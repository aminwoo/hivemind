"""Where an RL trainer meets the machines that play its self-play games.

The trainer publishes a work order: the iteration, the generator network to
play, and whether games are still wanted. Contributors play games with that
generator and submit each batch as a folder of HDST chunks with a manifest;
the trainer collects the batches into its runs and closes them. Two backends:

  * HubExchange (hf://NAMESPACE/NAME): a Hugging Face dataset repository. The
    work order is its status.json, the generator is downloaded from the model
    repository at the revision the order pins, and each batch is a pull
    request adding one folder, merged once the trainer has collected it. Any
    Hugging Face account can open one.
  * DirectoryExchange (a path): a directory every machine can reach, such as
    a shared drive.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import NamedTuple

# Bump when self-play settings or the submission format change: contributors
# and the trainer must agree on how games are played.
PROTOCOL = 1
PR_TITLE = "Self-play games: "


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def install(source, destination):
    """Copy atomically, so readers never see a partial file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Submission(NamedTuple):
    folder: str  # it7/NAME-SEED: unique across contributors
    manifest: dict
    handle: object  # the backend's own reference: a pull request or a directory


def open_exchange(spec, model_repo, api=None):
    """The exchange named by `spec`: hf://NAMESPACE/NAME or a directory."""
    if spec.startswith("hf://"):
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi()
        return HubExchange(api, spec[len("hf://"):], model_repo)
    return DirectoryExchange(spec)


class DirectoryExchange:
    def __init__(self, root):
        self.root = Path(root)

    def __str__(self):
        return str(self.root)

    def create(self):
        for name in ("generators", "incoming", "rejected"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def publish(self, order, generator):
        digest = sha256(generator)
        stored = self.root / "generators" / f"{digest}.onnx"
        if not stored.is_file():
            install(generator, stored)
        order = {**order, "generator": {"file": f"generators/{digest}.onnx", "name": generator.name,
                                        "sha256": digest}, "updated": now()}
        temporary = self.root / "status.json.tmp"
        temporary.write_text(json.dumps(order, indent=2) + "\n")
        temporary.replace(self.root / "status.json")

    def order(self):
        path = self.root / "status.json"
        return json.loads(path.read_text()) if path.is_file() else None

    def generator(self, order, directory):
        spec = order["generator"]
        destination = Path(directory) / f"{Path(spec['name']).stem}-{spec['sha256'][:12]}.onnx"
        if not (destination.is_file() and sha256(destination) == spec["sha256"]):
            install(self.root / spec["file"], destination)
        return destination

    def submit(self, folder, source, manifest):
        name = folder.replace("/", "--")
        temporary = self.root / "incoming" / f".{name}.tmp"
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True)
        for chunk in sorted(Path(source).glob("*.dst")):
            shutil.copyfile(chunk, temporary / chunk.name)
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.rename(self.root / "incoming" / name)
        return str(self.root / "incoming" / name)

    def submissions(self):
        for path in sorted((self.root / "incoming").glob("[!.]*")):
            manifest = json.loads((path / "manifest.json").read_text())
            yield Submission(manifest["folder"], manifest, path)

    def fetch(self, submission, destination):
        destination = Path(destination)
        temporary = destination.with_name(f".{destination.name}.tmp")
        shutil.rmtree(temporary, ignore_errors=True)
        shutil.copytree(submission.handle, temporary)
        temporary.rename(destination)

    def finish(self, submission, accepted, note):
        if accepted:
            shutil.rmtree(submission.handle)
        else:
            rejected = self.root / "rejected" / submission.handle.name
            shutil.rmtree(rejected, ignore_errors=True)
            submission.handle.rename(rejected)
            (rejected / "reason.txt").write_text(note + "\n")


class HubExchange:
    def __init__(self, api, repo, model_repo):
        self.api, self.repo, self.model_repo = api, repo, model_repo

    def __str__(self):
        return f"hf://{self.repo}"

    def create(self):
        # Private until it is opened to other contributors (a repository setting).
        self.api.create_repo(self.repo, repo_type="dataset", private=True, exist_ok=True)

    def publish(self, order, generator):
        """The generator must already be on the model repository (rl-loop's
        upload stage puts it there): the order pins that revision."""
        digest = sha256(generator)
        revision = self.api.model_info(self.model_repo).sha
        info = self.api.get_paths_info(self.model_repo, [generator.name], revision=revision)
        if not (info and info[0].lfs and info[0].lfs.sha256 == digest):
            raise RuntimeError(f"{generator.name} on {self.model_repo} is not the generator {digest[:12]}")
        order = {**order, "generator": {"repo": self.model_repo, "revision": revision, "file": generator.name,
                                        "name": generator.name, "sha256": digest}, "updated": now()}
        state = "open" if order["accepting"] else "closed"
        self.api.upload_file(path_or_fileobj=(json.dumps(order, indent=2) + "\n").encode(),
                             path_in_repo="status.json", repo_id=self.repo, repo_type="dataset",
                             commit_message=f"Work order: iteration {order['iteration']}, {state}")

    def order(self):
        from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
        try:
            path = self.api.hf_hub_download(self.repo, "status.json", repo_type="dataset")
        except (EntryNotFoundError, RepositoryNotFoundError):
            return None
        return json.loads(Path(path).read_text())

    def generator(self, order, directory):
        spec = order["generator"]
        destination = Path(directory) / f"{Path(spec['name']).stem}-{spec['sha256'][:12]}.onnx"
        if not (destination.is_file() and sha256(destination) == spec["sha256"]):
            path = self.api.hf_hub_download(spec["repo"], spec["file"], revision=spec["revision"])
            if sha256(path) != spec["sha256"]:
                raise ValueError(f"{spec['file']} at {spec['revision']} is not {spec['sha256'][:12]}")
            install(path, destination)
        return destination

    def submit(self, folder, source, manifest):
        from huggingface_hub import CommitOperationAdd
        operations = [CommitOperationAdd(path_in_repo=f"{folder}/{chunk.name}", path_or_fileobj=str(chunk))
                      for chunk in sorted(Path(source).glob("*.dst"))]
        operations.append(CommitOperationAdd(path_in_repo=f"{folder}/manifest.json",
                                             path_or_fileobj=(json.dumps(manifest, indent=2) + "\n").encode()))
        commit = self.api.create_commit(self.repo, operations, commit_message=PR_TITLE + folder,
                                        repo_type="dataset", create_pr=True)
        return commit.pr_url

    def submissions(self):
        # Pull requests made through the API start as drafts.
        for pr in self.api.get_repo_discussions(self.repo, discussion_type="pull_request",
                                                discussion_status="open", repo_type="dataset"):
            if pr.status not in ("open", "draft") or not pr.title.startswith(PR_TITLE):
                continue
            folder = pr.title[len(PR_TITLE):]
            path = self.api.hf_hub_download(self.repo, f"{folder}/manifest.json", repo_type="dataset",
                                            revision=f"refs/pr/{pr.num}")
            yield Submission(folder, json.loads(Path(path).read_text()), pr)

    def fetch(self, submission, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            self.api.snapshot_download(self.repo, repo_type="dataset", revision=f"refs/pr/{submission.handle.num}",
                                       allow_patterns=[f"{submission.folder}/*"], local_dir=temporary)
            Path(temporary, submission.folder).rename(destination)

    def finish(self, submission, accepted, note):
        pr = submission.handle
        if not accepted:
            self.api.change_discussion_status(self.repo, pr.num, "closed", comment=note, repo_type="dataset")
            return
        if pr.status == "draft":
            self.api.change_discussion_status(self.repo, pr.num, "open", repo_type="dataset")
        self.api.merge_pull_request(self.repo, pr.num, comment=note, repo_type="dataset")
