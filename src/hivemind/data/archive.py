"""Move data directories to a private Hugging Face dataset, and back.

archive() packs a directory's files into a zstd tarball, reads it back against
them, uploads it and checks the Hub's copy, and only then deletes the files.
The directory keeps an archive.json listing every tarball and the files in it,
so restore() can bring them back; files added later go in the next tarball.
In the dataset each tarball sits at the directory's path under data/:
data/distill/train becomes distill/train-000.tar.zst.
"""

import json
from pathlib import Path
import subprocess
import tempfile

from hivemind.distill.exchange import sha256
from hivemind.paths import DATA_DIRECTORY

MANIFEST = "archive.json"
# The posix format keeps mtimes to the nanosecond, so a restored file is
# recognised as one the archive already holds. Level 9 is 15-20% smaller than
# zstd's default and still packs hundreds of MB/s, far faster than an upload.
TAR = ["tar", "--format=posix", "-I", "zstd -9 -T0"]


class ArchiveError(Exception):
    pass


def manifest(directory):
    """DIRECTORY/archive.json: the tarballs DIRECTORY's files went to, and the files in each."""
    path = Path(directory) / MANIFEST
    return json.loads(path.read_text()) if path.is_file() else {"archives": []}


def file_stat(path):
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def files(directory, paths=None):
    """{path under DIRECTORY: [size, mtime_ns]} of the files on disk but
    archive.json, only those under one of PATHS (relative to DIRECTORY) when
    given."""
    directory = Path(directory)
    found = {}
    for path in sorted(directory.rglob("*")) if directory.is_dir() else []:
        relative = path.relative_to(directory)
        if (path.is_file() and relative.as_posix() != MANIFEST
                and (paths is None or any(relative.is_relative_to(p) for p in paths))):
            found[relative.as_posix()] = file_stat(path)
    return found


def held(directory):
    """{path: [size, mtime_ns]} of every file an archived tarball of DIRECTORY holds, as last archived."""
    return {path: stat for entry in manifest(directory)["archives"] for path, stat in entry["files"].items()}


def hub_path(directory):
    """Where DIRECTORY's tarballs go in the dataset: its path under data/, or
    its absolute path for a directory elsewhere."""
    directory = Path(directory).resolve()
    try:
        return directory.relative_to(DATA_DIRECTORY.resolve()).as_posix()
    except ValueError:
        return directory.as_posix().lstrip("/")


def evict(directory):
    """Delete DIRECTORY's files that an archived tarball holds unchanged, and
    the subdirectories that leaves empty. Returns the bytes freed."""
    directory = Path(directory)
    freed = 0
    for path, stat in held(directory).items():
        file = directory / path
        if file.is_file() and file_stat(file) == stat:
            file.unlink()
            freed += stat[0]
    for sub in sorted((d for d in directory.rglob("*") if d.is_dir()), key=lambda d: len(d.parts), reverse=True):
        if not any(sub.iterdir()):
            sub.rmdir()
    return freed


def archive(api, repo, directory, log=print, paths=None):
    """Move DIRECTORY's files (only those under PATHS, relative to it, when
    given) to the private dataset REPO, creating it if need be. Files no
    tarball holds yet go in a new one. Returns the bytes freed."""
    directory = Path(directory)
    record = manifest(directory)
    stored = held(directory)
    new = {path: stat for path, stat in files(directory, paths).items() if stored.get(path) != stat}
    if new:
        # Numbered in archive.json's order. A tarball uploaded but never
        # recorded is overwritten by the next one, which holds its files too.
        path_in_repo = f"{hub_path(directory)}-{len(record['archives']):03d}.tar.zst"
        with tempfile.TemporaryDirectory(dir=directory.parent, prefix=f".{directory.name}-archive-") as scratch:
            tarball, listing = Path(scratch) / "archive.tar.zst", Path(scratch) / "files"
            listing.write_text("".join(f"{path}\0" for path in new))
            subprocess.run(TAR + ["-cf", str(tarball), "-C", str(directory), "--null", "-T", str(listing)],
                           check=True)
            # --compare reads every member back and checks its contents, size
            # and mtime against the file on disk.
            subprocess.run(TAR + ["--compare", "-f", str(tarball), "-C", str(directory)], check=True)
            digest, size = sha256(tarball), tarball.stat().st_size
            api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
            api.upload_file(path_or_fileobj=str(tarball), path_in_repo=path_in_repo, repo_id=repo,
                            repo_type="dataset", commit_message=f"Archive {len(new)} files of {hub_path(directory)}")
            remote = api.get_paths_info(repo, [path_in_repo], repo_type="dataset")
            if not remote or remote[0].lfs is None or remote[0].lfs.sha256 != digest:
                raise ArchiveError(f"{repo}/{path_in_repo} on the Hub does not match the tarball of {directory}")
            record["archives"].append({"repo": repo, "path": path_in_repo, "sha256": digest, "bytes": size,
                                       "files": new})
            temporary = Path(scratch) / MANIFEST
            temporary.write_text(json.dumps(record) + "\n")
            temporary.replace(directory / MANIFEST)
        log(f"archived {len(new)} files of {directory} ({sum(s for s, _ in new.values()) / 1e9:.2f} GB) "
            f"as {repo}/{path_in_repo} ({size / 1e9:.2f} GB)")
    return evict(directory)


def hub_download(repo, path_in_repo, directory):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, path_in_repo, repo_type="dataset", local_dir=directory))


def restore(directory, log=print, download=None):
    """Bring back DIRECTORY's archived files that are missing from disk, each
    from the last tarball that holds it. A file on disk is never overwritten.
    Returns how many came back."""
    directory = Path(directory)
    record = manifest(directory)
    latest = {path: index for index, entry in enumerate(record["archives"]) for path in entry["files"]}
    wanted = {}
    for path, index in latest.items():
        if not (directory / path).exists():
            wanted.setdefault(index, []).append(path)
    for index in sorted(wanted):
        entry = record["archives"][index]
        with tempfile.TemporaryDirectory(dir=directory.parent, prefix=f".{directory.name}-restore-") as scratch:
            tarball = (download or hub_download)(entry["repo"], entry["path"], Path(scratch))
            if sha256(tarball) != entry["sha256"]:
                raise ArchiveError(f"{entry['repo']}/{entry['path']} changed since it was archived")
            listing, unpacked = Path(scratch) / "files", Path(scratch) / "unpacked"
            listing.write_text("".join(f"{path}\0" for path in wanted[index]))
            unpacked.mkdir()
            subprocess.run(TAR + ["-xf", str(tarball), "-C", str(unpacked), "--null", "-T", str(listing)],
                           check=True)
            # Moved into place whole, so an interrupted restore leaves no
            # partial file for the next one to take as already there.
            for path in wanted[index]:
                (directory / path).parent.mkdir(parents=True, exist_ok=True)
                (unpacked / path).replace(directory / path)
        log(f"restored {len(wanted[index])} files of {directory} from {entry['repo']}/{entry['path']}")
    return sum(len(paths) for paths in wanted.values())
