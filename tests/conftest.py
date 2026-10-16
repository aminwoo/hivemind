import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from hivemind.distill.exchange import sha256


class ArchiveHub:
    """The HfApi calls hivemind.data.archive makes, kept in a directory."""

    def __init__(self, root):
        self.root, self.corrupt, self.uploads = root, False, []

    def create_repo(self, repo, repo_type, private, exist_ok):
        assert (repo_type, private, exist_ok) == ("dataset", True, True)

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
        target = self.root / repo_id / path_in_repo
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path_or_fileobj, target)
        if self.corrupt:
            with target.open("ab") as stream:
                stream.write(b"!")
        self.uploads.append(path_in_repo)

    def get_paths_info(self, repo, paths, repo_type):
        return [SimpleNamespace(path=path, lfs=SimpleNamespace(sha256=sha256(self.root / repo / path)))
                for path in paths if (self.root / repo / path).is_file()]

    def download(self, repo, path_in_repo, directory):
        return Path(shutil.copy(self.root / repo / path_in_repo, directory))


@pytest.fixture
def archive_hub(tmp_path):
    if shutil.which("zstd") is None:
        pytest.skip("needs the zstd binary")
    return ArchiveHub(tmp_path / "hub")
