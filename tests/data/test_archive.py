"""Moving data directories to a Hugging Face dataset and back."""

import os

import pytest

from hivemind.data import archive
from hivemind.paths import DATA_DIRECTORY

FILES = {
    "train/seg00/search_7000_00000.dst": os.urandom(4096) * 8,
    "train/seg01/search_7001_00000.dst": os.urandom(4096) * 8,
    "train_run/seg00/games.pgn": b"[Event ?]\n1. e4 *\n",
    "val/search_7999_00000.dst": os.urandom(1024),
    "logs/seg00.log": b"selfplay game 1\n",
}


def write(directory, files):
    for name, data in files.items():
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(data)
    return directory


def quiet(message):
    pass


def test_tarballs_go_where_the_directory_sits_under_data():
    assert archive.hub_path(DATA_DIRECTORY / "distill" / "train") == "distill/train"
    assert archive.hub_path("/elsewhere/rl-it7/search") == "elsewhere/rl-it7/search"


def test_archive_moves_files_to_the_hub_and_back(tmp_path, archive_hub):
    directory = write(tmp_path / "search", FILES)
    on_disk = archive.files(directory)
    assert archive.archive(archive_hub, "me/archive", directory, quiet) == sum(map(len, FILES.values()))
    assert archive.files(directory) == {} and list(directory.iterdir()) == [directory / archive.MANIFEST]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["hub", "search"]  # no scratch left behind
    assert archive_hub.uploads == [f"{archive.hub_path(directory)}-000.tar.zst"]
    [entry] = archive.manifest(directory)["archives"]
    assert entry["repo"] == "me/archive" and entry["path"] == archive_hub.uploads[0]
    assert entry["files"] == on_disk

    assert archive.restore(directory, quiet, archive_hub.download) == len(FILES)
    assert {name: (directory / name).read_bytes() for name in FILES} == FILES
    assert archive.files(directory) == on_disk  # mtimes too, so they count as archived
    assert sorted(path.name for path in tmp_path.iterdir()) == ["hub", "search"]

    # Archiving restored files again only deletes them: the Hub already holds them.
    assert archive.archive(archive_hub, "me/archive", directory, quiet) == sum(map(len, FILES.values()))
    assert len(archive_hub.uploads) == 1 and archive.files(directory) == {}


def test_files_added_later_go_in_a_new_tarball(tmp_path, archive_hub):
    directory = write(tmp_path / "search", FILES)
    archive.archive(archive_hub, "me/archive", directory, quiet)
    write(directory, {"train/seg-late/search_1_00000.dst": b"late batch"})
    assert archive.archive(archive_hub, "me/archive", directory, quiet) == len(b"late batch")
    first, second = archive.manifest(directory)["archives"]
    assert [entry["path"] for entry in (first, second)] == archive_hub.uploads
    assert archive_hub.uploads[1].endswith("search-001.tar.zst")
    assert set(first["files"]) == set(FILES) and set(second["files"]) == {"train/seg-late/search_1_00000.dst"}
    assert archive.restore(directory, quiet, archive_hub.download) == len(FILES) + 1
    assert (directory / "train/seg-late/search_1_00000.dst").read_bytes() == b"late batch"


def test_restore_takes_the_last_copy_and_never_overwrites_the_disk(tmp_path, archive_hub):
    directory = write(tmp_path / "search", {"a.dst": b"first", "b.dst": b"kept"})
    archive.archive(archive_hub, "me/archive", directory, quiet)
    write(directory, {"a.dst": b"second version"})
    archive.archive(archive_hub, "me/archive", directory, quiet)
    assert archive.restore(directory, quiet, archive_hub.download) == 2
    assert (directory / "a.dst").read_bytes() == b"second version"
    assert (directory / "b.dst").read_bytes() == b"kept"
    write(directory, {"b.dst": b"edited on disk"})
    (directory / "a.dst").unlink()
    assert archive.restore(directory, quiet, archive_hub.download) == 1
    assert (directory / "b.dst").read_bytes() == b"edited on disk"


def test_only_the_paths_given_are_archived(tmp_path, archive_hub):
    directory = write(tmp_path / "search", FILES)
    archive.archive(archive_hub, "me/archive", directory, quiet, ["train/seg00", "logs/seg00.log", "val_run"])
    assert set(archive.files(directory)) == {"train/seg01/search_7001_00000.dst", "train_run/seg00/games.pgn",
                                             "val/search_7999_00000.dst"}
    # Nothing new to archive: the call only reports what is already gone.
    assert archive.archive(archive_hub, "me/archive", directory, quiet, ["train/seg00"]) == 0
    assert len(archive_hub.uploads) == 1


def test_files_stay_when_the_hub_copy_does_not_match(tmp_path, archive_hub):
    directory = write(tmp_path / "search", FILES)
    on_disk = archive.files(directory)
    archive_hub.corrupt = True
    with pytest.raises(archive.ArchiveError, match="does not match"):
        archive.archive(archive_hub, "me/archive", directory, quiet)
    assert archive.files(directory) == on_disk
    assert not (directory / archive.MANIFEST).exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["hub", "search"]


def test_restore_refuses_a_tarball_that_changed_on_the_hub(tmp_path, archive_hub):
    directory = write(tmp_path / "search", FILES)
    archive.archive(archive_hub, "me/archive", directory, quiet)
    with (archive_hub.root / "me/archive" / archive_hub.uploads[0]).open("ab") as stream:
        stream.write(b"!")
    with pytest.raises(archive.ArchiveError, match="changed since it was archived"):
        archive.restore(directory, quiet, archive_hub.download)
    assert archive.files(directory) == {}
