"""RL loop bookkeeping: resuming, data discovery, match results and uploads."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hivemind.cli import rl_loop
from hivemind.data import archive


def test_next_iteration_resumes_after_the_last_finished_one(tmp_path):
    assert rl_loop.next_iteration(tmp_path) == 1
    pilot = tmp_path / "rl-it1" / "arms" / "C"
    pilot.mkdir(parents=True)
    (pilot / "last.pt").touch()
    assert rl_loop.next_iteration(tmp_path) == 2
    (tmp_path / "rl-it2" / "stages").mkdir(parents=True)
    (tmp_path / "rl-it2" / "stages" / "iteration.done").write_text("{}\n")
    (tmp_path / "rl-it3" / "stages").mkdir(parents=True)
    assert rl_loop.next_iteration(tmp_path) == 3


def test_train_dirs_cover_both_layouts_and_only_finished_segments(tmp_path):
    pilot = tmp_path / "rl-it1" / "search" / "train"
    pilot.mkdir(parents=True)
    (pilot / "search_31_00000.dst").touch()
    assert rl_loop.train_dirs(tmp_path, 1) == [pilot]
    stages = tmp_path / "rl-it2" / "stages"
    stages.mkdir(parents=True)
    (stages / "seg01.done").touch()
    (stages / "seg00.done").touch()
    (stages / "val.done").touch()
    train = tmp_path / "rl-it2" / "search" / "train"
    assert rl_loop.train_dirs(tmp_path, 2) == [train / "seg00", train / "seg01"]


def test_replay_window_stops_at_the_first_iteration():
    assert rl_loop.replay_iterations(7, 3) == [6, 5, 4]
    assert rl_loop.replay_iterations(2, 3) == [1]
    assert rl_loop.replay_iterations(1, 3) == []


@pytest.mark.parametrize("completed, segment_games, target, expected", [
    ({k: 1000 for k in range(91)}, 10_000, 100_000, [("seg91", 9000)]),
    ({0: 10_000}, 1000, 12_500, [("seg01", 1000), ("seg02", 1000), ("seg03", 500)]),
    ({0: 1000, 2: 1000}, 1000, 3000, [("seg01", 1000)]),
    ({0: 10_000}, 1000, 10_000, []),
    ({}, 1000, 1500, [("seg00", 1000), ("seg01", 500)]),
])
def test_iteration_generates_remaining_games_before_validation(
        tmp_path, completed, segment_games, target, expected):
    loop = rl_loop.Loop.__new__(rl_loop.Loop)
    loop.runs, loop.exchange = tmp_path, None
    loop.it = tmp_path / "rl-it7"
    stages = loop.it / "stages"
    stages.mkdir(parents=True)
    (stages / "generator").write_text("generator\n")
    for k, games in completed.items():
        (stages / f"seg{k:02d}.done").write_text(json.dumps({"games": games}))
    loop.args = SimpleNamespace(games=target, segment_games=segment_games, val_games=1000,
                                keep_selfplay=True)
    loop.generator = tmp_path / "generator.onnx"
    loop.fixed = tmp_path / "missing-it0.onnx"
    loop.log = lambda message: None
    calls = []

    def selfplay(name, games, seed, data, records):
        calls.append((name, games))
        assert seed == 7000 + (999 if name == "val" else int(name[3:]))
        assert data == loop.it / "search" / ("val" if name == "val" else f"train/{name}")
        if name == "val":
            assert loop.games_played() >= target
        loop.mark(name, {"games": games, "seed": seed})

    loop.selfplay = selfplay
    loop.train = lambda: calls.append(("train", 0))
    loop.match = lambda *args: None
    loop.promote = lambda: None
    loop.upload = lambda: None
    loop.iteration(7)
    assert calls == expected + [("val", 1000), ("train", 0)]
    assert loop.games_played() == target
    for k, games in completed.items():
        assert json.loads((stages / f"seg{k:02d}.done").read_text()) == {"games": games}


def test_the_clock_model_reaches_self_play_and_matches(tmp_path, monkeypatch):
    loop = rl_loop.Loop.__new__(rl_loop.Loop)
    loop.runs = tmp_path
    loop.n, loop.it = 3, tmp_path / "rl-it3"
    loop.arms = loop.it / "arms"
    for directory in ("stages", "arms", "models", "search"):
        (loop.it / directory).mkdir(parents=True)
    loop.generator = tmp_path / "net.onnx"
    loop.generator.write_bytes(b"net")
    loop.generator_sp1 = tmp_path / "net-sp1.onnx"
    (loop.it / "stages" / "generator").write_text(rl_loop.sha256(loop.generator) + "\n")
    loop.args = SimpleNamespace(selfplay_engine=tmp_path / "sp1", match_engine=tmp_path / "engine",
                                parallel_games=6, time_model="clock", match_games=2,
                                match_movetime=1000, openings=tmp_path / "openings.tsv",
                                pause_for="", keep_selfplay=True)
    loop.log = lambda message: None
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command[1] == "selfplay":
            out = Path(command[command.index("--distill-output") + 1])
            out.mkdir(parents=True)
            Path(kwargs["stdout"].name).write_text("selfplay game 1/1\n")
        return SimpleNamespace(returncode=0)

    class Process:
        def __init__(self, command, **kwargs):
            commands.append(command)
            out = Path(command[command.index("--output") + 1])
            out.mkdir(parents=True)
            write_summary(out / "summary.json", 1, 1, 0)

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(rl_loop.subprocess, "run", run)
    monkeypatch.setattr(rl_loop.subprocess, "Popen", Process)
    monkeypatch.setattr(rl_loop.train, "positions", lambda dirs: 10)
    loop.selfplay("seg00", 1, 3000, loop.it / "search" / "train" / "seg00",
                  loop.it / "search" / "train_run" / "seg00")
    loop.match("generator", loop.generator)

    selfplay_command, match_command = commands
    assert selfplay_command[selfplay_command.index("--time-model") + 1] == "clock"
    assert match_command[match_command.index("--time-model") + 1] == "clock"
    assert match_command[match_command.index("--movetime") + 1] == "1000"
    marker = json.loads((loop.it / "stages" / "match_generator.done").read_text())
    assert marker["time_model"] == "clock" and marker["movetime"] == 1000
    assert "--baseline-time-model" not in match_command
    assert "baseline_time_model" not in marker

    # A pre-clock it0 searches on the binary advantage inside the clocked match.
    loop.match("it0", tmp_path / "it0.onnx", "flag")
    it0_command = commands[-1]
    assert it0_command[it0_command.index("--time-model") + 1] == "clock"
    assert it0_command[it0_command.index("--baseline-time-model") + 1] == "flag"
    marker = json.loads((loop.it / "stages" / "match_it0.done").read_text())
    assert marker["baseline_time_model"] == "flag"


def write_summary(path, wins, losses, draws):
    path.write_text(json.dumps({
        "contender_wins": wins, "baseline_wins": losses, "draws": draws,
        "contender_elo": -30.5, "elo_confidence_95": [-63.0, 1.5],
        "performance": {"contender": {"nps": 73728.7}, "baseline": {"nps": 73261.2}},
    }))


def test_match_summary_scores_complete_matches_only(tmp_path):
    path = tmp_path / "summary.json"
    write_summary(path, 70, 84, 6)
    summary = rl_loop.match_summary(path, 160)
    assert summary["score"] == 0.45625
    assert summary["nps"] == [73728.7, 73261.2]
    with pytest.raises(rl_loop.StageFailed):
        rl_loop.match_summary(path, 162)


class FakeHub:
    def __init__(self, files, remote_sha):
        self.files, self.remote_sha, self.commits = set(files), remote_sha, []

    def list_repo_files(self, repo):
        return sorted(self.files)

    def get_paths_info(self, repo, paths):
        return [SimpleNamespace(path=path, lfs=SimpleNamespace(sha256=self.remote_sha[path]))
                for path in paths if path in self.remote_sha]

    def create_commit(self, repo_id, commit_message, operations):
        self.commits.append(([op.path_in_repo for op in operations], commit_message))
        return SimpleNamespace(oid="abc123")


def trained_iteration(runs, n, weights):
    it = runs / f"rl-it{n}"
    (it / "stages").mkdir(parents=True)
    (it / "stages" / "train.done").write_text("{}\n")
    (it / "arms" / "C").mkdir(parents=True)
    (it / "arms" / "C" / "last.pt").write_bytes(weights)
    return it


def test_iteration_checkpoints_are_the_finished_trainings(tmp_path):
    pilot = tmp_path / "rl-it1" / "arms" / "C"
    pilot.mkdir(parents=True)
    (pilot / "last.pt").touch()
    trained_iteration(tmp_path, 2, b"it2")
    trained_iteration(tmp_path / "aside", 2, b"aborted")
    (tmp_path / "aside" / "rl-it2").rename(tmp_path / "rl-it2-unsegmented-aborted")
    unfinished = tmp_path / "rl-it3" / "arms" / "C"
    (tmp_path / "rl-it3" / "stages").mkdir(parents=True)
    unfinished.mkdir(parents=True)
    (unfinished / "last.pt").touch()
    assert rl_loop.iteration_checkpoints(tmp_path) == [
        (1, pilot / "last.pt"), (2, tmp_path / "rl-it2" / "arms" / "C" / "last.pt")]


def test_sync_uploads_what_the_hub_lacks_in_one_commit(tmp_path):
    models, runs = tmp_path / "models", tmp_path / "runs"
    models.mkdir()
    for n in (0, 1, 2):
        (models / f"net-it{n}.onnx").write_bytes(f"it{n}".encode())
    generator, checkpoint = models / "net.onnx", tmp_path / "best.pt"
    generator.write_bytes(b"it2")
    checkpoint.write_bytes(b"it2 weights")
    trained_iteration(runs, 1, b"it1 weights")
    it2 = trained_iteration(runs, 2, b"it2 weights")
    (it2 / "stages" / "match_generator.done").write_text(json.dumps({"wins": 81, "losses": 75, "draws": 4,
                                                                     "elo": 13.0}))
    hub = FakeHub({"net-it0.onnx", "net-it1.onnx", "net-it1.pt", "net.onnx"},
                  {"net.onnx": rl_loop.sha256(models / "net-it1.onnx")})
    rl_loop.sync_hub(hub, "repo", generator, checkpoint, lambda message: None, runs)
    assert hub.commits == [(["net-it2.onnx", "net-it2.pt", "net.onnx", "net.pt"],
                            "Upload net from RL iteration 2: 81-75-4 vs the previous generator at 100 ms "
                            "(+13 Elo)")]

    hub = FakeHub({"net-it0.onnx", "net-it1.onnx", "net-it2.onnx", "net-it1.pt", "net-it2.pt"},
                  {"net.onnx": rl_loop.sha256(generator), "net.pt": rl_loop.sha256(checkpoint)})
    rl_loop.sync_hub(hub, "repo", generator, checkpoint, lambda message: None, runs)
    assert hub.commits == []

    # The checkpoint of an unchanged generator, as when it was first published.
    hub.remote_sha.pop("net.pt")
    rl_loop.sync_hub(hub, "repo", generator, checkpoint, lambda message: None, runs)
    assert hub.commits == [(["net.pt"], "Upload net.pt")]


def test_generator_checkpoint_is_downloaded_only_for_the_published_network(tmp_path, monkeypatch):
    from hivemind.cli import train

    generator, checkpoint = tmp_path / "net.onnx", tmp_path / "ckpt" / "best.pt"
    generator.write_bytes(b"published")
    downloads = []

    def fetch(variant, output_dir):
        downloads.append(variant)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "net.pt").write_bytes(b"weights")
        return output_dir / "net.pt"

    monkeypatch.setattr(train, "fetch_network", fetch)
    artifact = train.ARTIFACTS["onnx"]._replace(sha256=rl_loop.sha256(generator))
    monkeypatch.setitem(train.ARTIFACTS, "onnx", artifact)
    assert train.generator_checkpoint(generator, checkpoint) == checkpoint
    assert checkpoint.read_bytes() == b"weights" and downloads == ["checkpoint"]
    assert train.generator_checkpoint(generator, checkpoint) == checkpoint
    assert downloads == ["checkpoint"]
    generator.write_bytes(b"promoted locally")
    checkpoint.unlink()
    with pytest.raises(FileNotFoundError):
        train.generator_checkpoint(generator, checkpoint)


class FakeAccount:
    def __init__(self, account):
        self.account = account

    def whoami(self):
        if self.account is None:
            raise OSError("offline")
        return self.account


def test_uploads_are_skipped_without_write_access():
    owner = FakeAccount({"name": "aminwoo", "orgs": []})
    assert rl_loop.upload_blocker(owner, "aminwoo/bughouse-twin-s", None).startswith("not logged in")
    assert rl_loop.upload_blocker(owner, "aminwoo/bughouse-twin-s", "token") is None
    member = FakeAccount({"name": "someone", "orgs": [{"name": "aminwoo"}]})
    assert rl_loop.upload_blocker(member, "aminwoo/bughouse-twin-s", "token") is None
    other = FakeAccount({"name": "someone", "orgs": []})
    assert "cannot upload" in rl_loop.upload_blocker(other, "aminwoo/bughouse-twin-s", "token")
    assert rl_loop.upload_blocker(FakeAccount(None), "aminwoo/bughouse-twin-s", "token") is None


def test_collect_files_batches_under_their_iteration(tmp_path, monkeypatch):
    from hivemind.distill.exchange import PROTOCOL, DirectoryExchange

    runs, shared = tmp_path / "runs", tmp_path / "shared"
    for n, generator in ((6, "g4"), (7, "g7")):
        (runs / f"rl-it{n}" / "stages").mkdir(parents=True)
        (runs / f"rl-it{n}" / "stages" / "generator").write_text(generator + "\n")
    exchange = DirectoryExchange(shared)
    exchange.create()

    def submit(folder, n, generator):
        source = tmp_path / folder.replace("/", "_")
        source.mkdir()
        (source / "search_1_00000.dst").write_bytes(b"x")
        exchange.submit(folder, source, {"protocol": PROTOCOL, "iteration": n, "generator": generator,
                                         "games": 10, "contributor": "mac", "folder": folder})

    submit("it7/mac-01", 7, "g7")  # current
    submit("it6/mac-02", 6, "g4")  # late: still the generator of iteration 6
    submit("it7/mac-03", 7, "g4")  # an old generator
    loop = rl_loop.Loop.__new__(rl_loop.Loop)
    loop.runs, loop.exchange, loop.log = runs, exchange, lambda message: None
    submit("it7/mac-04", 7, "g7")  # broken values
    monkeypatch.setattr(rl_loop.train, "positions", lambda dirs: 5)
    monkeypatch.setattr(rl_loop, "chunk_problem",
                        lambda path: "non-finite plane_values" if "mac-04" in str(path) else None)
    loop.collect()
    assert sorted(p.name for p in (runs / "rl-it7" / "stages").glob("seg*.done")) == ["seg-mac-01.done"]
    assert sorted(p.name for p in (runs / "rl-it6" / "stages").glob("seg*.done")) == ["seg-mac-02.done"]
    assert (runs / "rl-it7" / "search" / "train" / "seg-mac-01" / "search_1_00000.dst").is_file()
    assert json.loads((runs / "rl-it7" / "stages" / "seg-mac-01.done").read_text())["games"] == 10
    assert list(exchange.submissions()) == []
    assert (shared / "rejected" / "it7--mac-03").is_dir()
    assert (shared / "rejected" / "it7--mac-04").is_dir()
    assert not (runs / "rl-it7" / "search" / "train" / "seg-mac-04").exists()
    assert rl_loop.train_dirs(runs, 7) == [runs / "rl-it7" / "search" / "train" / "seg-mac-01"]


def write(directory, files):
    for name, data in files.items():
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(data)


def test_finished_selfplay_is_every_finished_stage(tmp_path):
    it = tmp_path / "rl-it7"
    assert rl_loop.finished_selfplay(it) is None  # the pilot's layout: all of it is finished
    (it / "stages").mkdir(parents=True)
    assert rl_loop.finished_selfplay(it) == []
    for stage in ("seg00", "seg-mac-01", "val"):
        (it / "stages" / f"{stage}.done").touch()
    assert rl_loop.finished_selfplay(it) == [
        "train/seg-mac-01", "train_run/seg-mac-01", "logs/seg-mac-01.log",
        "train/seg00", "train_run/seg00", "logs/seg00.log", "val", "val_run", "logs/val.log"]


def archiving_loop(runs, hub, **args):
    loop = rl_loop.Loop.__new__(rl_loop.Loop)
    loop.runs, loop.n, loop.api, loop.teacher_val = runs, 7, hub, []
    loop.args = SimpleNamespace(**{"keep_selfplay": False, "archive": "me/archive", **args})
    loop.log = lambda message: None
    return loop


def test_loop_archives_finished_selfplay_and_what_was_moved_aside(tmp_path, archive_hub):
    it = tmp_path / "rl-it7"
    write(it / "search", {"train/seg00/a.dst": b"done", "train_run/seg00/games.pgn": b"pgn",
                          "logs/seg00.log": b"log", "train/seg01/a.dst": b"interrupted",
                          "logs/seg01.log": b"interrupted"})
    write(it / "aborted", {"search_train_seg07-20261006-001519/a.dst": b"moved aside"})
    (it / "stages").mkdir()
    (it / "stages" / "seg00.done").touch()
    archiving_loop(tmp_path, archive_hub).archive(6, 7)
    assert set(archive.files(it / "search")) == {"train/seg01/a.dst", "logs/seg01.log"}
    assert archive.files(it / "aborted") == {}
    assert len(archive_hub.uploads) == 2

    archiving_loop(tmp_path, archive_hub, keep_selfplay=True).archive(7)
    assert len(archive_hub.uploads) == 2


# Stands in for distill-train: notes the chunks it was given and leaves a model.
TRAINER = """import json, sys
from pathlib import Path
out = Path(sys.argv[1])
out.mkdir(parents=True)
(out / "seen.json").write_text(json.dumps(sorted(p.name for d in sys.argv[2:] for p in Path(d).glob("*.dst"))))
(out / "last.pt").write_text("weights")
(out / "distill-twin-s-v3.0.onnx").write_text("model")
"""


def test_training_downloads_its_games_and_deletes_them_again(tmp_path, archive_hub, monkeypatch):
    runs = tmp_path / "runs"
    for n in (5, 6, 7):
        write(runs / f"rl-it{n}" / "search", {f"train/seg00/search_{n}_00000.dst": b"games",
                                              f"val/search_{n}_00999.dst": b"val"})
        (runs / f"rl-it{n}" / "stages").mkdir()
        for stage in ("seg00", "val"):
            (runs / f"rl-it{n}" / "stages" / f"{stage}.done").touch()
    write(runs / "rl-it6" / "arms" / "C", {"last.pt": b"previous"})
    (runs / "rl-it7" / "arms").mkdir()
    teacher_val = tmp_path / "teacher_val"
    write(teacher_val, {"distill_2_0_00000.dst": b"teacher"})
    loop = archiving_loop(runs, archive_hub, window=2, new_passes=1.5, replay_passes=0.5, batch_size=2,
                          pause_for="")
    loop.archive(5, 6, 7)
    archive.archive(archive_hub, "me/archive", teacher_val, loop.log)
    uploads = list(archive_hub.uploads)
    loop.it, loop.arms, loop.teacher_val = runs / "rl-it7", runs / "rl-it7" / "arms", [teacher_val]
    loop.generator = tmp_path / "models" / "twin.onnx"
    monkeypatch.setattr(archive, "hub_download", archive_hub.download)
    monkeypatch.setattr(rl_loop.train, "positions", lambda dirs: 16 * sum(len(list(d.glob("*.dst"))) for d in dirs))
    monkeypatch.setattr(rl_loop.train, "distill_command", lambda init, new, val, old, *args: [
        sys.executable, "-c", TRAINER, str(args[-2]), *map(str, [*new, *val, *old, *args[-1]])])

    loop.train()
    assert json.loads((runs / "rl-it7" / "arms" / "C" / "seen.json").read_text()) == [
        "distill_2_0_00000.dst", "search_5_00000.dst", "search_6_00000.dst", "search_7_00000.dst",
        "search_7_00999.dst"]
    assert (runs / "rl-it7" / "stages" / "train.done").is_file()
    for directory in [*(runs / f"rl-it{n}" / "search" for n in (5, 6, 7)), teacher_val]:
        assert archive.files(directory) == {}
    assert archive_hub.uploads == uploads  # put back by deleting: the Hub already had them
