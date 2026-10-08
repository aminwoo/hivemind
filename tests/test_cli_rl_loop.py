"""RL loop bookkeeping: resuming, data discovery, match results and uploads."""

import json
from types import SimpleNamespace

import pytest

from hivemind.cli import rl_loop


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
    monkeypatch.setattr(rl_loop.train, "positions", lambda dirs: 5)
    loop.collect()
    assert sorted(p.name for p in (runs / "rl-it7" / "stages").glob("seg*.done")) == ["seg-mac-01.done"]
    assert sorted(p.name for p in (runs / "rl-it6" / "stages").glob("seg*.done")) == ["seg-mac-02.done"]
    assert (runs / "rl-it7" / "search" / "train" / "seg-mac-01" / "search_1_00000.dst").is_file()
    assert json.loads((runs / "rl-it7" / "stages" / "seg-mac-01.done").read_text())["games"] == 10
    assert list(exchange.submissions()) == []
    assert (shared / "rejected" / "it7--mac-03").is_dir()
    assert rl_loop.train_dirs(runs, 7) == [runs / "rl-it7" / "search" / "train" / "seg-mac-01"]
