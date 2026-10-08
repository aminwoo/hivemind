"""Work orders and self-play batches passed between an RL loop and contributors."""

import json
from types import SimpleNamespace

from hivemind.distill.exchange import PROTOCOL, DirectoryExchange, HubExchange, sha256


def batch(tmp_path, name, payload=b"chunk"):
    source = tmp_path / name
    source.mkdir()
    (source / "search_1_00000.dst").write_bytes(payload)
    (source / "games").mkdir()
    return source


def test_directory_exchange_round_trip(tmp_path):
    exchange = DirectoryExchange(tmp_path / "shared")
    exchange.create()
    assert exchange.order() is None
    generator = tmp_path / "net.onnx"
    generator.write_bytes(b"network")
    exchange.publish({"protocol": PROTOCOL, "iteration": 7, "accepting": True, "games": 100}, generator)
    order = exchange.order()
    assert order["iteration"] == 7 and order["generator"]["sha256"] == sha256(generator)
    model = exchange.generator(order, tmp_path / "models")
    assert model.read_bytes() == b"network" and model.name == f"net-{sha256(generator)[:12]}.onnx"

    manifest = {"folder": "it7/mac-00ff", "iteration": 7, "games": 3}
    exchange.submit("it7/mac-00ff", batch(tmp_path, "mac-00ff"), manifest)
    exchange.submit("it7/pc-0abc", batch(tmp_path, "pc-0abc"), {**manifest, "folder": "it7/pc-0abc"})
    first, second = exchange.submissions()
    assert first.folder == "it7/mac-00ff" and first.manifest == manifest
    exchange.fetch(first, tmp_path / "runs" / "seg-mac-00ff")
    assert sorted(p.name for p in (tmp_path / "runs" / "seg-mac-00ff").iterdir()) == [
        "manifest.json", "search_1_00000.dst"]
    exchange.finish(first, True, "Collected.")
    exchange.finish(second, False, "Not played with the generator of its iteration.")
    assert list(exchange.submissions()) == []
    assert (tmp_path / "shared" / "rejected" / "it7--pc-0abc" / "reason.txt").is_file()


class FakeHub:
    def __init__(self, generator_sha):
        self.generator_sha, self.calls = generator_sha, []

    def model_info(self, repo):
        return SimpleNamespace(sha="rev123")

    def get_paths_info(self, repo, paths, revision):
        return [SimpleNamespace(path=paths[0], lfs=SimpleNamespace(sha256=self.generator_sha))]

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
        self.calls.append(("upload", path_in_repo, json.loads(path_or_fileobj), commit_message))

    def create_commit(self, repo, operations, commit_message, repo_type, create_pr):
        self.calls.append(("commit", [op.path_in_repo for op in operations], commit_message, create_pr))
        return SimpleNamespace(pr_url="https://huggingface.co/datasets/x/discussions/1")

    def change_discussion_status(self, repo, num, status, comment=None, repo_type=None):
        self.calls.append(("status", num, status))

    def merge_pull_request(self, repo, num, comment=None, repo_type=None):
        self.calls.append(("merge", num))


def test_hub_exchange_pins_the_generator_and_submits_pull_requests(tmp_path):
    generator = tmp_path / "net.onnx"
    generator.write_bytes(b"network")
    hub = FakeHub(sha256(generator))
    exchange = HubExchange(hub, "me/games", "me/net")
    exchange.publish({"protocol": PROTOCOL, "iteration": 7, "accepting": True, "games": 100}, generator)
    (_, path, order, message), = hub.calls
    assert path == "status.json" and message == "Work order: iteration 7, open"
    assert order["generator"] == {"repo": "me/net", "revision": "rev123", "file": "net.onnx",
                                  "name": "net.onnx", "sha256": sha256(generator)}

    exchange.submit("it7/mac-00ff", batch(tmp_path, "mac-00ff"), {"games": 3})
    assert hub.calls[-1] == ("commit", ["it7/mac-00ff/search_1_00000.dst", "it7/mac-00ff/manifest.json"],
                             "Self-play games: it7/mac-00ff", True)

    from hivemind.distill.exchange import Submission
    exchange.finish(Submission("it7/mac-00ff", {}, SimpleNamespace(num=4, status="draft")), True, "Collected.")
    exchange.finish(Submission("it7/pc-0abc", {}, SimpleNamespace(num=5, status="open")), False, "Stale.")
    assert hub.calls[-3:] == [("status", 4, "open"), ("merge", 4), ("status", 5, "closed")]


def test_hub_exchange_refuses_a_generator_the_model_repo_lacks(tmp_path):
    import pytest

    generator = tmp_path / "net.onnx"
    generator.write_bytes(b"promoted locally")
    exchange = HubExchange(FakeHub("0" * 64), "me/games", "me/net")
    with pytest.raises(RuntimeError):
        exchange.publish({"protocol": PROTOCOL, "iteration": 7, "accepting": True, "games": 100}, generator)
