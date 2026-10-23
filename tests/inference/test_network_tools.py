"""Regression coverage for downloaded artifacts and ONNX inference precision."""

import hashlib
import io
import inspect
import sys
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from hivemind.inference.onnx import run_onnx
from hivemind.inference.onnx_graph import topologically_sort_graph
from hivemind.cli import fetch_network as downloader
from hivemind.network import Artifact


@pytest.mark.parametrize("tensor_type,dtype", [
    ("tensor(float)", np.float32), ("tensor(float16)", np.float16),
])
def test_inference_matches_model_input_precision(tensor_type, dtype):
    class Session:
        def get_inputs(self):
            return [SimpleNamespace(name="data", type=tensor_type)]

        def run(self, outputs, inputs):
            assert inputs["data"].dtype == dtype
            assert inputs["data"].flags.c_contiguous
            return [inputs["data"]]

    source = np.zeros((2, 74, 8, 8), dtype=np.float64)
    result = run_onnx(Session(), source)
    np.testing.assert_array_equal(result[0], source)


def test_graph_sort_orders_dependencies_and_preserves_outputs():
    graph = helper.make_graph(
        [helper.make_node("Identity", ["middle"], ["output"]),
         helper.make_node("Identity", ["input"], ["middle"])],
        "unsorted",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph)
    topologically_sort_graph(model)
    onnx.checker.check_model(model)
    assert model.graph.node[0].output == ["middle"]


def test_graph_sort_rejects_missing_dependency():
    model = helper.make_model(helper.make_graph(
        [helper.make_node("Identity", ["missing"], ["output"])], "invalid", [], [],
    ))
    with pytest.raises(RuntimeError, match="unsortable"):
        topologically_sort_graph(model)


def test_download_verifies_and_reuses_existing_artifact(tmp_path, monkeypatch):
    payload = b"test network"
    monkeypatch.setitem(downloader.ARTIFACTS, "test", Artifact(
        "owner/repo", "abc123", "network.onnx", hashlib.sha256(payload).hexdigest()))
    calls = []

    def open_url(url, timeout):
        calls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(downloader, "urlopen", open_url)
    path = downloader.fetch_network("test", tmp_path)
    assert path.read_bytes() == payload
    assert downloader.fetch_network("test", tmp_path) == path
    assert calls == ["https://huggingface.co/owner/repo/resolve/abc123/network.onnx"]


def test_corrupt_download_preserves_existing_file_and_cleans_temporary(tmp_path, monkeypatch):
    path = tmp_path / "network.onnx"
    path.write_bytes(b"existing")
    monkeypatch.setitem(downloader.ARTIFACTS, "test", Artifact(
        "owner/repo", "abc123", path.name, "incorrect digest"))
    monkeypatch.setattr(downloader, "urlopen", lambda *a, **kw: io.BytesIO(b"corrupt"))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        downloader.fetch_network("test", tmp_path)
    assert path.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("variant", [None, "onnx"])
def test_cli_download_defaults_to_clock_and_accepts_legacy_override(tmp_path, monkeypatch, capsys, variant):
    selected = variant or "clock"
    payload = b"verified network"
    artifact = downloader.ARTIFACTS[selected]._replace(sha256=hashlib.sha256(payload).hexdigest())
    monkeypatch.setitem(downloader.ARTIFACTS, selected, artifact)
    urls = []

    def open_url(url, timeout):
        urls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(downloader, "urlopen", open_url)
    argv = ["fetch-network", "--output-dir", str(tmp_path)]
    if variant:
        argv += ["--variant", variant]
    monkeypatch.setattr(sys, "argv", argv)
    downloader.main()
    expected = tmp_path / artifact.filename
    assert expected.read_bytes() == payload
    assert str(expected) in capsys.readouterr().out
    assert urls == [f"https://huggingface.co/{artifact.repository}/resolve/{artifact.revision}/{artifact.filename}"]


def test_download_api_and_inference_use_the_same_clock_default(tmp_path, monkeypatch):
    from hivemind.network import DEFAULT_ONNX_PATH

    payload = b"clock network"
    artifact = downloader.ARTIFACTS["clock"]._replace(sha256=hashlib.sha256(payload).hexdigest())
    monkeypatch.setitem(downloader.ARTIFACTS, "clock", artifact)
    monkeypatch.setattr(downloader, "urlopen", lambda *a, **kw: io.BytesIO(payload))
    downloaded = downloader.fetch_network(output_dir=tmp_path)
    assert downloaded.name == DEFAULT_ONNX_PATH.name == "twin-s-clock.onnx"
    assert downloaded.read_bytes() == payload


def test_twin_checkpoint_installs_where_training_continues_it():
    from hivemind.cli import rl_loop, train
    from hivemind.network import ARTIFACTS, DEFAULT_ONNX_PATH, DESTINATIONS, TWIN_CHECKPOINT_PATH, TWIN_ONNX_PATH

    assert ARTIFACTS["checkpoint"].revision == ARTIFACTS["onnx"].revision
    assert DESTINATIONS["checkpoint"] == TWIN_CHECKPOINT_PATH
    assert TWIN_ONNX_PATH.name == ARTIFACTS["onnx"].filename
    assert TWIN_ONNX_PATH != DEFAULT_ONNX_PATH
    assert inspect.signature(train.generator_checkpoint).parameters["generator"].default == TWIN_ONNX_PATH
    assert inspect.signature(rl_loop.Loop).parameters["generator"].default == TWIN_ONNX_PATH
