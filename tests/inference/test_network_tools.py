"""Regression coverage for downloaded artifacts and ONNX inference precision."""

import hashlib
import io
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from hivemind.config.train_config import TrainObjects
from hivemind.inference.onnx import run_onnx
from hivemind.inference.onnx_graph import topologically_sort_graph
from hivemind.cli import fetch_network as downloader


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


def test_training_objects_do_not_share_mutable_settings():
    first, second = TrainObjects(), TrainObjects()
    first.phase_weights[0] = 0
    assert second.phase_weights[0] == 1


def test_download_verifies_and_reuses_existing_artifact(tmp_path, monkeypatch):
    payload = b"test network"
    monkeypatch.setitem(downloader.ARTIFACTS, "test", ("network.onnx", hashlib.sha256(payload).hexdigest()))
    calls = []

    def open_url(url, timeout):
        calls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(downloader, "urlopen", open_url)
    path = downloader.fetch_network("test", tmp_path)
    assert path.read_bytes() == payload
    assert downloader.fetch_network("test", tmp_path) == path
    assert len(calls) == 1


def test_corrupt_download_preserves_existing_file_and_cleans_temporary(tmp_path, monkeypatch):
    path = tmp_path / "network.onnx"
    path.write_bytes(b"existing")
    monkeypatch.setitem(downloader.ARTIFACTS, "test", (path.name, "incorrect digest"))
    monkeypatch.setattr(downloader, "urlopen", lambda *a, **kw: io.BytesIO(b"corrupt"))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        downloader.fetch_network("test", tmp_path)
    assert path.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [path]
