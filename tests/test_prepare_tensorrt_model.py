"""A TensorRT bundle must ship a network the engine accepts without Python."""
import importlib.util
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

SCRIPT = Path(__file__).resolve().parents[1] / "engine/scripts/prepare_tensorrt_model.py"
spec = importlib.util.spec_from_file_location("prepare_tensorrt_model", SCRIPT)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def _network(io_type, weight_type):
    """Like the portable network: FP16 weights, optionally behind FP32 I/O."""
    weight_dtype = np.float16 if weight_type == TensorProto.FLOAT16 else np.float32
    weight = numpy_helper.from_array(
        np.linspace(-1, 1, 8, dtype=weight_dtype).reshape(2, 4), "w")
    nodes = []
    x = "x"
    if io_type != weight_type:
        nodes.append(helper.make_node("Cast", ["x"], ["x_cast"], to=weight_type))
        x = "x_cast"
    nodes.append(helper.make_node("MatMul", [x, "w"], ["y_raw"]))
    nodes.append(helper.make_node("Cast", ["y_raw"], ["y"], to=io_type))
    graph = helper.make_graph(
        nodes, "net",
        [helper.make_tensor_value_info("x", io_type, ["batch", 2])],
        [helper.make_tensor_value_info("y", io_type, ["batch", 4])],
        [weight])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _run(path, x):
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return session.run(None, {"x": x})[0]


def test_fp32_io_network_is_converted_to_fp16(tmp_path):
    src, dst = tmp_path / "in.onnx", tmp_path / "out.onnx"
    onnx.save(_network(TensorProto.FLOAT, TensorProto.FLOAT16), src)
    assert prepare.fp32_tensors(onnx.load(src)) == ["x", "y"]

    assert prepare.prepare(str(src), str(dst))

    converted = onnx.load(dst)
    assert prepare.fp32_tensors(converted) == []
    x = np.array([[0.5, -0.25], [1.0, 2.0]], dtype=np.float32)
    np.testing.assert_allclose(
        _run(str(dst), x.astype(np.float16)), _run(str(src), x), rtol=1e-3)


def test_fp32_weights_are_converted_too(tmp_path):
    src, dst = tmp_path / "in.onnx", tmp_path / "out.onnx"
    onnx.save(_network(TensorProto.FLOAT, TensorProto.FLOAT), src)

    assert prepare.prepare(str(src), str(dst))
    assert prepare.fp32_tensors(onnx.load(dst)) == []


def test_fp16_network_is_copied_unchanged(tmp_path):
    src, dst = tmp_path / "in.onnx", tmp_path / "out.onnx"
    onnx.save(_network(TensorProto.FLOAT16, TensorProto.FLOAT16), src)

    assert not prepare.prepare(str(src), str(dst))
    assert dst.read_bytes() == src.read_bytes()


def test_leftover_fp32_fails_packaging(tmp_path, monkeypatch):
    src, dst = tmp_path / "in.onnx", tmp_path / "out.onnx"
    onnx.save(_network(TensorProto.FLOAT, TensorProto.FLOAT16), src)
    monkeypatch.setattr(prepare, "load_converter", lambda: _NoOpConverter)

    with pytest.raises(SystemExit, match="left FP32 tensors: x, y"):
        prepare.prepare(str(src), str(dst))


class _NoOpConverter:
    @staticmethod
    def convert(input_path, output_path):
        Path(output_path).write_bytes(Path(input_path).read_bytes())
