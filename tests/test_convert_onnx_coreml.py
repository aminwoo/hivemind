"""The Core ML rewrite must remove the ops Core ML lacks without changing outputs."""
import importlib.util
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

SCRIPT = Path(__file__).resolve().parents[1] / "engine/scripts/convert_onnx_coreml.py"
spec = importlib.util.spec_from_file_location("convert_onnx_coreml", SCRIPT)
coreml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coreml)


def _se_block(gate_shape_source: str = "x", elem_type=TensorProto.FLOAT):
    """x -> squeeze-excitation gate broadcast with Expand(gate, Shape(.)) -> Neg."""
    nodes = [
        helper.make_node("GlobalAveragePool", ["x"], ["pooled"]),
        helper.make_node("HardSigmoid", ["pooled"], ["gate"]),
        helper.make_node("Shape", [gate_shape_source], ["shape"]),
        helper.make_node("Expand", ["gate", "shape"], ["expanded"]),
        helper.make_node("Mul", ["x", "expanded"], ["scaled"]),
        helper.make_node("Neg", ["scaled"], ["y"]),
    ]
    inputs = [helper.make_tensor_value_info("x", elem_type, ["batch", 4, 2, 2])]
    if gate_shape_source != "x":
        inputs.append(helper.make_tensor_value_info(
            gate_shape_source, elem_type, ["batch", 4, 2, 2]))
    graph = helper.make_graph(
        nodes, "se", inputs,
        [helper.make_tensor_value_info("y", elem_type, ["batch", 4, 2, 2])])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _convert(tmp_path, model):
    src, dst = tmp_path / "in.onnx", tmp_path / "out.onnx"
    onnx.save(model, src)
    coreml.convert(str(src), str(dst))
    return src, dst, onnx.load(dst)


def _run(path, feeds):
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return session.run(None, feeds)[0]


def test_rewrite_removes_expand_and_neg_and_keeps_outputs(tmp_path):
    src, dst, converted = _convert(tmp_path, _se_block())

    ops = [node.op_type for node in converted.graph.node]
    assert "Expand" not in ops and "Neg" not in ops and "Shape" not in ops
    assert {p.key: p.value for p in converted.metadata_props} == {
        coreml.METADATA_KEY: "1"}

    x = np.random.default_rng(0).standard_normal((3, 4, 2, 2)).astype(np.float32)
    np.testing.assert_array_equal(_run(dst, {"x": x}), _run(src, {"x": x}))


def test_expand_against_another_tensor_is_kept(tmp_path):
    # Mul(x, Expand(gate, Shape(other))) is not Mul(x, gate) in general.
    _, _, converted = _convert(tmp_path, _se_block(gate_shape_source="other"))
    assert "Expand" in [node.op_type for node in converted.graph.node]


@pytest.mark.parametrize("elem_type", [TensorProto.FLOAT, TensorProto.FLOAT16])
def test_negation_constant_matches_the_graph_precision(tmp_path, elem_type):
    _, _, converted = _convert(tmp_path, _se_block(elem_type=elem_type))
    onnx.shape_inference.infer_shapes(converted, strict_mode=True)
    (minus_one,) = [t for t in converted.graph.initializer
                    if t.name.endswith("_minus_one")]
    assert minus_one.data_type == elem_type


def test_abs_becomes_max_of_value_and_negation(tmp_path):
    # twin-s feeds |A - B| of the two boards' value features to its head.
    graph = helper.make_graph(
        [helper.make_node("Sub", ["a", "b"], ["diff"]),
         helper.make_node("Abs", ["diff"], ["y"])],
        "value_diff",
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch", 8])
         for name in ("a", "b")],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 8])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    src, dst, converted = _convert(tmp_path, model)

    ops = [node.op_type for node in converted.graph.node]
    assert "Abs" not in ops and "Max" in ops

    rng = np.random.default_rng(1)
    feeds = {name: rng.standard_normal((3, 8)).astype(np.float32) for name in "ab"}
    session = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    np.testing.assert_array_equal(session.run(None, feeds)[0],
                                  ort.InferenceSession(src, providers=["CPUExecutionProvider"])
                                  .run(None, feeds)[0])


def test_converting_twice_keeps_one_marker(tmp_path):
    _, dst, _ = _convert(tmp_path, _se_block())
    again = tmp_path / "again.onnx"
    coreml.convert(str(dst), str(again))
    keys = [p.key for p in onnx.load(again).metadata_props]
    assert keys.count(coreml.METADATA_KEY) == 1
