#!/usr/bin/env python3
"""Rewrite a Hivemind network so Core ML can run all of it on the Mac GPU.

ONNX Runtime's Core ML execution provider has no Expand, Neg or Abs kernel,
and the exported networks use them: every squeeze-excitation block of the
crossboard network broadcasts its gate with Expand(gate, Shape(x)) before
multiplying by x, both networks negate one WDL logit in the value head, and
twin-s feeds |A - B| of the two boards' value features to it. Each unsupported
node splits the graph, and every split costs a GPU -> CPU -> GPU round trip
per batch. All three rewrites are exact:

  * Mul(Expand(gate, Shape(x)), x) -> Mul(gate, x), since Mul broadcasts.
  * Neg(x) -> Mul(x, -1).
  * Abs(x) -> Max(x, Mul(x, -1)).

The output still runs on every backend. Keep the FP16 network as the input:
Core ML computes in FP16 on the GPU, while engine/scripts/convert_onnx_fp32.py
is only for the CPU provider.

    python3 engine/scripts/convert_onnx_coreml.py in.onnx out.onnx
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

# The engine reads this to tell whether a network has been through here.
METADATA_KEY = "hivemind_coreml"
BROADCASTING_OPS = {"Add", "Sub", "Mul", "Div"}


def _consumers(graph):
    consumers = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    return consumers


def _bypass_expands(graph) -> int:
    producers = {out: node for node in graph.node for out in node.output}
    consumers = _consumers(graph)
    graph_outputs = {output.name for output in graph.output}
    removed = []
    for node in graph.node:
        if node.op_type != "Expand" or node.output[0] in graph_outputs:
            continue
        shape = producers.get(node.input[1])
        if shape is None or shape.op_type != "Shape" or shape.attribute:
            continue
        users = consumers.get(node.output[0], [])
        # Dropping the Expand is exact only when each consumer broadcasts the
        # unexpanded tensor against the very tensor whose shape was taken.
        if not users or not all(
            user.op_type in BROADCASTING_OPS
            and len(user.input) == 2
            and shape.input[0] in user.input
            and user.input[0] != user.input[1]
            for user in users
        ):
            continue
        for user in users:
            for i, name in enumerate(user.input):
                if name == node.output[0]:
                    user.input[i] = node.input[0]
        removed.append(node)
    for node in removed:
        graph.node.remove(node)
    return len(removed)


def _elem_types(model) -> dict[str, int]:
    inferred = onnx.shape_inference.infer_shapes(model)
    graph = inferred.graph
    types = {value.name: value.type.tensor_type.elem_type
             for value in (list(graph.value_info) + list(graph.input)
                           + list(graph.output))
             if value.type.tensor_type.elem_type}
    types.update({tensor.name: tensor.data_type for tensor in graph.initializer})
    return types


def _replace_neg_and_abs(model) -> tuple[int, int]:
    graph = model.graph
    types = None
    counts = {"Neg": 0, "Abs": 0}
    nodes = []
    for node in graph.node:
        if node.op_type not in counts:
            nodes.append(node)
            continue
        if types is None:
            types = _elem_types(model)
        if node.input[0] not in types:
            raise SystemExit(f"Cannot infer the element type of {node.input[0]}")
        dtype = helper.tensor_dtype_to_np_dtype(types[node.input[0]])
        constant = f"{node.output[0]}_minus_one"
        graph.initializer.append(
            numpy_helper.from_array(np.array(-1, dtype=dtype), constant))
        if node.op_type == "Neg":
            nodes.append(helper.make_node(
                "Mul", [node.input[0], constant], list(node.output),
                name=node.name or None))
        else:
            negated = f"{node.output[0]}_negated"
            nodes.append(helper.make_node(
                "Mul", [node.input[0], constant], [negated],
                name=f"{node.name}_negate" if node.name else None))
            nodes.append(helper.make_node(
                "Max", [node.input[0], negated], list(node.output),
                name=node.name or None))
        counts[node.op_type] += 1
    del graph.node[:]
    graph.node.extend(nodes)
    return counts["Neg"], counts["Abs"]


def _prune_dead_nodes(graph) -> int:
    graph_outputs = {output.name for output in graph.output}
    pruned = 0
    while True:
        used = {name for node in graph.node for name in node.input} | graph_outputs
        dead = [node for node in graph.node
                if not any(out in used for out in node.output if out)]
        if not dead:
            return pruned
        for node in dead:
            graph.node.remove(node)
        pruned += len(dead)


def _coreml_supported_ops() -> set[str] | None:
    """Read ONNX Runtime's own Core ML op table, if onnxruntime is installed."""
    spec = importlib.util.find_spec("onnxruntime")
    if spec is None or spec.origin is None:
        return None
    table = (Path(spec.origin).parent / "tools" / "mobile_helpers"
             / "coreml_supported_mlprogram_ops.md")
    if not table.exists():
        return None
    ops = set()
    for line in table.read_text().splitlines():
        if line.startswith("|ai.onnx:"):
            ops.add(line.split("|")[1].removeprefix("ai.onnx:"))
    return ops


def convert(src: str, dst: str) -> None:
    model = onnx.load(src)
    graph = model.graph

    expands = _bypass_expands(graph)
    negs, abses = _replace_neg_and_abs(model)
    pruned = _prune_dead_nodes(graph)

    for prop in list(model.metadata_props):
        if prop.key == METADATA_KEY:
            model.metadata_props.remove(prop)
    model.metadata_props.add(key=METADATA_KEY, value="1")

    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, dst)
    print(f"{src} -> {dst}\n"
          f"  expands removed {expands}, negs replaced {negs}, "
          f"abs replaced {abses}, dead nodes pruned {pruned}")

    supported = _coreml_supported_ops()
    if supported is not None:
        # A Gather on a Shape is batch-size arithmetic that ONNX Runtime folds
        # into a constant once the engine fixes the batch size.
        producers = {out: node for node in graph.node for out in node.output}
        missing = sorted({
            node.op_type for node in graph.node
            if node.op_type not in supported
            and not (node.op_type == "Gather"
                     and getattr(producers.get(node.input[0]), "op_type", "")
                     == "Shape")
        })
        if missing:
            print("  warning: Core ML has no kernel for "
                  + ", ".join(missing)
                  + "; those nodes will run on the CPU")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    convert(sys.argv[1], sys.argv[2])
