#!/usr/bin/env python3

import argparse

import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16


def topologically_sort_graph(model: onnx.ModelProto) -> None:
    available_values = {
        value.name for value in [*model.graph.input, *model.graph.initializer]
    }
    pending_nodes = list(model.graph.node)
    sorted_nodes = []
    while pending_nodes:
        ready_nodes = [
            node
            for node in pending_nodes
            if all(not name or name in available_values for name in node.input)
        ]
        if not ready_nodes:
            raise RuntimeError("FP16 ONNX conversion produced an unsortable graph")
        for node in ready_nodes:
            sorted_nodes.append(node)
            available_values.update(name for name in node.output if name)
            pending_nodes.remove(node)

    del model.graph.node[:]
    model.graph.node.extend(sorted_nodes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    model = onnx.load(args.input)
    model_fp16 = convert_float_to_float16(model, keep_io_types=False)
    topologically_sort_graph(model_fp16)
    onnx.checker.check_model(model_fp16)
    onnx.save(model_fp16, args.output)


if __name__ == "__main__":
    main()