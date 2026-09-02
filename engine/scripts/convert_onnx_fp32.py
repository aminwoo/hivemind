#!/usr/bin/env python3
"""Convert an FP16 Hivemind network to FP32.

The released networks are FP16 because TensorRT wants them that way. ONNX
Runtime's CPU provider has no native FP16 kernels, so it wraps every node in
cast pairs and runs orders of magnitude slower. Converting once, ahead of time,
is the difference between a usable engine and an unusable one.

The file roughly doubles in size (27 MB -> 54 MB); that is the cost of a
portable backend.

    python3 engine/scripts/convert_onnx_fp32.py in.onnx out.onnx
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

F16, F32 = TensorProto.FLOAT16, TensorProto.FLOAT


def _convert_tensor(tensor) -> bool:
    if tensor.data_type != F16:
        return False
    array = numpy_helper.to_array(tensor).astype(np.float32)
    tensor.CopyFrom(numpy_helper.from_array(array, tensor.name))
    return True


def _convert_value_info(value_info) -> bool:
    tensor_type = value_info.type.tensor_type
    if tensor_type.elem_type != F16:
        return False
    tensor_type.elem_type = F32
    return True


def convert(src: str, dst: str) -> None:
    model = onnx.load(src)
    graph = model.graph

    initializers = sum(_convert_tensor(t) for t in graph.initializer)
    value_infos = sum(
        _convert_value_info(v)
        for v in list(graph.input) + list(graph.output) + list(graph.value_info)
    )

    casts = attrs = 0
    for node in graph.node:
        for attribute in node.attribute:
            # Retarget explicit Cast-to-fp16 nodes.
            if attribute.name == "to" and attribute.i == F16:
                attribute.i = F32
                casts += 1
            if attribute.type == onnx.AttributeProto.TENSOR:
                attrs += _convert_tensor(attribute.t)
            elif attribute.type == onnx.AttributeProto.TENSORS:
                for tensor in attribute.tensors:
                    attrs += _convert_tensor(tensor)

    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, dst)
    print(
        f"{src} -> {dst}\n"
        f"  initializers {initializers}, value infos {value_infos}, "
        f"casts {casts}, attribute tensors {attrs}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    convert(sys.argv[1], sys.argv[2])
