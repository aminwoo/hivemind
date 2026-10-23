#!/usr/bin/env python3
"""Write the FP16 network a TensorRT bundle ships.

The engine builds a TensorRT plan only from a network whose tensors are all
FP16. Given anything else it runs convert_onnx_fp16.py on the user's machine,
which needs a Python with onnx and onnxruntime that release users do not have,
so the first launch fails. The portable network keeps FP16 weights behind FP32
inputs and outputs, which is enough to trigger that path. Converting here, at
packaging time, lets every bundle start without Python.

    python3 engine/scripts/prepare_tensorrt_model.py in.onnx out.onnx
"""
import importlib.util
import shutil
import sys
from pathlib import Path

import onnx
from onnx import TensorProto

CONVERTER = Path(__file__).resolve().parent / "convert_onnx_fp16.py"


def fp32_tensors(model: onnx.ModelProto) -> list[str]:
    graph = model.graph
    values = [*graph.input, *graph.output]
    names = [value.name for value in values
             if value.type.tensor_type.elem_type == TensorProto.FLOAT]
    names += [tensor.name for tensor in graph.initializer
              if tensor.data_type == TensorProto.FLOAT]
    return names


def load_converter():
    spec = importlib.util.spec_from_file_location("convert_onnx_fp16", CONVERTER)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    return converter


def prepare(input_path: str, output_path: str) -> bool:
    """Writes an FP16 copy of the network; returns whether it converted."""
    if not fp32_tensors(onnx.load(input_path)):
        shutil.copyfile(input_path, output_path)
        return False

    load_converter().convert(input_path, output_path)

    remaining = fp32_tensors(onnx.load(output_path))
    if remaining:
        raise SystemExit(
            f"FP16 conversion left FP32 tensors: {', '.join(remaining)}")
    return True


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} in.onnx out.onnx")
    converted = prepare(sys.argv[1], sys.argv[2])
    print(("Converted" if converted else "Copied already-FP16")
          + f" network to {sys.argv[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
