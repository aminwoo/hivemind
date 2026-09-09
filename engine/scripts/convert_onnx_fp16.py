#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

source_dir = Path(__file__).resolve().parents[2] / "src"
if source_dir.is_dir():
    sys.path.insert(0, str(source_dir))
from hivemind.inference.onnx_graph import topologically_sort_graph

import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16


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