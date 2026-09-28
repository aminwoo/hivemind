#!/usr/bin/env python3
"""Quantize a Hivemind network to INT8 for the TensorRT backend.

TensorRT 11 only runs INT8 from explicit quantize/dequantize (Q/DQ) nodes in
the ONNX model, so the quantization happens here, ahead of time, with NVIDIA
ModelOpt. The result keeps FP16 everywhere it is not INT8, including its
inputs and outputs, which is what the engine's buffers and plan cache expect.

ModelOpt is not a project dependency; install it into its own environment:

    python3 -m venv ~/.cache/hivemind-quantize
    ~/.cache/hivemind-quantize/bin/pip install "nvidia-modelopt[onnx]"

Calibrate on positions encoded by the engine itself, so the activations match
what the search feeds the network (a few thousand are enough):

    ./build-ninja/hivemind dumpplanes --fens ../data/nnue/val/nnue_99_0_00000.fen \\
        --output calib.f32 --every 122
    ~/.cache/hivemind-quantize/bin/python scripts/quantize_int8.py \\
        models/network.onnx calib.f32 models/network-int8.onnx

Max calibration matters: entropy calibration clips activations this network
needs and cost board-B policy agreement (92% vs 77% top-1 against FP16).
Check a new model's outputs through TensorRT (trtexec --loadInputs and
--exportOutput), not ONNX Runtime, whose CPU provider mis-executes the mixed
FP16/Q-DQ graph.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto

sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert_onnx_fp32  # noqa: E402

PLANES_SHAPE = (74, 8, 8)


def fp16_interface(model: onnx.ModelProto) -> int:
    """Remove ModelOpt's boundary Casts so inputs and outputs are FP16.

    ModelOpt keeps the source model's FP32 interface around an FP16 graph: a
    Cast to FP16 after each input and a Cast to FP32 before each output.
    Returns the number of Casts removed.
    """
    graph = model.graph
    producers = {output: node for node in graph.node for output in node.output}
    removed = set()

    for graph_input in graph.input:
        for node in [n for n in graph.node if graph_input.name in n.input]:
            if node.op_type == "Cast" and node.attribute[0].i == TensorProto.FLOAT16:
                for other in graph.node:
                    other.input[:] = [graph_input.name if name == node.output[0] else name
                                      for name in other.input]
                removed.add(id(node))
        graph_input.type.tensor_type.elem_type = TensorProto.FLOAT16

    for graph_output in graph.output:
        node = producers[graph_output.name]
        if node.op_type == "Cast" and node.attribute[0].i == TensorProto.FLOAT:
            inner = node.input[0]
            for other in graph.node:
                other.output[:] = [graph_output.name if name == inner else name
                                   for name in other.output]
                other.input[:] = [graph_output.name if name == inner else name
                                  for name in other.input]
            removed.add(id(node))
        graph_output.type.tensor_type.elem_type = TensorProto.FLOAT16

    kept_nodes = [node for node in graph.node if id(node) not in removed]
    del graph.node[:]
    graph.node.extend(kept_nodes)
    kept_info = [info for info in graph.value_info
                 if info.type.tensor_type.elem_type != TensorProto.FLOAT]
    del graph.value_info[:]
    graph.value_info.extend(kept_info)
    return len(removed)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="FP16 or FP32 Hivemind ONNX network")
    parser.add_argument("calibration", type=Path,
                        help="float32 planes, N x 74 x 8 x 8, from `hivemind dumpplanes`")
    parser.add_argument("output", type=Path)
    parser.add_argument("--calibration-method", default="max", choices=["max", "entropy"])
    args = parser.parse_args()

    planes = np.fromfile(args.calibration, dtype=np.float32)
    if planes.size == 0 or planes.size % int(np.prod(PLANES_SHAPE)) != 0:
        raise SystemExit(f"{args.calibration} is not a float32 array of 74x8x8 planes")
    planes = planes.reshape(-1, *PLANES_SHAPE)
    print(f"calibrating on {len(planes)} positions")

    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        calibration = work / "calibration.npy"
        np.save(calibration, planes)
        fp32 = work / "model-fp32.onnx"
        convert_onnx_fp32.convert(str(args.model), str(fp32))
        quantized = work / "model-int8.onnx"
        subprocess.run(
            [sys.executable, "-m", "modelopt.onnx.quantization",
             "--onnx_path", str(fp32),
             "--quantize_mode", "int8",
             "--calibration_data", str(calibration),
             "--calibration_method", args.calibration_method,
             "--high_precision_dtype", "fp16",
             "--output_path", str(quantized)],
            check=True)
        model = onnx.load(quantized)

    casts = fp16_interface(model)
    onnx.checker.check_model(model)
    onnx.save(model, args.output)
    qdq = sum(node.op_type in ("QuantizeLinear", "DequantizeLinear") for node in model.graph.node)
    print(f"{args.output}: {qdq} Q/DQ nodes, {casts} boundary casts removed, FP16 interface")


if __name__ == "__main__":
    main()
