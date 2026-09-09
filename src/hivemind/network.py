"""Pinned public network artifacts and project-relative defaults."""

from hivemind.paths import MODEL_DIRECTORY

MODEL_REPOSITORY = "aminwoo/bughouse-rise-v3"
MODEL_REVISION = "eac7b7a415061b63bdce672d064aa5b8030e9ae8"
MODEL_NAME = "hivemind-it04-crossboard-risev33-loss1.556-p82.0"
# SHA-256 values from the Hugging Face LFS metadata at MODEL_REVISION.
ARTIFACTS = {
    "onnx": (f"{MODEL_NAME}.onnx", "5adb5f450d34098e089a6f1bf275ef31f1be8569a11aa37b3d92183b840883cc"),
    "fp16": (f"{MODEL_NAME}_fp16.onnx", "a0ec548c21a2b001642f99f9347c3b0b512cd6a5a8b5a00e5388bb0e5898149e"),
    "checkpoint": (f"{MODEL_NAME}.tar", "6dd470599ee8e0d168911336126eec6c6f9bed51915cc2ff9e3935ecdd8160e2"),
}
DEFAULT_ONNX_PATH = MODEL_DIRECTORY / ARTIFACTS["onnx"][0]
DEFAULT_CHECKPOINT_PATH = MODEL_DIRECTORY / ARTIFACTS["checkpoint"][0]
