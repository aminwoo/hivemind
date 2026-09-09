"""ONNX loading and input conversion shared by command-line inference tools."""

from pathlib import Path

import numpy as np
import onnxruntime as ort


def load_onnx_model(model_path):
    path = Path(model_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"ONNX model not found: {path}. Download it with python tools/fetch_network.py"
        )
    available = ort.get_available_providers()
    providers = [provider for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if provider in available]
    session = ort.InferenceSession(str(path), providers=providers)
    print(f"Loaded {path}; execution providers: {session.get_providers()}")
    return session


def run_onnx(session, planes):
    """Match the input precision, including networks with native FP16 inputs."""
    model_input = session.get_inputs()[0]
    dtypes = {"tensor(float)": np.float32, "tensor(float16)": np.float16}
    if model_input.type not in dtypes:
        raise ValueError(f"Unsupported model input type: {model_input.type}")
    batch = np.ascontiguousarray(planes, dtype=dtypes[model_input.type])
    return session.run(None, {model_input.name: batch})
