"""Pinned public network artifacts and project-relative defaults."""

import subprocess
import sys
from typing import NamedTuple

from hivemind.paths import MODEL_DIRECTORY, PROJECT_ROOT

COREML_CONVERTER = PROJECT_ROOT / "engine" / "scripts" / "convert_onnx_coreml.py"
FP32_CONVERTER = PROJECT_ROOT / "engine" / "scripts" / "convert_onnx_fp32.py"


class Artifact(NamedTuple):
    repository: str
    revision: str
    filename: str
    sha256: str


# Legacy twin-s-noattn after RL iteration 4: it beat
# the distilled iteration 0 99-57-4 at 100 ms per move (+93 Elo) and searches
# about 3x as many nodes per second as the network it was distilled from. Its training
# checkpoint, which `hivemind train` and `hivemind rl-loop` continue, is pinned
# at the same revision.
TWIN_REPOSITORY = "aminwoo/bughouse-twin-s"
TWIN_REVISION = "c9603be902877d163b892b71a4e39bde045e300e"
# Where `hivemind rl-loop --distributed` publishes work orders and contributors
# (`hivemind contribute`) submit self-play games: a Hugging Face dataset.
SELFPLAY_EXCHANGE = f"hf://{TWIN_REPOSITORY}-selfplay"
# Where `hivemind rl-loop` moves finished self-play, and `hivemind archive`
# any data directory, as zstd tarballs: a private Hugging Face dataset.
SELFPLAY_ARCHIVE = f"{TWIN_REPOSITORY}-selfplay-archive"
# SHA-256 values from the Hugging Face LFS metadata at each revision.
ARTIFACTS = {
    "clock": Artifact("aminwoo/bughouse-twin-s-clock",
                      "30995e780f192d56393cce94acdae234f25f6331", "twin-s-clock.onnx",
                      "40b1f9b6c6e875a068372a6510d060d14acc5c85a8c88527d39d0194e12b2ab4"),
    "onnx": Artifact(TWIN_REPOSITORY, TWIN_REVISION, "twin-s-noattn.onnx",
                     "efef8ca12634a26d55f8d34313459d36ae1469411208384d5c5f636bc43742c2"),
    "checkpoint": Artifact(TWIN_REPOSITORY, TWIN_REVISION, "twin-s-noattn.pt",
                           "89d67816acf331f9141a8ebbffe379e16642de17fe06abdd78194f87939bc05e"),
}
DEFAULT_NETWORK_VARIANT = "clock"
DEFAULT_ONNX_PATH = MODEL_DIRECTORY / ARTIFACTS[DEFAULT_NETWORK_VARIANT].filename
# The legacy training generator must stay paired with its published checkpoint.
TWIN_ONNX_PATH = MODEL_DIRECTORY / ARTIFACTS["onnx"].filename
# The generator's training checkpoint, where the RL loop promotes each new one.
TWIN_CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "distill" / "twin-s-noattn" / "best.pt"
# Variants installed somewhere other than MODEL_DIRECTORY by default.
DESTINATIONS = {"checkpoint": TWIN_CHECKPOINT_PATH}


def for_this_machine(model):
    """On macOS, the network rewritten so Core ML runs all of it on the GPU
    (the outputs are identical), converted next to it on first use."""
    if sys.platform != "darwin":
        return model
    converted = model.with_name(f"{model.stem}-coreml{model.suffix}")
    if not converted.is_file():
        subprocess.run([sys.executable, str(COREML_CONVERTER), str(model), str(converted)], check=True)
    return converted


def for_cpu(model):
    """The network converted to FP32 for ONNX Runtime's CPU provider, which
    has no FP16 kernels, converted next to it on first use."""
    converted = model.with_name(f"{model.stem}-fp32{model.suffix}")
    if not converted.is_file():
        subprocess.run([sys.executable, str(FP32_CONVERTER), str(model), str(converted)], check=True)
    return converted
