"""Pinned public network artifacts and project-relative defaults."""

from typing import NamedTuple

from hivemind.paths import MODEL_DIRECTORY, PROJECT_ROOT


class Artifact(NamedTuple):
    repository: str
    revision: str
    filename: str
    sha256: str


# twin-s-noattn after RL iteration 4, the strongest network measured: it beat
# the distilled iteration 0 99-57-4 at 100 ms per move (+93 Elo) and searches
# about 3x as many nodes per second as the crossboard teacher. Its training
# checkpoint, which `hivemind train` and `hivemind rl-loop` continue, is pinned
# at the same revision.
TWIN_REPOSITORY = "aminwoo/bughouse-twin-s"
TWIN_REVISION = "c9603be902877d163b892b71a4e39bde045e300e"
# Where `hivemind rl-loop --distributed` publishes work orders and contributors
# (`hivemind contribute`) submit self-play games: a Hugging Face dataset.
SELFPLAY_EXCHANGE = f"hf://{TWIN_REPOSITORY}-selfplay"
# The crossboard-risev33 teacher twin-s was distilled from. Its training
# checkpoint is the only one published, and checkpoint inference loads only
# the RISE architectures.
CROSSBOARD_REPOSITORY = "aminwoo/bughouse-rise-v3"
CROSSBOARD_REVISION = "eac7b7a415061b63bdce672d064aa5b8030e9ae8"
CROSSBOARD_NAME = "hivemind-it04-crossboard-risev33-loss1.556-p82.0"
# SHA-256 values from the Hugging Face LFS metadata at each revision.
ARTIFACTS = {
    "onnx": Artifact(TWIN_REPOSITORY, TWIN_REVISION, "twin-s-noattn.onnx",
                     "efef8ca12634a26d55f8d34313459d36ae1469411208384d5c5f636bc43742c2"),
    "checkpoint": Artifact(TWIN_REPOSITORY, TWIN_REVISION, "twin-s-noattn.pt",
                           "89d67816acf331f9141a8ebbffe379e16642de17fe06abdd78194f87939bc05e"),
    "crossboard": Artifact(CROSSBOARD_REPOSITORY, CROSSBOARD_REVISION, f"{CROSSBOARD_NAME}.onnx",
                           "5adb5f450d34098e089a6f1bf275ef31f1be8569a11aa37b3d92183b840883cc"),
    "crossboard-fp16": Artifact(CROSSBOARD_REPOSITORY, CROSSBOARD_REVISION, f"{CROSSBOARD_NAME}_fp16.onnx",
                                "a0ec548c21a2b001642f99f9347c3b0b512cd6a5a8b5a00e5388bb0e5898149e"),
    "crossboard-checkpoint": Artifact(CROSSBOARD_REPOSITORY, CROSSBOARD_REVISION, f"{CROSSBOARD_NAME}.tar",
                                      "6dd470599ee8e0d168911336126eec6c6f9bed51915cc2ff9e3935ecdd8160e2"),
}
DEFAULT_ONNX_PATH = MODEL_DIRECTORY / ARTIFACTS["onnx"].filename
DEFAULT_CHECKPOINT_PATH = MODEL_DIRECTORY / ARTIFACTS["crossboard-checkpoint"].filename
# The generator's training checkpoint, where the RL loop promotes each new one.
TWIN_CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "distill" / "twin-s-noattn" / "best.pt"
# Variants installed somewhere other than MODEL_DIRECTORY by default.
DESTINATIONS = {"checkpoint": TWIN_CHECKPOINT_PATH}
