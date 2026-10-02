"""Reader for the engine's `gennnue` chunks (HNUE v1)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hivemind.nnue.features import NUM_FEATURES

MAGIC = b"HNUE"
VERSION = 1
OUTCOME_UNKNOWN = 2


@dataclass
class NnueChunk:
    value: np.ndarray      # f32[N], teacher value for the side to move
    wdl: np.ndarray        # f32[N, 3], teacher loss/draw/win probabilities
    outcome: np.ndarray    # i8[N], -1/0/1 game result, 2 unfinished
    ply: np.ndarray        # u16[N]
    counts: np.ndarray     # u8[N], active features per position
    features: np.ndarray   # u16[F], side-to-move feature indices, concatenated

    def __len__(self):
        return len(self.value)


def read_chunk_header(path) -> tuple[int, int]:
    """(positions, total active features) without reading the body."""
    with open(path, "rb") as stream:
        raw = stream.read(28)
    if raw[:4] != MAGIC:
        raise ValueError(f"{path}: not an HNUE chunk")
    n, total = np.frombuffer(raw, np.uint64, 2, 12)
    return int(n), int(total)


def read_chunk(path) -> NnueChunk:
    raw = Path(path).read_bytes()
    if raw[:4] != MAGIC:
        raise ValueError(f"{path}: not an HNUE chunk")
    version, feature_space = np.frombuffer(raw, np.uint32, 2, 4)
    if version != VERSION or feature_space != NUM_FEATURES:
        raise ValueError(f"{path}: version {version}, {feature_space} features; "
                         f"expected {VERSION}, {NUM_FEATURES}")
    n, total = (int(x) for x in np.frombuffer(raw, np.uint64, 2, 12))
    offset = 28

    def take(dtype, count):
        nonlocal offset
        array = np.frombuffer(raw, dtype, count, offset)
        offset += array.nbytes
        return array

    chunk = NnueChunk(
        value=take(np.float32, n),
        wdl=take(np.float32, 3 * n).reshape(n, 3),
        outcome=take(np.int8, n),
        ply=take(np.uint16, n),
        counts=take(np.uint8, n),
        features=take(np.uint16, total),
    )
    if offset != len(raw) or int(chunk.counts.sum(dtype=np.int64)) != total:
        raise ValueError(f"{path}: truncated or inconsistent chunk")
    return chunk


def chunk_paths(directories) -> list[Path]:
    paths = []
    for directory in directories:
        directory = Path(directory)
        paths.extend(sorted(directory.glob("*.bin")) if directory.is_dir() else [directory])
    return paths


def concatenate(chunks: list[NnueChunk]) -> NnueChunk:
    return NnueChunk(*(np.concatenate([getattr(c, name) for c in chunks])
                       for name in ("value", "wdl", "outcome", "ply", "counts", "features")))
