"""Reader for the engine's `gennnue --format distill` chunks (HDST v2)."""

from pathlib import Path

import numpy as np
import torch

MAGIC = b"HDST"
VERSION = 2
HEADER_BYTES = 28
PLANES = 74
POLICY_SIZE = 4672


def _header(raw, path):
    if raw[:4] != MAGIC:
        raise ValueError(f"{path}: not an HDST chunk")
    version, planes = np.frombuffer(raw, np.uint32, 2, 4)
    if version != VERSION or planes != PLANES:
        raise ValueError(f"{path}: version {version}, {planes} planes (expected {VERSION}, {PLANES})")
    n, entries = (int(v) for v in np.frombuffer(raw, np.uint64, 2, 12))
    return n, entries


def chunk_positions(path):
    """Positions in a chunk, from its header alone."""
    with open(path, "rb") as stream:
        return _header(stream.read(HEADER_BYTES), path)[0]


def read_chunk(path):
    raw = Path(path).read_bytes()
    n, entries = _header(raw, path)
    offset = HEADER_BYTES

    def take(dtype, count):
        nonlocal offset
        array = np.frombuffer(raw, dtype, count, offset)
        offset += array.nbytes
        return array

    chunk = {
        "plane_bits": take(np.int64, n * PLANES).reshape(n, PLANES),
        "plane_values": take(np.float16, n * PLANES).reshape(n, PLANES),
        "value": take(np.float32, n),
        "wdl": take(np.float32, 3 * n).reshape(n, 3),
        "moves_left": take(np.float16, n),
        "ply": take(np.uint16, n),
        "outcome": take(np.int8, n),
        "policy_count": take(np.uint16, 2 * n).reshape(n, 2),
        "policy_index": take(np.uint16, entries),
        "policy_probability": take(np.float16, entries),
    }
    if offset != len(raw) or int(chunk["policy_count"].sum()) != entries:
        raise ValueError(f"{path}: truncated or inconsistent chunk")
    return chunk


def head_chunk(chunk, count):
    """The first `count` positions of a chunk (policy entries follow the counts)."""
    count = min(count, len(chunk["value"]))
    entries = int(chunk["policy_count"][:count].astype(np.int64).sum())
    per_position = {"plane_bits", "plane_values", "value", "wdl", "moves_left", "ply", "outcome",
                    "policy_count"}
    return {key: array[:count] if key in per_position else array[:entries]
            for key, array in chunk.items()}


def chunk_paths(sources):
    paths = []
    for source in sources:
        source = Path(source)
        paths.extend(sorted(source.glob("*.dst")) if source.is_dir() else [source])
    return paths


class DistillDataset:
    """Positions on one device; planes are rebuilt from bitboards per batch."""

    def __init__(self, chunks, device, source=0):
        """`source` tags every row (0 = teacher data, 1 = search data) so the
        loss can weight the two kinds of target separately."""
        cat = lambda key: np.concatenate([c[key] for c in chunks])
        self.device = device
        self.source = source
        self.plane_bits = torch.from_numpy(cat("plane_bits")).to(device)
        self.plane_values = torch.from_numpy(cat("plane_values").copy()).to(device)
        self.value = torch.from_numpy(cat("value").copy()).to(device)
        self.wdl = torch.from_numpy(cat("wdl").copy()).to(device)
        self.moves_left = torch.from_numpy(cat("moves_left").astype(np.float32)).to(device)
        counts = torch.from_numpy(cat("policy_count").astype(np.int64)).to(device)
        self.policy_counts = counts                                  # [N, 2]
        flat = counts.flatten()
        self.policy_starts = (torch.cumsum(flat, 0) - flat).view(-1, 2)
        # Compact on the device (indices fit int16); widened per batch.
        self.policy_index = torch.from_numpy(cat("policy_index").astype(np.int16)).to(device)
        self.policy_probability = torch.from_numpy(cat("policy_probability").copy()).to(device)
        self.shifts = torch.arange(64, device=device, dtype=torch.int64)

    def __len__(self):
        return len(self.value)

    def planes(self, index):
        bits = self.plane_bits[index]                                # [B, 74]
        values = self.plane_values[index].float()
        squares = ((bits.unsqueeze(-1) >> self.shifts) & 1).float()  # [B, 74, 64]
        filled = torch.where(bits.unsqueeze(-1) != 0, squares, torch.ones_like(squares))
        return (filled * values.unsqueeze(-1)).view(-1, PLANES, 8, 8)

    def policy_targets(self, index, board):
        """One board's teacher policy as dense [B, 4672] targets.

        Returns (target, legal, mask): the teacher's probabilities (summing to
        one), which policy indices are legal moves or the pass, and which rows
        have a board on turn at all. Rows without one are marked all-legal so a
        legal-only softmax stays finite; their loss is masked out.
        """
        counts = self.policy_counts[index, board]
        starts = self.policy_starts[index, board]
        total = int(counts.sum())
        rows = torch.repeat_interleave(torch.arange(len(index), device=self.device), counts,
                                       output_size=total)
        offsets = torch.cumsum(counts, 0) - counts
        within = torch.arange(total, device=self.device) - torch.repeat_interleave(
            offsets, counts, output_size=total)
        entries = torch.repeat_interleave(starts, counts, output_size=total) + within
        columns = self.policy_index[entries].long()
        target = torch.zeros(len(index), POLICY_SIZE, device=self.device)
        target.index_put_((rows, columns), self.policy_probability[entries].float(), accumulate=True)
        legal = torch.zeros(len(index), POLICY_SIZE, dtype=torch.bool, device=self.device)
        legal[rows, columns] = True
        mask = counts > 0
        legal[~mask] = True
        mass = target.sum(1, keepdim=True)
        return target / mass.clamp_min(1e-8), legal, mask

    def batch(self, index):
        return {
            "planes": self.planes(index),
            "value": self.value[index],
            "wdl": self.wdl[index],
            "moves_left": self.moves_left[index],
            "policy": [self.policy_targets(index, board) for board in (0, 1)],
            "source": torch.full((len(index),), self.source, device=self.device, dtype=torch.int8),
        }
