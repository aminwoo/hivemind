import struct
import sys

import numpy as np

from hivemind.nnue.data import MAGIC, VERSION, read_chunk
from hivemind.nnue.features import NUM_FEATURES


def write_chunk(path, rng, positions=512):
    counts = rng.integers(40, 80, positions).astype(np.uint8)
    features = rng.integers(0, NUM_FEATURES, int(counts.sum())).astype(np.uint16)
    value = np.tanh(rng.normal(size=positions)).astype(np.float32)
    wdl = np.full((positions, 3), 1 / 3, dtype=np.float32)
    with open(path, "wb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("<II", VERSION, NUM_FEATURES))
        stream.write(struct.pack("<QQ", positions, len(features)))
        for array in (value, wdl, np.zeros(positions, np.int8), np.zeros(positions, np.uint16), counts, features):
            stream.write(array.tobytes())


def test_chunk_round_trip_and_training_exports_a_network(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    for index in range(3):
        write_chunk(tmp_path / "train" / f"c{index}.bin", rng)
    write_chunk(tmp_path / "val" / "v.bin", rng)
    assert len(read_chunk(tmp_path / "val" / "v.bin")) == 512

    from hivemind.nnue import train
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "nnue-train", "--data", str(tmp_path / "train"), "--val", str(tmp_path / "val"),
        "--out", str(out), "--hidden", "32", "--l1", "8", "--l2", "8",
        "--batch-size", "256", "--epochs", "2", "--window-positions", "1000"])
    train.main()

    raw = (out / "best.nnue").read_bytes()
    assert raw[:8] == b"HMNNUE02"
    features, hidden, l1, l2, scale, buckets = struct.unpack("<6I", raw[8:32])
    assert (features, hidden, l1, l2, scale, buckets) == (NUM_FEATURES, 32, 8, 8, 255, 1)
    head = (8 * 64 + 8) + (8 * 8 + 8) + 8 + 1
    assert len(raw) == 32 + 64 + 2 * (NUM_FEATURES * 32 + 32) + 4 * head
