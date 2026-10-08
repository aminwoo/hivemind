import json
import struct

import numpy as np
import pytest
import torch

from hivemind.architectures.twin_board import get_twin_board_model
from hivemind.distill.data import MAGIC, PLANES, VERSION, DistillDataset, head_chunk, read_chunk
from hivemind.distill.train import concat_batches, losses


def _pack(planes):
    """The engine's packing: a 0/1 bitboard with value 1, or one value everywhere."""
    bits = np.zeros(PLANES, np.int64)
    values = np.zeros(PLANES, np.float16)
    for index, plane in enumerate(planes.reshape(PLANES, 64)):
        if np.all(plane == plane[0]):
            values[index] = plane[0]
        else:
            bits[index] = sum(1 << int(s) for s in np.flatnonzero(plane == 1)) - (1 << 64 if plane[63] == 1 else 0)
            values[index] = 1.0
    return bits, values


def _write_chunk(path, rng, n=16):
    planes = np.zeros((n, PLANES, 64), np.float32)
    planes[:, :12] = rng.random((n, 12, 64)) < 0.1          # piece bitboards
    planes[:, 12:22] = (rng.integers(0, 5, (n, 10, 1)) / 16.0)  # pocket fills
    planes[:, 0, 63] = 1                                     # exercise bit 63 (sign bit)
    packed = [_pack(p) for p in planes]
    counts = rng.integers(0, 6, (n, 2)).astype(np.uint16)
    counts[0] = (0, 3)                                       # a board not on turn
    entries = int(counts.sum())
    index = rng.integers(0, 4672, entries).astype(np.uint16)
    probability = rng.random(entries).astype(np.float16)
    with open(path, "wb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("<II", VERSION, PLANES))
        stream.write(struct.pack("<QQ", n, entries))
        stream.write(np.stack([p[0] for p in packed]).tobytes())
        stream.write(np.stack([p[1] for p in packed]).tobytes())
        stream.write(np.tanh(rng.normal(size=n)).astype(np.float32).tobytes())
        stream.write(np.full((n, 3), 1 / 3, np.float32).tobytes())
        stream.write(rng.random(n).astype(np.float16).tobytes())
        stream.write(np.zeros(n, np.uint16).tobytes())
        stream.write(np.zeros(n, np.int8).tobytes())
        stream.write(counts.tobytes())
        stream.write(index.tobytes())
        stream.write(probability.tobytes())
    return planes.reshape(n, PLANES, 8, 8), counts


def test_planes_round_trip_and_policy_targets(tmp_path):
    rng = np.random.default_rng(0)
    planes, counts = _write_chunk(tmp_path / "c.dst", rng)
    dataset = DistillDataset([read_chunk(tmp_path / "c.dst")], torch.device("cpu"))
    index = torch.arange(len(planes))
    torch.testing.assert_close(dataset.planes(index), torch.from_numpy(planes), atol=1e-3, rtol=0)
    target, legal, mask = dataset.policy_targets(index, 0)
    assert not mask[0] and mask.tolist() == (counts[:, 0] > 0).tolist()
    torch.testing.assert_close(target[mask].sum(1), torch.ones(int(mask.sum())))
    assert legal[~mask].all() and (legal[mask].sum(1) <= torch.from_numpy(counts[:, 0].astype(np.int64))[mask]).all()
    assert (target[~legal] == 0).all()


def test_policy_loss_ignores_illegal_logits(tmp_path):
    """The engine never sees illegal moves' logits, so the loss must not either."""
    rng = np.random.default_rng(2)
    _write_chunk(tmp_path / "c.dst", rng)
    dataset = DistillDataset([read_chunk(tmp_path / "c.dst")], torch.device("cpu"))
    batch = dataset.batch(torch.arange(16))
    model = get_twin_board_model("twin-s").eval()

    def policy_loss(shift_illegal):
        class Shifted(torch.nn.Module):
            def forward(self, x):
                value, (a, b), aux, wdl, plys = model(x)
                a = a + shift_illegal * (~batch["policy"][0][1]).float()
                b = b + shift_illegal * (~batch["policy"][1][1]).float()
                return value, (a, b), aux, wdl, plys
        _, terms, _ = losses(Shifted(), batch, {k: 1.0 for k in ("policy_a", "policy_b", "wdl", "value", "moves_left")})
        return float(terms["policy_a"]), float(terms["policy_b"])

    with torch.no_grad():
        base, shifted = policy_loss(0.0), policy_loss(50.0)
    assert np.isfinite(base).all()
    np.testing.assert_allclose(base, shifted, rtol=1e-5)


def test_a_training_step_reduces_the_loss(tmp_path):
    rng = np.random.default_rng(1)
    _write_chunk(tmp_path / "c.dst", rng, n=32)
    dataset = DistillDataset([read_chunk(tmp_path / "c.dst")], torch.device("cpu"))
    model = get_twin_board_model("twin-s")
    weights = {"policy_a": 1.0, "policy_b": 1.0, "wdl": 1.0, "value": 1.0, "moves_left": 0.1}
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    batch = dataset.batch(torch.arange(32))
    first = None
    for _ in range(20):
        total, _, _ = losses(model, batch, weights)
        first = float(total) if first is None else first
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
    assert float(total) < first


def test_search_records_have_no_wdl_or_moves_left_loss(tmp_path):
    """Search records (all-zero WDL) train value and policy only."""
    rng = np.random.default_rng(3)
    _write_chunk(tmp_path / "c.dst", rng)
    dataset = DistillDataset([read_chunk(tmp_path / "c.dst")], torch.device("cpu"))
    raw = dataset.batch(torch.arange(8))
    search = dataset.batch(torch.arange(8, 16))
    search["wdl"] = torch.zeros_like(search["wdl"])
    search["moves_left"] = torch.full_like(search["moves_left"], 1e6)  # would dominate if used
    model = get_twin_board_model("twin-s").eval()
    weights = {k: 1.0 for k in ("policy_a", "policy_b", "wdl", "value", "moves_left")}
    with torch.no_grad():
        _, raw_terms, _ = losses(model, raw, weights)
        _, mixed_terms, _ = losses(model, concat_batches(raw, search), weights)
    torch.testing.assert_close(mixed_terms["wdl"], raw_terms["wdl"])
    torch.testing.assert_close(mixed_terms["moves_left"], raw_terms["moves_left"])
    merged = concat_batches(raw, search)
    assert merged["planes"].shape[0] == 16 and len(merged["policy"][0]) == 3


def test_head_chunk_keeps_policies_aligned(tmp_path):
    rng = np.random.default_rng(4)
    _write_chunk(tmp_path / "c.dst", rng)
    chunk = read_chunk(tmp_path / "c.dst")
    head = head_chunk(chunk, 5)
    full = DistillDataset([chunk], torch.device("cpu"))
    part = DistillDataset([head], torch.device("cpu"))
    index = torch.arange(5)
    for board in (0, 1):
        torch.testing.assert_close(part.policy_targets(index, board)[0], full.policy_targets(index, board)[0])
    torch.testing.assert_close(part.planes(index), full.planes(index))


def test_zero_weighted_source_neither_contributes_nor_dilutes(tmp_path):
    rng = np.random.default_rng(5)
    _write_chunk(tmp_path / "c.dst", rng)
    chunk = read_chunk(tmp_path / "c.dst")
    raw = DistillDataset([chunk], torch.device("cpu"), source=0).batch(torch.arange(8))
    search = DistillDataset([chunk], torch.device("cpu"), source=1).batch(torch.arange(8, 16))
    search["wdl"] = torch.nn.functional.one_hot(torch.arange(8) % 3, 3).float()  # game results
    model = get_twin_board_model("twin-s").eval()
    ones = {k: 1.0 for k in ("policy_a", "policy_b", "wdl", "value", "moves_left")}
    with torch.no_grad():
        ignored, _, _ = losses(model, concat_batches(raw, search),
                               {"raw": ones, "search": {**ones, "wdl": 0.0, "moves_left": 0.0}})
        unlabelled = dict(search, wdl=torch.zeros_like(search["wdl"]))
        reference, _, _ = losses(model, concat_batches(raw, unlabelled), {"raw": ones, "search": ones})
        used, _, _ = losses(model, concat_batches(raw, search), {"raw": ones, "search": ones})
    torch.testing.assert_close(ignored, reference)
    assert not torch.isclose(used, reference)


def test_training_on_search_data_alone(tmp_path, monkeypatch):
    """--search-fraction 1 without --data trains on self-play records only."""
    import sys
    from hivemind.distill import train

    rng = np.random.default_rng(6)
    (tmp_path / "search").mkdir()
    _write_chunk(tmp_path / "search" / "c.dst", rng, n=32)
    _write_chunk(tmp_path / "val.dst", rng, n=16)
    monkeypatch.setattr(train, "export", lambda *args, **kwargs: None)
    evaluation_batches = []
    real_evaluate = train.evaluate

    def recording_evaluate(model, dataset, weights, batch_size):
        evaluation_batches.append(batch_size)
        return real_evaluate(model, dataset, weights, batch_size=batch_size)

    monkeypatch.setattr(train, "evaluate", recording_evaluate)
    monkeypatch.setattr(sys, "argv", [
        "distill-train",
        "--search-data", str(tmp_path / "search"), "--search-val", str(tmp_path / "val.dst"),
        "--search-fraction", "1", "--size", "twin-s", "--batch-size", "8", "--steps", "3",
        "--eval-every", "3", "--eval-batch-size", "4", "--train-probe", "8",
        "--no-compile", "--out", str(tmp_path / "out")])
    train.main()
    history = json.loads((tmp_path / "out" / "history.json").read_text())
    assert history and (tmp_path / "out" / "last.pt").exists()
    assert evaluation_batches == [4, 4]
    assert history[-1]['selection_loss'] == history[-1]['search_loss']


def test_selfplay_checkpoint_ignores_teacher_validation_loss(tmp_path, monkeypatch):
    import sys
    from hivemind.distill import train

    rng = np.random.default_rng(16)
    (tmp_path / "search").mkdir()
    _write_chunk(tmp_path / "search/c.dst", rng, n=32)
    _write_chunk(tmp_path / "val.dst", rng, n=16)
    teacher_losses = iter([100.0, 1.0, 0.0])
    selfplay_losses = iter([1.0, 2.0, 3.0])

    def fake_evaluate(model, dataset, weights, batch_size):
        return {"loss": next(selfplay_losses if dataset.source else teacher_losses)}

    monkeypatch.setattr(train, "evaluate", fake_evaluate)
    monkeypatch.setattr(train, "export", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", [
        "distill-train", "--val", str(tmp_path / "val.dst"),
        "--search-data", str(tmp_path / "search"), "--search-val", str(tmp_path / "val.dst"),
        "--search-fraction", "1", "--size", "twin-s", "--batch-size", "8", "--steps", "3",
        "--eval-every", "1", "--train-probe", "0", "--no-compile", "--out", str(tmp_path / "out")])
    train.main()
    best = torch.load(tmp_path / "out/best.pt", map_location="cpu", weights_only=True)
    assert best['step'] == 1
    assert best['selection_loss'] == 1.0


def test_replay_data_takes_its_share_of_each_batch(tmp_path, monkeypatch):
    """--replay-data is a third stream inside the search share of every batch."""
    import sys
    from hivemind.distill import train

    rng = np.random.default_rng(7)
    for name in ("search", "replay"):
        (tmp_path / name).mkdir()
        _write_chunk(tmp_path / name / "c.dst", rng, n=32)
    _write_chunk(tmp_path / "val.dst", rng, n=16)
    sizes = []
    real_losses = train.losses
    def recording_losses(model, batch, weights):
        sizes.append((len(batch["value"]), int((batch["source"] == 1).sum())))
        return real_losses(model, batch, weights)
    monkeypatch.setattr(train, "losses", recording_losses)
    monkeypatch.setattr(train, "export", lambda *args, **kwargs: None)
    argv = ["distill-train", "--val", str(tmp_path / "val.dst"),
            "--search-data", str(tmp_path / "search"), "--search-val", str(tmp_path / "val.dst"),
            "--search-fraction", "1", "--size", "twin-s", "--batch-size", "8", "--steps", "3",
            "--eval-every", "3", "--train-probe", "0", "--no-compile", "--out", str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", argv + ["--replay-data", str(tmp_path / "replay"),
                                             "--replay-fraction", "0.25"])
    replayed = []
    real_batches = train.DistillDataset.batch
    def recording_batch(self, index):
        batch = real_batches(self, index)
        replayed.append(len(index))
        return batch
    monkeypatch.setattr(train.DistillDataset, "batch", recording_batch)
    train.main()
    assert sizes == [(8, 8)] * 3
    # Each step draws 6 new and 2 replay rows (validation batches are larger).
    assert replayed[:6] == [6, 2] * 3
    for bad in (["--replay-fraction", "0.25"], ["--replay-data", str(tmp_path / "replay")],
                ["--replay-data", str(tmp_path / "replay"), "--replay-fraction", "1"]):
        monkeypatch.setattr(sys, "argv", argv + bad)
        with pytest.raises(SystemExit):
            train.main()
