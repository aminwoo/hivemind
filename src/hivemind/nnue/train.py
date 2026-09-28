"""Distill the teacher network's value into an NNUE student.

    hivemind nnue-train --data data/nnue/train --val data/nnue/val --out artifacts/nnue/run1
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from hivemind.nnue.data import chunk_paths, read_chunk, read_chunk_header
from hivemind.nnue.features import MIRROR_TABLE, SWAP_TABLE
from hivemind.nnue.model import NnueModel, export_network, quantized_copy
from hivemind.paths import PROJECT_ROOT


class DeviceDataset:
    """Positions resident on one device as CSR feature lists.

    Buffers are allocated once for `capacity` chunks and refilled in place, so
    a pool larger than device memory is trained on one random window of
    chunks per epoch. Chunks are copied one at a time, so host memory never
    holds more than one; 48M positions take about 8 GB on the device.
    """

    def __init__(self, chunks, device, capacity=None):
        capacity = capacity or len(chunks)
        largest = sorted(chunks, key=lambda chunk: -len(chunk))[:capacity]
        positions = sum(len(chunk) for chunk in largest)
        features = max(sum(chunk.width for chunk in window)
                       for window in (sorted(chunks, key=lambda chunk: -chunk.width)[:capacity],))
        self.device = device
        self.counts = torch.empty(positions, dtype=torch.int16, device=device)
        # int16 halves the footprint; indices are < 2048 so the cast is exact.
        self.features = torch.empty(features, dtype=torch.int16, device=device)
        self.value = torch.empty(positions, dtype=torch.float32, device=device)
        self.mirror = torch.from_numpy(MIRROR_TABLE).to(device)
        self.swap = torch.from_numpy(SWAP_TABLE).to(device)
        self.size = 0

    def fill(self, chunks):
        position, feature = 0, 0
        for lazy in chunks:
            chunk = lazy()
            size, width = len(chunk), len(chunk.features)
            self.counts[position:position + size] = torch.from_numpy(chunk.counts.astype(np.int16))
            self.features[feature:feature + width] = torch.from_numpy(chunk.features.view(np.int16).copy())
            self.value[position:position + size] = torch.from_numpy(chunk.value.copy())
            position, feature = position + size, feature + width
        self.size = position
        counts = self.counts[:position].long()
        self.starts = torch.cumsum(counts, 0) - counts
        return self

    def __len__(self):
        return self.size

    def batch(self, index, swap_probability=0.0):
        """Side-to-move and mirrored feature indices, their batch rows, and targets.

        With `swap_probability`, that share of positions has boards A and B
        swapped, a symmetry of bughouse that leaves the value unchanged.
        """
        counts = self.counts[index].long()
        offsets = torch.cumsum(counts, 0) - counts
        total = int(offsets[-1] + counts[-1])
        within = torch.arange(total, device=self.device) - torch.repeat_interleave(offsets, counts, output_size=total)
        flat = self.features[torch.repeat_interleave(self.starts[index], counts, output_size=total) + within].long()
        rows = torch.repeat_interleave(torch.arange(len(index), device=self.device), counts, output_size=total)
        if swap_probability > 0.0:
            swapped = torch.rand(len(index), device=self.device) < swap_probability
            flat = torch.where(swapped[rows], self.swap[flat], flat)
        return flat, self.mirror[flat], rows, self.value[index]


class LazyChunk:
    """Defers reading a chunk until the dataset copies it."""

    def __init__(self, path):
        self.path = path
        header = read_chunk_header(path)
        self.size, self.width = header

    def __len__(self):
        return self.size

    @property
    def features(self):
        return range(self.width)

    def __call__(self):
        return read_chunk(self.path)


def load(paths, device):
    chunks = [LazyChunk(path) for path in paths]
    return DeviceDataset(chunks, device).fill(chunks)


def value_loss(logits, target):
    return torch.mean((torch.tanh(logits) - target) ** 2)


@torch.no_grad()
def evaluate(model, dataset, batch_size=65536):
    model.eval()
    squared, absolute, count = 0.0, 0.0, 0
    predictions, targets = [], []
    for start in range(0, len(dataset), batch_size):
        index = torch.arange(start, min(start + batch_size, len(dataset)), device=dataset.device)
        stm, nstm, rows, target = dataset.batch(index)
        prediction = torch.tanh(model(stm, nstm, rows, len(index)))
        squared += float(((prediction - target) ** 2).sum())
        absolute += float((prediction - target).abs().sum())
        count += len(index)
        predictions.append(prediction)
        targets.append(target)
    model.train()
    prediction, target = torch.cat(predictions), torch.cat(targets)
    correlation = float(torch.corrcoef(torch.stack([prediction, target]))[0, 1])
    sign = float(((prediction > 0) == (target > 0)).float().mean())
    return {"mse": squared / count, "mae": absolute / count, "corr": correlation, "sign": sign}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="+", required=True, help="Chunk files or directories")
    parser.add_argument("--val", nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "artifacts" / "nnue" / "run")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--l1", type=int, default=16)
    parser.add_argument("--l2", type=int, default=32)
    parser.add_argument("--king-buckets", default="none", choices=["none", "k4", "k8"])
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--epochs", type=float, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--window-positions", type=float, default=48e6,
                        help="Positions held on the device at once; larger pools rotate "
                             "a random window of chunks each epoch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--swap-probability", type=float, default=0.5,
                        help="Share of training positions with boards A and B swapped")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    started = time.time()
    pool = [LazyChunk(path) for path in chunk_paths(args.data)]
    window = len(pool)
    while window > 1 and sum(sorted(len(chunk) for chunk in pool)[-window:]) > args.window_positions:
        window -= 1
    rotation = np.random.default_rng(args.seed)

    def window_chunks():
        if window == len(pool):
            return pool
        return [pool[i] for i in rotation.choice(len(pool), window, replace=False)]

    train = DeviceDataset(pool, device, window).fill(window_chunks())
    val = load(chunk_paths(args.val), device)
    n_train = len(train)
    print(f"pool {sum(len(c) for c in pool)} positions in {len(pool)} chunks; window {window} chunks "
          f"({n_train} positions); val {len(val)}; loaded in {time.time() - started:.0f}s", flush=True)

    model = NnueModel(args.hidden, args.l1, args.l2, args.king_buckets).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, n_train // args.batch_size)
    total_steps = int(args.epochs * steps_per_epoch)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 0.5 * (1 + math.cos(math.pi * min(step, total_steps) / total_steps)) * 0.99 + 0.01)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    history = []
    best = math.inf
    running = 0.0
    tick = time.time()
    for step in range(1, total_steps + 1):
        index = torch.randint(0, len(train), (args.batch_size,), device=device, generator=generator)
        stm, nstm, rows, target = (t.to(device, non_blocking=True)
                                   for t in train.batch(index, args.swap_probability))
        loss = value_loss(model(stm, nstm, rows, args.batch_size), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        model.clip_weights()
        running += loss.detach()

        if step % steps_per_epoch == 0 or step == total_steps:
            epoch = step / steps_per_epoch
            metrics = evaluate(quantized_copy(model), val)
            metrics.update(epoch=round(epoch, 2), train_mse=float(running) / steps_per_epoch,
                           lr=schedule.get_last_lr()[0], seconds=round(time.time() - tick, 1))
            running = 0.0
            history.append(metrics)
            print(json.dumps(metrics), flush=True)
            torch.save({"model": model.state_dict(), "config": model.config(), "args": vars(args) | {"out": str(args.out)}},
                       args.out / "last.pt")
            if metrics["mse"] < best:
                best = metrics["mse"]
                torch.save({"model": model.state_dict(), "config": model.config()}, args.out / "best.pt")
                export_network(model, args.out / "best.nnue")
            if window < len(pool) and step < total_steps:
                train.fill(window_chunks())

    (args.out / "history.json").write_text(json.dumps(history, indent=1))
    print(f"best val mse {best:.5f}; network at {args.out / 'best.nnue'}")


if __name__ == "__main__":
    main()
