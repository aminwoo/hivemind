"""Train a student network on the teacher's policy, WDL and moves-left.

    hivemind distill-train --data data/distill/train --val data/distill/val \\
        --size twin-m --out artifacts/distill/twin-m

Data comes from `hivemind gennnue --format distill` (raw teacher outputs)
and, optionally, `hivemind selfplay --distill-output` (--search-data: MCTS
visit distributions, root Q, and finished-game WDL and moves-left targets). Every
batch mixes the two in a fixed proportion (--search-fraction, by default
their share of all positions). Generate validation sets with their own seeds
so they share no games with the training sets.

An epoch visits every raw position once: chunks are shuffled, loaded a window
at a time, and each window is shuffled and consumed without replacement; the
search pool cycles the same way alongside, and so does --replay-data (older
self-play, a replay buffer) in its own share of each batch. The best
checkpoint by validation loss is exported to ONNX with the same inputs and
outputs as the teacher, so the engine loads it with --model.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hivemind.architectures.twin_board import TWIN_BOARD_SIZES, get_twin_board_model
from hivemind.distill.data import DistillDataset, chunk_paths, chunk_positions, head_chunk, read_chunk
from hivemind.paths import PROJECT_ROOT


TERMS = ("policy_a", "policy_b", "wdl", "value", "moves_left")


def _loss_rows(model, batch):
    """Loss numerators and target masks before normalization."""
    value, (policy_a, policy_b), _, wdl_logits, plys = model(batch["planes"])
    per_row = {}
    for name, logits, (target, legal, mask) in (("policy_a", policy_a, batch["policy"][0]),
                                                ("policy_b", policy_b, batch["policy"][1])):
        # The engine softmaxes over legal moves and the pass only; so does the loss.
        log_probs = F.log_softmax(logits.float().masked_fill(~legal, float("-inf")), dim=1)
        per_row[name] = (-torch.where(legal, target * log_probs, 0.0).sum(1), mask.float())
    has_wdl = (batch["wdl"].sum(1) > 0).float()
    per_row["wdl"] = (-(batch["wdl"] * F.log_softmax(wdl_logits.float(), dim=1)).sum(1), has_wdl)
    per_row["value"] = ((value.float().squeeze(1) - batch["value"]) ** 2, torch.ones_like(has_wdl))
    per_row["moves_left"] = ((plys.float().squeeze(1) - batch["moves_left"]) ** 2, has_wdl)
    return per_row, (value, policy_a, policy_b)


def _source_masks(batch, weights):
    source = batch.get("source", torch.zeros_like(batch["value"]))
    if "raw" in weights:
        return (("raw", source == 0), ("search", source > 0))
    return ((None, torch.ones_like(source, dtype=torch.bool)),)


def losses(model, batch, weights):
    """Normalize each term within its source, then add fixed source weights.

    Flat weights retain a single mean over all applicable rows. Unlabelled
    games contribute to neither WDL nor moves-left.
    """
    per_row, outputs = _loss_rows(model, batch)
    total, terms = 0.0, {}
    for name, (loss, valid) in per_row.items():
        for source, mask in _source_masks(batch, weights):
            active = valid * mask
            weight = weights[source][name] if source is not None else weights[name]
            total = total + weight * (loss * active).sum() / active.sum().clamp_min(1)
        terms[name] = (loss * valid).sum() / valid.sum().clamp_min(1)
    return total, terms, outputs


@torch.no_grad()
def evaluate(model, dataset, weights, batch_size=2048):
    model.eval()
    sums, counts, count = {}, {}, 0
    agree = [0, 0]
    rows = [0, 0]
    value_error = 0.0
    for start in range(0, len(dataset), batch_size):
        index = torch.arange(start, min(start + batch_size, len(dataset)), device=dataset.device)
        batch = dataset.batch(index)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dataset.device.type == "cuda"):
            per_row, (value, policy_a, policy_b) = _loss_rows(model, batch)
        for name, (loss, valid) in per_row.items():
            for source, mask in _source_masks(batch, weights):
                key = (source, name)
                active = valid * mask
                sums[key] = sums.get(key, 0.0) + float((loss * active).sum())
                counts[key] = counts.get(key, 0.0) + float(active.sum())
        count += len(index)
        value_error += float((value.float().squeeze(1) - batch["value"]).abs().sum())
        for board, logits in enumerate((policy_a, policy_b)):
            target, legal, mask = batch["policy"][board]
            # Top-move agreement among the legal moves.
            student_best = logits.float().masked_fill(~legal, float("-inf")).argmax(1)
            teacher_best = target.argmax(1)
            agree[board] += int(((student_best == teacher_best) & mask).sum())
            rows[board] += int(mask.sum())
    model.train()
    metrics = {}
    total = 0.0
    for name in TERMS:
        keys = [key for key in sums if key[1] == name]
        metrics[name] = sum(sums[key] for key in keys) / max(1, sum(counts[key] for key in keys))
        for source, term in keys:
            weight = weights[source][term] if source is not None else weights[term]
            total += weight * sums[(source, term)] / max(1, counts[(source, term)])
    metrics["loss"] = total
    metrics["value_mae"] = value_error / count
    metrics["top1_a"] = agree[0] / max(1, rows[0])
    metrics["top1_b"] = agree[1] / max(1, rows[1])
    return metrics


def concat_batches(first, second):
    """One batch from two (the raw and search streams)."""
    merged = {key: torch.cat((first[key], second[key])) for key in first if key != "policy"}
    merged["policy"] = [tuple(torch.cat(pair) for pair in zip(a, b))
                        for a, b in zip(first["policy"], second["policy"])]
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="*", default=[],
                        help="Teacher (raw) HDST chunks; omit with --search-fraction 1 to train on "
                             "self-play alone")
    parser.add_argument("--val", nargs="+", required=True)
    parser.add_argument("--search-data", nargs="+", default=[], help="Search (selfplay) HDST chunks")
    parser.add_argument("--search-val", nargs="+", default=[])
    parser.add_argument("--search-fraction", type=float, default=None,
                        help="Share of each batch from --search-data (default: its share of positions)")
    parser.add_argument("--search-window-chunks", type=int, default=2)
    parser.add_argument("--replay-data", nargs="+", default=[],
                        help="Older self-play HDST chunks (a replay buffer), weighted like --search-data")
    parser.add_argument("--replay-fraction", type=float, default=0.0,
                        help="Share of each batch from --replay-data, taken out of the search share")
    parser.add_argument("--replay-window-chunks", type=int, default=2)
    parser.add_argument("--search-policy-weight", type=float, default=1.0)
    parser.add_argument("--search-value-weight", type=float, default=1.0,
                        help="Weight of root-Q value MSE on search rows")
    parser.add_argument("--search-wdl-weight", type=float, default=1.0,
                        help="Weight of game-result WDL on search rows (only finished games carry one)")
    parser.add_argument("--search-moves-left-weight", type=float, default=None,
                        help="Default: --moves-left-weight")
    parser.add_argument("--init", type=Path, default=None,
                        help="Start from this checkpoint (its size and variant replace --size and the "
                             "architecture flags)")
    parser.add_argument("--steps", type=int, default=None, help="Train this many steps (overrides --epochs)")
    parser.add_argument("--export", default="best", choices=["best", "last"],
                        help="Export the best checkpoint by validation loss or the last one")
    parser.add_argument("--train-probe", type=int, default=200_000,
                        help="Training positions evaluated like the validation set, for the gap (0 = off)")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "artifacts" / "distill" / "run")
    parser.add_argument("--size", default="twin-m", choices=sorted(TWIN_BOARD_SIZES))
    parser.add_argument("--mid-attention", action="store_true",
                        help="Add a cross-board attention halfway through the trunk")
    parser.add_argument("--value-diff", action="store_true",
                        help="Feed the value head |A - B| next to A + B")
    parser.add_argument("--blocks", type=int, default=None, help="Override the size's block count")
    parser.add_argument("--channels", type=int, default=None, help="Override the size's trunk width")
    parser.add_argument("--attention-dim", type=int, default=None,
                        help="Override the final attention width (0 = no attention)")
    parser.add_argument("--attention-style", default="full", choices=["full", "lite"])
    parser.add_argument("--block", default="bottleneck", choices=["bottleneck", "dense"])
    parser.add_argument("--se-every", type=int, default=4, help="ECA every this many blocks (0 = none)")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--window-chunks", type=int, default=4,
                        help="Chunks held on the device at once")
    parser.add_argument("--eval-every", type=int, default=2000, help="Steps between validations")
    parser.add_argument("--moves-left-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip torch.compile of the residual blocks (about 1.5x slower)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    raw_weights = {"policy_a": 1.0, "policy_b": 1.0, "wdl": 1.0, "value": 1.0,
                   "moves_left": args.moves_left_weight}
    search_moves_left = (args.moves_left_weight if args.search_moves_left_weight is None
                         else args.search_moves_left_weight)
    search_weights = {"policy_a": args.search_policy_weight, "policy_b": args.search_policy_weight,
                      "wdl": args.search_wdl_weight, "value": args.search_value_weight,
                      "moves_left": search_moves_left}
    weights = {"raw": raw_weights, "search": search_weights}

    pool = chunk_paths(args.data)
    positions = sum(chunk_positions(path) for path in pool)
    rotation = np.random.default_rng(args.seed)
    window = min(args.window_chunks, len(pool))

    search_pool = chunk_paths(args.search_data)
    search_positions = sum(chunk_positions(path) for path in search_pool)
    fraction = args.search_fraction
    if fraction is None:
        fraction = search_positions / (positions + search_positions)
    search_rows = round(args.batch_size * fraction) if search_pool else 0
    raw_rows = args.batch_size - search_rows
    if raw_rows < 0 or (raw_rows == 0) != (not pool) or (not pool and not search_pool):
        raise SystemExit("--data is needed unless --search-fraction is 1 with --search-data, "
                         "and is unused then")
    replay_pool = chunk_paths(args.replay_data)
    replay_rows = round(args.batch_size * args.replay_fraction) if replay_pool else 0
    if bool(replay_pool) != (args.replay_fraction > 0):
        raise SystemExit("--replay-data and --replay-fraction go together")
    if replay_pool and not 0 < replay_rows < search_rows:
        raise SystemExit("--replay-fraction must leave part of the search share for --search-data")
    search_rows -= replay_rows

    def batches(paths, window, rows, source):
        """Endless batches of `rows` positions; every position once per pass."""
        while True:
            order = rotation.permutation(len(paths))
            for start in range(0, len(order), window):
                data = None  # free the previous window before loading the next
                data = DistillDataset([read_chunk(paths[i]) for i in order[start:start + window]], device,
                                      source=source)
                shuffled = torch.randperm(len(data), device=device)
                for first in range(0, len(data) - rows + 1, rows):
                    yield data.batch(shuffled[first:first + rows])

    raw_stream = batches(pool, window, raw_rows, source=0) if raw_rows else None
    search_stream = (batches(search_pool, min(args.search_window_chunks, len(search_pool)), search_rows,
                             source=1)
                     if search_rows else None)
    replay_stream = (batches(replay_pool, min(args.replay_window_chunks, len(replay_pool)), replay_rows,
                             source=1)
                     if replay_rows else None)

    def next_batch():
        parts = [next(stream) for stream in (raw_stream, search_stream, replay_stream) if stream is not None]
        batch = parts[0]
        for part in parts[1:]:
            batch = concat_batches(batch, part)
        return batch

    val = DistillDataset([read_chunk(p) for p in chunk_paths(args.val)], device)
    # A fixed slice of the training data, scored like the validation set: the
    # gap between the two says whether more data would help.
    train_probe = (DistillDataset([head_chunk(read_chunk((pool or search_pool)[0]), args.train_probe)], device,
                                  source=0 if pool else 1)
                   if args.train_probe else None)
    search_val = (DistillDataset([read_chunk(p) for p in chunk_paths(args.search_val)], device, source=1)
                  if args.search_val else None)
    print(f"train {len(pool)} chunks ({positions} positions, window {window}), val {len(val)}", flush=True)
    if search_pool:
        print(f"search {len(search_pool)} chunks ({search_positions} positions), "
              f"{search_rows}/{args.batch_size} of each batch, val {len(search_val) if search_val else 0}",
              flush=True)
    if replay_pool:
        print(f"replay {len(replay_pool)} chunks "
              f"({sum(chunk_positions(path) for path in replay_pool)} positions), "
              f"{replay_rows}/{args.batch_size} of each batch", flush=True)

    variant = {"mid_attention": args.mid_attention, "value_diff": args.value_diff,
               "blocks": args.blocks, "channels": args.channels, "attention_dim": args.attention_dim,
               "attention_style": args.attention_style, "block": args.block, "se_every": args.se_every}
    size = args.size
    initial = None
    if args.init:
        initial = torch.load(args.init, map_location="cpu")
        size, variant = initial["size"], initial.get("variant", {})
        print(f"starting from {args.init} ({size}, {variant})", flush=True)
    model = get_twin_board_model(size, **variant)
    if initial is not None:
        model.load_state_dict(initial["model"])
    model = model.to(device)
    if device.type == "cuda" and not args.no_compile:
        # In place, so state_dict keys are unchanged. The whole model does not
        # compile (its heads return views of shared tensors).
        torch.backends.cudnn.benchmark = True
        for block in model.blocks:
            block.compile()
    print(f"{size}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # An epoch is one pass over the raw data, or over the search data without it.
    steps_per_epoch = max(1, positions // raw_rows if raw_rows else search_positions // search_rows)
    total_steps = args.steps if args.steps else int(args.epochs * steps_per_epoch)
    warmup = min(1000, total_steps // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(1, warmup))
        * (0.5 * (1 + math.cos(math.pi * min(step, total_steps) / total_steps)) * 0.99 + 0.01))

    best, history, started = math.inf, [], time.time()
    for step in range(1, total_steps + 1):
        batch = next_batch()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            total, _, _ = losses(model, batch, weights)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        schedule.step()

        if step % args.eval_every == 0 or step == total_steps:
            metrics = evaluate(model, val, weights)
            if train_probe is not None:
                probe = evaluate(model, train_probe, weights)
                metrics.update(train_loss=probe["loss"], train_top1_a=probe["top1_a"],
                               train_value_mae=probe["value_mae"])
            if search_val is not None:
                search_metrics = evaluate(model, search_val, weights)
                metrics.update({f"search_{k}": v for k, v in search_metrics.items()})
                # Source weights are fixed coefficients, independent of batch proportions.
                metrics["selection_loss"] = (metrics["loss"]
                                             + (search_metrics["loss"] if search_rows else 0.0))
            metrics.update(epoch=round(step / steps_per_epoch, 2), lr=schedule.get_last_lr()[0],
                           seconds=round(time.time() - started))
            history.append(metrics)
            print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in metrics.items()}),
                  flush=True)
            selection = metrics.get("selection_loss", metrics["loss"])
            if selection < best:
                best = selection
                torch.save({"model": model.state_dict(), "size": size, "variant": variant},
                           args.out / "best.pt")

    torch.save({"model": model.state_dict(), "size": size, "variant": variant}, args.out / "last.pt")
    (args.out / "history.json").write_text(json.dumps(history, indent=1))
    export(args.out, size, checkpoint_name=f"{args.export}.pt")


def export(out, size, checkpoint_name="best.pt"):
    """ONNX export with the teacher's inputs and outputs (value, pi_a, pi_b, wdl_out, moves_left)."""
    from hivemind.training.trainer_agent import export_to_onnx

    checkpoint = torch.load(out / checkpoint_name)
    model = get_twin_board_model(size, **checkpoint.get("variant", {}))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    export_to_onnx(model, 1, torch.ones(1, 74, 8, 8), out, f"distill-{size}",
                   has_auxiliary_output=True, dynamic_batch_size=True)
    print(f"exported {out}")


if __name__ == "__main__":
    main()
