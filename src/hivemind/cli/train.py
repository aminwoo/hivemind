#!/usr/bin/env python3
"""Train twin-s on self-play games with the settings of the RL loop that
trained iterations 1 to 6 (`hivemind rl-loop`): policy from root visits,
WDL and moves-left from game results, no teacher data, lr 1e-4, 1.5 passes
over the new games plus 0.5 over each replayed one, exporting the last
weights. Runs `hivemind distill-train`."""

import argparse
from datetime import datetime
import math
import os
from pathlib import Path
import sys

from hivemind.cli.fetch_network import fetch_network, file_digest
from hivemind.network import ARTIFACTS, DEFAULT_ONNX_PATH, TWIN_CHECKPOINT_PATH
from hivemind.paths import PROJECT_ROOT

GENERATOR_CHECKPOINT = TWIN_CHECKPOINT_PATH
TEACHER_VAL = PROJECT_ROOT / "data" / "distill" / "val"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "distill" / "rl"


def generator_checkpoint(generator=DEFAULT_ONNX_PATH, checkpoint=GENERATOR_CHECKPOINT):
    """The generator's training checkpoint, downloading the published one when
    it is missing and the generator is the published network."""
    if checkpoint.is_file():
        return checkpoint
    if not generator.is_file() or file_digest(generator) != ARTIFACTS["onnx"].sha256:
        raise FileNotFoundError(f"No training checkpoint at {checkpoint} for {generator}, and it is "
                                f"not the published network")
    print(f"Downloading the generator's training checkpoint to {checkpoint}", flush=True)
    fetch_network("checkpoint", checkpoint.parent).replace(checkpoint)
    return checkpoint


def plan(new_positions, replay_positions, new_passes, replay_passes, batch_size):
    """Training steps and the replay share of each batch, as rl_loop.sh
    computes them."""
    samples_new = new_passes * new_positions
    samples_old = replay_passes * replay_positions
    steps = math.ceil((samples_new + samples_old) / batch_size)
    return steps, round(samples_old / (samples_new + samples_old), 4)


def positions(dirs):
    from hivemind.distill.data import chunk_paths, chunk_positions
    return sum(chunk_positions(path) for path in chunk_paths(dirs))


# Settings the RL loop trained with, applied to the distill-train process.
TRAINING_ENV = {"OMP_NUM_THREADS": "2", "PYTORCH_ALLOC_CONF": "expandable_segments:True"}


def distill_command(init, data, val, replay, steps, fraction, batch_size, seed, out,
                    teacher_val=(), extra=()):
    """The `hivemind distill-train` command line of one RL iteration."""
    return [
        sys.executable, "-m", "hivemind", "distill-train",
        "--init", str(Path(init).resolve()),
        "--search-fraction", "1",
        *(["--val", *map(str, teacher_val)] if teacher_val else []),
        "--search-data", *map(str, data),
        "--search-val", *map(str, val),
        *(["--replay-data", *map(str, replay), "--replay-fraction", str(fraction)]
          if fraction > 0 else []),
        "--steps", str(steps),
        "--lr", "1e-4",
        "--eval-every", str(max(1, steps // 4)),
        "--batch-size", str(batch_size),
        "--eval-batch-size", str(batch_size),
        "--search-window-chunks", "2",
        "--replay-window-chunks", "2",
        "--export", "last",
        "--seed", str(seed),
        "--search-value-weight", "0",
        "--search-wdl-weight", "1",
        "--search-moves-left-weight", "0.1",
        "--out", str(out),
        *extra,
    ]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Unrecognised options are passed through to `hivemind distill-train`.",
    )
    parser.add_argument("--data", nargs="+", type=Path, required=True,
                        help="This iteration's self-play directories (from hivemind selfplay)")
    parser.add_argument("--val", nargs="+", type=Path, required=True,
                        help="Self-play validation directories")
    parser.add_argument("--replay", nargs="+", type=Path, default=[],
                        help="Earlier iterations' self-play directories (the loop replayed the "
                             "previous three)")
    parser.add_argument("--init", type=Path, default=GENERATOR_CHECKPOINT,
                        help="Checkpoint to continue (default: the generator's, "
                             "artifacts/distill/twin-s-noattn/best.pt, downloaded when missing)")
    parser.add_argument("--out", type=Path,
                        help="Output directory (default: artifacts/distill/rl/<time>)")
    parser.add_argument("--new-passes", type=float, default=1.5)
    parser.add_argument("--replay-passes", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0, help="The loop used the iteration number")
    parser.add_argument("--teacher-val", nargs="*", type=Path,
                        default=[TEACHER_VAL] if TEACHER_VAL.is_dir() else [],
                        help="Teacher validation chunks, scored for reference only "
                             "(default: data/distill/val when present; pass no value to skip)")
    parser.add_argument("--dry-run", action="store_true", help="Print the command and exit")
    args, extra = parser.parse_known_args()
    if args.batch_size < 1 or args.new_passes <= 0 or args.replay_passes < 0:
        parser.error("--batch-size and --new-passes must be positive, --replay-passes non-negative")
    if not args.init.is_file():
        if args.init != GENERATOR_CHECKPOINT:
            sys.exit(f"Checkpoint not found at {args.init}; pass --init.")
        if not args.dry_run:
            try:
                generator_checkpoint()
            except FileNotFoundError as error:
                sys.exit(f"{error}; pass --init.")

    new, val, old = positions(args.data), positions(args.val), positions(args.replay)
    if not new or not val:
        sys.exit("--data and --val must contain HDST chunks (*.dst).")
    steps, fraction = plan(new, old, args.new_passes, args.replay_passes, args.batch_size)
    out = args.out or OUTPUT_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"{steps} steps: {new} new positions x{args.new_passes}, {old} replay positions "
          f"x{args.replay_passes} (replay share {fraction}, batch {args.batch_size})", flush=True)
    command = distill_command(args.init, args.data, args.val, args.replay, steps, fraction,
                              args.batch_size, args.seed, out, args.teacher_val, extra)
    print(" ".join(command), flush=True)
    if args.dry_run:
        return 0
    for name, value in TRAINING_ENV.items():
        os.environ.setdefault(name, value)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
