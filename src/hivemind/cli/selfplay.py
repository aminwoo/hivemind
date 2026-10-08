#!/usr/bin/env python3
"""Play twin-s self-play games with the settings of the RL loop that trained
iterations 1 to 6 (`hivemind rl-loop`): 800 nodes +-5% per move, no mate
search, resignation off, 6 parallel games on the one-worker engine. Writes
HDST chunks of the search targets for `hivemind train`."""

import argparse
from datetime import datetime
import filecmp
import os
from pathlib import Path
import shutil
import sys

from hivemind.network import DEFAULT_ONNX_PATH
from hivemind.paths import PROJECT_ROOT

# Built with -DHIVEMIND_SEARCH_WORKERS=1: one search worker per game, so many
# games share the GPU.
DEFAULT_ENGINE = PROJECT_ROOT / "engine" / "build-sp1" / "hivemind"
SELFPLAY_DIR = PROJECT_ROOT / "data" / "distill" / "selfplay"
BUILD_COMMAND = (
    "cmake -S engine -B engine/build-sp1 -G Ninja -DCMAKE_BUILD_TYPE=Release "
    "-DHIVEMIND_SEARCH_WORKERS=1 && cmake --build engine/build-sp1 --target hivemind"
)


def one_worker_copy(model):
    """The copy of `model` the one-worker engine loads. TensorRT caches are
    named after the ONNX file, and this engine's (one optimization profile)
    would otherwise replace the UCI engine's (four) on every switch."""
    if model.stem.endswith("-sp1"):
        return model
    return model.with_name(f"{model.stem}-sp1{model.suffix}")


def refresh_copy(model, copy):
    if copy == model or (copy.is_file() and filecmp.cmp(model, copy, shallow=False)):
        return
    temporary = copy.with_name(f"{copy.name}.tmp")
    shutil.copyfile(model, temporary)
    temporary.replace(copy)


def engine_command(engine, model, games, seed, output, records, parallel_games=6, extra=()):
    """The engine command line: HDST chunks go to `output`, PGN to `records`."""
    return [
        str(engine), "selfplay",
        "--model", str(model),
        "--games", str(games),
        "--nodes", "800",
        "--node-random-factor", "0.05",
        "--seed", str(seed),
        "--fairy-stockfish-mate-nodes", "0",
        "--resign-threshold", "0",
        "--parallel-games", str(parallel_games),
        "--training-chunks", "false",
        "--output", str(records),
        "--distill-output", str(output),
        *extra,
    ]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Unrecognised options are passed through to the engine's selfplay command "
               "(see ./engine/build-sp1/hivemind --help).",
    )
    parser.add_argument("--games", type=int, default=10_000,
                        help="Games to play (default: 10000, one RL loop segment)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Self-play seed (default: 0, from the clock); the loop used "
                             "1000*iteration + segment, and 1000*iteration + 999 for validation")
    parser.add_argument("--output", type=Path,
                        help="Fresh directory for the HDST chunks; game records go in its games "
                             "subdirectory (default: data/distill/selfplay/<time>)")
    parser.add_argument("--model", type=Path, default=DEFAULT_ONNX_PATH,
                        help="Generator network; run on a -sp1 copy next to it")
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE,
                        help="One-search-worker engine binary")
    parser.add_argument("--parallel-games", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true", help="Print the command and exit")
    args, extra = parser.parse_known_args()

    if not args.engine.is_file():
        sys.exit(f"Engine binary not found at {args.engine}; build the one-worker engine with\n"
                 f"  {BUILD_COMMAND}\nor pass --engine.")
    if not args.model.is_file():
        sys.exit(f"Model not found at {args.model}; run `hivemind fetch-network` or pass --model.")
    output = args.output or SELFPLAY_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    if output.exists() and any(output.iterdir()):
        sys.exit(f"Output directory {output} must be fresh.")
    model = one_worker_copy(args.model.resolve())

    command = engine_command(args.engine, model, args.games, args.seed, output.resolve(),
                             output.resolve() / "games", args.parallel_games, extra)
    print(" ".join(command), flush=True)
    if args.dry_run:
        return 0
    refresh_copy(args.model.resolve(), model)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
