#!/usr/bin/env python3
"""Run engine self-play with training defaults: 800 MCTS nodes and a
100,000-node Fairy-Stockfish mate budget per searched position."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from hivemind.network import DEFAULT_ONNX_PATH
from hivemind.paths import PROJECT_ROOT

DEFAULT_ENGINE = PROJECT_ROOT / "engine" / "build-ninja" / "hivemind"
DEFAULT_OUTPUT = PROJECT_ROOT / "engine" / "selfplay_games"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Unrecognised options are passed through to `hivemind selfplay` "
               "(see ./engine/build-ninja/hivemind --help).",
    )
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--nodes", type=int, default=800, help="MCTS iterations per move")
    parser.add_argument("--mate-nodes", type=int, default=100_000,
                        help="Fairy-Stockfish node budget per searched position (0 disables)")
    parser.add_argument("--model", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE,
                        help="Path to the built engine binary")
    parser.add_argument("--dry-run", action="store_true", help="Print the command and exit")
    args, extra = parser.parse_known_args()

    if not args.engine.is_file():
        sys.exit(f"Engine binary not found at {args.engine}; build it first "
                 f"(cmake --build --preset ninja-fast) or pass --engine.")
    if not args.model.is_file():
        sys.exit(f"Model not found at {args.model}; run `hivemind fetch-network` or pass --model.")

    command = [
        str(args.engine), "selfplay",
        "--model", str(args.model.resolve()),
        "--games", str(args.games),
        "--nodes", str(args.nodes),
        "--fairy-stockfish-mate-nodes", str(args.mate_nodes),
        "--output", str(args.output),
        *extra,
    ]
    print(" ".join(command), flush=True)
    if args.dry_run:
        return 0
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
