"""Command dispatch; heavyweight dependencies are loaded only when needed."""

import argparse
from importlib import import_module
import sys

COMMANDS = {
    "engine": ("hivemind.cli.engine", "Start the UCI engine (the default command)"),
    "lichess-bot": ("hivemind.cli.lichess_bot", "Accept Chess, Crazyhouse, Antichess, Chess960, Atomic, and Three-check challenges"),
    "build-engine": ("hivemind.cli.build_engine", "Configure and build the C++ engine"),
    "infer": ("hivemind.cli.infer_from_fen", "Evaluate a pair of FENs with ONNX"),
    "selfplay": ("hivemind.cli.selfplay", "Play twin-s self-play games with the RL loop's settings"),
    "train": ("hivemind.cli.train", "Train twin-s on self-play games with the RL loop's settings"),
    "rl-loop": ("hivemind.cli.rl_loop", "Self-play, train, match and upload twin-s networks, forever"),
    "contribute": ("hivemind.cli.contribute", "Play self-play games for a distributed rl-loop, forever"),
    "archive": ("hivemind.cli.archive", "Move data directories to a private Hugging Face dataset"),
    "restore": ("hivemind.cli.restore", "Download archived data directories back"),
    "fetch-network": ("hivemind.cli.fetch_network", "Download verified network weights"),
    "distill-train": ("hivemind.distill.train", "Distil the teacher into a faster policy-value network"),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="hivemind", description="Hivemind training and inference tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Commands:\n" + "\n".join(
            f"  {name:20} {description}" for name, (_, description) in COMMANDS.items()
        ) + "\n\nWith no COMMAND, starts the engine. Use hivemind COMMAND --help for command options.",
    )
    parser.add_argument("command", choices=COMMANDS, metavar="COMMAND")
    if not argv:
        argv = ["engine"]
    args = parser.parse_args(argv[:1])
    module_name, _ = COMMANDS[args.command]
    previous_argv = sys.argv
    try:
        sys.argv = [f"hivemind {args.command}", *argv[1:]]
        return import_module(module_name).main()
    finally:
        sys.argv = previous_argv
