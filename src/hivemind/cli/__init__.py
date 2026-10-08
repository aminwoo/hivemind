"""Command dispatch; heavyweight dependencies are loaded only when needed."""

import argparse
from importlib import import_module
import sys

COMMANDS = {
    "engine": ("hivemind.cli.engine", "Start the UCI engine (the default command)"),
    "lichess-bot": ("hivemind.cli.lichess_bot", "Accept Chess, Crazyhouse, Antichess, Chess960, Atomic, and Three-check challenges"),
    "build-engine": ("hivemind.cli.build_engine", "Configure and build the C++ engine"),
    "infer": ("hivemind.cli.infer_from_fen", "Evaluate a pair of FENs with ONNX"),
    "checkpoint": ("hivemind.inference.checkpoint", "Run PyTorch checkpoint inference"),
    "selfplay": ("hivemind.cli.selfplay", "Play twin-s self-play games with the RL loop's settings"),
    "evaluate": ("hivemind.cli.evaluate_model", "Evaluate a network on self-play data"),
    "train": ("hivemind.cli.train", "Train twin-s on self-play games with the RL loop's settings"),
    "rl-loop": ("hivemind.cli.rl_loop", "Self-play, train, match and upload twin-s networks, forever"),
    "contribute": ("hivemind.cli.contribute", "Play self-play games for a distributed rl-loop, forever"),
    "rise-selfplay": ("hivemind.cli.rise_selfplay", "Run RISE self-play (800 nodes, 100k mate nodes)"),
    "rise-train": ("hivemind.training.train_loop", "Train RISE networks with supervised or self-play data"),
    "prepare": ("hivemind.cli.train_from_games_parquet", "Prepare game shards and train"),
    "convert-selfplay": ("hivemind.data.convert_selfplay_data", "Convert self-play chunks to Parquet"),
    "fetch-network": ("hivemind.cli.fetch_network", "Download verified network weights"),
    "analyze": ("hivemind.cli.analyze_training_data", "Inspect training positions"),
    "search": ("hivemind.cli.search_training_fen", "Search training data by FEN"),
    "inspect-loader": ("hivemind.cli.inspect_data_loader", "Inspect loaded and augmented samples"),
    "nnue-train": ("hivemind.nnue.train", "Distill the teacher's value into an NNUE"),
    "distill-train": ("hivemind.distill.train", "Distil the teacher into a faster policy-value network"),
    "verify-augmentation": ("hivemind.cli.verify_augmentation", "Display board-swap augmentation"),
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
