"""Command dispatch; heavyweight dependencies are loaded only when needed."""

import argparse
from importlib import import_module
import sys

COMMANDS = {
    "build-engine": ("hivemind.cli.build_engine", "Configure and build the C++ engine"),
    "infer": ("hivemind.cli.infer_from_fen", "Evaluate a pair of FENs with ONNX"),
    "checkpoint": ("hivemind.inference.checkpoint", "Run PyTorch checkpoint inference"),
    "selfplay": ("hivemind.cli.selfplay", "Run engine self-play (800 nodes, 100k mate nodes)"),
    "evaluate": ("hivemind.cli.evaluate_model", "Evaluate a network on self-play data"),
    "train": ("hivemind.training.train_loop", "Train with supervised or self-play data"),
    "prepare": ("hivemind.cli.train_from_games_parquet", "Prepare game shards and train"),
    "convert-selfplay": ("hivemind.data.convert_selfplay_data", "Convert self-play chunks to Parquet"),
    "fetch-network": ("hivemind.cli.fetch_network", "Download verified network weights"),
    "analyze": ("hivemind.cli.analyze_training_data", "Inspect training positions"),
    "search": ("hivemind.cli.search_training_fen", "Search training data by FEN"),
    "inspect-loader": ("hivemind.cli.inspect_data_loader", "Inspect loaded and augmented samples"),
    "verify-augmentation": ("hivemind.cli.verify_augmentation", "Display board-swap augmentation"),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="hivemind", description="Hivemind training and inference tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Commands:\n" + "\n".join(
            f"  {name:20} {description}" for name, (_, description) in COMMANDS.items()
        ) + "\n\nUse hivemind COMMAND --help for command options.",
    )
    parser.add_argument("command", choices=COMMANDS, metavar="COMMAND")
    args = parser.parse_args(argv[:1])
    module_name, _ = COMMANDS[args.command]
    previous_argv = sys.argv
    try:
        sys.argv = [f"hivemind {args.command}", *argv[1:]]
        return import_module(module_name).main()
    finally:
        sys.argv = previous_argv
