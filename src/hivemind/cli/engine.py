#!/usr/bin/env python3
"""Start the built UCI engine with the default network."""

import argparse
import os
from pathlib import Path
import sys

from hivemind.network import DEFAULT_ONNX_PATH
from hivemind.paths import PROJECT_ROOT

ENGINE_CANDIDATES = (
    PROJECT_ROOT / "engine" / "build-ninja" / "hivemind",
    PROJECT_ROOT / "engine" / "build-ort" / "hivemind",
)


def default_engine():
    """The most recently built engine, or None when none has been built."""
    built = [path for path in ENGINE_CANDIDATES if path.is_file()]
    return max(built, key=lambda path: path.stat().st_mtime, default=None)


def default_model():
    """The default network, preferring its Core ML conversion on macOS."""
    if sys.platform == "darwin":
        coreml = DEFAULT_ONNX_PATH.with_name(f"{DEFAULT_ONNX_PATH.stem}-coreml.onnx")
        if coreml.is_file():
            return coreml
    return DEFAULT_ONNX_PATH if DEFAULT_ONNX_PATH.is_file() else None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Unrecognised options are passed through to the engine "
               "(see ./engine/build-ort/hivemind --help).",
    )
    parser.add_argument("--engine", type=Path, help="Path to the built engine binary")
    parser.add_argument("--model", type=Path, help="ONNX network to load")
    parser.add_argument("--dry-run", action="store_true", help="Print the command and exit")
    args, extra = parser.parse_known_args()

    engine = args.engine or default_engine()
    if engine is None or not engine.is_file():
        sys.exit("Engine binary not found; build it first (see engine/README.md) or pass --engine.")
    model = args.model or default_model()

    command = [str(engine)]
    if model is not None:
        command += ["--model", str(model.resolve())]
    command += extra
    print(" ".join(command), file=sys.stderr, flush=True)
    if args.dry_run:
        return 0
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
