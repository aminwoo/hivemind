#!/usr/bin/env python3
"""Start the UCI engine with the default network, building and downloading
whatever is missing on first use."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from hivemind.cli import build_engine
from hivemind.network import DEFAULT_ONNX_PATH, for_cpu, for_this_machine
from hivemind.paths import PROJECT_ROOT

# In order of preference: the TensorRT build runs on the GPU, while the ONNX
# Runtime build only accelerates through Core ML and is CPU-only elsewhere.
ENGINE_CANDIDATES = (
    PROJECT_ROOT / "engine" / "build-ninja" / "hivemind",
    PROJECT_ROOT / "engine" / "build-ort" / "hivemind",
)
ONNXRUNTIME_DIR = PROJECT_ROOT / "third_party" / "onnxruntime"
ONNXRUNTIME_FETCHER = PROJECT_ROOT / "tools" / "fetch_onnxruntime.py"


def default_engine():
    """The preferred built engine, or None when none has been built."""
    return next((path for path in ENGINE_CANDIDATES if path.is_file()), None)


def engine_backend(engine):
    """The inference backend an engine was configured with, from the CMake
    cache in its build directory, or None when it cannot be told."""
    try:
        cache = (Path(engine).parent / "CMakeCache.txt").read_text()
    except OSError:
        return None
    for line in cache.splitlines():
        if line.startswith("HIVEMIND_BACKEND:"):
            return line.partition("=")[2].strip()
    return None


def build_default_engine():
    """Builds the engine for this machine: TensorRT when it is installed,
    otherwise ONNX Runtime (downloading the runtime first)."""
    backend = "tensorrt" if build_engine.tensorrt_available() else "onnxruntime"
    print(f"No engine has been built yet; building one with {backend} "
          "(this takes a few minutes, once).", file=sys.stderr, flush=True)
    if backend == "onnxruntime" and not ONNXRUNTIME_DIR.is_dir():
        subprocess.run([sys.executable, str(ONNXRUNTIME_FETCHER)], check=True)
    tensorrt = backend == "tensorrt"
    return build_engine.build(
        "ninja-release", backend,
        tensorrt_dir=build_engine.environment_tensorrt_dir() if tensorrt else None,
        cuda_root=build_engine.environment_cuda_root() if tensorrt else None)


def default_model(backend=None):
    """The default network, converted for Core ML on macOS and to FP32 for an
    ONNX Runtime engine elsewhere, which runs it on the CPU."""
    if not DEFAULT_ONNX_PATH.is_file():
        return None
    try:
        if backend == "onnxruntime" and sys.platform != "darwin":
            return for_cpu(DEFAULT_ONNX_PATH)
        return for_this_machine(DEFAULT_ONNX_PATH)
    except subprocess.CalledProcessError:
        print("Network conversion failed; loading the unconverted network.", file=sys.stderr)
        return DEFAULT_ONNX_PATH


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Unrecognised options are passed through to the engine "
               "(see ./engine/build-ninja/hivemind --help).",
    )
    parser.add_argument("--engine", type=Path, help="Path to the built engine binary")
    parser.add_argument("--model", type=Path, help="ONNX network to load")
    parser.add_argument("--dry-run", action="store_true", help="Print the command and exit")
    args, extra = parser.parse_known_args()

    engine = args.engine or default_engine()
    if args.engine is not None and not engine.is_file():
        sys.exit(f"Engine binary not found: {engine}")
    if engine is None:
        if args.dry_run:
            sys.exit("No engine has been built yet; run without --dry-run to build one.")
        try:
            engine = build_default_engine()
        except FileNotFoundError as error:
            sys.exit(f"Cannot build the engine ({error}). Install CMake 3.25+, Ninja and "
                     "a C++23 compiler (see engine/README.md), or pass --engine.")
        except subprocess.CalledProcessError:
            sys.exit("Building the engine failed; see the output above and engine/README.md, "
                     "or pass --engine.")
    if args.model is None and not DEFAULT_ONNX_PATH.is_file() and not args.dry_run:
        from hivemind.cli.fetch_network import fetch_network
        print("Downloading the default network.", file=sys.stderr, flush=True)
        fetch_network()
    model = args.model or default_model(engine_backend(engine))

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
