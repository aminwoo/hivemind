#!/usr/bin/env python3
"""Configure and build the C++ Hivemind engine with a CMake preset."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

from hivemind.paths import PROJECT_ROOT

# Where CMake looks for TensorRT when TensorRT_DIR is not given, and where
# distribution packages install its headers.
TENSORRT_HEADER_DIRS = (
    Path("/usr/src/tensorrt/include"),
    Path("/usr/include"),
    Path("/usr/include/x86_64-linux-gnu"),
    Path("/usr/include/aarch64-linux-gnu"),
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def environment_tensorrt_dir():
    """TensorRT_DIR (or TENSORRT_DIR) from the environment, if set."""
    value = os.environ.get("TensorRT_DIR") or os.environ.get("TENSORRT_DIR")
    return Path(value).expanduser() if value else None


def environment_cuda_root():
    """The CUDA root named by CUDA_TOOLKIT_ROOT_DIR, CUDA_HOME or CUDA_PATH, if any."""
    for name in ("CUDA_TOOLKIT_ROOT_DIR", "CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(name):
            return Path(os.environ[name]).expanduser()
    return None


def tensorrt_available() -> bool:
    """Whether a TensorRT build can be configured here: TensorRT's headers and a CUDA install."""
    if sys.platform == "darwin":
        return False
    tensorrt_dir = environment_tensorrt_dir()
    header_dirs = ((tensorrt_dir / "include",) if tensorrt_dir else ()) + TENSORRT_HEADER_DIRS
    cuda_root = environment_cuda_root() or Path("/usr/local/cuda")
    return (any((directory / "NvInfer.h").is_file() for directory in header_dirs)
            and (cuda_root / "include" / "cuda_runtime.h").is_file())


def build(preset="ninja-fast", backend=None, jobs=None, tensorrt_dir=None, cuda_root=None,
          clean_first=False, configure_only=False) -> Path:
    """Configures and builds the engine; returns the executable's path."""
    engine_dir = PROJECT_ROOT / "engine"
    if not (engine_dir / "CMakeLists.txt").is_file():
        raise FileNotFoundError(f"C++ engine source not found at {engine_dir}")

    configure = ["cmake", "--preset", preset]
    backend = backend or ("onnxruntime" if sys.platform == "darwin" else None)
    if backend:
        configure.append(f"-DHIVEMIND_BACKEND={backend}")
    if sys.platform == "darwin":
        # The C++ tests have not been built on macOS (see engine/README.md).
        configure.append("-DBUILD_TESTING=OFF")
    if tensorrt_dir:
        configure.append(f"-DTensorRT_DIR={Path(tensorrt_dir).expanduser().resolve()}")
    if cuda_root:
        configure.append(f"-DCUDA_TOOLKIT_ROOT_DIR={Path(cuda_root).expanduser().resolve()}")

    print(shlex.join(configure), flush=True)
    subprocess.run(configure, cwd=engine_dir, check=True)
    executable = engine_dir / "build-ninja" / "hivemind"
    if configure_only:
        return executable

    command = ["cmake", "--build", "--preset", preset, "--target", "hivemind"]
    if jobs:
        command.extend(("--parallel", str(jobs)))
    if clean_first:
        command.append("--clean-first")

    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=engine_dir, check=True)
    return executable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=("ninja-fast", "ninja-release"),
        default="ninja-fast",
        help="CMake configure/build preset (default: ninja-fast)",
    )
    parser.add_argument(
        "--backend",
        choices=("tensorrt", "onnxruntime"),
        help="override the inference backend (default: onnxruntime on macOS, "
        "which has no TensorRT; otherwise CMake's tensorrt)",
    )
    parser.add_argument("--jobs", type=_positive_int, help="parallel build jobs")
    parser.add_argument("--tensorrt-dir", type=Path, help="TensorRT installation root")
    parser.add_argument("--cuda-root", type=Path, help="CUDA Toolkit or runtime root")
    parser.add_argument(
        "--clean-first", action="store_true", help="clean the target before building"
    )
    parser.add_argument(
        "--configure-only", action="store_true", help="configure without compiling"
    )
    args = parser.parse_args()

    try:
        executable = build(args.preset, args.backend, args.jobs, args.tensorrt_dir,
                           args.cuda_root, args.clean_first, args.configure_only)
    except FileNotFoundError as error:
        parser.error(str(error))
    if not args.configure_only:
        print(f"Built {executable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
