#!/usr/bin/env python3
"""Configure and build the C++ Hivemind engine with a CMake preset."""

import argparse
from pathlib import Path
import shlex
import subprocess

from hivemind.paths import PROJECT_ROOT


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


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
        help="override the inference backend",
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

    engine_dir = PROJECT_ROOT / "engine"
    if not (engine_dir / "CMakeLists.txt").is_file():
        parser.error(f"C++ engine source not found at {engine_dir}")

    configure = ["cmake", "--preset", args.preset]
    if args.backend:
        configure.append(f"-DHIVEMIND_BACKEND={args.backend}")
    if args.tensorrt_dir:
        configure.append(f"-DTensorRT_DIR={args.tensorrt_dir.expanduser().resolve()}")
    if args.cuda_root:
        configure.append(
            f"-DCUDA_TOOLKIT_ROOT_DIR={args.cuda_root.expanduser().resolve()}"
        )

    print(shlex.join(configure), flush=True)
    subprocess.run(configure, cwd=engine_dir, check=True)
    if args.configure_only:
        return 0

    build = ["cmake", "--build", "--preset", args.preset, "--target", "hivemind"]
    if args.jobs:
        build.extend(("--parallel", str(args.jobs)))
    if args.clean_first:
        build.append("--clean-first")

    print(shlex.join(build), flush=True)
    subprocess.run(build, cwd=engine_dir, check=True)
    executable = engine_dir / "build-ninja" / "hivemind"
    print(f"Built {executable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
