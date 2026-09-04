#!/usr/bin/env python3
"""Build a self-contained Windows x86-64 TensorRT release bundle."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


ENGINE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = ENGINE_DIR.parent


def run(command: list[str]) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_one(root: Path, name: str) -> Path:
    matches = [path for path in root.rglob(name) if path.is_file()]
    if len(matches) != 1:
        raise SystemExit(f"Expected one {name} below {root}, found: {matches}")
    return matches[0]


def copy_license(root: Path, destination: Path, target_name: str) -> None:
    for name in ("LICENSE.txt", "LICENSE"):
        candidates = [path for path in root.rglob(name) if path.is_file()]
        if candidates:
            candidates.sort(key=lambda path: len(path.parts))
            shutil.copy2(candidates[0], destination / target_name)
            return

    # NVIDIA's redistributable TensorRT package ships no license file, only the
    # documents gathered below. Bundle whatever attribution it does carry and
    # warn, rather than shipping the binaries with nothing alongside them.
    prefix = target_name.removesuffix("-LICENSE.txt")
    bundled = []
    for name in ("Acknowledgements.txt", "README.txt"):
        candidates = [path for path in root.rglob(name) if path.is_file()]
        if candidates:
            candidates.sort(key=lambda path: len(path.parts))
            shutil.copy2(candidates[0], destination / f"{prefix}-{name}")
            bundled.append(name)
    if bundled:
        print(f"warning: no license text below {root}; "
              f"bundled {', '.join(bundled)} instead")
    else:
        print(f"warning: no license or attribution files below {root}")


def main() -> int:
    if os.name != "nt":
        raise SystemExit("The Windows TensorRT bundle must be built natively on Windows.")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path,
                        help="FP16 ONNX model to include")
    parser.add_argument("--tensorrt-root", required=True, type=Path,
                        help="Extracted NVIDIA TensorRT SDK root")
    parser.add_argument("--cuda-root", required=True, type=Path,
                        help="Extracted NVIDIA CUDA Runtime root")
    parser.add_argument("--output", type=Path, default=WORKSPACE_DIR / "dist")
    parser.add_argument("--name", default="hivemind-v2.2.2-windows-x86_64-tensorrt")
    parser.add_argument("--build-dir", type=Path,
                        default=ENGINE_DIR / "build-windows-tensorrt-release")
    args = parser.parse_args()

    model = args.model.resolve()
    tensorrt_root = args.tensorrt_root.resolve()
    cuda_root = args.cuda_root.resolve()
    output_dir = args.output.resolve()
    build_dir = args.build_dir.resolve()

    if not model.is_file() or model.suffix.lower() != ".onnx":
        raise SystemExit(f"A readable ONNX model is required: {model}")
    for label, root in (("TensorRT", tensorrt_root), ("CUDA Runtime", cuda_root)):
        if not root.is_dir():
            raise SystemExit(f"{label} root not found: {root}")

    run([
        "cmake", "-S", str(ENGINE_DIR), "-B", str(build_dir), "-A", "x64",
        "-DHIVEMIND_BACKEND=tensorrt",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_TESTING=OFF",
        "-DHIVEMIND_FAST_BUILD=OFF",
        "-DHIVEMIND_NATIVE_ARCH=OFF",
        f"-DTensorRT_DIR={tensorrt_root}",
        f"-DCUDA_TOOLKIT_ROOT_DIR={cuda_root}",
    ])
    run(["cmake", "--build", str(build_dir), "--config", "Release", "--parallel"])

    engine = find_one(build_dir, "hivemind.exe")
    required_runtime = [
        find_one(tensorrt_root, "nvinfer_11.dll"),
        find_one(tensorrt_root, "nvonnxparser_11.dll"),
        find_one(cuda_root, "cudart64_13.dll"),
    ]
    builder_resources = sorted(
        path for path in tensorrt_root.rglob("nvinfer_builder_resource_*_11.dll")
        if path.is_file()
    )
    if not builder_resources:
        raise SystemExit(f"TensorRT builder resource DLLs not found below {tensorrt_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hivemind-release-", dir=output_dir) as temp:
        bundle = Path(temp) / args.name
        bin_dir = bundle / "bin"
        models_dir = bundle / "models"
        bin_dir.mkdir(parents=True)
        models_dir.mkdir()

        shutil.copy2(engine, bin_dir / "hivemind.exe")
        for runtime in required_runtime + builder_resources:
            shutil.copy2(runtime, bin_dir / runtime.name)
        shutil.copy2(ENGINE_DIR / "scripts" / "convert_onnx_fp16.py", bin_dir)
        shutil.copy2(model, models_dir / "hivemind.onnx")
        shutil.copy2(WORKSPACE_DIR / "LICENSE", bundle / "LICENSE.txt")
        shutil.copy2(ENGINE_DIR / "LICENSE", bundle / "FAIRY-STOCKFISH-LICENSE.txt")
        copy_license(tensorrt_root, bundle, "TENSORRT-LICENSE.txt")
        copy_license(cuda_root, bundle, "CUDA-RUNTIME-LICENSE.txt")

        (bundle / "README.txt").write_text(
            f"Hivemind {args.name}\n"
            f"{'=' * (10 + len(args.name))}\n\n"
            "Self-contained FP16 NVIDIA TensorRT build for 64-bit Windows.\n\n"
            "Requirements:\n"
            "  - 64-bit Windows 10 or newer\n"
            "  - A supported NVIDIA GPU and current proprietary NVIDIA driver\n"
            "  - Microsoft Visual C++ 2019 or newer Redistributable\n"
            "  - Enough free disk space for the GPU-specific TensorRT plan\n\n"
            "Configure bin\\hivemind.exe as the engine in a UCI-compatible GUI. "
            "The bundled FP16 model is found automatically even when the GUI uses "
            "another working directory.\n\n"
            "The first launch builds and caches a TensorRT plan beside the bundled "
            "ONNX model. This can take several minutes. Later launches reuse that "
            "plan. TensorRT plans are GPU- and software-version-specific.\n",
            encoding="utf-8",
        )

        zip_path = output_dir / f"{args.name}.zip"
        zip_path.unlink(missing_ok=True)
        with zipfile.ZipFile(
            zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
        ) as archive:
            for path in sorted(bundle.rglob("*")):
                archive.write(path, path.relative_to(bundle.parent))

    checksum_path = zip_path.with_suffix(zip_path.suffix + ".sha256")
    checksum_path.write_text(
        f"{sha256(zip_path)}  {zip_path.name}\n", encoding="ascii"
    )
    print(f"Created {zip_path}")
    print(f"Created {checksum_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
