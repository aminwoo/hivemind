#!/usr/bin/env python3
"""Build a redistributable ONNX Runtime release for the current platform."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


ENGINE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = ENGINE_DIR.parent


def host_slug() -> str:
    names = {"Linux": "linux", "Windows": "windows", "Darwin": "macos"}
    try:
        system = names[platform.system()]
    except KeyError as error:
        raise SystemExit(f"Unsupported release platform: {platform.system()}") from error
    machine = platform.machine().lower()
    arch = "arm64" if machine in {"arm64", "aarch64"} else "x86_64"
    return f"{system}-{arch}"


def run(command: list[str]) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def find_built_engine(build_dir: Path) -> Path:
    filename = "hivemind.exe" if os.name == "nt" else "hivemind.bin"
    candidates = [path for path in build_dir.rglob(filename) if path.is_file()]
    if not candidates:
        raise SystemExit(f"CMake did not produce {filename} below {build_dir}")
    # Multi-config generators put the Release binary in a Release directory.
    candidates.sort(key=lambda path: ("Release" not in path.parts, len(path.parts)))
    return candidates[0]


def find_runtime(root: Path) -> Path:
    if os.name == "nt":
        names = ("onnxruntime.dll",)
    elif sys.platform == "darwin":
        names = ("libonnxruntime.dylib",)
    else:
        # Copy the ELF SONAME (normally .so.1), not just the developer symlink;
        # that is the filename recorded in the executable's DT_NEEDED entry.
        versioned = sorted((root / "lib").glob("libonnxruntime.so.*"),
                           key=lambda path: (len(path.name), path.name))
        if versioned:
            return versioned[0]
        names = ("libonnxruntime.so",)
    for name in names:
        for directory in (root / "lib", root / "bin"):
            candidate = directory / name
            if candidate.exists():
                return candidate
    raise SystemExit(f"ONNX Runtime shared library not found below {root}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path,
                        help="FP16 ONNX model to include")
    parser.add_argument("--onnxruntime-root", type=Path,
                        default=WORKSPACE_DIR / "third_party" / "onnxruntime")
    parser.add_argument("--output", type=Path, default=WORKSPACE_DIR / "dist")
    parser.add_argument("--name", help="archive/root directory name")
    parser.add_argument("--build-dir", type=Path,
                        default=ENGINE_DIR / "build-portable-release")
    args = parser.parse_args()

    model = args.model.resolve()
    runtime_root = args.onnxruntime_root.resolve()
    output_dir = args.output.resolve()
    build_dir = args.build_dir.resolve()
    bundle_name = args.name or f"hivemind-v2.2.1-{host_slug()}-onnxruntime"

    if not model.is_file() or model.suffix.lower() != ".onnx":
        raise SystemExit(f"A readable ONNX model is required: {model}")
    if not runtime_root.is_dir():
        raise SystemExit(
            f"ONNX Runtime not found at {runtime_root}; run "
            f"{sys.executable} tools/fetch_onnxruntime.py first."
        )

    run([
        "cmake", "-S", str(ENGINE_DIR), "-B", str(build_dir),
        "-DHIVEMIND_BACKEND=onnxruntime",
        "-DHIVEMIND_ORT_FP16=ON",
        "-DHIVEMIND_PORTABLE_BUNDLE=ON",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_TESTING=OFF",
        "-DHIVEMIND_FAST_BUILD=OFF",
        "-DHIVEMIND_NATIVE_ARCH=OFF",
        f"-DONNXRuntime_ROOT={runtime_root}",
    ])
    run(["cmake", "--build", str(build_dir), "--config", "Release", "--parallel"])

    engine = find_built_engine(build_dir)
    runtime = find_runtime(runtime_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="hivemind-release-", dir=output_dir) as temp:
        bundle = Path(temp) / bundle_name
        bin_dir = bundle / "bin"
        models_dir = bundle / "models"
        bin_dir.mkdir(parents=True)
        models_dir.mkdir()

        shutil.copy2(engine, bin_dir / engine.name)
        if os.name == "nt":
            shutil.copy2(runtime, bin_dir / runtime.name)
        shutil.copy2(model, models_dir / "hivemind.onnx")
        shutil.copy2(WORKSPACE_DIR / "LICENSE", bundle / "LICENSE.txt")
        shutil.copy2(ENGINE_DIR / "LICENSE", bundle / "FAIRY-STOCKFISH-LICENSE.txt")
        for source_name, target_name in (
            ("LICENSE", "ONNXRUNTIME-LICENSE.txt"),
            ("ThirdPartyNotices.txt", "ONNXRUNTIME-THIRD-PARTY-NOTICES.txt"),
        ):
            source = runtime_root / source_name
            if source.is_file():
                shutil.copy2(source, bundle / target_name)

        if os.name != "nt":
            lib_dir = bundle / "lib"
            lib_dir.mkdir(exist_ok=True)
            shutil.copy2(runtime, lib_dir / runtime.name)
            launcher = bundle / "hivemind"
            launcher.write_text(
                "#!/bin/sh\n"
                "set -eu\n"
                "bundle_dir=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\n"
                "export LD_LIBRARY_PATH=\"$bundle_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\"\n"
                "exec \"$bundle_dir/bin/hivemind.bin\" \"$@\"\n",
                encoding="utf-8",
            )
            launcher.chmod(0o755)

        executable = "bin\\hivemind.exe" if os.name == "nt" else "./hivemind"
        requirements = (
            "- 64-bit Windows 10 or newer\n"
            "  - Microsoft Visual C++ 2019 or newer Redistributable\n"
            if os.name == "nt"
            else "- A recent 64-bit Linux distribution\n"
        )
        (bundle / "README.txt").write_text(
            f"Hivemind {bundle_name}\n"
            f"{'=' * (10 + len(bundle_name))}\n\n"
            "Portable FP16 release powered by ONNX Runtime. No CUDA, TensorRT, "
            "or NVIDIA GPU is required.\n\n"
            f"Requirements:\n  {requirements}"
            "- A UCI-compatible chess GUI, or a terminal\n\n"
            f"Run {executable}. The bundled FP16 model is discovered automatically, "
            "including when a GUI starts the engine from another working directory.\n\n"
            "The bundled network remains in FP16 on both Windows and Linux. "
            "CPU performance depends on the host processor and ONNX Runtime.\n",
            encoding="utf-8",
        )

        zip_path = output_dir / f"{bundle_name}.zip"
        zip_path.unlink(missing_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(bundle.rglob("*")):
                archive.write(path, path.relative_to(bundle.parent))

    checksum_path = zip_path.with_suffix(zip_path.suffix + ".sha256")
    checksum_path.write_text(f"{sha256(zip_path)}  {zip_path.name}\n", encoding="ascii")
    print(f"Created {zip_path}")
    print(f"Created {checksum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
