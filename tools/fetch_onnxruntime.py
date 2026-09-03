#!/usr/bin/env python3
"""Download the ONNX Runtime release the portable backend links against.

The CPU build is ~11 MB compressed and is all the engine needs to run without
CUDA. CMake picks it up automatically from third_party/onnxruntime; pass
-DONNXRuntime_ROOT=<dir> to use one you installed yourself instead.

    python3 tools/fetch_onnxruntime.py
"""
import argparse
import io
import platform
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_VERSION = "1.29.0"
ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "third_party" / "onnxruntime"


def asset_name(version: str) -> str:
    system, machine = platform.system(), platform.machine().lower()
    if system == "Linux":
        arch = "aarch64" if machine in ("aarch64", "arm64") else "x64"
        return f"onnxruntime-linux-{arch}-{version}.tgz"
    if system == "Darwin":
        arch = "arm64" if machine in ("arm64", "aarch64") else "x86_64"
        return f"onnxruntime-osx-{arch}-{version}.tgz"
    if system == "Windows":
        return f"onnxruntime-win-x64-{version}.zip"
    raise SystemExit(f"Unsupported platform: {system} {machine}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=DEFAULT_VERSION)
    ap.add_argument("--force", action="store_true",
                    help="re-download even if third_party/onnxruntime exists")
    a = ap.parse_args()

    if TARGET.exists() and not a.force:
        print(f"{TARGET} already present; pass --force to replace it.")
        return 0

    name = asset_name(a.version)
    url = (
        "https://github.com/microsoft/onnxruntime/releases/download/"
        f"v{a.version}/{name}"
    )
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response:
        payload = response.read()
    print(f"  {len(payload) / 1e6:.1f} MB")

    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.parent.mkdir(parents=True, exist_ok=True)

    # Both archive kinds contain a single versioned top-level directory; strip
    # it so the include/ and lib/ paths CMake looks for land directly under
    # third_party/onnxruntime.
    staging = TARGET.parent / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    if name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.extractall(staging)
    else:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            archive.extractall(staging)

    extracted = [p for p in staging.iterdir() if p.is_dir()]
    if len(extracted) != 1:
        raise SystemExit(f"Unexpected archive layout: {extracted}")
    extracted[0].rename(TARGET)
    shutil.rmtree(staging)

    print(f"Installed {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
