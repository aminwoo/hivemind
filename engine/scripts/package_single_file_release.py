#!/usr/bin/env python3
"""Build a single-file ONNX Runtime release for the current platform.

The executable links ONNX Runtime and the C++ runtime statically and carries
the network and license texts inside it, so it runs with nothing beside it
(`hivemind --licenses` prints the licenses). Build the static ONNX Runtime
first with scripts/build_static_onnxruntime.py:

    python3 engine/scripts/package_single_file_release.py --model twin-s-clock.onnx \\
        --onnxruntime-root third_party/onnxruntime-static
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = ENGINE_DIR.parent
LICENSES = (
    ("Hivemind", WORKSPACE_DIR / "LICENSE"),
    ("Fairy-Stockfish", ENGINE_DIR / "LICENSE"),
)


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


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_licenses(onnxruntime_root: Path, output: Path) -> None:
    """The license texts the executable must carry, in one file."""
    sections = [(name, path.read_text(encoding="utf-8")) for name, path in LICENSES]
    for name, filename in (("ONNX Runtime", "LICENSE"),
                           ("ONNX Runtime third-party notices", "ThirdPartyNotices.txt")):
        path = onnxruntime_root / filename
        if not path.is_file():
            raise SystemExit(f"{path} is missing; the release must carry it")
        sections.append((name, path.read_text(encoding="utf-8", errors="replace")))
    output.write_text("".join(f"{'=' * 72}\n{name}\n{'=' * 72}\n\n{text.rstrip()}\n\n"
                              for name, text in sections), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, type=Path,
                        help="released ONNX network, converted to FP32 (Core ML on macOS) and embedded")
    parser.add_argument("--onnxruntime-root", type=Path,
                        default=WORKSPACE_DIR / "third_party" / "onnxruntime-static",
                        help="static ONNX Runtime from build_static_onnxruntime.py")
    parser.add_argument("--output", type=Path, default=WORKSPACE_DIR / "dist")
    parser.add_argument("--name", help="file name, without .exe")
    parser.add_argument("--build-dir", type=Path, default=ENGINE_DIR / "build-single-file")
    parser.add_argument("--macos-sign-identity",
                        help="Developer ID Application identity (default: ad-hoc signature)")
    parser.add_argument("--macos-notary-profile",
                        help="notarytool keychain profile; requires --macos-sign-identity")
    args = parser.parse_args()
    if args.macos_notary_profile and not args.macos_sign_identity:
        parser.error("--macos-notary-profile requires --macos-sign-identity")

    model = args.model.resolve()
    runtime_root = args.onnxruntime_root.resolve()
    build_dir = args.build_dir.resolve()
    output_dir = args.output.resolve()
    name = args.name or f"hivemind-{host_slug()}"
    if not model.is_file():
        raise SystemExit(f"A readable ONNX model is required: {model}")
    if not (runtime_root / "lib").is_dir():
        raise SystemExit(f"Static ONNX Runtime not found at {runtime_root}; "
                         "run engine/scripts/build_static_onnxruntime.py first.")

    with tempfile.TemporaryDirectory(prefix="hivemind-single-file-") as temp:
        temp_dir = Path(temp)
        # ONNX Runtime's CPU provider needs FP32; on macOS the network runs
        # through Core ML, which needs the rewrite that keeps it in one partition.
        converter = "convert_onnx_coreml.py" if sys.platform == "darwin" else "convert_onnx_fp32.py"
        embedded_model = temp_dir / "hivemind.onnx"
        run([sys.executable, str(ENGINE_DIR / "scripts" / converter), str(model), str(embedded_model)])
        licenses = temp_dir / "LICENSES.txt"
        write_licenses(runtime_root, licenses)

        run([
            "cmake", "-S", str(ENGINE_DIR), "-B", str(build_dir),
            "-DHIVEMIND_BACKEND=onnxruntime",
            "-DHIVEMIND_ORT_STATIC=ON",
            f"-DONNXRuntime_ROOT={runtime_root}",
            f"-DHIVEMIND_EMBED_MODEL={embedded_model}",
            f"-DHIVEMIND_EMBED_LICENSES={licenses}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TESTING=OFF",
            "-DHIVEMIND_FAST_BUILD=OFF",
            "-DHIVEMIND_NATIVE_ARCH=OFF",
        ])
        run(["cmake", "--build", str(build_dir), "--config", "Release", "--parallel",
             "--target", "hivemind"])

        engine = find_built_engine(build_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / (name + (".exe" if os.name == "nt" else ""))
        shutil.copy2(engine, target)

    if sys.platform == "darwin":
        run(["strip", "-x", str(target)])
        # Apple silicon runs only signed code; stripping drops the linker's
        # ad-hoc signature, so sign again.
        identity = args.macos_sign_identity or "-"
        run(["codesign", "--force", "--options", "runtime", "--sign", identity, str(target)])
        if args.macos_notary_profile:
            # notarytool takes archives, not bare executables.
            with tempfile.TemporaryDirectory(prefix="hivemind-notary-") as temp:
                archive = Path(temp) / (target.name + ".zip")
                run(["ditto", "-c", "-k", "--keepParent", str(target), str(archive)])
                run(["xcrun", "notarytool", "submit", str(archive), "--keychain-profile",
                     args.macos_notary_profile, "--wait"])
    elif os.name != "nt":
        run(["strip", str(target)])
    target.chmod(0o755)

    checksum = target.with_name(target.name + ".sha256")
    checksum.write_text(f"{sha256(target)}  {target.name}\n", encoding="ascii")
    print(f"Created {target} ({target.stat().st_size / 1e6:.1f} MB)")
    print(f"Created {checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
