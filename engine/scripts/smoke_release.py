#!/usr/bin/env python3
"""Verify a release checksum, then exercise its UCI engine from another directory.

The release is a ZIP bundle or a single-file executable; a single file is run
alone in an empty directory and must use its built-in network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import zipfile


def verify_archive(archive: Path) -> None:
    # Version numbers put dots in single-file names, so append rather than
    # replace a suffix.
    checksum = archive.with_name(archive.name + ".sha256")
    expected, filename = checksum.read_text().strip().split(maxsplit=1)
    if filename != archive.name:
        raise ValueError(f"Checksum names {filename}, expected {archive.name}")
    with archive.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {archive.name}")


def exercise(engine: Path, backend: str, timeout: float, cwd: Path,
             embedded: bool = False) -> dict:
    env = os.environ.copy()
    env["HIVEMIND_COREML"] = "ane" if backend == "coreml" else "off"
    # Dependencies must come from the bundle, not the developer's library path.
    env.pop("LD_LIBRARY_PATH", None)
    output: queue.Queue[str | None] = queue.Queue()
    lines: list[str] = []
    started = time.monotonic()
    deadline = started + timeout
    process = subprocess.Popen(
        [str(engine)], cwd=cwd, env=env, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )

    def read_output() -> None:
        assert process.stdout
        for line in process.stdout:
            output.put(line.rstrip())
        output.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()

    def send(command: str) -> None:
        assert process.stdin
        process.stdin.write(command + "\n")
        process.stdin.flush()

    def until(prefix: str) -> str:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for {prefix}")
            try:
                line = output.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(f"Timed out waiting for {prefix}") from None
            if line is None:
                raise RuntimeError(f"Engine exited before {prefix}")
            lines.append(line)
            print(line, flush=True)
            if line.startswith(prefix):
                return line

    try:
        send("uci")
        until("uciok")
        send("isready")
        until("readyok")
        moves = []
        for _ in range(2):
            begin = len(lines)
            send("position startpos")
            send("go nodes 200")
            move = until("bestmove ")
            if any(invalid in move for invalid in ("0000", "(none)")):
                raise RuntimeError(f"Invalid starting-position move: {move}")
            counts = [int(n) for n in re.findall(
                r"\bnodes (\d+)", "\n".join(lines[begin:]))]
            if not counts or max(counts) < 200:
                raise RuntimeError("Search did not complete 200 nodes")
            moves.append(move)
        transcript = "\n".join(lines)
        markers = {
            "cpu": "(CPU) model",
            "coreml": "(Core ML Neural Engine) model",
            "tensorrt": "loaded TensorRT engine:",
        }
        if markers[backend].lower() not in transcript.lower():
            raise RuntimeError(f"Expected {backend} backend was not confirmed")
        if embedded and "model <embedded>" not in transcript:
            raise RuntimeError("The single-file engine did not load its built-in network")
        send("quit")
        process.wait(timeout=max(1, deadline - time.monotonic()))
        if process.returncode != 0:
            raise RuntimeError(f"Engine exited with status {process.returncode}")
        return {"backend": backend, "platform": platform.platform(),
                "searches": moves, "elapsed_seconds": round(time.monotonic() - started, 2)}
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        reader.join(timeout=5)
        if process.stdin:
            process.stdin.close()
        if process.stdout:
            process.stdout.close()


def exercise_single_file(executable: Path, backend: str, timeout: float, root: Path) -> dict:
    """Runs a single-file engine as a user would: alone in a directory, with
    no models or libraries beside it."""
    alone = root / "alone"
    alone.mkdir()
    engine = alone / ("hivemind.exe" if os.name == "nt" else "hivemind")
    shutil.copy2(executable, engine)
    engine.chmod(0o755)
    # The notices are UTF-8 whatever the console's code page.
    licenses = subprocess.run([str(engine), "--licenses"], capture_output=True,
                              encoding="utf-8", errors="replace", timeout=60,
                              check=True).stdout
    for required in ("Hivemind", "Fairy-Stockfish", "ONNX Runtime"):
        if required not in licenses:
            raise RuntimeError(f"--licenses does not include {required}")
    working = root / "elsewhere"
    working.mkdir()
    report = exercise(engine, backend, timeout, working, embedded=True)
    report["single_file_bytes"] = executable.stat().st_size
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--backend", choices=("cpu", "coreml", "tensorrt"), required=True)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    archive = args.archive.resolve()
    verify_archive(archive)
    with tempfile.TemporaryDirectory(prefix="hivemind-smoke-") as temp:
        root = Path(temp)
        if not zipfile.is_zipfile(archive):
            report = exercise_single_file(archive, args.backend, args.timeout, root)
            report["archive"] = archive.name
            if args.report:
                args.report.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
            return
        extracted = root / "bundle"
        extracted.mkdir()
        if os.name == "nt":
            with zipfile.ZipFile(archive) as source:
                source.extractall(extracted)
        else:
            # unzip preserves executable permissions and TensorRT symlinks.
            subprocess.run(["unzip", "-q", str(archive), "-d", str(extracted)], check=True)
        bundles = list(extracted.iterdir())
        if len(bundles) != 1 or not bundles[0].is_dir():
            raise ValueError("Expected one bundle directory")
        engine = bundles[0] / ("bin/hivemind.exe" if os.name == "nt" else "hivemind")
        report = exercise(engine, args.backend, args.timeout, root)
    report["archive"] = archive.name
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
