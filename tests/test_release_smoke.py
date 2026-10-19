"""Failure modes that must prevent a broken bundle from passing release checks."""

import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest


spec = importlib.util.spec_from_file_location(
    "smoke_release", Path(__file__).resolve().parents[1] / "engine/scripts/smoke_release.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def test_archive_checksum_rejects_corruption(tmp_path):
    archive = tmp_path / "release.zip"
    archive.write_bytes(b"original")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n")
    smoke.verify_archive(archive)
    archive.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        smoke.verify_archive(archive)


@pytest.fixture
def engine(tmp_path):
    path = tmp_path / "fake-engine"
    path.write_text(f'''#!{sys.executable}
import sys
for command in sys.stdin:
    command = command.strip()
    if command == 'uci':
        print('uciok', flush=True)
    elif command == 'isready':
        print('info string backend ONNX Runtime (CPU) model bundled.onnx', flush=True)
        print('readyok', flush=True)
    elif command.startswith('go '):
        print('info depth 2 nodes 200 score cp 0', flush=True)
        print('bestmove (e2e4,e7e5)', flush=True)
    elif command == 'quit':
        break
''')
    path.chmod(0o755)
    return path


def test_protocol_handshakes_and_two_searches(engine, tmp_path):
    report = smoke.exercise(engine, "cpu", 5, tmp_path)
    assert len(report["searches"]) == 2


def test_coreml_check_rejects_cpu_fallback(engine, tmp_path):
    with pytest.raises(RuntimeError, match="Expected coreml backend"):
        smoke.exercise(engine, "coreml", 5, tmp_path)


def test_early_engine_exit_is_a_failure(engine, tmp_path):
    engine.write_text(f"#!{sys.executable}\nraise SystemExit(0)\n")
    with pytest.raises((RuntimeError, BrokenPipeError)):
        smoke.exercise(engine, "cpu", 5, tmp_path)


def test_engine_that_stalls_cannot_pass(engine, tmp_path):
    engine.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(60)\n")
    with pytest.raises(TimeoutError):
        smoke.exercise(engine, "cpu", 0.2, tmp_path)
