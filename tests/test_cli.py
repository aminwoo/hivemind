"""CLI dispatch and workspace path behavior independent of training data."""

from types import SimpleNamespace
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hivemind import cli, paths


def test_dispatch_forwards_options_and_restores_process_arguments(monkeypatch):
    original_argv = sys.argv
    captured = {}

    def command_main():
        captured["argv"] = sys.argv[:]
        return 7

    def import_command(name):
        captured["module"] = name
        return SimpleNamespace(main=command_main)

    monkeypatch.setattr(cli, "import_module", import_command)
    assert cli.main(["infer", "--starting", "--top-k", "3"]) == 7
    assert captured == {
        "module": "hivemind.cli.infer_from_fen",
        "argv": ["hivemind infer", "--starting", "--top-k", "3"],
    }
    assert sys.argv is original_argv


def test_help_does_not_import_command_dependencies(monkeypatch, capsys):
    def unexpected_import(name):
        pytest.fail(f"Help imported {name}")

    monkeypatch.setattr(cli, "import_module", unexpected_import)
    with pytest.raises(SystemExit) as result:
        cli.main(["--help"])
    assert result.value.code == 0
    assert "fetch-network" in capsys.readouterr().out


def test_workspace_override_is_resolved(tmp_path, monkeypatch):
    monkeypatch.setenv("HIVEMIND_WORKSPACE", str(tmp_path))
    assert paths.workspace_root() == tmp_path


def test_editable_workspace_does_not_follow_working_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("HIVEMIND_WORKSPACE", raising=False)
    expected = paths.workspace_root()
    monkeypatch.chdir(tmp_path)
    assert paths.workspace_root() == expected
    assert (expected / "pyproject.toml").is_file()


def test_training_outputs_are_outside_source_tree():
    assert paths.TRAINING_OUTPUT_DIR == paths.PROJECT_ROOT / "artifacts/training"


def test_no_command_starts_the_engine(monkeypatch):
    captured = {}

    def import_command(name):
        captured["module"] = name
        return SimpleNamespace(main=lambda: 0)

    monkeypatch.setattr(cli, "import_module", import_command)
    assert cli.main([]) == 0
    assert captured["module"] == "hivemind.cli.engine"


def test_engine_command_converts_the_network_for_coreml_on_macos(tmp_path, monkeypatch):
    from hivemind import network as networks
    from hivemind.cli import engine

    network = tmp_path / "net.onnx"
    converted = tmp_path / "net-coreml.onnx"
    monkeypatch.setattr(engine, "DEFAULT_ONNX_PATH", network)
    monkeypatch.setattr(networks.sys, "platform", "darwin")
    conversions = []

    def convert(command, check):
        conversions.append(command)
        Path(command[-1]).touch()

    monkeypatch.setattr(networks.subprocess, "run", convert)
    assert engine.default_model() is None
    network.touch()
    assert engine.default_model() == converted
    assert conversions == [[networks.sys.executable, str(networks.COREML_CONVERTER), str(network), str(converted)]]
    assert engine.default_model() == converted
    assert len(conversions) == 1
    monkeypatch.setattr(networks.sys, "platform", "linux")
    assert engine.default_model() == network


def test_engine_command_falls_back_when_coreml_conversion_fails(tmp_path, monkeypatch):
    from hivemind import network as networks
    from hivemind.cli import engine

    network = tmp_path / "net.onnx"
    network.touch()
    monkeypatch.setattr(engine, "DEFAULT_ONNX_PATH", network)
    monkeypatch.setattr(networks.sys, "platform", "darwin")

    def fail(command, check):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(networks.subprocess, "run", fail)
    assert engine.default_model() == network


def test_engine_command_prefers_the_tensorrt_build(tmp_path, monkeypatch):
    from hivemind.cli import engine

    tensorrt = tmp_path / "build-ninja" / "hivemind"
    onnxruntime = tmp_path / "build-ort" / "hivemind"
    monkeypatch.setattr(engine, "ENGINE_CANDIDATES", (tensorrt, onnxruntime))
    assert engine.default_engine() is None
    onnxruntime.parent.mkdir()
    onnxruntime.touch()
    assert engine.default_engine() == onnxruntime
    tensorrt.parent.mkdir()
    tensorrt.touch()
    os.utime(tensorrt, (0, 0))
    assert engine.default_engine() == tensorrt


def test_train_plan_matches_the_rl_loop():
    from hivemind.cli.train import plan

    # rl-it6: 7,705,551 new positions x1.5 and 23,096,955 replayed x0.5.
    assert plan(7_705_551, 23_096_955, 1.5, 0.5, 1024) == (22566, 0.4998)
    assert plan(1000, 0, 1.5, 0.5, 1024) == (2, 0.0)


def test_selfplay_runs_on_a_one_worker_copy(tmp_path):
    from hivemind.cli.selfplay import one_worker_copy, refresh_copy

    model = tmp_path / "net.onnx"
    model.write_bytes(b"it4")
    copy = one_worker_copy(model)
    assert copy == tmp_path / "net-sp1.onnx"
    assert one_worker_copy(copy) == copy
    refresh_copy(model, copy)
    assert copy.read_bytes() == b"it4"
    model.write_bytes(b"it6")
    refresh_copy(model, copy)
    assert copy.read_bytes() == b"it6"


def test_engine_backend_reads_the_cmake_cache(tmp_path):
    from hivemind.cli import engine

    binary = tmp_path / "hivemind"
    assert engine.engine_backend(binary) is None
    (tmp_path / "CMakeCache.txt").write_text(
        "CMAKE_BUILD_TYPE:STRING=Release\nHIVEMIND_BACKEND:STRING=onnxruntime\n")
    assert engine.engine_backend(binary) == "onnxruntime"


def test_onnxruntime_engine_gets_the_fp32_network_off_macos(tmp_path, monkeypatch):
    from hivemind import network as networks
    from hivemind.cli import engine

    network = tmp_path / "net.onnx"
    network.touch()
    monkeypatch.setattr(engine, "DEFAULT_ONNX_PATH", network)
    monkeypatch.setattr(engine.sys, "platform", "linux")
    conversions = []

    def convert(command, check):
        conversions.append(command)
        Path(command[-1]).touch()

    monkeypatch.setattr(networks.subprocess, "run", convert)
    assert engine.default_model("tensorrt") == network
    assert engine.default_model("onnxruntime") == tmp_path / "net-fp32.onnx"
    assert engine.default_model("onnxruntime") == tmp_path / "net-fp32.onnx"
    assert conversions == [[networks.sys.executable, str(networks.FP32_CONVERTER),
                            str(network), str(tmp_path / "net-fp32.onnx")]]


def _missing_engine(tmp_path, monkeypatch):
    from hivemind.cli import engine

    monkeypatch.setattr(engine, "ENGINE_CANDIDATES", (tmp_path / "build-ninja" / "hivemind",))
    monkeypatch.setattr(engine, "ONNXRUNTIME_DIR", tmp_path / "third_party" / "onnxruntime")
    network = tmp_path / "net.onnx"
    network.touch()
    monkeypatch.setattr(engine, "DEFAULT_ONNX_PATH", network)
    monkeypatch.setattr(engine, "default_model", lambda backend=None: network)
    calls = []
    built = tmp_path / "build-ninja" / "hivemind"

    def build(preset, backend, **kwargs):
        calls.append(("build", preset, backend, kwargs))
        return built

    monkeypatch.setattr(engine.build_engine, "build", build)
    monkeypatch.setattr(engine.subprocess, "run",
                        lambda command, check: calls.append(("run", command)))
    monkeypatch.setattr(engine.os, "execv", lambda path, command: calls.append(("exec", command)))
    return engine, calls, built, network


def test_engine_command_builds_tensorrt_when_it_is_installed(tmp_path, monkeypatch):
    engine, calls, built, network = _missing_engine(tmp_path, monkeypatch)
    monkeypatch.setattr(engine.build_engine, "tensorrt_available", lambda: True)
    monkeypatch.setenv("TENSORRT_DIR", str(tmp_path / "trt"))
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "cuda"))
    monkeypatch.setattr(sys, "argv", ["hivemind engine"])
    engine.main()
    assert calls == [
        ("build", "ninja-release", "tensorrt",
         {"tensorrt_dir": tmp_path / "trt", "cuda_root": tmp_path / "cuda"}),
        ("exec", [str(built), "--model", str(network.resolve())]),
    ]


def test_engine_command_builds_onnxruntime_without_tensorrt(tmp_path, monkeypatch):
    engine, calls, built, network = _missing_engine(tmp_path, monkeypatch)
    monkeypatch.setattr(engine.build_engine, "tensorrt_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["hivemind engine"])
    engine.main()
    assert calls == [
        ("run", [sys.executable, str(engine.ONNXRUNTIME_FETCHER)]),
        ("build", "ninja-release", "onnxruntime", {"tensorrt_dir": None, "cuda_root": None}),
        ("exec", [str(built), "--model", str(network.resolve())]),
    ]


def test_engine_command_does_not_build_on_a_dry_run(tmp_path, monkeypatch):
    engine, calls, _, _ = _missing_engine(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["hivemind engine", "--dry-run"])
    with pytest.raises(SystemExit):
        engine.main()
    assert calls == []


def test_engine_command_explains_a_missing_toolchain(tmp_path, monkeypatch):
    engine, _, _, _ = _missing_engine(tmp_path, monkeypatch)
    monkeypatch.setattr(engine.build_engine, "tensorrt_available", lambda: True)

    def no_cmake(*args, **kwargs):
        raise FileNotFoundError("cmake")

    monkeypatch.setattr(engine.build_engine, "build", no_cmake)
    monkeypatch.setattr(sys, "argv", ["hivemind engine"])
    with pytest.raises(SystemExit, match="Install CMake"):
        engine.main()


def test_tensorrt_is_available_with_its_headers_and_cuda(tmp_path, monkeypatch):
    from hivemind.cli import build_engine

    monkeypatch.setattr(build_engine.sys, "platform", "linux")
    monkeypatch.setattr(build_engine, "TENSORRT_HEADER_DIRS", ())
    monkeypatch.delenv("TensorRT_DIR", raising=False)
    monkeypatch.setenv("TENSORRT_DIR", str(tmp_path / "trt"))
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "cuda"))
    assert not build_engine.tensorrt_available()
    (tmp_path / "trt" / "include").mkdir(parents=True)
    (tmp_path / "trt" / "include" / "NvInfer.h").touch()
    assert not build_engine.tensorrt_available()
    (tmp_path / "cuda" / "include").mkdir(parents=True)
    (tmp_path / "cuda" / "include" / "cuda_runtime.h").touch()
    assert build_engine.tensorrt_available()
    monkeypatch.setattr(build_engine.sys, "platform", "darwin")
    assert not build_engine.tensorrt_available()
