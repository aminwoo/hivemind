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
