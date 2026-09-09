"""CLI dispatch and workspace path behavior independent of training data."""

from types import SimpleNamespace
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
