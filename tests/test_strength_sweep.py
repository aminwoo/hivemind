"""Exercise sweep resumption and effective option ordering with a fake engine."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "engine/scripts/run_strength_sweep.py"
spec = importlib.util.spec_from_file_location("strength_sweep", SCRIPT)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


@pytest.fixture
def sweep_run(tmp_path):
    engine = tmp_path / "engine"
    model = tmp_path / "model.onnx"
    engine.write_bytes(b"engine-v1")
    model.write_bytes(b"model-v1")
    args = [str(SCRIPT), "--engine", str(engine), "--model", str(model),
            "--output", str(tmp_path / "results"), "--games", "4", "--nodes", "100",
            "--axis", "root-pw-coefficient=4", "--axis", "pw-coefficient=1", "--resume"]
    commands = []

    def tournament(command, check):
        commands.append(command)
        output = Path(command[command.index("--output") + 1])
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(json.dumps(
            {"games": 4, "sprt": {"decision": "continue"}, "contender_score": .5}))
        return subprocess.CompletedProcess(command, 0)

    def run():
        with patch.object(sys, "argv", args), patch.object(sweep.subprocess, "run", tournament):
            assert sweep.main() == 0
        return commands

    return run, tmp_path, model


def test_explicit_root_override_applied_after_internal_coefficient(sweep_run):
    run, _, _ = sweep_run
    command = run()[0]
    assert command.index("--contender-pw-coefficient") < command.index("--contender-root-pw-coefficient")
    assert command[command.index("--dirichlet-epsilon") + 1] == "0.0"


def test_resume_requires_success_marker_and_unchanged_summary(sweep_run):
    run, root, _ = sweep_run
    assert len(run()) == 1
    assert len(run()) == 1
    marker = next((root / "results").glob("*/completed.json"))
    marker.unlink()
    assert len(run()) == 2
    marker.write_text('{"identity":')
    assert len(run()) == 3
    summary = next((root / "results").glob("*/summary.json"))
    summary.write_text(json.dumps({"games": 2, "sprt": {"decision": "continue"}}))
    assert len(run()) == 4
    assert json.loads((root / "results/sweep.json").read_text())["runs"][0]["games"] == 4


def test_changed_model_content_creates_new_run(sweep_run):
    run, root, model = sweep_run
    assert len(run()) == 1
    model.write_bytes(b"model-v2")
    assert len(run()) == 2
    assert len(list((root / "results").glob("*/completed.json"))) == 2


def test_failed_run_is_not_marked_complete(sweep_run):
    _, root, model = sweep_run
    args = [str(SCRIPT), "--engine", str(root / "engine"), "--model", str(model),
            "--output", str(root / "failed"), "--games", "4", "--nodes", "100",
            "--axis", "pw-mass=.35"]
    with patch.object(sys, "argv", args), patch.object(sweep.subprocess, "run",
            side_effect=subprocess.CalledProcessError(1, "tournament")):
        with pytest.raises(subprocess.CalledProcessError):
            sweep.main()
    assert not list((root / "failed").glob("*/completed.json"))
