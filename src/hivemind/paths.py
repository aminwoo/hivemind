"""Workspace paths shared by commands and library code.

Editable installs use the repository root. Wheel installs use the current
working directory. HIVEMIND_WORKSPACE overrides either default.
"""

import os
from pathlib import Path


def workspace_root() -> Path:
    override = os.environ.get("HIVEMIND_WORKSPACE")
    if override:
        return Path(override).expanduser().resolve()
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "pyproject.toml").is_file() and (source_root / "engine").is_dir():
        return source_root
    return Path.cwd()


PROJECT_ROOT = workspace_root()
DATA_DIRECTORY = PROJECT_ROOT / "data"
TRAINING_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "training"
MODEL_DIRECTORY = PROJECT_ROOT / "engine" / "models"
