# Development layout

The repository has two build targets: the Python package under `src/hivemind`
and the C++ engine under `engine`. Python imports start with `hivemind`; `src`
is a source directory, not an importable application package.

## Python

```bash
uv sync
uv run hivemind --help
uv run hivemind fetch-network
uv run hivemind infer --starting
uv run hivemind train --help
uv run hivemind prepare --help
uv run python -m pytest
```

Commands live in `hivemind.cli` and load dependencies only when selected.
`python -m hivemind` is equivalent to the installed `hivemind` command.
Library code belongs in the domain, data, architectures, inference, or training
packages. Configuration defaults live in `hivemind.config`; workspace paths
are centralized in `hivemind.paths`.

Python tests mirror those subsystems under `tests/`. Visual inspection commands
such as `hivemind inspect-loader` and `hivemind verify-augmentation` live with
the CLI, separate from automated tests.

## Files produced at runtime

- `data/`: source games and prepared training datasets.
- `artifacts/training/weights/{supervised,rl}/`: training checkpoints and exports.
- `artifacts/training/logs/`: training logs.
- `engine/models/`: downloaded networks and the engine's cached plans.

An editable install resolves these paths from the repository root, regardless
of the command's working directory. A wheel install uses the current directory.
Set `HIVEMIND_WORKSPACE=/path/to/workspace` to override either default. Explicit
command-line paths remain available for selecting data and networks.

## Engine and bootstrap tools

CMake and C++ tests remain within `engine/`; see its [build guide](../engine/README.md).
Engine packaging, network precision conversion, and UCI utilities live in
`engine/scripts/`. Release bundles include the FP16 converter's shared Python
helper so conversion does not require installing the training package.

The two scripts in `tools/` bootstrap dependencies before Python installation:

```bash
python tools/fetch_network.py
python tools/fetch_onnxruntime.py
```

## Migrating existing commands

| Previous command/path | Replacement |
| --- | --- |
| `python scripts/infer_from_fen.py` | `uv run hivemind infer` |
| `python scripts/evaluate_model.py` | `uv run hivemind evaluate` |
| `python scripts/train_from_games_parquet.py` | `uv run hivemind prepare` |
| `python src/training/train_loop.py` | `uv run hivemind train` |
| `python -m src.main` | `uv run hivemind checkpoint` |
| `python tools/uci_drive.py` | `python engine/scripts/uci_drive.py` |
| `src/training/weights/` | `artifacts/training/weights/` |
| `src/training/logs/` | `artifacts/training/logs/` |

Existing checkpoints remain compatible. If you have outputs from an older
checkout, move them to `artifacts/training/` or pass their paths explicitly.
