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
uv run hivemind rl-loop --help
uv run python -m pytest
```

Commands live in `hivemind.cli` and load dependencies only when selected.
`python -m hivemind` is equivalent to the installed `hivemind` command.
Library code belongs in the domain, data, architectures, inference, or distill
packages. Workspace paths are centralized in `hivemind.paths`, and pinned
networks in `hivemind.network`.

Python tests mirror those subsystems under `tests/`.

## Files produced at runtime

- `data/distill/`: self-play and distillation data, and RL loop runs.
- `artifacts/distill/`: training checkpoints and ONNX exports.
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
