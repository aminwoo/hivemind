## Inference backends

The engine builds against either of two backends, selected with
`-DHIVEMIND_BACKEND=`:

|                      | `tensorrt` (default)                  | `onnxruntime`                   |
| -------------------- | ------------------------------------- | ------------------------------- |
| Requires             | CUDA 13+, TensorRT 10.14+, NVIDIA GPU | ONNX Runtime                    |
| Platforms            | Linux / Windows + NVIDIA              | Linux / macOS / Windows         |
| Precision            | FP16                                  | FP32 recommended; FP16 optional |
| Redistributable size | ~2 GB                                 | ~85 MB uncompressed             |

The TensorRT path is the default and is unchanged — existing build commands
behave exactly as before. The ONNX Runtime path is opt-in and exists so the
engine can be built and shipped without CUDA at all.

### Building the portable backend

```bash
python3 tools/fetch_onnxruntime.py          # ~11 MB, into third_party/
python3 -m pip install numpy onnx
python3 engine/scripts/convert_onnx_fp32.py model-fp16.onnx model-fp32.onnx
cmake -S engine -B engine/build-ort -G Ninja \
    -DHIVEMIND_BACKEND=onnxruntime \
    -DCMAKE_BUILD_TYPE=Release
cmake --build engine/build-ort -j "$(nproc)"
```

CMake finds the runtime in `third_party/onnxruntime`; pass
`-DONNXRuntime_ROOT=<dir>` to use one installed elsewhere.

Load the converted FP32 network on both Windows and Linux:

```bash
./engine/build-ort/hivemind.bin --model model-fp32.onnx
```

An FP16-compatible build remains available by adding
`-DHIVEMIND_ORT_FP16=ON`, but FP16 execution is usually much slower on the CPU.

On Windows, run these commands from a Developer PowerShell for Visual Studio:

```powershell
py tools/fetch_onnxruntime.py
py -m pip install numpy onnx
py engine/scripts/convert_onnx_fp32.py model-fp16.onnx model-fp32.onnx
cmake -S engine -B engine/build-ort -A x64 `
  -DHIVEMIND_BACKEND=onnxruntime `
  -DONNXRuntime_ROOT=third_party/onnxruntime
cmake --build engine/build-ort --config Release --parallel
```

The executable and `onnxruntime.dll` are written to
`engine/build-ort/Release`. MSVC builds use native compiler options; no CUDA,
Unix shell, or POSIX compatibility layer is needed.

### Building the TensorRT backend

On Linux:

```bash
cmake --preset ninja-fast \
    -DTensorRT_DIR=/path/to/TensorRT -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
cmake --build --preset ninja-fast -j "$(nproc)"
./build-ninja/hivemind \
	--network "$(realpath ../src/training/weights/rl/model-rl-final-v3.0.onnx)"
```

Self-play allocates a fixed Fairy-Stockfish mate-search budget per searched
position. The default 8,000,000 nodes are split across the active boards:

```bash
./build-ninja/hivemind selfplay \
  --network "$(realpath ../src/training/weights/rl/model-rl-final-v3.0.onnx)" \
  --games 1000 --nodes 400 --fairy-stockfish-mate-nodes 8000000
```

Set `--fairy-stockfish-mate-nodes 0` to disable the probe.

On Windows, use a Developer PowerShell for Visual Studio and point CMake at
the extracted TensorRT SDK and CUDA Runtime redistributable:

```powershell
cmake -S engine -B engine/build-tensorrt -A x64 `
  -DHIVEMIND_BACKEND=tensorrt `
  -DTensorRT_DIR=C:\path\to\TensorRT-11.1.0.106 `
  -DCUDA_TOOLKIT_ROOT_DIR=C:\path\to\cuda-runtime
cmake --build engine/build-tensorrt --config Release --parallel
```

The Windows backend loads TensorRT builder-resource DLLs from the executable
directory. FP32-to-FP16 conversion uses `.venv\\Scripts\\python.exe` or
`python` and can be overridden with `HIVEMIND_PYTHON`.

When a TensorRT plan is missing or stale, Hivemind accepts an FP32 ONNX model
and converts it to a temporary all-FP16 graph before building the plan. Plan
generation requires `onnx` and `onnxruntime` in `../.venv` or `python3`; set
`HIVEMIND_PYTHON` to select another Python interpreter. The cached TensorRT
plan uses FP16 inputs, outputs, weights, and floating-point activations.

## Experimental cross-board supply search

Two optional UCI settings make the search more sensitive to attacks supplied by
captures on the other board:

```text
setoption name SupplyPolicyWeightPermille value 400
setoption name SupplyValueWeightPermille value 500
```

Both default to **0 (disabled)** and accept 0–500. Changing either setting stops
the current search and clears retained search state so old priors and values are
not reused with the new configuration.

`SupplyPolicyWeightPermille` blends a fraction of each board's neural policy
with weights for useful captures, quiet moves that newly attack transferable
pieces, checks against an exposed king, and pawn moves that vacate a potential
flight square beside both the king and an attacked king-only shield pawn. For
example, it recognizes `...Bb4` as preparation to capture a knight that can be
dropped against the partner-board king, and `...e6` as opening e7 before an f7
sacrifice. The blend guides ordinary search below the root as well as at the root;
400 means 40% of marginal policy mass, **not** a guarantee of 40% of visits or
wall-clock time. Quiet positions retain their original policy.

`SupplyValueWeightPermille` adds a bounded checking-drop pressure term to
nonterminal neural evaluations. It distinguishes pieces already in hand,
attacked pieces on the feed board, and pieces that might be exchanged later.
It uses the most exposed king on each team, accounts for promoted pieces
transferring as pawns, and contributes nothing with bare kings and no supply.
Checking-drop potential is bounded per piece type rather than multiplied by
the number of geometrically available checking squares.
500 caps its additive contribution at 0.5 on the neural value scale [-1, 1].

These features use geometric threat estimates. They do not fabricate pocket
pieces, change move/sitting legality, or turn a heuristic into a solved mate.
They add bounded work to leaf preparation, within the existing search deadline,
without extra inference calls. The current exposure heuristic requires the
victim king to have left its back rank; the flight-square policy hint can act
before that happens. Neither is a complete king-safety model. Supply captures
are geometric estimates and can be pinned.

The initial 200/500 experiment changed one of the repeated opening exchange
decisions and retained the baseline's 47/48 tactical-suite matches at 700 ms.
A 12-game paired test at 100 ms finished 6–5–1, which is inconclusive and includes
one move-limit draw. The feature remains experimental and opt-in; it does not
establish an overall playing-strength improvement.

The final flight-square experiment does not select `...e6` reliably: even at
400/500 it sometimes returns `...Nc6`, and 500/500 returns `...Nc6` in repeated
3000-ms searches in the original orientation. The engine change is therefore
candidate discovery only, not the mechanism used to guarantee this opening.
It passes 250 engine tests. No updated tournament strength claim is made and
both UCI defaults remain disabled.

A subsequent direct comparison cautions against treating `...Be6` as best:
after forcing each candidate alongside partner Nf3 and giving the opponent a
3000-ms search, the disabled engine rates `...e6` at -292/-269 cp versus
`...Be6` at -406/-406 cp across board orientations (scores from the original
team's perspective). The 200/500 heuristic reverses this ranking. Legal replay
also confirms that `...e6` blocks e6 checking drops and opens Ke7 after N@g5+.
Thus interrupting the original sequence is not enough to justify a general
engine deployment. Nachos handles the asserted exact move through a dedicated
one-position forced opening book instead.

Compare settings using the same model on each side:

```sh
./build-supply/hivemind tournament \
  --contender /path/to/network.onnx --baseline /path/to/network.onnx \
  --contender-batch-size 16 --baseline-batch-size 16 \
  --contender-supply-policy-weight 0.2 --baseline-supply-policy-weight 0 \
  --contender-supply-value-weight 0.5 --baseline-supply-value-weight 0 \
  --games 100 --movetime 700 --seed 19 --output tournament_results/supply
```

Tournament arguments use fractions (0–0.5); UCI uses permille. Settings follow
the contestant across color swaps and are included in `summary.json`.

## Progressive-widening tournament

Use the same network on both sides to isolate different progressive-widening
coefficients. Each value controls widening at both root and non-root nodes, and
the settings follow contender and baseline when colors swap:

```bash
NETWORK=/path/to/network.onnx

./build-ninja/hivemind tournament \
	--contender "$NETWORK" \
	--baseline "$NETWORK" \
	--games 100 \
	--nodes 800 \
	--output tournament_results/pw-1.5-vs-1.0 \
	--seed 1 \
	--contender-pw-coefficient 1.5 \
	--baseline-pw-coefficient 1.0
```

Both coefficients are recorded in `summary.json`.

For a fixed-time batch-size comparison, use the same network on both sides:

```bash
./build-ninja/hivemind tournament \
	--contender "$NETWORK" \
	--baseline "$NETWORK" \
	--games 20 \
	--movetime 1000 \
	--contender-batch-size 8 \
	--baseline-batch-size 16 \
	--output tournament_results/batch-8-vs-16 \
	--seed 1
```

For multi-parameter strength sweeps, the resumable runner forms the Cartesian
product of repeated `--axis` values. It supports batch size, worker count,
MCGS/transpositions, root mate search, progressive widening, WDL weight,
moves-left discount, and Q selection:

```bash
./scripts/run_strength_sweep.py \
    --engine ./build-ninja/hivemind.bin \
    --model "$NETWORK" \
    --games 400 --nodes 800 \
    --axis batch-size=8,16,32 \
    --axis threads=1,2,4 \
    --sprt-elo0 0 --sprt-elo1 8 \
    --positions tournament_positions.tsv \
    --output tournament_results/batch-workers --resume
```

The optional positions file is tab-separated: `dual FEN`, `white|black` team
to play, and `true|false` time advantage. Comment lines begin with `#`. Each
position is used for a complete color-swapped pair. Sequential stopping is
evaluated only after both games in a pair; `summary.json` also records measured
full-search NPS for each contestant and every effective search parameter.

cmake --preset ninja-release -DTensorRT_DIR=/home/ben/opt/TensorRT-11.1.0.106 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
cmake --build --preset ninja-release -j "$(nproc)"

## TensorRT release bundles

Build a generic x86-64 Linux ZIP containing the engine, an ONNX network, and
the local TensorRT and CUDA runtime libraries:

```bash
./scripts/package_ubuntu_release.sh \
    --model /path/to/model.onnx \
    --name hivemind-v2.2.2-linux-x86_64-tensorrt
```

Recipients need a supported NVIDIA GPU and proprietary NVIDIA driver, but do
not need to install CUDA or TensorRT. The first launch builds a TensorRT plan
for their GPU and caches it beside the bundled model.

Build the equivalent package from a Developer PowerShell on Windows:

```powershell
python engine/scripts/package_windows_tensorrt_release.py `
  --model C:\path\to\model.onnx `
  --tensorrt-root C:\path\to\TensorRT-11.1.0.106 `
  --cuda-root C:\path\to\cuda-runtime `
  --name hivemind-v2.2.2-windows-x86_64-tensorrt
```

## Portable Windows and Linux release bundles

Build the ONNX Runtime CPU bundle natively on either operating system:

```bash
python tools/fetch_onnxruntime.py
python -m pip install numpy onnx
python engine/scripts/package_portable_release.py \
    --model model-fp16.onnx --name hivemind-v2.2.2-linux-x86_64-onnxruntime
```

On Windows, use the same Python command and a
`hivemind-v2.2.2-windows-x86_64-onnxruntime` name. The ZIP includes the engine,
an automatically converted FP32 model, ONNX Runtime, licenses, and checksums.
It does not require a GPU.
Tagged builds create ONNX Runtime and TensorRT archives for both Linux and
Windows through `.github/workflows/portable-release.yml`.
