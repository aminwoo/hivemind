## Inference backends

Hivemind supports Bughouse, Chess, Crazyhouse, Antichess, Chess960, Atomic,
and Three-check with the same ONNX network.
Use `--model engine/models/twin-s-noattn.onnx` from the repository root for all
variants. The default UCI variant is Bughouse; select single-board play with
`setoption name UCI_Variant value chess`, `crazyhouse`, `antichess`, `chess960`,
`atomic`, or `3check`. Chess960 also accepts the standard combination of
`UCI_Variant=chess` and `UCI_Chess960=true`, with king-to-rook castling notation.

Single-board variants accept standard UCI positions and moves, including
castling, en passant, all promotions, and Crazyhouse drops such as `N@f3`.
Their `bestmove` is a single move; Bughouse keeps its dual-FEN positions,
board-prefixed move history, and joint `(moveA,moveB)` output. Single-board
play uses a neural PUCT search with legal moves and terminal rules from
Fairy-Stockfish, without Bughouse sitting or partner capture feeds. Its active
position occupies board A's 37 input planes; board B's 37 planes encode a
fixed starting position with White to move, empty pockets, and full castling
rights, using the usual opposite-team perspective. Both time-advantage planes
are zero. Draw contempt defaults to zero. Rook, bishop, and Antichess king
promotions share the queen promotion's policy prior but are evaluated as their
actual resulting positions.

The network was trained for Bughouse. This input adaptation adds rule-correct
single-board play without another network; playing strength in these variants
has not been established by a rating test. Antichess and Atomic kings use the
existing king input planes. Three-check counters are tracked by the rules and
search but have no dedicated input planes in this Bughouse-trained model.

Download the [published network](https://huggingface.co/aminwoo/bughouse-rise-v3)
from the repository root with `python tools/fetch_network.py`. Files are verified
and installed into `engine/models`. See the [network setup instructions](../README.md#download-the-network)
for native FP16, CPU conversion, and training checkpoints. Pass `--model PATH`
to explicitly select the network for your backend.

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
	--network "$(realpath ../artifacts/training/weights/rl/model-rl-final-v3.0.onnx)"
```

Self-play allocates a fixed Fairy-Stockfish mate-search budget per searched
position. The default 8,000,000 nodes are split across the active boards:

```bash
./build-ninja/hivemind selfplay \
  --network "$(realpath ../artifacts/training/weights/rl/model-rl-final-v3.0.onnx)" \
  --games 1000 --nodes 400 --fairy-stockfish-mate-nodes 8000000
```

Set `--fairy-stockfish-mate-nodes 0` to disable the probe.

### Background search for UCI frontends

MCTS keeps searching the opponents' replies after emitting `bestmove`. Stopping
that background search retains its tree for reuse on the next move.

If a frontend plays a different move, such as an opening-book move, send the
actual position followed by `go background` to restart this work there. Keep
`Team` and `TimeAdvantage` set for our team; the engine searches the opponents'
side internally. The command emits no `bestmove` and ends on the next
`position`, `go`, or `stop`, or when the background limits are reached.

Check for `option name BackgroundSearch type check default true` in the UCI
handshake before using the command. Setting `BackgroundSearch` to `false`
disables explicit `go background` commands; automatic background work after
normal MCTS searches continues. In alpha-beta mode, `go background` is ignored.

### INT8 networks

An INT8 copy of a network evaluates about 17% more positions per second in
MCTS on an RTX 4070 (four batch-8 workers). At 100 ms per move it beat the FP16
network 51-29 over 80 paired games (+98 Elo, 95% CI +41 to +161), while an
FP16-vs-FP16 control scored 40-39-1. At 1 s per move it scored 22-16-2 over 40
games (+53 Elo, 95% CI -4 to +112). A single inference stream gains nothing;
the extra throughput comes from the workers overlapping.

TensorRT 11 runs INT8 only from explicit quantize/dequantize nodes, so the
network is quantized ahead of time with NVIDIA ModelOpt, calibrated on
positions encoded by the engine itself:

```bash
python3 -m venv ~/.cache/hivemind-quantize
~/.cache/hivemind-quantize/bin/pip install "nvidia-modelopt[onnx]"
./build-ninja/hivemind dumpplanes --fens ../data/nnue/val/nnue_99_0_00000.fen \
    --output calib.f32 --every 122
~/.cache/hivemind-quantize/bin/python scripts/quantize_int8.py \
    models/network.onnx calib.f32 models/network-int8.onnx
./build-ninja/hivemind --model models/network-int8.onnx
```

The engine recognises a Q/DQ network and builds it at builder optimization
level 3: at the default level 5, TensorRT 11 times a fused kernel for the
cross-board blocks that faults on the device. Against FP16, the INT8 teacher
agrees on the top move in 93% (board A) and 92% (board B) of positions, with
a mean value difference of 0.023.

### Internal mate-probe experiment

The `InternalMateProbe` UCI combo is disabled by default and separates the
experiment's effects:

| Mode          | Candidate telemetry | Selection bias | Exact solver update                      |
| ------------- | ------------------- | -------------- | ---------------------------------------- |
| `off`         | no                  | no             | no                                       |
| `telemetry`   | yes                 | no             | no                                       |
| `bias`        | yes                 | yes            | no                                       |
| `certify`     | yes                 | yes            | only after exact two-board certification |
| `certifyonly` | yes                 | no             | only after exact two-board certification |

Use `setoption name InternalMateProbe value telemetry` to measure candidate
hit rates without changing play. That mode preserves 90% of the Fairy probe's
time for the established root probe. The strength-affecting `bias` and
`certify` modes cap that initial root phase at 10%, allowing internal hints to
arrive early enough to accumulate visits. All modes retain the shared node
budget, with 90% of its nodes reserved for the root probe in every mode. For an
untimed complete probe, the wall-clock cap does not apply, so the root probe
still runs first and may consume its full node share. A Fairy hit remains in
the hint table when exact certification fails; only successful certification
may update MCTS solver state.

`certifyonly` is the proof-only variant. A Fairy hit is neither promoted nor
published as a hint, so an unproven line cannot influence the move; it goes
straight to the exact certifier and only a certificate touches solver state.
The other strength modes skip a hit whose root edge the search already scores
as decided (|Q| above 0.8, about 5.4 pawns); `certifyonly` proves at any Q,
since a position already scored as lost is where a move into a forced mate
most needs telling apart from one that merely stays lost. It also probes with
the clock as it stands rather than assuming the target team can sit, so a
team behind on time is never credited with a single-board line.

### Selected-move certification

Independently of the experiment above, `CertifySelectedMove` (default `true`)
holds the moves MCTS is likely to play to a proof when this team is behind on
time. While the tree grows, the probe thread takes the leading root actions
by visits, plays each on a private board, asks Fairy-Stockfish for the
opponents' single-board mate from there (certified exactly before it counts)
and, failing that, the exact joint solver for a cross-board one - checks on
either board, pieces fed by capture - in slices, so an action that stops
leading stops being searched. A proof marks the child a solved loss at once,
which takes the action out of selection; an exhaustive refutation within
the six-attacker-move bound is final for the search; anything cut short
stays eligible for another slice. Each action keeps its proof cache between
slices. The thread is bounded by the move time and a node ceiling.

If the action finally chosen has no conclusive verdict - it took the lead
too late, or its search was cut short - a small serial tail (3% of the move,
at most 100ms, reserved off the front) asks the same question once more
and falls back through the next-most-visited alternatives on a proof.
Nothing runs when this team has the time advantage, since it may then sit
the threatened board and no single-board line is forced. Verbose output
reports both stages: `info string concurrent verifier: ...` and
`info string selected move certification: ...`.

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

## Alpha-beta NNUE engine

A second search mode replaces MCTS with alpha-beta over an NNUE value network
distilled from an ONNX teacher. It runs on the CPU, so a build with either
backend can use it without a GPU or a model file:

```bash
./build-ninja/hivemind --nnue ../artifacts/nnue/h512v2/best.nnue
```

The UCI options `NNUEFile` (loads a network and switches mode) and
`SearchMode` (`mcts` or `alphabeta`) select it at runtime; `go` accepts
`movetime`, `nodes`, `depth`, and `infinite`, and `bestmove` uses the same
`(moveA,moveB)` joint notation as MCTS. `Threads` runs Lazy SMP helpers that
share the transposition table, and `ABParams` takes comma-separated
`name=value` search parameters (`lmp_base`, `lmp_scale`, `pv_lmp_scale`,
`root_lmp_scale`, `lmr_divisor`, `qsearch_plies`, `rfp_margin`,
`futility_margin`, `qsearch_checks`, `check_extension`, `pair_ordering`,
`mate_probe`, `threads`).

When the team holds the time advantage, the Fairy-Stockfish root mate probe
that MCTS uses runs on its own thread beside the search. A mate that survives
the probe's two-board replay and the mate-race veto stops the search and is
played; `mate_probe=false` disables it.

Each team turn is searched as two nested choices by the same player, first on
board A and then on board B, with board B's options taken from the position
before board A's move. Terminal positions use the engine's own bughouse mate,
blockable-mate, and repetition rules. Pairs are pruned on the product of the
two boards' move-order ranks, and quiescence covers single-board captures plus
checks on its first turn.

### Distilling a network

1. Generate teacher-labelled positions. Games sample the teacher's raw policy
   with per-game temperature and occasional random moves; every position is
   stored with its sparse features and the teacher's value and WDL:

   ```bash
   ./build-ninja/hivemind gennnue --model teacher.onnx --positions 60000000 \
       --threads 4 --batch-size 256 --seed 1001 --output ../data/nnue/train
   ./build-ninja/hivemind gennnue --model teacher.onnx --positions 500000 \
       --seed 99 --fens true --output ../data/nnue/val
   ```

2. Train and export (`best.nnue` is written whenever validation improves).
   Pools larger than device memory rotate a random window of chunks per epoch;
   half of the positions have boards A and B swapped, an exact symmetry:

   ```bash
   uv run hivemind nnue-train --data data/nnue/train --val data/nnue/val \
       --out artifacts/nnue/run --hidden 512 --epochs 40
   ```

3. Check that the engine reproduces the trainer's outputs, then measure
   strength. `absearchbench` reports depth and nodes on FEN lines, and the
   tournament accepts an alpha-beta side through `--contender-nnue` or
   `--baseline-nnue` (search parameters via `--{contender,baseline}-ab-set`):

   ```bash
   ./build-ninja/hivemind nnueeval --nnue best.nnue --fens ../data/nnue/val/<chunk>.fen
   ./build-ninja/hivemind tournament --contender-nnue best.nnue \
       --contender-ab-movetime 500 --baseline teacher.onnx --nodes 128 \
       --games 40 --positions openings.tsv --output tournament_results/nnue
   ```

The feature layout is defined once in `src/nnue/features.h` and mirrored in
`src/hivemind/nnue/features.py`; `tests/test_nnue.cc` checks the team mirror
and board-swap permutations and that incremental accumulator updates match a
full refresh.

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
