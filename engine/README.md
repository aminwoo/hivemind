# Hivemind engine

The C++ engine: build instructions, backends, UCI options, and engine tools.
For ready-to-run downloads, start with the [download guide](../README.md#download-and-run).
The [distribution guide](../docs/distribution.md) lists exact release requirements,
first-launch behavior, checksums, and hardware validation commands.

## Inference backends

Hivemind supports Bughouse, Chess, Crazyhouse, Antichess, Chess960, Atomic,
and Three-check with the same ONNX network.
Use `--model engine/models/twin-s-clock.onnx` from the repository root for all
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
position occupies board A's input planes; board B's planes encode a
fixed starting position with White to move, empty pockets, and full castling
rights, using the usual opposite-team perspective. Both time-advantage planes
and all four clock planes are zero. Draw contempt defaults to zero. Rook, bishop, and Antichess king
promotions share the queen promotion's policy prior but are evaluated as their
actual resulting positions.

The network was trained for Bughouse. This input adaptation adds rule-correct
single-board play without another network; playing strength in these variants
has not been established by a rating test. Antichess and Atomic kings use the
existing king input planes. Three-check counters are tracked by the rules and
search but have no dedicated input planes in this Bughouse-trained model.

Download the [published network](https://huggingface.co/aminwoo/bughouse-twin-s-clock)
from the repository root with `python tools/fetch_network.py --variant clock`. Files are verified
and installed into `engine/models`. See the [network setup instructions](../README.md#download-the-network)
for native FP16, CPU conversion, and training checkpoints. Pass `--model PATH`
to explicitly select the network for your backend.

The engine builds against either of two backends, selected with
`-DHIVEMIND_BACKEND=`:

|                      | `tensorrt` (default)                  | `onnxruntime`                                  |
| -------------------- | ------------------------------------- | ---------------------------------------------- |
| Requires             | CUDA 13+, TensorRT 10.14+, NVIDIA GPU | ONNX Runtime                                   |
| Platforms            | Linux / Windows + NVIDIA              | Linux / macOS / Windows                        |
| Runs the network on  | NVIDIA GPU                            | CPU; Apple GPU through Core ML on macOS        |
| Precision            | FP16                                  | CPU: FP32 recommended; Core ML: FP16 network   |
| Redistributable size | ~2 GB                                 | ~85 MB uncompressed                            |

TensorRT is the default backend on Linux and Windows. ONNX Runtime lets the
engine be built and shipped without CUDA.

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

### Running on the Apple GPU (macOS)

On macOS the same backend runs the network on the GPU through ONNX Runtime's
Core ML provider, which the macOS runtime from `tools/fetch_onnxruntime.py`
includes; the CPU is left to the search. Keep the published FP16 network
rather than the FP32 conversion, and prepare it once for Core ML:

```bash
xcode-select --install
brew install cmake ninja
python3 tools/fetch_onnxruntime.py
python3 tools/fetch_network.py
python3 -m venv .venv && .venv/bin/pip install numpy onnx
.venv/bin/python engine/scripts/convert_onnx_coreml.py \
  engine/models/twin-s-clock.onnx \
  engine/models/twin-s-clock-coreml.onnx
cmake -S engine -B engine/build-ort -G Ninja \
    -DHIVEMIND_BACKEND=onnxruntime \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF
cmake --build engine/build-ort -j "$(sysctl -n hw.ncpu)"
./engine/build-ort/hivemind --model engine/models/twin-s-clock-coreml.onnx
```

The engine then reports `info string backend ONNX Runtime (Core ML Neural Engine)`.
`-DBUILD_TESTING=OFF` skips the C++ tests, which have not been built on
macOS.

As exported, a network splits into several Core ML partitions, because Core
ML has no kernel for a few of its ops, such as the Neg and Abs in twin-s's
value head. Every split adds a GPU-to-CPU round trip per
batch. The conversion replaces these ops with exact equivalents (the outputs
are bit-identical), so the whole graph is one partition. The engine warns
when it is given a network that has not been converted.

The engine also pins the batch dimension, since Core ML compiles for fixed
shapes, and gives each search worker its own Core ML session: ONNX Runtime
runs one prediction at a time per Core ML model, so separate sessions let the
workers' batches overlap on the accelerator, as TensorRT's per-worker contexts do.

`HIVEMIND_COREML` selects the compute units: `ane` (the default), `gpu`, `all`
(lets Core ML choose between them), or `off` for the CPU provider. On an M5
at the default batch of 8, the Neural Engine searches about 37k nps against
the GPU's 23k.
`hivemind bench` runs one worker at a time, so compare settings by the
search's nps instead:

```bash
(printf 'uci\nisready\nposition startpos\ngo movetime 10000\n'; sleep 12; echo quit) |
    HIVEMIND_COREML=gpu ./engine/build-ort/hivemind \
    --model engine/models/twin-s-clock-coreml.onnx | grep ' nps ' | tail -1
```

The GPU may also prefer larger batches than the default of 8; try
`--batch-size 16` and `32`.

### Building the TensorRT backend

On Linux:

```bash
cmake --preset ninja-fast \
    -DTensorRT_DIR=/path/to/TensorRT -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
cmake --build --preset ninja-fast -j "$(nproc)"
./build-ninja/hivemind --model models/twin-s-clock.onnx
```

Use the `ninja-release` preset instead for an optimized build. Both presets
write to `build-ninja` and optimize for the local CPU.

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

### Self-play

Self-play allocates a fixed Fairy-Stockfish mate-search budget per searched
position. The default 10,000,000 nodes are split across the active boards:

```bash
./build-ninja/hivemind selfplay --model models/twin-s-clock.onnx \
  --games 1000 --nodes 400 --fairy-stockfish-mate-nodes 10000000
```

Set `--fairy-stockfish-mate-nodes 0` to disable the probe.

`--time-model clock` plays every game on a simulated clock instead of letting
one team sit all game. Each game starts level with probability
`--level-start-fraction` (0.5); otherwise the white team's lead is drawn
log-uniformly from `--start-lead-min-ms` to `--start-lead-max-ms` (100 to
10000) with a random sign. Each team's move time is drawn log-uniformly from
`--move-time-min-ms` to `--move-time-max-ms` (100 to 3000). Searches and
adjudication follow the lead, and the PGN records the starting and final lead
and both move times.

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
normal MCTS searches continues.

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
./build-ninja/hivemind dumpplanes --fens ../data/distill/val/distill_99_0_00000.fen \
    --output calib.f32 --every 122
~/.cache/hivemind-quantize/bin/python scripts/quantize_int8.py \
    models/network.onnx calib.f32 models/network-int8.onnx
./build-ninja/hivemind --model models/network-int8.onnx
```

A network from before the clock planes reads 74 planes; calibrate it on
`dumpplanes --legacy-planes true`, which drops the clock planes.

The engine recognises a Q/DQ network and builds it at builder optimization
level 3: at the default level 5, TensorRT 11 times a fused kernel for the
cross-board blocks that faults on the device. Against FP16, the INT8 teacher
agrees on the top move in 93% (board A) and 92% (board B) of positions, with
a mean value difference of 0.023.

### Mate-probe threads

MCTS checks mate in one before starting neural inference. Its deeper exact
root mate scan searches up to M8 on a separate cancellable thread, with a
private two-board position. It can use the whole MCTS search window, subject
to its scaled node budget. The first verified proof stops MCTS and the loss
scan and returns the mating move; stop requests, the move deadline and a
finished MCTS search cancel it. Threads are joined before returning a move or
replacing the tree.

`MateThreads` (default `4`, max `256`) sets how many Fairy-Stockfish workers
each mate probe searches with. The workers share
one transposition table, and the probe's node budget is their combined total,
so extra workers search more within the same move time and use up a node
budget sooner. Each worker adds about 20 MB of resident memory. On a
12-thread CPU, one probe searched 0.94M nodes/s with one worker, 3.6M with
four and 5.8M with eight, but eight reached no deeper in a second than four.
MCTS there keeps about three cores busy while it waits on the GPU, and lost
0.6% of its nodes per move to four workers. Lower the count on a machine with
fewer cores. The hand-rolled check-only mate scans are unaffected.

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

With `--time-model clock` and the self-play clock options, each pair draws one
clock and both of its games use it, so the contender plays each team's side of
the lead once. The positions file's time-advantage flags are then unused, and
the breakdown splits games by the contender's starting lead: up, down or level.

## Teacher distillation data

`gendistill` plays games by sampling a teacher network's raw policy, with a
per-game temperature and occasional random moves, and writes every position
as HDST distillation chunks (packed planes, the teacher's value, WDL,
moves-left and legal-move policies, averaged over both board orders) for
`hivemind distill-train`. `--fens true` also writes each chunk's positions
as `fenA;fenB;team;advantage` lines, which `dumpplanes` and `searchdiag` read:

```bash
./build-ninja/hivemind gendistill --model teacher.onnx --positions 500000 \
    --seed 99 --fens true --output ../data/distill/val
```

See the [training guide](../docs/training.md#distilling-from-a-teacher).

## TensorRT release bundles

Build a generic x86-64 Linux ZIP containing the engine, an ONNX network, and
the local TensorRT and CUDA runtime libraries:

```bash
./scripts/package_ubuntu_release.sh \
    --model /path/to/model.onnx \
    --name hivemind-v3.0.3-linux-x86_64-tensorrt
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
  --name hivemind-v3.0.3-windows-x86_64-tensorrt
```

## Single-file Windows, Linux, and macOS releases

The CPU and Mac releases are one executable each: ONNX Runtime and the C++
runtime are linked statically, and the network and license texts are
compiled in (`HIVEMIND_ORT_STATIC`, `HIVEMIND_EMBED_MODEL` and
`HIVEMIND_EMBED_LICENSES`). With a network compiled in, the engine uses it
unless `--model` names another; `--licenses` prints the licenses.

Microsoft ships ONNX Runtime only as a shared library, so first build it
from source as one static library (CPU only, plus Core ML on macOS; about 15
minutes on four cores, once). Then build and package the executable, natively
on each target operating system:

```bash
python -m pip install ninja numpy onnx
python engine/scripts/build_static_onnxruntime.py
python engine/scripts/package_single_file_release.py \
    --model model-fp16.onnx --name hivemind-v3.0.3-linux-x86_64
```

On Windows, run them from a Developer PowerShell (for `lib.exe`). The
network is converted to FP32 for the CPU, or for Core ML on macOS, before it
is embedded.

Tagged builds create the single executables and TensorRT bundles for
Linux/Windows through `.github/workflows/portable-release.yml`, which caches
the static ONNX Runtime. Assets are staged in a draft release for hardware
validation before publication; see the [release process](../docs/distribution.md#release-process).
