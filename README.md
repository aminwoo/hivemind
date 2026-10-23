<div align="center">

![hivemind-logo](https://github.com/aminwoo/hivemind/assets/124148472/d42c6a6e-ab2e-4d7a-bf90-4876d59c9558)

# Hivemind

A free bughouse chess engine and self-play training system, powered by neural networks and Monte Carlo graph search.

[Latest release](https://github.com/aminwoo/hivemind/releases/latest) · [Build guide](engine/README.md) · [Training guide](docs/training.md) · [Development guide](docs/development.md)

</div>

Hivemind searches both bughouse boards together, including partner captures,
piece drops, waiting, and clock-dependent decisions. Release bundles include
**twin-s-clock**, a small two-board network improved through self-play RL.
Inference runs on CPU, NVIDIA GPUs through TensorRT, or Apple silicon through
Core ML.

## Choose how to use Hivemind

| Goal | Start here | Tools required |
|---|---|---|
| Run the released engine in a GUI or terminal | [Download and run](#download-and-run) | No CMake, compiler, Python, or uv |
| Build or modify the engine from a checkout | [Build from a clone](#build-from-a-clone) | Python, uv, CMake, a C++ compiler, and backend dependencies |
| Generate games for a distributed training run | [Contribute self-play games](#contribute-self-play-games) | Python setup plus a one-worker engine build |
| Train or continue a network | [Training](#training) | Python training dependencies, datasets, and a matching checkpoint |

## Download and run

[Download the latest release](https://github.com/aminwoo/hivemind/releases/latest).
Each ZIP includes the compiled engine, network, inference runtime, and licenses.
You do not need to clone this repository or download a separate model.

| Your computer | ZIP name contains | Approximate download |
|---|---|---|
| Windows x86-64, CPU | `windows-x86_64-onnxruntime` | 9 MB |
| Ubuntu 24.04+ x86-64, CPU | `linux-x86_64-onnxruntime` | 15 MB |
| Apple silicon, macOS 14+ | `macos-arm64-onnxruntime` | 15 MB |
| Windows x86-64, NVIDIA GPU | `windows-x86_64-tensorrt` | 2 GB |
| Ubuntu 24.04+ x86-64, NVIDIA GPU | `linux-x86_64-tensorrt` | 2 GB |

1. Download the matching ZIP and `.zip.sha256` file, then [verify the checksum](docs/distribution.md#verify-your-download).
2. Extract the entire ZIP into a writable folder, keeping its directories together.
3. Select `bin\hivemind.exe` on Windows or the top-level `hivemind` launcher on Linux/macOS in a compatible GUI.

From a terminal, run `.\bin\hivemind.exe` on Windows or `./hivemind` on
Linux/macOS. The program speaks UCI and waits for commands; it does not open
a graphical board. Bughouse requires a frontend that understands two-board
positions and Hivemind's joint moves.

The CPU bundles are the easiest starting point on Windows/Linux. NVIDIA
bundles include TensorRT 11.1 and CUDA Runtime 13.3; use a Turing or newer GPU
(compute capability 7.5+, such as RTX 20 series+) and driver R580 or newer.
R610+ is recommended for the CUDA 13.3 baseline. No separate CUDA or TensorRT
installation is needed. First launch builds a GPU-specific cache and can take
several minutes. Allow at least 6 GB free for the download, extracted files,
and cache. Windows builds also require the Visual C++ x64 Redistributable.

The macOS release launcher requests the Neural Engine through Core ML.
Current macOS downloads are unsigned and not notarized; see the
[macOS launch instructions](docs/distribution.md#macos-launch).
Release notes record which hardware paths were validated.

## Build from a clone

**A clone contains source code, not a ready-to-run engine.** Build the C++
engine before running it or generating self-play games. `uv sync` installs the
Python package and dependencies; it does not install CMake, a compiler, CUDA,
or TensorRT, and it does not build or automatically download an engine.

### Prerequisites

- Git, Python 3.13+, and [uv](https://docs.astral.sh/uv/).
- A C++23 compiler and system development tools. On macOS, install Xcode
  Command Line Tools; on Windows, use a Visual Studio Developer PowerShell
  with the C++ build tools available.
- CMake **3.25+** and Ninja for `hivemind build-engine`. The supplied CMake
  presets use schema version 6, which requires CMake 3.25.
- ONNX Runtime for CPU/Core ML builds, downloaded below; or a compatible
  TensorRT/CUDA development installation for NVIDIA builds. The release
  packaging baseline is TensorRT 11.1.0.106 with CUDA 13.3.

Install system tools before following the build examples. Full platform
instructions and manual CMake commands are in the [engine build guide](engine/README.md).

### Install the Python tools

Run from a terminal:

```bash
git clone https://github.com/aminwoo/hivemind.git
cd hivemind
uv sync
uv run hivemind --help
uv run hivemind fetch-network
```

The download is pinned and SHA-256 verified. It installs
`engine/models/twin-s-clock.onnx`, the model bundled in releases.
The Python environment currently includes training dependencies such as
PyTorch even if you only need an engine helper.

Examples below run from the repository root and use `uv run`. To run commands
directly, activate the environment once in each terminal:

```bash
source .venv/bin/activate   # Linux/macOS, bash or zsh
hivemind --help
```

On Windows PowerShell, activation is `.\.venv\Scripts\Activate.ps1`.
`python -m hivemind` provides the same CLI. With no subcommand, `hivemind`
starts the engine; it requires an existing build.

### CPU build on Linux

```bash
uv run python tools/fetch_onnxruntime.py
uv run hivemind build-engine --backend onnxruntime --preset ninja-release
uv run python engine/scripts/convert_onnx_fp32.py \
  engine/models/twin-s-clock.onnx engine/models/twin-s-clock-fp32.onnx
uv run hivemind engine --model engine/models/twin-s-clock-fp32.onnx
```

The model conversion happens once and avoids slow FP16 execution on CPU.
Both Ninja presets write to `engine/build-ninja`; `ninja-release` optimizes
for runtime performance, while the default `ninja-fast` favors build speed.
The presets optimize for the local CPU. Use release ZIPs when sharing binaries.

For Windows CPU builds, follow the [Windows CMake commands](engine/README.md#building-the-portable-backend).
They produce `engine/build-ort/Release/hivemind.exe` and copy ONNX Runtime
beside it. Convert the model to FP32 as above and pass its path explicitly.

### Apple silicon build

After the Python setup and clock-network download:

```bash
uv run python tools/fetch_onnxruntime.py
uv run hivemind build-engine --backend onnxruntime --preset ninja-release
uv run python engine/scripts/convert_onnx_coreml.py \
  engine/models/twin-s-clock.onnx engine/models/twin-s-clock-coreml.onnx
uv run hivemind engine
```

The conversion rewrites the graph for Core ML; `hivemind engine` also runs it
on first launch if you skip it. The engine runs on the Neural Engine by
default; `HIVEMIND_COREML=gpu` selects GPU, `all` allows both, and `off`
forces CPU. The engine prints the selected provider and warns if Core ML
initialization falls back to CPU.

### NVIDIA build on Linux

After installing TensorRT and CUDA development dependencies and downloading
the clock network, supply their actual installation directories:

```bash
uv run hivemind build-engine --backend tensorrt --preset ninja-release \
  --tensorrt-dir /path/to/TensorRT --cuda-root /path/to/cuda
uv run hivemind engine --model engine/models/twin-s-clock.onnx
```

On Linux/Windows, `hivemind build-engine` defaults to TensorRT unless you
pass `--backend onnxruntime`; macOS defaults to ONNX Runtime. The first
TensorRT launch builds and caches a plan for the model, GPU, and batch size.
For Windows NVIDIA builds, see the [TensorRT build guide](engine/README.md#building-the-tensorrt-backend).

### Use an existing engine

The Python launcher can also use an extracted release or another build:

```bash
uv run hivemind engine --engine /path/to/release/hivemind \
  --model /path/to/release/models/hivemind.onnx
```

On Windows, point `--engine` to `bin\hivemind.exe`. Other Python commands
have their own engine/model arguments; the self-play requirements below
also apply when selecting an existing executable.

## Download the network

Release ZIPs already contain their model. Source users can choose a verified
artifact with `hivemind fetch-network` or the bootstrap script:

```bash
uv run hivemind fetch-network                                  # default: released twin-s-clock
uv run hivemind fetch-network --variant onnx                   # older twin-s-noattn
uv run hivemind fetch-network --variant checkpoint             # older twin-s training checkpoint
```

`python tools/fetch_network.py --variant clock` downloads the same artifact
without installing the Python package. Models go to `engine/models`; the
`checkpoint` variant goes to `artifacts/distill/twin-s-noattn/best.pt`.
Pins and hashes live in [network.py](src/hivemind/network.py).

The RL loop still defaults to the older twin-s-noattn generator, because
training needs its PyTorch checkpoint and an ONNX file cannot substitute for
one. See the [training guide](docs/training.md#reinforcement-learning).

## Engine usage

These examples use the Linux/macOS source-build launcher. For a release,
use its top-level `./hivemind`; on Windows, use the corresponding `.exe`.

```bash
./engine/build-ninja/hivemind --model engine/models/twin-s-clock.onnx
./engine/build-ninja/hivemind perft 5
./engine/build-ninja/hivemind bench --model engine/models/twin-s-clock.onnx
```

For an ONNX Runtime CPU build, use the FP32-converted model; on macOS,
use the Core ML-converted model. Pass the chosen model explicitly rather
than relying on directory scanning when several networks are present.

### Team and clocks

`Team white` means White on board A and Black on board B. UCI play defaults
to **`TimeModel clock`**, with a level lead and 100 ms move times. Send the
current team lead and move-time estimates before searching:

```text
uci
setoption name Team value white
setoption name TimeModel value clock
setoption name TimeLeadMs value 4200
setoption name TeamMoveTimeMs value 1000
setoption name OpponentMoveTimeMs value 1500
isready
position startpos
go movetime 1000
```

The search updates the lead along each path and only allows waiting while
the team's lead lasts. `RootMoveTimeMs` can set the cost of the searched root
move explicitly. For the older fixed sit-right behavior, select
`TimeModel flag` and set `TimeAdvantage true` or `false`.
The Python RL loop and native self-play retain **flag** defaults; request
`--time-model clock` when generating clock-aware games.

### Analysis and variants

Set `MultiPV` for several ranked lines and `ReportIntervalMs` for periodic
reports throughout a bughouse MCTS search:

```text
setoption name MultiPV value 3
setoption name ReportIntervalMs value 250
```

`ReportIntervalMs 0` reports rank 1 at new depths and the full block at the
end. `OpeningNoise true` enables optional opening diversity; it is off by
default. `BatchSize` controls neural evaluation batching; each TensorRT
batch size needs its own cached plan.

The same engine/network also supports Chess, Crazyhouse, Antichess,
Chess960, Atomic, and Three-check through `UCI_Variant`. It was trained for
bughouse; strength in the other variants has not been established by a
rating test. See the [variant and engine reference](engine/README.md) and
[Lichess bot guide](docs/lichess-bot.md).

## Contribute self-play games

Complete the Python setup and install your backend's build dependencies
first. Distributed contributors use an engine compiled with **one search
worker per game**, separate from the usual four-worker UCI build.
The default release bundles are four-worker builds, so build a one-worker
engine for this workflow.

On Linux with NVIDIA/TensorRT, configure with your TensorRT and CUDA paths
(or an environment already configured as described in the build guide):

```bash
cmake -S engine -B engine/build-sp1 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DHIVEMIND_BACKEND=tensorrt \
  -DHIVEMIND_SEARCH_WORKERS=1 \
  -DTensorRT_DIR=/path/to/TensorRT -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda
cmake --build engine/build-sp1 --target hivemind --parallel
```

On macOS, download ONNX Runtime first, then build:

```bash
uv run python tools/fetch_onnxruntime.py
cmake -S engine -B engine/build-ort-sp1 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DHIVEMIND_BACKEND=onnxruntime \
  -DBUILD_TESTING=OFF -DHIVEMIND_SEARCH_WORKERS=1
cmake --build engine/build-ort-sp1 --target hivemind --parallel
```

Get the exchange URL from the training-run operator and authenticate with a
Hugging Face account/token that can access it and submit batches. A private
exchange requires access from its owner. Replace `OWNER/DATASET` below:

```bash
uv run hf auth login
uv run hivemind contribute --exchange hf://OWNER/DATASET
```

The command chooses `engine/build-sp1/hivemind` on Linux or
`engine/build-ort-sp1/hivemind` on macOS; `--engine` overrides it. It downloads
the network specified by each work order, prepares Core ML on macOS, plays
batches, and submits them as dataset pull requests. It waits while no work
is accepting games. Use `--once` to stop after one batch, or
`--games N --parallel-games N` to adjust the workload.

The built-in exchange default is `hf://aminwoo/bughouse-twin-s-selfplay`;
confirm which exchange the operator is currently using.

## Training

Training uses the Python package, a matching PyTorch checkpoint, and
self-play data. Generating games requires the compiled C++ engine.
The RL loop additionally needs a one-worker self-play build and a match
build; installing the Python package alone is not sufficient.

```bash
uv run hivemind selfplay --help
uv run hivemind train --help
uv run hivemind rl-loop --help
uv run hivemind distill-train --help
```

`selfplay` writes HDST search-target chunks; `train` continues a twin-s
checkpoint. `rl-loop` coordinates self-play, replay, training, matches,
promotion, and optional Hugging Face uploads. It runs indefinitely unless
bounded with `--last`. The loop uses POSIX locking and process control;
run it on Linux/macOS. `distill-train` trains a twin-board network from
search targets or from a teacher network's outputs.

See the [training guide](docs/training.md) for checkpoint setup, clock-aware
lineages, distributed runs, archives, manual training stages, and teacher
distillation.

## Neural network architecture

The released twin-s-clock model processes each board through a shared
convolutional trunk with symmetric board exchanges. It takes 78 input planes
(39 per board), including the team's time lead and relative move speed,
and returns policies for both boards plus value, WDL, and moves-left heads.
Older flag networks use 74 planes and remain loadable. See the
[architecture reference](docs/training.md#neural-network-architecture).

### Input representation

The [channel layout](docs/training.md#input-representation) documents the
78-plane clock input and the older 74-plane format. Encoding is implemented
in [planes.cc](engine/src/environment/planes.cc) and mirrored by the Python
representation code.

## Development and tests

The [development guide](docs/development.md) covers the package layout.
Python tests run with:

```bash
uv run python -m pytest
```

For a Linux ONNX Runtime test build, after downloading the runtime:

```bash
cmake -S engine -B engine/build-test -G Ninja \
  -DHIVEMIND_BACKEND=onnxruntime -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build engine/build-test --target hivemind_tests --parallel
ctest --test-dir engine/build-test --output-on-failure
```

[RULES.md](RULES.md) summarizes how the engine scores mate races.
Source code lives in `src/hivemind/` (Python) and `engine/` (C++). Downloaded
models and cached plans live in `engine/models/`; generated data and training
artifacts live under `data/` and `artifacts/`. Release packaging and hardware
validation are documented in the [distribution guide](docs/distribution.md).

## License

MIT License — see [LICENSE](LICENSE).

## Acknowledgments

- [Fairy-Stockfish](https://github.com/fairy-stockfish/Fairy-Stockfish) for move generation.
- [CrazyAra](https://github.com/QueensGambit/CrazyAra) for architecture inspiration.
