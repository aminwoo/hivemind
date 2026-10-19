<div align="center">
  
  ![hivemind-logo](https://github.com/aminwoo/hivemind/assets/124148472/d42c6a6e-ab2e-4d7a-bf90-4876d59c9558)
  
  # Hivemind

A free and strong UCI Bughouse chess engine powered by deep reinforcement learning.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

## Download and run

[Download the latest release](https://github.com/aminwoo/hivemind/releases/latest).
The engine, **twin-s-clock network**, and inference runtime are included.
No Python, source build, or separate network download is needed.

| Your computer | Choose | Approximate ZIP size |
|---|---|---|
| Windows x86-64 | `windows-x86_64-onnxruntime` — easiest starting point | 9 MB |
| Linux x86-64 (Ubuntu 24.04+) | `linux-x86_64-onnxruntime` — easiest starting point | 15 MB |
| Mac with Apple silicon (macOS 14+) | `macos-arm64-onnxruntime` — Core ML / Neural Engine | 15 MB |
| Windows or Linux with a supported NVIDIA GPU | Matching `tensorrt` bundle — GPU performance option | About 2 GB |

1. Download the matching ZIP and its `.sha256` file, and [verify the checksum](docs/distribution.md#verify-your-download).
2. Extract the entire ZIP into a writable folder. Keep `bin`, `lib`, and `models` together.
3. Select `bin\hivemind.exe` (Windows) or the top-level `hivemind` launcher
   (Linux/macOS) in your bughouse-compatible GUI. From a terminal, run
   `.\bin\hivemind.exe` or `./hivemind`; it speaks UCI and waits for commands.

For NVIDIA builds, use a Turing or newer GPU (compute capability 7.5+, such as
RTX 20 series or newer) and driver R580+; R610+ is recommended with bundled
CUDA 13.3. You do not need to install CUDA or TensorRT. **First launch can take
several minutes** while building a GPU-specific cache. Allow at least 6 GB free
for the download, extracted bundle, and cache. CPU bundles start without this step.

The macOS download is currently unsigned and not notarized. See
[macOS launch instructions and compatibility details](docs/distribution.md).
The release notes list the validation performed for each platform.

## Overview

Hivemind is a neural network-based engine for [Bughouse chess](https://en.wikipedia.org/wiki/Bughouse_chess), a four-player chess variant played on two boards. The engine uses Monte Carlo Tree Search (MCTS) with a deep neural network for position evaluation and move prediction.

### Key Features

- **Neural Network Policy & Value Estimation** - Release bundles include twin-s-clock, a small two-board network improved by self-play RL with clock-aware evaluation
- **Monte Carlo Graph Search (MCGS)** - Shares nodes across transpositions for improved search efficiency
- **TensorRT Acceleration** - High-performance GPU inference using NVIDIA TensorRT
- **UCI Protocol** - Standard Universal Chess Interface for GUI compatibility
- **Self-Play Training** - RL training pipeline with self-play game generation
- **Alpha-Beta NNUE Mode** - CPU alpha-beta search on an NNUE distilled from the network (`--nnue`, see [engine/README.md](engine/README.md#alpha-beta-nnue-engine))

## Project Structure

```text
hivemind/
├── src/hivemind/        # Installable Python package
│   ├── architectures/  # Neural network definitions and model configuration
│   ├── cli/            # Training, inference, and data inspection commands
│   ├── config/         # Training and representation settings
│   ├── data/           # Game acquisition, encoding, and Parquet preparation
│   ├── domain/         # Bughouse board and move representation
│   ├── inference/      # ONNX and checkpoint inference
│   └── training/       # Data loaders, optimization, metrics, and schedules
├── engine/             # C++ UCI engine, with its own build and test setup
│   ├── src/            # Engine implementation and vendored Fairy-Stockfish
│   ├── tests/          # C++ tests
│   ├── scripts/        # Engine build, packaging, conversion, and UCI tools
│   └── models/         # Downloaded inference networks (ignored)
├── tests/              # Python tests grouped by subsystem
├── tools/              # Network and runtime bootstrapping without installation
├── docs/               # Development and layout guide
├── data/               # Game archives and prepared datasets (ignored)
└── artifacts/training/ # Generated checkpoints and logs (ignored)
```

See [the development guide](docs/development.md) for commands and migration notes.

## Requirements

### Engine (C++)

- CMake 3.16+
- C++23 compatible compiler
- Windows, Linux or macOS with ONNX Runtime (portable CPU build; on macOS the
  network runs on the Apple GPU through Core ML), or
- Windows or Linux with CUDA 13+ and TensorRT 10.14+ (NVIDIA GPU build)

### Training (Python)

- Python 3.13+
- PyTorch 2.9+
- See `pyproject.toml` for full dependencies

## Download the Network

Release bundles already include **twin-s-clock (RL iteration 3)** from
[aminwoo/bughouse-twin-s-clock](https://huggingface.co/aminwoo/bughouse-twin-s-clock).
For a source build, download the same pinned, SHA-256-verified network:

```bash
python tools/fetch_network.py --variant clock
```

The training tools retain the older twin-s defaults for compatibility with
existing training runs. The older network and its crossboard teacher remain available:

```bash
python tools/fetch_network.py --variant onnx                   # older twin-s
python tools/fetch_network.py --variant crossboard             # crossboard teacher
python tools/fetch_network.py --variant crossboard-fp16        # teacher, native FP16 input/output
python tools/fetch_network.py --variant crossboard-checkpoint  # teacher's PyTorch training checkpoint
```

Files go into `engine/models`, which the engine searches automatically. Use
`--model PATH` to select a specific file when multiple networks are present.
The published ONNX networks contain FP16 weights internally. For the portable
CPU backend, convert the download once:

```bash
python engine/scripts/convert_onnx_fp32.py \
  engine/models/twin-s-clock.onnx \
  engine/models/twin-s-clock-fp32.onnx
```

Python inference defaults to the standard downloaded ONNX file:

```bash
hivemind infer --starting
hivemind checkpoint --device cpu  # Requires --variant crossboard-checkpoint above
```

Training and checkpoint inference share architecture options in
`src/hivemind/architectures/model_config.py`; ONNX inference helpers live in
`src/hivemind/inference/onnx.py`. Network provenance and default paths live in
`src/hivemind/network.py`. To run the Python regression suite after installing dependencies:

```bash
uv run python -m pytest
```

## Building the Engine

```bash
uv run hivemind build-engine
```

This configures the fast Ninja preset and builds the C++ engine at
`engine/build-ninja/hivemind`. Use `--preset ninja-release` for a fully
optimized build, or run `uv run hivemind build-engine --help` for backend and
dependency-path options.

The equivalent manual build is:

```bash
cd engine
mkdir build && cd build
cmake ..
make -j$(nproc)
```

For CPU and NVIDIA GPU builds, plus ready-to-distribute Windows/Linux ZIPs,
see [the engine build and release guide](engine/README.md).

## Installation (Python)

Using [uv](https://github.com/astral-sh/uv):

```bash
uv sync
uv run hivemind --help
```

`uv sync` installs the Python package and its `hivemind` command.
`python -m hivemind` provides the same interface in an activated environment.

## Usage

### Running the Engine

```bash
./engine/build-ninja/hivemind \
  --model "$(realpath artifacts/training/weights/rl/model-rl-final-v3.0.onnx)"
```

The engine communicates via UCI protocol. Use with any UCI-compatible chess GUI.
Passing an explicit model path via `--model` (or `--network`) makes startup independent
of the current working directory. Without explicit path, the engine searches `./models`,
`./engine/models`, and legacy `./networks` for the latest ONNX model.

### Engine Commands

```bash
# Run inference benchmark (add --batch-size to compare batch throughput)
./hivemind bench
./hivemind bench 1000 --batch-size 64

# Run move generation benchmark
./hivemind perft 5

# Run RISE self-play with training defaults (800 MCTS nodes, 100k
# Fairy-Stockfish mate nodes per position); extra flags pass through to the
# engine. For twin-s self-play, see Training.
uv run hivemind rise-selfplay --games 1000
uv run hivemind rise-selfplay --games 1000 --nodes 400 --mate-nodes 0 --seed 7

# Or call the engine directly
./engine/build-ninja/hivemind selfplay \
  --model artifacts/training/weights/rl/model-rl-final-v3.0.onnx \
  --games 1000 --nodes 400 --output engine/selfplay_games
```

Self-play also runs the bounded Fairy-Stockfish mate search for each searched
position. It receives a fixed total allowance of 10,000,000 nodes across the two
boards by default and is allowed to finish after the MCGS node budget is reached,
so mate-aware generation trades some throughput for stronger tactical play.
Adjust the node allowance, or disable it, with:

```bash
./engine/build-ninja/hivemind selfplay \
  --fairy-stockfish-mate-nodes 4000000 ...

./engine/build-ninja/hivemind selfplay \
  --fairy-stockfish-mate-nodes 0 ...
```

### Inference Batch Size

The number of leaves gathered per neural network evaluation defaults to
`SearchParams::BATCH_SIZE` and can be overridden without recompiling:

```bash
# UCI mode, set at startup
./engine/build-ninja/hivemind --model models/bughouse-rise-v3.onnx --batch-size 32

# UCI mode, set at runtime (reloads the network)
setoption name BatchSize value 32

# Self-play
./engine/build-ninja/hivemind selfplay --batch-size 32 ...

# Head-to-head, one batch size per side
./engine/build-ninja/hivemind tournament \
  --contender models/bughouse-rise-v3.onnx --baseline models/bughouse-rise-v3.onnx \
  --contender-batch-size 32 --baseline-batch-size 8 \
  --games 200 --nodes 800 --output engine/batch_ab
```

Each batch size needs its own TensorRT engine. A cached one loads in about a
second; the first use of a new size builds it from the ONNX, which takes a few
minutes. Larger batches raise throughput but coarsen search: more leaves are
selected under virtual loss before any of them is evaluated, so the trade-off is
worth confirming with a paired tournament rather than by nodes per second alone.

Self-play diversifies each opening with raw-policy initialization. Its length is
sampled from an exponential distribution with a mean of 8 macro plies and a
maximum of 30; these positions are recorded in PGN but excluded from the
training chunks.
Subsequent actions sample MCTS visits with temperature 0.8, decayed by 0.93
every two macro plies. Search budgets are randomized by ±5% per position.

### Team and Time Advantage

Two options tell the engine which half of the four players it is playing, and
whether that team is ahead on the clocks:

```bash
# Our team plays White on board A and Black on board B (the default)
setoption name Team value white

# Our team is ahead on the clocks, so it may sit and double-sit
setoption name TimeAdvantage value true
```

`TimeAdvantage` gates the bughouse waiting rules: a team that is up on time may
pass on a board it is on turn for, and may pass on both. Without it, passing on
an on-turn board is only legal when the partner board captures. The flag also
feeds the network as an input plane, so it changes the evaluation as well as the
legal joint actions — set it to match the real clocks before searching.

> `setoption name Mode value go|sit` is the deprecated spelling of the same
> setting (`sit` = `TimeAdvantage true`). It is still accepted but no longer
> advertised.

#### Clock model

`TimeAdvantage` is one bit fixed for the whole search. The clock model searches
on the lead itself:

```text
setoption name TimeModel value clock
setoption name TimeLeadMs value 4200        # our lead over the diagonal opponents
setoption name TeamMoveTimeMs value 1000     # how long our team takes per action
setoption name OpponentMoveTimeMs value 1500
```

Without increment a team's lead is the same on both diagonals (White on A minus
White on B equals Black on B minus Black on A), and the other team's is its
negation. It moves only while one team is on turn on both boards: each action
that team plays costs it its own move time. The search charges every path that
way, so a team may sit only while its lead lasts, and at a lead of exactly zero
neither team may. Both move times default to 100 ms, chess.com's fastest move.
`TimeModel flag` (the default) keeps the `TimeAdvantage` behaviour.

Networks read the clock through two planes at the end of each board's block
(see [Input Representation](#input-representation)): the lead, counted in the
moves the leading team can afford to wait, and the ratio of the two teams'
move times. Networks from before those planes read 74 planes and still load;
the engine drops the clock planes before running them. Those networks were
trained with one team sitting for the whole game, so they misjudge level
clocks until retrained. Training from one widens its first layer with zero
weights for the two new planes (`hivemind.architectures.twin_board.
widen_to_clock_planes`), so it starts out computing exactly what it did, and
training data from before the planes is widened as it is read: the lead plane
is +1 or -1 by the sit right, the speed plane 0.

### Opening diversity

Normal UCI play is deterministic by default. To vary early play without the
coverage and maintenance cost of a Bughouse opening book, enable Dirichlet
noise at the search root:

```text
setoption name OpeningNoise value true
setoption name OpeningNoisePlies value 16
setoption name OpeningNoiseAlphaPermille value 100
setoption name OpeningNoiseEpsilonPermille value 600
```

`OpeningNoisePlies` is measured in ordinary game plies on each board. Noise is
used only while the larger of the two board ply counts is below the limit, so a
setting of 16 covers roughly the first eight moves per board. Alpha controls
the shape of the sampled alternatives; epsilon controls how much of that sample
is mixed into the network policy (600 means 60%). Each noisy search uses a fresh
sample and starts a fresh root, while later positions retain the normal
deterministic search and tree reuse. Set `OpeningNoise` to `false` (the default),
`OpeningNoisePlies` to 0, or epsilon to 0 to disable it.

### Reports While Searching

By default a search prints its rank-1 line (`info depth … pv …`) whenever the
search reaches a new depth, which stops after the first few thousand nodes.
Set `ReportIntervalMs` to print the whole MultiPV block, in the same format as
the final block, every interval instead. It applies to `go nodes`,
`go movetime`, timed and `go infinite` searches of the bughouse MCTS search;
single-board variants and `SearchMode alphabeta` keep their usual output.
Periodic blocks leave out root moves no visit has reached yet, except rank 1,
which is always the move `bestmove` would play at that moment.

```text
# Print every ranked line every 250 ms; 0 (the default) prints only the rank-1 line at each new depth
setoption name ReportIntervalMs value 250
```

A GUI can tell a build that can report every ranked line while it searches by
the `option name ReportIntervalMs` line in its `uci` reply.

### Paired Model Tournament

```bash
./engine/build-ninja/hivemind tournament \
  --contender artifacts/training/weights/rl/model-rl-final-v3.0.onnx \
  --baseline engine/models/model-rl-final-v3.0.onnx \
  --games 100 --nodes 800 --output engine/tournament_results --seed 1
```

Tournament games are paired, so `--games` must be even. Each model controls
the White team in one game and the Black team in the other, with the starting
team alternated between pairs. Both games in a pair use the same seeded root
noise schedule, and moves are selected by maximum root visits. The command
writes complete games to `games.pgn` and incremental W/D/L, score, and Elo
results to `summary.json`. Set `--dirichlet-epsilon 0` for deterministic games.

### Training

The default network, twin-s, improves by reinforcement learning from its own
games. One command runs the whole loop, forever:

```bash
uv run hivemind rl-loop
```

Each iteration plays 100,000 self-play games and 1,000 validation games with
the current generator, trains on them plus a replay of the previous three
iterations' games, and plays 1,000 games at 100 ms per move against the
generator, and as many against the fixed first network (it0). The new network
replaces the generator unless it scores below 50% against it.
Every iteration's network and training checkpoint are uploaded to Hugging Face
as `twin-s-noattn-itN.onnx` and `twin-s-noattn-itN.pt`, and a promoted one as
`twin-s-noattn.onnx` and `twin-s-noattn.pt`, when `hf auth login` has logged
this machine in to an account that can write to the repository (`--repo`);
otherwise they stay local. Progress goes to
`data/distill/runs/rl-loop.log`. Every finished stage is recorded under
`data/distill/runs/rl-itN/stages`, so rerunning the command resumes where it
stopped; `touch data/distill/runs/rl-loop.stop` stops it between stages.
Matches open from the 100 positions in `engine/openings.tsv`.
`hivemind fetch-network` keeps installing the network and checkpoint pinned
in `src/hivemind/network.py`; update that pin to publish a promotion.

#### Training on the clock

`--time-model clock` trains a network that reads the clock planes: self-play
plays every game on a simulated clock (`selfplay --time-model clock`: half the
games start level, the rest with a lead of 0.1-10 s either way, and each team
moves in 0.1-3 s), and matches run on one too. Such a network cannot replace
the flag model's generator, which engines from before the clock planes still
load, so it trains as a lineage of its own, from a copy of the generator it
starts from:

```bash
cp engine/models/twin-s-noattn.onnx engine/models/twin-s-clock.onnx
cp engine/models/twin-s-noattn.onnx engine/models/twin-s-clock-it0.onnx
mkdir -p artifacts/distill/twin-s-clock
cp data/distill/runs/rl-it10/arms/C/last.pt artifacts/distill/twin-s-clock/last.pt
uv run hivemind rl-loop --time-model clock --runs data/distill/runs-clock \
    --generator engine/models/twin-s-clock.onnx \
    --generator-checkpoint artifacts/distill/twin-s-clock/last.pt \
    --selfplay-engine engine/build-clock-sp1/hivemind --match-engine engine/build-clock/hivemind
```

The first iteration widens the checkpoint's input layer with zero weights for
the clock planes, so it starts out playing exactly as its parent. The engines
must be builds with the clock model: one search worker for self-play
(`-DHIVEMIND_SEARCH_WORKERS=1`) and the usual four for matches. Uploads name
the lineage after its generator (`twin-s-clock-itN.onnx`), and contributors to
a `--distributed` clock loop play clock games too (work-order protocol 2).

#### Keeping data off the disk

An iteration's games take about 9 GB on disk. The loop moves each finished
segment to the private Hugging Face dataset
`aminwoo/bughouse-twin-s-selfplay-archive` (`--archive`) as a zstd tarball
about 13x smaller. It deletes the games only once the Hub's copy matches.
Training downloads the games it needs, about 3 GB for the new iteration and
the replay window, and deletes them again when it finishes. Use
`--keep-selfplay` to keep everything on disk; without uploads (`--no-upload`,
or no `hf auth login`) the loop keeps everything on disk anyway.
`hivemind rl-loop --archive-only` moves all finished self-play and exits.

`archive` and `restore` do the same for any data directory:

```bash
uv run hivemind archive data/nnue/train   # leaves data/nnue/train/archive.json
uv run hivemind restore data/nnue/train   # brings the files back
```

`hivemind train` restores the directories it is given by itself; other
training commands need a `restore` first.

#### Sharing self-play with other machines

Self-play is most of each iteration, and any number of machines can share it.
Start the loop with `--distributed`, and run `contribute` on every other
machine:

```bash
uv run hivemind rl-loop --distributed   # the training machine
uv run hivemind contribute              # each contributing machine
```

The loop publishes a work order in the Hugging Face dataset
`aminwoo/bughouse-twin-s-selfplay` (created private; `--exchange` names
another, or a directory every machine can reach). A contributor downloads the
generator the order names, plays 1,000-game batches with the loop's settings,
and submits each as a pull request, so any logged-in Hugging Face account can
contribute (`hf auth login`). The loop collects and merges the batches,
keeps playing 1,000-game segments itself (unless `--no-local-selfplay`), and
trains once the iteration has its games. Batches that arrive later still
count, as replay data for later iterations; while the loop trains and plays
its matches, contributors wait for the next order. Contributors need an
engine built with one search worker per game, as `hivemind contribute --help`
shows; on macOS that is the ONNX Runtime backend, and the contributor runs
the generator through Core ML. Each batch adds about 85 MB to the dataset.

`selfplay` and `train` run the stages of one iteration by hand:

```bash
# 100,000 games in 10,000-game segments (seed 1000 * iteration + segment),
# then 1,000 validation games (seed 1000 * iteration + 999)
uv run hivemind selfplay --seed 7000 --output data/distill/rl/it7/seg00
uv run hivemind selfplay --seed 7001 --output data/distill/rl/it7/seg01
...
uv run hivemind selfplay --games 1000 --seed 7999 --output data/distill/rl/it7/val

# Continue the generator's checkpoint on the new games, replaying the
# previous three iterations'
uv run hivemind train --seed 7 \
  --data data/distill/rl/it7/seg* --val data/distill/rl/it7/val \
  --replay data/distill/rl/it{4,5,6}/seg* --out artifacts/distill/rl/it7
```

Self-play searches 800 nodes ±5% per move with no mate search and no
resignation, and plays 6 games at once on an engine built with one search
worker per game:

```bash
cmake -S engine -B engine/build-sp1 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DHIVEMIND_SEARCH_WORKERS=1
cmake --build engine/build-sp1 --target hivemind
```

It loads the network from a `-sp1` copy next to it, so this engine and the UCI
engine keep separate TensorRT caches. Each output directory receives HDST
chunks of the search targets, with the PGN in its `games` subdirectory.
Training continues `artifacts/distill/twin-s-noattn/best.pt` unless given
`--init`, downloading the published network's checkpoint there when it is
missing (`uv run hivemind fetch-network --variant checkpoint` does the same); it learns policy from root visits and WDL and moves-left from game
results, with no teacher data, at lr 1e-4 over 1.5 passes of the new games
and 0.5 of each replayed one, and exports the last weights as `last.pt` and
`distill-twin-s-v3.0.onnx`.

The RISE architectures, including the crossboard teacher twin-s was distilled
from, train with `rise-train`:

```bash
# Supervised learning on human games
uv run hivemind rise-train --mode sl

# Train the explicit cross-board coordination architecture from scratch
uv run hivemind rise-train --mode sl \
  --architecture crossboard-risev33

# Train the staged dual-stream architecture with persistent latent memory
uv run hivemind rise-train --mode sl \
  --architecture dualstream-memory-risev33

# Generate an isolated >=2250 corpus and train cross-board RISEv3 on it
uv run hivemind prepare \
  --games data/games.parquet \
  --min-rating 2250 \
  --train-planes-dir data/planes/sl_2250/train \
  --val-shard data/planes/sl_2250/val/evaluation_shard.parquet \
  --train-eval-shard data/planes/sl_2250/train_eval/evaluation_shard.parquet \
  --architecture crossboard-risev33 \
  --batch-size 256

# RL training directly from native HVM5 self-play data (from before the clock
# planes: self-play now writes HVM6 chunks with 78 planes, which these RISE
# commands do not read)
uv run hivemind rise-train --mode rl --checkpoint artifacts/training/weights/rl/model-rl-final.tar --selfplay-dir engine/selfplay_games/iteration-2/training_data --architecture crossboard-risev33
```

RL training reads `engine/selfplay_games/training_data` by default, creates a
deterministic game-level 98/2 train/validation split under
`engine/selfplay_games/rl_data`, and then starts training. Original HVM chunks
are preserved. Use `--selfplay-dir` for a different self-play directory, or
provide both `--rl-data-dir` and `--val-data-dir` to train from existing Parquet
data. Supervised artifacts are written under `artifacts/training/weights/supervised`;
RL artifacts, including resumable `model-rl-final.tar` and deployable
`model-rl-final-v3.0.onnx`, are written under `artifacts/training/weights/rl`.
For later RL iterations, pass
`--checkpoint artifacts/training/weights/rl/model-rl-final.tar`.
Cross-board and dual-stream checkpoints must be continued with their original
`--architecture`; legacy RISEv3 checkpoints are not shape compatible with the
new attention and policy heads.

```bash
# Train on iteration 3 with CrazyAra-style replay from iteration 2
uv run hivemind rise-train --mode rl \
  --checkpoint artifacts/training/weights/rl/model-rl-final.tar \
  --selfplay-dir engine/selfplay_games/iteration-3 \
  --replay-dir engine/selfplay_games/iteration-2 \
  --architecture crossboard-risev33
```

HVM5 stores the sparse joint root-visit distribution in addition to both
marginal policies. The joint compatibility residual is optional; the default RL
configuration disables it and trains only the two marginal policy heads. Older
HVM3/HVM4 chunks therefore remain valid training inputs.

With `--replay-dir`, RL preparation adds five archived HVM chunks selected
deterministically from the newest 5% of that directory, matching CrazyAra's
replay-memory defaults. Replay games are training-only; validation is made only
from the current iteration. Adjust this with `--replay-files`,
`--replay-selection-fraction`, and `--split-seed`.

RL sample shuffling is bounded to one decoded Parquet shard at a time so dense
policy tensors from completed shards can be released. Generated RL Parquet
shards contain 4,096 samples by default. On hosts with limited RAM, use
`--shuffle-buffer-size 1000` to reduce the default 10,000-sample within-shard
buffer. RL training saves resumable `.tar` checkpoints when validation loss
improves but defers ONNX conversion until training completes. Resume an
interrupted run with its latest intermediate checkpoint and `--resume-training`.

The end-to-end supervised script filters all four players, performs a
deterministic whole-game 98/2 train/validation split, doubles samples by board
swap, builds a fixed training-metrics shard, and then launches training. Using
the `sl_2250` paths above preserves the existing default corpus.

## Neural Network Architecture

Hivemind uses **RISEv3** (Residual Inverted Squeeze-Excitation), a mobile-optimized architecture combining:

- Mixed depthwise convolutions
- Squeeze-and-excitation blocks
- Pre-activation residual connections

The optional `crossboard-risev33` architecture retains the RISEv3 convolutional
tower and adds explicit post-tower coordination:

- 64 spatial tokens for board A and 64 for board B
- Four pocket tokens covering both teams on both boards
- Two board-local side-to-move tokens and one shared time-advantage token
- Two bidirectional cross-attention layers
- Independent spatial policy heads for boards A and B

The coordinated board maps are fused only for the value, WDL, and moves-left
heads. The deployable ONNX interface remains `value`, `pi_a`, `pi_b`, `wdl_out`,
and `moves_left`.

The optional `dualstream-memory-risev33` architecture instead runs each board
through the same stem and three shared-weight five-block stages. Two
intermediate communication stages update a persistent eight-token latent
workspace, apply one latent self-attention block, and symmetrically feed the
result back to both boards. A final direct square-to-square attention layer
preserves exact tactical communication. State-dependent residual gates control
the latent and direct pathways independently, and only the value-side heads
fuse the final board maps.

Cross-board and dual-stream training both default to batch size 256. The
dual-stream model still processes two sets of trunk activations, so reduce this
value explicitly if GPU memory is insufficient.
Use `--batch-size` to tune this for another GPU; the legacy RISEv3 default
remains unchanged. Dual-stream training uses BF16 model execution by default on
CUDA while retaining FP32 parameters, losses, optimizer state, and checkpoints.
Use `--precision fp32` for an exact full-precision fallback. During supervised training,
intermediate checks process at most 64 batches from each metrics loader and run
every 2,048,000 training samples. Complete train/validation evaluation runs at
each epoch boundary and at the end of training.

### Input Representation

The network uses a **78-channel input** (78×8×8), with 39 channels per board:

| Per-board channels | Description                             |
| ------------------ | --------------------------------------- |
| 0-11               | Piece positions (own and opponent)      |
| 12-21              | Pocket piece counts                     |
| 22-23              | Promoted pieces and en passant          |
| 24-26              | Perspective, side to move, and constant |
| 27-30              | Castling rights                         |
| 31                 | Time advantage                          |
| 32-33              | Last move source and destination        |
| 34                 | Halfmove clock                          |
| 35-36              | Twofold and threefold repetition        |
| 37                 | Time lead: sign(lead) * min(1, log2(1 + moves the leader can wait) / 10) |
| 38                 | Speed: log2(own move time / opponents' move time) / 5, clamped to ±1 |

Both boards carry the team's same two clock values. Under `TimeModel flag` the
lead plane is +1 or -1 by the sit right and the speed plane 0. Networks from
before the clock planes (RISE, and twin-s up to iteration 10) read channels
0-36 of each board, 74 in all.

### Output

- **Policy heads**: 4672 move probabilities for each board
- **Value head**: Scalar outcome prediction
- **WDL head**: Win/draw/loss classification
- **Moves-left head**: Remaining team-decision estimate

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Fairy-Stockfish](https://github.com/fairy-stockfish/Fairy-Stockfish) for move generation
- [CrazyAra](https://github.com/QueensGambit/CrazyAra) for architecture inspiration
