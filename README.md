<div align="center">
  
  ![hivemind-logo](https://github.com/aminwoo/hivemind/assets/124148472/d42c6a6e-ab2e-4d7a-bf90-4876d59c9558)
  
  # Hivemind

A free and strong UCI Bughouse chess engine powered by deep reinforcement learning.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

## Overview

Hivemind is a neural network-based engine for [Bughouse chess](https://en.wikipedia.org/wiki/Bughouse_chess), a four-player chess variant played on two boards. The engine uses Monte Carlo Tree Search (MCTS) with a deep neural network for position evaluation and move prediction.

### Key Features

- **Neural Network Policy & Value Estimation** - Uses the RISEv3 architecture for move prediction and position evaluation
- **Monte Carlo Graph Search (MCGS)** - Shares nodes across transpositions for improved search efficiency
- **TensorRT Acceleration** - High-performance GPU inference using NVIDIA TensorRT
- **UCI Protocol** - Standard Universal Chess Interface for GUI compatibility
- **Self-Play Training** - RL training pipeline with self-play game generation

## Project Structure

```
hivemind/
├── engine/              # C++ UCI engine
│   ├── src/             # Modular engine source code
│   │   ├── common/      # Globals, logging, utilities
│   │   ├── environment/ # Bughouse board, joint actions, planes, zobrist
│   │   ├── nn/          # TensorRT inference engine, ONNX utilities
│   │   ├── search/      # MCTS/MCGS search, agent, nodes, transposition table
│   │   ├── interface/   # UCI protocol interface
│   │   ├── tools/       # Selfplay, tournament, benchmarks
│   │   └── Fairy-Stockfish/  # Move generation library
│   └── models/          # ONNX and TensorRT engine model files
├── src/                 # Python training code
│   ├── architectures/   # Neural network architectures (RISEv3)
│   ├── domain/          # Board representation and move encoding
│   ├── preprocessing/   # Data preprocessing utilities
│   ├── training/        # Training loop and data loaders
│   └── utils/           # Utility functions
├── scripts/             # Utility scripts
│   ├── analyze_training_data.py  # Inspect training samples
│   ├── evaluate_model.py         # Evaluate model on data
│   ├── infer_from_fen.py         # Run inference on positions
│   └── search_training_fen.py    # Search for positions in data
├── tests/               # Test suite
├── configs/             # Configuration files
└── data/                # Training data and game archives
```

## Requirements

### Engine (C++)

- CMake 3.16+
- C++23 compatible compiler
- Windows or Linux with ONNX Runtime (portable CPU build), or
- Linux with CUDA Toolkit 13.0+ and TensorRT 10.14+ (NVIDIA GPU build)

### Training (Python)

- Python 3.13+
- PyTorch 2.9+
- See `pyproject.toml` for full dependencies

## Building the Engine

```bash
cd engine
mkdir build && cd build
cmake ..
make -j$(nproc)
```

For a Windows-compatible CPU build or ready-to-distribute Windows/Linux zip,
see [the engine build and release guide](engine/README.md).

## Installation (Python)

Using [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

## Usage

### Running the Engine

```bash
./engine/build-ninja/hivemind \
  --model "$(realpath src/training/weights/rl/model-rl-final-v3.0.onnx)"
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

# Run self-play for training data generation
./engine/build-ninja/hivemind selfplay \
  --model src/training/weights/rl/model-rl-final-v3.0.onnx \
  --games 1000 --nodes 400 --output engine/selfplay_games
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
maximum of 30; these positions are recorded in PGN but excluded from HVM5.
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

### Paired Model Tournament

```bash
./engine/build-ninja/hivemind tournament \
  --contender src/training/weights/rl/model-rl-final-v3.0.onnx \
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

```bash
# Supervised learning on human games
uv run python src/training/train_loop.py --mode sl

# Train the explicit cross-board coordination architecture from scratch
uv run python src/training/train_loop.py --mode sl \
  --architecture crossboard-risev33

# Train the staged dual-stream architecture with persistent latent memory
uv run python src/training/train_loop.py --mode sl \
  --architecture dualstream-memory-risev33

# Generate an isolated >=2250 corpus and train cross-board RISEv3 on it
uv run python scripts/train_from_games_parquet.py \
  --games data/games.parquet \
  --min-rating 2250 \
  --train-planes-dir data/planes/sl_2250/train \
  --val-shard data/planes/sl_2250/val/evaluation_shard.parquet \
  --train-eval-shard data/planes/sl_2250/train_eval/evaluation_shard.parquet \
  --architecture crossboard-risev33 \
  --batch-size 256

# RL training directly from native HVM5 self-play data
uv run python src/training/train_loop.py --mode rl --checkpoint /home/ben/hivemind/src/training/weights/rl/model-rl-final.tar --selfplay-dir /home/ben/hivemind/engine/selfplay_games/iteration-2/training_data --architecture crossboard-risev33
```

RL training reads `engine/selfplay_games/training_data` by default, creates a
deterministic game-level 98/2 train/validation split under
`engine/selfplay_games/rl_data`, and then starts training. Original HVM chunks
are preserved. Use `--selfplay-dir` for a different self-play directory, or
provide both `--rl-data-dir` and `--val-data-dir` to train from existing Parquet
data. Supervised artifacts are written under `src/training/weights/supervised`;
RL artifacts, including resumable `model-rl-final.tar` and deployable
`model-rl-final-v3.0.onnx`, are written under `src/training/weights/rl`.
For later RL iterations, pass
`--checkpoint src/training/weights/rl/model-rl-final.tar`.
Cross-board and dual-stream checkpoints must be continued with their original
`--architecture`; legacy RISEv3 checkpoints are not shape compatible with the
new attention and policy heads.

```bash
# Train on iteration 3 with CrazyAra-style replay from iteration 2
uv run python src/training/train_loop.py --mode rl \
  --checkpoint src/training/weights/rl/model-rl-final.tar \
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

The network uses a **74-channel input** (74×8×8), with 37 channels per board:

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
