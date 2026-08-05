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
│   ├── src/             # Engine source code
│   │   ├── Fairy-Stockfish/  # Move generation library
│   │   └── rl/          # Self-play and training data generation
│   └── networks/        # ONNX model files
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
- CUDA Toolkit 13.0+
- TensorRT 10.14+

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

## Installation (Python)

Using [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

## Usage

### Running the Engine

```bash
./engine/build-ninja/hivemind \
  --network "$(realpath src/training/weights/rl/model-rl-final-v3.0.onnx)"
```

The engine communicates via UCI protocol. Use with any UCI-compatible chess GUI.
Passing an explicit network path makes startup independent of the current working
directory. Without `--network`, the engine retains the legacy behavior of loading
the most recently modified ONNX model from `./networks`.

### Engine Commands

```bash
# Run inference benchmark
./hivemind bench

# Run move generation benchmark
./hivemind perft 5

# Run self-play for training data generation
./engine/build-ninja/hivemind selfplay \
  --network src/training/weights/rl/model-rl-final-v3.0.onnx \
  --games 1000 --nodes 400 --output engine/selfplay_games
```

Self-play diversifies each opening with raw-policy initialization. Its length is
sampled from an exponential distribution with a mean of 8 macro plies and a
maximum of 30; these positions are recorded in PGN but excluded from HVM3.
Subsequent actions sample MCTS visits with temperature 0.8, decayed by 0.93
every two macro plies. Search budgets are randomized by ±5% per position.

### Network Tournament

```bash
./engine/build-ninja/hivemind tournament \
  --contender src/training/weights/rl/model-rl-final-v3.0.onnx \
  --baseline src/training/weights/supervised/model-0.89016-0.702-0136-v3.0.onnx \
  --games 100 --nodes 800 --output engine/tournament_results --seed 1
```

Tournament games are paired, so `--games` must be even. Each network controls
the White team in one game and the Black team in the other, with the starting
team alternated between pairs. Both games in a pair use the same seeded root
noise schedule, and moves are selected by maximum root visits. The command
writes complete games to `games.pgn` and incremental W/D/L, score, and Elo
results to `summary.json`. Set `--dirichlet-epsilon 0` for deterministic games.

### Training

```bash
# Supervised learning on human games
uv run python src/training/train_loop.py --mode sl

# RL training directly from native HVM3 self-play data
uv run python src/training/train_loop.py --mode rl \
  --checkpoint src/training/weights/supervised/model-0.89016-0.702-0136.tar
```

RL training reads `engine/selfplay_games/training_data` by default, creates a
deterministic game-level 98/2 train/validation split under
`engine/selfplay_games/rl_data`, and then starts training. Original HVM3 chunks
are preserved. Use `--selfplay-dir` for a different self-play directory, or
provide both `--rl-data-dir` and `--val-data-dir` to train from existing Parquet
data. Supervised artifacts are written under `src/training/weights/supervised`;
RL artifacts, including resumable `model-rl-final.tar` and deployable
`model-rl-final-v3.0.onnx`, are written under `src/training/weights/rl`.
For later RL iterations, pass
`--checkpoint src/training/weights/rl/model-rl-final.tar`.

## Neural Network Architecture

Hivemind uses **RISEv3** (Residual Inverted Squeeze-Excitation), a mobile-optimized architecture combining:

- Mixed depthwise convolutions
- Squeeze-and-excitation blocks
- Pre-activation residual connections

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
