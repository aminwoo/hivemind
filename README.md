<div align="center">
  
  ![hivemind-logo](https://github.com/aminwoo/hivemind/assets/124148472/d42c6a6e-ab2e-4d7a-bf90-4876d59c9558)
  
  # HiveMind

  A free and strong UCI Bughouse chess engine powered by deep reinforcement learning.

  [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

## Overview

HiveMind is a neural network-based engine for [Bughouse chess](https://en.wikipedia.org/wiki/Bughouse_chess), a four-player chess variant played on two boards. The engine uses Monte Carlo Tree Search (MCTS) with a deep neural network for position evaluation and move prediction.

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
./engine/build/hivemind
```

The engine communicates via UCI protocol. Use with any UCI-compatible chess GUI.

### Engine Commands

```bash
# Run inference benchmark
./hivemind bench

# Run move generation benchmark  
./hivemind perft 5

# Run self-play for training data generation
./hivemind selfplay 1000

# Evaluate two models against each other
./hivemind eval --new model_a.onnx --old model_b.onnx --games 100
```

### Training

```bash
# Supervised learning on human games
uv run python src/training/train_loop.py --mode supervised

# RL training on self-play data
uv run python src/training/train_loop.py --mode rl \
  --rl-data-dir engine/selfplay_games/training_data_parquet
```

### Convert Self-Play Data

```bash
uv run python src/preprocessing/convert_selfplay_data.py \
  engine/selfplay_games/training_data \
  engine/selfplay_games/training_data_parquet
```

## Neural Network Architecture

HiveMind uses **RISEv3** (Residual Inverted Squeeze-Excitation), a mobile-optimized architecture combining:
- Mixed depthwise convolutions
- Squeeze-and-excitation blocks
- Pre-activation residual connections

### Input Representation

The network uses a **64-channel input** (64×8×8) encoding both Bughouse boards:

| Channels | Description |
|----------|-------------|
| 0-11, 32-43 | Piece positions (own and opponent) |
| 12-21, 44-53 | Pocket pieces available for drops |
| 22-23, 54-55 | Promoted piece masks |
| 24, 56 | En passant squares |
| 25, 57 | Side to move |
| 26, 58 | Constant plane (all 1s) |
| 27-30, 59-62 | Castling rights |
| 31, 63 | Time advantage indicator |

### Output

- **Policy head**: 4672 move probabilities per board (73 planes × 8 × 8)
- **Value head**: Win/Draw/Loss prediction for the position

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Fairy-Stockfish](https://github.com/fairy-stockfish/Fairy-Stockfish) for move generation
- [CrazyAra](https://github.com/QueensGambit/CrazyAra) for architecture inspiration
