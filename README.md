<div align="center">
  
  ![hivemind-logo](https://github.com/aminwoo/hivemind/assets/124148472/d42c6a6e-ab2e-4d7a-bf90-4876d59c9558)
  
  <h3>HiveMind</h3>

<<<<<<< HEAD
  A free & strong UCI Bughouse engine.
=======
A free & strong UCI Bughouse engine.
>>>>>>> feat/multi-pv

</div>

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda-repo-ubuntu2404-13-0-local_13.0.2-580.95.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-13-0-local_13.0.2-580.95.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-0

wget https://developer.download.nvidia.com/compute/tensorrt/10.14.1/local_installers/nv-tensorrt-local-repo-ubuntu2404-10.14.1-cuda-13.0_1.0-1_amd64.deb
os="ubuntu2404"
tag="10.14.1-cuda-13.0"
sudo dpkg -i nv-tensorrt-local-repo-${os}-${tag}_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-${os}-${tag}/*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install tensorrt
```

```
$$M = k \cdot N^{\alpha}$$$k$ (Widening Constant): Set this between $1.0$ and $2.0$.$\alpha$ (Widening Factor): Set this around $0.5$.
```

```
trtexec \
    --onnx=model-2.40025-0.604-0001-v3.0.onnx \
    --fp16 \
    --shapes=obs:64x64x8x8 \
    --warmUp=500 \
    --duration=10 \
    --avgTiming=100 \
    --dumpProfile
```

# Bughouse Chess Input Representation
<<<<<<< HEAD
The neural network uses a **64-channel input representation** (64×8×8) to encode the complete game state for bughouse chess.
## Channel Layout
The input is organized as two 32-channel blocks representing each bughouse board:
=======

The neural network uses a **64-channel input representation** (64×8×8) to encode the complete game state for bughouse chess.

## Channel Layout

The input is organized as two 32-channel blocks representing each bughouse board:

>>>>>>> feat/multi-pv
- **Channels 0-31**: Board A
- **Channels 32-63**: Board B

All channels are from the current team's perspective with appropriate board orientations.
<<<<<<< HEAD
## Channel Breakdown

| Channels | Description |
| --- | --- |
| **0-11, 32-43** | **Piece Positions** - Own pieces (0-5) and opponent pieces (6-11) for each piece type: Pawn, Knight, Bishop, Rook, Queen, King |
| **12-21, 44-53** | **Pocket Pieces** - Available drops for own team (12-16) and opponent (17-21): Pawn, Knight, Bishop, Rook, Queen (normalized by max drops) |
| **22-23, 54-55** | **Promoted Pieces** - Binary mask for promoted pieces (own team, opponent team) |
| **24, 56** | **En Passant** - En passant target squares |
| **25, 57** | **Turn** - 1.0 if it's the team's turn on this board |
| **26, 58** | **Constant** - All 1.0 (reference plane) |
| **27-30, 59-62** | **Castling Rights** - Kingside/queenside castling availability for both teams |
| **31, 63** | **Time Advantage** - 1.0 if team has time advantage (can "sit") |
=======

## Channel Breakdown

| Channels         | Description                                                                                                                                 |
| ---------------- |---------------------------------------------------------------------------------------------------------------------------------------------|
| **0-11, 32-43**  | **Piece Positions** - Own pieces (0-5) and opponent pieces (6-11) for each piece type: Pawn, Knight, Bishop, Rook, Queen, King              |
| **12-21, 44-53** | **Pocket Pieces** - AvailabPle drops for own team (12-16) and opponent (17-21): Pawn, Knight, Bishop, Rook, Queen (normalized by max drops) |
| **22-23, 54-55** | **Promoted Pieces** - Binary mask for promoted pieces (own team, opponent team)                                                             |
| **24, 56**       | **En Passant** - En passant target squares                                                                                                  |
| **25, 57**       | **Turn** - 1.0 if it's the team's turn on this board                                                                                        |
| **26, 58**       | **Constant** - All 1.0 (reference plane)                                                                                                    |
| **27-30, 59-62** | **Castling Rights** - Kingside/queenside castling availability for both teams                                                               |
| **31, 63**       | **Time Advantage** - 1.0 if team has time advantage (can "sit")                                                                             |

uv run src/preprocessing/convert_selfplay_data.py \
 engine/selfplay_games/training_data \
 engine/selfplay_games/training_data_parquet

cd src/training
python train_loop.py --mode rl \
 --rl-data-dir ../../engine/selfplay_games/training_data_parquet \

python train_loop.py --mode rl \
 --rl-data-dir ../../engine/selfplay_games/training_data_parquet \
 --checkpoint path/to/model.tar

uv run src/training/train_loop.py --mode rl \
 --rl-data-dir engine/selfplay_games/training_data_parquet \
 --checkpoint src/training/weights/model-0.97878-0.683-0224.tar
>>>>>>> feat/multi-pv
