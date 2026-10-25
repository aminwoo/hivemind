# Training and network reference

Start with the [repository setup](../README.md#build-from-a-clone) and
[one-worker build instructions](../README.md#contribute-self-play-games).
`uv sync` installs Python dependencies; it does not compile either C++ engine.
Commands below run from the repository root. The RL loop uses POSIX locking
and process control, so these loop instructions are for Linux/macOS. NVIDIA GPU training is the main training workflow;
Apple silicon contribution uses Core ML for inference.

These commands create large datasets and can run indefinitely. Use
`hivemind COMMAND --help` for current flags and `--last N` to bound a loop.
An ONNX release network is for inference; continuing training also requires
its matching PyTorch checkpoint. Defaults and pinned artifacts are defined in
[`network.py`](../src/hivemind/network.py).

## Reinforcement learning

The Python training defaults still use the older twin-s-noattn flag lineage.
Release bundles use twin-s-clock; downloading that ONNX file does not provide
a matching training checkpoint. After building the one-worker self-play engine
and the match engine, one command runs the default loop until stopped:

```bash
uv run hivemind fetch-network --variant onnx
uv run hivemind fetch-network --variant checkpoint
uv run hivemind rl-loop
```

Each iteration plays 100,000 self-play games and 1,000 validation games with
the current generator, trains on them plus a replay of the previous three
iterations' games, and plays 1,000 games at 100 ms per move against the
generator, and as many against the fixed first network (it0). The new network
replaces the generator unless it is significantly worse: the upper end of
its 95% Elo interval against it is below zero.
When uploads are enabled and access is available, each iteration's network
and training checkpoint are uploaded to Hugging Face
as `twin-s-noattn-itN.onnx` and `twin-s-noattn-itN.pt`, and a promoted one as
`twin-s-noattn.onnx` and `twin-s-noattn.pt`, when `hf auth login` has logged
this machine in to an account that can write to the repository (`--repo`);
otherwise they stay local. Progress goes to
`data/distill/runs/rl-loop.log`. Every finished stage is recorded under
`data/distill/runs/rl-itN/stages`, so rerunning the command resumes where it
stopped; `touch data/distill/runs/rl-loop.stop` stops it between stages.
Matches open from the 100 positions in `engine/openings.tsv`.
`hivemind fetch-network` defaults to the released clock network. Explicit
`--variant onnx` and `--variant checkpoint` select the paired legacy training
artifacts pinned in `src/hivemind/network.py`; update those pins to publish
a promotion.

### Training on the clock

`--time-model clock` trains a network that reads the clock planes: self-play
plays every game on a simulated clock (`selfplay --time-model clock`: half the
games start level, the rest with a lead of 0.1-10 s either way, and each team
moves in 0.1-3 s), and matches run on one too. Such a network cannot replace
the flag model's generator, which engines from before the clock planes still
load, so it trains as a lineage of its own, from a copy of the generator it
starts from:

```bash
uv run hivemind fetch-network --variant onnx
uv run hivemind fetch-network --variant checkpoint
cp engine/models/twin-s-noattn.onnx engine/models/my-clock.onnx
cp engine/models/twin-s-noattn.onnx engine/models/my-clock-it0.onnx
mkdir -p artifacts/distill/my-clock
cp artifacts/distill/twin-s-noattn/best.pt artifacts/distill/my-clock/last.pt
uv run hivemind rl-loop --time-model clock --runs data/distill/runs-clock \
    --fixed-time-model flag --no-upload \
    --generator engine/models/my-clock.onnx \
    --generator-checkpoint artifacts/distill/my-clock/last.pt \
    --selfplay-engine engine/build-sp1/hivemind --match-engine engine/build-ninja/hivemind
```

The first iteration widens the checkpoint's input layer with zero weights for
the clock planes, so it starts out playing exactly as its parent. This
example creates a new local lineage from the pinned flag network and its
checkpoint. Use fresh output paths and do not overwrite an existing clock
lineage. Build the named engines from the current source: one search worker
for self-play (`-DHIVEMIND_SEARCH_WORKERS=1`) and the usual four for matches.
On macOS, use the one-worker Core ML engine path from the README instead. Uploads name
the lineage after its generator (`my-clock-itN.onnx`), and contributors to
a `--distributed` clock loop play clock games too (work-order protocol 2).

### Keeping data off the disk

A full iteration can occupy several GB, depending on game length. The loop
moves each finished segment to the private Hugging Face dataset
`aminwoo/bughouse-twin-s-selfplay-archive` (`--archive`) as a compressed zstd
tarball. It deletes the games only once the Hub's copy matches.
Training downloads the games it needs for the new iteration and
the replay window, and deletes them again when it finishes. Use
`--keep-selfplay` to keep everything on disk; without uploads (`--no-upload`,
or no `hf auth login`) the loop keeps everything on disk anyway.
`hivemind rl-loop --archive-only` moves all finished self-play and exits.

`archive` and `restore` do the same for any data directory:

```bash
uv run hivemind archive data/distill/train   # leaves data/distill/train/archive.json
uv run hivemind restore data/distill/train   # brings the files back
```

`hivemind train` restores the directories it is given by itself; other
training commands need a `restore` first.

### Sharing self-play with other machines

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
and submits each as a pull request. Contributors need an authenticated
Hugging Face account with access to the exchange (`uv run hf auth login`);
a private exchange also requires the operator to grant access. The loop
collects and merges the batches,
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

### Distilling from a teacher

`distill-train` can also train a twin-board student from a teacher network's
raw outputs. The engine's `gendistill` command plays games by sampling the
teacher's policy and writes every position with the teacher's value, WDL,
moves-left and policies, averaged over both board orders:

```bash
cd engine
./build-ninja/hivemind gendistill --model teacher.onnx --positions 60000000 \
    --threads 4 --batch-size 256 --seed 1001 --output ../data/distill/train
./build-ninja/hivemind gendistill --model teacher.onnx --positions 500000 \
    --seed 99 --output ../data/distill/val
cd ..
uv run hivemind distill-train --data data/distill/train --val data/distill/val \
    --size twin-s --out artifacts/distill/twin-s
```

`--search-data` mixes in self-play search targets as well; see
`hivemind distill-train --help`.

## Neural network architecture

The **twin-board** network (`twin-s`, `twin-m`, `twin-l` in
[`twin_board.py`](../src/hivemind/architectures/twin_board.py)) runs each
board's planes through the same convolutional trunk as their own 8x8 image,
and the boards exchange information through explicit, symmetric exchange
steps. Each board's planes are written from the team's point of view, so
swapping the two boards is a symmetry of the game: swapping the input blocks
swaps the two policies and leaves value, WDL and moves-left unchanged, and
every training position teaches both boards. The released twin-s-clock
network is `twin-s` with the clock planes below.

### Input Representation

Clock-aware twin-s networks use a **78-channel input** (78×8×8), with
39 channels per board. Older flag networks use 74 channels:

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
before the clock planes (the twin-s-noattn flag lineage) read channels 0-36
of each board, 74 in all.

### Output

- **Policy heads**: 4672 move probabilities for each board
- **Value head**: Scalar outcome prediction
- **WDL head**: Win/draw/loss classification
- **Moves-left head**: Remaining team-decision estimate
