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
replaces the generator unless it scores below 50% against it.
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
uv run hivemind archive data/nnue/train   # leaves data/nnue/train/archive.json
uv run hivemind restore data/nnue/train   # brings the files back
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

## Neural network architecture

The released **twin-s-clock** model uses a shared convolutional trunk for
each board, symmetric board exchanges, and clock inputs. The older
**RISEv3** teacher architectures remain available for training experiments.
RISEv3 (Residual Inverted Squeeze-Excitation) combines:

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
before the clock planes (RISE, and twin-s up to iteration 10) read channels
0-36 of each board, 74 in all.

### Output

- **Policy heads**: 4672 move probabilities for each board
- **Value head**: Scalar outcome prediction
- **WDL head**: Win/draw/loss classification
- **Moves-left head**: Remaining team-decision estimate
