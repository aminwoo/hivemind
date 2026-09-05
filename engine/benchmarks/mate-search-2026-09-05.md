# Mate search optimization — 2026-09-05

Compared against the working-tree source present before this change, including
its existing root mate, capture-feed, reverse-loss and UCI changes. No search
budgets, network weights or search parameter defaults were changed.

## Changes

- Generate legal checks by filtering pseudo-legal moves for check before
  testing legality. This avoids legality work on non-checking drops and retains
  checking evasions, castling, en passant and underpromotions.
- Reserve the exact capacity when returning a full legal move list.
- Validate a specific move directly instead of generating every legal move.
- Retain defender replies from the single-board terminal scan and search
  checks with fewer replies first. Terminal draws and losses are not expanded.
- Use the checking-move path in the waiting-board mate detector shared with
  MCTS.

## Results

Input: the current `nachos/positions.csv`, 29 source rows and their board-swapped
equivalents (58 cases). Duplicate source rows were retained. These are tactical
regressions, not an independent playing-strength or Elo measurement.

| Measurement | Before | After |
| --- | ---: | ---: |
| Full CPU UCI benchmark, expected results matched | 42/50 | 44/50 |
| Full CPU UCI benchmark, errors | 0 | 0 |
| Direct root solver, mates at 10,000-node allowance | 30/58 | 34/58 |
| Direct root solver, mates at 100,000-node allowance | 34/58 | 34/58 |
| Direct root solver, total time at 100,000-node allowance | 1,166 ms | 593 ms |

The direct solver timing is the sum of per-case medians over three runs of
`Agent::find_root_mate`, using the same configured allowance on each side and
no wall-clock deadline. Initialization and neural inference were excluded.
Both binaries used the existing ONNX Runtime Release build configuration with
`HIVEMIND_FAST_BUILD=ON`. The 49% reduction in time (1.97x speedup) applies to
this mate-solver workload, not overall engine NPS.

At the smaller allowance, positions 11 (`Qxh5`, a capture feeding a bishop
mate) and 17 (`Qh5`, a queen sacrifice) are newly proven in both orientations.
At 100,000 nodes, their unflipped searches fall from approximately 68 to 20 ms
and 69 to 21 ms respectively. Position 29's `Qxd6` proof falls from 222 to
106 ms and is newly selected in both orientations by the full UCI benchmark.
No previously matched UCI case was lost. Positions 5, 10 and 16 still miss the
expected result in both orientations.

The full benchmark used 1,000 ms requested move time, batch size 16, four search
workers, draw contempt 1000, and the existing
`hivemind-it04-crossboard-risev33-loss1.556-p82.0.onnx` network. CPU inference
often exceeded the move time, so full-search timing is not evidence of a GPU
speedup. Existing mate-proof UCI lines also report synthetic `nodes 1 nps 1000
time 0`; those numbers were not used for the speed comparison.

## Verification and reproduction

All 227 tests pass in the CPU build. The four new tactical tests also pass in
the optimized TensorRT build without requiring neural inference.

The new `test_tactical_movegen.cc` compares the optimized move operations with
full legal move generation on special-move positions and deterministic games
with all five pocket piece types. It also requires the two deeper mates to be
found within the 10,000-node allowance in both board orientations and verifies
that the solver restores the board and its search hash.

```sh
cmake --build engine/build-ort-release-test -j 6
ctest --test-dir engine/build-ort-release-test --output-on-failure -j 2
```

Run the original end-to-end suite from `nachos`:

```sh
.venv/bin/python benchmark.py \
  --engine ../hivemind/engine/build-ort-release-test/hivemind \
  --model ../models/hivemind-it04-crossboard-risev33-loss1.556-p82.0.onnx \
  --movetime 1000 --output /tmp/hivemind-cpu-results.json
```

The TensorRT build is also rebuilt at `engine/build-ninja/hivemind`, the path
configured by Nachos. Its runtime performance could not be measured because
the NVIDIA driver was unavailable in this environment. On a GPU host, run
Nachos's usual `benchmark.py` command to measure the production backend.

## Follow-up: early exit after a concurrent MCTS proof

Position 16 exposed a separate delay: workers stopped on a solved root, but
the calling thread's reverse-loss scan only watched its deadline and explicit
UCI cancellation. It could therefore occupy the remainder of the move time
after MCTS had already proved a mate.

All root-scan budgets now also watch the concurrent root's solved status,
including nested opponent proofs, capture-feed scans and immediate scans.
Cancellation unwinds the board normally and preserves the MCTS proof. A high
neural evaluation alone does not trigger this stop condition.

A regression uses position 16 and its flip with a simulated worker publishing
a root proof after 20 ms. Before the fix, each reverse scan returned at about
1,001 ms; afterward, each returned at about 20 ms. This measures the delay
after a proof becomes available, not the time a real network needs to find it.
