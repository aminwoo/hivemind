# Mate benchmark: 10 chess.com bughouse games (October 2026)

Forced mates hivemind reports in every position of ten consecutive chess.com
bughouse games (all vs vjbaker), for auditing missed mates.

## Games

186251789121, 186251789181, 186251789413, 186251789497, 186251789537,
186251789605, 186252389103, 186252389157, 186252389205, 186252389241
(raw chess.com callback payloads with their partner boards in `games/`).
Positions come from the bughouse-analysis site's timeline code, so plies and
clocks match `https://bughouse-analysis.vercel.app/?game=<id>&ply=<ply>`.

## Searches

- Every mainline position before a board is mated, for **each team on turn**
  there: 1,546 searches.
- Engine: hivemind `5f9bb75` (`build-ninja`), network `twin-s-clock.onnx`
  (RL iteration 2, sha256 `f751f611...`).
- `go movetime 1000` (main files; a 300 ms run is kept in `runs/300ms/`), MultiPV 1, `TimeModel clock`, `TimeLeadMs` = the team's
  real diagonal clock lead at that position, `TeamMoveTimeMs` =
  `OpponentMoveTimeMs` = 100. The game's move history is sent, so repetitions
  count.
- Mates come from MCTS (its solver), the exact root scans (up to 8 attacking
  moves, checks only, plus clock-limited sat-partner proofs) and the
  Fairy-Stockfish root probe (up to #10, but only while the team leads on the
  clock, capped at lead / 100 ms moves, and only mate in 1 when the defenders'
  partner can capture on the other board).

## Files

- `mates.jsonl`: one record per search that reported a forced mate: game, ply,
  team, `outcome` (win/loss for that team), `mate_in`, clock lead, two-board FEN,
  full move history, the engine's first action and PV, whether the game's next
  move by that team was the engine's mating move.
- `mates.csv`: the same positions in the nachos benchmark format
  (`fen,color,lead,move,eval`; lead in seconds; winning rows expect the engine's
  mating action, losing rows only the eval). Run with
  `--team-move-time 100 --opponent-move-time 100`.
- `searches.jsonl`: every search, mate or not, for finding positions with a mate
  hivemind missed.
- `tools/`: the search driver and dataset builder.

## Results

| Run | Mates reported | Winning | Losing |
|---|---|---|---|
| 1 s (main) | 120 | 68 | 52 |
| 300 ms (`runs/300ms/`) | 114 | 64 | 50 |

By length at 1 s: 1:12, 2:16, 3:15, 4:24, 5:20, 6:10, 7:16, 8:4, 9:1, 10:2. One
position can appear twice, once per team (a win for one, a loss for the other),
and a mate is reported again at each step of its line.

The 1 s run found 6 mates the 300 ms run did not (186252389103 plies 116 and
118 white #4, 186252389157 ply 65 black #9, 186252389241 ply 89 black loses
#6, ply 92 white #5, ply 93 black loses #7) and lost none; 20 positions
changed the reported length by a move or two.
