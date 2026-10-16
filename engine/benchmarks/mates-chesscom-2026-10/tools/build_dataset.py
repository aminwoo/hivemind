"""Turn results.jsonl into the mate benchmark dataset.

Usage: python build_dataset.py results.jsonl OUT_DIR
Writes OUT_DIR/searches.jsonl (every search), mates.jsonl (searches that
reported a forced mate, either way) and mates.csv (nachos benchmark format).
"""
import csv
import json
import sys
from pathlib import Path

results_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
rows = [json.loads(line) for line in results_path.open()]
rows.sort(key=lambda r: (r["game"], r["ply"], r["team"]))


def pv_first(r):
    return (r.get("pv") or [None])[0]


def played_matches(r):
    """Whether the game's next move by this team is on the board where the
    engine's first action moves, and is that move."""
    nm, first = r.get("nextMove"), pv_first(r)
    if not nm or not first or nm["team"] != r["team"]:
        return None
    a, b = first.strip("()").split(",")
    engine_move = a if nm["board"] == "A" else b
    return engine_move == nm["uci"]


with (out_dir / "searches.jsonl").open("w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

mates = []
for r in rows:
    if r.get("score_kind") != "mate":
        continue
    m = {
        "id": f"{r['game']}-{r['ply']}-{r['team']}",
        "game": r["game"],
        "partner_game": r["partner"],
        "url": f"https://www.chess.com/game/live/{r['game']}",
        "analysis_url": f"https://bughouse-analysis.vercel.app/?game={r['game']}&ply={r['ply']}",
        "ply": r["ply"],
        "team": r["team"],
        "outcome": "win" if r["score"] > 0 else "loss",
        "mate_in": abs(r["score"]),
        "lead_ms": r["leadMs"],
        "team_move_time_ms": 100,
        "opponent_move_time_ms": 100,
        "fen": r["fen"],
        "position": r["position"],
        "moves": r["moves"],
        "engine_action": pv_first(r),
        "engine_pv": r.get("pv"),
        "search_ms": r.get("elapsed_ms"),
        "nodes": r.get("nodes"),
        "next_game_move": r.get("nextMove"),
        "game_move_matches_engine": played_matches(r),
    }
    mates.append(m)

with (out_dir / "mates.jsonl").open("w") as f:
    for m in mates:
        f.write(json.dumps(m) + "\n")

# Nachos benchmark format: fen,color,lead,move,eval. A winning mate expects the
# engine's mating action (any other mate is a legitimate alternative, so audit
# misses by eval first); a losing one only expects the eval.
with (out_dir / "mates.csv").open("w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["fen", "color", "lead", "move", "eval"])
    for m in mates:
        move = m["engine_action"] if m["outcome"] == "win" else ""
        w.writerow([m["fen"], m["team"], f"{m['lead_ms'] / 1000:g}", move,
                    "winning" if m["outcome"] == "win" else "losing"])

wins = sum(m["outcome"] == "win" for m in mates)
print(f"{len(rows)} searches, {len(mates)} mates ({wins} winning, {len(mates) - wins} losing)")
by_len = {}
for m in mates:
    by_len.setdefault(m["mate_in"], 0)
    by_len[m["mate_in"]] += 1
print("by length:", dict(sorted(by_len.items())))
errors = [r for r in rows if r.get("error")]
print(f"{len(errors)} searches failed")
