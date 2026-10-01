#!/usr/bin/env bash
# Twin-s-noattn RL iteration: self-play, then training arms, then matches.
#   A: search policy + teacher anchor (value from the teacher)
#   B: A plus game-result WDL and moves-left from self-play
#   C: self-play only (no teacher data): policy from visits, WDL and moves-left from results
set -uo pipefail
cd /home/ben/hivemind || exit 1
GAMES=2550
VAL_GAMES=250
OUT=
ARMS=both
PASSES=3
PARALLEL=6
OPENINGS=/tmp/claude-1000/-home-ben-hivemind/dcfc0c59-e54c-4798-abc0-738a4b25289c/scratchpad/openings.tsv
while (($#)); do
    case "$1" in
        --games|--val-games|--out|--arm|--openings|--passes|--parallel-games)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 1; }
            case "$1" in
                --games) GAMES=$2 ;;
                --val-games) VAL_GAMES=$2 ;;
                --out) OUT=$2 ;;
                --arm) ARMS=$2 ;;
                --openings) OPENINGS=$2 ;;
                --passes) PASSES=$2 ;;
                --parallel-games) PARALLEL=$2 ;;
            esac
            shift 2 ;;
        *) echo "Usage: $0 [--games N] [--val-games N] [--out NEW_DIRECTORY] [--arm A|B|C|both] [--passes N] [--parallel-games N] [--openings TSV]" >&2; exit 1 ;;
    esac
done
[[ "$GAMES" =~ ^[1-9][0-9]*$ && "$VAL_GAMES" =~ ^[1-9][0-9]*$ ]] || exit 1
[[ "$ARMS" == A || "$ARMS" == B || "$ARMS" == C || "$ARMS" == both ]] || exit 1
[[ "$PASSES" =~ ^[1-9][0-9]*$ && "$PARALLEL" =~ ^[1-9][0-9]*$ ]] || exit 1
[[ -f "$OPENINGS" ]] || { echo "Missing openings: $OPENINGS" >&2; exit 1; }
if [[ -z "$OUT" ]]; then
    mkdir -p runs || exit 1
    OUT=$(mktemp -d "$PWD/runs/rl-pilot-XXXXXXXX") || exit 1
else
    mkdir -p "$(dirname "$OUT")" || exit 1
    mkdir "$OUT" || { echo "Output must be a fresh directory: $OUT" >&2; exit 1; }
fi
OUT=$(realpath "$OUT") || exit 1
D=$OUT/search
A=$OUT/arms
LOG=$OUT/pilot.log
mkdir -p "$D" "$A" "$OUT/models" || exit 1
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# Hash inputs and outputs so each corpus, checkpoint and match is identifiable.
manifest() {
    OMP_NUM_THREADS=2 .venv/bin/python - "$OUT" "$GAMES" "$VAL_GAMES" "$ARMS" "$OPENINGS" "$1" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
out, games, val_games, arms, openings, stage = sys.argv[1:]
root = Path(out)
paths = [Path('artifacts/distill/twin-s-noattn/best.pt'),
         Path('engine/models/twin-s-noattn-sp1.onnx'), Path('engine/models/twin-s-noattn.onnx'),
         Path('engine/build-sp1/hivemind'), Path('engine/build-ninja/hivemind'),
         Path(openings),
         Path('data/distill/rl_pilot.sh')]
# The launchers exec these; hash them where they exist (not in test trees).
paths += [p for p in (Path('engine/build-sp1/hivemind.bin'), Path('engine/build-ninja/hivemind.bin'))
          if p.exists()]
paths += sorted(Path('data/distill/train').glob('*.dst'))
paths += sorted(Path('data/distill/val').glob('*.dst'))
paths += sorted(p for p in root.rglob('*') if p.suffix in ('.dst', '.pt', '.onnx') or p.name == 'summary.json')
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
files = {str(p.resolve()): digest(p) for p in paths}
(root / 'manifest.json').write_text(json.dumps(dict(stage=stage, games=int(games),
    val_games=int(val_games), arms=arms, train_seed=31, val_seed=32, match_games=160,
    init='artifacts/distill/twin-s-noattn/best.pt', export='last', files=files), indent=2))
PY
}

selfplay() {
    local name=$1 games=$2 seed=$3 count
    log "self-play $name: $games games, seed $seed" || return 1
    if ! (cd engine && ./build-sp1/hivemind selfplay --model models/twin-s-noattn-sp1.onnx \
            --games "$games" --nodes 800 --node-random-factor 0.05 --seed "$seed" \
            --fairy-stockfish-mate-nodes 0 --resign-threshold 0 --parallel-games "$PARALLEL" \
            --training-chunks false --output "$D/${name}_run" --distill-output "$D/$name" \
            > "$D/$name.log" 2>&1); then
        log "self-play $name failed"; return 1
    fi
    count=$(grep -c '^selfplay game ' "$D/$name.log") || return 1
    [[ "$count" == "$games" ]] || { log "Expected $games games, got $count"; return 1; }
    OMP_NUM_THREADS=2 .venv/bin/python - "$D/$name" <<'PY'
import sys
from hivemind.distill.data import chunk_paths, chunk_positions
paths = chunk_paths([sys.argv[1]])
assert paths and sum(chunk_positions(p) for p in paths) > 0, 'Empty search corpus'
PY
}

arm() {
    local name=$1; shift
    local anchor=(--data data/distill/train --search-fraction 0.5) search_positions steps
    [[ "$name" == C ]] && anchor=(--search-fraction 1)
    # PASSES presentations of each self-play position at batch 1024.
    search_positions=$(OMP_NUM_THREADS=2 .venv/bin/python -c "
from hivemind.distill.data import chunk_paths, chunk_positions
print(sum(chunk_positions(p) for p in chunk_paths(['$D/train'])))") || return 1
    local rows
    [[ "$name" == C ]] && rows=1024 || rows=512
    steps=$(( (PASSES * search_positions + rows - 1) / rows ))
    log "training arm $name: $steps steps ($PASSES passes over $search_positions positions): $*" || return 1
    if ! PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 .venv/bin/hivemind distill-train \
            --init artifacts/distill/twin-s-noattn/best.pt \
            "${anchor[@]}" --val data/distill/val \
            --search-data "$D/train" --search-val "$D/val" \
            --steps "$steps" --lr 1e-4 --eval-every "$(( steps / 4 > 0 ? steps / 4 : 1 ))" --window-chunks 2 \
            --search-window-chunks 2 --export last \
            --search-value-weight 0 "$@" --out "$A/$name" > "$A/$name.log" 2>&1; then
        log "arm $name failed (see $A/$name.log)"; return 1
    fi
    [[ -s "$A/$name/last.pt" && -s "$A/$name/distill-twin-s-v3.0.onnx" ]] || return 1
    cp "$A/$name/distill-twin-s-v3.0.onnx" "$OUT/models/$name.onnx" || return 1
    cmp -s "$A/$name/distill-twin-s-v3.0.onnx" "$OUT/models/$name.onnx" || return 1
    manifest "trained-$name" || return 1
}

match() {
    local name=$1 summary
    log "match $name vs twin-s-noattn (100 ms, 160 games)" || return 1
    if ! (cd engine && ./build-ninja/hivemind tournament --contender "$OUT/models/$name.onnx" \
        --baseline models/twin-s-noattn.onnx --movetime 100 --games 160 --seed 4 \
        --positions "$OPENINGS" --output "$A/match_$name" > "$A/match_$name.log" 2>&1); then
        log "match $name failed"; return 1
    fi
    summary=$(OMP_NUM_THREADS=2 .venv/bin/python - "$A/match_$name/summary.json" <<'PY'
import json
import math
import sys
d = json.load(open(sys.argv[1]))
counts = [d[k] for k in ('contender_wins', 'baseline_wins', 'draws')]
assert all(type(n) is int and n >= 0 for n in counts) and sum(counts) == 160, 'Incomplete match'
# The tournament reports null Elo or interval for one-sided results (e.g. 0-160).
elo, interval = d['contender_elo'], d['elo_confidence_95']
assert elo is None or math.isfinite(elo)
assert interval is None or (len(interval) == 2 and all(n is None or math.isfinite(n) for n in interval))
fmt = lambda n: 'undefined' if n is None else f'{n:+.0f}'
ci = 'undefined' if interval is None else f'{fmt(interval[0])} to {fmt(interval[1])}'
print(f"{counts[0]}-{counts[1]}-{counts[2]}, Elo {fmt(elo)} (95% CI {ci}), "
      f"nps {d['performance']['contender']['nps']:.0f} vs {d['performance']['baseline']['nps']:.0f}")
PY
    ) || { log "Invalid match summary for $name"; return 1; }
    log "$name vs twin-s-noattn: $summary" || return 1
    manifest "matched-$name" || return 1
}

manifest initialized || exit 1
selfplay train "$GAMES" 31 || exit 1
selfplay val "$VAL_GAMES" 32 || exit 1
manifest generated || exit 1
if [[ "$ARMS" == both || "$ARMS" == A ]]; then
    arm A --search-wdl-weight 0 --search-moves-left-weight 0 || exit 1
    match A || exit 1
fi
if [[ "$ARMS" == both || "$ARMS" == B ]]; then
    arm B --search-wdl-weight 1 --search-moves-left-weight 0.1 || exit 1
    match B || exit 1
fi
if [[ "$ARMS" == C ]]; then
    arm C --search-wdl-weight 1 --search-moves-left-weight 0.1 || exit 1
    match C || exit 1
fi
manifest finished || exit 1
log "pilot finished: $OUT" || exit 1
