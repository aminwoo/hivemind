#!/usr/bin/env bash
# Twin-s-noattn AlphaZero loop. Each iteration N:
#   1. plays --games self-play games with the current generator, in segments
#   2. plays --val-games validation games
#   3. trains from iteration N-1's last weights on the new games (--new-passes)
#      plus a replay buffer of the previous --window iterations (--replay-passes
#      over each, per iteration)
#   4. matches the result against the generator; it becomes the generator
#      unless it scores below 50%
#   5. matches it against the fixed pre-RL network (twin-s-noattn-it0)
# Every finished stage leaves a marker in runs/rl-itN/stages, so rerunning the
# script resumes at the first unfinished one (an interrupted segment or match is
# moved to runs/rl-itN/aborted and redone). Matches and training wait while a
# game is running on the GPU, and a match restarts if one starts mid-way.
#   Stop between stages: touch data/distill/runs/rl-loop.stop
set -uo pipefail
cd /home/ben/hivemind || exit 1
ROOT=$PWD
FIRST=
LAST=8
GAMES=100000
SEGMENT=10000
VAL_GAMES=1000
WINDOW=3
NEW_PASSES=1.5
REPLAY_PASSES=0.5
PARALLEL=6
MATCH_GAMES=160
BATCH_SIZE=1024
OPENINGS=$ROOT/data/distill/openings.tsv
RUNS=$ROOT/data/distill/runs
# Process names (exact) that mean a game is running: Steam starts every game
# under reaper, and Proton runs wineserver.
GAMING=${RL_GAMING_PATTERN:-reaper|wineserver}
QUIET_SECONDS=${RL_QUIET_SECONDS:-300}
POLL_SECONDS=${RL_POLL_SECONDS:-20}
RETRY_SECONDS=${RL_RETRY_SECONDS:-60}
GENERATOR=$ROOT/engine/models/twin-s-noattn.onnx
GENERATOR_SP1=$ROOT/engine/models/twin-s-noattn-sp1.onnx
GENERATOR_PT=$ROOT/artifacts/distill/twin-s-noattn/best.pt
FIXED=$ROOT/engine/models/twin-s-noattn-it0.onnx
usage="Usage: $0 [--first N] [--last N] [--games N] [--segment-games N] [--val-games N] [--window N]
          [--new-passes X] [--replay-passes X] [--parallel-games N] [--match-games N] [--batch-size N] [--openings TSV]"
while (($#)); do
    (($# >= 2)) || { echo "$usage" >&2; exit 1; }
    case "$1" in
        --first) FIRST=$2 ;;
        --last) LAST=$2 ;;
        --games) GAMES=$2 ;;
        --segment-games) SEGMENT=$2 ;;
        --val-games) VAL_GAMES=$2 ;;
        --window) WINDOW=$2 ;;
        --new-passes) NEW_PASSES=$2 ;;
        --replay-passes) REPLAY_PASSES=$2 ;;
        --parallel-games) PARALLEL=$2 ;;
        --match-games) MATCH_GAMES=$2 ;;
        --batch-size) BATCH_SIZE=$2 ;;
        --openings) OPENINGS=$2 ;;
        *) echo "$usage" >&2; exit 1 ;;
    esac
    shift 2
done
for n in "$LAST" "$GAMES" "$SEGMENT" "$VAL_GAMES" "$WINDOW" "$PARALLEL" "$MATCH_GAMES" "$BATCH_SIZE" ${FIRST:+"$FIRST"}; do
    [[ "$n" =~ ^[1-9][0-9]*$ ]] || { echo "$usage" >&2; exit 1; }
done
[[ "$NEW_PASSES" =~ ^[0-9]*\.?[0-9]+$ && "$REPLAY_PASSES" =~ ^[0-9]*\.?[0-9]+$ ]] || { echo "$usage" >&2; exit 1; }
OPENINGS=$(realpath "$OPENINGS") || exit 1
for path in "$OPENINGS" "$GENERATOR" "$GENERATOR_SP1" "$GENERATOR_PT" "$FIXED"; do
    [[ -f "$path" ]] || { echo "Missing $path" >&2; exit 1; }
done
mkdir -p "$RUNS" || exit 1
exec 9> "$RUNS/rl-loop.lock" || exit 1
flock -n 9 || { echo "Another rl_loop.sh is running" >&2; exit 1; }
if [[ -z "$FIRST" ]]; then
    FIRST=2
    while [[ -f "$RUNS/rl-it$FIRST/stages/iteration.done" ]]; do FIRST=$((FIRST + 1)); done
fi
((FIRST >= 2)) || { echo "--first must be at least 2 (iteration 1 is the pilot)" >&2; exit 1; }

log() { echo "$(date '+%F %T') $*" | tee -a "$RUNS/rl-loop.log"; }
sha() { sha256sum "$1" | cut -c1-64; }
py() { OMP_NUM_THREADS=2 .venv/bin/python "$@"; }
gaming() { pgrep -x "$GAMING" > /dev/null; }
stop_requested() {
    [[ -e "$RUNS/rl-loop.stop" ]] || return 1
    log "stop requested ($RUNS/rl-loop.stop): exiting before the next stage"
}

# Write a stage marker atomically; its contents describe what the stage did.
mark() {
    printf '%s\n' "$2" > "$IT/stages/$1.tmp" && mv "$IT/stages/$1.tmp" "$IT/stages/$1.done"
}
done_stage() { [[ -f "$IT/stages/$1.done" ]]; }

# Move leftovers of an unfinished stage out of the way.
aside() {
    local path rel stamp
    stamp=$(date +%Y%m%d-%H%M%S)
    for path in "$@"; do
        [[ -e "$path" ]] || continue
        rel=${path#"$IT"/}
        mkdir -p "$IT/aborted" && mv "$path" "$IT/aborted/${rel//\//_}-$stamp" || return 1
    done
}

wait_for_quiet() {
    local quiet=0
    gaming || return 0
    log "waiting: a game is running ($GAMING)"
    while ((quiet < QUIET_SECONDS)); do
        if gaming; then quiet=0; else quiet=$((quiet + POLL_SECONDS)); fi
        sleep "$POLL_SECONDS"
    done
    log "no game for ${QUIET_SECONDS}s: continuing"
}

# Self-play training data directories of iteration $1, oldest layout first
# (iteration 1 wrote its chunks straight into search/train).
train_dirs() {
    local dir=$RUNS/rl-it$1/search/train marker
    compgen -G "$dir/*.dst" > /dev/null && echo "$dir"
    for marker in "$RUNS/rl-it$1"/stages/seg*.done; do
        [[ -e "$marker" ]] && echo "$dir/$(basename "$marker" .done)"
    done
}

positions() {
    py - "$@" <<'PY'
import sys
from hivemind.distill.data import chunk_paths, chunk_positions
print(sum(chunk_positions(p) for p in chunk_paths(sys.argv[1:])))
PY
}

# One self-play run: name, games, seed, data dir, run dir.
selfplay() {
    local name=$1 games=$2 seed=$3 data=$4 run=$5 log=$IT/search/logs/$1.log count n
    done_stage "$name" && return 0
    stop_requested && exit 0
    [[ "$(sha "$GENERATOR_SP1")" == "$(cat "$IT/stages/generator")" ]] \
        || { log "generator changed during it$N"; return 1; }
    aside "$data" "$run" "$log" || return 1
    mkdir -p "$IT/search/logs" || return 1
    log "it$N self-play $name: $games games, seed $seed"
    if ! (cd engine && ./build-sp1/hivemind selfplay --model "$GENERATOR_SP1" \
            --games "$games" --nodes 800 --node-random-factor 0.05 --seed "$seed" \
            --fairy-stockfish-mate-nodes 0 --resign-threshold 0 --parallel-games "$PARALLEL" \
            --training-chunks false --output "$run" --distill-output "$data" \
            > "$log" 2>&1); then
        log "it$N self-play $name failed (see $log)"; return 1
    fi
    count=$(grep -c '^selfplay game ' "$log")
    [[ "$count" == "$games" ]] || { log "it$N $name: expected $games games, got $count"; return 1; }
    n=$(positions "$data") && ((n > 0)) || { log "it$N $name: empty corpus"; return 1; }
    mark "$name" "{\"games\": $games, \"seed\": $seed, \"positions\": $n, \"generator\": \"$(cat "$IT/stages/generator")\"}"
}

train() {
    local out=$A/C init=$RUNS/rl-it$((N - 1))/arms/C/last.pt new=() old=() m dir plan steps fraction
    local replay=() npos opos
    done_stage train && return 0
    stop_requested && exit 0
    [[ -f "$init" ]] || { log "missing $init"; return 1; }
    mapfile -t new < <(train_dirs "$N")
    for ((m = N - 1; m >= 1 && m >= N - WINDOW; m--)); do
        while read -r dir; do old+=("$dir"); done < <(train_dirs "$m")
    done
    plan=$(py - "$NEW_PASSES" "$REPLAY_PASSES" "$BATCH_SIZE" "${#new[@]}" "${new[@]}" "${old[@]}" <<'PY'
import math
import sys
from hivemind.distill.data import chunk_paths, chunk_positions
new_passes, replay_passes = float(sys.argv[1]), float(sys.argv[2])
batch_size, count = int(sys.argv[3]), int(sys.argv[4])
dirs = sys.argv[5:]
new = sum(chunk_positions(p) for p in chunk_paths(dirs[:count]))
old = sum(chunk_positions(p) for p in chunk_paths(dirs[count:])) if dirs[count:] else 0
samples_new, samples_old = new_passes * new, replay_passes * old
assert new > 0
print(math.ceil((samples_new + samples_old) / batch_size), round(samples_old / (samples_new + samples_old), 4), new, old)
PY
    ) || return 1
    read -r steps fraction npos opos <<< "$plan"
    ((${#old[@]})) && replay=(--replay-data "${old[@]}" --replay-fraction "$fraction")
    aside "$out" "$A/C.log" || return 1
    wait_for_quiet
    log "it$N training: $steps steps from $init, $npos new positions x$NEW_PASSES," \
        "$opos replay positions x$REPLAY_PASSES (replay share $fraction, ${#old[@]} dirs, batch $BATCH_SIZE)"
    if ! PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 .venv/bin/hivemind distill-train \
            --init "$init" --search-fraction 1 --val data/distill/val \
            --search-data "${new[@]}" --search-val "$IT/search/val" "${replay[@]}" \
            --steps "$steps" --lr 1e-4 --eval-every "$(( steps / 4 > 0 ? steps / 4 : 1 ))" \
            --batch-size "$BATCH_SIZE" --eval-batch-size "$BATCH_SIZE" \
            --search-window-chunks 2 --replay-window-chunks 2 --export last --seed "$N" \
            --search-value-weight 0 --search-wdl-weight 1 --search-moves-left-weight 0.1 \
            --out "$out" > "$A/C.log" 2>&1; then
        log "it$N training failed (see $A/C.log)"; return 1
    fi
    [[ -s "$out/last.pt" && -s "$out/distill-twin-s-v3.0.onnx" ]] || return 1
    cp "$out/distill-twin-s-v3.0.onnx" "$IT/models/C.onnx" \
        && cmp -s "$out/distill-twin-s-v3.0.onnx" "$IT/models/C.onnx" \
        && cp "$IT/models/C.onnx" "engine/models/twin-s-noattn-it$N.onnx" || return 1
    mark train "{\"steps\": $steps, \"replay_fraction\": $fraction, \"new_positions\": $npos, \"replay_positions\": $opos, \"init\": \"$(sha "$init")\", \"onnx\": \"$(sha "$IT/models/C.onnx")\"}"
}

# A timed match of this iteration's network against $2; restarts if a game starts.
match() {
    local name=$1 baseline=$2 out=$A/match_$1 pid status summary
    done_stage "match_$name" && return 0
    stop_requested && exit 0
    while :; do
        aside "$out" "$out.log" || return 1
        wait_for_quiet
        log "it$N match vs $name ($baseline, ${MATCH_GAMES} games at 100 ms)"
        (cd engine && exec ./build-ninja/hivemind tournament --contender "$IT/models/C.onnx" \
            --baseline "$baseline" --movetime 100 --games "$MATCH_GAMES" --seed "$N" \
            --positions "$OPENINGS" --output "$out" > "$out.log" 2>&1) &
        pid=$!
        while kill -0 "$pid" 2> /dev/null && ! gaming; do sleep "$POLL_SECONDS"; done
        if kill -0 "$pid" 2> /dev/null; then
            kill "$pid"; wait "$pid"
            log "it$N match vs $name: a game started, restarting the match"
            continue
        fi
        wait "$pid"; status=$?
        ((status == 0)) || { log "it$N match vs $name failed (see $out.log)"; return 1; }
        break
    done
    summary=$(py - "$out/summary.json" "$MATCH_GAMES" <<'PY'
import json
import math
import sys
d = json.load(open(sys.argv[1]))
games = int(sys.argv[2])
w, l, dr = (d[k] for k in ('contender_wins', 'baseline_wins', 'draws'))
assert all(type(n) is int and n >= 0 for n in (w, l, dr)) and w + l + dr == games, 'Incomplete match'
elo, ci = d['contender_elo'], d['elo_confidence_95']
assert elo is None or math.isfinite(elo)
print(json.dumps(dict(wins=w, losses=l, draws=dr, score=(w + dr / 2) / games, elo=elo, ci=ci,
                      nps=[d['performance']['contender']['nps'], d['performance']['baseline']['nps']])))
PY
    ) || { log "it$N match vs $name: invalid summary"; return 1; }
    log "it$N vs $name: $summary"
    mark "match_$name" "$summary"
}

install() { cp "$1" "$2.tmp" && mv "$2.tmp" "$2"; }

promote() {
    local score
    done_stage promote && return 0
    score=$(py -c "import json, sys; print(json.load(open(sys.argv[1]))['score'])" "$IT/stages/match_generator.done") \
        || return 1
    if py -c "import sys; sys.exit(float(sys.argv[1]) < 0.5)" "$score"; then
        install "$IT/models/C.onnx" "$GENERATOR" && install "$IT/models/C.onnx" "$GENERATOR_SP1" \
            && install "$A/C/last.pt" "$GENERATOR_PT" || { log "it$N promotion failed"; return 1; }
        log "it$N promoted (score $score): it is the generator for it$((N + 1))"
        mark promote "{\"promoted\": true, \"score\": $score}"
    else
        log "it$N not promoted (score $score): the generator stays"
        mark promote "{\"promoted\": false, \"score\": $score}"
    fi
}

# Retry a stage (a crash, an out-of-memory error) before giving up.
attempt() {
    local try
    for try in 1 2 3; do
        "$@" && return 0
        log "it$N: $1 failed (attempt $try of 3)"
        sleep "$RETRY_SECONDS"
    done
    return 1
}

log "loop: iterations $FIRST..$LAST, $GAMES games ($SEGMENT per segment), window $WINDOW," \
    "passes new $NEW_PASSES / replay $REPLAY_PASSES"
for ((N = FIRST; N <= LAST; N++)); do
    IT=$RUNS/rl-it$N A=$RUNS/rl-it$N/arms
    done_stage iteration && continue
    if [[ -e "$IT" && ! -d "$IT/stages" ]]; then
        log "$IT exists but was not made by this loop"; exit 1
    fi
    mkdir -p "$IT/stages" "$IT/search" "$A" "$IT/models" || exit 1
    # The generator is fixed for a whole iteration's self-play.
    [[ -f "$IT/stages/generator" ]] || sha "$GENERATOR_SP1" > "$IT/stages/generator" || exit 1
    for ((k = 0; k * SEGMENT < GAMES; k++)); do
        games=$(( GAMES - k * SEGMENT < SEGMENT ? GAMES - k * SEGMENT : SEGMENT ))
        attempt selfplay "$(printf 'seg%02d' "$k")" "$games" $((1000 * N + k)) \
            "$IT/search/train/$(printf 'seg%02d' "$k")" "$IT/search/train_run/$(printf 'seg%02d' "$k")" || exit 1
    done
    attempt selfplay val "$VAL_GAMES" $((1000 * N + 999)) "$IT/search/val" "$IT/search/val_run" || exit 1
    attempt train || exit 1
    attempt match generator "$GENERATOR" || exit 1
    promote || exit 1
    attempt match it0 "$FIXED" || exit 1
    mark iteration "{}"
    log "it$N finished: vs generator $(cat "$IT/stages/match_generator.done"), vs it0 $(cat "$IT/stages/match_it0.done")"
done
log "loop finished at it$LAST"
