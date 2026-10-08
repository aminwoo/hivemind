#!/usr/bin/env python3
"""Screen Bughouse search settings, select on validation, test one held-out finalist.

Uses one opening per teacher-policy game and disjoint books between stages.
Writes a resumable report; never changes engine defaults automatically.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import struct
import subprocess
import sys
from pathlib import Path

from run_strength_sweep import file_digest


CANDIDATES = {
    "control": {"pw-mass": 0},
    "mass-legacy": {"pw-mass": .35},
    "mass-legal": {"pw-mass": .35, "pw-mass-normalize": "true"},
    "mass-low": {"pw-mass": .20, "pw-mass-exponent": .10, "pw-mass-normalize": "true"},
    "mass-high": {"pw-mass": .50, "pw-mass-exponent": .25, "pw-mass-normalize": "true"},
    "mass-root-wide": {"pw-mass": .25, "root-pw-mass": .65,
                       "pw-mass-exponent": .10, "pw-mass-normalize": "true"},
    "mass-root-only": {"pw-mass": 0, "root-pw-mass": .60, "pw-mass-normalize": "true"},
    "count-narrow": {"pw-coefficient": 1, "root-pw-coefficient": 4},
    "count-balanced": {"pw-coefficient": 1.5, "root-pw-coefficient": 3},
    "count-wide": {"pw-coefficient": 3, "root-pw-coefficient": 6},
    "count-slow": {"pw-exponent": .20},
    "cpuct-low": {"cpuct-init": 2},
    "cpuct-high": {"cpuct-init": 4},
    "moves-left-off": {"moves-left-discount": 0},
    "wdl-quarter": {"wdl-eval": "true", "wdl-weight": .25},
    "q-half": {"q-value-weight": .5},
}


def prepare_books(source: Path, output: Path, seed: int,
                  screen_pairs: int = 1000, validate_pairs: int = 1000,
                  holdout_pairs: int = 1000) -> dict:
    """Read HNUE v1 ply metadata to preserve game boundaries across chunks."""
    if min(screen_pairs, validate_pairs, holdout_pairs) <= 0:
        raise ValueError("opening pair counts must be positive")
    rng = random.Random(seed)
    openings: list[str] = []
    game: list[str] = []
    seen: set[str] = set()

    def finish_game() -> None:
        if game:
            chosen = rng.choice(game)
            if chosen not in seen:
                seen.add(chosen)
                openings.append(chosen)
        game.clear()

    sources = {}
    for chunk in sorted(source.glob("*.bin")):
        with chunk.open("rb") as stream:
            magic, version, _, count, _ = struct.unpack("<4sIIQQ", stream.read(28))
            if magic != b"HNUE" or version != 1:
                raise ValueError(f"unsupported opening source: {chunk}")
            stream.seek(28 + count * 17)
            plies = struct.unpack(f"<{count}H", stream.read(count * 2))
        fen_file = chunk.with_suffix(".fen")
        fens = fen_file.read_text().splitlines()
        if len(fens) != count:
            raise ValueError(f"FEN/record count differs: {chunk}")
        sources[str(chunk.resolve())] = file_digest(chunk)
        sources[str(fen_file.resolve())] = file_digest(fen_file)
        for ply, fen in zip(plies, fens):
            if ply == 0:
                finish_game()
            if 6 <= ply <= 18:
                a, b, team, advantage = fen.split(";")
                game.append(f"{a}|{b}\t{team}\t{advantage}")
    finish_game()
    rng.shuffle(openings)
    required = screen_pairs + validate_pairs + holdout_pairs
    if len(openings) < required:
        raise ValueError(f"need at least {required} distinct games; found {len(openings)}")
    output.mkdir(parents=True, exist_ok=True)
    validation_end = screen_pairs + validate_pairs
    splits = {"screen": openings[:screen_pairs],
              "validate": openings[screen_pairs:validation_end],
              "holdout": openings[validation_end:]}
    for name, lines in splits.items():
        (output / f"{name}.tsv").write_text("\n".join(lines) + "\n")
    manifest = {"seed": seed, "source_sha256": sources, "opening_ply_range": [6, 18],
                "one_position_per_game": True, "counts": {k: len(v) for k, v in splits.items()},
                "sha256": {k: file_digest(output / f"{k}.tsv") for k in splits}}
    (output / "openings.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def write_report(output: Path, state: dict) -> None:
    lines = ["# Bughouse search tuning", "", f"Status: {state['status']}", "",
             "All comparisons use the same twin-s-noattn network, four workers, batch 8,",
             "color-swapped opening pairs, and no per-move Dirichlet noise. Stages use",
             "disjoint openings, each sampled from a different teacher-policy game.", "",
             "Discovery estimates select candidates; only the independent held-out",
             "finalist test is used to judge a strength gain. Defaults are unchanged.", "",
             "| Stage | Candidate | Games | Score | Elo | 95% Elo interval |",
             "|---|---|---:|---:|---:|---|" ]
    for row in state["results"]:
        result = row["summary"]
        elo = result.get("contender_elo")
        interval = result.get("elo_confidence_95")
        bounds = ", ".join("unbounded" if x is None else f"{x:+.1f}" for x in interval) if interval else "unavailable"
        lines.append(f"| {row['stage']} | {row['name']} | {result['games']} | "
                     f"{result['contender_score']:.1%} | {elo:+.1f} | {bounds} |" if elo is not None else
                     f"| {row['stage']} | {row['name']} | {result['games']} | "
                     f"{result['contender_score']:.1%} | unbounded | {bounds} |")
    lines += ["", state.get("conclusion", "Strength validation is still pending."), "",
              "Parameters and commands: see campaign.json and each run's invocation.json.",
              "Confidence intervals use paired-opening normal approximation; short-budget",
              "selfplay results need confirmation at the intended playing budget."]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opening-source", type=Path)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--screen-games", type=int, default=2000)
    parser.add_argument("--screen-ms", type=int, default=30)
    parser.add_argument("--validate-games", type=int, default=2000)
    parser.add_argument("--validate-ms", type=int, default=60)
    parser.add_argument("--holdout-games", type=int, default=2000)
    parser.add_argument("--holdout-ms", type=int, default=100)
    args = parser.parse_args()
    for stage, games in [("screen", args.screen_games), ("validate", args.validate_games),
                         ("holdout", args.holdout_games)]:
        if games <= 0 or games % 2:
            parser.error(f"{stage} needs a positive even game count")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    books = output / "openings"
    if not (books / "openings.json").exists():
        if not args.opening_source:
            parser.error("--opening-source required for first run")
        prepare_books(args.opening_source, books, args.seed,
                      args.screen_games // 2, args.validate_games // 2,
                      args.holdout_games // 2)
    manifest = json.loads((books / "openings.json").read_text())
    for stage, games in [("screen", args.screen_games), ("validate", args.validate_games),
                         ("holdout", args.holdout_games)]:
        if games <= 0 or games % 2 or games // 2 > manifest["counts"][stage]:
            parser.error(f"{stage} needs a positive even game count and enough distinct openings")
        if file_digest(books / f"{stage}.tsv") != manifest["sha256"][stage]:
            raise RuntimeError(f"opening book changed: {stage}")
    snapshot = output / "hivemind-tuning.bin"
    if not snapshot.exists():
        shutil.copy2(args.engine, snapshot)
    runner = output / "run_strength_sweep.py"
    if not runner.exists():
        shutil.copy2(Path(__file__).with_name("run_strength_sweep.py"), runner)
    identity = {"engine_sha256": file_digest(snapshot), "model_sha256": file_digest(args.model),
                "runner_sha256": file_digest(runner),
                "controller_sha256": file_digest(Path(__file__)),
                "books": manifest["sha256"], "candidates": CANDIDATES,
                "budgets": {"screen": [args.screen_games, args.screen_ms],
                            "validate": [args.validate_games, args.validate_ms],
                            "holdout": [args.holdout_games, args.holdout_ms]}, "seed": args.seed}
    state_path = output / "campaign.json"
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if previous["identity"] != identity:
            raise RuntimeError("campaign identity changed; use a new output directory")
    state = {"identity": identity, "status": "running", "results": []}

    def run(stage: str, name: str) -> dict:
        if file_digest(args.model) != identity["model_sha256"]:
            raise RuntimeError("model changed during campaign")
        games, milliseconds = identity["budgets"][stage]
        state["status"] = f"running {stage}: {name} ({games} games, {milliseconds} ms/move)"
        write_json(state_path, state)
        write_report(output, state)
        directory = output / stage / name
        directory.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(runner), "--engine", str(snapshot),
                   "--model", str(args.model.resolve()), "--output", str(directory),
                   "--positions", str(books / f"{stage}.tsv"), "--games", str(games),
                   "--movetime", str(milliseconds), "--seed", str(args.seed + ["screen", "validate", "holdout"].index(stage)),
                   "--dirichlet-epsilon", "0", "--resume"]
        for key, value in {"batch-size": 8, "threads": 4, **CANDIDATES[name]}.items():
            command += ["--axis", f"{key}={value}"]
        for key, value in {"batch-size": 8, "threads": 4}.items():
            command += ["--baseline", f"{key}={value}"]
        print(state["status"], flush=True)
        with (directory / "run.log").open("a") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((directory / "sweep.json").read_text())["runs"][0]
        state["results"].append({"stage": stage, "name": name, "summary": summary})
        print(f"{stage} {name}: {summary['contender_score']:.1%}, Elo {summary['contender_elo']}", flush=True)
        write_json(state_path, state)
        write_report(output, state)
        return summary

    try:
        screening = [(name, run("screen", name)) for name in CANDIDATES]
        ranked = sorted((row for row in screening if row[0] != "control"),
                        key=lambda row: row[1]["contender_score"], reverse=True)
        validation = [(name, run("validate", name)) for name, _ in ranked[:3]]
        name, result = max(validation, key=lambda row: row[1]["contender_score"])
        final = run("holdout", name)
        low, high = final["score_confidence_95"]
        if low > .5:
            state["conclusion"] = (f"Held-out evidence supports {name} at {args.holdout_ms} ms/move: "
                                   f"{final['contender_elo']:+.1f} Elo. Review results before changing defaults.")
        elif high < .5:
            state["conclusion"] = f"Held-out testing finds {name} weaker than the baseline; retain current defaults."
        else:
            state["conclusion"] = f"No statistically established gain for {name}; retain current defaults."
        state["status"] = "complete"
    except Exception as error:
        state["status"] = "failed"
        state["error"] = str(error)
        write_json(state_path, state)
        write_report(output, state)
        raise
    write_json(state_path, state)
    write_report(output, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
