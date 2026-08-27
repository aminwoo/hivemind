#!/usr/bin/env python3
"""Run resumable paired Hivemind parameter sweeps.

Example:
  ./scripts/run_strength_sweep.py --engine ./build-ninja/hivemind.bin \
    --model ../models/net.onnx --positions positions.tsv --games 200 \
    --nodes 1600 --axis batch-size=8,16,32 --axis threads=1,2,4 \
    --sprt-elo0 0 --sprt-elo1 8 --output tournament_results/sweep
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
from pathlib import Path


PARAMETERS = {
    "batch-size": int,
    "threads": int,
    "mcgs": str,
    "transpositions": str,
    "root-mate-search": str,
    "wdl-eval": str,
    "pw-coefficient": float,
    "root-pw-coefficient": float,
    "wdl-weight": float,
    "moves-left-discount": float,
    "q-value-weight": float,
    "q-veto-delta": float,
}


def assignment(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("expected NAME=VALUE")
    name, value = text.split("=", 1)
    if name not in PARAMETERS:
        raise argparse.ArgumentTypeError(f"unknown parameter {name!r}")
    return name, value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path)
    parser.add_argument("--positions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=200)
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--nodes", type=int)
    budget.add_argument("--movetime", type=int)
    parser.add_argument("--axis", type=assignment, action="append", required=True,
                        help="repeatable Cartesian axis, e.g. threads=1,2,4")
    parser.add_argument("--baseline", type=assignment, action="append", default=[])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sprt-elo0", type=float, default=0.0)
    parser.add_argument("--sprt-elo1", type=float, default=0.0)
    parser.add_argument("--sprt-alpha", type=float, default=0.05)
    parser.add_argument("--sprt-beta", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def normalize(name: str, value: str) -> str:
    converter = PARAMETERS[name]
    if converter is str:
        lowered = value.lower()
        if lowered not in {"true", "false", "0", "1", "on", "off"}:
            raise ValueError(f"{name} expects a boolean, got {value!r}")
        return lowered
    return str(converter(value))


def main() -> int:
    args = parse_args()
    axes: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    for name, values in args.axis:
        if name in seen:
            raise SystemExit(f"duplicate axis: {name}")
        seen.add(name)
        axes.append((name, [normalize(name, item) for item in values.split(",")]))
    baseline = {name: normalize(name, value) for name, value in args.baseline}

    args.output.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, object]] = []
    baseline_model = args.baseline_model or args.model
    for values in itertools.product(*(values for _, values in axes)):
        contender = dict(zip((name for name, _ in axes), values))
        identity = json.dumps({
            "contender": contender,
            "baseline": baseline,
            "model": str(args.model.resolve()),
            "baseline_model": str(baseline_model.resolve()),
            "positions": str(args.positions.resolve()) if args.positions else None,
            "games": args.games,
            "nodes": args.nodes,
            "movetime": args.movetime,
            "seed": args.seed,
            "sprt": [args.sprt_elo0, args.sprt_elo1,
                     args.sprt_alpha, args.sprt_beta],
        }, sort_keys=True)
        run_id = hashlib.sha256(identity.encode()).hexdigest()[:10]
        run_dir = args.output / run_id
        summary_path = run_dir / "summary.json"
        if args.resume and summary_path.exists():
            runs.append(json.loads(summary_path.read_text()))
            continue

        command = [str(args.engine), "tournament",
                   "--contender", str(args.model),
                   "--baseline", str(baseline_model),
                   "--games", str(args.games),
                   "--output", str(run_dir),
                   "--seed", str(args.seed)]
        command += (["--nodes", str(args.nodes)] if args.nodes is not None
                    else ["--movetime", str(args.movetime)])
        if args.positions:
            command += ["--positions", str(args.positions)]
        if args.sprt_elo1 > args.sprt_elo0:
            command += ["--sprt-elo0", str(args.sprt_elo0),
                        "--sprt-elo1", str(args.sprt_elo1),
                        "--sprt-alpha", str(args.sprt_alpha),
                        "--sprt-beta", str(args.sprt_beta)]
        for side, settings in (("contender", contender), ("baseline", baseline)):
            for name, value in settings.items():
                command += [f"--{side}-{name}", value]

        subprocess.run(command, check=True)
        summary = json.loads(summary_path.read_text())
        summary["sweep_parameters"] = contender
        runs.append(summary)
        (args.output / "sweep.json").write_text(
            json.dumps({"axes": dict(axes), "runs": runs}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
