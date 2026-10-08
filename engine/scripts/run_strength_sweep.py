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
    "pw-exponent": float,
    "pw-mass": float,
    "root-pw-mass": float,
    "pw-mass-exponent": float,
    "pw-mass-cap": float,
    "pw-mass-normalize": str,
    "cpuct-init": float,
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
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.0,
                        help="root noise per move; default off for strength tests")
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


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_flags(side: str, settings: dict[str, str]) -> list[str]:
    # The legacy coefficient flag also sets the root coefficient. Always
    # apply an explicit root override last, regardless of axis order.
    names = sorted(settings, key=lambda name: (name == "root-pw-coefficient", name))
    return [item for name in names for item in (f"--{side}-{name}", settings[name])]


def completed_run(run_dir: Path, identity: str) -> dict | None:
    marker = run_dir / "completed.json"
    summary = run_dir / "summary.json"
    if not marker.exists() or not summary.exists():
        return None
    try:
        saved = json.loads(marker.read_text())
        if saved.get("identity") != identity or saved.get("summary_sha256") != file_digest(summary):
            return None
        return json.loads(summary.read_text())
    except (OSError, ValueError, AttributeError):
        # A crash while writing an older checkpoint must restart that match.
        return None


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


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
    fingerprints = {"engine": file_digest(args.engine), "model": file_digest(args.model),
                    "baseline_model": file_digest(baseline_model),
                    "positions": file_digest(args.positions) if args.positions else None}
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
            "noise": [args.dirichlet_alpha, args.dirichlet_epsilon],
            "sha256": fingerprints,
        }, sort_keys=True)
        run_id = hashlib.sha256(identity.encode()).hexdigest()[:10]
        run_dir = args.output / run_id
        summary_path = run_dir / "summary.json"
        summary = completed_run(run_dir, identity) if args.resume else None
        if summary is None:
            command = [str(args.engine.resolve()), "tournament",
                       "--contender", str(args.model),
                       "--baseline", str(baseline_model),
                       "--games", str(args.games),
                       "--output", str(run_dir),
                       "--seed", str(args.seed),
                       "--dirichlet-alpha", str(args.dirichlet_alpha),
                       "--dirichlet-epsilon", str(args.dirichlet_epsilon)]
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
                command += parameter_flags(side, settings)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "invocation.json").write_text(json.dumps(
                {"identity": json.loads(identity), "command": command}, indent=2) + "\n")
            subprocess.run(command, check=True)
            summary = json.loads(summary_path.read_text())
            if summary["games"] != args.games and summary["sprt"]["decision"] == "continue":
                raise RuntimeError(f"incomplete tournament: {run_dir}")
            write_json(run_dir / "completed.json",
                       {"identity": identity, "summary_sha256": file_digest(summary_path)})
        summary["sweep_parameters"] = contender
        runs.append(summary)
        write_json(args.output / "sweep.json", {"axes": dict(axes), "runs": runs})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
