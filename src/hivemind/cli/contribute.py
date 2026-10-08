#!/usr/bin/env python3
"""Play self-play games for an RL loop (`hivemind rl-loop --distributed`) and
submit them, forever. Each batch reads the loop's work order, downloads the
generator it names, plays --games games with the loop's settings, and submits
them for the order's iteration. A batch that arrives after the iteration has
its games still counts, as replay data for the next iterations. Batches that
could not be submitted are kept and retried.

On Hugging Face (the default exchange) each batch is a pull request, so any
account can contribute once logged in with `hf auth login`.

The engine must play with one search worker per game, like the loop:
  NVIDIA (TensorRT):  cmake -S engine -B engine/build-sp1 -G Ninja \\
                        -DCMAKE_BUILD_TYPE=Release -DHIVEMIND_SEARCH_WORKERS=1
                      cmake --build engine/build-sp1 --target hivemind
  macOS (Core ML):    cmake -S engine -B engine/build-ort-sp1 -G Ninja \\
                        -DHIVEMIND_BACKEND=onnxruntime -DCMAKE_BUILD_TYPE=Release \\
                        -DBUILD_TESTING=OFF -DHIVEMIND_SEARCH_WORKERS=1
                      cmake --build engine/build-ort-sp1 --target hivemind"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import platform
import re
import secrets
import shutil
import socket
import subprocess
import sys
import time

from hivemind.cli import selfplay, train
from hivemind.distill.exchange import PROTOCOL, HubExchange, now, open_exchange
from hivemind.network import SELFPLAY_EXCHANGE, TWIN_REPOSITORY
from hivemind.paths import PROJECT_ROOT

ENGINE_DIR = PROJECT_ROOT / "engine"
WORK_DIR = PROJECT_ROOT / "data" / "distill" / "contribute"
COREML_CONVERTER = ENGINE_DIR / "scripts" / "convert_onnx_coreml.py"
MAC_ENGINE = ENGINE_DIR / "build-ort-sp1" / "hivemind"
POLL_SECONDS = 60


def say(message):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}", flush=True)


def default_engine():
    return MAC_ENGINE if sys.platform == "darwin" else selfplay.DEFAULT_ENGINE


def search_workers(engine):
    """Search workers per game the engine was built with, from its CMake
    cache, or None when it has none (a release bundle)."""
    cache = Path(engine).parent / "CMakeCache.txt"
    match = cache.is_file() and re.search(r"^HIVEMIND_SEARCH_WORKERS:\w+=(\d+)$", cache.read_text(), re.M)
    return int(match.group(1)) if match else None


def for_this_machine(model):
    """On macOS, the network rewritten so Core ML runs all of it on the GPU
    (the outputs are identical)."""
    if sys.platform != "darwin":
        return model
    converted = model.with_name(f"{model.stem}-coreml{model.suffix}")
    if not converted.is_file():
        subprocess.run([sys.executable, str(COREML_CONVERTER), str(model), str(converted)], check=True)
    return converted


def contributor_name(exchange):
    name = socket.gethostname().split(".")[0]
    if isinstance(exchange, HubExchange):
        name = f"{exchange.api.whoami()['name']}-{name}"
    return re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-") or "contributor"


def play(engine, model, games, parallel_games, batch, work):
    """Play one batch into work/batch; returns that directory."""
    out, log = work / batch, work / f"{batch}.log"
    seed = int(batch.rsplit("-", 1)[1], 16)
    command = selfplay.engine_command(engine, model, games, seed, out, out / "games", parallel_games)
    count = 0
    with log.open("w") as stream:
        process = subprocess.Popen(command, cwd=ENGINE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True)
        for line in process.stdout:
            stream.write(line)
            if line.startswith("selfplay game "):
                count += 1
                if count % 100 == 0:
                    say(f"  {count} of {games} games")
        status = process.wait()
    if status or count != games:
        raise RuntimeError(f"self-play stopped after {count} of {games} games (see {log})")
    log.unlink()
    return out


def submit_pending(exchange, work):
    """Submit every finished batch in `work`; keep those that fail for later."""
    for path in sorted(work.glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        try:
            where = exchange.submit(manifest["folder"], path.parent, manifest)
        except Exception as error:
            say(f"submitting {manifest['folder']} failed, will retry: {error!r}")
            continue
        shutil.rmtree(path.parent)
        say(f"submitted {manifest['games']} games for iteration {manifest['iteration']}: {where}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exchange", default=SELFPLAY_EXCHANGE,
                        help="The loop's --exchange: hf://NAMESPACE/NAME or a shared directory")
    parser.add_argument("--games", type=int, default=1000, help="Games per submitted batch")
    parser.add_argument("--parallel-games", type=int, default=6)
    parser.add_argument("--engine", type=Path, default=default_engine())
    parser.add_argument("--name", help="Contributor name (default: Hugging Face account and host name)")
    parser.add_argument("--work-dir", type=Path, default=WORK_DIR)
    parser.add_argument("--once", action="store_true", help="Stop after one batch")
    args = parser.parse_args()
    if args.games < 1 or args.parallel_games < 1:
        parser.error("--games and --parallel-games must be positive")

    engine = args.engine.resolve()
    if not engine.is_file():
        sys.exit(f"Engine not found at {engine}; build it as `hivemind contribute --help` shows, "
                 f"or pass --engine.")
    workers = search_workers(engine)
    if workers not in (None, 1):
        sys.exit(f"{engine} plays with {workers} search workers per game and the loop with 1; build "
                 f"the engine as `hivemind contribute --help` shows.")
    exchange = open_exchange(args.exchange, TWIN_REPOSITORY)
    if isinstance(exchange, HubExchange):
        from huggingface_hub import get_token
        if get_token() is None:
            sys.exit("Log in with `hf auth login`: each batch is a pull request from your account.")
    name = args.name or contributor_name(exchange)
    work = args.work_dir.resolve()
    (work / "models").mkdir(parents=True, exist_ok=True)
    say(f"contributing to {exchange} as {name}, {args.games} games per batch")

    submit_pending(exchange, work)
    waiting = False
    while True:
        try:
            order = exchange.order()
        except Exception as error:
            say(f"reading the work order failed: {error!r}")
            time.sleep(POLL_SECONDS)
            continue
        if order is not None and order.get("protocol") != PROTOCOL:
            sys.exit(f"The loop plays protocol {order.get('protocol')} and this hivemind {PROTOCOL}: "
                     f"update hivemind.")
        if not (order and order.get("accepting")):
            if not waiting:
                say("the loop wants no games right now; waiting")
                waiting = True
            time.sleep(POLL_SECONDS)
            continue
        waiting = False
        n = order["iteration"]
        model = for_this_machine(exchange.generator(order, work / "models"))
        batch = f"{name}-{secrets.randbits(63) or 1:016x}"
        say(f"iteration {n}: playing {args.games} games ({batch})")
        out = play(engine, model, args.games, args.parallel_games, batch, work)
        manifest = {"protocol": PROTOCOL, "iteration": n, "generator": order["generator"]["sha256"],
                    "games": args.games, "positions": train.positions([out]),
                    "seed": int(batch.rsplit("-", 1)[1], 16), "contributor": name, "folder": f"it{n}/{batch}",
                    "platform": platform.platform(), "created": now()}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        submit_pending(exchange, work)
        if args.once:
            return


if __name__ == "__main__":
    main()
