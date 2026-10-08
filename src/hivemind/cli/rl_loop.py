#!/usr/bin/env python3
"""Improve twin-s by self-play, forever. Each iteration N:
  1. plays --games self-play games with the generator, in segments
  2. plays --val-games validation games
  3. trains from iteration N-1's last weights on the new games plus a replay
     buffer of the previous --window iterations
  4. matches the result against the generator; it becomes the generator unless
     it scores below 50%
  5. uploads every iteration's network and training checkpoint, and the
     generator with its checkpoint, to Hugging Face
     (skipped when not logged in to an account that can write to --repo)
  6. matches the result against the fixed pre-RL network (twin-s-noattn-it0)

Self-play and training use the settings of `hivemind selfplay` and `hivemind
train`. Every finished stage leaves a marker in RUNS/rl-itN/stages, so a rerun
resumes at the first unfinished one (an interrupted segment or match is moved
to rl-itN/aborted and redone); the layout is that of data/distill/rl_loop.sh,
whose runs this continues. Training and matches wait while a game is running
(see --pause-for), and a match restarts if one starts mid-way.

With --distributed, other machines share step 1: the loop publishes a work
order on --exchange, `hivemind contribute` plays games for it on any number of
machines, and the loop collects their batches (and keeps playing its own
segments, unless --no-local-selfplay) until the iteration has --games games.
Batches that arrive later still count, as replay data for later iterations.

Stop between stages with: touch RUNS/rl-loop.stop"""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from hivemind.cli import selfplay, train
from hivemind.distill.exchange import PROTOCOL, HubExchange, install, open_exchange, sha256
from hivemind.network import DEFAULT_ONNX_PATH, SELFPLAY_EXCHANGE, TWIN_REPOSITORY
from hivemind.paths import PROJECT_ROOT

RUNS = PROJECT_ROOT / "data" / "distill" / "runs"
# The 100 positions every RL match since iteration 2 opened from.
OPENINGS = PROJECT_ROOT / "engine" / "openings.tsv"
ENGINE_DIR = PROJECT_ROOT / "engine"
MATCH_ENGINE = ENGINE_DIR / "build-ninja" / "hivemind"
QUIET_SECONDS = 300
POLL_SECONDS = 20
COLLECT_SECONDS = 60
RETRY_SECONDS = 60
ATTEMPTS = 3


class StageFailed(Exception):
    pass


def next_iteration(runs):
    """The first iteration without stages/iteration.done (the pilot, rl-it1,
    has no stages and counts as finished once it has trained)."""
    n = 1
    while ((runs / f"rl-it{n}" / "stages" / "iteration.done").is_file()
           or (n == 1 and (runs / "rl-it1" / "arms" / "C" / "last.pt").is_file())):
        n += 1
    return n


def train_dirs(runs, n):
    """Self-play training data directories of iteration n, oldest layout first
    (the pilot wrote its chunks straight into search/train)."""
    it = runs / f"rl-it{n}"
    data = it / "search" / "train"
    dirs = [data] if any(data.glob("*.dst")) else []
    return dirs + [data / marker.stem for marker in sorted((it / "stages").glob("seg*.done"))]


def replay_iterations(n, window):
    return list(range(n - 1, max(1, n - window) - 1, -1))


def match_summary(path, games):
    d = json.loads(Path(path).read_text())
    wins, losses, draws = d["contender_wins"], d["baseline_wins"], d["draws"]
    if not (all(type(k) is int and k >= 0 for k in (wins, losses, draws))
            and wins + losses + draws == games):
        raise StageFailed(f"incomplete match in {path}")
    elo = d["contender_elo"]
    if elo is not None and not math.isfinite(elo):
        raise StageFailed(f"invalid Elo in {path}")
    return dict(wins=wins, losses=losses, draws=draws, score=(wins + draws / 2) / games, elo=elo,
                ci=d["elo_confidence_95"],
                nps=[d["performance"]["contender"]["nps"], d["performance"]["baseline"]["nps"]])


def archive_iteration(path):
    return int(path.stem.rsplit("-it", 1)[1])


def iteration_checkpoints(runs):
    """(n, last.pt) of every iteration whose training finished."""
    found = []
    for it in runs.glob("rl-it*"):
        if not re.fullmatch(r"rl-it\d+", it.name):
            continue  # an aborted run moved aside, such as rl-it2-unsegmented-aborted
        n, last = int(it.name[len("rl-it"):]), it / "arms" / "C" / "last.pt"
        trained = (it / "stages" / "train.done").is_file() or (n == 1 and not (it / "stages").exists())
        if trained and last.is_file():
            found.append((n, last))
    return sorted(found)


def sync_hub(api, repo, generator, checkpoint, log, runs):
    """Upload, in one commit, what the Hub lacks: every iteration's network
    (NAME-itN.onnx) and training checkpoint (NAME-itN.pt), and the generator
    (NAME.onnx) and its checkpoint (NAME.pt) when they differ from the Hub's."""
    from huggingface_hub import CommitOperationAdd

    stem = generator.stem
    present = set(api.list_repo_files(repo))
    archives = sorted(generator.parent.glob(f"{stem}-it*.onnx"), key=archive_iteration)
    uploads = [(path, path.name) for path in archives if path.name not in present]
    uploads += [(path, f"{stem}-it{n}.pt") for n, path in iteration_checkpoints(runs)
                if f"{stem}-it{n}.pt" not in present]
    remote = {info.path: info.lfs.sha256 for info in api.get_paths_info(repo, [generator.name, f"{stem}.pt"])
              if info.lfs}
    local = sha256(generator)
    promoted = local != remote.get(generator.name)
    if promoted:
        uploads.append((generator, generator.name))
    if checkpoint.is_file() and sha256(checkpoint) != remote.get(f"{stem}.pt"):
        uploads.append((checkpoint, f"{stem}.pt"))
    if not uploads:
        return
    names = [name for _, name in uploads]
    message = f"Upload {', '.join(names)}"
    n = next((archive_iteration(p) for p in archives if sha256(p) == local), None)
    if promoted and n is not None:
        message = f"Upload {stem} from RL iteration {n}"
        marker = runs / f"rl-it{n}" / "stages" / "match_generator.done"
        if marker.is_file():
            m = json.loads(marker.read_text())
            elo = "" if m["elo"] is None else f" ({m['elo']:+.0f} Elo)"
            message += f": {m['wins']}-{m['losses']}-{m['draws']} vs the previous generator at 100 ms{elo}"
    commit = api.create_commit(
        repo_id=repo, commit_message=message,
        operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for path, name in uploads])
    log(f"uploaded {', '.join(names)} to {repo}: {message} ({commit.oid})")


def upload_blocker(api, repo, token):
    """Why this machine cannot upload to `repo`, or None if it can."""
    if token is None:
        return "not logged in to Hugging Face (run `hf auth login` to upload)"
    try:
        account = api.whoami()
    except Exception:
        return None  # offline for now: each upload retries
    owners = {account["name"], *(org["name"] for org in account.get("orgs", []))}
    if repo.split("/")[0] not in owners:
        return f"Hugging Face account {account['name']} cannot upload to {repo}"
    return None


class Loop:
    def __init__(self, args, generator=DEFAULT_ONNX_PATH, checkpoint=train.GENERATOR_CHECKPOINT,
                 teacher_val=train.TEACHER_VAL):
        self.args = args
        self.runs = args.runs
        self.generator = generator
        self.generator_sp1 = selfplay.one_worker_copy(generator)
        self.checkpoint = checkpoint
        self.fixed = generator.with_name(f"{generator.stem}-it0{generator.suffix}")
        self.teacher_val = [teacher_val] if teacher_val.is_dir() else []
        self.api = None
        self.exchange = None  # set for --distributed

    def log(self, message):
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}"
        print(line, flush=True)
        with (self.runs / "rl-loop.log").open("a") as stream:
            stream.write(line + "\n")

    def check_stop(self):
        if (self.runs / "rl-loop.stop").exists():
            self.log(f"stop requested ({self.runs / 'rl-loop.stop'}): exiting before the next stage")
            sys.exit(0)

    def gaming(self):
        return bool(self.args.pause_for) and subprocess.run(
            ["pgrep", "-x", self.args.pause_for], stdout=subprocess.DEVNULL).returncode == 0

    def wait_for_quiet(self):
        if not self.gaming():
            return
        self.log(f"waiting: a game is running ({self.args.pause_for})")
        quiet = 0
        while quiet < QUIET_SECONDS:
            quiet = 0 if self.gaming() else quiet + POLL_SECONDS
            time.sleep(POLL_SECONDS)
        self.log(f"no game for {QUIET_SECONDS}s: continuing")

    # Stage bookkeeping for the current iteration.
    def done(self, stage):
        return (self.it / "stages" / f"{stage}.done").is_file()

    def mark(self, stage, value):
        temporary = self.it / "stages" / f"{stage}.tmp"
        temporary.write_text(json.dumps(value) + "\n")
        temporary.replace(self.it / "stages" / f"{stage}.done")

    def aside(self, *paths):
        """Move leftovers of an unfinished stage out of the way."""
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        for path in paths:
            if path.exists():
                (self.it / "aborted").mkdir(exist_ok=True)
                rel = "_".join(path.relative_to(self.it).parts)
                path.rename(self.it / "aborted" / f"{rel}-{stamp}")

    def attempt(self, stage, *args):
        """Retry a stage (a crash, an out-of-memory error) before giving up."""
        for count in range(1, ATTEMPTS + 1):
            try:
                return stage(*args)
            except Exception as error:
                self.log(f"it{self.n}: {stage.__name__} failed: {error} (attempt {count} of {ATTEMPTS})")
                if count < ATTEMPTS:
                    time.sleep(RETRY_SECONDS)
        sys.exit(1)

    def selfplay(self, name, games, seed, data, records):
        if self.done(name):
            return
        self.check_stop()
        generator = (self.it / "stages" / "generator").read_text().strip()
        selfplay.refresh_copy(self.generator, self.generator_sp1)
        if sha256(self.generator_sp1) != generator:
            raise StageFailed(f"the generator changed during it{self.n}")
        log = self.it / "search" / "logs" / f"{name}.log"
        self.aside(data, records, log)
        log.parent.mkdir(parents=True, exist_ok=True)
        self.log(f"it{self.n} self-play {name}: {games} games, seed {seed}")
        command = selfplay.engine_command(self.args.selfplay_engine, self.generator_sp1, games, seed,
                                          data, records, self.args.parallel_games)
        with log.open("w") as stream:
            if subprocess.run(command, cwd=ENGINE_DIR, stdout=stream, stderr=subprocess.STDOUT).returncode:
                raise StageFailed(f"self-play {name} failed (see {log})")
        with log.open() as stream:
            count = sum(line.startswith("selfplay game ") for line in stream)
        if count != games:
            raise StageFailed(f"{name}: expected {games} games, got {count}")
        positions = train.positions([data])
        if not positions:
            raise StageFailed(f"{name}: empty corpus")
        self.mark(name, {"games": games, "seed": seed, "positions": positions, "generator": generator})

    def games_played(self):
        return sum(json.loads(marker.read_text())["games"] for marker in (self.it / "stages").glob("seg*.done"))

    def publish(self, accepting):
        order = {"protocol": PROTOCOL, "iteration": self.n, "accepting": accepting, "games": self.args.games}
        try:
            self.exchange.publish(order, self.generator)
        except Exception as error:
            self.log(f"it{self.n}: publishing the work order on {self.exchange} failed: {error!r}")
            return False
        self.log(f"it{self.n}: work order on {self.exchange} {'open' if accepting else 'closed'}")
        return True

    def collect(self):
        """Take in contributors' batches. A batch played with the generator of
        its iteration joins that iteration's games, even once it has trained;
        any other batch is closed unused."""
        for submission in self.exchange.submissions():
            manifest = submission.manifest
            n, tail = manifest.get("iteration"), submission.folder.rsplit("/", 1)[-1]
            stages = self.runs / f"rl-it{n}" / "stages"
            name = f"seg-{tail}"
            if (stages / f"{name}.done").is_file():  # collected before the batch could be closed
                self.exchange.finish(submission, True, "Collected.")
                continue
            if (type(n) is not int or not re.fullmatch(r"[A-Za-z0-9._-]+", tail)
                    or manifest.get("protocol") != PROTOCOL or not (stages / "generator").is_file()
                    or manifest.get("generator") != (stages / "generator").read_text().strip()):
                self.exchange.finish(submission, False, "Not played with the generator of its iteration.")
                self.log(f"closed {submission.folder} from {manifest.get('contributor')} unused: "
                         f"not played with the generator of its iteration")
                continue
            data = self.runs / f"rl-it{n}" / "search" / "train" / name
            shutil.rmtree(data, ignore_errors=True)
            self.exchange.fetch(submission, data)
            positions = train.positions([data])
            if not positions:
                shutil.rmtree(data)
                self.exchange.finish(submission, False, "No positions.")
                continue
            marker = {"games": manifest["games"], "positions": positions, "generator": manifest["generator"],
                      "contributor": manifest.get("contributor"), "folder": submission.folder}
            (stages / f"{name}.tmp").write_text(json.dumps(marker) + "\n")
            (stages / f"{name}.tmp").replace(stages / f"{name}.done")
            self.exchange.finish(submission, True, f"Collected for iteration {n}. Thank you!")
            self.log(f"it{n}: collected {manifest['games']} games ({positions} positions) from "
                     f"{manifest.get('contributor')}")

    def gather(self):
        try:
            self.collect()
        except Exception as error:  # the network: the next poll retries
            self.log(f"collecting games from {self.exchange} failed: {error!r}")

    def distributed_selfplay(self):
        """Self-play shared with contributors until --games games are in."""
        if self.done("selfplay"):
            return
        published, reported, k = False, None, 0
        while True:
            self.check_stop()
            self.gather()
            played = self.games_played()
            if played >= self.args.games:
                break
            if not published:
                published = self.publish(True)
            if self.args.local_selfplay:
                while self.done(f"seg{k:02d}"):
                    k += 1
                if k >= 999:
                    raise StageFailed("out of local self-play seeds")
                name = f"seg{k:02d}"
                self.attempt(self.selfplay, name, min(self.args.segment_games, self.args.games - played),
                             1000 * self.n + k, self.it / "search" / "train" / name,
                             self.it / "search" / "train_run" / name)
            else:
                if played != reported:
                    self.log(f"it{self.n}: {played} of {self.args.games} games in")
                    reported = played
                time.sleep(COLLECT_SECONDS)
        self.publish(False)
        self.mark("selfplay", {"games": played})
        self.log(f"it{self.n}: {played} self-play games in")

    def train(self):
        if self.done("train"):
            return
        self.check_stop()
        previous = self.runs / f"rl-it{self.n - 1}"
        init = (previous / "arms" / "C" / "last.pt" if previous.exists()
                else train.generator_checkpoint(self.generator, self.checkpoint))
        if not init.is_file():
            raise StageFailed(f"missing {init}")
        new = train_dirs(self.runs, self.n)
        old = [d for m in replay_iterations(self.n, self.args.window) for d in train_dirs(self.runs, m)]
        npos, opos = train.positions(new), train.positions(old)
        if not npos:
            raise StageFailed("no new self-play positions")
        steps, fraction = train.plan(npos, opos, self.args.new_passes, self.args.replay_passes,
                                     self.args.batch_size)
        out, log = self.arms / "C", self.arms / "C.log"
        self.aside(out, log)
        self.wait_for_quiet()
        self.log(f"it{self.n} training: {steps} steps from {init}, {npos} new positions "
                 f"x{self.args.new_passes}, {opos} replay positions x{self.args.replay_passes} "
                 f"(replay share {fraction}, {len(old)} dirs, batch {self.args.batch_size})")
        command = train.distill_command(init, new, [self.it / "search" / "val"], old, steps, fraction,
                                        self.args.batch_size, self.n, out, self.teacher_val)
        with log.open("w") as stream:
            if subprocess.run(command, cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT,
                              env={**os.environ, **train.TRAINING_ENV}).returncode:
                raise StageFailed(f"training failed (see {log})")
        onnx = next(out.glob("distill-*-v3.0.onnx"), None)
        if not (out / "last.pt").is_file() or onnx is None:
            raise StageFailed(f"training left no last.pt and ONNX in {out}")
        install(onnx, self.it / "models" / "C.onnx")
        install(onnx, self.generator.with_name(f"{self.generator.stem}-it{self.n}.onnx"))
        self.mark("train", {"steps": steps, "replay_fraction": fraction, "new_positions": npos,
                            "replay_positions": opos, "init": sha256(init),
                            "onnx": sha256(self.it / "models" / "C.onnx")})

    def match(self, name, baseline):
        if self.done(f"match_{name}"):
            return
        self.check_stop()
        out, log = self.arms / f"match_{name}", self.arms / f"match_{name}.log"
        games = self.args.match_games
        while True:
            self.aside(out, log)
            self.wait_for_quiet()
            self.log(f"it{self.n} match vs {name} ({baseline}, {games} games at 100 ms)")
            command = [str(self.args.match_engine), "tournament",
                       "--contender", str(self.it / "models" / "C.onnx"), "--baseline", str(baseline),
                       "--movetime", "100", "--games", str(games), "--seed", str(self.n),
                       "--positions", str(self.args.openings), "--output", str(out)]
            with log.open("w") as stream:
                process = subprocess.Popen(command, cwd=ENGINE_DIR, stdout=stream, stderr=subprocess.STDOUT)
                status = None
                while status is None:
                    try:
                        status = process.wait(timeout=POLL_SECONDS)
                    except subprocess.TimeoutExpired:
                        if self.gaming():
                            process.terminate()
                            process.wait()
                            break
            if status is None:
                self.log(f"it{self.n} match vs {name}: a game started, restarting the match")
                continue
            if status:
                raise StageFailed(f"match vs {name} failed (see {log})")
            break
        summary = match_summary(out / "summary.json", games)
        self.log(f"it{self.n} vs {name}: {json.dumps(summary)}")
        self.mark(f"match_{name}", summary)

    def promote(self):
        if self.done("promote"):
            return
        score = json.loads((self.it / "stages" / "match_generator.done").read_text())["score"]
        if score >= 0.5:
            install(self.it / "models" / "C.onnx", self.generator)
            install(self.it / "models" / "C.onnx", self.generator_sp1)
            install(self.arms / "C" / "last.pt", self.checkpoint)
            self.log(f"it{self.n} promoted (score {score}): it is the generator for it{self.n + 1}")
        else:
            self.log(f"it{self.n} not promoted (score {score}): the generator stays")
        self.mark("promote", {"promoted": score >= 0.5, "score": score})

    def upload(self):
        """Not fatal: the next iteration's upload retries whatever is missing."""
        if self.args.no_upload or self.done("upload"):
            return
        for count in range(1, ATTEMPTS + 1):
            try:
                sync_hub(self.api, self.args.repo, self.generator, self.checkpoint, self.log, self.runs)
                self.mark("upload", {"repo": self.args.repo, "generator": sha256(self.generator)})
                return
            except Exception as error:
                self.log(f"it{self.n}: upload to {self.args.repo} failed: {error!r} "
                         f"(attempt {count} of {ATTEMPTS})")
                if count < ATTEMPTS:
                    time.sleep(RETRY_SECONDS)
        self.log(f"it{self.n}: continuing without uploading")

    def iteration(self, n):
        self.n, self.it = n, self.runs / f"rl-it{n}"
        self.arms = self.it / "arms"
        if self.done("iteration"):
            return
        if self.it.exists() and not (self.it / "stages").is_dir():
            self.log(f"{self.it} exists but was not made by this loop")
            sys.exit(1)
        for directory in ("stages", "search", "arms", "models"):
            (self.it / directory).mkdir(parents=True, exist_ok=True)
        # The generator is fixed for a whole iteration's self-play.
        if not (self.it / "stages" / "generator").is_file():
            (self.it / "stages" / "generator").write_text(sha256(self.generator) + "\n")
        args = self.args
        if self.exchange:
            self.distributed_selfplay()
        else:
            for k in range(math.ceil(args.games / args.segment_games)):
                name = f"seg{k:02d}"
                self.attempt(self.selfplay, name, min(args.segment_games, args.games - k * args.segment_games),
                             1000 * n + k, self.it / "search" / "train" / name,
                             self.it / "search" / "train_run" / name)
        self.attempt(self.selfplay, "val", args.val_games, 1000 * n + 999, self.it / "search" / "val",
                     self.it / "search" / "val_run")
        if self.exchange:
            self.gather()
        self.attempt(self.train)
        self.attempt(self.match, "generator", self.generator)
        self.promote()
        self.upload()
        if self.fixed.is_file():
            self.attempt(self.match, "it0", self.fixed)
        else:
            self.log(f"it{n}: no {self.fixed.name}, skipping the match against it")
        self.mark("iteration", {})
        results = {name: (self.it / "stages" / f"match_{name}.done") for name in ("generator", "it0")}
        self.log(f"it{n} finished: " + ", ".join(
            f"vs {name} {path.read_text().strip()}" for name, path in results.items() if path.is_file()))

    def run(self, first, last):
        self.log(f"loop: iterations {first}..{last or 'forever'}, {self.args.games} games "
                 f"({self.args.segment_games} per segment), window {self.args.window}, "
                 f"passes new {self.args.new_passes} / replay {self.args.replay_passes}, "
                 f"upload {'off' if self.args.no_upload else self.args.repo}"
                 + (f", distributed via {self.exchange}" if self.exchange else ""))
        n = first
        while last is None or n <= last:
            self.iteration(n)
            n += 1
        self.log(f"loop finished at it{last}")


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def even_int(value):
    parsed = positive_int(value)
    if parsed % 2:
        raise argparse.ArgumentTypeError("must be even")
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--first", type=positive_int,
                        help="First iteration (default: the first unfinished one)")
    parser.add_argument("--last", type=positive_int, help="Last iteration (default: run forever)")
    parser.add_argument("--games", type=positive_int, default=100_000)
    parser.add_argument("--segment-games", type=positive_int,
                        help="Games per local self-play segment (default: 10000, or 1000 with --distributed)")
    parser.add_argument("--val-games", type=positive_int, default=1000)
    parser.add_argument("--window", type=positive_int, default=3,
                        help="Earlier iterations replayed in training")
    parser.add_argument("--new-passes", type=float, default=1.5)
    parser.add_argument("--replay-passes", type=float, default=0.5)
    parser.add_argument("--parallel-games", type=positive_int, default=6)
    parser.add_argument("--match-games", type=even_int, default=200,
                        help="Games per match, in colour-swapped pairs")
    parser.add_argument("--batch-size", type=positive_int, default=1024)
    parser.add_argument("--openings", type=Path, default=OPENINGS, help="Match opening positions (TSV)")
    parser.add_argument("--runs", type=Path, default=RUNS, help="Directory of the rl-itN runs")
    parser.add_argument("--repo", default=TWIN_REPOSITORY,
                        help="Hugging Face model repository; uploads are skipped when this machine "
                             "is not logged in to an account that can write to it")
    parser.add_argument("--no-upload", action="store_true", help="Do not upload to Hugging Face")
    parser.add_argument("--pause-for", default="reaper|wineserver",
                        help="pgrep -x pattern of processes (Steam and Proton games by default) that "
                             "pause training and matches; empty disables")
    parser.add_argument("--distributed", action="store_true",
                        help="Share self-play with machines running `hivemind contribute`")
    parser.add_argument("--exchange", default=SELFPLAY_EXCHANGE,
                        help="Where work orders and games are exchanged with --distributed: "
                             "hf://NAMESPACE/NAME (a Hugging Face dataset) or a shared directory")
    parser.add_argument("--no-local-selfplay", action="store_true",
                        help="With --distributed, leave self-play to contributors")
    parser.add_argument("--selfplay-engine", type=Path, default=selfplay.DEFAULT_ENGINE)
    parser.add_argument("--match-engine", type=Path, default=MATCH_ENGINE)
    args = parser.parse_args()
    if args.new_passes <= 0 or args.replay_passes < 0:
        parser.error("--new-passes must be positive and --replay-passes non-negative")
    if args.last is not None and args.first is not None and args.last < args.first:
        parser.error("--last is before --first")
    if args.segment_games is None:
        args.segment_games = 1000 if args.distributed else 10_000
    args.local_selfplay = not args.no_local_selfplay

    if not DEFAULT_ONNX_PATH.is_file():
        sys.exit(f"Generator not found at {DEFAULT_ONNX_PATH}; run `hivemind fetch-network`.")
    if not args.selfplay_engine.is_file():
        sys.exit(f"Self-play engine not found at {args.selfplay_engine}; build it with\n"
                 f"  {selfplay.BUILD_COMMAND}")
    if not args.match_engine.is_file():
        sys.exit(f"Match engine not found at {args.match_engine}; run `hivemind build-engine`.")
    if not args.openings.is_file():
        sys.exit(f"Openings not found at {args.openings}; pass --openings.")
    args.openings = args.openings.resolve()
    args.selfplay_engine = args.selfplay_engine.resolve()
    args.match_engine = args.match_engine.resolve()
    args.runs = args.runs.resolve()

    loop = Loop(args)
    skipped = None
    if not args.no_upload:
        from huggingface_hub import HfApi, get_token
        loop.api = HfApi()
        skipped = upload_blocker(loop.api, args.repo, get_token())
        args.no_upload = skipped is not None

    import fcntl
    args.runs.mkdir(parents=True, exist_ok=True)
    lock = open(args.runs / "rl-loop.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"Another RL loop is running on {args.runs}.")
    stop = args.runs / "rl-loop.stop"
    if stop.exists():
        stop.unlink()
        loop.log(f"cleared an earlier stop request ({stop})")
    if skipped:
        loop.log(f"not uploading: {skipped}")
    if args.distributed:
        loop.exchange = open_exchange(args.exchange, args.repo, loop.api)
        if isinstance(loop.exchange, HubExchange) and args.no_upload:
            sys.exit(f"--distributed on {args.exchange} needs uploads to Hugging Face: "
                     f"{skipped or 'drop --no-upload'}.")
        loop.exchange.create()
    loop.run(args.first or next_iteration(args.runs), args.last)


if __name__ == "__main__":
    main()
