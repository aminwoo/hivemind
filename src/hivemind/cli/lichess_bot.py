"""Play Lichess variants with one Hivemind ONNX model."""

import argparse
from contextlib import contextmanager
from functools import partial
import json
import logging
import os
from pathlib import Path
import signal
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

import chess
import chess.engine
import chess.variant

from hivemind.paths import MODEL_DIRECTORY, PROJECT_ROOT

LOG = logging.getLogger(__name__)
VARIANTS = {
    "standard": chess.Board,
    "crazyhouse": chess.variant.CrazyhouseBoard,
    "antichess": chess.variant.AntichessBoard,
    "chess960": partial(chess.Board, chess960=True),
    "atomic": chess.variant.AtomicBoard,
    "threeCheck": chess.variant.ThreeCheckBoard,
}
ENGINE_VARIANTS = {"chess", "crazyhouse", "antichess", "atomic", "3check", "bughouse"}


class LichessAPI:
    def __init__(self, token, stop=None):
        self._token = token
        self.stop = stop or threading.Event()

    @contextmanager
    def response(self, path, data=None):
        request = Request(
            "https://lichess.org" + path,
            data=urlencode(data).encode() if data is not None else None,
            headers={"Authorization": "Bearer " + self._token,
                     "Accept": "application/json", "User-Agent": "HivemindChess/0.1"},
        )
        # Every request honors Lichess's one-minute pause after a 429.
        for attempt in range(5):
            try:
                response = urlopen(request, timeout=35)
                break
            except HTTPError as error:
                if error.code != 429 or attempt == 4:
                    raise
                LOG.warning("Lichess rate limit; waiting one minute")
                if self.stop.wait(60):
                    raise RuntimeError("Bot stopped during rate-limit wait") from None
        try:
            yield response
        finally:
            response.close()

    def get(self, path):
        with self.response(path) as response:
            return json.load(response)

    def post(self, path, data=None):
        with self.response(path, data or {}) as response:
            return json.load(response)

    def stream(self, path):
        with self.response(path) as response:
            for line in response:
                if self.stop.is_set():
                    break
                if line.strip():
                    yield json.loads(line)


def challenge_decline_reason(challenge, busy=False):
    if challenge.get("variant", {}).get("key") not in VARIANTS:
        return "variant"
    if challenge.get("speed") == "ultraBullet":
        return "tooFast"
    if busy:
        return "later"
    return None


def board_from_state(full, state):
    variant = full["variant"]["key"]
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported Lichess variant: {variant}")
    initial = full.get("initialFen", "startpos")
    board = VARIANTS[variant]() if initial == "startpos" else VARIANTS[variant](initial)
    for move in state.get("moves", "").split():
        board.push_uci(move)
    return board


def move_budget(state, color, maximum_ms):
    remaining = state.get("wtime" if color == chess.WHITE else "btime")
    increment = state.get("winc" if color == chess.WHITE else "binc", 0)
    if remaining is None:
        return maximum_ms / 1000
    # Reserve time for the move request as well as future turns.
    available = max(1, remaining - min(1000, remaining // 4))
    return max(0.001, min(maximum_ms, available, remaining / 40 + increment / 2) / 1000)


class Bot:
    def __init__(self, api, account, engine_command, move_time_ms=500):
        self.api = api
        self.account = account
        self.engine_command = engine_command
        self.move_time_ms = move_time_ms
        self.stop = api.stop
        self.lock = threading.Lock()
        self.games = {}
        self.pending = set()

    def challenge(self, challenge):
        if challenge.get("challenger", {}).get("id", "").lower() == self.account.lower():
            return  # Outgoing challenges are also sent on the event stream.
        challenge_id = challenge["id"]
        with self.lock:
            reason = challenge_decline_reason(challenge, bool(self.games or self.pending))
            if reason is None:
                self.pending.add(challenge_id)
        if reason:
            self.api.post(f"/api/challenge/{quote(challenge_id)}/decline", {"reason": reason})
            LOG.info("Declined challenge %s: %s", challenge_id, reason)
            return
        try:
            self.api.post(f"/api/challenge/{quote(challenge_id)}/accept")
            LOG.info("Accepted %s challenge %s", challenge["variant"]["key"], challenge_id)
        except Exception:
            with self.lock:
                self.pending.discard(challenge_id)
            raise

    def start_game(self, game):
        game_id = game["id"]
        with self.lock:
            self.pending.discard(game_id)
            if game_id in self.games:
                return
            worker = threading.Thread(target=self.play_game, args=(game_id,), daemon=True)
            self.games[game_id] = worker
        worker.start()

    def play_game(self, game_id):
        full = None
        submitted = None
        try:
            with chess.engine.SimpleEngine.popen_uci(self.engine_command, timeout=60) as engine:
                engine.configure({"DrawContemptPermille": 0})
                delay = 1
                while not self.stop.is_set():
                    try:
                        for event in self.api.stream(f"/api/bot/game/stream/{quote(game_id)}"):
                            delay = 1
                            if event["type"] == "gameFull":
                                full = event
                                state = event["state"]
                                LOG.info("Playing %s: https://lichess.org/%s", full["variant"]["key"], game_id)
                            elif event["type"] == "gameState" and full:
                                state = event
                            else:
                                continue
                            if state.get("status") not in ("created", "started"):
                                LOG.info("Game %s ended: %s (%s)", game_id,
                                         state.get("status"), state.get("winner", "draw"))
                                return
                            board = board_from_state(full, state)
                            color = chess.WHITE if full["white"].get("id", "").lower() == self.account.lower() else chess.BLACK
                            moves = state.get("moves", "")
                            if board.turn != color or moves == submitted or board.is_game_over():
                                continue
                            result = engine.play(board, chess.engine.Limit(
                                time=move_budget(state, color, self.move_time_ms)), game=game_id)
                            if result.move is None or result.move not in board.legal_moves:
                                raise RuntimeError("Engine did not return a legal move")
                            move_uci = board.uci(result.move, chess960=True)
                            self.api.post(f"/api/bot/game/{quote(game_id)}/move/{quote(move_uci)}")
                            submitted = moves
                            LOG.info("Game %s ply %d: %s", game_id, len(board.move_stack) + 1, result.move.uci())
                    except (HTTPError, URLError, TimeoutError, OSError) as error:
                        if isinstance(error, HTTPError) and error.code == 404:
                            return
                        LOG.warning("Game %s connection interrupted (%s); reconnecting", game_id, type(error).__name__)
                    if self.stop.wait(delay):
                        return
                    delay = min(30, delay * 2)
        except Exception as error:
            LOG.error("Game %s stopped: %s", game_id, error)
        finally:
            with self.lock:
                self.games.pop(game_id, None)

    def run(self):
        delay = 1
        while not self.stop.is_set():
            try:
                for event in self.api.stream("/api/stream/event"):
                    delay = 1
                    try:
                        if event["type"] == "challenge":
                            self.challenge(event["challenge"])
                        elif event["type"] == "gameStart":
                            self.start_game(event["game"])
                        elif event["type"] in ("challengeCanceled", "challengeDeclined"):
                            with self.lock:
                                self.pending.discard(event["challenge"]["id"])
                    except (HTTPError, URLError, TimeoutError, OSError) as error:
                        LOG.warning("Event request failed: %s", type(error).__name__)
            except (HTTPError, URLError, TimeoutError, OSError) as error:
                LOG.warning("Event stream interrupted (%s); reconnecting", type(error).__name__)
            if self.stop.wait(delay):
                break
            delay = min(30, delay * 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=PROJECT_ROOT / "engine/build-ort/hivemind.bin")
    parser.add_argument("--model", type=Path, default=MODEL_DIRECTORY / "twin-s-noattn.onnx")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--move-time-ms", type=int, default=500)
    parser.add_argument("--token-file", type=Path, help="Read a token from a private file; otherwise use LICHESS_TOKEN")
    parser.add_argument("--run-seconds", type=int, help="Stop listening after this many seconds")
    parser.add_argument("--check", action="store_true", help="Check BOT authentication and engine variants, then exit")
    args = parser.parse_args()
    if args.move_time_ms <= 0 or not 1 <= args.batch_size <= 1024:
        parser.error("Use a positive move time and a batch size from 1 to 1024")
    token = args.token_file.read_text().strip() if args.token_file else os.environ.get("LICHESS_TOKEN", "").strip()
    if not token:
        parser.error("Set LICHESS_TOKEN or provide --token-file")
    if not args.engine.is_file() or not args.model.is_file():
        parser.error("The engine executable and ONNX model must exist")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stop = threading.Event()
    api = LichessAPI(token, stop)
    account = api.get("/api/account")
    if account.get("title") != "BOT":
        parser.error(f"{account['username']} is not a BOT account; upgrading must be done explicitly")
    command = [str(args.engine.resolve()), "--model", str(args.model.resolve()), "--batch-size", str(args.batch_size)]
    # Check protocol support before accepting any challenge.
    with chess.engine.SimpleEngine.popen_uci(command, timeout=60) as engine:
        option = engine.options.get("UCI_Variant")
        if not option or not ENGINE_VARIANTS.issubset(option.var) or "UCI_Chess960" not in engine.options:
            parser.error("The engine must support all configured variants and UCI_Chess960")
        engine.configure({"DrawContemptPermille": 0})
    LOG.info("Authenticated BOT %s; model %s; accepting %s challenges; draw contempt 0",
             account["username"], args.model.name, ", ".join(VARIANTS))
    if args.check:
        return 0
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    timer = None
    if args.run_seconds:
        timer = threading.Timer(args.run_seconds, stop.set)
        timer.daemon = True
        timer.start()
    try:
        Bot(api, account["id"], command, args.move_time_ms).run()
    finally:
        if timer:
            timer.cancel()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
