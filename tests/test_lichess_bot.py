import threading
from types import SimpleNamespace

import chess
import pytest

from hivemind.cli.lichess_bot import Bot, board_from_state, challenge_decline_reason, move_budget


@pytest.mark.parametrize("variant", ["standard", "crazyhouse", "antichess", "chess960", "atomic", "threeCheck"])
def test_supported_challenges_are_accepted(variant):
    assert challenge_decline_reason({"variant": {"key": variant}, "speed": "blitz"}) is None
    assert challenge_decline_reason({"variant": {"key": variant}}, busy=True) == "later"


def test_unsupported_challenges_and_ultrabullet_are_declined():
    assert challenge_decline_reason({"variant": {"key": "horde"}}) == "variant"
    assert challenge_decline_reason({"variant": {"key": "standard"}, "speed": "ultraBullet"}) == "tooFast"


def test_crazyhouse_history_preserves_capture_pockets_and_drops():
    full = {"variant": {"key": "crazyhouse"}, "initialFen": "startpos"}
    board = board_from_state(full, {"moves": "e2e4 d7d5 e4d5 g8f6"})
    assert board.pockets[chess.WHITE].count(chess.PAWN) == 1
    board = board_from_state(full, {"moves": "e2e4 d7d5 e4d5 g8f6 P@e4"})
    assert board.pockets[chess.WHITE].count(chess.PAWN) == 0
    assert board.piece_at(chess.E4) == chess.Piece(chess.PAWN, chess.WHITE)


def test_custom_starting_fen_and_standard_castling_history():
    full = {"variant": {"key": "standard"},
            "initialFen": "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"}
    board = board_from_state(full, {"moves": "e1h1 e8a8"})
    assert board.king(chess.WHITE) == chess.G1
    assert board.king(chess.BLACK) == chess.C8


def test_clock_budget_reserves_network_time_and_obeys_maximum():
    assert move_budget({"wtime": 300000, "winc": 2000}, chess.WHITE, 500) == 0.5
    assert move_budget({"btime": 100, "binc": 0}, chess.BLACK, 500) <= 0.003
    assert move_budget({}, chess.WHITE, 500) == 0.5


class FakeAPI:
    def __init__(self):
        self.stop = threading.Event()
        self.requests = []

    def post(self, path, data=None):
        self.requests.append((path, data))
        return {"ok": True}


def test_accept_handler_reserves_capacity_and_ignores_own_challenges():
    api = FakeAPI()
    bot = Bot(api, "hivemindchess", [])
    challenge = {"id": "abc", "variant": {"key": "crazyhouse"}, "challenger": {"id": "opponent"}}
    bot.challenge(challenge)
    assert api.requests == [("/api/challenge/abc/accept", None)]
    assert bot.pending == {"abc"}
    bot.challenge({**challenge, "id": "def", "variant": {"key": "standard"}})
    assert api.requests[-1] == ("/api/challenge/def/decline", {"reason": "later"})
    bot.challenge({**challenge, "challenger": {"id": "HivemindChess"}})
    assert len(api.requests) == 2


def test_failed_accept_does_not_reserve_capacity():
    api = FakeAPI()
    def fail(*_):
        raise OSError("disconnected")
    api.post = fail
    bot = Bot(api, "hivemindchess", [])
    with pytest.raises(OSError):
        bot.challenge({"id": "abc", "variant": {"key": "standard"}})
    assert not bot.pending


@pytest.mark.parametrize("variant", ["standard", "crazyhouse", "antichess", "chess960", "atomic", "threeCheck"])
def test_game_worker_plays_once_per_turn_and_handles_finished_games(monkeypatch, variant):
    api = FakeAPI()
    full = {"type": "gameFull", "variant": {"key": variant},
            "initialFen": "startpos", "white": {"id": "hivemindchess"},
            "black": {"id": "opponent"}, "state": {"status": "started", "moves": ""}}
    api.stream = lambda _: iter([
        full,
        {"type": "gameState", "status": "started", "moves": ""},
        {"type": "gameState", "status": "started", "moves": "e2e4"},
        {"type": "gameState", "status": "started", "moves": "e2e4 e7e5"},
        {"type": "gameState", "status": "resign", "moves": "e2e4 e7e5 g1f3"},
    ])
    calls = []
    class Engine:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def configure(self, options): assert options == {"DrawContemptPermille": 0}
        def play(self, board, limit, game):
            calls.append((board.uci_variant, limit.time, game))
            return SimpleNamespace(move=board.parse_uci("e2e4" if not board.move_stack else "g1f3"))
    monkeypatch.setattr("chess.engine.SimpleEngine.popen_uci", lambda *_, **__: Engine())
    bot = Bot(api, "hivemindchess", [])
    bot.games["abc"] = None
    bot.play_game("abc")
    assert len(calls) == 2
    uci_variant = {"standard": "chess", "chess960": "chess", "threeCheck": "3check"}.get(variant, variant)
    assert all(call[0] == uci_variant for call in calls)
    assert api.requests == [("/api/bot/game/abc/move/e2e4", None),
                            ("/api/bot/game/abc/move/g1f3", None)]
    assert not bot.games


def test_chess960_castling_uses_rook_square_and_preserves_history():
    full = {"variant": {"key": "chess960"}, "initialFen": "4k3/8/8/8/8/8/8/RK5R w HA - 0 1"}
    board = board_from_state(full, {"moves": "b1h1"})
    assert board.chess960
    assert board.king(chess.WHITE) == chess.G1
    assert board.piece_at(chess.F1) == chess.Piece(chess.ROOK, chess.WHITE)


def test_threecheck_history_and_lichess_fen_preserve_check_counts():
    full = {"variant": {"key": "threeCheck"}, "initialFen": "4k3/8/8/8/8/8/8/R3K3 w - - 0 1 +2+1"}
    board = board_from_state(full, {"moves": "a1a8"})
    assert board.remaining_checks[chess.WHITE] == 0
    assert board.remaining_checks[chess.BLACK] == 2
    assert board.is_variant_loss()
