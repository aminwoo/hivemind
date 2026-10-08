"""`hivemind infer` must rank only the actions the engine would consider."""
import chess
import numpy as np

from hivemind.cli import infer_from_fen as infer
from hivemind.domain.board import BughouseBoard

AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"


def _board(fen_a=chess.STARTING_FEN, fen_b=chess.STARTING_FEN):
    board = BughouseBoard()
    board.set_fen(f"{fen_a} | {fen_b}")
    return board


def test_team_moves_on_its_board_and_passes_on_the_other():
    board = _board()
    legal_a = infer.legal_policy_indices(board, 0, chess.WHITE)
    assert {infer.LABELS[i] for i in legal_a} == {
        move.uci() for move in board.boards[0].legal_moves}
    # White's partner plays black on board B, where white is to move.
    assert infer.legal_policy_indices(board, 1, chess.WHITE) == [infer.PASS_INDEX]


def test_black_moves_are_mirrored_to_the_side_to_move():
    legal = infer.legal_policy_indices(_board(fen_a=AFTER_E4), 0, chess.BLACK)
    assert infer.LABEL_INDEX["e2e4"] in legal  # ...e7e5
    assert infer.LABEL_INDEX["g1f3"] in legal  # ...Ng8f6
    assert infer.PASS_INDEX not in legal


def test_time_advantage_allows_sitting(monkeypatch):
    board = _board()
    monkeypatch.setattr(board, "time_advantage", lambda side: 1)
    assert infer.PASS_INDEX in infer.legal_policy_indices(board, 0, chess.WHITE)


def test_illegal_logits_never_reach_the_ranking():
    class Session:
        def get_inputs(self):
            return [type("Input", (), {"name": "data", "type": "tensor(float)"})()]

        def run(self, outputs, inputs):
            # Untrained illegal logits dominate, as they do for twin-s.
            policy = np.full((1, len(infer.LABELS)), 50.0, dtype=np.float32)
            policy[0, infer.LABEL_INDEX["e2e4"]] = 1.0
            return [np.zeros((1, 1)), policy, policy.copy()]

    results = infer.infer_from_fens(Session(), chess.STARTING_FEN, chess.STARTING_FEN,
                                    top_k=1)
    legal_a = infer.legal_policy_indices(_board(), 0, chess.WHITE)
    assert set(np.flatnonzero(results["policy_a"])) == set(legal_a)
    np.testing.assert_allclose(results["policy_a"].sum(), 1.0, rtol=1e-6)
    assert results["top_moves_a"][0][0] in {infer.LABELS[i] for i in legal_a}
    assert results["top_moves_b"] == [("pass", 1.0)]
