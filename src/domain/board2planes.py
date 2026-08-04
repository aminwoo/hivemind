import chess
import chess.variant
import numpy as np

from src.domain.board import BughouseBoard
from src.constants import (BOARD_A, BOARD_B, BOARD_HEIGHT, BOARD_WIDTH,
                           MAX_NUM_DROPS, MAX_NUM_NO_PROGRESS,
                           NUM_BUGHOUSE_CHANNELS, NUM_BUGHOUSE_CHANNELS_PER_BOARD)


_PIECE_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
_SQUARE_FLIP_VERTICAL = np.arange(64, dtype=np.int16).reshape(8, 8)[::-1, :].reshape(-1)

_LAST_MOVE_FROM_CHANNEL = 32
_LAST_MOVE_TO_CHANNEL = 33
_HALFMOVE_CLOCK_CHANNEL = 34
_REPETITION_2_CHANNEL = 35
_REPETITION_3_CHANNEL = 36


def _set_plane_from_bitboard(plane: np.ndarray, bitboard: int, needs_vertical_flip: bool) -> None:
    while bitboard:
        lsb = bitboard & -bitboard
        sq = lsb.bit_length() - 1
        if needs_vertical_flip:
            sq = int(_SQUARE_FLIP_VERTICAL[sq])
        plane.flat[sq] = 1.0
        bitboard ^= lsb


def _orient_square(square: int, needs_vertical_flip: bool) -> int:
    if not needs_vertical_flip:
        return square
    return int(_SQUARE_FLIP_VERTICAL[square])


def _last_move_for_board(board: BughouseBoard, board_idx: int):
    for idx in range(len(board.board_order) - 1, -1, -1):
        if board.board_order[idx] == board_idx:
            return board.move_history[idx]
    return None


def _safe_repetition_flag(single_board, count: int) -> float:
    try:
        # `is_repetition` can traverse history via push/pop internally.
        # Run it on a copy so live bughouse board state is never mutated.
        board_copy = single_board.copy(stack=True)
        return float(board_copy.is_repetition(count))
    except Exception:
        # Some historical game records can violate pocket consistency during
        # deep repetition probes; do not fail sample generation for this.
        return 0.0


def board2planes(board: BughouseBoard, team_side: chess.Color, flip=False) -> np.ndarray:
    offset = NUM_BUGHOUSE_CHANNELS_PER_BOARD
    planes = np.zeros(
        (NUM_BUGHOUSE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32
    )

    board_a = board.boards[BOARD_A]
    board_b = board.boards[BOARD_B]
    flip_board_a = team_side == chess.BLACK
    flip_board_b = team_side == chess.WHITE

    # we evaluate castling rights BEFORE we apply transform since king position is used to determine castling rights
    planes[27][:, :] = float(board_a.has_kingside_castling_rights(team_side))
    planes[28][:, :] = float(board_a.has_queenside_castling_rights(team_side))
    planes[29][:, :] = float(board_a.has_kingside_castling_rights(not team_side))
    planes[30][:, :] = float(board_a.has_queenside_castling_rights(not team_side))

    planes[offset + 27][:, :] = float(board_b.has_kingside_castling_rights(not team_side))
    planes[offset + 28][:, :] = float(board_b.has_queenside_castling_rights(not team_side))
    planes[offset + 29][:, :] = float(board_b.has_kingside_castling_rights(team_side))
    planes[offset + 30][:, :] = float(board_b.has_queenside_castling_rights(team_side))

    # pieces
    c = 0
    for color in [team_side, not team_side]:
        for pt in _PIECE_TYPES:
            _set_plane_from_bitboard(
                planes[c],
                int(board_a.pieces_mask(pt, color)),
                flip_board_a,
            )
            _set_plane_from_bitboard(
                planes[offset + c],
                int(board_b.pieces_mask(pt, not color)),
                flip_board_b,
            )
            c += 1

    # pocket pieces
    c = 12
    for color in [team_side, not team_side]:
        for pt in range(1, 6):
            count_a = board_a.pockets[color].count(pt)
            planes[c][:, :] = count_a / MAX_NUM_DROPS

            count_b = board_b.pockets[not color].count(pt)
            planes[offset + c][:, :] = count_b / MAX_NUM_DROPS
            c += 1

    # promoted pieces
    c = 22
    for color in [team_side, not team_side]:
        mask_a = int(board_a.occupied_co[color] & board_a.promoted)
        _set_plane_from_bitboard(planes[c], mask_a, flip_board_a)

        mask_b = int(board_b.occupied_co[not color] & board_b.promoted)
        _set_plane_from_bitboard(planes[offset + c], mask_b, flip_board_b)
        c += 1

    # en passant square
    if board_a.ep_square is not None:
        sq = board_a.ep_square
        if flip_board_a:
            sq = int(_SQUARE_FLIP_VERTICAL[sq])
        planes[24].flat[sq] = 1.0
    if board_b.ep_square is not None:
        sq = board_b.ep_square
        if flip_board_b:
            sq = int(_SQUARE_FLIP_VERTICAL[sq])
        planes[offset + 24].flat[sq] = 1.0

    # on turn
    planes[25][:, :] = 1.0 if board_a.turn == team_side else 0.0
    planes[offset + 25][:, :] = 1.0 if board_b.turn == (not team_side) else 0.0

    # constant plane
    planes[26][:, :] = 1.0
    planes[offset + 26][:, :] = 1.0

    # has time advantage (can sit)
    has_time_advantage = 1.0 if board.time_advantage(team_side) > 0 else 0.0
    planes[31][:, :] = has_time_advantage
    planes[offset + 31][:, :] = has_time_advantage

    # last move planes
    last_move_a = _last_move_for_board(board, BOARD_A)
    if last_move_a is not None:
        if last_move_a.drop is None:
            sq = _orient_square(last_move_a.from_square, flip_board_a)
            planes[_LAST_MOVE_FROM_CHANNEL].flat[sq] = 1.0
        sq = _orient_square(last_move_a.to_square, flip_board_a)
        planes[_LAST_MOVE_TO_CHANNEL].flat[sq] = 1.0

    last_move_b = _last_move_for_board(board, BOARD_B)
    if last_move_b is not None:
        if last_move_b.drop is None:
            sq = _orient_square(last_move_b.from_square, flip_board_b)
            planes[offset + _LAST_MOVE_FROM_CHANNEL].flat[sq] = 1.0
        sq = _orient_square(last_move_b.to_square, flip_board_b)
        planes[offset + _LAST_MOVE_TO_CHANNEL].flat[sq] = 1.0

    # halfmove clock and repetition context
    planes[_HALFMOVE_CLOCK_CHANNEL][:, :] = min(board_a.halfmove_clock, MAX_NUM_NO_PROGRESS) / MAX_NUM_NO_PROGRESS
    planes[offset + _HALFMOVE_CLOCK_CHANNEL][:, :] = min(board_b.halfmove_clock, MAX_NUM_NO_PROGRESS) / MAX_NUM_NO_PROGRESS

    planes[_REPETITION_2_CHANNEL][:, :] = _safe_repetition_flag(board_a, 2)
    planes[offset + _REPETITION_2_CHANNEL][:, :] = _safe_repetition_flag(board_b, 2)
    planes[_REPETITION_3_CHANNEL][:, :] = _safe_repetition_flag(board_a, 3)
    planes[offset + _REPETITION_3_CHANNEL][:, :] = _safe_repetition_flag(board_b, 3)

    if flip:
        a_block = planes[:offset].copy()
        planes[:offset] = planes[offset:]
        planes[offset:] = a_block

    return planes
