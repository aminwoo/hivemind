import chess
import chess.variant
import numpy as np

from src.domain.board import BughouseBoard
from src.constants import (BOARD_A, BOARD_B, BOARD_HEIGHT, BOARD_WIDTH,
                           MAX_NUM_DROPS, NUM_BUGHOUSE_CHANNELS)


def board2planes(board: BughouseBoard, team_side: chess.Color, flip=False) -> np.ndarray:
    offset = NUM_BUGHOUSE_CHANNELS // 2
    planes = np.zeros((NUM_BUGHOUSE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH), dtype=float)

    boards = [board.boards[BOARD_A].copy(), board.boards[BOARD_B].copy()]

    # we evaluate castling rights BEFORE we apply transform since king position is used to determine castling rights
    planes[27][:, :] = float(boards[BOARD_A].has_kingside_castling_rights(team_side))
    planes[28][:, :] = float(boards[BOARD_A].has_queenside_castling_rights(team_side))
    planes[29][:, :] = float(boards[BOARD_A].has_kingside_castling_rights(not team_side))
    planes[30][:, :] = float(boards[BOARD_A].has_queenside_castling_rights(not team_side))

    planes[offset + 27][:, :] = float(boards[BOARD_B].has_kingside_castling_rights(not team_side))
    planes[offset + 28][:, :] = float(boards[BOARD_B].has_queenside_castling_rights(not team_side))
    planes[offset + 29][:, :] = float(boards[BOARD_B].has_kingside_castling_rights(team_side))
    planes[offset + 30][:, :] = float(boards[BOARD_B].has_queenside_castling_rights(team_side))

    # representation is always from the perspective of the team
    if team_side == chess.BLACK:
        boards[BOARD_A].apply_transform(chess.flip_vertical)
    else:
        boards[BOARD_B].apply_transform(chess.flip_vertical)

    # pieces
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    c = 0
    for color in [team_side, not team_side]:
        for pt in piece_types:
            for sq in boards[BOARD_A].pieces(pt, color):
                planes[c].flat[sq] = 1
            for sq in boards[BOARD_B].pieces(pt, not color):
                planes[offset + c].flat[sq] = 1
            c += 1

    # pocket pieces
    c = 12
    for color in [team_side, not team_side]:
        for pt in range(1, 6):
            count_a = board.boards[BOARD_A].pockets[color].count(pt)
            planes[c][:, :] = count_a / MAX_NUM_DROPS

            count_b = board.boards[BOARD_B].pockets[not color].count(pt)
            planes[offset + c][:, :] = count_b / MAX_NUM_DROPS
            c += 1

    # promoted pieces
    c = 22
    for i, color in enumerate([team_side, not team_side]):
        mask_a = boards[BOARD_A].occupied_co[color] & boards[BOARD_A].promoted
        for sq in chess.SquareSet(mask_a):
            planes[c].flat[sq] = 1

        mask_b = boards[BOARD_B].occupied_co[not color] & boards[BOARD_B].promoted
        for sq in chess.SquareSet(mask_b):
            planes[offset + c].flat[sq] = 1
        c += 1

    # en passant square
    if boards[BOARD_A].ep_square is not None:
        planes[24].flat[boards[BOARD_A].ep_square] = 1
    if boards[BOARD_B].ep_square is not None:
        planes[offset + 24].flat[boards[BOARD_B].ep_square] = 1

    # on turn
    planes[25][:, :] = 1.0 if board.boards[BOARD_A].turn == team_side else 0.0
    planes[offset + 25][:, :] = 1.0 if board.boards[BOARD_B].turn == (not team_side) else 0.0

    # constant plane
    planes[26][:, :] = 1.0
    planes[offset + 26][:, :] = 1.0

    # has time advantage (can sit)
    planes[31][:, :] = 1.0 if board.time_advantage(team_side) > 0 else 0.0
    planes[offset + 31][:, :] = 1.0 if board.time_advantage(team_side) > 0 else 0.0

    if flip:
        a_block = planes[:offset].copy()
        planes[:offset] = planes[offset:]
        planes[offset:] = a_block

    return planes
