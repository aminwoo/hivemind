import chess
import chess.variant
import jax.numpy as jnp
import numpy as np
import sys
from matplotlib import pyplot as plt

from src.types import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    MAX_NUM_DROPS,
    MAX_TIME,
    NUM_BUGHOUSE_CHANNELS,
    BOARD_A,
    BOARD_B,
)
from src.domain.board import BughouseBoard


def board2planes(board: BughouseBoard, side: chess.Color) -> np.ndarray:
    """
    Converts a Bughouse board object into a planar representation for the neural net.

    Args:
        board: 
        side: The side of the left board of the team on turn 
    """
    planes = np.zeros(
        (NUM_BUGHOUSE_CHANNELS, BOARD_HEIGHT, 2 * BOARD_WIDTH), dtype=float
    )
    boards = board.boards.copy()

    # Castling
    planes[27][:, :8] = boards[BOARD_A].has_kingside_castling_rights(side)
    planes[27][:, 8:] = boards[BOARD_B].has_kingside_castling_rights(not side)
    planes[28][:, :8] = boards[BOARD_A].has_queenside_castling_rights(side)
    planes[28][:, 8:] = boards[BOARD_B].has_queenside_castling_rights(not side)
    planes[29][:, :8] = boards[BOARD_A].has_kingside_castling_rights(not side)
    planes[29][:, 8:] = boards[BOARD_B].has_kingside_castling_rights(side)
    planes[30][:, :8] = boards[BOARD_A].has_queenside_castling_rights(not side)
    planes[30][:, 8:] = boards[BOARD_B].has_queenside_castling_rights(side)

    # Flip if black
    if side == chess.BLACK:
        boards[BOARD_A].apply_transform(chess.flip_vertical)
        # boards[BOARD_A].apply_transform(chess.flip_horizontal)

    if (not side) == chess.BLACK:
        boards[BOARD_B].apply_transform(chess.flip_vertical)
        # boards[BOARD_B].apply_transform(chess.flip_horizontal)

    pieces = chess.PIECE_TYPES
    # Pieces
    c = 0
    for color in [side, not side]:
        for piece in pieces:
            a = np.zeros(BOARD_HEIGHT * BOARD_WIDTH)
            b = np.zeros(BOARD_HEIGHT * BOARD_WIDTH)
            a[list(boards[BOARD_A].pieces(piece, color))] = 1
            b[list(boards[BOARD_B].pieces(piece, not color))] = 1
            planes[c] = np.concatenate(
                (
                    a.reshape(BOARD_HEIGHT, BOARD_WIDTH),
                    b.reshape(BOARD_HEIGHT, BOARD_WIDTH),
                ),
                axis=1,
            )
            c += 1

    # Drops
    c = 12
    for color in [side, not side]:
        for piece in range(1, 6):
            a = boards[BOARD_A].pockets[color].count(piece) * np.ones(
                BOARD_HEIGHT * BOARD_WIDTH
            )
            b = boards[BOARD_B].pockets[not color].count(piece) * np.ones(
                BOARD_HEIGHT * BOARD_WIDTH
            )
            planes[c] = np.concatenate(
                (
                    a.reshape(BOARD_HEIGHT, BOARD_WIDTH),
                    b.reshape(BOARD_HEIGHT, BOARD_WIDTH),
                ),
                axis=1,
            )
            c += 1
    planes[12:22] /= MAX_NUM_DROPS

    # Promoted
    for color in [side, not side]:
        a = np.zeros(BOARD_HEIGHT * BOARD_WIDTH)
        b = np.zeros(BOARD_HEIGHT * BOARD_WIDTH)

        bb = boards[BOARD_A].occupied_co[color] & boards[BOARD_A].promoted
        a[list(boards[BOARD_A].piece_map(mask=bb).keys())] = 1

        bb = boards[BOARD_B].occupied_co[not color] & boards[BOARD_B].promoted
        b[list(boards[BOARD_B].piece_map(mask=bb).keys())] = 1

        planes[c] = np.concatenate(
            (
                a.reshape(BOARD_HEIGHT, BOARD_WIDTH),
                b.reshape(BOARD_HEIGHT, BOARD_WIDTH),
            ),
            axis=1,
        )
        c += 1

    # En passant
    a = np.zeros(BOARD_HEIGHT * BOARD_WIDTH)
    b = np.zeros(BOARD_HEIGHT * BOARD_WIDTH)
    if boards[side].ep_square:
        a[boards[side].ep_square] = 1
    if boards[not side].ep_square:
        b[boards[not side].ep_square] = 1
    planes[24] = np.concatenate(
        (a.reshape(BOARD_HEIGHT, BOARD_WIDTH), b.reshape(BOARD_HEIGHT, BOARD_WIDTH)),
        axis=1,
    )
    # Turns
    planes[25] = boards[BOARD_A].turn == side
    planes[26] = boards[BOARD_B].turn == (not side)

    # Time advantage
    planes[31] = 0.5 + board.time_advantage(side) / MAX_TIME

    if side == chess.BLACK:
        # boards[BOARD_A].apply_transform(chess.flip_horizontal)
        boards[BOARD_A].apply_transform(chess.flip_vertical)

    if (not side) == chess.BLACK:
        # boards[BOARD_B].apply_transform(chess.flip_horizontal)
        boards[BOARD_B].apply_transform(chess.flip_vertical)

    return planes.transpose(1, 2, 0)