import chess
import chess.variant
import numpy as np
import sys
from matplotlib import pyplot as plt

from utils.constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    MAX_NUM_DROPS,
    MAX_TIME,
    NUM_BUGHOUSE_CHANNELS,
    BOARD_A, BOARD_B)
from utils.bugboard import BughouseBoard


def board2planes(board, side):
    planes = np.zeros((NUM_BUGHOUSE_CHANNELS, BOARD_HEIGHT, 2 * BOARD_WIDTH), dtype=float)
    boards = board.get_boards().copy()

    # castling
    planes[27][:,:8] = boards[BOARD_A].has_kingside_castling_rights(side)
    planes[27][:,8:] = boards[BOARD_B].has_kingside_castling_rights(not side)
    planes[28][:,:8] = boards[BOARD_A].has_queenside_castling_rights(side)
    planes[28][:,8:] = boards[BOARD_B].has_queenside_castling_rights(not side)
    planes[29][:,:8] = boards[BOARD_A].has_kingside_castling_rights(not side)
    planes[29][:,8:] = boards[BOARD_B].has_kingside_castling_rights(side)
    planes[30][:,:8] = boards[BOARD_A].has_queenside_castling_rights(not side)
    planes[30][:,8:] = boards[BOARD_B].has_queenside_castling_rights(side)

    # flip if black
    if side == chess.BLACK:
        boards[BOARD_A].apply_transform(chess.flip_vertical)
        #boards[BOARD_A].apply_transform(chess.flip_horizontal)

    if (not side) == chess.BLACK:
        boards[BOARD_B].apply_transform(chess.flip_vertical)
        #boards[BOARD_B].apply_transform(chess.flip_horizontal)

    pieces = chess.PIECE_TYPES
    # pieces
    c = 0
    for color in [side, not side]:
        for piece in pieces:
            a = np.zeros(BOARD_HEIGHT * BOARD_WIDTH)
            b = np.zeros(BOARD_HEIGHT * BOARD_WIDTH)
            a[list(boards[BOARD_A].pieces(piece, color))] = 1
            b[list(boards[BOARD_B].pieces(piece, not color))] = 1
            planes[c] = np.concatenate((a.reshape(BOARD_HEIGHT, BOARD_WIDTH), b.reshape(BOARD_HEIGHT, BOARD_WIDTH)),
                                       axis=1)
            c += 1

    # drops
    c = 12
    for color in [side, not side]:
        for piece in range(1, 6):
            a = boards[BOARD_A].pockets[color].count(piece) * np.ones(BOARD_HEIGHT * BOARD_WIDTH)
            b = boards[BOARD_B].pockets[not color].count(piece) * np.ones(BOARD_HEIGHT * BOARD_WIDTH)
            planes[c] = np.concatenate((a.reshape(BOARD_HEIGHT, BOARD_WIDTH), b.reshape(BOARD_HEIGHT, BOARD_WIDTH)),
                                       axis=1)
            c += 1
    planes[12:22] /= MAX_NUM_DROPS

    # promoted
    for color in [side, not side]:
        a = np.zeros(BOARD_HEIGHT * BOARD_WIDTH)
        b = np.zeros(BOARD_HEIGHT * BOARD_WIDTH)

        bb = boards[BOARD_A].occupied_co[color] & boards[BOARD_A].promoted
        a[list(boards[BOARD_A].piece_map(mask=bb).keys())] = 1

        bb = boards[BOARD_B].occupied_co[not color] & boards[BOARD_B].promoted
        b[list(boards[BOARD_B].piece_map(mask=bb).keys())] = 1

        planes[c] = np.concatenate((a.reshape(BOARD_HEIGHT, BOARD_WIDTH), b.reshape(BOARD_HEIGHT, BOARD_WIDTH)),
                                   axis=1)
        c += 1

    # en passant
    a = np.zeros(BOARD_HEIGHT * BOARD_WIDTH)
    b = np.zeros(BOARD_HEIGHT * BOARD_WIDTH)
    if boards[side].ep_square:
        a[boards[side].ep_square] = 1
    if boards[not side].ep_square:
        b[boards[not side].ep_square] = 1
    planes[24] = np.concatenate((a.reshape(BOARD_HEIGHT, BOARD_WIDTH), b.reshape(BOARD_HEIGHT, BOARD_WIDTH)),
                                axis=1)
    # turns
    planes[25] = boards[BOARD_A].turn == side
    planes[26] = boards[BOARD_B].turn == (not side)

    # time + scramble
    planes[31] = 0.5 + board.get_time_diff(side) / MAX_TIME
    planes[32] = board.is_scramble()

    if side == chess.BLACK:
        #boards[BOARD_A].apply_transform(chess.flip_horizontal)
        boards[BOARD_A].apply_transform(chess.flip_vertical)

    if (not side) == chess.BLACK:
        #boards[BOARD_B].apply_transform(chess.flip_horizontal)
        boards[BOARD_B].apply_transform(chess.flip_vertical)

    return planes


def visualize_planes(planes):
    h, w = 4, 3
    fig, ax = plt.subplots(h, w, figsize=(10, 10))

    titles = ["team 1 pawns", "team 1 knights", "team 1 bishops", "team 1 rooks", "team 1 queens", "team 1 kings",
              "team 2 pawns", "team 2 knights", "team 2 bishops", "team 2 rooks", "team 2 queens", "team 2 kings",
              ]

    for i in range(12):
        c = i % w
        r = i // w
        im = ax[r, c].imshow(planes[i], vmin=0, vmax=1, cmap="binary")
        ax[r, c].axvline(x=7.5, color='black', label='axvline - full height')
        ax[r, c].set_xticks([])
        ax[r, c].set_yticks([])
        ax[r, c].set_title(titles[i])
        ax[r, c].invert_yaxis()

    fig.tight_layout()
    fig.colorbar(im, ax=ax.ravel().tolist())
    plt.show()

    h, w = 2, 5
    fig, ax = plt.subplots(h, w, figsize=(10, 10))
    titles = ["team 1 pawns", "team 1 knights", "team 1 bishops", "team 1 rooks", "team 1 queens",
              "team 2 pawns", "team 2 knights", "team 2 bishops", "team 2 rooks", "team 2 queens",
              ]

    for i in range(10):
        c = i % w
        r = i // w
        im = ax[r, c].imshow(planes[12 + i], vmin=0, vmax=1, cmap="binary")
        ax[r, c].axvline(x=7.5, color='black', label='axvline - full height')
        ax[r, c].set_xticks([])
        ax[r, c].set_yticks([])
        ax[r, c].set_title(titles[i])
        ax[r, c].invert_yaxis()

    fig.tight_layout()
    fig.colorbar(im, ax=ax.ravel().tolist())
    plt.show()

    h, w = 2, 1
    fig, ax = plt.subplots(h, w, figsize=(10, 10))

    titles = ["team 1 promoted", "team 2 promoted"]

    for i in range(2):
        im = ax[i].imshow(planes[22 + i], vmin=0, vmax=1, cmap="binary")
        ax[i].axvline(x=7.5, color='black', label='axvline - full height')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title(titles[i])
        ax[i].invert_yaxis()

    fig.tight_layout()
    fig.colorbar(im, ax=ax.ravel().tolist())
    plt.show()

    h, w = 2, 2
    fig, ax = plt.subplots(h, w, figsize=(10, 10))

    titles = ["team 1 king", "team 1 queen", "team 2 king", "team 2 queen"]

    for i in range(4):
        c = i % w
        r = i // w
        im = ax[r, c].imshow(planes[27 + i], vmin=0, vmax=1, cmap="binary")
        ax[r, c].axvline(x=7.5, color='black', label='axvline - full height')
        ax[r, c].set_xticks([])
        ax[r, c].set_yticks([])
        ax[r, c].set_title(titles[i])
        ax[r, c].invert_yaxis()

    fig.tight_layout()
    fig.colorbar(im, ax=ax.ravel().tolist())
    plt.show()


if __name__ == "__main__":
    board = BughouseBoard()
    board.set_fen(
        'rnbqkbnr/pp1ppppp/8/8/1PpPP3/8/P1P2PPP/RNBQKBNR b KQkq b3 0 3 | r~3kb1r~/pppbqppp/4P3/3pN~3/3Pn3/8/PPP2PPP/R1BQK2R[] b Kq')
    board.set_time([[1089, 781], [155, 35]])
    boards = board.get_boards().copy()
    planes = board2planes(board, chess.WHITE)
    np.set_printoptions(threshold=sys.maxsize)
    print(planes)
    #visualize_planes(planes)
