import numpy as np
import chess

from src.domain.board import BughouseBoard
from src.domain.board2planes import board2planes


def test_initial_pawn_placement():
    """Verify that the starting rank pawns appear in the correct channels."""
    board = BughouseBoard()  # Default starting position
    planes = board2planes(board, chess.WHITE)

    # White pawns on Board A should be in channel 0
    # On an 8x8 grid, row 1 (index 1) should be all 1s
    pawn_row = planes[0][1, :]
    assert np.all(pawn_row == 1)

    # Verify the empty space above
    assert np.all(planes[0][2:6, :] == 0)


def test_side_perspective_flip():
    """Verify that when playing as Black, the board is vertically inverted."""
    board = BughouseBoard()
    # As White, White pawns are at row 1.
    planes_white = board2planes(board, chess.WHITE)
    # As Black, Black pawns (originally row 6) should now be at row 1 for the CNN.
    planes_black = board2planes(board, chess.BLACK)

    # Channel 0 for White perspective (White Pawns)
    # Channel 0 for Black perspective (Black Pawns)
    assert np.all(planes_white[0][1, :] == 1)
    assert np.all(planes_black[0][1, :] == 1)


def test_pocket_count_normalization():
    board = BughouseBoard()
    board.boards[0].pockets[chess.WHITE].add(chess.KNIGHT)
    planes = board2planes(board, chess.WHITE)

    expected_val = 1.0 / 16.0
    assert np.all(planes[13] == expected_val)


def test_castling_rights_channels():
    board = BughouseBoard()
    planes = board2planes(board, chess.WHITE)

    # Verify initial rights (should be 1.0)
    assert np.all(planes[27:31] == 1.0)

    # Use the standard python-chess way to clear castling rights
    # We cast to a string FEN and remove the castling part or use set_castling_fen
    board.boards[0].set_castling_fen("-")

    planes_no_castle = board2planes(board, chess.WHITE)
    assert np.all(planes_no_castle[27:31] == 0.0)


def test_flip_augmentation_swap():
    """Verify the 'flip' parameter swaps the entire 32-channel blocks."""
    board = BughouseBoard()
    # Put a specific marker (e.g. Turn info) in Board A but not Board B
    planes_normal = board2planes(board, chess.WHITE, flip=False)
    planes_flipped = board2planes(board, chess.WHITE, flip=True)

    # The first 32 channels of 'flipped' should match the last 32 of 'normal'
    assert np.array_equal(planes_flipped[:32], planes_normal[32:])
    assert np.array_equal(planes_flipped[32:], planes_normal[:32])


def test_partner_board_piece_placement():
    board = BughouseBoard()
    planes = board2planes(board, chess.WHITE)
    offset = 32

    # Because (not side) is BLACK, Board B was flipped.
    # Black pawns move from index 6 to index 1.
    pawn_row_board_b = planes[offset + 0][1, :]
    assert np.all(pawn_row_board_b == 1)

def test_partner_pocket_flow():
    """Verify that the partner's pocket is tracked in Board B channels."""
    board = BughouseBoard()
    # If I am White, my partner is Black.
    # Give the partner (Black) a Knight on Board B.
    board.boards[1].pockets[chess.BLACK].add(chess.KNIGHT)

    planes = board2planes(board, chess.WHITE)
    offset = 32

    # Partner's friendly pocket knight is at offset + 13
    # 1/16 = 0.0625
    assert planes[offset + 13][0, 0] == 0.0625


def test_partner_castling_rights():
    board = BughouseBoard()
    assert board.boards[1].has_kingside_castling_rights(chess.BLACK) == True

    planes = board2planes(board, chess.WHITE)
    offset = 32

    assert np.all(planes[offset + 27] == 1.0)


def test_turn_indicator_per_board():
    """Verify that turn channels correctly identify whose move it is on which board."""
    board = BughouseBoard()
    # Default: Both boards turn = WHITE.
    # If side = WHITE:
    # Board A (White) turn is 1.0
    # Board B (Black) turn is 0.0 (because it's White's turn on Board B, not Black's)

    planes = board2planes(board, chess.WHITE)
    assert np.all(planes[25] == 1.0)  # My turn on Board A
    assert np.all(planes[32 + 25] == 0.0)  # Not my partner's turn on Board B