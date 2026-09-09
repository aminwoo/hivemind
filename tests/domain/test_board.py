import copy


from hivemind.domain.board import BughouseBoard


def test_copy_has_independent_clock_pairs():
    board = BughouseBoard()
    board.set_times([[100, 200], [300, 400]])

    board_copy = copy.copy(board)
    board_copy.update_time(board_num=0, time_left=190, move_time=5)

    assert board_copy.times == [[100, 190], [300, 395]]
    assert board.times == [[100, 200], [300, 400]]
