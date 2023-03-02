import unittest
import chess
from utils.bugboard import BughouseBoard
from utils.board2planes import board2planes


class TestSum(unittest.TestCase):

    def test_sum(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")

    def test_sum_tuple(self):
        self.assertEqual(sum((1, 2, 2)), 6, "Should be 6")

    def test_planes(self):
        board = BughouseBoard()
        planes = board2planes(board, 0)
        print(planes)


if __name__ == '__main__':
    unittest.main()
