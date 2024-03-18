import chess
from chess.variant import CrazyhouseBoard
from typing import List, Any


class BughouseBoard(object):
    def __init__(self, time_control: int = 1800) -> None:
        self.boards = [CrazyhouseBoard(), CrazyhouseBoard()]
        self.times = [[time_control for _ in range(2)] for _ in range(2)]
        self.board_order = []
        self.move_history = []
        self.reset()

    def reset(self) -> None:
        colors = [chess.BLACK, chess.WHITE]
        for board in self.boards:
            board.set_fen(chess.STARTING_FEN)
            for color in colors:
                board.pockets[color].reset()

    def set_time(self, times: List[int]) -> None:
        self.times = times

    def set_fen(self, fen: str) -> None:
        fen = fen.split(" | ")
        self.boards[0].set_fen(fen[0])
        self.boards[1].set_fen(fen[1])

    def get_fens(self) -> List[str]:
        return (
            self.boards[0].fen(),
            self.boards[1].fen(),
        )

    def get_turn(self, board_num: int) -> int:
        return self.boards[board_num].turn

    def swap_boards(self) -> None:
        self.boards = self.boards[::-1]
        self.times = self.times[::-1]

    def get_boards(self):
        return self.boards

    def get_times(self, board_num: int) -> List[int]:
        board = self.boards[board_num]
        other = self.boards[not board_num]
        return [
            self.times[board_num][board.turn],
            self.times[board_num][not board.turn],
            self.times[not board_num][other.turn],
            self.times[not board_num][not other.turn],
        ]

    def get_time_diff(self, side: Any) -> int:
        return self.times[0][not side] - self.times[1][not side]

    def update_time(self, board_num: int, time_left: int, move_time: int) -> None:
        board = self.boards[board_num]
        other = self.boards[not board_num]
        self.times[board_num][board.turn] = time_left
        self.times[not board_num][other.turn] -= move_time

    def push(self, board_num: int, move: chess.Move) -> None:
        board = self.boards[board_num]
        other = self.boards[not board_num]

        is_capture = False if move.drop else board.is_capture(move)
        captured = None
        if is_capture:
            captured = board.piece_type_at(move.to_square)
            if captured is None:
                captured = chess.PAWN
            is_promotion = board.promoted & (1 << move.to_square)
            if is_promotion:
                captured = chess.PAWN
            partner_pocket = other.pockets[not board.turn]
            partner_pocket.add(captured)

        board.push(move)
        if is_capture:
            opponent_pocket = board.pockets[not board.turn]
            opponent_pocket.remove(captured)

        self.move_history.append(move)
        self.board_order.append(board_num)

    def pop(self) -> None:
        last_move = self.move_history.pop()
        last_board = self.board_order.pop()

        board = self.boards[last_board]
        other = self.boards[not last_board]
        board.pop()

        if board.is_capture(last_move):
            captured = board.piece_type_at(last_move.to_square)
            if captured is None:
                captured = chess.PAWN
            is_promotion = board.promoted & (1 << last_move.to_square)
            if is_promotion:
                captured = chess.PAWN
            partner_pocket = other.pockets[not board.turn]
            partner_pocket.remove(captured)
