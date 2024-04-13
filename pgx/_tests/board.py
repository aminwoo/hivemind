import time 
from typing import List
import chess
from chess.variant import CrazyhouseBoard

class BughouseBoard(object):
    def __init__(self, time_control: int = 1800) -> None:
        self.boards = [CrazyhouseBoard(), CrazyhouseBoard()]
        self.times = [[time_control for _ in range(2)] for _ in range(2)]
        self.board_order = []
        self.move_history = []
        self.current_player = chess.BLACK 
        self.reset()

    def legal_moves(self) -> List[str]: 
        moves = [] 
        if self.boards[0].turn == self.current_player: 
            moves += ['0' + move.uci() for move in self.boards[0].legal_moves]
        if self.boards[1].turn != self.current_player: 
            moves += ['1' + move.uci() for move in self.boards[1].legal_moves]
        
        return moves

    def is_game_over(self):
        return self.boards[0].is_game_over() or self.boards[1].is_game_over()
    
    def is_checkmate(self): 
        return self.boards[0].is_checkmate() or self.boards[1].is_checkmate()

    def reset(self) -> None:
        colors = [chess.BLACK, chess.WHITE]
        for board in self.boards:
            board.set_fen(chess.STARTING_FEN)
            for color in colors:
                board.pockets[color].reset()

    def set_times(self, times: List[int]) -> None:
        self.times = times

    @classmethod
    def from_fen(cls, fen: str) -> 'BughouseBoard':
        board = cls() 
        board.set_fen(fen)
        return board

    def set_fen(self, fen: str) -> None:
        fen = fen.split("|")
        self.boards[0].set_fen(fen[0])
        self.boards[1].set_fen(fen[1])

    def fen(self) -> List[str]:
        return (
            self.boards[0].fen(),
            self.boards[1].fen(),
        )

    def turn(self, board_num: int) -> int:
        return self.boards[board_num].turn

    def swap_boards(self) -> None:
        self.boards = self.boards[::-1]
        self.times = self.times[::-1]

    def time_advantage(self, side: chess.Color) -> int:
        return self.times[0][side] - self.times[1][side]

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

    def push_san(self, board_num: int, move_str: str) -> None:
        move = self.parse_san(board_num, move_str)
        self.push(board_num, move)
    
    def push_uci(self, board_num: int, move_str: str) -> None:
        move = self.parse_uci(board_num, move_str)
        self.push(board_num, move)

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
    
    def parse_uci(self, board_num: int, move_uci: str) -> chess.Move: 
        return self.boards[board_num].parse_uci(move_uci) 
    
    def parse_san(self, board_num: int, move_san: str) -> chess.Move: 
        return self.boards[board_num].parse_san(move_san) 


