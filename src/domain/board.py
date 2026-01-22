from typing import List

import chess
from chess.variant import CrazyhouseBoard, CrazyhousePocket


class BughouseBoard(object):
    def __init__(self, time_control: int = 1800) -> None:
        self.boards = [CrazyhouseBoard(), CrazyhouseBoard()]
        self.times = [[time_control for _ in range(2)] for _ in range(2)]
        self.board_order = []
        self.move_history = []
        self.reset()

    def __copy__(self):
        ret = BughouseBoard(1800)

        fen = self.fen()
        ret.boards[0].set_fen(fen[0])
        ret.boards[1].set_fen(fen[1])
        ret.times = self.times.copy()
        return ret

    def result(self):
        if self.boards[0].is_checkmate() or self.boards[0].is_stalemate():
            if self.turn(0) == chess.WHITE:
                return '0-1'
            else:
                return '1-0'
        if self.boards[1].is_checkmate() or self.boards[1].is_stalemate():
            if self.turn(1) == chess.WHITE:
                return '1-0'
            else:
                return '0-1'

        return '1/2-1/2'

    def is_checkmate(self):
        return self.boards[0].is_checkmate() or self.boards[1].is_checkmate()

    def is_diagonal(self):
        return self.boards[0].turn == self.boards[1].turn

    def reset(self) -> None:
        colors = [chess.BLACK, chess.WHITE]
        for board in self.boards:
            board.set_fen(chess.STARTING_FEN)
            for color in colors:
                board.pockets[color].reset()

    def reset_board(self, board_num: int) -> None:
        self.boards[board_num].set_fen(chess.STARTING_FEN)

    def set_times(self, times: List[int]) -> None:
        self.times = times

    def set_fen(self, fen: str) -> None:
        fen = fen.split(" | ")
        self.boards[0].set_fen(fen[0])
        self.boards[1].set_fen(fen[1])

    def fen(self) -> tuple[str, str]:
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

    def push_san(self, board_num: int, move_str: str) -> str:
        move = self.parse_san(board_num, move_str)
        self.push(board_num, move)
        return move.uci()

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

    def to_san(self, board_num: int, move_uci: str) -> str:
        try:
            return self.boards[board_num].san(self.parse_uci(board_num, move_uci))
        except:
            return self.boards[board_num].san(self.parse_uci(board_num, move_uci + 'q'))

    def push_unchecked(self, board_num: int, move: chess.Move) -> None:
        """
        Pushes a move onto the board without verifying its legality.

        For a drop move (i.e. when move.drop is not None):
          - If the mover's pocket does not already contain the piece being dropped,
            the piece is first added to the pocket.
          - The piece is then removed from the mover's pocket to perform the drop.

        For a normal move:
          - The move is played as usual (which will handle captures, etc.).

        Finally, the move is played on the board via a call to self.push.

        The parameter `board_num` identifies which board instance is used,
        which is useful if multiple boards are maintained.
        """

        # If the move is a drop move, ensure the mover's pocket contains the piece.
        if move.drop is not None:
            # Retrieve the current player's pocket from the specified board.
            pocket = self.boards[board_num].pockets[self.turn(board_num)]
            # If the piece to be dropped is not available in the pocket, add it first.
            if pocket.count(move.drop) < 1:
                pocket.add(move.drop)

        # Play the move as normal (drop or standard move).
        self.push(board_num, move)

    def update_hand(self, board_num: int, new_hand: str) -> None:
        """
        Updates the pocket (hand) for a given board state based on a string input.

        The parameter `new_hand` is a string representing the pocket contents.
        Uppercase letters denote pieces for white and lowercase letters denote pieces
        for black. For example, if `new_hand` is "ppPP":
          - The white pocket should contain two pieces corresponding to 'P' (e.g., white pawns).
          - The black pocket should contain two pieces corresponding to 'p' (e.g., black pawns).

        This function:
          1. Retrieves the board instance using board_num.
          2. Separates the input symbols into those for white and black.
             (Note: here we convert uppercase symbols to lowercase for white to ensure
             proper indexing in CrazyhousePocket if it relies on lowercase representations.)
          3. Creates new CrazyhousePocket instances for each color.
          4. Assigns the newly created pockets to the corresponding sides on the board.
        """
        board = self.boards[board_num]

        # Extract symbols: uppercase symbols indicate white pieces, lowercase indicate black.
        white_symbols = "".join(symbol.lower() for symbol in new_hand if symbol.isupper())
        black_symbols = "".join(symbol for symbol in new_hand if symbol.islower())

        # Create new CrazyhousePocket instances using the extracted symbols.
        white_pocket = CrazyhousePocket(white_symbols)
        black_pocket = CrazyhousePocket(black_symbols)

        # Update the board's pockets for white and black.
        board.pockets[chess.WHITE] = white_pocket
        board.pockets[chess.BLACK] = black_pocket

    def get_hand(self, board_num: int) -> str:
        hand = ""
        symbols = ['', 'p', 'n', 'b', 'r', 'q', 'k']
        for piece in chess.PIECE_TYPES:
            hand += symbols[piece].upper() * self.boards[board_num].pockets[chess.WHITE].count(piece)
            hand += symbols[piece] * self.boards[board_num].pockets[chess.BLACK].count(piece)
        return hand

    def is_legal(self, board_num: int, move: str) -> bool:
        return chess.Move.from_uci(move) in self.boards[board_num].legal_moves


def clean_fen(extended_fen):
    # Split the FEN string by whitespace into its components.
    parts = extended_fen.split()

    # Expecting at least 4 parts: board, active color, castling, en passant.
    if len(parts) < 4:
        raise ValueError("FEN string does not contain enough fields.")

    board_field = parts[0]
    active_color = parts[1]
    castling = parts[2]
    en_passant = parts[3]

    # Remove any piece drop extension from the board field if present.
    if '[' in board_field:
        board_field = board_field.split('[')[0]

    # Reconstruct the FEN without the halfmove clock and fullmove number.
    clean = f"{board_field} {active_color} {castling} {en_passant}"
    return clean


if __name__ == "__main__":
    board = BughouseBoard()
    board.update_hand(0, "pQqq")
    print(board.is_legal(0, "P@e5"))
    # print(board.boards[0].pockets)
    # board.push_unchecked(0, chess.Move(chess.E4, chess.E4, None, chess.QUEEN))
    # board.reset_board(0)
    print(board.get_hand(0))
    print(board.fen()[0].split(" ")[-1])

    board.update_hand(0, "p")

    import json

    with open('chessbot/book.json', 'r') as file:
        book = json.load(file)
    move = book.get(clean_fen(board.fen()[0]), "Move not found for this position.")
    print(move)
