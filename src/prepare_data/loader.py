import gzip
import json

import chess
import numpy as np
import torch
from torch.utils.data import Dataset

from src.domain.board import BughouseBoard
from src.domain.board2planes import board2planes
from src.domain.move2planes import mirrorMoveUCI
from src.types import (BOARD_A, BOARD_B, BOARD_HEIGHT, BOARD_WIDTH,
                       NUM_BUGHOUSE_CHANNELS, POLICY_LABELS)
from src.utils.tcn import tcn_decode


class JSONGameReader(object):

    def __init__(self, game):
        self.result = self.get_game_result(game)
        self.time_control = self.get_game_time_control(game)
        self.ratings = [
            [game["a"]["white"]["rating"], game["a"]["black"]["rating"]],
            [game["b"]["white"]["rating"], game["b"]["black"]["rating"]],
        ]
        moves = [
            self.get_board_moves(game["a"]),
            self.get_board_moves(game["b"]),
        ]
        times = [
            self.get_board_times(game["a"]),
            self.get_board_times(game["b"]),
        ]
        deltas = [
            self.get_board_deltas(times[0]),
            self.get_board_deltas(times[1]),
        ]

        move_order = self.get_naive_move_order(deltas)
        self.moves = self.verify_move_order(move_order, moves, times, deltas)

    def get_result(self, board_num, turn):
        if (board_num == BOARD_A and turn == chess.WHITE) or (
            board_num == BOARD_B and turn == chess.BLACK
        ):
            return self.result
        return -self.result

    def get_rating(self, board_num, color):
        return self.ratings[board_num][not color]

    def get_board_deltas(self, board_times):
        white_times = board_times[::2]
        black_times = board_times[1::2]
        white_times.insert(0, self.time_control)
        black_times.insert(0, self.time_control)
        white_deltas = [a - b for a, b in zip(white_times[:-1], white_times[1:])]
        black_deltas = [a - b for a, b in zip(black_times[:-1], black_times[1:])]
        board_deltas = {
            chess.WHITE: white_deltas,
            chess.BLACK: black_deltas,
        }
        return board_deltas
    
    @staticmethod
    def get_naive_move_order(deltas):
        # Get time_deltas
        a_deltas_w = np.array(deltas[BOARD_A][chess.WHITE])
        a_deltas_b = np.array(deltas[BOARD_A][chess.BLACK])
        b_deltas_w = np.array(deltas[BOARD_B][chess.WHITE])
        b_deltas_b = np.array(deltas[BOARD_B][chess.BLACK])

        # Interleave player time deltas
        a_deltas = np.empty(a_deltas_w.size + a_deltas_b.size)
        a_deltas[0::2] = a_deltas_w
        a_deltas[1::2] = a_deltas_b
        b_deltas = np.empty(b_deltas_w.size + b_deltas_b.size)
        b_deltas[0::2] = b_deltas_w
        b_deltas[1::2] = b_deltas_b

        # Get accumulated player times
        a_times = np.cumsum(a_deltas)
        b_times = np.cumsum(b_deltas)

        all_times = np.concatenate((a_times, b_times))
        all_indices = np.argsort(all_times)
        move_order = np.digitize(all_indices, [0, a_times.shape[0]]) - 1
        return move_order

    @staticmethod
    def get_game_result(game):
        winner_color = game["a"].get("winner")
        if winner_color == "white":
            return 1
        elif winner_color == "black":
            return -1
        else:
            return 0

    @staticmethod
    def get_board_moves(board):
        tcn_moves = board["tcn"]
        board_moves = tcn_decode(tcn_moves)
        return board_moves

    @staticmethod
    def get_board_times(board):
        board_times = [int(t) for t in board["time_stamps"].split(",")]
        return board_times

    @staticmethod
    def get_game_time_control(game):
        try:
            time_control = int(game["a"]["time_control"]) * 10
        except ValueError:
            time_control = 9999

        return time_control

    @staticmethod
    def verify_move_order(move_order, moves, times, deltas):
        board = BughouseBoard()
        new_moves = []

        q = [i for i in range(len(move_order))]
        while q:
            stuck_board = -1
            for i in q:
                board_num = move_order[i]
                if moves[board_num]:
                    move = moves[board_num][0]
                else:
                    q.remove(i)
                    break
                pockets = board.boards[board_num].pockets
                turn = board.boards[board_num].turn
                time_left = times[board_num][0]
                move_time = deltas[board_num][turn][0]
                if board_num == stuck_board or (
                    move.drop and pockets[turn].count(move.drop) <= 0
                ):
                    stuck_board = board_num
                    continue
                else:
                    new_moves.append((board_num, move, time_left, move_time))
                    board.push(board_num, move)
                    moves[board_num].pop(0)
                    times[board_num].pop(0)
                    deltas[board_num][turn].pop(0)
                    q.remove(i)
                    break

        return new_moves


class BughouseDataset(Dataset):

    def __init__(self, path):
        self.path = path
        self.games = []
        self.board_data = []
        self.move_data = []
        self.result_data = []
        self.num_samples = 0
        self.game_idx = 0

    def load(self):
        with gzip.open(self.path) as f:
            self.games = json.load(f)

    def reset(self):
        self.num_samples = 0
        self.game_idx = 0

    def load_chunk(self, chunk_size=204800):
        if not self.games:
            self.load()

        self.board_data = []
        self.move_data = []
        self.result_data = []

        while len(self.board_data) < chunk_size and self.game_idx < len(self.games):
            JSONGameReader = JSONGameReader(self.games[self.game_idx])
            self.game_idx += 1
            if JSONGameReader.time_control not in [1800, 1200]:
                continue

            board = BughouseBoard(JSONGameReader.time_control)
            for i, (board_num, move, time_left, move_time) in enumerate(JSONGameReader.moves):
                board.update_time(board_num, time_left, move_time)
                boards = board.get_boards()

                # Define move planes (one hot encoding)
                move_planes = np.zeros((2, len(POLICY_LABELS)))
                if boards[board_num].turn == chess.WHITE:
                    move_planes[board_num][POLICY_LABELS.index(str(move))] = 1
                else:
                    move_planes[board_num][
                        POLICY_LABELS.index(mirrorMoveUCI(str(move)))
                    ] = 1

                # Move for partner board
                if (
                    boards[board_num].turn != boards[1 - board_num].turn
                    and i + 1 < len(JSONGameReader.moves)
                    and JSONGameReader.moves[i + 1][0] != board_num
                ):
                    partner_move = JSONGameReader.moves[i + 1][1]
                    if boards[1 - board_num].turn == chess.WHITE:
                        move_planes[1 - board_num][
                            POLICY_LABELS.index(str(partner_move))
                        ] = 1
                    else:
                        move_planes[1 - board_num][
                            POLICY_LABELS.index(mirrorMoveUCI(str(partner_move)))
                        ] = 1
                else:
                    move_planes[1 - board_num] = 0  # NO action

                turn = board.get_turn(board_num)
                self.board_data.append(
                    board2planes(board, turn if board_num == BOARD_A else not turn)
                )
                self.move_data.append(move_planes)
                self.result_data.append(np.array([JSONGameReader.get_result(board_num, turn)]))
                board.push(board_num, move)

        self.num_samples = len(self.board_data)
        return self.num_samples > 0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.board_data[idx], self.move_data[idx], self.result_data[idx]


if __name__ == "__main__":
    """with open("data/games.json") as f:
        games = json.load(f)
    print(games[0]["a"]["id"])
    JSONGameReader = JSONGameReader(games[0])
    board = BughouseBoard(time_control=JSONGameReader.time_control)
    for board_num, move, time_left, move_time in JSONGameReader.moves:
        print(board_num, move, time_left, move_time)
        board.update_time(board_num, time_left, move_time)"""
    generator = BughouseDataset("data/games.json")
    from torch.utils.data import DataLoader

    while generator.load_chunk():
        train_loader = DataLoader(
            generator, batch_size=1024, shuffle=True, num_workers=8
        )
        for board_planes, (y_policy, y_value) in train_loader:
            print(board_planes.shape, y_policy.shape, y_value.shape)
