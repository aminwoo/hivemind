import chess
import numpy as np
from tcn import tcn_decode
from board import BughouseBoard
import json

import torch
from torch.utils.data import Dataset

from board2planes import board2planes
from move2planes import mirrorMoveUCI
from constants import (
    POLICY_LABELS, BOARD_HEIGHT, NUM_BUGHOUSE_CHANNELS, BOARD_WIDTH, BOARD_A, BOARD_B)


class Parser(object):

    def __init__(self, game):
        self.result = self.get_game_result(game)
        self.ratings = [
            [game["a"]["white"]["rating"],
             game["a"]["black"]["rating"]],
            [game["b"]["white"]["rating"],
             game["b"]["black"]["rating"]]
        ]
        self.time_control = self.get_game_time_control(game)
        if self.time_control not in [1800, 1200]:
            return None 

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
        if (board_num == BOARD_A and turn == chess.WHITE) or (board_num == BOARD_B and turn == chess.BLACK):
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
        # Overwrites the first white and black move times to be equal to the time control
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
    def get_naive_move_order(deltas):
        # get time_deltas
        a_deltas_b = np.array(deltas[BOARD_A][chess.BLACK])
        a_deltas_w = np.array(deltas[BOARD_A][chess.WHITE])
        b_deltas_b = np.array(deltas[BOARD_B][chess.BLACK])
        b_deltas_w = np.array(deltas[BOARD_B][chess.WHITE])

        # interleave player time deltas
        a_deltas = np.empty(a_deltas_w.size + a_deltas_b.size)
        a_deltas[0::2] = a_deltas_w
        a_deltas[1::2] = a_deltas_b
        b_deltas = np.empty(b_deltas_w.size + b_deltas_b.size)
        b_deltas[0::2] = b_deltas_w
        b_deltas[1::2] = b_deltas_b

        # get accumulated player times
        a_times = np.cumsum(a_deltas)
        b_times = np.cumsum(b_deltas)

        all_times = np.concatenate((a_times, b_times))
        all_indices = np.argsort(all_times)
        move_order = np.digitize(all_indices, [0, a_times.shape[0]]) - 1
        return move_order

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
                if board_num == stuck_board or (move.drop and pockets[turn].count(move.drop) <= 0):
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
        with open(self.path) as f:
            self.games = json.load(f)

    def reset(self):
        self.num_samples = 0
        self.game_idx = 0

    def load_chunk(self, chunk_size=1024):
        if not self.games:
            self.load() 

        self.board_data = []
        self.move_data = []
        self.result_data = []

        while len(self.board_data) < chunk_size and self.game_idx < len(self.games):
            parser = Parser(self.games[self.game_idx])
            self.game_idx += 1
            if not parser:
                continue 

            board = BughouseBoard(parser.time_control)
            for board_num, move, time_left, move_time in parser.moves:
                board.update_time(board_num, time_left, move_time)
                boards = board.get_boards()

                move_planes = np.zeros((2, len(POLICY_LABELS)))
                if boards[board_num].turn == chess.WHITE:
                    move_planes[board_num][POLICY_LABELS.index(str(move))] = 1
                else:
                    move_planes[board_num][POLICY_LABELS.index(mirrorMoveUCI(str(move)))] = 1

                turn = board.get_turn(board_num)
                self.board_data.append(board2planes(board, turn if board_num == BOARD_A else not turn).flatten())
                self.move_data.append(move_planes.flatten())
                self.result_data.append(parser.get_result(board_num, turn))
                board.push(board_num, move)

        self.num_samples = len(self.board_data)
        return self.num_samples > 0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_planes = torch.tensor(self.board_data[idx].reshape(NUM_BUGHOUSE_CHANNELS, BOARD_HEIGHT, 2 * BOARD_WIDTH), dtype=torch.float)
        y_value = torch.tensor(self.result_data[idx], dtype=torch.float)
        y_policy = torch.tensor(self.move_data[idx].reshape(2 * len(POLICY_LABELS)), dtype=torch.float)
        return input_planes, (y_value, y_policy)
    

if __name__ == '__main__':
    with open("data/games.json") as f:
        games = json.load(f)
    print(games[0]["a"]["id"])
    parser = Parser(games[0])
    board = BughouseBoard(time_control=parser.time_control)
    for board_num, move, time_left, move_time in parser.moves:
        print(board_num, move, time_left, move_time)
        board.update_time(board_num, time_left, move_time)
