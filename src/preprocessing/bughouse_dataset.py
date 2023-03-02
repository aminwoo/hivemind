import chess
import numpy as np
import torch

from torch.utils.data import Dataset
from utils.bugboard import BughouseBoard
from utils.game_parser import Parser

from utils.board2planes import board2planes
from utils.move2planes import mirrorMoveUCI
from utils.constants import (
    POLICY_LABELS, BOARD_HEIGHT, NUM_BUGHOUSE_CHANNELS, BOARD_WIDTH, BOARD_A)


class BughouseDataset(Dataset):
    def __init__(self, games):
        self.games = games
        self.board_data = None
        self.move_data = None
        self.result_data = None
        self.num_samples = 0
        self.game_idx = 0

    def reset(self):
        self.num_samples = 0
        self.game_idx = 0

    def load_chunk(self, chunk_size=1024):
        self.board_data = []
        self.move_data = []
        self.result_data = []

        while len(self.board_data) < chunk_size and self.game_idx < len(self.games):
            try:
                parser = Parser(self.games[self.game_idx])
            except Exception:
                self.game_idx += 1
                continue
            self.game_idx += 1

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