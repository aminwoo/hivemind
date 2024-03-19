import os 
import json 
import numpy as np 
import chess

from board2planes import board2planes
from move2planes import mirrorMoveUCI
from board import BughouseBoard
from loader import Parser
from constants import POLICY_LABELS, BOARD_HEIGHT, BOARD_WIDTH, NUM_BUGHOUSE_CHANNELS, BOARD_A


def generate_planes(samples=409600): 
    os.makedirs(f"data/training_data", exist_ok=True)
    with open('data/games.json') as f:
        games = json.load(f)
    
    errors = 0 

    idx = 0 
    checkpoint = 0
    board_planes = np.zeros((samples, BOARD_HEIGHT, 2 * BOARD_WIDTH, NUM_BUGHOUSE_CHANNELS))
    move_planes = np.zeros((samples, 2))
    value_planes = np.zeros((samples))
    
    for game in games:
        try:
            parser = Parser(game)
            if parser.time_control not in [1800, 1200]:
                continue
            board = BughouseBoard(parser.time_control)

            for i, (board_num, move, time_left, move_time) in enumerate(parser.moves):
                board.update_time(board_num, time_left, move_time)
                boards = board.get_boards()

                # One hot encoding for expected move
                if boards[board_num].turn == chess.WHITE:
                    move_planes[idx][board_num] = POLICY_LABELS.index(str(move))
                else:
                    move_planes[idx][board_num] = POLICY_LABELS.index(mirrorMoveUCI(str(move)))

                # Move for partner board
                if (
                    boards[board_num].turn != boards[1 - board_num].turn
                    and i + 1 < len(parser.moves)
                    and parser.moves[i + 1][0] != board_num
                ):
                    partner_move = parser.moves[i + 1][1]
                    if boards[1 - board_num].turn == chess.WHITE:
                        move_planes[idx][1 - board_num] = POLICY_LABELS.index(str(partner_move))
                    else:
                        move_planes[idx][1 - board_num] = POLICY_LABELS.index(mirrorMoveUCI(str(partner_move)))
                else:
                    move_planes[idx][1 - board_num] = 0  # NO action

                turn = board.get_turn(board_num)
                board_planes[idx] = board2planes(board, turn if board_num == BOARD_A else not turn)
                value_planes[idx] = parser.get_result(board_num, turn)

                idx += 1
                if idx >= samples: 
                    np.savez_compressed(f'data/training_data/checkpoint{checkpoint}', board_planes=board_planes, move_planes=move_planes, value_planes=value_planes)
                    board_planes = np.zeros((samples, BOARD_HEIGHT, 2 * BOARD_WIDTH, NUM_BUGHOUSE_CHANNELS))
                    move_planes = np.zeros((samples, 2))
                    value_planes = np.zeros((samples))
                    checkpoint += 1
                    idx = 0

                board.push(board_num, move)

        except Exception as e:
            print(e)
            errors += 1

    print(errors)


if __name__ == '__main__': 
    generate_planes() 