import re
import os
import chess
import bz2
import chardet
import glob
import numpy as np 
from tqdm import tqdm 

from board import BughouseBoard
from board2planes import board2planes
from move2planes import mirrorMoveUCI
from constants import POLICY_LABELS, BOARD_HEIGHT, BOARD_WIDTH, NUM_BUGHOUSE_CHANNELS, BOARD_A


def parse_bpgn_file(file_content):
    games = []
    current_game = {}
    for line in file_content.split(b"\n"):
        line = line.strip()
        if line.startswith(b"["):
            key, value = line[1:-1].split(b" ", 1)
            key = key.decode()
            value = value.strip(b'"').decode()
            if key == "Event":
                if current_game:
                    games.append(current_game)
                    current_game = {}
            current_game[key] = value
        elif line:
            if "moves" not in current_game:
                current_game["moves"] = []

            result = chardet.detect(line)
            encoding = result['encoding']
            if encoding:
                current_game["moves"].append(line.decode(encoding))

    if current_game:
        games.append(current_game)

    filtered_games = [] 
    for game in games:
        try:
            for player_key in ("WhiteA", "BlackA", "WhiteB", "BlackB"):
                player_info = game[player_key].split("]")
                username = player_info[0].strip('"')
                rating_info = player_info[1].strip("[").split(" ")
                rating = int(rating_info[1].strip('"'))
                game[player_key] = {
                    "username": username,
                    "rating": rating
                }
            filtered_games.append(game)
        except:
            continue

    return filtered_games


if __name__ == "__main__": 
    os.makedirs(f"data/fics_training_data", exist_ok=True)

    samples = 2**16 
    idx = 0 
    checkpoint = 0
    board_planes = np.zeros((samples, BOARD_HEIGHT, 2 * BOARD_WIDTH, NUM_BUGHOUSE_CHANNELS))
    move_planes = np.zeros((samples, 2))
    value_planes = np.zeros((samples, 1))
    
    min_rating = 2000

    errors = 0 
    for path in tqdm(glob.glob("data/fics_db/*")):
        with bz2.open(path) as file:
            file_content = file.read()
            games = parse_bpgn_file(file_content)

        for game in games:
            try:
                if 'Result' not in game:
                    continue

                result = game['Result'].split(']')[0].strip('""')

                if result == '1-0':
                    reward = [[-1, 1], [1, -1]]
                elif result == '0-1':
                    reward = [[1, -1], [-1, 1]]
                elif result == '1/2-1/2':
                    reward = [[0, 0], [0, 0]]
                else:
                    continue

                if game['TimeControl'] == "120+0":
                    board = BughouseBoard(1200)
                    times = [[1200, 1200], [1200, 1200]] 
                elif game['TimeControl'] == "180+0":
                    board = BughouseBoard(1800)
                    times = [[1800, 1800], [1800, 1800]]
                else:
                    continue

                turn = [True, True]
                if game['WhiteA']['rating'] > min_rating and game['BlackA']['rating'] > min_rating and game['WhiteB']['rating'] > min_rating and game['BlackB']['rating'] > min_rating:
                    pattern = r"(\d+[A-Za-z]\.\s\S+\{[-\d\.]+\})"
                    matches = re.findall(pattern, game["moves"][1])

                    data = [] 
                    for match in matches: 
                        move, time = match.split("{")
                        move_number, move_san = move.split('. ')
                        time = time.rstrip("}")
                        time_left = int(float(time) * 10)

                        board_num = 0
                        if move_number[-1].lower() == 'b': 
                            board_num = 1
                        data.append((board_num, move_san, time_left))
                        
                    for i in range(len(data)):
                        board_num, move_san, time_left = data[i] 
                        times[not board_num][turn[not board_num]] -= times[board_num][turn[board_num]] - time_left
                        times[board_num][turn[board_num]] = time_left 
                        board.set_time(times)

                        boards = board.get_boards()

                        # One hot encoding for expected move
                        if boards[board_num].turn == chess.WHITE:
                            move_planes[idx][board_num] = POLICY_LABELS.index(str(board.parse_san(board_num, move_san)))
                        else:
                            move_planes[idx][board_num] = POLICY_LABELS.index(mirrorMoveUCI(str(board.parse_san(board_num, move_san))))

                        try:
                            # Move for partner board
                            if (
                                boards[board_num].turn != boards[1 - board_num].turn
                                and i + 1 < len(data)
                                and data[i + 1][0] != board_num
                            ):
                                partner_move = data[i + 1][1]
                                if boards[1 - board_num].turn == chess.WHITE:
                                    move_planes[idx][1 - board_num] = POLICY_LABELS.index(str(board.parse_san(1 - board_num, partner_move)))
                                else:
                                    move_planes[idx][1 - board_num] = POLICY_LABELS.index(mirrorMoveUCI(str(board.parse_san(1 - board_num, partner_move))))
                            else:
                                move_planes[idx][1 - board_num] = 0  # NO action
                        except:
                            move_planes[idx][1 - board_num] = 0  # NO action

                        board_planes[idx] = board2planes(board, turn[0] if board_num == BOARD_A else not turn[1])
                        value_planes[idx] = reward[board_num][turn[board_num]]
                        
                        turn[board_num] = not turn[board_num]
                        board.push_san(board_num, move_san)

                        idx += 1
                        if idx >= samples: 
                            np.savez_compressed(f'data/fics_training_data/checkpoint{checkpoint}', board_planes=board_planes, move_planes=move_planes, value_planes=value_planes)
                            board_planes = np.zeros((samples, BOARD_HEIGHT, 2 * BOARD_WIDTH, NUM_BUGHOUSE_CHANNELS))
                            move_planes = np.zeros((samples, 2))
                            value_planes = np.zeros((samples, 1))
                            checkpoint += 1
                            idx = 0
            except Exception as e:
                print(e)
                errors += 1

    print(errors)