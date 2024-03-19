import re
import os
import chess
import gzip
import json 
import bz2
import chardet
import glob
import numpy as np 

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

    samples = 2 ** 18 
    idx = 0 
    checkpoint = 0
    board_planes = np.zeros((samples, BOARD_HEIGHT, 2 * BOARD_WIDTH, NUM_BUGHOUSE_CHANNELS))
    move_planes = np.zeros((samples, 2))
    value_planes = np.zeros((samples))
    
    min_rating = 2000

    for path in glob.glob("data/fics_db/*"):
        with bz2.open(path) as file:
            file_content = file.read()
            games = parse_bpgn_file(file_content)

        for game in games:
            game = {'Event': 'FICS rated bughouse game', 'Site': 'FICS - freechess.org', 'Date': '2017.03.04', 'Time': '16:28:40', 'BughouseDBGameNo': '3611438', 'WhiteA': {'username': 'cheesybread', 'rating': 2701}, 'BlackA': {'username': 'tantheman', 'rating': 2487}, 'WhiteB': {'username': 'helmsknight', 'rating': 2443}, 'BlackB': {'username': 'ChiCkenCROSSrOaD', 'rating': 2611}, 'TimeControl': '120+0', 'Lag': '122 374 18 88 517 37 18 71 84 20 726 36 19 46 220 384 165 86 36 20 382 358 35 483 21 35 102 19 241 36 85 573 86 240 20 86 511 114 19 120 87 17 36 21 239 35 85 512 20 83 36 19 36 29 36 22 35 27 240 84 240 84 511 85 241 36 84 25 35 150 460 206 592 204 36 60 321 42 57 219 59 76 307 44 179 349 208 17 191 331 217 17 106 311 20 103 271 20 106 518 17 95 394 19 260 206', 'Result': '0-1', 'moves': ['{C:This is game number 3611438 at http://www.bughouse-db.org}', '1A. d4{119.051} 1a. Nc6{119.861} 1B. d4{118.328} 2A. Nc3{118.900} 2a. Nf6{119.761} 1b. Nf6{119.313} 2B. Nf3{118.228} 2b. d5{119.213} 3A. Nf3{118.620} 3B. Bf4{117.637} 3a. e6{119.661} 3b. Bf5{118.934} 4B. e3{117.537} 4b. e6{118.609} 4A. e4{118.047} 4a. Bb4{119.426} 5B. Bd3{116.567} 5A. e5{117.699} 5b. Ne4{117.969} 6B. Ne5{115.984} 5a. Ne4{118.567} 6A. Bd3{117.599} 6b. Bd6{117.424} 6a. d5{118.467} 7B. O-O{115.160} 7b. O-O{116.836} 7A. O-O{116.383} 8B. f3{114.637} 7a. Bxc3{118.103} 8b. f6{116.384} 8A. bxc3{116.283} 8a. Nxc3{118.003} 9A. Bg5{115.782} 9a. Nxd1{116.768} 9B. P@f7+{111.607} 10A. Bxd8{114.373} 10a. Kxd8{116.668} 9b. Rxf7{114.524} 10B. Nxf7{110.993} 10b. Kxf7{114.424} 11A. Raxd1{112.727} 11B. fxe4{109.174} 11b. Bxf4{113.615} 12B. Rxf4{107.786} 11a. R@g6{112.523} 12b. Q@g5{109.897} 12A. Bxg6{109.695} 12a. hxg6{112.423} 13B. exf5{106.343} 13A. P@f6{108.884} 13b. N@h3+{109.030} 14B. Kf1{105.175} 14b. R@h1+{97.814} 15B. Ke2{104.522} 15b. Nxf4+{71.593} 16B. exf4{102.684} 16b. Qxg2+{71.223} 17B. B@f2{102.209} 13a. N@e2+{69.197} 14A. Kh1{107.842} 14a. N@g3+{68.731} 15A. fxg3{107.440} 15a. Nxg3+{68.631} 16A. Kg1{107.111} 16a. B@e3+{68.245} 17b. B@f3+{66.619} 17A. Rf2{106.838} 18B. Kd2{99.783} 18b. Rxd1+{65.543} 19B. Kc3{99.683} 17a. Bxf2+{2.717} 18A. Kxf2{106.738} 18a. Ne4+{2.617} 19A. Kf1{106.638} 19b. B@a5+{1.852} 20B. Q@b4{99.583} 19a. gxf6{2.274} 20b. Bxb4+{1.300} 21B. Kb3{99.483} 20A. Ke2{106.538} 21b. a5{1.137} 22B. B@h5+{99.383} 20a. B@b5+{1.878} 22b. Ke7{0.640} 21A. Rd3{106.438} 21a. Nxd4+{1.391} 22A. Kd1{106.338} 23B. N@c6+{98.265} 23b. Kd6{0.540} 22a. Bxd3{1.167} 23A. Q@e7+{106.238} 24B. R@d7+{97.435} 24b. Kxd7{0.440} 23a. Kxe7{0.719} 25B. Nxb8+{96.938} 25b. Kc8{0.340} 24A. Kc1{106.138} 26B. Nc6{96.471} 26b. Qd7{0.240} 24a. Kd7{0.619} 27B. Nd8{96.203} 27b. Kb8{0.140} 25A. Q@d6+{106.038} 28B. Nc6+{95.720} 25a. cxd6{-0.576} 26A. Kb2{105.938}', '{ChiCkenCROSSrOaD forfeits on time} 0-1']}
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
                        value_planes = np.zeros((samples))
                        checkpoint += 1
                        idx = 0