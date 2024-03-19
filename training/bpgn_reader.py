import re
import chess.pgn
import gzip
import json 
import bz2
import chardet
import glob

from board import BughouseBoard


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

    min_rating = 1600
    with bz2.open("data/fics_db/export2023.bpgn.bz2") as file:
        file_content = file.read()
        games = parse_bpgn_file(file_content)

    board = BughouseBoard(1200)
    for game in games:
        print(game['TimeControl'])
        if game['WhiteA']['rating'] > min_rating and game['BlackA']['rating'] > min_rating and game['WhiteB']['rating'] > min_rating and game['BlackB']['rating'] > min_rating:
            pattern = r"(\d+[A-Za-z]\.\s\S+\{[\d\.]+\})"
            matches = re.findall(pattern, game["moves"][1])
            for match in matches:
                move, time = match.split("{")
                move_number, move_str = move.split('. ')
                time = time.rstrip("}")

                board_num = 0
                if move_number[-1].lower() == 'b': 
                    board_num = 1

                board.push_san(board_num, move_str)
                #print(f"Move: {move}, Time: {time}")
            break

'''def parse_bpgn_file(file_content):
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

count = 0
for path in glob.glob("data/fics_db/*"):
    with bz2.open(path) as file:
        file_content = file.read()
        games = parse_bpgn_file(file_content)

    for game in games:
        if game['WhiteA']['rating'] > 2000 and game['BlackA']['rating'] > 2000 and game['WhiteB']['rating'] > 2000 and game['BlackB']['rating'] > 2000:
            count += 1

print(count)

'''