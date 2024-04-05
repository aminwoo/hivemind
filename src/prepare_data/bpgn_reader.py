import bz2
import glob
import os
import re

import chardet
import chess
import jax
import jax.numpy as jnp
import numpy as np
from pgx.bughouse import Action, Bughouse, State, _observe
from pgx.experimental.bughouse import make_policy_labels
from tqdm import tqdm

from src.domain.board import BughouseBoard
from src.domain.move2planes import mirrorMoveUCI


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


def generate_planes():
    os.makedirs(f"data/training_data", exist_ok=True)

    seed = 42 
    key = jax.random.PRNGKey(seed)
    env = Bughouse() 
    init_fn = jax.jit(env.init)
    step_fn = jax.jit(env.step)

    labels = make_policy_labels()
    print("Num labels:", len(labels))
     
    samples = 2**16 
    idx = 0 
    checkpoint = 0
    board_planes = np.zeros((samples, 8, 16, 32))
    move_planes = np.zeros((samples))
    value_planes = np.zeros((samples))
    
    min_rating = 2000

    errors = 0 
    for path in tqdm(glob.glob("data/fics_db/*")):
        with bz2.open(path) as file:
            file_content = file.read()
            games = parse_bpgn_file(file_content)

        for game in games:
            try:
                state = init_fn(key)

                if 'Result' not in game:
                    continue

                result = game['Result'].split(']')[0].strip('""')
                if result == '1-0':
                    reward = [[1, -1], [-1, 1]]
                elif result == '0-1':
                    reward = [[-1, 1], [1, -1]]
                elif result == '1/2-1/2':
                    reward = [[0, 0], [0, 0]]
                else:
                    continue

                if game['TimeControl'] == "120+0":
                    state = state.replace(_clock=jnp.int32([[1200, 1200], [1200, 1200]]))
                    board = BughouseBoard(1200)
                    times = [[1200, 1200], [1200, 1200]] 
                elif game['TimeControl'] == "180+0":
                    state = state.replace(_clock=jnp.int32([[1800, 1800], [1800, 1800]]))
                    board = BughouseBoard(1800)
                    times = [[1800, 1800], [1800, 1800]]
                else:
                    continue

                turn = [0, 0]
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
                        times[1 - board_num][turn[1 - board_num]] -= (times[board_num][turn[board_num]] - time_left)
                        times[board_num][turn[board_num]] = time_left 
                        assert (times[0][0] - times[1][0]) == (times[1][1] - times[0][1])

                        move_uci = str(board.parse_san(board_num, move_san))
                        if board.turn(board_num) == chess.BLACK: 
                            move_uci = mirrorMoveUCI(move_uci)
                        move_uci = str(board_num) + move_uci 
                        if move_uci.endswith("q"): # Treat queen promotion as default move
                            move_uci = move_uci[:-1]

                        t = times.copy()
                        if turn[0] != 0:
                            t[0] = t[0][::-1]
                        if turn[1] != 0:
                            t[1] = t[1][::-1]

                        state = state.replace(_clock=jnp.int32(t))
                        state = state.replace(current_player=turn[board_num] if board_num == 0 else 1 - turn[board_num])
                        planes = np.array(_observe(state, player_id=state.current_player)).reshape(8, 16, 32)

                        assert np.isclose(planes[0, 0, 31], (0.5 + (times[board_num][turn[board_num]] - times[1 - board_num][turn[board_num]]) / 300), atol=1e-05)
                        assert (jnp.int32(turn) == state._turn).all() 
                        move_planes[idx] = labels.index(move_uci)
                        board_planes[idx] = planes
                        value_planes[idx] = reward[board_num][turn[board_num]]

                        turn[board_num] = 1 - turn[board_num]
                        board.push_san(board_num, move_san)
                        state = step_fn(state, labels.index(move_uci))

                        idx += 1
                        if idx >= samples: 
                            np.savez_compressed(f'data/training_data/checkpoint{checkpoint}', board_planes=board_planes, move_planes=move_planes, value_planes=value_planes)
                            board_planes = np.zeros((samples, 8, 16, 32))
                            move_planes = np.zeros((samples))
                            value_planes = np.zeros((samples))
                            checkpoint += 1
                            idx = 0
            except Exception as e:
                print(game)
                print(e)
                errors += 1

    print(errors)

if __name__ == "__main__": 
    generate_planes()