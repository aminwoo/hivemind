import os 
import json 
import jax
import jax.numpy as jnp 
import numpy as np 
import chess
from tqdm import tqdm
from pgx.bughouse import Bughouse, State, _observe
from pgx.experimental.bughouse import make_policy_labels
from src.domain.move2planes import mirrorMoveUCI
from src.domain.board import BughouseBoard
from src.prepare_data.loader import JSONGameReader


def generate_planes(samples=2**16): 
    os.makedirs(f"data/training_data", exist_ok=True)
    with open('data/games.json') as f:
        games = json.load(f)

    seed = 42 
    key = jax.random.PRNGKey(seed)
    env = Bughouse() 
    init_fn = jax.jit(env.init)
    step_fn = jax.jit(env.step)

    labels = make_policy_labels()

    errors = 0 
    idx = 0 
    checkpoint = 466
    board_planes = np.zeros((samples, 8, 16, 32))
    move_planes = np.zeros((samples))
    value_planes = np.zeros((samples))
    
    for game in tqdm(games):
        try:
            state = init_fn(key)
            reader = JSONGameReader(game)
            if reader.time_control not in [1800, 1200]:
                continue
            board = BughouseBoard(reader.time_control)

            result = reader.result
            reward = [[result, -result], [-result, result]]
            if reader.time_control == 1200:
                state = state.replace(_clock=jnp.int32([[1200, 1200], [1200, 1200]]))
                board = BughouseBoard(1200)
                times = [[1200, 1200], [1200, 1200]] 
            else:
                state = state.replace(_clock=jnp.int32([[1800, 1800], [1800, 1800]]))
                board = BughouseBoard(1800)
                times = [[1800, 1800], [1800, 1800]]

            turn = [0, 0]
            for board_num, move, time_left, move_time in reader.moves:
                times[1 - board_num][turn[1 - board_num]] -= (times[board_num][turn[board_num]] - time_left)
                times[board_num][turn[board_num]] = time_left 
                assert (times[0][0] - times[1][0]) == (times[1][1] - times[0][1])

                move_uci = move.uci()
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
                board.push(board_num, move)
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
            print(e)
            errors += 1
    print(errors)


if __name__ == '__main__': 
    generate_planes() 