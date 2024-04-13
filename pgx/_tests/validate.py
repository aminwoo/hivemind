from benchmarking.board import BughouseBoard

import jax
import jax.numpy as jnp
import chess
from tqdm import tqdm
from pgx.bughouse import Bughouse, Action, State, _is_promotion

def mirrorMoveUCI(uci_move):
    move = chess.Move.from_uci(uci_move)
    return mirrorMove(move).uci()


def mirrorMove(move):
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        move.promotion,
        move.drop,
    )

@jax.jit
def act_randomly(rng_key, obs, mask):
    """Ignore observation and choose randomly from legal actions"""
    del obs
    probs = mask / mask.sum()
    logits = jnp.maximum(jnp.log(probs), jnp.finfo(probs.dtype).min)
    return jax.random.categorical(rng_key, logits=logits, axis=-1)

env = Bughouse()

#state = State._from_fen('1R4k1/5ppp/7N/8/8/8/PPP4r/1K6/ b - - 0 1|rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR/ w KQkq - 0 1')
state = State._from_fen('r1bqkbnr/ppp2Qpp/2np4/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR/ b KQkq - 0 4|rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR/ w KQkq - 0 1')
print(state.terminated)

'''init_fn = jax.jit(env.init)
step_fn = jax.jit(env.step)

simulations = 10000
for seed in tqdm(range(simulations)):
    key = jax.random.PRNGKey(seed)
    state = init_fn(key)

    board = BughouseBoard()
    board.current_player = (state.current_player == 0)
    while ~state.terminated:
        key, subkey = jax.random.split(key)

        action = act_randomly(subkey, state.observation, state.legal_action_mask)

        actions = [] 
        for i in range(2 * 64 * 78):
            if state.legal_action_mask[i]:
                actions.append(Action._from_label(i)._to_string()) 
        
        assert len(actions) == len(board.legal_moves())

        move_uci = Action._from_label(action)._to_string()
        if _is_promotion(state, action) and len(move_uci) < 6:
            move_uci += 'q'

        board.current_player = not board.current_player
        state = step_fn(state, action)
        if move_uci == "pass":
            continue
        
        board_num = int(move_uci[0])
        if board.boards[board_num].turn == chess.BLACK:
            board.push_uci(board_num, mirrorMoveUCI(move_uci[1:]))
        else:
            board.push_uci(board_num, move_uci[1:])

        assert board.is_checkmate() == (1 in state.rewards)
        assert board.is_game_over() == state.terminated or state._step_count >= 256 '''