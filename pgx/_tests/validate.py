from pgx._tests.board import BughouseBoard

import jax
import jax.numpy as jnp
import numpy as np 
import chess
from tqdm import tqdm
from pgx.bughouse import Bughouse, Action, _is_promotion, _on_turn, _time_advantage, _legal_drops


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


def legal_moves(state):
    moves = []  
    for i in range(9985): 
        if state.legal_action_mask[i]:
            moves.append(Action._from_label(i)._to_string())
    return moves


env = Bughouse()
init_fn = jax.jit(env.init)
step_fn = jax.jit(env.step)

simulations = 10000
for seed in tqdm(range(simulations)):
    key = jax.random.PRNGKey(seed)
    state = init_fn(key)

    while ~state.terminated:
        board = BughouseBoard.from_fen(state._to_fen())
        board.current_player = (state.current_player == 0)
        assert jnp.sum(state.legal_action_mask[:9984]) == len(board.legal_moves()), (legal_moves(state), board.legal_moves(), state._to_fen())

        key, subkey = jax.random.split(key)
        action = act_randomly(subkey, state.observation, state.legal_action_mask)
        state = step_fn(state, action, subkey)

    board = BughouseBoard.from_fen(state._to_fen())
    assert board.is_checkmate() or state._step_count >= 256, state._to_fen()
