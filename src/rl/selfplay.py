import datetime
import os
import random
import shutil
import time
from functools import partial
from typing import Optional

import chex
import jax
import jax.numpy as jnp
import mctx
import numpy as np
import pgx
import requests
from flax.training import train_state
from pgx.bughouse import Action, _time_advantage
from pydantic import BaseModel
from tqdm import tqdm

from src.architectures.azresnet import AZResnet, AZResnetConfig
from src.training.trainer import TrainerModule
from src.utils.bpgn import write_bpgn

import warnings

warnings.filterwarnings("ignore")


class TrainState(train_state.TrainState):
    batch_stats: chex.ArrayTree


model_configs = AZResnetConfig(
    num_blocks=15,
    channels=256,
    policy_channels=4,
    value_channels=8,
    num_policy_labels=2 * 64 * 78 + 1,
)
net = AZResnet(model_configs)
trainer = TrainerModule(
    model_class=AZResnet,
    model_configs=model_configs,
    optimizer_name="lion",
    optimizer_params={"learning_rate": 1},
    x=jnp.ones((1, 8, 16, 32)),
)
state = trainer.load_checkpoint(1)

params = {"params": state["params"], "batch_stats": state["batch_stats"]}
forward = jax.jit(partial(net.apply, train=False))

devices = jax.local_devices()
num_devices = len(devices)
print("Number of devices:", num_devices)


class Config(BaseModel):
    env_id: pgx.EnvId = "bughouse"
    seed: int = random.randint(0, 999999999)
    max_num_iters: int = 1000
    # selfplay params
    selfplay_batch_size: int = 1
    num_simulations: int = 800
    max_num_steps: int = 512

    class Config:
        extra = "forbid"


config: Config = Config()
env = pgx.make(config.env_id)


def winning_action_mask(state, rng_key):
    """
    Finds all actions that would immediately win the game for the given player.
    """

    # Play all actions and check the reward.
    # Remember that the reward is for the current player, so we expect it to be 1.

    legal_action_mask = state.legal_action_mask

    def step_and_check(action, winning_action_mask):
        winning_action_mask = jax.lax.cond(
            legal_action_mask[action],
            lambda: winning_action_mask.at[action].set(
                env.step(state, action, rng_key).rewards[0] != 0
            ),
            lambda: winning_action_mask,
        )
        return winning_action_mask

    winning_action_mask = jnp.zeros(9985)
    winning_action_mask = jax.lax.fori_loop(
        0, 9985, step_and_check, winning_action_mask
    )

    return winning_action_mask


def recurrent_fn(params, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
    current_player = state.current_player
    state = jax.vmap(env.step)(state, action)

    logits, value = forward(params, state.observation)
    # mask invalid actions
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
    value = jnp.where(state.terminated, 0.0, value)
    discount = -1.0 * jnp.ones_like(value)
    discount = jnp.where(state.terminated, 0.0, discount)

    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=logits,
        value=value,
    )
    return recurrent_fn_output, state


@partial(jax.jit, static_argnums=(2,))
def run_mcts(state, key, num_simulations: int, tree: Optional[mctx.Tree] = None):
    key1, key2 = jax.random.split(key)

    logits, value = forward(params, state.observation)
    root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

    policy_output = mctx.alphazero_policy(
        params=params,
        rng_key=key1,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        invalid_actions=~state.legal_action_mask,
        search_tree=None,
        qtransform=partial(mctx.qtransform_by_min_max, min_value=-1, max_value=1),
    )
    return policy_output


if __name__ == "__main__":
    os.makedirs("data/run1", exist_ok=True)
    game_dir = "data/games"
    if os.path.exists(game_dir):
        shutil.rmtree(game_dir)
    os.makedirs(game_dir)

    init_fn = jax.jit(jax.vmap(env.init))
    step_fn = jax.jit(jax.vmap(env.step))

    print("Running selfplay with initial seed", config.seed)

    rng_key = jax.random.PRNGKey(config.seed)

    for _ in tqdm(range(config.max_num_iters)):
        game_id = random.randint(0, 999999999)
        print(f"Playing game id: {game_id}")

        rng_key, sub_key = jax.random.split(rng_key)
        keys = jax.random.split(sub_key, config.selfplay_batch_size)
        state = init_fn(keys)
        tree = None
        actions = []
        times = []

        obs = []
        policy_tgt = []
        value_tgt = []

        while ~state.terminated.all():
            rng_key, sub_key = jax.random.split(rng_key)
            policy_output = run_mcts(state, sub_key, config.num_simulations, tree)
            # tree = mctx.get_subtree(policy_output.search_tree, policy_output.action)
            obs.append(state.observation.ravel())
            policy_tgt.append(policy_output.action_weights.ravel())

            action = policy_output.action.item()
            keys = jax.random.split(sub_key, config.selfplay_batch_size)
            state = step_fn(state, policy_output.action, keys)

            move_uci = Action._from_label(action)._to_string()
            actions.append(move_uci)
            times.append(state._clock[0].tolist())

        reward = abs(int(state.rewards[0][0]))
        for i in range(len(obs)):
            value_tgt.append(reward)
            reward *= -1
            # assert np.sum(policy_tgt[i]) == 1

        value_tgt = value_tgt[::-1]
        assert value_tgt[-1] != -1
        write_bpgn(game_id, actions, times)

        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=0)))
        filepath = (
            f"data/run1/game_{game_id}-{now.strftime('%Y%m%d')}-{now.strftime('%H%M')}"
        )
        np.savez_compressed(
            filepath, obs=obs, policy_tgt=policy_tgt, value_tgt=value_tgt
        )

        """url = f"http://ec2-3-84-181-213.compute-1.amazonaws.com:8000/upload"
        file = {"file": open(filepath + ".npz", "rb")}

        response = requests.post(url=url, files=file)

        if response.status_code == 200:
            print("Game sent successfully!")
        else:
            print(f"Error: {response.status_code}")"""
