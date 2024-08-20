from functools import partial
from time import time
import json

import jax
import jax.numpy as jnp
import mctx
import pgx

from pgx.bughouse import State, _set_current_player, Action, _set_clock, _time_advantage

from src.architectures.azresnet import AZResnet, AZResnetConfig
from src.training.trainer import TrainerModule

trainer = TrainerModule(
    model_class=AZResnet,
    model_configs=AZResnetConfig(
        num_blocks=15,
        channels=256,
        policy_channels=4,
        value_channels=8,
        num_policy_labels=2 * 64 * 78 + 1,
    ),
    optimizer_name="lion",
    optimizer_params={"learning_rate": 0.00001},
    x=jnp.ones((1, 8, 16, 32)),
)
state = trainer.load_checkpoint("2")

variables = {"params": state["params"], "batch_stats": state["batch_stats"]}
model = AZResnet(
    AZResnetConfig(
        num_blocks=15,
        channels=256,
        policy_channels=4,
        value_channels=8,
        num_policy_labels=2 * 64 * 78 + 1,
    )
)
forward = jax.jit(partial(model.apply, train=False))

seed = 42
key = jax.random.PRNGKey(seed)
keys = jax.random.split(key, 1)
env = pgx.make("bughouse")
step_fn = jax.jit(jax.vmap(env.step))
init_fn = jax.jit(jax.vmap(env.init))


@partial(jax.jit, static_argnums=(1,))
def search(state, simulations):
    def recurrent_fn(variables, rng_key: jnp.ndarray, action: jnp.ndarray, state):
        current_player = state.current_player
        state = step_fn(state, action)

        policy_logits, value = forward(variables, state.observation)
        policy_logits = policy_logits - jnp.max(policy_logits, axis=-1, keepdims=True)
        policy_logits = jnp.where(state.legal_action_mask, policy_logits, jnp.finfo(policy_logits.dtype).min)
        
        reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
        value = jnp.where(state.terminated, 0.0, value)
        discount = -1.0 * jnp.ones_like(value)
        discount = jnp.where(state.terminated, 0.0, discount)

        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=policy_logits,
            value=value,
        )
        return recurrent_fn_output, state

    policy_logits, value = forward(variables, state.observation)
    root = mctx.RootFnOutput(prior_logits=policy_logits, value=value, embedding=state)

    
    policy_output = mctx.alphazero_policy(
        params=variables,
        rng_key=key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=simulations,
        invalid_actions=~state.legal_action_mask,
        qtransform=partial(mctx.qtransform_by_min_max, min_value=-1, max_value=1),
        dirichlet_fraction=0.0,
        temperature=0.0
    )
    return policy_output