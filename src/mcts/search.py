from functools import partial
from time import time

import jax
import jax.numpy as jnp
import mctx
from pgx.bughouse import Action, Bughouse, State

from src.architectures.azresnet import AZResnet, AZResnetConfig
from src.training.trainer import TrainerModule

trainer = TrainerModule(model_name='AZResNet', model_class=AZResnet, model_configs=AZResnetConfig(
    num_blocks=15,
    channels=256,
    policy_channels=4, 
    value_channels=8,
    num_policy_labels=2*64*78+1
), optimizer_name='lion', optimizer_params={'learning_rate': 0.00001}, x=jnp.ones((1, 8, 16, 32)))
state = trainer.load_checkpoint('0')

variables = {'params': state['params'], 'batch_stats': state['batch_stats']}
model = AZResnet(
    AZResnetConfig(
        num_blocks=15,
        channels=256,
        policy_channels=4,
        value_channels=8,
        num_policy_labels=2*64*78+1,
    )
)
forward = jax.jit(partial(model.apply, train=False))

seed = 42
key = jax.random.PRNGKey(seed)
key1, key2 = jax.random.split(key)
keys = jax.random.split(key, 1)
env = Bughouse()
step_fn = jax.jit(jax.vmap(env.step))
init_fn = jax.jit(jax.vmap(env.init))

@jax.jit
def search(state):

    def recurrent_fn(variables, rng_key: jnp.ndarray, action: jnp.ndarray, state):
        del rng_key
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

    policy_output = mctx.gumbel_muzero_policy(
        params=variables,
        rng_key=key1,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=400,
        invalid_actions=~state.legal_action_mask,
        qtransform=mctx.qtransform_completed_by_mix_value,
        max_num_considered_actions=512,
        gumbel_scale=1.0,
    )
    return policy_output

if __name__ == '__main__':
    state = init_fn(keys)
    out = search(state)
    print(Action._from_label(out.action[0])._to_string())