from time import time 
import jax
import jax.numpy as jnp
import numpy as np 

from pgx.chess import Chess
from pgx.bughouse import Bughouse, Action

@jax.jit
def act_randomly(rng_key, obs, mask):
    """Ignore observation and choose randomly from legal actions"""
    del obs
    probs = mask / mask.sum()
    logits = jnp.maximum(jnp.log(probs), jnp.finfo(probs.dtype).min)
    return jax.random.categorical(rng_key, logits=logits, axis=-1)

seed = 42 
env = Bughouse()
init_fn = jax.jit(env.init)
step_fn = jax.jit(env.step)

key = jax.random.PRNGKey(seed)
state = init_fn(key)

times = [] 
iterations = 100
for seed in range(iterations): 
    while ~state.terminated:
        key, subkey = jax.random.split(key)
        action = act_randomly(subkey, state.observation, state.legal_action_mask)
        #print(Action._from_label(action)._to_string())
        st = time() 
        state = step_fn(state, action, key)
        end = time() 
        if end - st < 1:
            times.append(end - st)

print(np.mean(times))