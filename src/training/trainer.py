import datetime
import os

from typing import Any
from tqdm.auto import tqdm

import numpy as np
import jax
import jax.numpy as jnp
import optax
import orbax
import chex
import wandb
import pgx
import orbax.checkpoint as ocp

from flax.training import orbax_utils
from flax.training.train_state import TrainState
from flax import linen as nn
from flax.training import train_state


def extract_params(state: TrainState): 
    if isinstance(state, dict):
        return {'params': state['params'], 'batch_stats': state['batch_stats']}
    return {'params': state.params, 'batch_stats': state.batch_stats}


class TrainState(train_state.TrainState):
    batch_stats: chex.ArrayTree


class TrainerModule:

    def __init__(
        self,
        model_class: nn.Module,
        model_configs: Any,
        optimizer_name: str,
        optimizer_params: dict,
        x: Any,
        ckpt_dir: str = os.getcwd() + '/checkpoints',
        seed=42,
    ):
        '''
        Module for summarizing all training functionalities for classification.

        Inputs:
            model_class - Class implementing the neural network
            model_hparams - Hyperparameters of the model, used as input to model constructor
            optimizer_name - String of the optimizer name, supporting ['sgd', 'adam', 'adamw']
            optimizer_params - Hyperparameters of the optimizer, including learning rate as 'lr'
            x - Example imgs, used as input to initialize the model
            seed - Seed to use in the model initialization
        '''
        super().__init__()
        self.model_class = model_class
        self.model_configs = model_configs
        self.optimizer_name = optimizer_name
        self.optimizer_params = optimizer_params
        self.seed = seed
        # Create empty model. Note: no parameters yet
        self.model = self.model_class(model_configs)

        self.ckpt_dir = ckpt_dir
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model(x)

    def init_model(self, x):
        # Initialize model
        init_rng = jax.random.PRNGKey(self.seed)
        variables = self.model.init(init_rng, x, train=True)
        self.init_params, self.init_batch_stats = (
            variables['params'],
            variables['batch_stats'],
        )
        self.state = None

    def init_optimizer(self):
        # Initialize learning rate schedule and optimizer
        if self.optimizer_name.lower() == 'adam':
            opt_class = optax.adam
        elif self.optimizer_name.lower() == 'adamw':
            opt_class = optax.adamw
        elif self.optimizer_name.lower() == 'sgd':
            opt_class = optax.sgd
        elif self.optimizer_name.lower() == 'lion':
            opt_class = optax.lion
        else:
            assert False, f'Unknown optimizer "{opt_class}"'

        # Initialize training state
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=self.init_params if self.state is None else self.state['params'],
            batch_stats=(
                self.init_batch_stats if self.state is None else self.state['batch_stats']
            ),
            tx=opt_class(**self.optimizer_params),
        )

    def create_functions(self):
        # Function to calculate the classification loss and accuracy for a model
        def train_step(state: TrainState, obs, policy_tgt, value_tgt, value_mask):
            def calculate_loss(params, batch_stats):
                logits, new_model_state = state.apply_fn(
                    {'params': params, 'batch_stats': batch_stats},
                    obs,
                    train=True,
                    mutable=['batch_stats'],
                )
                policy_logits, value = logits
                policy_loss = optax.softmax_cross_entropy(policy_logits, policy_tgt).mean()

                value_loss = optax.l2_loss(value, value_tgt)
                value_loss = jnp.mean(value_loss * value_mask)
                loss = 0.99 * policy_loss + 0.01 * value_loss

                return loss, (new_model_state, policy_loss, value_loss)

            loss_fn = lambda params: calculate_loss(
                params, state.batch_stats
            )

            # Get loss, gradients for loss, and other outputs of loss function
            (loss, (new_model_state, policy_loss, value_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params
            )
            # Update parameters and batch statistics
            state = state.apply_gradients(
                grads=grads, batch_stats=new_model_state['batch_stats']
            )
            return state, policy_loss, value_loss
        
        def eval_step(state: TrainState, board_planes, y_policy, y_value):
            policy_logits, value = state.apply_fn(
                {'params': self.state.params, 
                'batch_stats': self.state.batch_stats},
                board_planes,
                train=False,
            )

            policy_acc = np.sum(np.argmax(policy_logits, axis=1) == y_policy) 
            value_acc = np.sum((value > 0) == (y_value > 0)) 
            return policy_acc, value_acc

        def evaluate(variables, baseline, batch_size=1024):
            """A simplified evaluation by sampling. Only for debugging. 
            Please use MCTS and run tournaments for serious evaluation."""
            my_player = 0

            env = pgx.make('bughouse')
            rng_key = jax.random.PRNGKey(self.seed)
            key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, batch_size)
            state = jax.vmap(env.init)(keys)

            def body_fn(val):
                key, state, R = val
                my_logits, _ = self.model.apply(variables, state.observation, train=False)
                opp_logits, _ = self.model.apply(baseline, state.observation, train=False)
                is_my_turn = (state.current_player == my_player).reshape((-1, 1))
                logits = jnp.where(is_my_turn, my_logits, opp_logits)
                key, subkey = jax.random.split(key)
                action = jax.random.categorical(subkey, logits, axis=-1)
                keys = jax.random.split(subkey, batch_size)
                state = jax.vmap(env.step)(state, action, keys)
                R = R + state.rewards[jnp.arange(batch_size), my_player]
                return (key, state, R)

            _, _, R = jax.lax.while_loop(
                lambda x: ~(x[1].terminated.all()), body_fn, (key, state, jnp.zeros(batch_size))
            )
            return R

        # jit for efficiency
        #self.train_step = jax.jit(train_step)
        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)
        self.evaluate = jax.jit(evaluate)

    def train_loop(self, train_files, eval_files=None, epochs=7, batch_size=1024):
        """Training model with provided files"""
        baseline = extract_params(self.state)
        log = {} 

        for epoch in range(epochs):
            print(f'Training Epoch: {epoch + 1}/{epochs}')
            policy_losses, value_losses = [], []
            for file in tqdm(train_files):
                data = np.load(file)
                with tf.device('/CPU:0'):
                    train_loader = tf.data.Dataset.from_tensor_slices((data['obs'], data['policy_tgt'], data['value_tgt'], data['value_mask']))
                    train_loader = train_loader.batch(batch_size)
                for obs, policy_tgt, value_tgt, value_mask in train_loader:
                    obs = obs.numpy()
                    policy_tgt = policy_tgt.numpy()
                    value_tgt = value_tgt.numpy()
                    value_mask = value_mask.numpy()
                    self.state, policy_loss, value_loss = self.train_step(self.state, obs, policy_tgt, value_tgt, value_mask)
                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())

            policy_loss = sum(policy_losses) / len(policy_losses)
            value_loss = sum(value_losses) / len(value_losses)
            R = self.evaluate(extract_params(self.state), baseline)
            log.update(
                {
                    'train/policy_loss': policy_loss,
                    'train/value_loss': value_loss,
                    f'eval/vs_baseline/avg_R': R.mean().item(),
                    f'eval/vs_baseline/win_rate': ((R == 1).sum() / R.size).item(),
                    f'eval/vs_baseline/draw_rate': ((R == 0).sum() / R.size).item(),
                    f'eval/vs_baseline/lose_rate': ((R == -1).sum() / R.size).item(),
                }
            )
            self.save_checkpoint(step=epoch+1)
            print(log)
            wandb.log(log)

    def eval_model(self, files, batch_size):
        policy_acc = 0
        value_acc = 0

        for file in tqdm(files):
            data = np.load(file)
            with tf.device('/CPU:0'):
                val_loader = tf.data.Dataset.from_tensor_slices((data['obs'], data['policy_tgt'], data['value_tgt']))
                val_loader = val_loader.batch(batch_size)

            for obs, policy_tgt, value_tgt in val_loader:
                batch_policy_acc, batch_value_acc = self.eval_step(self.state, obs.numpy(), policy_tgt.numpy(), value_tgt.numpy())

                policy_acc += batch_policy_acc
                value_acc += batch_value_acc

        policy_acc /= len(val_loader) * batch_size * len(files)
        value_acc /= len(val_loader) * batch_size * len(files)

        return policy_acc, value_acc

    def save_checkpoint(self, step=0):
        # Save current model at certain training iteration
        options = ocp.CheckpointManagerOptions()
        with ocp.CheckpointManager(
            self.ckpt_dir,
            options=options,
            item_handlers=ocp.PyTreeCheckpointHandler()
        ) as mngr:
            mngr.save(step, args=ocp.args.PyTreeSave(self.state))
            mngr.wait_until_finished()
    
    def load_checkpoint(self, step=0) -> TrainState:
        # Load model. We use different checkpoint for pretrained models
        options = ocp.CheckpointManagerOptions()
        with ocp.CheckpointManager(
            self.ckpt_dir,
            options=options,
            item_handlers=ocp.PyTreeCheckpointHandler()
        ) as mngr:
            self.state = mngr.restore(step)
            return self.state

if __name__ == '__main__':
    from src.architectures.azresnet import AZResnet, AZResnetConfig
    import tensorflow as tf
    import glob

    wandb.init()

    trainer = TrainerModule(model_class=AZResnet, model_configs=AZResnetConfig(
        num_blocks=15,
        channels=256,
        policy_channels=4, 
        value_channels=8,
        num_policy_labels=2*64*78+1, 
    ), optimizer_name='lion', optimizer_params={'learning_rate': 8e-6}, x=jnp.ones((1024, 8, 16, 32)))
    trainer.load_checkpoint(0)
    trainer.init_optimizer()

    files = glob.glob('data/run2/*')
    trainer.train_loop(files, epochs=1) 