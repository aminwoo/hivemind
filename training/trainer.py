import os
import numpy as np
from typing import Any
from tqdm.auto import tqdm

import jax.numpy as jnp
import jax
import flax
import optax
import orbax
import chex

from flax.training import orbax_utils
from flax.training.train_state import TrainState
from flax import linen as nn
from flax.training import train_state


def extract_params(state: TrainState) -> chex.ArrayTree:
    if hasattr(state, 'batch_stats'):
        return {'params': state.params, 'batch_stats': state.batch_stats}
    return {'params': state.params}

class TrainState(train_state.TrainState):
    batch_stats: chex.ArrayTree


class TrainerModule:

    def __init__(
        self,
        model_name: str,
        model_class: nn.Module,
        model_configs: Any,
        optimizer_name: str,
        optimizer_params: dict,
        x: Any,
        ckpt_dir: str = '/tmp/checkpoints',
        max_checkpoints: int = 2,
        seed=42,
    ):
        '''
        Module for summarizing all training functionalities for classification on CIFAR10.

        Inputs:
            model_name - String of the class name, used for logging and saving
            model_class - Class implementing the neural network
            model_hparams - Hyperparameters of the model, used as input to model constructor
            optimizer_name - String of the optimizer name, supporting ['sgd', 'adam', 'adamw']
            optimizer_params - Hyperparameters of the optimizer, including learning rate as 'lr'
            x - Example imgs, used as input to initialize the model
            seed - Seed to use in the model initialization
        '''
        super().__init__()
        self.model_name = model_name
        self.model_class = model_class
        self.model_configs = model_configs
        self.optimizer_name = optimizer_name
        self.optimizer_params = optimizer_params
        self.seed = seed
        # Create empty model. Note: no parameters yet
        self.model = self.model_class(model_configs)

        self.ckpt_dir = ckpt_dir
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(
            max_to_keep=max_checkpoints, create=True
        )
        self.checkpoint_manager = orbax.checkpoint.CheckpointManager(
            self.ckpt_dir, orbax_checkpointer, options
        )

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
            params=self.init_params if self.state is None else self.state.params,
            batch_stats=(
                self.init_batch_stats if self.state is None else self.state.batch_stats
            ),
            tx=opt_class(**self.optimizer_params),
        )

    def create_functions(self):
        # Function to calculate the classification loss and accuracy for a model
        def train_step(state: TrainState, board_planes, y_policy, y_value):
            def calculate_loss(params, batch_stats, train):
                logits, new_model_state = state.apply_fn(
                    {'params': params, 'batch_stats': batch_stats},
                    board_planes,
                    train=train,
                    mutable=['batch_stats'],
                )

                policy_logits, value_logits = logits
                policy_loss = optax.softmax_cross_entropy_with_integer_labels(
                    logits=policy_logits[0], labels=y_policy[:, 0]
                ).mean()
                policy_loss += optax.softmax_cross_entropy_with_integer_labels(
                    logits=policy_logits[1], labels=y_policy[:, 1]
                ).mean()
                value_loss = optax.l2_loss(value_logits, y_value).mean()
                loss = 0.5 * policy_loss + 0.01 * value_loss

                return loss, (new_model_state)

            loss_fn = lambda params: calculate_loss(
                params, state.batch_stats, train=True
            )

            # Get loss, gradients for loss, and other outputs of loss function
            (loss, (new_model_state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params
            )
            # Update parameters and batch statistics
            state = state.apply_gradients(
                grads=grads, batch_stats=new_model_state['batch_stats']
            )
            return state, loss

        # jit for efficiency
        self.train_step = jax.jit(train_step)

    def train_model(self, train_loader):
        for board_planes, y_policy, y_value in train_loader:
            self.state, loss = self.train_step(self.state, board_planes.numpy(), y_policy.numpy().astype(np.int32), y_value.numpy())

    def eval_model(self, val_loader, batch_size):
        policy_acc = 0
        value_acc = 0

        for board_planes, y_policy, y_value in val_loader:
            policy_logits, value = self.state.apply_fn(
                {'params': self.state.params, 
                 'batch_stats': self.state.batch_stats},
                board_planes,
                train=False,
            )

            policy_acc += np.sum(np.argmax(policy_logits[0], axis=1) == y_policy[:,0]) 
            policy_acc += np.sum(np.argmax(policy_logits[1], axis=1) == y_policy[:,1]) 
            value_acc += np.sum((value > 0) == (y_value > 0)) 

        policy_acc /= len(val_loader) * batch_size * 2
        value_acc /= len(val_loader) * batch_size

        return policy_acc, value_acc

    def save_checkpoint(self, train_state: TrainState, step=0):
        # Save current model at certain training iteration
        ckpt = {'train_state': train_state}
        save_args = orbax_utils.save_args_from_target(ckpt)
        self.checkpoint_manager.save(step, ckpt, save_kwargs={'save_args': save_args})

    def load_checkpoint(self, checkpoint_path: str) -> TrainState:
        # Load model. We use different checkpoint for pretrained models
        ckpt = self.checkpoint_manager.restore(checkpoint_path)
        return ckpt['train_state']

if __name__ == '__main__':
    import jax.numpy as jnp 
    from architectures.azresnet import AZResnet, AZResnetConfig
    from constants import POLICY_LABELS, BOARD_HEIGHT, BOARD_WIDTH, NUM_BUGHOUSE_CHANNELS
    import tensorflow as tf
    import glob

    batch_size = 1024

    trainer = TrainerModule(model_name='AZResNet', model_class=AZResnet, model_configs=AZResnetConfig(
        num_blocks=15,
        channels=256,
        policy_channels=4, 
        value_channels=8,
        num_policy_labels=len(POLICY_LABELS)
    ), optimizer_name='lion', optimizer_params={'learning_rate': 0.00001}, x=jnp.ones((batch_size, BOARD_HEIGHT, 2 * BOARD_WIDTH, NUM_BUGHOUSE_CHANNELS)))
    trainer.init_optimizer()

    data = np.load('data/fics_training_data/checkpoint0.npz')
    with tf.device('/CPU:0'):
        val_loader = tf.data.Dataset.from_tensor_slices((data['board_planes'], data['move_planes'], data['value_planes']))
        val_loader = val_loader.shuffle(buffer_size=2**16).batch(batch_size)

    steps = 0
    eval_steps = 100

    for path in tqdm(glob.glob('data/*training_data/*')):
        data = np.load(path)
        with tf.device('/CPU:0'):
            train_loader = tf.data.Dataset.from_tensor_slices((data['board_planes'], data['move_planes'], data['value_planes']))
            train_loader = train_loader.shuffle(buffer_size=2**16).batch(batch_size)
        trainer.train_model(train_loader) 

        if steps % eval_steps == 0:
            policy_acc, value_acc =  trainer.eval_model(val_loader, batch_size)
            print(policy_acc, value_acc)

        steps += 1
    
    trainer.save_checkpoint(trainer.state)