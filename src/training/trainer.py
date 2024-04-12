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

from flax.training import orbax_utils
from flax.training.train_state import TrainState
from flax import linen as nn
from flax.training import train_state


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
        ckpt_dir: str = os.getcwd() + '/checkpoints',
        max_checkpoints: int = 999,
        seed=42,
    ):
        '''
        Module for summarizing all training functionalities for classification.

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
            params=self.init_params if self.state is None else self.state['params'],
            batch_stats=(
                self.init_batch_stats if self.state is None else self.state['batch_stats']
            ),
            tx=opt_class(**self.optimizer_params),
        )

    def create_functions(self):
        # Function to calculate the classification loss and accuracy for a model
        def train_step(state: TrainState, obs, policy_tgt, value_tgt):
            def calculate_loss(params, batch_stats):
                logits, new_model_state = state.apply_fn(
                    {'params': params, 'batch_stats': batch_stats},
                    obs,
                    train=True,
                    mutable=['batch_stats'],
                )
                policy_logits, value = logits
                policy_loss = optax.softmax_cross_entropy(policy_logits, policy_tgt).mean()

                value_loss = optax.l2_loss(value, value_tgt).mean()
                loss = policy_loss + value_loss

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

        # jit for efficiency
        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)

    def train_loop(self, train_files, eval_files=None, epochs=7, batch_size=1024):
        log = {} 
        for epoch in range(epochs):
            print(f'Training Epoch: {epoch}/{epochs}')
            policy_losses, value_losses = [], []
            for file in tqdm(train_files):
                data = np.load(file)
                with tf.device('/CPU:0'):
                    train_loader = tf.data.Dataset.from_tensor_slices((data['obs'], data['policy_tgt'], data['value_tgt']))
                    train_loader = train_loader.shuffle(buffer_size=2**16).batch(batch_size)
                for obs, policy_tgt, value_tgt in train_loader:
                    self.state, policy_loss, value_loss = self.train_step(self.state, obs.numpy(), policy_tgt.numpy(), value_tgt.numpy())
                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())

            policy_loss = sum(policy_losses) / len(policy_losses)
            value_loss = sum(value_losses) / len(value_losses)
            log.update(
                {
                    "train/policy_loss": policy_loss,
                    "train/value_loss": value_loss,
                }
            )
            print(log)
            self.save_checkpoint(self.state, epoch=epoch)

    def eval_model(self, files, batch_size):
        policy_acc = 0
        value_acc = 0

        for file in tqdm(files):
            data = np.load(file)
            with tf.device('/CPU:0'):
                val_loader = tf.data.Dataset.from_tensor_slices((data['obs'], data['policy_tgt'], data['value_tgt']))
                val_loader = val_loader.shuffle(buffer_size=2**16).batch(batch_size)

            for obs, policy_tgt, value_tgt in val_loader:
                batch_policy_acc, batch_value_acc = self.eval_step(self.state, obs.numpy(), policy_tgt.numpy(), value_tgt.numpy())

                policy_acc += batch_policy_acc
                value_acc += batch_value_acc

        policy_acc /= len(val_loader) * batch_size * len(files)
        value_acc /= len(val_loader) * batch_size * len(files)

        return policy_acc, value_acc

    def save_checkpoint(self, train_state: TrainState, epoch=0):
        # Save current model at certain training iteration
        ckpt = {'train_state': train_state}
        save_args = orbax_utils.save_args_from_target(ckpt)
        self.checkpoint_manager.save(epoch, ckpt, save_kwargs={'save_args': save_args})

    def load_checkpoint(self, checkpoint_path: str) -> TrainState:
        # Load model. We use different checkpoint for pretrained models
        ckpt = self.checkpoint_manager.restore(checkpoint_path)
        return ckpt['train_state']

if __name__ == '__main__':
    from src.architectures.azresnet import AZResnet, AZResnetConfig
    import tensorflow as tf
    import glob

    trainer = TrainerModule(model_name='AZResNet', model_class=AZResnet, model_configs=AZResnetConfig(
        num_blocks=15,
        channels=256,
        policy_channels=4, 
        value_channels=8,
        num_policy_labels=2*64*78+1, 
    ), optimizer_name='lion', optimizer_params={'learning_rate': 0.00001}, x=jnp.ones((1024, 8, 16, 32)))
    print(os.getcwd() + '/checkpoints')
    trainer.save_checkpoint(trainer.state)
    #trainer.state = trainer.load_checkpoint('20240406121656')
    #trainer.init_optimizer()

    #files = glob.glob('data/run1/*')
    #trainer.train_loop(files, epochs=1) 