import os
from typing import Any

import numpy as np 
from tqdm.auto import tqdm

import jax.numpy as jnp
import jax
import flax
from flax import linen as nn
from flax.training import train_state
import optax
from flax.training import orbax_utils

from collections import defaultdict

from architectures.azresnet import AZResnet, AZResnetConfig
from constants import POLICY_LABELS

from flax.training.train_state import TrainState
import orbax
import orbax.checkpoint as ocp

from torch.utils.data import DataLoader
import chex

from jax import grad, jit, vmap

from constants import (BOARD_HEIGHT, BOARD_WIDTH, NUM_BUGHOUSE_CHANNELS)

class TrainState(train_state.TrainState):
    batch_stats: chex.ArrayTree

class TrainerModule:

    def __init__(self,
                 model_name: str,
                 model_class: nn.Module,
                 model_configs: Any,
                 optimizer_name: str,
                 optimizer_hparams: dict,
                 input: Any,
                 ckpt_dir: str = "/tmp/checkpoints",
                 max_checkpoints: int = 2,
                 seed=42):
        """
        Module for summarizing all training functionalities for classification on CIFAR10.

        Inputs:
            model_name - String of the class name, used for logging and saving
            model_class - Class implementing the neural network
            model_hparams - Hyperparameters of the model, used as input to model constructor
            optimizer_name - String of the optimizer name, supporting ['sgd', 'adam', 'adamw']
            optimizer_hparams - Hyperparameters of the optimizer, including learning rate as 'lr'
            exmp_imgs - Example imgs, used as input to initialize the model
            seed - Seed to use in the model initialization
        """
        super().__init__()
        self.model_name = model_name
        self.model_class = model_class
        self.model_configs = model_configs
        self.optimizer_name = optimizer_name
        self.optimizer_hparams = optimizer_hparams
        self.seed = seed
        # Create empty model. Note: no parameters yet
        self.model = self.model_class(model_configs)

        self.ckpt_dir = ckpt_dir
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=max_checkpoints, create=True)
        self.checkpoint_manager = orbax.checkpoint.CheckpointManager(
            self.ckpt_dir, orbax_checkpointer, options)
        
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model(input)

    def init_model(self, input):
        # Initialize model
        init_rng = jax.random.PRNGKey(self.seed)
        variables = self.model.init(init_rng, input, train=True)
        self.init_params, self.init_batch_stats = variables['params'], variables['batch_stats']
        self.state = None

    def init_optimizer(self):
        # Initialize learning rate schedule and optimizer
        if self.optimizer_name.lower() == 'adam':
            opt_class = optax.adam
        elif self.optimizer_name.lower() == 'adamw':
            opt_class = optax.adamw
        elif self.optimizer_name.lower() == 'sgd':
            opt_class = optax.sgd
        else:
            assert False, f'Unknown optimizer "{opt_class}"'

        # Initialize training state
        self.state = TrainState.create(apply_fn=self.model.apply,
                                       params=self.init_params if self.state is None else self.state.params,
                                       batch_stats=self.init_batch_stats if self.state is None else self.state.batch_stats,
                                       tx=opt_class(**self.optimizer_hparams))


    def create_functions(self):
        # Function to calculate the classification loss and accuracy for a model
        def calculate_loss(params, batch_stats, batch, train):
            board_planes, policy, value = batch
            batched_predict = vmap(self.model.apply, in_axes=(None, 0))
            print(board_planes.shape)
            # Run model. During training, we need to update the BatchNorm statistics.
            out = batched_predict({'params': params, 'batch_stats': batch_stats},
                                    board_planes,
                                    train=train,
                                    mutable=['batch_stats'] if train else False)
            print(out)


            logits, new_model_state = outs if train else (outs, None)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
            acc = (logits.argmax(axis=-1) == labels).mean()
            return loss, (acc, new_model_state)
        # Training function
        def train_step(state, batch):
            loss_fn = lambda params: calculate_loss(params, state.batch_stats, batch, train=True)
            # Get loss, gradients for loss, and other outputs of loss function
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            loss, acc, new_model_state = ret[0], *ret[1]
            # Update parameters and batch statistics
            state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
            return state, loss, acc
        # Eval function
        def eval_step(state, batch):
            # Return the accuracy for a single batch
            _, (acc, _) = calculate_loss(state.params, state.batch_stats, batch, train=False)
            return acc
        # jit for efficiency
        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)

    def train_epoch(self, train_set, epoch):
        # Train model for one epoch, and log avg loss and accuracy
        while train_set.load_chunk():
            train_loader = DataLoader(train_set, batch_size=1024, shuffle=True, num_workers=8)
            for batch in train_loader:
                self.state, loss, acc = self.train_step(self.state, list(map(lambda x: jnp.float32(x), batch)))

    def train_model(self, train_loader, val_loader, num_epochs=200):
        # Train model for defined number of epochs
        # We first need to create optimizer and the scheduler for the given number of epochs
        self.init_optimizer(num_epochs, len(train_loader))
        # Track best eval accuracy
        best_eval = 0.0
        for epoch_idx in tqdm(range(1, num_epochs+1)):
            self.train_epoch(train_loader, epoch=epoch_idx)
            if epoch_idx % 2 == 0:
                eval_acc = self.eval_model(val_loader)
                self.logger.add_scalar('val/acc', eval_acc, global_step=epoch_idx)
                if eval_acc >= best_eval:
                    best_eval = eval_acc
                    self.save_model(step=epoch_idx)
                self.logger.flush()

    def eval_model(self, val_set):
        # Test model on all images of a data loader and return avg loss
        correct_class, count = 0, 0

        while val_set.load_chunk(1024):
            dataloader = DataLoader(val_set, batch_size=1024, shuffle=False)
            for board_planes, (y_value, y_policy) in dataloader:

                val_policy_acc += torch.sum(torch.argmax(policy, dim=1) == torch.argmax(y_policy, dim=1)) / \
                                policy.size()[0]

                steps += 1

        for board_planes, (y_value, y_policy) in data_loader:
            acc = self.eval_step(self.state, batch)
            correct_class += acc * batch[0].shape[0]
            count += batch[0].shape[0]
        eval_acc = (correct_class / count).item()
        return eval_acc

    def save_checkpoint(self, step=0):
        # Save current model at certain training iteration
        ckpt = {'train_state': self.state}
        save_args = orbax_utils.save_args_from_target(ckpt)
        self.checkpoint_manager.save(step, ckpt, save_kwargs={'save_args': save_args})

    def load_checkpoint(self):
        # Load model. We use different checkpoint for pretrained models
        self.state = self.checkpoint_manager.restore(self.ckpt_dir + "/0")['train_state']
    

trainer = TrainerModule(model_name="AZResNet", model_class=AZResnet, model_configs=AZResnetConfig(
    num_blocks=15,
    channels=256,
    policy_channels=4, 
    value_channels=8,
    num_policy_labels=len(POLICY_LABELS)
), optimizer_name='adam', optimizer_hparams={'learning_rate': 0.0001}, input=jnp.ones((BOARD_HEIGHT, 2 * BOARD_WIDTH, NUM_BUGHOUSE_CHANNELS)))

trainer.init_optimizer()

from loader import BughouseDataset
train_set = BughouseDataset("data/games.json")
trainer.train_epoch(train_set, 1)

from torch.utils.data import DataLoader
while generator.load_chunk():
    train_loader = DataLoader(generator, batch_size=1024, shuffle=True, num_workers=8)
    for board_planes, (y_policy, y_value) in train_loader:
        print(jnp.float32(board_planes))
        #print(board_planes.shape, y_policy.shape, y_value.shape)
#trainer.save_checkpoint()
#trainer.load_checkpoint()
