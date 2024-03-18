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

from torch.utils.data import DataLoader


class TrainState(train_state.TrainState):
    batch_stats: chex.ArrayTree


class TrainerModule:

    def __init__(
        self,
        model_name: str,
        model_class: nn.Module,
        model_configs: Any,
        optimizer_name: str,
        optimizer_hparams: dict,
        x: Any,
        ckpt_dir: str = "/tmp/checkpoints",
        max_checkpoints: int = 2,
        seed=42,
    ):
        """
        Module for summarizing all training functionalities for classification on CIFAR10.

        Inputs:
            model_name - String of the class name, used for logging and saving
            model_class - Class implementing the neural network
            model_hparams - Hyperparameters of the model, used as input to model constructor
            optimizer_name - String of the optimizer name, supporting ['sgd', 'adam', 'adamw']
            optimizer_hparams - Hyperparameters of the optimizer, including learning rate as 'lr'
            x - Example imgs, used as input to initialize the model
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
            variables["params"],
            variables["batch_stats"],
        )
        self.state = None

    def init_optimizer(self):
        # Initialize learning rate schedule and optimizer
        if self.optimizer_name.lower() == "adam":
            opt_class = optax.adam
        elif self.optimizer_name.lower() == "adamw":
            opt_class = optax.adamw
        elif self.optimizer_name.lower() == "sgd":
            opt_class = optax.sgd
        elif self.optimizer_name.lower() == "lion":
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
            tx=opt_class(**self.optimizer_hparams),
        )

    def create_functions(self):
        # Function to calculate the classification loss and accuracy for a model
        def train_step(state: TrainState, batch: jnp.ndarray):

            def calculate_loss(params, batch_stats, batch, train):
                board_planes, y_policy, y_value = batch
                logits, new_model_state = state.apply_fn(
                    {"params": params, "batch_stats": batch_stats},
                    board_planes,
                    train=train,
                    mutable=["batch_stats"],
                )
                print(policy_logits[0].shape, y_policy[:, 0, :].shape)
                policy_logits, value_logits = logits
                policy_loss = optax.softmax_cross_entropy(
                    logits=policy_logits[0], labels=y_policy[:, 0, :]
                ).mean()
                policy_loss += optax.softmax_cross_entropy(
                    logits=policy_logits[1], labels=y_policy[:, 1, :]
                ).mean()
                value_loss = optax.l2_loss(value_logits, y_value).mean()
                loss = 0.5 * policy_loss + 0.01 * value_loss

                return loss, new_model_state

            loss_fn = lambda params: calculate_loss(
                params, state.batch_stats, batch, train=True
            )

            # Get loss, gradients for loss, and other outputs of loss function
            (loss, new_model_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params
            )
            # Update parameters and batch statistics
            state = state.apply_gradients(
                grads=grads, batch_stats=new_model_state["batch_stats"]
            )
            return state, loss

        # jit for efficiency
        self.train_step = jax.jit(train_step)

    def train_epoch(self, train_set):
        # Train model for one epoch, and log avg loss and accuracy
        while train_set.load_chunk():
            train_loader = DataLoader(train_set, batch_size=1024, shuffle=True)
            for batch in train_loader:
                batch = list(map(lambda x: np.array(x), batch))
                self.state, loss = self.train_step(self.state, batch)

    def train_model(self, train_loader, val_loader, num_epochs=200):
        # Train model for defined number of epochs
        # We first need to create optimizer and the scheduler for the given number of epochs
        self.init_optimizer(num_epochs, len(train_loader))
        # Track best eval accuracy
        best_eval = 0.0
        for epoch_idx in tqdm(range(1, num_epochs + 1)):
            self.train_epoch(train_loader, epoch=epoch_idx)
            if epoch_idx % 2 == 0:
                eval_acc = self.eval_model(val_loader)
                self.logger.add_scalar("val/acc", eval_acc, global_step=epoch_idx)
                if eval_acc >= best_eval:
                    best_eval = eval_acc
                    self.save_model(step=epoch_idx)
                self.logger.flush()

    def save_checkpoint(self, step=0):
        # Save current model at certain training iteration
        ckpt = {"train_state": self.state}
        save_args = orbax_utils.save_args_from_target(ckpt)
        self.checkpoint_manager.save(step, ckpt, save_kwargs={"save_args": save_args})

    def load_checkpoint(self):
        # Load model. We use different checkpoint for pretrained models
        self.state = self.checkpoint_manager.restore(self.ckpt_dir + "/0")[
            "train_state"
        ]
