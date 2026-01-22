import glob
import polars as pl
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from configs.train_config import TrainConfig, TrainObjects
from configs.main_config import main_config
from src.training.data_loaders import load_parquet_shard

from src.training.lr_schedules.lr_schedules import *
from src.architectures.rise_mobile_v3 import get_rise_v33_model
from src.training.trainer_agent import TrainerAgentPytorch, save_torch_state,\
    load_torch_state, export_to_onnx, get_context, get_data_loader, evaluate_metrics
from src.training.train_util import get_metrics, value_to_wdl_label, prepare_plys_label


if __name__ == '__main__':
    tc = TrainConfig()
    to = TrainObjects()
    to.metrics = get_metrics(tc)

    tc.nb_parts = len(glob.glob(main_config['planes_train_dir'] + '*'))

    # Load validation data using load_parquet_shard instead of numpy
    # Assuming you have a validation parquet file, replace the path as needed
    x_val, y_val_value, y_val_policy = load_parquet_shard('../../data/planes/val/evaluation_shard.parquet')
    dataset = TensorDataset(x_val, y_val_value, y_val_policy)
    val_data = DataLoader(dataset, batch_size=tc.batch_size, shuffle=False)

    nb_it_per_epoch = (2**16 * tc.nb_parts) // tc.batch_size  # calculate how many iterations per epoch exist
    # one iteration is defined by passing 1 batch and doing backprop
    tc.total_it = int(nb_it_per_epoch * tc.nb_training_epochs)

    to.lr_schedule = OneCycleSchedule(start_lr=tc.max_lr / 8, max_lr=tc.max_lr, cycle_length=tc.total_it * .3,
                                      cooldown_length=tc.total_it * .6, finish_lr=tc.min_lr)
    to.lr_schedule = LinearWarmUp(to.lr_schedule, start_lr=tc.min_lr, length=tc.total_it / 30)
    to.momentum_schedule = MomentumSchedule(to.lr_schedule, tc.min_lr, tc.max_lr, tc.min_momentum, tc.max_momentum)

    class Args:
        def __init__(self):
            # Model type, e.g., "risev33" (specific architecture or version of the model)
            self.model_type = "risev33"

            # Input version, e.g., "1.0" (version of the input data or model configuration)
            self.input_version = "1.0"

            # Directory where the model checkpoints will be exported or saved
            self.export_dir = "../../checkpoints"

            # Device ID for running the model (e.g., GPU device ID)
            self.device_id = 0

            # Context in which the model will run, e.g., "gpu" or "cpu"
            self.context = "gpu"

            # Input shape of the model, represented as (channels, height, width)
            # Example: 64 channels, 8 rows (height), and 8 columns (width) for an 8x8 board
            self.input_shape = (64, 8, 8)

            # Number of labels for the policy head (output layer for move predictions)
            # Example: 4672 possible moves or actions in the policy output
            self.n_labels = 0

            # Number of channels in the policy head (output layer for move predictions)
            # Example: 75 channels in the policy head
            self.channels_policy_head = 73

            # Whether to select the policy directly from the plane (spatial output)
            # If True, the policy is derived from the spatial dimensions of the output
            self.select_policy_from_plane = True

            # Whether to use a Win/Draw/Loss (WDL) head
            # If True, the model will predict the game outcome (win, draw, or loss)
            self.use_wdl = False

            # Whether to use a "plys to end" head
            # If True, the model will predict the number of plies (half-moves) remaining until the end of the game
            self.use_plys_to_end = False

            # Whether to use a Multi-Layer Perceptron (MLP) for the WDL and "plys to end" heads
            # If True, an MLP will be used to process these outputs instead of a simpler method
            self.use_mlp_wdl_ply = False


    # Create an instance of the Args class
    args = Args()

    model = get_rise_v33_model(args)

    trainer = TrainerAgentPytorch(model, val_data, tc, to, True)
    trainer.train()