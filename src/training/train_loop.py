import argparse
import glob
import re
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
import polars as pl
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from configs.train_config import TrainConfig, TrainObjects, rl_train_config
from configs.main_config import main_config
from src.training.data_loaders import load_parquet_shard, load_rl_data_from_directory, load_rl_parquet_shard, RLDataset, StreamingRLDataset, CombinedRLDataset

from src.training.lr_schedules.lr_schedules import *
from src.architectures.rise_mobile_v3 import get_rise_v33_model
from src.training.trainer_agent import TrainerAgentPytorch, save_torch_state,\
    load_torch_state, export_to_onnx, get_context, get_data_loader, evaluate_metrics
from src.training.train_util import get_metrics, value_to_wdl_label, prepare_plys_label
from src.constants import NUM_BUGHOUSE_CHANNELS


def _resolve_validation_shard() -> str:
    preferred = Path('../../data/planes/val/evaluation_shard.parquet')
    if preferred.exists():
        return str(preferred)

    val_dir = Path(main_config['planes_val_dir'])
    parquet_files = sorted(val_dir.glob('*.parquet'))
    if parquet_files:
        return str(parquet_files[0])

    raise FileNotFoundError(
        f"No validation parquet shard found. Checked {preferred} and {val_dir}"
    )


def _resolve_train_evaluation_shard(shard_path: str = None) -> str:
    path = (
        Path(shard_path).expanduser().resolve()
        if shard_path
        else Path('../../data/planes/train_eval/evaluation_shard.parquet')
    )
    if not path.is_file():
        raise FileNotFoundError(f"Training evaluation shard not found: {path}")
    return str(path)


def get_model_args(train_config=None):
    """Get model configuration arguments."""
    class Args:
        def __init__(self):
            self.model_type = "risev33"
            self.input_version = "1.0"
            self.export_dir = "../../checkpoints"
            self.device_id = 0
            self.context = "gpu"
            self.input_shape = (NUM_BUGHOUSE_CHANNELS, 8, 8)
            self.n_labels = 0
            self.channels_policy_head = 73
            self.select_policy_from_plane = True
            self.use_wdl = bool(train_config and train_config.use_wdl)
            self.use_plys_to_end = bool(
                train_config and train_config.use_plys_to_end
            )
            self.use_mlp_wdl_ply = bool(
                train_config and train_config.use_mlp_wdl_ply
            )
            self.shared_policy_trunk = bool(train_config and train_config.use_wdl)
    return Args()


def _checkpoint_k_steps(checkpoint_path: Path) -> int:
    match = re.search(r"-(\d+)\.tar$", checkpoint_path.name)
    if match is None:
        raise ValueError(
            "Cannot infer training progress from checkpoint filename. "
            "Expected a name ending in '-<step>.tar'."
        )
    return int(match.group(1))


def train_supervised(
    checkpoint_path: str = None,
    train_eval_shard: str = None,
):
    """Run supervised learning training on human game data."""
    tc = TrainConfig()
    tc.use_wdl = True
    tc.use_plys_to_end = True
    tc.policy_loss_factor = 0.978
    checkpoint = None
    if checkpoint_path:
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        tc.k_steps_initial = _checkpoint_k_steps(checkpoint)
    to = TrainObjects()
    to.metrics = get_metrics(tc)

    train_shards = glob.glob(main_config['planes_train_dir'] + '*.parquet')
    tc.nb_parts = len(train_shards)
    if tc.nb_parts == 0:
        raise FileNotFoundError(
            f"No training parquet shards found in {main_config['planes_train_dir']}"
        )

    # Load validation data
    val_shard = _resolve_validation_shard()
    val_tensors = load_parquet_shard(
        val_shard,
        include_auxiliary=True,
    )
    dataset = TensorDataset(*val_tensors)
    val_data = DataLoader(dataset, batch_size=tc.batch_size, shuffle=False)

    train_eval_tensors = load_parquet_shard(
        _resolve_train_evaluation_shard(train_eval_shard),
        include_auxiliary=True,
    )
    train_eval_data = DataLoader(
        TensorDataset(*train_eval_tensors),
        batch_size=tc.batch_size,
        shuffle=False,
    )

    nb_it_per_epoch = (2**16 * tc.nb_parts) // tc.batch_size
    tc.total_it = int(nb_it_per_epoch * tc.nb_training_epochs)

    to.lr_schedule = OneCycleSchedule(start_lr=tc.max_lr / 8, max_lr=tc.max_lr, cycle_length=tc.total_it * .3,
                                      cooldown_length=tc.total_it * .6, finish_lr=tc.min_lr)
    to.lr_schedule = LinearWarmUp(to.lr_schedule, start_lr=tc.min_lr, length=tc.total_it / 30)
    to.momentum_schedule = MomentumSchedule(to.lr_schedule, tc.min_lr, tc.max_lr, tc.min_momentum, tc.max_momentum)

    args = get_model_args(tc)
    model = get_rise_v33_model(args)

    trainer = TrainerAgentPytorch(
        model,
        val_data,
        tc,
        to,
        use_rtpt=True,
        is_rl=False,
        train_eval_loader=train_eval_data,
    )
    if checkpoint is not None:
        print(
            f"Resuming supervised training from {checkpoint} "
            f"at step {tc.k_steps_initial} ({tc.k_steps_initial * tc.batch_steps} batches)"
        )
        load_torch_state(model, trainer.optimizer, checkpoint, tc.device_id)
    trainer.train(cur_it=tc.k_steps_initial * tc.batch_steps)


def train_rl(rl_data_dir: str, val_data_dir: str, checkpoint_path: str = None, augment_flip: bool = True):
    """
    Run RL training on self-play data.
    
    Args:
        rl_data_dir: Directory containing RL parquet files (converted from binary)
        val_data_dir: Directory containing validation parquet files
        checkpoint_path: Optional path to load model weights from
        augment_flip: If True, use board flip augmentation to double training data
    """
    import glob
    from pathlib import Path as PathLib
    
    tc = rl_train_config()
    to = TrainObjects()
    to.metrics = get_metrics(tc)
    
    # Set export directory and ensure it exists
    tc.export_dir = str(project_root / "src/training/")
    weights_dir = Path(tc.export_dir) / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(tc.export_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure ONNX export is enabled
    tc.export_weights = True
    
    # Find all training parquet files
    parquet_files = sorted(glob.glob(str(PathLib(rl_data_dir) / "*.parquet")))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {rl_data_dir}")
    
    print(f"Found {len(parquet_files)} training parquet files in {rl_data_dir}")
    
    # Load validation data from separate directory
    val_parquet_files = sorted(glob.glob(str(PathLib(val_data_dir) / "*.parquet")))
    if not val_parquet_files:
        raise ValueError(f"No parquet files found in validation directory {val_data_dir}")
    
    print(f"Found {len(val_parquet_files)} validation parquet files in {val_data_dir}")
    print(f"Loading validation data...")
    
    val_samples = []
    for vf in val_parquet_files:
        x, y_val, pol_a, pol_b = load_rl_parquet_shard(vf)
        for i in range(len(x)):
            val_samples.append((x[i], y_val[i], pol_a[i], pol_b[i]))
    
    if not val_samples:
        raise ValueError(f"No validation samples found in {val_data_dir}")
    
    # Convert validation to tensors
    x_val = torch.stack([s[0] for s in val_samples])
    y_val = torch.stack([s[1] for s in val_samples])
    pol_a_val = torch.stack([s[2] for s in val_samples])
    pol_b_val = torch.stack([s[3] for s in val_samples])
    
    val_dataset = RLDataset(x_val, y_val, pol_a_val, pol_b_val, augment_flip=augment_flip)
    val_loader = DataLoader(val_dataset, batch_size=tc.batch_size, shuffle=False)
    print(f"Loaded {len(val_dataset)} validation samples")
    
    # Estimate training samples
    estimated_samples_per_shard = 16384  # Typical shard size
    estimated_training_samples = len(parquet_files) * estimated_samples_per_shard
    if augment_flip:
        estimated_training_samples *= 2  # Double with augmentation
    n_train = estimated_training_samples
    
    print(f"Board flip augmentation: {'ENABLED' if augment_flip else 'DISABLED'}")
    if augment_flip:
        print(f"Training data will be doubled through board flip augmentation")
    print(f"Estimated training samples: ~{n_train}")
    
    # Create streaming training dataset from all training files
    train_dataset = StreamingRLDataset(parquet_files, shuffle_files=True, shuffle_buffer_size=10000, augment_flip=augment_flip)
    train_loader = DataLoader(train_dataset, batch_size=tc.batch_size, num_workers=0)
    
    # Calculate iterations (approximate)
    nb_it_per_epoch = max(1, n_train // tc.batch_size)
    tc.total_it = int(nb_it_per_epoch * tc.nb_training_epochs)
    tc.nb_parts = len(parquet_files)
    
    print(f"Iterations per epoch: ~{nb_it_per_epoch}, Total iterations: ~{tc.total_it}")
    
    # LR schedule: Cosine Annealing with 25% warm-up (as per CrazyAra RL paper)
    # The warm-up helps with context drift in the training data
    cosine_schedule = CosineAnnealingSchedule(min_lr=tc.min_lr, max_lr=tc.max_lr, cycle_length=tc.total_it)
    to.lr_schedule = LinearWarmUp(cosine_schedule, start_lr=tc.min_lr, length=int(tc.total_it * 0.25))
    to.momentum_schedule = MomentumSchedule(to.lr_schedule, tc.min_lr, tc.max_lr, tc.min_momentum, tc.max_momentum)
    
    # Load model
    args = get_model_args(tc)
    model = get_rise_v33_model(args)
    
    # Optionally load checkpoint
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Train with streaming RL data loader
    trainer = TrainerAgentPytorch(model, val_loader, tc, to, use_rtpt=True, 
                                  is_rl=True, rl_train_loader=train_loader)
    trainer.train()
    
    # Export final model to ONNX
    print("\nExporting final model to ONNX...")
    ctx = get_context(tc.context, tc.device_id)
    dummy_input = torch.zeros(1, NUM_BUGHOUSE_CHANNELS, 8, 8).to(ctx)
    model_prefix = f"model-rl-final"
    export_to_onnx(model, 1, dummy_input, weights_dir, model_prefix, False, True)
    print(f"ONNX model exported to {weights_dir}/{model_prefix}.onnx")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Hivemind neural network')
    parser.add_argument('--mode', type=str, default='sl', choices=['sl', 'rl'],
                        help='Training mode: sl (supervised learning) or rl (reinforcement learning)')
    parser.add_argument('--rl-data-dir', type=str, default='../../engine/selfplay_games/training_data_parquet',
                        help='Directory containing RL training parquet files')
    parser.add_argument('--val-data-dir', type=str, default='/home/ben/hivemind/engine/selfplay_games/val_data_parquet',
                        help='Directory containing RL validation parquet files')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--train-eval-shard', type=str, default=None,
                        help='Fixed representative training shard used only for metrics')
    
    args = parser.parse_args()

    if args.mode == 'sl':
        train_supervised(
            checkpoint_path=args.checkpoint,
            train_eval_shard=args.train_eval_shard,
        )
    else:
        train_rl(args.rl_data_dir, args.val_data_dir, checkpoint_path=args.checkpoint)
