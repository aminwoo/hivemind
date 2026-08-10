import argparse
import glob
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
from src.preprocessing.convert_selfplay_data import (
    DEFAULT_REPLAY_FILES,
    DEFAULT_REPLAY_SELECTION_FRACTION,
    DEFAULT_RL_VALIDATION_FRACTION,
    convert_to_split_parquet,
)

from src.training.lr_schedules.lr_schedules import *
from src.architectures.rise_mobile_v3 import (
    get_cross_board_rise_v33_model,
    get_dual_stream_memory_rise_v33_model,
    get_rise_v33_model,
)
from src.training.trainer_agent import TrainerAgentPytorch, save_torch_state,\
    restore_torch_state, export_to_onnx, get_context, get_data_loader, evaluate_metrics
from src.training.train_util import get_metrics, value_to_wdl_label, prepare_plys_label
from src.constants import NUM_BUGHOUSE_CHANNELS


TRAINING_OUTPUT_DIR = project_root / "src" / "training"
SUPERVISED_WEIGHTS_DIR = TRAINING_OUTPUT_DIR / "weights" / "supervised"
RL_WEIGHTS_DIR = TRAINING_OUTPUT_DIR / "weights" / "rl"
MODEL_FACTORIES = {
    "risev33": get_rise_v33_model,
    "crossboard-risev33": get_cross_board_rise_v33_model,
    "dualstream-memory-risev33": get_dual_stream_memory_rise_v33_model,
}
CROSS_BOARD_DEFAULT_BATCH_SIZE = 256
DUAL_STREAM_DEFAULT_BATCH_SIZE = 256
SUPERVISED_EVAL_BATCHES = 64
SUPERVISED_EVAL_INTERVAL_MULTIPLIER = 2


def _resolve_validation_shard(validation_shard: str = None) -> str:
    if validation_shard:
        path = Path(validation_shard).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Validation shard not found: {path}")
        return str(path)

    preferred = project_root / 'data' / 'planes' / 'val' / 'evaluation_shard.parquet'
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
    if shard_path:
        path = Path(shard_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Training evaluation shard not found: {path}")
        return str(path)

    preferred = (
        project_root / 'data' / 'planes' / 'train_eval' /
        'evaluation_shard.parquet'
    )
    if preferred.is_file():
        return str(preferred)

    train_dir = Path(main_config['planes_train_dir'])
    parquet_files = sorted(train_dir.glob('*.parquet'))
    if parquet_files:
        return str(parquet_files[0])

    raise FileNotFoundError(
        f"No training evaluation shard found. Checked {preferred} and {train_dir}"
    )


def get_model_args(train_config=None, architecture="risev33"):
    """Get model configuration arguments."""
    class Args:
        def __init__(self):
            self.model_type = architecture
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


def create_model(args):
    return MODEL_FACTORIES[args.model_type](args)


def configure_batch_size(train_config, architecture, batch_size=None):
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive")

    original_batch_size = train_config.batch_size
    selected_batch_size = batch_size
    if selected_batch_size is None:
        selected_batch_size = {
            "crossboard-risev33": CROSS_BOARD_DEFAULT_BATCH_SIZE,
            "dualstream-memory-risev33": DUAL_STREAM_DEFAULT_BATCH_SIZE,
        }.get(architecture)
    if selected_batch_size is None:
        return

    train_config.batch_size = selected_batch_size
    train_config.batch_steps = max(
        1,
        round(
            train_config.batch_steps
            * original_batch_size
            / selected_batch_size
        ),
    )
    print(
        f"Training batch size: {train_config.batch_size} "
        f"(evaluate every {train_config.batch_steps} batches)"
    )


def configure_precision(train_config, architecture, precision=None, is_rl=False):
    if precision is None:
        precision = (
            "bf16"
            if is_rl or architecture == "dualstream-memory-risev33"
            else "fp32"
        )
    train_config.mixed_precision = precision
    print(f"Training precision: {precision}")


def configure_supervised_evaluation(train_config):
    train_config.batch_steps *= SUPERVISED_EVAL_INTERVAL_MULTIPLIER
    train_config.eval_batches = SUPERVISED_EVAL_BATCHES
    train_config.full_eval_each_epoch = True


def _checkpoint_progress(checkpoint) -> tuple[int, int]:
    required_keys = {"training_iteration", "evaluation_step"}
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(
            f"Checkpoint is missing required progress metadata: {missing}"
        )
    return (
        int(checkpoint["training_iteration"]),
        int(checkpoint["evaluation_step"]),
    )


def save_final_rl_checkpoint(
    model,
    optimizer,
    weights_dir: Path,
    training_iteration: int = None,
    evaluation_step: int = None,
    batch_steps: int = None,
) -> Path:
    """Save the final trainable RL state for the next self-play iteration."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = weights_dir / "model-rl-final.tar"
    save_torch_state(
        model,
        optimizer,
        checkpoint_path,
        training_iteration=training_iteration,
        evaluation_step=evaluation_step,
        batch_steps=batch_steps,
    )
    return checkpoint_path


def train_supervised(
    checkpoint_path: str = None,
    train_eval_shard: str = None,
    architecture: str = "risev33",
    batch_size: int = None,
    precision: str = None,
    train_data_dir: str = None,
    validation_shard: str = None,
):
    """Run supervised learning training on human game data."""
    tc = TrainConfig()
    tc.export_dir = f"{TRAINING_OUTPUT_DIR}/"
    tc.weights_dir = str(SUPERVISED_WEIGHTS_DIR)
    tc.use_wdl = True
    tc.use_plys_to_end = True
    tc.policy_loss_factor = 0.978
    configure_supervised_evaluation(tc)
    configure_batch_size(tc, architecture, batch_size)
    configure_precision(tc, architecture, precision)
    print(
        f"Intermediate evaluation: up to {tc.eval_batches} batches per loader "
        f"every {tc.batch_steps} training batches; full evaluation at epoch end"
    )
    checkpoint = None
    if checkpoint_path:
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    to = TrainObjects()
    to.metrics = get_metrics(tc)

    train_dir = (
        Path(train_data_dir).expanduser().resolve()
        if train_data_dir else Path(main_config['planes_train_dir'])
    )
    train_shards = sorted(str(path) for path in train_dir.glob('*.parquet'))
    tc.nb_parts = len(train_shards)
    if tc.nb_parts == 0:
        raise FileNotFoundError(
            f"No training parquet shards found in {train_dir}"
        )

    # Load validation data
    val_shard = _resolve_validation_shard(validation_shard)
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

    args = get_model_args(tc, architecture)
    model = create_model(args)

    checkpoint_state = None
    resume_iteration = 0
    if checkpoint is not None:
        checkpoint_state = torch.load(checkpoint, map_location="cpu")
        resume_iteration, tc.k_steps_initial = _checkpoint_progress(
            checkpoint_state
        )

    trainer = TrainerAgentPytorch(
        model,
        val_data,
        tc,
        to,
        use_rtpt=True,
        is_rl=False,
        train_eval_loader=train_eval_data,
    )
    trainer.ordering = train_shards
    if checkpoint_state is not None:
        restore_torch_state(
            model,
            trainer.optimizer,
            checkpoint_state,
        )
        print(
            f"Resuming supervised training from {checkpoint} "
            f"at iteration {resume_iteration}"
        )
    trainer.train(cur_it=resume_iteration)


def train_rl(rl_data_dir: str, val_data_dir: str, checkpoint_path: str = None,
             augment_flip: bool = True, architecture: str = "risev33",
             batch_size: int = None, precision: str = None):
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
    configure_batch_size(tc, architecture, batch_size)
    configure_precision(tc, architecture, precision, is_rl=True)
    to = TrainObjects()
    to.metrics = get_metrics(tc)
    
    # Set export directory and ensure it exists
    tc.export_dir = f"{TRAINING_OUTPUT_DIR}/"
    tc.weights_dir = str(RL_WEIGHTS_DIR)
    weights_dir = RL_WEIGHTS_DIR
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
        x, y_val, pol_a, pol_b, wdl, moves_left = load_rl_parquet_shard(vf)
        for i in range(len(x)):
            val_samples.append((x[i], y_val[i], pol_a[i], pol_b[i], wdl[i], moves_left[i]))
    
    if not val_samples:
        raise ValueError(f"No validation samples found in {val_data_dir}")
    
    # Convert validation to tensors
    x_val = torch.stack([s[0] for s in val_samples])
    y_val = torch.stack([s[1] for s in val_samples])
    pol_a_val = torch.stack([s[2] for s in val_samples])
    pol_b_val = torch.stack([s[3] for s in val_samples])
    wdl_val = torch.stack([s[4] for s in val_samples])
    moves_left_val = torch.stack([s[5] for s in val_samples])
    
    val_dataset = RLDataset(
        x_val,
        y_val,
        pol_a_val,
        pol_b_val,
        wdl_val,
        moves_left_val,
        augment_flip=augment_flip,
    )
    val_loader = DataLoader(val_dataset, batch_size=tc.batch_size, shuffle=False)
    print(f"Loaded {len(val_dataset)} validation samples")
    
    # Count rows from Parquet metadata so partial shards are handled exactly.
    training_samples = int(
        pl.scan_parquet(parquet_files).select(pl.len()).collect().item()
    )
    if augment_flip:
        training_samples *= 2
    n_train = training_samples
    
    print(f"Board flip augmentation: {'ENABLED' if augment_flip else 'DISABLED'}")
    if augment_flip:
        print(f"Training data will be doubled through board flip augmentation")
    print(f"Training samples: {n_train}")
    
    # Create streaming training dataset from all training files
    train_dataset = StreamingRLDataset(parquet_files, shuffle_files=True, shuffle_buffer_size=10000, augment_flip=augment_flip)
    train_loader = DataLoader(train_dataset, batch_size=tc.batch_size, num_workers=0)
    
    nb_it_per_epoch = max(1, (n_train + tc.batch_size - 1) // tc.batch_size)
    tc.total_it = int(nb_it_per_epoch * tc.nb_training_epochs)
    tc.nb_parts = len(parquet_files)
    
    print(f"Iterations per epoch: {nb_it_per_epoch}, Total iterations: {tc.total_it}")
    
    # LR schedule: Cosine Annealing with 25% warm-up (as per CrazyAra RL paper)
    # The warm-up helps with context drift in the training data
    cosine_schedule = CosineAnnealingSchedule(min_lr=tc.min_lr, max_lr=tc.max_lr, cycle_length=tc.total_it)
    to.lr_schedule = LinearWarmUp(cosine_schedule, start_lr=tc.min_lr, length=int(tc.total_it * 0.25))
    to.momentum_schedule = MomentumSchedule(to.lr_schedule, tc.min_lr, tc.max_lr, tc.min_momentum, tc.max_momentum)
    
    # Load model
    args = get_model_args(tc, architecture)
    model = create_model(args)
    
    # Optionally load checkpoint
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Train with streaming RL data loader
    trainer = TrainerAgentPytorch(model, val_loader, tc, to, use_rtpt=True, 
                                  is_rl=True, rl_train_loader=train_loader)
    trainer.train()
    
    # Save a trainable checkpoint before ONNX conversion mutates export state.
    final_checkpoint = save_final_rl_checkpoint(
        model,
        trainer.optimizer,
        weights_dir,
        training_iteration=trainer.cur_it,
        evaluation_step=trainer.k_steps,
        batch_steps=tc.batch_steps,
    )
    print(f"\nFinal RL checkpoint saved to {final_checkpoint}")

    print("Exporting final model to ONNX...")
    ctx = get_context(tc.context, tc.device_id)
    dummy_input = torch.zeros(1, NUM_BUGHOUSE_CHANNELS, 8, 8).to(ctx)
    model_prefix = f"model-rl-final"
    export_to_onnx(model, 1, dummy_input, weights_dir, model_prefix, True, True)
    print(f"Final RL ONNX exported under {weights_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Hivemind neural network')
    parser.add_argument('--mode', type=str, default='sl', choices=['sl', 'rl'],
                        help='Training mode: sl (supervised learning) or rl (reinforcement learning)')
    parser.add_argument(
        '--architecture',
        choices=sorted(MODEL_FACTORIES),
        default='risev33',
        help='Network architecture (default: risev33)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help=(
            'Training batch size (defaults: 256 for crossboard-risev33, '
            '256 for dualstream-memory-risev33, otherwise config value)'
        ),
    )
    parser.add_argument(
        '--precision',
        choices=('fp32', 'bf16'),
        default=None,
        help=(
            'Model execution precision (default: bf16 for RL and '
            'dualstream-memory-risev33, fp32 otherwise)'
        ),
    )
    parser.add_argument('--rl-data-dir', type=str, default=None,
                        help='Preconverted RL training Parquet directory (requires --val-data-dir)')
    parser.add_argument('--val-data-dir', type=str, default=None,
                        help='Preconverted RL validation Parquet directory (requires --rl-data-dir)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--selfplay-dir', type=str,
                        default=str(project_root / 'engine/selfplay_games'),
                        help='Self-play output directory, or its training_data directory, containing HVM3 chunks')
    parser.add_argument('--replay-dir', type=str, default=None,
                        help='Archived self-play directory used as CrazyAra-style replay memory')
    parser.add_argument('--replay-files', type=int, default=DEFAULT_REPLAY_FILES,
                        help='Number of archived HVM3 chunks to include (default: 5)')
    parser.add_argument(
        '--replay-selection-fraction',
        type=float,
        default=DEFAULT_REPLAY_SELECTION_FRACTION,
        help='Newest fraction of replay chunks eligible for selection (default: 0.05)',
    )
    parser.add_argument('--rl-output-dir', type=str, default=None,
                        help='Generated train/validation Parquet directory (default: <selfplay-dir>/rl_data)')
    parser.add_argument(
        '--validation-fraction',
        type=float,
        default=DEFAULT_RL_VALIDATION_FRACTION,
        help='Fraction of complete games reserved for validation (default: 0.02)',
    )
    parser.add_argument('--split-seed', type=int, default=42,
                        help='Deterministic game-level split seed (default: 42)')
    parser.add_argument('--train-eval-shard', type=str, default=None,
                        help='Fixed representative training shard used only for metrics')
    parser.add_argument('--sl-train-data-dir', type=str, default=None,
                        help='Supervised training Parquet directory')
    parser.add_argument('--sl-validation-shard', type=str, default=None,
                        help='Supervised validation Parquet shard')
    
    args = parser.parse_args()

    if args.mode == 'sl':
        train_supervised(
            checkpoint_path=args.checkpoint,
            train_eval_shard=args.train_eval_shard,
            architecture=args.architecture,
            batch_size=args.batch_size,
            precision=args.precision,
            train_data_dir=args.sl_train_data_dir,
            validation_shard=args.sl_validation_shard,
        )
    else:
        if args.rl_data_dir or args.val_data_dir:
            if not args.rl_data_dir or not args.val_data_dir:
                parser.error('--rl-data-dir and --val-data-dir must be provided together')
            train_rl(
                args.rl_data_dir,
                args.val_data_dir,
                checkpoint_path=args.checkpoint,
                architecture=args.architecture,
                batch_size=args.batch_size,
                precision=args.precision,
            )
        else:
            selfplay_path = Path(args.selfplay_dir).expanduser().resolve()
            has_training_data_dir = (selfplay_path / "training_data").is_dir()
            hvm_path = selfplay_path / "training_data" if has_training_data_dir else selfplay_path
            default_output_root = (
                selfplay_path
                if has_training_data_dir or selfplay_path.name != "training_data"
                else selfplay_path.parent
            )
            output_path = (
                Path(args.rl_output_dir).expanduser().resolve()
                if args.rl_output_dir
                else default_output_root / "rl_data"
            )
            replay_path = None
            if args.replay_dir:
                replay_root = Path(args.replay_dir).expanduser().resolve()
                replay_path = (
                    replay_root / "training_data"
                    if (replay_root / "training_data").is_dir()
                    else replay_root
                )
            train_path, val_path, _, _ = convert_to_split_parquet(
                hvm_path,
                output_path,
                validation_fraction=args.validation_fraction,
                split_seed=args.split_seed,
                replay_input_dir=replay_path,
                replay_files=args.replay_files,
                replay_selection_fraction=args.replay_selection_fraction,
            )
            train_rl(
                str(train_path),
                str(val_path),
                checkpoint_path=args.checkpoint,
                architecture=args.architecture,
                batch_size=args.batch_size,
                precision=args.precision,
            )
