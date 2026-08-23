
"""
Data loading utilities for parquet files
"""
import numpy as np
import polars as pl
import torch
from src.constants import NUM_BUGHOUSE_CHANNELS, NUM_BUGHOUSE_CHANNELS_PER_BOARD


# Constants matching the Python representation
NB_INPUT_CHANNELS = NUM_BUGHOUSE_CHANNELS
NB_BUGHOUSE_BOARD_CHANNELS = NUM_BUGHOUSE_CHANNELS_PER_BOARD
BOARD_SIZE = 8
NB_INPUT_VALUES = NB_INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE
NB_POLICY_CHANNELS = 73
NB_POLICY_VALUES = NB_POLICY_CHANNELS * BOARD_SIZE * BOARD_SIZE  # 4672
NB_JOINT_POLICY_VALUES = NB_POLICY_VALUES + 1  # dedicated pass action
MAX_JOINT_POLICY_ENTRIES = 256
def _decode_plane_bytes(encoded_planes) -> np.ndarray:
    planes = np.frombuffer(encoded_planes, dtype=np.uint8)
    if planes.size != NB_INPUT_VALUES:
        raise ValueError(
            f"Expected {NB_INPUT_VALUES} input values, found {planes.size}"
        )
    return planes


def _decode_fixed_width_binary_column(values, dtype, width) -> np.ndarray:
    if not values:
        return np.empty((0, width), dtype=dtype)

    expected_nbytes = width * np.dtype(dtype).itemsize
    for idx, value in enumerate(values):
        if len(value) != expected_nbytes:
            raise ValueError(
                f"Invalid binary payload size at row {idx}: expected {expected_nbytes}, found {len(value)}"
            )

    joined = b"".join(values)
    return np.frombuffer(joined, dtype=dtype).reshape(len(values), width)


def _normalize_input_planes(x_tensor: torch.Tensor) -> torch.Tensor:
    return x_tensor / 255.0


def flip_bughouse_sample(x, policy_a, policy_b):
    """
    Apply bughouse board flip augmentation by swapping the two boards.
    
    In bughouse, due to symmetry, we can swap Board A and Board B to create
    an equivalent position from the partner's perspective. This effectively
    doubles the training data.
    
    IMPORTANT: When swapping boards, we must also mirror the moves in each policy
    because the board perspective changes (like viewing from the opposite side).
    
    Args:
        x: Input planes (C, 8, 8) where C depends on NUM_BUGHOUSE_CHANNELS
        policy_a: Policy distribution for Board A (4672,)
        policy_b: Policy distribution for Board B (4672,)
    
    Returns:
        Tuple of (flipped_x, flipped_policy_a, flipped_policy_b) where:
        - Board A and B channels are swapped
        - Policy distributions are swapped AND mirrored
    """
    # Clone to avoid modifying originals
    flipped_x = x.clone()
    
    # Swap the board channel blocks (Board A ↔ Board B)
    offset = NB_BUGHOUSE_BOARD_CHANNELS
    flipped_x[:offset] = x[offset:NB_INPUT_CHANNELS]
    flipped_x[offset:NB_INPUT_CHANNELS] = x[:offset]
    
    # Swap the policy distributions (NO MIRRORING - just swap)
    # What was Board B becomes Board A -> use policy_b as-is
    # What was Board A becomes Board B -> use policy_a as-is
    # The boards are not being flipped vertically, just swapped, so moves stay the same
    flipped_policy_a = policy_b.clone()
    flipped_policy_b = policy_a.clone()
    
    return flipped_x, flipped_policy_a, flipped_policy_b


def load_parquet_shard(file_path, include_auxiliary=False):
    """
    Loads a single supervised learning parquet shard and converts it to PyTorch tensors.
    x: uniformly quantized board planes (74, 8, 8)
    y_value: game outcome
    y_policy_idx: tuple containing (move_index, ...)
    """
    required_columns = ['x', 'y_value', 'y_policy_idx']
    if include_auxiliary:
        required_columns.append('y_plys_to_end')

    df = pl.read_parquet(file_path, columns=required_columns)

    x_array = _decode_fixed_width_binary_column(
        df['x'].to_list(),
        dtype=np.uint8,
        width=NB_INPUT_VALUES,
    )
    x_tensor = torch.from_numpy(x_array.copy()).float().view(
        -1, NB_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE
    )
    x_tensor = _normalize_input_planes(x_tensor)

    y_val_tensor = torch.tensor(df['y_value'].to_list(), dtype=torch.float32)
    y_pol_tensor = torch.tensor(df['y_policy_idx'].to_list(), dtype=torch.long)

    if include_auxiliary:
        if 'y_plys_to_end' not in df.columns:
            raise ValueError(
                "Moves-left training requires a y_plys_to_end column; regenerate this shard"
            )
        if not torch.all((y_val_tensor == -1) | (y_val_tensor == 0) | (y_val_tensor == 1)):
            raise ValueError("WDL training requires y_value labels in {-1, 0, 1}")
        wdl_tensor = (y_val_tensor + 1).long()
        plys_tensor = torch.tensor(
            df['y_plys_to_end'].to_list(), dtype=torch.float32
        ).clamp_(0, 100) / 100.0
        return x_tensor, y_val_tensor, y_pol_tensor, wdl_tensor, plys_tensor

    return x_tensor, y_val_tensor, y_pol_tensor


def load_rl_parquet_shard(file_path, include_joint_policy=False):
    """
    Loads a single RL/self-play parquet shard and converts it to PyTorch tensors.
    
    RL data format (from C++ self-play):
    - x: bytes (NB_INPUT_VALUES uint8 planes)
    - policy_a: bytes (4672 float32 dense policy distribution for board A)
    - policy_b: bytes (4672 float32 dense policy distribution for board B)
    - y_value: float (game outcome)
    - y_wdl: int in {0, 1, 2}
    - y_moves_left: remaining team decisions
    
    Returns:
        x_tensor: (N, 74, 8, 8) float32 input planes
        y_val_tensor: (N,) float32 value targets
        policy_a_tensor: (N, 4672) float32 policy distribution for board A
        policy_b_tensor: (N, 4672) float32 policy distribution for board B
    """
    columns = ['x', 'policy_a', 'policy_b', 'y_value', 'y_wdl', 'y_moves_left']
    joint_columns = [
        'joint_policy_a',
        'joint_policy_b',
        'joint_policy_probability',
        'joint_policy_count',
    ]
    available_columns = set(pl.scan_parquet(file_path).collect_schema().names())
    has_joint_policy = all(column in available_columns for column in joint_columns)
    if include_joint_policy and has_joint_policy:
        columns.extend(joint_columns)
    df = pl.read_parquet(file_path, columns=columns)

    x_array = _decode_fixed_width_binary_column(
        df['x'].to_list(),
        dtype=np.uint8,
        width=NB_INPUT_VALUES,
    )
    x_tensor = torch.from_numpy(x_array.copy()).float().view(
        -1, NB_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE
    )
    x_tensor = _normalize_input_planes(x_tensor)
    
    # Process Y_Value
    y_val_tensor = torch.tensor(df['y_value'].to_list(), dtype=torch.float32)
    
    policy_a_array = _decode_fixed_width_binary_column(
        df['policy_a'].to_list(),
        dtype=np.float32,
        width=NB_POLICY_VALUES,
    )
    policy_b_array = _decode_fixed_width_binary_column(
        df['policy_b'].to_list(),
        dtype=np.float32,
        width=NB_POLICY_VALUES,
    )
    policy_a_tensor = torch.tensor(policy_a_array.copy(), dtype=torch.float32)
    policy_b_tensor = torch.tensor(policy_b_array.copy(), dtype=torch.float32)
    wdl_tensor = torch.tensor(df['y_wdl'].to_list(), dtype=torch.long)
    moves_left_tensor = torch.tensor(
        df['y_moves_left'].to_list(), dtype=torch.float32
    ).clamp_(0, 100) / 100.0

    base = (
        x_tensor,
        y_val_tensor,
        policy_a_tensor,
        policy_b_tensor,
        wdl_tensor,
        moves_left_tensor,
    )
    if not include_joint_policy:
        return base

    sample_count = len(x_tensor)
    if has_joint_policy:
        joint_a_array = _decode_fixed_width_binary_column(
            df['joint_policy_a'].to_list(),
            dtype=np.uint16,
            width=MAX_JOINT_POLICY_ENTRIES,
        )
        joint_b_array = _decode_fixed_width_binary_column(
            df['joint_policy_b'].to_list(),
            dtype=np.uint16,
            width=MAX_JOINT_POLICY_ENTRIES,
        )
        joint_probability_array = _decode_fixed_width_binary_column(
            df['joint_policy_probability'].to_list(),
            dtype=np.float32,
            width=MAX_JOINT_POLICY_ENTRIES,
        )
        joint_count = torch.tensor(
            df['joint_policy_count'].to_list(), dtype=torch.long
        )
    else:
        joint_a_array = np.zeros(
            (sample_count, MAX_JOINT_POLICY_ENTRIES), dtype=np.uint16
        )
        joint_b_array = np.zeros_like(joint_a_array)
        joint_probability_array = np.zeros(
            (sample_count, MAX_JOINT_POLICY_ENTRIES), dtype=np.float32
        )
        joint_count = torch.zeros(sample_count, dtype=torch.long)

    return base + (
        torch.from_numpy(joint_a_array.astype(np.int64)),
        torch.from_numpy(joint_b_array.astype(np.int64)),
        torch.from_numpy(joint_probability_array.copy()),
        joint_count,
    )


class RLDataset(torch.utils.data.Dataset):
    """
    Dataset for RL/self-play data with dual policy targets.
    """
    def __init__(self, x, y_value, policy_a, policy_b, wdl, moves_left, augment_flip=False):
        """
        Args:
            x: Input planes (N, 74, 8, 8)
            y_value: Value targets (N,)
            policy_a: Policy distribution for Board A (N, 4672)
            policy_b: Policy distribution for Board B (N, 4672)
            augment_flip: If True, doubles dataset size by including flipped samples
        """
        self.x = x
        self.y_value = y_value
        self.policy_a = policy_a
        self.policy_b = policy_b
        self.wdl = wdl
        self.moves_left = moves_left
        self.augment_flip = augment_flip
        
    def __len__(self):
        # If augmentation is enabled, double the dataset size
        return len(self.x) * (2 if self.augment_flip else 1)
    
    def __getitem__(self, idx):
        # Determine if this is an original or flipped sample
        if self.augment_flip and idx >= len(self.x):
            # This is a flipped sample
            original_idx = idx - len(self.x)
            x, policy_a, policy_b = flip_bughouse_sample(
                self.x[original_idx], 
                self.policy_a[original_idx], 
                self.policy_b[original_idx]
            )
            return (
                x,
                self.y_value[original_idx],
                policy_a,
                policy_b,
                self.wdl[original_idx],
                self.moves_left[original_idx],
            )
        else:
            return (
                self.x[idx],
                self.y_value[idx],
                self.policy_a[idx],
                self.policy_b[idx],
                self.wdl[idx],
                self.moves_left[idx],
            )


class StreamingRLDataset(torch.utils.data.IterableDataset):
    """
    Streaming dataset for RL/self-play data that loads parquet files on-demand.
    
    This avoids loading all data into memory at once by streaming through
    parquet files and yielding samples one at a time.
    """
    def __init__(
        self,
        parquet_files: list,
        shuffle_files: bool = True,
        shuffle_buffer_size: int = 10000,
        augment_flip: bool = False,
        shuffle_samples: bool = True,
        include_joint_policy: bool = False,
    ):
        """
        Args:
            parquet_files: List of paths to parquet files
            shuffle_files: Whether to shuffle file order each epoch
            shuffle_buffer_size: Size of buffer for sample shuffling within files
            augment_flip: If True, yields both original and flipped versions of each sample
            shuffle_samples: If False, yield samples directly without retaining a
                shuffle buffer. This is useful for validation, where order does
                not matter and dense policy targets consume substantial RAM.
        """
        self.parquet_files = parquet_files
        self.shuffle_files = shuffle_files
        self.shuffle_buffer_size = shuffle_buffer_size
        self.augment_flip = augment_flip
        self.shuffle_samples = shuffle_samples
        self.include_joint_policy = include_joint_policy
        
    def __iter__(self):
        files = self.parquet_files.copy()
        if self.shuffle_files:
            import random
            random.shuffle(files)
        
        buffer = []
        
        for pf in files:
            shard = (
                load_rl_parquet_shard(pf, include_joint_policy=True)
                if self.include_joint_policy
                else load_rl_parquet_shard(pf)
            )
            x, y_val, pol_a, pol_b, wdl, moves_left = shard[:6]
            joint = shard[6:]
            
            # Add samples to buffer (both original and flipped if augmentation is enabled)
            for i in range(len(x)):
                sample = (
                    x[i], y_val[i], pol_a[i], pol_b[i], wdl[i], moves_left[i]
                ) + tuple(values[i] for values in joint)
                if self.shuffle_samples:
                    buffer.append(sample)
                else:
                    yield sample
                
                # Add flipped sample if augmentation is enabled
                if self.augment_flip:
                    flipped_x, flipped_pol_a, flipped_pol_b = flip_bughouse_sample(
                        x[i], pol_a[i], pol_b[i]
                    )
                    flipped_sample = (
                        flipped_x,
                        y_val[i],
                        flipped_pol_a,
                        flipped_pol_b,
                        wdl[i],
                        moves_left[i],
                    )
                    if joint:
                        flipped_sample += (
                            joint[1][i],
                            joint[0][i],
                            joint[2][i],
                            joint[3][i],
                        )
                    if self.shuffle_samples:
                        buffer.append(flipped_sample)
                    else:
                        yield flipped_sample
                
                # When buffer is full, shuffle and yield half
                if (
                    self.shuffle_samples
                    and len(buffer) >= self.shuffle_buffer_size
                ):
                    import random
                    random.shuffle(buffer)
                    # Yield first half
                    for sample in buffer[:len(buffer)//2]:
                        yield sample
                    buffer = buffer[len(buffer)//2:]
        
        # Yield remaining samples in buffer
        if buffer:
            import random
            random.shuffle(buffer)
            for sample in buffer:
                yield sample


class CombinedRLDataset(torch.utils.data.IterableDataset):
    """
    Combines a regular RLDataset with a StreamingRLDataset.
    
    Yields all samples from the regular dataset first (shuffled),
    then streams from the streaming dataset.
    """
    def __init__(self, regular_dataset: RLDataset, streaming_dataset: StreamingRLDataset):
        self.regular_dataset = regular_dataset
        self.streaming_dataset = streaming_dataset
        
    def __iter__(self):
        import random
        
        # First yield from regular dataset (shuffled)
        indices = list(range(len(self.regular_dataset)))
        random.shuffle(indices)
        
        for idx in indices:
            yield self.regular_dataset[idx]
        
        # Then stream from streaming dataset
        for sample in self.streaming_dataset:
            yield sample


def load_rl_data_from_directory(data_dir, max_samples=None):
    """
    Load all RL parquet shards from a directory.
    
    Args:
        data_dir: Directory containing RL parquet files
        max_samples: Optional limit on total samples to load
        
    Returns:
        Tuple of (x, y_value, policy_a, policy_b, wdl, moves_left) tensors
    """
    import glob
    from pathlib import Path
    
    parquet_files = sorted(glob.glob(str(Path(data_dir) / "*.parquet")))
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")
    
    all_x = []
    all_y_value = []
    all_policy_a = []
    all_policy_b = []
    all_wdl = []
    all_moves_left = []
    
    total_loaded = 0
    for pf in parquet_files:
        x, y_val, pol_a, pol_b, wdl, moves_left = load_rl_parquet_shard(pf)
        all_x.append(x)
        all_y_value.append(y_val)
        all_policy_a.append(pol_a)
        all_policy_b.append(pol_b)
        all_wdl.append(wdl)
        all_moves_left.append(moves_left)
        
        total_loaded += len(x)
        if max_samples and total_loaded >= max_samples:
            break
    
    x_tensor = torch.cat(all_x, dim=0)
    y_val_tensor = torch.cat(all_y_value, dim=0)
    policy_a_tensor = torch.cat(all_policy_a, dim=0)
    policy_b_tensor = torch.cat(all_policy_b, dim=0)
    wdl_tensor = torch.cat(all_wdl, dim=0)
    moves_left_tensor = torch.cat(all_moves_left, dim=0)
    
    if max_samples:
        x_tensor = x_tensor[:max_samples]
        y_val_tensor = y_val_tensor[:max_samples]
        policy_a_tensor = policy_a_tensor[:max_samples]
        policy_b_tensor = policy_b_tensor[:max_samples]
        wdl_tensor = wdl_tensor[:max_samples]
        moves_left_tensor = moves_left_tensor[:max_samples]
    
    return (
        x_tensor,
        y_val_tensor,
        policy_a_tensor,
        policy_b_tensor,
        wdl_tensor,
        moves_left_tensor,
    )
