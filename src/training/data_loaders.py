
"""
Data loading utilities for parquet files
"""
<<<<<<< HEAD
=======
import numpy as np
>>>>>>> feat/multi-pv
import polars as pl
import torch


<<<<<<< HEAD
def load_parquet_shard(file_path):
    """
    Loads a single parquet shard and converts it to PyTorch tensors.
=======
# Constants matching C++ code
NB_INPUT_CHANNELS = 64
BOARD_SIZE = 8
NB_INPUT_VALUES = NB_INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE  # 4096
NB_POLICY_CHANNELS = 73
NB_POLICY_VALUES = NB_POLICY_CHANNELS * BOARD_SIZE * BOARD_SIZE  # 4672


def flip_bughouse_sample(x, policy_a, policy_b):
    """
    Apply bughouse board flip augmentation by swapping the two boards.
    
    In bughouse, due to symmetry, we can swap Board A and Board B to create
    an equivalent position from the partner's perspective. This effectively
    doubles the training data.
    
    IMPORTANT: When swapping boards, we must also mirror the moves in each policy
    because the board perspective changes (like viewing from the opposite side).
    
    Args:
        x: Input planes (64, 8, 8) where channels 0-31 are Board A, 32-63 are Board B
        policy_a: Policy distribution for Board A (4672,)
        policy_b: Policy distribution for Board B (4672,)
    
    Returns:
        Tuple of (flipped_x, flipped_policy_a, flipped_policy_b) where:
        - Board A and B channels are swapped
        - Policy distributions are swapped AND mirrored
    """
    # Clone to avoid modifying originals
    flipped_x = x.clone()
    
    # Swap the 32-channel blocks (Board A ↔ Board B)
    flipped_x[:32] = x[32:64]  # Board A gets Board B's channels
    flipped_x[32:64] = x[:32]  # Board B gets Board A's channels
    
    # Swap the policy distributions (NO MIRRORING - just swap)
    # What was Board B becomes Board A -> use policy_b as-is
    # What was Board A becomes Board B -> use policy_a as-is
    # The boards are not being flipped vertically, just swapped, so moves stay the same
    flipped_policy_a = policy_b.clone()
    flipped_policy_b = policy_a.clone()
    
    return flipped_x, flipped_policy_a, flipped_policy_b


def load_parquet_shard(file_path):
    """
    Loads a single supervised learning parquet shard and converts it to PyTorch tensors.
>>>>>>> feat/multi-pv
    x: board planes (64, 8, 8)
    y_value: game outcome
    y_policy_idx: tuple containing (move_index, ...)
    """
    # Read the parquet file
    df = pl.read_parquet(file_path)

    # 1. Process X (Planes): Convert list of floats to (Batch, 64, 8, 8)
    # If x is stored as a flat list of 4096 values, reshape it.
    x_tensor = torch.tensor(df['x'].to_list(), dtype=torch.float32).view(-1, 64, 8, 8)

    # 2. Process Y_Value
    y_val_tensor = torch.tensor(df['y_value'].to_list(), dtype=torch.float32)

    # 3. Process Y_policy
    y_pol_tensor = torch.tensor(df['y_policy_idx'].to_list(), dtype=torch.long)

<<<<<<< HEAD
    return x_tensor, y_val_tensor, y_pol_tensor
=======
    return x_tensor, y_val_tensor, y_pol_tensor


def load_rl_parquet_shard(file_path):
    """
    Loads a single RL/self-play parquet shard and converts it to PyTorch tensors.
    
    RL data format (from C++ self-play):
    - x: bytes (4096 uint8 planes)
    - policy_a: bytes (4672 float32 dense policy distribution for board A)
    - policy_b: bytes (4672 float32 dense policy distribution for board B)
    - y_value: float (game outcome)
    
    Returns:
        x_tensor: (N, 64, 8, 8) float32 input planes
        y_val_tensor: (N,) float32 value targets
        policy_a_tensor: (N, 4672) float32 policy distribution for board A
        policy_b_tensor: (N, 4672) float32 policy distribution for board B
    """
    df = pl.read_parquet(file_path)
    
    # Process X (Planes): Convert bytes to (Batch, 64, 8, 8)
    x_list = []
    for x_bytes in df['x']:
        planes = np.frombuffer(x_bytes, dtype=np.uint8).astype(np.float32)
        x_list.append(planes)
    x_tensor = torch.tensor(np.stack(x_list), dtype=torch.float32).view(-1, 64, 8, 8)
    
    # Normalize pocket planes (channels 12-21 and 44-53): stored as 0-16, convert to 0.0-1.0
    x_tensor[:, 12:22, :, :] /= 16.0
    x_tensor[:, 44:54, :, :] /= 16.0
    
    # Process Y_Value
    y_val_tensor = torch.tensor(df['y_value'].to_list(), dtype=torch.float32)
    
    # Process Policy A (dense float32 distribution)
    policy_a_list = []
    for p_bytes in df['policy_a']:
        policy = np.frombuffer(p_bytes, dtype=np.float32)
        policy_a_list.append(policy)
    policy_a_tensor = torch.tensor(np.stack(policy_a_list), dtype=torch.float32)
    
    # Process Policy B (dense float32 distribution)
    policy_b_list = []
    for p_bytes in df['policy_b']:
        policy = np.frombuffer(p_bytes, dtype=np.float32)
        policy_b_list.append(policy)
    policy_b_tensor = torch.tensor(np.stack(policy_b_list), dtype=torch.float32)
    
    return x_tensor, y_val_tensor, policy_a_tensor, policy_b_tensor


class RLDataset(torch.utils.data.Dataset):
    """
    Dataset for RL/self-play data with dual policy targets.
    """
    def __init__(self, x, y_value, policy_a, policy_b, augment_flip=False):
        """
        Args:
            x: Input planes (N, 64, 8, 8)
            y_value: Value targets (N,)
            policy_a: Policy distribution for Board A (N, 4672)
            policy_b: Policy distribution for Board B (N, 4672)
            augment_flip: If True, doubles dataset size by including flipped samples
        """
        self.x = x
        self.y_value = y_value
        self.policy_a = policy_a
        self.policy_b = policy_b
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
            return x, self.y_value[original_idx], policy_a, policy_b
        else:
            # This is an original sample
            return self.x[idx], self.y_value[idx], self.policy_a[idx], self.policy_b[idx]


class StreamingRLDataset(torch.utils.data.IterableDataset):
    """
    Streaming dataset for RL/self-play data that loads parquet files on-demand.
    
    This avoids loading all data into memory at once by streaming through
    parquet files and yielding samples one at a time.
    """
    def __init__(self, parquet_files: list, shuffle_files: bool = True, shuffle_buffer_size: int = 10000, augment_flip: bool = False):
        """
        Args:
            parquet_files: List of paths to parquet files
            shuffle_files: Whether to shuffle file order each epoch
            shuffle_buffer_size: Size of buffer for sample shuffling within files
            augment_flip: If True, yields both original and flipped versions of each sample
        """
        self.parquet_files = parquet_files
        self.shuffle_files = shuffle_files
        self.shuffle_buffer_size = shuffle_buffer_size
        self.augment_flip = augment_flip
        
    def __iter__(self):
        files = self.parquet_files.copy()
        if self.shuffle_files:
            import random
            random.shuffle(files)
        
        buffer = []
        
        for pf in files:
            x, y_val, pol_a, pol_b = load_rl_parquet_shard(pf)
            
            # Add samples to buffer (both original and flipped if augmentation is enabled)
            for i in range(len(x)):
                # Add original sample
                buffer.append((x[i], y_val[i], pol_a[i], pol_b[i]))
                
                # Add flipped sample if augmentation is enabled
                if self.augment_flip:
                    flipped_x, flipped_pol_a, flipped_pol_b = flip_bughouse_sample(
                        x[i], pol_a[i], pol_b[i]
                    )
                    buffer.append((flipped_x, y_val[i], flipped_pol_a, flipped_pol_b))
                
                # When buffer is full, shuffle and yield half
                if len(buffer) >= self.shuffle_buffer_size:
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
        Tuple of (x, y_value, policy_a, policy_b) tensors
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
    
    total_loaded = 0
    for pf in parquet_files:
        x, y_val, pol_a, pol_b = load_rl_parquet_shard(pf)
        all_x.append(x)
        all_y_value.append(y_val)
        all_policy_a.append(pol_a)
        all_policy_b.append(pol_b)
        
        total_loaded += len(x)
        if max_samples and total_loaded >= max_samples:
            break
    
    x_tensor = torch.cat(all_x, dim=0)
    y_val_tensor = torch.cat(all_y_value, dim=0)
    policy_a_tensor = torch.cat(all_policy_a, dim=0)
    policy_b_tensor = torch.cat(all_policy_b, dim=0)
    
    if max_samples:
        x_tensor = x_tensor[:max_samples]
        y_val_tensor = y_val_tensor[:max_samples]
        policy_a_tensor = policy_a_tensor[:max_samples]
        policy_b_tensor = policy_b_tensor[:max_samples]
    
    return x_tensor, y_val_tensor, policy_a_tensor, policy_b_tensor
>>>>>>> feat/multi-pv
