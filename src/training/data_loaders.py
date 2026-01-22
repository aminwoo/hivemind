
"""
Data loading utilities for parquet files
"""
import polars as pl
import torch


def load_parquet_shard(file_path):
    """
    Loads a single parquet shard and converts it to PyTorch tensors.
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

    return x_tensor, y_val_tensor, y_pol_tensor