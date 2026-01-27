#!/usr/bin/env python3
"""
Convert binary training data from C++ selfplay to parquet format.

Binary format v2 (written by training_data_writer.cc):
  Header (16 bytes):
    - Magic bytes: "HVM2" (4 bytes)
    - Version: uint32 (4 bytes) = 2
    - Number of samples: uint64 (8 bytes)
  
  Per sample:
    - Planes: 4096 bytes (64 channels * 8 * 8)
    - Policy A num entries: uint16 (2 bytes)
    - Policy A entries: [uint16 index, float32 prob] * num_entries
    - Policy B num entries: uint16 (2 bytes)
    - Policy B entries: [uint16 index, float32 prob] * num_entries
    - Value: float32 (4 bytes)

Output parquet format (AlphaZero-style):
    - x: bytes (planes as uint8)
    - policy_a: bytes (sparse policy as dense float32 array)
    - policy_b: bytes (sparse policy as dense float32 array)
    - y_value: float (game outcome)
"""

import argparse
import os
import struct
import uuid
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

# Constants matching C++ code
NB_INPUT_CHANNELS = 64
BOARD_SIZE = 8
NB_INPUT_VALUES = NB_INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE  # 4096
NB_POLICY_CHANNELS = 73
NB_POLICY_VALUES = NB_POLICY_CHANNELS * BOARD_SIZE * BOARD_SIZE  # 4672


def read_sparse_policy(f) -> np.ndarray:
    """Read sparse policy entries and convert to dense array."""
    num_entries = struct.unpack('<H', f.read(2))[0]
    
    # Create dense policy array
    policy = np.zeros(NB_POLICY_VALUES, dtype=np.float32)
    
    for _ in range(num_entries):
        index = struct.unpack('<H', f.read(2))[0]
        prob = struct.unpack('<f', f.read(4))[0]
        if index < NB_POLICY_VALUES:
            policy[index] = prob
    
    return policy


def read_binary_shard(filepath: str) -> list[dict]:
    """Read a binary shard file and return list of sample dicts."""
    samples = []
    
    with open(filepath, 'rb') as f:
        # Read header
        magic = f.read(4)
        if magic == b'HVMD':
            # Legacy format v1 - not supported anymore
            raise ValueError(f"Legacy format v1 in {filepath} - please regenerate data")
        elif magic != b'HVM2':
            raise ValueError(f"Invalid magic bytes in {filepath}: {magic}")
        
        version = struct.unpack('<I', f.read(4))[0]
        if version != 2:
            raise ValueError(f"Unsupported version {version} in {filepath}")
        
        num_samples = struct.unpack('<Q', f.read(8))[0]
        
        # Read samples
        for _ in range(num_samples):
            # Read planes
            planes = np.frombuffer(f.read(NB_INPUT_VALUES), dtype=np.uint8)
            
            # Read policy A (sparse -> dense)
            policy_a = read_sparse_policy(f)
            
            # Read policy B (sparse -> dense)
            policy_b = read_sparse_policy(f)
            
            # Read value
            value = struct.unpack('<f', f.read(4))[0]
            
            samples.append({
                'x': planes.tobytes(),
                'policy_a': policy_a.tobytes(),
                'policy_b': policy_b.tobytes(),
                'y_value': float(value)
            })
    
    return samples


def convert_to_parquet(input_dir: str, output_dir: str, samples_per_shard: int = 65536):
    """Convert all binary shards in input_dir to parquet in output_dir.
    
    Uses streaming to avoid loading all data into memory at once.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all binary shard files
    bin_files = sorted(input_path.glob('shard_*.bin'))
    
    if not bin_files:
        print(f"No binary shard files found in {input_dir}")
        return
    
    print(f"Found {len(bin_files)} binary shard files")
    
    # Process files in streaming fashion to avoid OOM
    current_shard_samples = []
    total_samples = 0
    num_output_shards = 0
    
    def write_shard(samples):
        """Write a batch of samples to a parquet file."""
        nonlocal num_output_shards
        shard_id = uuid.uuid4().hex[:8]
        output_file = output_path / f"shard_{shard_id}.parquet"
        
        df = pl.DataFrame(samples)
        df.write_parquet(str(output_file), compression='zstd')
        num_output_shards += 1
        
        if num_output_shards % 10 == 0:
            print(f"  Written {num_output_shards} shards, {total_samples} total samples...")
    
    for bin_file in tqdm(bin_files, desc="Converting files"):
        try:
            samples = read_binary_shard(str(bin_file))
            
            for sample in samples:
                current_shard_samples.append(sample)
                total_samples += 1
                
                # Write shard when we have enough samples
                if len(current_shard_samples) >= samples_per_shard:
                    write_shard(current_shard_samples)
                    current_shard_samples = []  # Clear for next batch
                    
        except Exception as e:
            print(f"Error reading {bin_file}: {e}")
            continue
    
    # Write remaining samples
    if current_shard_samples:
        write_shard(current_shard_samples)
    
    print(f"\nConversion complete!")
    print(f"  Input: {len(bin_files)} binary shards")
    print(f"  Output: {num_output_shards} parquet shards")
    print(f"  Total samples: {total_samples}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert binary selfplay data to parquet format'
    )
    parser.add_argument(
        'input_dir',
        help='Directory containing binary shard files (shard_*.bin)'
    )
    parser.add_argument(
        'output_dir',
        help='Directory to write parquet files'
    )
    parser.add_argument(
        '--samples-per-shard',
        type=int,
        default=16384,
        help='Number of samples per output parquet shard (default: 16384)'
    )
    parser.add_argument(
        '--delete-binary',
        action='store_true',
        help='Delete binary files after successful conversion'
    )
    
    args = parser.parse_args()
    
    convert_to_parquet(args.input_dir, args.output_dir, args.samples_per_shard)
    
    if args.delete_binary:
        input_path = Path(args.input_dir)
        bin_files = list(input_path.glob('shard_*.bin'))
        for bin_file in bin_files:
            bin_file.unlink()
        print(f"Deleted {len(bin_files)} binary files")


if __name__ == '__main__':
    main()
