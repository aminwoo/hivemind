#!/usr/bin/env python3
"""Convert native HVM3 self-play chunks to Parquet training shards."""

import argparse
import hashlib
import re
import shutil
import struct
import uuid
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from src.constants import NUM_BUGHOUSE_CHANNELS, NUM_MOVE_CHANNELS

# Constants matching C++ code
NB_INPUT_CHANNELS = NUM_BUGHOUSE_CHANNELS
BOARD_SIZE = 8
NB_INPUT_VALUES = NB_INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE
NB_POLICY_VALUES = NUM_MOVE_CHANNELS * BOARD_SIZE * BOARD_SIZE
HEADER = struct.Struct('<4sIHHQ')
SAMPLE_METADATA = struct.Struct('<QIHHBBbB')
POLICY_ENTRY = struct.Struct('<Hf')
DEFAULT_RL_VALIDATION_FRACTION = 0.02


def read_exact(stream, size: int) -> bytes:
    payload = stream.read(size)
    if len(payload) != size:
        raise ValueError(
            f"Truncated HVM3 chunk: expected {size} bytes, found {len(payload)}"
        )
    return payload


def read_sparse_policy(f) -> np.ndarray:
    """Read sparse policy entries and convert to dense array."""
    num_entries = struct.unpack('<H', read_exact(f, 2))[0]
    
    # Create dense policy array
    policy = np.zeros(NB_POLICY_VALUES, dtype=np.float32)
    
    for _ in range(num_entries):
        index, probability = POLICY_ENTRY.unpack(
            read_exact(f, POLICY_ENTRY.size)
        )
        if index >= NB_POLICY_VALUES:
            raise ValueError(
                f"Policy index {index} exceeds {NB_POLICY_VALUES}"
            )
        policy[index] = probability

    if not np.isclose(policy.sum(), 1.0, atol=1e-4):
        raise ValueError(
            f"Sparse policy sums to {policy.sum()}, expected 1"
        )
    
    return policy


def read_binary_shard(filepath: str | Path) -> list[dict]:
    """Read one HVM3 chunk and return its samples."""
    samples = []
    
    with open(filepath, 'rb') as f:
        magic, version, channels, policy_values, num_samples = HEADER.unpack(
            read_exact(f, HEADER.size)
        )
        if magic != b'HVM3' or version != 3:
            raise ValueError(
                f"Unsupported self-play chunk {filepath}: {magic!r} v{version}"
            )
        if channels != NB_INPUT_CHANNELS:
            raise ValueError(
                f"Expected {NB_INPUT_CHANNELS} channels, found {channels}"
            )
        if policy_values != NB_POLICY_VALUES:
            raise ValueError(
                f"Expected {NB_POLICY_VALUES} policy values, found {policy_values}"
            )
        
        # Read samples
        for _ in range(num_samples):
            (
                game_id,
                nodes,
                macro_ply,
                moves_left,
                team,
                has_time_advantage,
                outcome,
                wdl,
            ) = SAMPLE_METADATA.unpack(read_exact(f, SAMPLE_METADATA.size))
            planes = read_exact(f, NB_INPUT_VALUES)
            policy_a = read_sparse_policy(f)
            policy_b = read_sparse_policy(f)

            samples.append({
                'x': planes,
                'policy_a': policy_a.tobytes(),
                'policy_b': policy_b.tobytes(),
                'y_value': float(outcome),
                'y_wdl': int(wdl),
                'y_moves_left': int(moves_left),
                'game_id': int(game_id),
                'macro_ply': int(macro_ply),
                'team': int(team),
                'has_time_advantage': bool(has_time_advantage),
                'search_nodes': int(nodes),
            })
        if f.read(1):
            raise ValueError(f"Trailing bytes in HVM3 chunk {filepath}")
    
    return samples


def _selfplay_run_id(path: Path) -> str:
    match = re.fullmatch(r"chunk_(.+)_\d+", path.stem)
    return match.group(1) if match else path.stem


def is_validation_game(
    run_id: str,
    game_id: int,
    validation_fraction: float,
    split_seed: int,
) -> bool:
    """Assign an entire self-play game to a deterministic split."""
    digest = hashlib.blake2b(
        f"{split_seed}:{run_id}:{game_id}".encode("ascii"),
        digest_size=8,
    ).digest()
    unit_value = int.from_bytes(digest, "little") / 2**64
    return unit_value < validation_fraction


def convert_to_split_parquet(
    input_dir: str | Path,
    output_dir: str | Path,
    validation_fraction: float = DEFAULT_RL_VALIDATION_FRACTION,
    split_seed: int = 42,
    samples_per_shard: int = 16384,
) -> tuple[Path, Path, int, int]:
    """Convert HVM3 chunks directly into game-disjoint train/validation shards."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if samples_per_shard <= 0:
        raise ValueError("samples_per_shard must be positive")

    input_path = Path(input_dir)
    binary_files = sorted(input_path.glob("*.hvm"))
    if not binary_files:
        raise FileNotFoundError(f"No HVM3 chunks found in {input_path}")

    output_path = Path(output_dir)
    resolved_input = input_path.resolve()
    resolved_output = output_path.resolve()
    if resolved_output == resolved_input or resolved_output in resolved_input.parents:
        raise ValueError("output_dir must not contain the HVM3 input directory")
    train_path = output_path / "train"
    validation_path = output_path / "val"
    shutil.rmtree(output_path, ignore_errors=True)
    train_path.mkdir(parents=True)
    validation_path.mkdir(parents=True)

    buffers = {"train": [], "val": []}
    counts = {"train": 0, "val": 0}
    shard_indices = {"train": 0, "val": 0}

    def flush(split: str) -> None:
        if not buffers[split]:
            return
        directory = validation_path if split == "val" else train_path
        destination = directory / f"shard_{shard_indices[split]:04d}.parquet"
        temporary = destination.with_suffix(".parquet.tmp")
        pl.DataFrame(buffers[split]).write_parquet(temporary, compression="zstd")
        temporary.replace(destination)
        shard_indices[split] += 1
        buffers[split].clear()

    for binary_file in tqdm(binary_files, desc="Converting HVM3 chunks"):
        run_id = _selfplay_run_id(binary_file)
        for sample in read_binary_shard(binary_file):
            split = "val" if is_validation_game(
                run_id,
                sample["game_id"],
                validation_fraction,
                split_seed,
            ) else "train"
            buffers[split].append(sample)
            counts[split] += 1
            if len(buffers[split]) >= samples_per_shard:
                flush(split)

    flush("train")
    flush("val")
    if counts["train"] == 0 or counts["val"] == 0:
        raise ValueError(
            "The game-level split produced an empty train or validation set; "
            "adjust validation_fraction or generate more games"
        )

    print(
        f"Prepared RL data: {counts['train']} train samples, "
        f"{counts['val']} validation samples"
    )
    return train_path, validation_path, counts["train"], counts["val"]


def convert_to_parquet(input_dir: str, output_dir: str, samples_per_shard: int = 65536):
    """Convert all binary shards in input_dir to parquet in output_dir.
    
    Uses streaming to avoid loading all data into memory at once.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    bin_files = sorted(input_path.glob('*.hvm'))
    
    if not bin_files:
        raise FileNotFoundError(f"No HVM3 chunks found in {input_dir}")
    
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
        
        temporary_file = output_file.with_suffix('.parquet.tmp')
        pl.DataFrame(samples).write_parquet(temporary_file, compression='zstd')
        temporary_file.replace(output_file)
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
    
    print("\nConversion complete!")
    print(f"  Input: {len(bin_files)} binary shards")
    print(f"  Output: {num_output_shards} parquet shards")
    print(f"  Total samples: {total_samples}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert binary selfplay data to parquet format'
    )
    parser.add_argument(
        'input_dir',
        help='Directory containing HVM3 chunks (*.hvm)'
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
        bin_files = list(input_path.glob('*.hvm'))
        for bin_file in bin_files:
            bin_file.unlink()
        print(f"Deleted {len(bin_files)} binary files")


if __name__ == '__main__':
    main()
