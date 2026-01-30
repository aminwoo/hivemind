#!/usr/bin/env python3
"""
Search for a specific FEN position in training data and display MCTS visits.

Supports both binary (.bin) and parquet (.parquet) training data formats.

Usage:
    python search_training_fen.py --fen "FEN_A|FEN_B"
    
Example:
    python search_training_fen.py --fen "r3k2r/ppp2ppp/2n1p3/3qP3/2B2B2/4Pb2/PPP1Q2P/4K2R[BBNNqbnnPP] w Kkq|N4k1r/pp2Nppp/3pp3/7b/2b5/2pnPP2/PP1Q2PP/R2K3R[RrpPP] w"
"""

import argparse
import os
import glob
import struct
import sys
import numpy as np
import chess

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

from src.domain.board import BughouseBoard
from src.domain.board2planes import board2planes
from src.domain.move2planes import make_map

# Constants matching training data format
NB_INPUT_CHANNELS = 64
NB_INPUT_VALUES = NB_INPUT_CHANNELS * 8 * 8  # 4096
NB_POLICY_VALUES = 73 * 8 * 8  # 4672


def fen_to_planes(fen: str, team_side: chess.Color = chess.WHITE) -> np.ndarray:
    """
    Convert a bughouse FEN string to planes.
    
    Args:
        fen: Bughouse FEN with two boards separated by |
        team_side: Perspective for the planes (WHITE or BLACK)
    
    Returns:
        np.ndarray of shape (64, 8, 8)
    """
    board = BughouseBoard()
    # Handle both | and " | " separators
    fen = fen.replace(" | ", "|")
    parts = fen.split("|")
    if len(parts) != 2:
        raise ValueError(f"Expected FEN with two boards separated by |, got: {fen}")
    
    fen_a = parts[0].strip()
    fen_b = parts[1].strip()
    board.set_fen(f"{fen_a} | {fen_b}")
    
    planes = board2planes(board, team_side, flip=False)
    return planes


def decode_move_index(move_idx: int, move_map: list) -> str:
    """Convert a move index to a human-readable move string."""
    if 0 <= move_idx < len(move_map):
        return move_map[move_idx]
    return f"unknown({move_idx})"


def compare_planes(planes1: np.ndarray, planes2: np.ndarray, tolerance: float = 0.01) -> bool:
    """Compare two plane arrays for approximate equality."""
    return np.allclose(planes1, planes2, atol=tolerance)


def read_sparse_policy(f) -> np.ndarray:
    """Read sparse policy entries and convert to dense array (from binary format)."""
    num_entries = struct.unpack('<H', f.read(2))[0]
    policy = np.zeros(NB_POLICY_VALUES, dtype=np.float32)
    for _ in range(num_entries):
        index = struct.unpack('<H', f.read(2))[0]
        prob = struct.unpack('<f', f.read(4))[0]
        if index < NB_POLICY_VALUES:
            policy[index] = prob
    return policy


def search_in_binary_file(file_path: str, target_planes: np.ndarray, move_map: list, verbose: bool = False) -> list:
    """
    Search for matching planes in a binary shard file.
    
    Returns list of (sample_idx, sample_dict) tuples for matches.
    """
    matches = []
    
    try:
        with open(file_path, 'rb') as f:
            # Read header
            magic = f.read(4)
            if magic != b'HVM2':
                if verbose:
                    print(f"  Skipping {os.path.basename(file_path)} - invalid magic bytes")
                return matches
            
            version = struct.unpack('<I', f.read(4))[0]
            if version != 2:
                if verbose:
                    print(f"  Skipping {os.path.basename(file_path)} - unsupported version {version}")
                return matches
            
            num_samples = struct.unpack('<Q', f.read(8))[0]
            
            if verbose:
                print(f"  Searching {num_samples} samples in {os.path.basename(file_path)}...", end="", flush=True)
            
            for i in range(num_samples):
                # Read planes
                planes_bytes = f.read(NB_INPUT_VALUES)
                sample_planes = np.frombuffer(planes_bytes, dtype=np.uint8).reshape(NB_INPUT_CHANNELS, 8, 8).astype(float)
                
                # Read policies
                policy_a = read_sparse_policy(f)
                policy_b = read_sparse_policy(f)
                
                # Read value
                value = struct.unpack('<f', f.read(4))[0]
                
                # Compare piece position channels (0-11 and 32-43)
                # The target uses float 0.0-1.0, sample uses uint8 0-1 for binary channels
                piece_channels = list(range(12)) + list(range(32, 44))
                
                target_pieces = target_planes[piece_channels]
                sample_pieces = sample_planes[piece_channels]  # Already 0 or 1 for piece channels
                
                if np.allclose(target_pieces, sample_pieces, atol=0.5):
                    sample = {
                        'x': planes_bytes,
                        'policy_a': policy_a,
                        'policy_b': policy_b,
                        'y_value': value
                    }
                    matches.append((i, sample))
            
            if verbose:
                print(f" found {len(matches)} matches")
    
    except Exception as e:
        if verbose:
            print(f"  Error reading {file_path}: {e}")
    
    return matches


def search_in_parquet_file(file_path: str, target_planes: np.ndarray, move_map: list, verbose: bool = False) -> list:
    """
    Search for matching planes in a parquet file.
    
    Returns list of (sample_idx, sample_dict) tuples for matches.
    """
    import polars as pl
    
    matches = []
    df = pl.read_parquet(file_path)
    total = len(df)
    
    if verbose:
        print(f"  Searching {total} samples in {os.path.basename(file_path)}...", end="", flush=True)
    
    for i in range(total):
        row = df.row(i, named=True)
        x_bytes = row['x']
        sample_planes = np.frombuffer(x_bytes, dtype=np.uint8).reshape(NB_INPUT_CHANNELS, 8, 8).astype(float)
        
        # Compare piece position channels (0-11 and 32-43)
        piece_channels = list(range(12)) + list(range(32, 44))
        target_pieces = target_planes[piece_channels]
        sample_pieces = sample_planes[piece_channels]
        
        if np.allclose(target_pieces, sample_pieces, atol=0.5):
            sample = {
                'x': x_bytes,
                'policy_a': np.frombuffer(row['policy_a'], dtype=np.float32),
                'policy_b': np.frombuffer(row['policy_b'], dtype=np.float32),
                'y_value': row['y_value']
            }
            matches.append((i, sample))
    
    if verbose:
        print(f" found {len(matches)} matches")
    
    return matches


def display_match(sample_idx: int, sample: dict, move_map: list):
    """Display MCTS visit distribution for a matched sample."""
    print(f"\n{'='*80}")
    print(f"MATCH FOUND - Sample #{sample_idx}")
    print(f"{'='*80}")
    
    # Read policies (already numpy arrays from our search functions)
    policy_a = sample['policy_a']
    policy_b = sample['policy_b']
    
    # Read outcome
    outcome = sample['y_value']
    
    print(f"\nOutcome: {outcome:.3f} (1.0=team 0 wins, -1.0=team 1 wins)")
    
    # Print policy sums for verification
    policy_a_sum = np.sum(policy_a)
    policy_b_sum = np.sum(policy_b)
    print(f"Policy sums: Board A = {policy_a_sum:.6f}, Board B = {policy_b_sum:.6f}")
    
    # Print move distribution for Board A
    print(f"\n--- Board A MCTS Visit Distribution ---")
    top_moves_a = np.argsort(policy_a)[-15:][::-1]
    for move_idx in top_moves_a:
        prob = policy_a[move_idx]
        if prob > 0.001:
            move_str = decode_move_index(move_idx, move_map)
            print(f"  {move_str:12s} (idx {move_idx:4d}): {prob:.4f} ({prob*100:.2f}%)")
    
    # Print move distribution for Board B
    print(f"\n--- Board B MCTS Visit Distribution ---")
    top_moves_b = np.argsort(policy_b)[-15:][::-1]
    for move_idx in top_moves_b:
        prob = policy_b[move_idx]
        if prob > 0.001:
            move_str = decode_move_index(move_idx, move_map)
            print(f"  {move_str:12s} (idx {move_idx:4d}): {prob:.4f} ({prob*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Search for a FEN position in training data'
    )
    parser.add_argument(
        '--fen',
        type=str,
        required=True,
        help='Bughouse FEN with two boards separated by |'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/home/ben/hivemind/engine/selfplay_games/training_data',
        help='Directory containing training data files (.bin or .parquet)'
    )
    parser.add_argument(
        '--team-side',
        type=str,
        default='white',
        choices=['white', 'black'],
        help='Team perspective for plane conversion'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Maximum number of files to search'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show progress information'
    )
    
    args = parser.parse_args()
    
    team_side = chess.WHITE if args.team_side == 'white' else chess.BLACK
    
    # Convert FEN to planes
    print(f"Converting FEN to planes...")
    print(f"  FEN: {args.fen}")
    print(f"  Team side: {'WHITE' if team_side == chess.WHITE else 'BLACK'}")
    
    try:
        target_planes = fen_to_planes(args.fen, team_side)
    except Exception as e:
        print(f"Error parsing FEN: {e}")
        return 1
    
    print(f"  Planes shape: {target_planes.shape}")
    
    # Load move map
    move_map = make_map()
    
    # Find training data files (prefer binary, fallback to parquet)
    bin_files = sorted(glob.glob(os.path.join(args.data_dir, 'shard_*.bin')))
    parquet_files = sorted(glob.glob(os.path.join(args.data_dir, '*.parquet')))
    
    if bin_files:
        data_files = bin_files
        file_type = 'binary'
    elif parquet_files:
        data_files = parquet_files
        file_type = 'parquet'
    else:
        print(f"No training data files found in {args.data_dir}")
        print(f"  Looked for: shard_*.bin or *.parquet")
        return 1
    
    if args.max_files:
        data_files = data_files[:args.max_files]
    
    print(f"\nSearching {len(data_files)} {file_type} files...")
    
    all_matches = []
    for file_path in data_files:
        if file_type == 'binary':
            matches = search_in_binary_file(file_path, target_planes, move_map, args.verbose)
        else:
            matches = search_in_parquet_file(file_path, target_planes, move_map, args.verbose)
        
        for sample_idx, sample in matches:
            all_matches.append((file_path, sample_idx, sample))
    
    if not all_matches:
        print("\nNo matches found.")
        print("\nTips:")
        print("  - Try searching with --team-side black")
        print("  - Verify the FEN is in the expected format: FEN_A|FEN_B")
        print("  - Check that castling rights and en passant match")
        return 1
    
    print(f"\n{'#'*80}")
    print(f"Found {len(all_matches)} total matches!")
    print(f"{'#'*80}")
    
    for file_path, sample_idx, sample in all_matches:
        print(f"\nFile: {os.path.basename(file_path)}")
        display_match(sample_idx, sample, move_map)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
