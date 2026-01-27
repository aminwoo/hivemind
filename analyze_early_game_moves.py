#!/usr/bin/env python3
"""
Analyze what moves are most common in early-game positions from RL training data.
This helps verify what the model SHOULD be learning from the RL data.
"""

import glob
import numpy as np
import torch
from collections import Counter
from pathlib import Path

from src.training.data_loaders import load_rl_parquet_shard
from src.domain.move2planes import make_map


def is_early_game(planes: torch.Tensor) -> bool:
    """Check if position is early game."""
    planes_np = planes.numpy()
    total_pocket_pieces = 0
    total_board_pieces = 0
    
    for board_offset in [0, 32]:
        for i in range(5):
            total_pocket_pieces += int(planes_np[board_offset + 12 + i, 0, 0] * 16 + 0.5)
            total_pocket_pieces += int(planes_np[board_offset + 17 + i, 0, 0] * 16 + 0.5)
        
        for piece_ch in range(12):
            total_board_pieces += np.sum(planes_np[board_offset + piece_ch] > 0.5)
    
    return total_pocket_pieces < 3 and total_board_pieces > 24


def main():
    data_dir = '/home/ben/hivemind/engine/selfplay_games/training_data_parquet'
    parquet_files = sorted(glob.glob(str(Path(data_dir) / "*.parquet")))
    
    if not parquet_files:
        print(f"No parquet files found in {data_dir}")
        return
    
    labels = make_map()
    
    # Track top moves across all early-game positions
    board_a_moves = Counter()
    board_b_moves = Counter()
    early_game_count = 0
    total_samples = 0
    
    print("Analyzing early-game positions across all parquet files...")
    print(f"Found {len(parquet_files)} files\n")
    
    for file_idx, file_path in enumerate(parquet_files[:5]):  # First 5 files
        print(f"Processing {Path(file_path).name}...")
        
        # Load data
        x, y_value, policy_a, policy_b = load_rl_parquet_shard(file_path)
        
        # Find early game positions
        for i in range(len(x)):
            total_samples += 1
            
            if is_early_game(x[i]):
                early_game_count += 1
                
                # Get top move for each board
                top_move_a = torch.argmax(policy_a[i]).item()
                top_move_b = torch.argmax(policy_b[i]).item()
                
                board_a_moves[labels[top_move_a]] += 1
                board_b_moves[labels[top_move_b]] += 1
        
        print(f"  Found {early_game_count} early-game positions so far")
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Total samples examined: {total_samples}")
    print(f"Early-game positions found: {early_game_count} ({100*early_game_count/total_samples:.1f}%)")
    
    print(f"\n{'='*80}")
    print(f"BOARD A - TOP 20 MOST COMMON MOVES IN EARLY GAME")
    print(f"{'='*80}")
    for move, count in board_a_moves.most_common(20):
        pct = 100 * count / early_game_count
        print(f"  {move:12s}: {count:5d} times ({pct:5.2f}%)")
    
    print(f"\n{'='*80}")
    print(f"BOARD B - TOP 20 MOST COMMON MOVES IN EARLY GAME")
    print(f"{'='*80}")
    for move, count in board_b_moves.most_common(20):
        pct = 100 * count / early_game_count
        print(f"  {move:12s}: {count:5d} times ({pct:5.2f}%)")
    
    # Check for a7a6 specifically
    print(f"\n{'='*80}")
    print(f"CHECKING FOR SPECIFIC MOVES")
    print(f"{'='*80}")
    suspicious_moves = ['a7a6', 'h7h6', 'a2a3', 'h2h3', 'pass']
    
    for move in suspicious_moves:
        count_a = board_a_moves.get(move, 0)
        count_b = board_b_moves.get(move, 0)
        pct_a = 100 * count_a / early_game_count if early_game_count > 0 else 0
        pct_b = 100 * count_b / early_game_count if early_game_count > 0 else 0
        print(f"  {move:12s}: Board A: {count_a:5d} ({pct_a:5.2f}%)  |  Board B: {count_b:5d} ({pct_b:5.2f}%)")


if __name__ == '__main__':
    main()
