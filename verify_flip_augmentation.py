#!/usr/bin/env python3
"""
Verify flip augmentation - check if moves should be mirrored when swapping boards.

This script loads a sample, applies flip augmentation, and shows:
1. Original position and top moves
2. Flipped position and top moves
3. Whether the moves make sense for the flipped position
"""

import numpy as np
import torch
import glob
from pathlib import Path

from src.training.data_loaders import load_rl_parquet_shard, flip_bughouse_sample
from src.domain.move2planes import make_map


def decode_board(planes: torch.Tensor, board_offset: int = 0) -> str:
    """Decode a single board from planes."""
    planes_np = planes.numpy()
    board_planes = planes_np[board_offset:board_offset+32]
    
    piece_chars = ['P', 'N', 'B', 'R', 'Q', 'K']
    board_lines = []
    
    for rank in range(7, -1, -1):
        rank_str = []
        for file in range(8):
            piece = '.'
            # Our pieces (0-5)
            for piece_idx in range(6):
                if board_planes[piece_idx, rank, file] > 0.5:
                    piece = piece_chars[piece_idx]
                    break
            # Opponent pieces (6-11)
            if piece == '.':
                for piece_idx in range(6):
                    if board_planes[6 + piece_idx, rank, file] > 0.5:
                        piece = piece_chars[piece_idx].lower()
                        break
            rank_str.append(piece)
        board_lines.append(' '.join(rank_str))
    
    return '\n'.join(board_lines)


def check_move_validity(board_str: str, move_str: str) -> str:
    """
    Check if a move seems valid for a board position.
    Returns a string explaining whether it makes sense.
    """
    # Parse the move
    if len(move_str) < 4:
        return "❓ Not a standard move"
    
    from_sq = move_str[:2]
    to_sq = move_str[2:4]
    
    # Check if it's a standard move (not a drop or pass)
    if not (from_sq[0].isalpha() and from_sq[1].isdigit()):
        return "❓ Not a standard move (might be drop/pass)"
    
    # Convert square notation to coordinates
    from_file = ord(from_sq[0]) - ord('a')
    from_rank = int(from_sq[1]) - 1
    to_file = ord(to_sq[0]) - ord('a')
    to_rank = int(to_sq[1]) - 1
    
    # Get board as list of lines
    board_lines = board_str.split('\n')
    
    # Board is displayed rank 8 to 1, so reverse index
    from_rank_idx = 7 - from_rank
    to_rank_idx = 7 - to_rank
    
    # Get the piece at from square
    from_line = board_lines[from_rank_idx].split()
    to_line = board_lines[to_rank_idx].split()
    
    if from_file >= len(from_line):
        return "❌ Invalid from square"
    
    from_piece = from_line[from_file]
    to_piece = to_line[to_file] if to_file < len(to_line) else '.'
    
    if from_piece == '.':
        return f"❌ No piece at {from_sq}"
    
    # Check if it's our piece (uppercase)
    if from_piece.islower():
        return f"❌ {from_sq} has opponent piece '{from_piece}'"
    
    return f"✓ Moves '{from_piece}' from {from_sq} to {to_sq}"


def main():
    data_dir = '/home/ben/hivemind/engine/selfplay_games/training_data_parquet'
    
    parquet_files = sorted(glob.glob(str(Path(data_dir) / "*.parquet")))
    if not parquet_files:
        print(f"No parquet files found in {data_dir}")
        return
    
    print(f"Loading from: {Path(parquet_files[0]).name}\n")
    
    # Load data
    x, y_value, policy_a, policy_b = load_rl_parquet_shard(parquet_files[0])
    
    labels = make_map()
    
    # Find an early game sample
    sample_idx = None
    for i in range(min(100, len(x))):
        # Check if it's early game (many pieces, few captures)
        planes_np = x[i].numpy()
        total_pocket = 0
        for board_offset in [0, 32]:
            for j in range(5):
                total_pocket += int(planes_np[board_offset + 12 + j, 0, 0] * 16 + 0.5)
                total_pocket += int(planes_np[board_offset + 17 + j, 0, 0] * 16 + 0.5)
        
        if total_pocket < 3:
            sample_idx = i
            break
    
    if sample_idx is None:
        print("No early game positions found, using first sample")
        sample_idx = 0
    
    print(f"Analyzing sample {sample_idx}\n")
    print("="*80)
    
    # Original sample
    x_orig = x[sample_idx]
    policy_a_orig = policy_a[sample_idx]
    policy_b_orig = policy_b[sample_idx]
    
    # Decode boards
    board_a_orig = decode_board(x_orig, 0)
    board_b_orig = decode_board(x_orig, 32)
    
    print("ORIGINAL CONFIGURATION:")
    print("\nBoard A (channels 0-31):")
    print(board_a_orig)
    print("\nBoard A top moves:")
    top_indices_a = torch.topk(policy_a_orig, k=5).indices
    for idx in top_indices_a:
        move = labels[idx.item()]
        prob = policy_a_orig[idx].item()
        validity = check_move_validity(board_a_orig, move)
        print(f"  {move:10s} ({prob:.4f}) - {validity}")
    
    print("\nBoard B (channels 32-63):")
    print(board_b_orig)
    print("\nBoard B top moves:")
    top_indices_b = torch.topk(policy_b_orig, k=5).indices
    for idx in top_indices_b:
        move = labels[idx.item()]
        prob = policy_b_orig[idx].item()
        validity = check_move_validity(board_b_orig, move)
        print(f"  {move:10s} ({prob:.4f}) - {validity}")
    
    print("\n" + "="*80)
    
    # Apply flip augmentation
    x_flipped, policy_a_flipped, policy_b_flipped = flip_bughouse_sample(
        x_orig, policy_a_orig, policy_b_orig
    )
    
    # Decode flipped boards
    board_a_flipped = decode_board(x_flipped, 0)
    board_b_flipped = decode_board(x_flipped, 32)
    
    print("\nAFTER FLIP AUGMENTATION:")
    print("\nBoard A (was Board B, channels 32-63 → 0-31):")
    print(board_a_flipped)
    print("\nBoard A top moves (policy_b_flipped with mirroring):")
    top_indices_a_flip = torch.topk(policy_a_flipped, k=5).indices
    for idx in top_indices_a_flip:
        move = labels[idx.item()]
        prob = policy_a_flipped[idx].item()
        validity = check_move_validity(board_a_flipped, move)
        print(f"  {move:10s} ({prob:.4f}) - {validity}")
    
    print("\nBoard B (was Board A, channels 0-31 → 32-63):")
    print(board_b_flipped)
    print("\nBoard B top moves (policy_a_flipped with mirroring):")
    top_indices_b_flip = torch.topk(policy_b_flipped, k=5).indices
    for idx in top_indices_b_flip:
        move = labels[idx.item()]
        prob = policy_b_flipped[idx].item()
        validity = check_move_validity(board_b_flipped, move)
        print(f"  {move:10s} ({prob:.4f}) - {validity}")
    
    print("\n" + "="*80)
    print("\nVERIFICATION:")
    print("\n1. Board comparison:")
    print(f"   Original Board A == Flipped Board B? {torch.allclose(x_orig[:32], x_flipped[32:64], atol=1e-5)}")
    print(f"   Original Board B == Flipped Board A? {torch.allclose(x_orig[32:64], x_flipped[:32], atol=1e-5)}")
    
    print("\n2. Move transformation check:")
    print("   Looking for specific move transformations...")
    
    # Check if any original move appears and its mirrored version appears in flipped
    for orig_idx in top_indices_a[:3]:
        orig_move = labels[orig_idx.item()]
        orig_prob = policy_a_orig[orig_idx].item()
        
        # What should this move become after mirroring?
        if len(orig_move) >= 4 and orig_move[0:2].isalpha() and orig_move[0:2][1].isdigit():
            from_file = orig_move[0]
            from_rank = orig_move[1]
            to_file = orig_move[2]
            to_rank = orig_move[3]
            
            # Mirror vertically: rank 1↔8, 2↔7, 3↔6, 4↔5
            rank_map = {'1': '8', '2': '7', '3': '6', '4': '5', 
                       '5': '4', '6': '3', '7': '2', '8': '1'}
            
            if from_rank in rank_map and to_rank in rank_map:
                expected_move = from_file + rank_map[from_rank] + to_file + rank_map[to_rank]
                if len(orig_move) > 4:
                    expected_move += orig_move[4:]
                
                if expected_move in labels:
                    expected_idx = labels.index(expected_move)
                    flipped_prob = policy_b_flipped[expected_idx].item()
                    
                    print(f"   Original Board A: {orig_move:10s} ({orig_prob:.4f})")
                    print(f"   Expected in Flipped Board B: {expected_move:10s}")
                    print(f"   Actual in Flipped Board B: {expected_move:10s} ({flipped_prob:.4f})")
                    print(f"   Probabilities match? {abs(orig_prob - flipped_prob) < 0.001}")
                    print()


if __name__ == '__main__':
    main()
