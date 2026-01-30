#!/usr/bin/env python3
"""
Eyeball test for data_loaders.py - verify that samples are loaded correctly.

This script loads samples using the data_loaders.py functions and displays them
in a similar format to analyze_training_data.py, allowing visual verification that:
- Boards are represented correctly
- Policies are distributed correctly
- Flip augmentation works as expected (boards swapped, policies mirrored)

Usage:
    # Test original samples
    python eyeball_test_dataloader.py --num-samples 2
    
    # Test augmentation
    python eyeball_test_dataloader.py --test-flip
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.domain.move2planes import make_map
from src.training.data_loaders import RLDataset, flip_bughouse_sample, load_rl_parquet_shard


def is_early_game(planes: torch.Tensor) -> bool:
    """
    Check if a position is from early in the game.
    
    Heuristics:
    - Few pieces in pockets (< 3 total pieces)
    - Most pieces still on board (> 24 pieces total across both boards)
    """
    planes_np = planes.numpy()
    
    # Check both boards
    total_pocket_pieces = 0
    total_board_pieces = 0
    
    for board_offset in [0, 32]:
        # Count pocket pieces (channels 12-16 and 17-21, normalized 0-1)
        for i in range(5):  # 5 piece types in pocket
            total_pocket_pieces += int(planes_np[board_offset + 12 + i, 0, 0] * 16 + 0.5)
            total_pocket_pieces += int(planes_np[board_offset + 17 + i, 0, 0] * 16 + 0.5)
        
        # Count pieces on board (channels 0-11)
        for piece_ch in range(12):  # Our pieces (0-5) + Opponent pieces (6-11)
            total_board_pieces += np.sum(planes_np[board_offset + piece_ch] > 0.5)
    
    # Early game: few captured pieces, many pieces on board
    return total_pocket_pieces < 3 and total_board_pieces > 24


def planes_to_board_representation(planes: torch.Tensor) -> tuple[str, str, dict]:
    """
    Convert input planes to board representation for both boards.
    
    Args:
        planes: Tensor of shape (64, 8, 8)
    
    Returns: (board_a_repr, board_b_repr, metadata)
    """
    planes_np = planes.numpy()
    
    def get_board_representation(board_planes: np.ndarray) -> tuple[str, dict]:
        """Convert 32 channels for one board to board representation."""
        
        on_turn = board_planes[25, 0, 0]
        
        piece_chars = ['P', 'N', 'B', 'R', 'Q', 'K']
        board_str = []
        
        # Build board from rank 7 (top) to rank 0 (bottom)
        for rank in range(7, -1, -1):
            rank_str = []
            for file in range(8):
                piece = '.'
                
                # Check our pieces (0-5)
                for piece_idx in range(6):
                    if board_planes[piece_idx, rank, file] > 0.5:
                        piece = piece_chars[piece_idx]
                        break
                
                # Check opponent pieces (6-11)
                if piece == '.':
                    for piece_idx in range(6):
                        if board_planes[6 + piece_idx, rank, file] > 0.5:
                            piece = piece_chars[piece_idx].lower()
                            break
                
                rank_str.append(piece)
            board_str.append(' '.join(rank_str))
        
        board_repr = '\n'.join(board_str)
        
        # Pocket pieces (normalized to 0-1, so multiply by 16)
        pocket_chars = ['P', 'N', 'B', 'R', 'Q']
        our_pocket = []
        opp_pocket = []
        
        for i, char in enumerate(pocket_chars):
            our_count = int(board_planes[12 + i, 0, 0] * 16 + 0.5)
            opp_count = int(board_planes[17 + i, 0, 0] * 16 + 0.5)
            if our_count > 0:
                our_pocket.append(f"{char}x{our_count}")
            if opp_count > 0:
                opp_pocket.append(f"{char.lower()}x{opp_count}")
        
        metadata = {
            'on_turn': bool(on_turn > 0.5),
            'our_pocket': ', '.join(our_pocket) if our_pocket else 'empty',
            'opp_pocket': ', '.join(opp_pocket) if opp_pocket else 'empty',
        }
        
        return board_repr, metadata
    
    # Split planes into two boards
    board_a_planes = planes_np[:32]
    board_b_planes = planes_np[32:]
    
    board_a_repr, metadata_a = get_board_representation(board_a_planes)
    board_b_repr, metadata_b = get_board_representation(board_b_planes)
    
    metadata = {'board_a': metadata_a, 'board_b': metadata_b}
    
    return board_a_repr, board_b_repr, metadata


def decode_move_index(move_idx: int, labels: list) -> str:
    """Convert a move index to a human-readable move string."""
    if 0 <= move_idx < len(labels):
        return labels[move_idx]
    return f"unknown({move_idx})"


def print_sample(sample_idx: int, x: torch.Tensor, y_value: torch.Tensor, 
                 policy_a: torch.Tensor, policy_b: torch.Tensor, labels: list, 
                 title: str = "Sample"):
    """Print detailed information about a sample."""
    
    # Convert planes to board representation
    board_a_repr, board_b_repr, metadata = planes_to_board_representation(x)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"{title} #{sample_idx}")
    print(f"{'='*80}")
    
    # Print boards side by side
    board_a_lines = board_a_repr.split('\n')
    board_b_lines = board_b_repr.split('\n')
    board_width = 15
    
    print(f"\n{'Board A':^{board_width}s}   {'Board B':^{board_width}s}")
    print(f"{'-'*board_width}   {'-'*board_width}")
    for line_a, line_b in zip(board_a_lines, board_b_lines):
        print(f"{line_a:<{board_width}s}   {line_b}")
    
    # Print metadata
    print(f"\n{'Board A Info':^24s}   {'Board B Info':^24s}")
    print(f"  On turn: {str(metadata['board_a']['on_turn']):14s}   On turn: {metadata['board_b']['on_turn']}")
    print(f"  Our pocket: {metadata['board_a']['our_pocket']:11s}   Our pocket: {metadata['board_b']['our_pocket']}")
    print(f"  Opp pocket: {metadata['board_a']['opp_pocket']:11s}   Opp pocket: {metadata['board_b']['opp_pocket']}")
    
    print(f"\n(uppercase=our pieces, lowercase=opponent)")
    print(f"\nValue: {y_value.item():.3f} (1.0=team 0 wins, -1.0=team 1 wins)")
    
    # Print policy sums
    policy_a_sum = policy_a.sum().item()
    policy_b_sum = policy_b.sum().item()
    print(f"\nPolicy sums: Board A = {policy_a_sum:.6f}, Board B = {policy_b_sum:.6f}")
    
    # Print top moves for Board A
    print(f"\nBoard A Top Moves:")
    top_k = 10
    top_probs_a, top_indices_a = torch.topk(policy_a, k=min(top_k, len(policy_a)))
    for i, (prob, idx) in enumerate(zip(top_probs_a, top_indices_a)):
        if prob.item() > 0.001 or i == 0:
            move_str = decode_move_index(idx.item(), labels)
            print(f"  {move_str:12s} (idx {idx.item():4d}): {prob.item():.4f} ({prob.item()*100:.2f}%)")
        if prob.item() <= 0.001 and i > 0:
            break
    
    # Print top moves for Board B
    print(f"\nBoard B Top Moves:")
    top_probs_b, top_indices_b = torch.topk(policy_b, k=min(top_k, len(policy_b)))
    for i, (prob, idx) in enumerate(zip(top_probs_b, top_indices_b)):
        if prob.item() > 0.001 or i == 0:
            move_str = decode_move_index(idx.item(), labels)
            print(f"  {move_str:12s} (idx {idx.item():4d}): {prob.item():.4f} ({prob.item()*100:.2f}%)")
        if prob.item() <= 0.001 and i > 0:
            break


def test_original_samples(data_dir: str, num_samples: int = 3, early_game: bool = False):
    """Test loading and displaying original samples."""
    print("\n" + "="*80)
    print("TESTING ORIGINAL SAMPLES FROM DATA_LOADERS.PY")
    print("="*80)
    
    parquet_files = sorted(glob.glob(str(Path(data_dir) / "*.parquet")))
    if not parquet_files:
        print(f"No parquet files found in {data_dir}")
        return
    
    print(f"\nLoading from: {Path(parquet_files[0]).name}")
    
    # Load data using data_loaders.py
    x, y_value, policy_a, policy_b = load_rl_parquet_shard(parquet_files[0])
    
    print(f"Loaded {len(x)} samples")
    print(f"Sample shape: x={x.shape}, y_value={y_value.shape}, policy_a={policy_a.shape}")
    
    labels = make_map()
    
    # Find sample indices
    if early_game:
        print("\nSearching for early-game positions...")
        sample_indices = []
        for i in range(len(x)):
            if is_early_game(x[i]):
                sample_indices.append(i)
                if len(sample_indices) >= num_samples:
                    break
        print(f"Found {len(sample_indices)} early-game positions")
    else:
        sample_indices = list(range(min(num_samples, len(x))))
    
    # Display samples
    for i in sample_indices:
        print_sample(i, x[i], y_value[i], policy_a[i], policy_b[i], labels, 
                    title="Original Sample")


def test_flip_augmentation(data_dir: str, num_samples: int = 2, early_game: bool = False):
    """Test flip augmentation by comparing original and flipped samples."""
    print("\n" + "="*80)
    print("TESTING FLIP AUGMENTATION")
    print("="*80)
    
    parquet_files = sorted(glob.glob(str(Path(data_dir) / "*.parquet")))
    if not parquet_files:
        print(f"No parquet files found in {data_dir}")
        return
    
    print(f"\nLoading from: {Path(parquet_files[0]).name}")
    
    # Load data
    x, y_value, policy_a, policy_b = load_rl_parquet_shard(parquet_files[0])
    
    labels = make_map()
    
    # Find sample indices
    if early_game:
        print("\nSearching for early-game positions...")
        sample_indices = []
        for i in range(len(x)):
            if is_early_game(x[i]):
                sample_indices.append(i)
                if len(sample_indices) >= num_samples:
                    break
        print(f"Found {len(sample_indices)} early-game positions")
    else:
        sample_indices = list(range(min(num_samples, len(x))))
    
    # Test flip on samples
    for i in sample_indices:
        # Show original
        print_sample(i, x[i], y_value[i], policy_a[i], policy_b[i], labels, 
                    title="ORIGINAL Sample")
        
        # Apply flip
        x_flipped, policy_a_flipped, policy_b_flipped = flip_bughouse_sample(
            x[i], policy_a[i], policy_b[i]
        )
        
        # Show flipped
        print_sample(i, x_flipped, y_value[i], policy_a_flipped, policy_b_flipped, 
                    labels, title="FLIPPED Sample")
        
        # Verify flip correctness
        print(f"\n{'='*80}")
        print("VERIFICATION")
        print(f"{'='*80}")
        
        # Check boards swapped
        boards_swapped = torch.allclose(x_flipped[:32], x[i][32:64], atol=1e-5)
        print(f"✓ Boards swapped: {boards_swapped}")
        
        # Check policy sums preserved
        orig_sum_a = policy_a[i].sum()
        orig_sum_b = policy_b[i].sum()
        flip_sum_a = policy_a_flipped.sum()
        flip_sum_b = policy_b_flipped.sum()
        
        sum_preserved = (abs(flip_sum_a - orig_sum_b) < 0.01 and 
                        abs(flip_sum_b - orig_sum_a) < 0.01)
        print(f"✓ Policy sums preserved: {sum_preserved}")
        print(f"  Original: A={orig_sum_a:.4f}, B={orig_sum_b:.4f}")
        print(f"  Flipped:  A={flip_sum_a:.4f}, B={flip_sum_b:.4f}")
        
        # Check for specific move mirroring
        e2e4_idx = labels.index('e2e4')
        e7e5_idx = labels.index('e7e5')
        
        if policy_a[i][e2e4_idx] > 0.01:
            expected = policy_a[i][e2e4_idx].item()
            actual = policy_b_flipped[e7e5_idx].item()
            print(f"✓ Move mirroring example: e2e4 ({expected:.4f}) → e7e5 ({actual:.4f})")
        elif policy_b[i][e7e5_idx] > 0.01:
            expected = policy_b[i][e7e5_idx].item()
            actual = policy_a_flipped[e2e4_idx].item()
            print(f"✓ Move mirroring example: e7e5 ({expected:.4f}) → e2e4 ({actual:.4f})")


def test_dataset_with_augmentation(data_dir: str, num_samples: int = 2, early_game: bool = False):
    """Test RLDataset with augmentation enabled."""
    print("\n" + "="*80)
    print("TESTING RLDataset WITH AUGMENTATION")
    print("="*80)
    
    parquet_files = sorted(glob.glob(str(Path(data_dir) / "*.parquet")))
    if not parquet_files:
        print(f"No parquet files found in {data_dir}")
        return
    
    print(f"\nLoading from: {Path(parquet_files[0]).name}")
    
    # Load data
    x, y_value, policy_a, policy_b = load_rl_parquet_shard(parquet_files[0])
    
    # Filter for early game if requested
    if early_game:
        print("\nFiltering for early-game positions...")
        early_indices = [i for i in range(len(x)) if is_early_game(x[i])]
        if len(early_indices) > 100:
            early_indices = early_indices[:100]
        print(f"Found {len(early_indices)} early-game positions")
        x = x[early_indices]
        y_value = y_value[early_indices]
        policy_a = policy_a[early_indices]
        policy_b = policy_b[early_indices]
    else:
        # Use first 100 samples
        x = x[:100]
        y_value = y_value[:100]
        policy_a = policy_a[:100]
        policy_b = policy_b[:100]
    
    # Create dataset with augmentation
    dataset = RLDataset(x, y_value, policy_a, policy_b, augment_flip=True)
    
    orig_size = len(x)
    print(f"Dataset size: {len(dataset)} ({orig_size} original + {orig_size} flipped)")
    
    labels = make_map()
    
    # Show original samples (first half)
    for i in range(min(num_samples, orig_size)):
        x_i, y_i, pa_i, pb_i = dataset[i]
        print_sample(i, x_i, y_i, pa_i, pb_i, labels, 
                    title="Dataset ORIGINAL Sample")
    
    # Show flipped samples (second half)
    for i in range(min(num_samples, orig_size)):
        idx = orig_size + i  # Flipped samples start after original samples
        x_i, y_i, pa_i, pb_i = dataset[idx]
        print_sample(i, x_i, y_i, pa_i, pb_i, labels, 
                    title="Dataset FLIPPED Sample")


def main():
    parser = argparse.ArgumentParser(
        description='Eyeball test for data_loaders.py'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/home/ben/hivemind/engine/selfplay_games/training_data_parquet',
        help='Directory containing parquet files'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=2,
        help='Number of samples to display'
    )
    parser.add_argument(
        '--test-flip',
        action='store_true',
        help='Test flip augmentation (compare original vs flipped)'
    )
    parser.add_argument(
        '--test-dataset',
        action='store_true',
        help='Test RLDataset with augmentation'
    )
    parser.add_argument(
        '--early-game',
        action='store_true',
        help='Filter to show only early-game positions (few captured pieces)'
    )
    
    args = parser.parse_args()
    
    if args.test_flip:
        test_flip_augmentation(args.data_dir, args.num_samples, args.early_game)
    elif args.test_dataset:
        test_dataset_with_augmentation(args.data_dir, args.num_samples, args.early_game)
    else:
        test_original_samples(args.data_dir, args.num_samples, args.early_game)


if __name__ == '__main__':
    main()
