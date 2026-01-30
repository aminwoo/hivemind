"""
Analyze move distribution in RL training data to investigate strange model behavior.
Specifically checking for a7a6 after 1.e2e4
"""
import glob
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm

from src.training.data_loaders import load_rl_parquet_shard


# Move encoding constants (73 channels for bughouse)
NB_POLICY_CHANNELS = 73
BOARD_SIZE = 8


def decode_policy_to_moves(policy_tensor, board_idx='A', top_k=10):
    """
    Decode policy tensor to top-k moves.
    Policy is shape (4672,) = 73 channels * 8 * 8
    
    Returns list of (probability, channel, from_sq, to_sq) tuples
    """
    # Reshape to (73, 8, 8)
    policy = policy_tensor.reshape(NB_POLICY_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    
    # Get top-k indices
    flat_policy = policy_tensor.flatten()
    top_k_values, top_k_indices = torch.topk(flat_policy, k=min(top_k, len(flat_policy)))
    
    moves = []
    for prob, idx in zip(top_k_values.tolist(), top_k_indices.tolist()):
        # Decode index to (channel, row, col)
        channel = idx // (BOARD_SIZE * BOARD_SIZE)
        remainder = idx % (BOARD_SIZE * BOARD_SIZE)
        row = remainder // BOARD_SIZE
        col = remainder % BOARD_SIZE
        
        # Convert to chess notation (approximate)
        from_sq = f"{chr(col + ord('a'))}{row + 1}"
        
        moves.append({
            'prob': prob,
            'channel': channel,
            'to_square': from_sq,
            'raw_idx': idx
        })
    
    return moves


def check_position_for_e2e4(x_planes, policy_a, policy_b, sample_idx):
    """
    Check if this position looks like the position after 1.e2e4
    Returns analysis dict if it matches, None otherwise
    """
    # After 1.e2e4, we expect:
    # - White pawn on e4 (channel 0, square e4 = col 4, row 3)
    # - No white pawn on e2 (col 4, row 1)
    
    white_pawns = x_planes[0]  # Channel 0 = white pawns on board A
    
    # Check if there's a pawn on e4 and not on e2
    has_pawn_e4 = white_pawns[3, 4] > 0.5  # e4 is row 3 (0-indexed), col 4
    no_pawn_e2 = white_pawns[1, 4] < 0.5  # e2 is row 1, col 4
    
    # Check that most other pawns are still in starting position
    starting_pawns = 0
    for col in range(8):
        if col != 4 and white_pawns[1, col] > 0.5:  # row 1 (rank 2)
            starting_pawns += 1
    
    if has_pawn_e4 and no_pawn_e2 and starting_pawns >= 6:
        # This looks like after 1.e2e4
        # Analyze the policy
        top_moves = decode_policy_to_moves(policy_a, board_idx='A', top_k=20)
        
        return {
            'sample_idx': sample_idx,
            'top_moves': top_moves,
            'policy_a': policy_a,
            'policy_b': policy_b,
            'planes': x_planes
        }
    
    return None


def analyze_opening_positions(data_dir, max_samples=100000):
    """
    Search through RL data for positions after 1.e2e4 and analyze move distribution.
    """
    parquet_files = sorted(glob.glob(str(Path(data_dir) / "*.parquet")))
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")
    
    print(f"Found {len(parquet_files)} parquet files")
    print(f"Searching for positions after 1.e2e4...")
    print()
    
    matching_positions = []
    total_samples_checked = 0
    
    # Track all moves seen after e2e4
    move_counter = Counter()
    
    for pf in tqdm(parquet_files[:10], desc="Processing files"):  # Check first 10 files
        x, y_value, policy_a, policy_b = load_rl_parquet_shard(pf)
        
        for i in range(len(x)):
            total_samples_checked += 1
            
            analysis = check_position_for_e2e4(x[i], policy_a[i], policy_b[i], total_samples_checked)
            if analysis:
                matching_positions.append(analysis)
                
                # Track the top move
                if analysis['top_moves']:
                    top_move = analysis['top_moves'][0]
                    move_counter[f"ch{top_move['channel']}_sq{top_move['to_square']}"] += 1
            
            if total_samples_checked >= max_samples:
                break
        
        if total_samples_checked >= max_samples:
            break
    
    print(f"\nTotal samples checked: {total_samples_checked}")
    print(f"Positions matching '1.e2e4': {len(matching_positions)}")
    print()
    
    if matching_positions:
        print("="*70)
        print("TOP MOVE DISTRIBUTION AFTER 1.e2e4")
        print("="*70)
        for move, count in move_counter.most_common(20):
            percentage = 100 * count / len(matching_positions)
            print(f"{move:30s} : {count:5d} times ({percentage:5.1f}%)")
        print()
        
        # Show detailed analysis of first few positions
        print("="*70)
        print("DETAILED ANALYSIS OF SAMPLE POSITIONS")
        print("="*70)
        for idx, pos in enumerate(matching_positions[:5]):
            print(f"\nSample {pos['sample_idx']}:")
            print(f"Top 10 moves (Board A):")
            for rank, move in enumerate(pos['top_moves'][:10], 1):
                print(f"  {rank:2d}. Channel {move['channel']:2d}, Square {move['to_square']}, Prob: {move['prob']:.6f}")
            
            # Check specifically for a7a6 pattern
            # a7 = col 0, row 6; a6 = col 0, row 5
            # Need to figure out which channel corresponds to pawn moves
            
        # Aggregate statistics across all matching positions
        print("\n" + "="*70)
        print("AGGREGATE STATISTICS")
        print("="*70)
        
        # Average policy distribution for top moves
        avg_top_channels = defaultdict(float)
        for pos in matching_positions:
            for move in pos['top_moves'][:5]:
                avg_top_channels[move['channel']] += 1
        
        print(f"\nMost common policy channels in top 5:")
        for channel, count in sorted(avg_top_channels.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  Channel {channel:2d}: appears in top-5 {count} times ({100*count/len(matching_positions):.1f}%)")
    else:
        print("No positions found matching '1.e2e4' pattern")
    
    return matching_positions


if __name__ == "__main__":
    data_dir = "/home/ben/hivemind/engine/selfplay_games/training_data_parquet"
    
    print("Analyzing RL training data for move distribution")
    print("Investigating a7a6 suggestion after 1.e2e4")
    print()
    
    positions = analyze_opening_positions(data_dir, max_samples=200000)
    
    if positions:
        print(f"\n✓ Analysis complete. Found {len(positions)} matching positions.")
    else:
        print("\n⚠ No matching positions found.")
