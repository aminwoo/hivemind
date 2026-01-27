"""
Deep dive analysis: Check if 'pass' moves are legitimate in the training data
"""
import glob
import torch
from pathlib import Path
from collections import Counter
from tqdm import tqdm

from src.training.data_loaders import load_rl_parquet_shard
from src.domain.move2planes import make_map


def analyze_pass_distribution(data_dir, max_samples=50000):
    """
    Check what percentage of training samples have 'pass' as a top move
    and whether this is appropriate.
    """
    parquet_files = sorted(glob.glob(str(Path(data_dir) / "*.parquet")))
    labels = make_map()
    
    print(f"Found {len(parquet_files)} parquet files")
    print("Analyzing pass move distribution...\n")
    
    # Track statistics
    total_samples = 0
    pass_as_top_move = 0
    pass_in_top_3 = 0
    pass_in_top_5 = 0
    
    # Distribution of which board has pass
    pass_board_a_only = 0
    pass_board_b_only = 0
    pass_both_boards = 0
    
    for pf in tqdm(parquet_files[:5], desc="Processing files"):
        x, y_value, policy_a, policy_b = load_rl_parquet_shard(pf)
        
        for i in range(min(len(x), max_samples - total_samples)):
            total_samples += 1
            
            # Get top moves for each board
            top_3_a = torch.topk(policy_a[i], k=3)
            top_3_b = torch.topk(policy_b[i], k=3)
            
            top_move_a = top_3_a.indices[0].item()
            top_move_b = top_3_b.indices[0].item()
            
            # Check if pass is in top positions
            # Pass moves are indices 0-63 (first 64 labels)
            top_is_pass_a = top_move_a < 64
            top_is_pass_b = top_move_b < 64
            
            if top_is_pass_a or top_is_pass_b:
                pass_as_top_move += 1
                
                if top_is_pass_a and not top_is_pass_b:
                    pass_board_a_only += 1
                elif top_is_pass_b and not top_is_pass_a:
                    pass_board_b_only += 1
                elif top_is_pass_a and top_is_pass_b:
                    pass_both_boards += 1
            
            # Check top 3
            if any(idx < 64 for idx in top_3_a.indices[:3].tolist()) or \
               any(idx < 64 for idx in top_3_b.indices[:3].tolist()):
                pass_in_top_3 += 1
            
            # Check top 5
            top_5_a = torch.topk(policy_a[i], k=5)
            top_5_b = torch.topk(policy_b[i], k=5)
            if any(idx < 64 for idx in top_5_a.indices.tolist()) or \
               any(idx < 64 for idx in top_5_b.indices.tolist()):
                pass_in_top_5 += 1
            
            if total_samples >= max_samples:
                break
        
        if total_samples >= max_samples:
            break
    
    print(f"\n{'='*70}")
    print("PASS MOVE DISTRIBUTION ANALYSIS")
    print(f"{'='*70}")
    print(f"Total samples analyzed: {total_samples:,}")
    print(f"\nPass as top move: {pass_as_top_move:,} ({100*pass_as_top_move/total_samples:.2f}%)")
    print(f"  - Board A only: {pass_board_a_only:,} ({100*pass_board_a_only/total_samples:.2f}%)")
    print(f"  - Board B only: {pass_board_b_only:,} ({100*pass_board_b_only/total_samples:.2f}%)")
    print(f"  - Both boards:  {pass_both_boards:,} ({100*pass_both_boards/total_samples:.2f}%)")
    print(f"\nPass in top 3: {pass_in_top_3:,} ({100*pass_in_top_3/total_samples:.2f}%)")
    print(f"Pass in top 5: {pass_in_top_5:,} ({100*pass_in_top_5/total_samples:.2f}%)")
    print(f"\n{'='*70}")
    print("\n⚠ WARNING: If pass moves appear too frequently, it suggests:")
    print("1. Training data may have too many positions where passing is appropriate")
    print("2. Model may be learning to pass instead of making active moves")
    print("3. Consider filtering training data to exclude excessive pass situations")


if __name__ == "__main__":
    data_dir = "/home/ben/hivemind/engine/selfplay_games/training_data_parquet"
    analyze_pass_distribution(data_dir, max_samples=50000)
