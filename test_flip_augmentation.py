#!/usr/bin/env python3
"""
Test flip augmentation for bughouse data.

Verifies that:
1. flip_bughouse_sample correctly swaps boards and policies
2. RLDataset with augment_flip doubles the dataset size
3. Real data augmentation works correctly
"""

import torch
import glob
from pathlib import Path

from src.training.data_loaders import (
    flip_bughouse_sample,
    RLDataset,
    load_rl_parquet_shard
)
from src.domain.move2planes import make_map


def test_flip_function():
    """Test that flip_bughouse_sample swaps boards and policies correctly."""
    print("Test 1: flip_bughouse_sample works correctly")
    
    # Create synthetic sample
    x = torch.randn(64, 8, 8)
    policy_a = torch.zeros(4672)
    policy_b = torch.zeros(4672)
    
    labels = make_map()
    
    e2e4_idx = labels.index('e2e4')
    e7e5_idx = labels.index('e7e5')
    g8f6_idx = labels.index('g8f6')
    g1f3_idx = labels.index('g1f3')
    
    policy_a[e2e4_idx] = 0.5  # e2e4 on Board A
    policy_a[g1f3_idx] = 0.3  # g1f3 on Board A
    policy_b[e7e5_idx] = 0.6  # e7e5 on Board B
    policy_b[g8f6_idx] = 0.4  # g8f6 on Board B
    
    # Apply flip
    flipped_x, flipped_pol_a, flipped_pol_b = flip_bughouse_sample(x, policy_a, policy_b)
    
    # Verify channels were swapped
    assert torch.allclose(flipped_x[:32], x[32:64]), "Board A should contain original Board B"
    assert torch.allclose(flipped_x[32:64], x[:32]), "Board B should contain original Board A"
    
    # Verify policies were swapped (NOT mirrored, just swapped)
    # What was Board B (with e7e5=0.6, g8f6=0.4) becomes Board A
    # So flipped_pol_a should have e7e5=0.6, g8f6=0.4 (same moves, just swapped)
    assert abs(flipped_pol_a[e7e5_idx] - 0.6) < 0.01, f"Expected e7e5=0.6, got {flipped_pol_a[e7e5_idx]}"
    assert abs(flipped_pol_a[g8f6_idx] - 0.4) < 0.01, f"Expected g8f6=0.4, got {flipped_pol_a[g8f6_idx]}"
    
    # What was Board A (with e2e4=0.5, g1f3=0.3) becomes Board B
    # So flipped_pol_b should have e2e4=0.5, g1f3=0.3 (same moves, just swapped)
    assert abs(flipped_pol_b[e2e4_idx] - 0.5) < 0.01, f"Expected e2e4=0.5, got {flipped_pol_b[e2e4_idx]}"
    assert abs(flipped_pol_b[g1f3_idx] - 0.3) < 0.01, f"Expected g1f3=0.3, got {flipped_pol_b[g1f3_idx]}"
    
    print("✓ flip_bughouse_sample works correctly (boards and policies swapped)")


def test_dataset_augmentation():
    """Test that RLDataset with augment_flip doubles the dataset size."""
    print("\nTest 2: Dataset augmentation works correctly")
    
    # Create small synthetic dataset
    x = torch.randn(100, 64, 8, 8)
    y_value = torch.randn(100)
    policy_a = torch.softmax(torch.randn(100, 4672), dim=1)
    policy_b = torch.softmax(torch.randn(100, 4672), dim=1)
    
    # Without augmentation
    dataset_no_aug = RLDataset(x, y_value, policy_a, policy_b, augment_flip=False)
    assert len(dataset_no_aug) == 100
    
    # With augmentation
    dataset_with_aug = RLDataset(x, y_value, policy_a, policy_b, augment_flip=True)
    assert len(dataset_with_aug) == 200  # Doubled
    
    # Check that first 100 samples are original
    for i in range(100):
        x_i, y_i, pa_i, pb_i = dataset_with_aug[i]
        assert torch.allclose(x_i, x[i])
        assert torch.allclose(pa_i, policy_a[i])
        assert torch.allclose(pb_i, policy_b[i])
    
    # Check that samples 100-199 are flipped versions
    for i in range(100):
        x_i, y_i, pa_i, pb_i = dataset_with_aug[100 + i]
        # Boards should be swapped
        assert torch.allclose(x_i[:32], x[i][32:64])
        assert torch.allclose(x_i[32:64], x[i][:32])
        # Policies should be swapped (not mirrored)
        assert torch.allclose(pa_i, policy_b[i])
        assert torch.allclose(pb_i, policy_a[i])
    
    print("  ✓ Dataset without augmentation: 100 samples")
    print("  ✓ Dataset with augmentation: 200 samples")
    print("  ✓ Dataset augmentation works correctly (policies swapped)")
    print()


def test_with_real_data():
    """Test augmentation with real parquet data."""
    print("Test 3: Real data augmentation works correctly")
    
    data_dir = '/home/ben/hivemind/engine/selfplay_games/training_data_parquet'
    parquet_files = sorted(glob.glob(str(Path(data_dir) / "*.parquet")))
    
    if not parquet_files:
        print("  ⚠ No parquet files found, skipping real data test")
        return
    
    # Load one shard
    x, y_value, policy_a, policy_b = load_rl_parquet_shard(parquet_files[0])
    n_samples = len(x)
    print(f"  Loaded {n_samples} samples from {Path(parquet_files[0]).name}")
    
    # Test dataset without augmentation
    dataset_no_aug = RLDataset(x, y_value, policy_a, policy_b, augment_flip=False)
    print(f"  Dataset size without augmentation: {len(dataset_no_aug)}")
    assert len(dataset_no_aug) == n_samples
    
    # Test dataset with augmentation
    dataset_with_aug = RLDataset(x, y_value, policy_a, policy_b, augment_flip=True)
    print(f"  Dataset size with augmentation: {len(dataset_with_aug)}")
    assert len(dataset_with_aug) == 2 * n_samples
    
    # Verify augmented samples
    print("  Verifying augmented samples...")
    for i in range(min(100, n_samples)):
        # Original sample
        x_orig, _, pa_orig, pb_orig = dataset_with_aug[i]
        
        # Flipped sample
        x_flip, _, pa_flip, pb_flip = dataset_with_aug[n_samples + i]
        
        # Boards should be swapped
        assert torch.allclose(x_flip[:32], x_orig[32:64], atol=1e-5)
        assert torch.allclose(x_flip[32:64], x_orig[:32], atol=1e-5)
        
        # Policies should be swapped (not mirrored)
        assert torch.allclose(pa_flip, pb_orig, atol=1e-5)
        assert torch.allclose(pb_flip, pa_orig, atol=1e-5)
    
    print("  ✓ Real data augmentation works correctly (policies swapped)")
    print()


if __name__ == '__main__':
    print("="*80)
    print("TESTING FLIP AUGMENTATION")
    print("="*80)
    print()
    
    test_flip_function()
    test_dataset_augmentation()
    test_with_real_data()
    
    print("="*80)
    print("ALL TESTS PASSED ✓")
    print("="*80)
