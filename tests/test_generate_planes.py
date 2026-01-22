import pytest
import os
import glob
import numpy as np
import polars as pl
from src.domain.move2planes import make_map


def test_generated_planes_validation():
    """
    Test that validates the generated planes in data/planes/train folder.
    Verifies that if both boards are on turn, the move labels cannot be pass for both boards.
    """
    labels = make_map()
    pass_label_idx = labels.index('pass')

    # Get all parquet files in the training data directory
    train_dir = "../data/planes/train"

    if not os.path.exists(train_dir):
        pytest.skip(f"Training data directory {train_dir} not found")

    parquet_files = glob.glob(os.path.join(train_dir, "*.parquet"))

    if not parquet_files:
        pytest.skip(f"No parquet files found in {train_dir}")

    total_samples = 0
    violation_count = 0

    # Process each parquet file
    for file_path in parquet_files[:10]:
        df = pl.read_parquet(file_path)

        # Extract data from the parquet file
        for row in df.iter_rows(named=True):
            # Convert bytes back to numpy array
            x_bytes = row['x']
            planes = np.frombuffer(x_bytes, dtype=np.uint8).reshape(64, 8, 8).astype(float)
            policy_idx = row['y_policy_idx']

            total_samples += 1

            # Check if both boards are on turn (channels 25 and 57)
            board_a_on_turn = planes[25, 0, 0] > 0.5  # Board A turn plane
            board_b_on_turn = planes[57, 0, 0] > 0.5  # Board B turn plane (channel 57 = 25 + 32)

            # Extract the move indices
            move_0_idx, move_1_idx = policy_idx

            # Check if both moves are 'pass'
            both_moves_are_pass = (move_0_idx == pass_label_idx and move_1_idx == pass_label_idx)

            # If both boards are on turn, both moves cannot be pass
            if board_a_on_turn and board_b_on_turn and both_moves_are_pass:
                violation_count += 1
                print(f"Violation found in {os.path.basename(file_path)}: "
                      f"Both boards on turn but both moves are pass. "
                      f"Board A on turn: {board_a_on_turn}, Board B on turn: {board_b_on_turn}, "
                      f"Move indices: {policy_idx}")

    print(f"Processed {total_samples} samples across {len(parquet_files)} files")
    print(f"Found {violation_count} violations")

    # Assert that no violations were found
    assert violation_count == 0, (
        f"Found {violation_count} violations where both boards are on turn "
        f"but both moves are 'pass' out of {total_samples} total samples"
    )