# Bughouse Board Flip Augmentation

## Overview

Implemented board flip augmentation for bughouse RL training data to effectively **double the training dataset size**. This leverages the symmetry in bughouse where swapping Board A and Board B creates an equivalent position from the partner's perspective.

## What Was Implemented

### 1. Core Augmentation Function (`src/training/data_loaders.py`)

```python
def flip_bughouse_sample(x, policy_a, policy_b):
    """
    Swap Board A (channels 0-31) with Board B (channels 32-63)
    and swap the policy distributions.
    """
```

### 2. Enhanced Datasets

#### `RLDataset`

- Added `augment_flip` parameter to constructor
- When enabled, dataset length is doubled
- First half contains original samples, second half contains flipped samples
- Automatically applies flip transformation on-the-fly

#### `StreamingRLDataset`

- Added `augment_flip` parameter
- Yields both original and flipped versions of each sample when enabled
- Maintains shuffling behavior with augmented samples mixed in

### 3. Updated Training Function

Modified `train_rl()` in `src/training/train_loop.py`:

- Added `augment_flip: bool = True` parameter (enabled by default)
- Applies augmentation to both in-memory and streaming datasets
- Correctly estimates total training samples with augmentation
- Prints augmentation status during training

## Usage

### In Training Code

```python
from src.training.train_loop import train_rl

# With augmentation (default, doubles data)
train_rl(
    rl_data_dir="/path/to/parquet",
    augment_flip=True  # 2x training data
)

# Without augmentation
train_rl(
    rl_data_dir="/path/to/parquet",
    augment_flip=False  # Original data size
)
```

### Direct Dataset Usage

```python
from src.training.data_loaders import RLDataset, load_rl_parquet_shard

x, y_value, policy_a, policy_b = load_rl_parquet_shard("shard.parquet")

# With augmentation (2x size)
dataset = RLDataset(x, y_value, policy_a, policy_b, augment_flip=True)
print(len(dataset))  # 2 * len(x)

# Without augmentation
dataset = RLDataset(x, y_value, policy_a, policy_b, augment_flip=False)
print(len(dataset))  # len(x)
```

## Benefits

1. **Doubles training data** without additional self-play games
2. **Zero storage overhead** - augmentation happens on-the-fly
3. **Minimal compute overhead** - just channel swapping and policy swapping
4. **Mathematically valid** - exploits true symmetry in bughouse
5. **Seamless integration** - works with both in-memory and streaming datasets

## Verification

A comprehensive test suite validates the implementation:

```bash
python test_flip_augmentation.py
```

Tests verify:

- ✓ Channel swapping (Board A ↔ Board B)
- ✓ Policy distribution swapping
- ✓ Value preservation
- ✓ Dataset size doubling
- ✓ Real parquet data compatibility

## How It Works

### Bughouse Plane Structure (64 channels)

- **Channels 0-31**: Board A state
- **Channels 32-63**: Board B state

### Flip Transformation

1. **Input planes**: Swap channels 0-31 with 32-63
2. **Policy A**: Becomes original Policy B
3. **Policy B**: Becomes original Policy A
4. **Value**: Unchanged (team-level value)

This creates a valid alternative representation of the same position from the partner's perspective.

## Files Modified

- `src/training/data_loaders.py` - Added flip function and augmentation support
- `src/training/train_loop.py` - Added augmentation parameter to training
- `test_flip_augmentation.py` - Comprehensive test suite (new file)
- `evaluate_model.py` - Model evaluation script (new file)

## Performance Impact

- **Memory**: Minimal (single sample stored temporarily during flip)
- **CPU**: Negligible (simple tensor indexing)
- **Training time**: Same per sample, but 2x more samples per epoch
- **Model quality**: Expected improvement from larger effective dataset

## Next Steps

When ready to train with RL data:

```python
python -m src.main train_rl \
    --rl-data-dir /home/ben/hivemind/engine/selfplay_games/training_data_parquet \
    --checkpoint /home/ben/hivemind/src/training/weights/model-0.97878-0.683-0224-v3.0.tar
```

The augmentation will automatically double your effective training data!
