"""
Evaluate supervised model on RL self-play data as a validation set.

Usage:
    python evaluate_model.py --model path/to/model.onnx --data path/to/parquet_dir
"""
import argparse
import glob
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from tqdm import tqdm

from src.training.data_loaders import load_rl_parquet_shard, RLDataset
from src.training.metrics import Accuracy, MSE, CrossEntropy, AccuracySign


def load_onnx_model(model_path):
    """Load ONNX model and return inference session."""
    print(f"Loading ONNX model from: {model_path}")
    
    # Configure ONNX Runtime for GPU if available
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    
    # Print model info
    print(f"Execution providers: {session.get_providers()}")
    print(f"Input name: {session.get_inputs()[0].name}")
    print(f"Input shape: {session.get_inputs()[0].shape}")
    
    return session


def evaluate_model_on_data(session, data_dir, batch_size=64, max_samples=None, augment_flip=False):
    """
    Evaluate ONNX model on RL parquet data.
    
    Args:
        session: ONNX Runtime inference session
        data_dir: Directory containing parquet files
        batch_size: Batch size for inference
        max_samples: Optional limit on total samples to evaluate
        augment_flip: If True, also evaluate on board-swapped augmentations
    
    Returns:
        Dictionary of evaluation metrics
    """
    parquet_files = sorted(glob.glob(str(Path(data_dir) / "*.parquet")))
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")
    
    print(f"\nFound {len(parquet_files)} parquet files")
    print(f"Evaluating with batch_size={batch_size}, augment_flip={augment_flip}")
    
    # Initialize metrics
    # RL data has dense policy labels, so sparse_policy_label=False
    policy_acc = Accuracy(sparse_policy_label=False)
    policy_loss = CrossEntropy(sparse_policy_label=False)
    value_mse = MSE()
    value_acc_sign = AccuracySign()
    
    # Since we have policy for both boards A and B, track them separately
    policy_acc_a = Accuracy(sparse_policy_label=False)
    policy_loss_a = CrossEntropy(sparse_policy_label=False)
    policy_acc_b = Accuracy(sparse_policy_label=False)
    policy_loss_b = CrossEntropy(sparse_policy_label=False)
    
    total_samples = 0
    
    # Process each parquet file
    for pf in tqdm(parquet_files, desc="Processing parquet files"):
        x, y_value, policy_a, policy_b = load_rl_parquet_shard(pf)
        
        # Create dataset with optional augmentation
        dataset = RLDataset(x, y_value, policy_a, policy_b, augment_flip=augment_flip)
        
        # Process in batches
        for i in range(0, len(dataset), batch_size):
            batch_data = []
            for j in range(i, min(i + batch_size, len(dataset))):
                batch_data.append(dataset[j])
            
            # Unpack batch
            batch_x = torch.stack([item[0] for item in batch_data])
            batch_y_value = torch.stack([item[1] for item in batch_data])
            batch_policy_a = torch.stack([item[2] for item in batch_data])
            batch_policy_b = torch.stack([item[3] for item in batch_data])
            
            # Run inference
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: batch_x.numpy()})
            
            # outputs[0] = value, outputs[1] = policy_a, outputs[2] = policy_b
            pred_value = torch.tensor(outputs[0], dtype=torch.float32)
            pred_policy_a = torch.tensor(outputs[1], dtype=torch.float32)
            pred_policy_b = torch.tensor(outputs[2], dtype=torch.float32)
            
            # Sample weights (all 1.0 for now)
            sample_weights = torch.ones(len(batch_x))
            
            # Update metrics for value head
            value_mse.update(pred_value.squeeze(), batch_y_value, sample_weights)
            value_acc_sign.update(pred_value.squeeze(), batch_y_value, sample_weights)
            
            # Update metrics for policy A
            pred_policy_a_labels = pred_policy_a.argmax(dim=1)
            policy_acc_a.update(pred_policy_a_labels, batch_policy_a, sample_weights)
            policy_loss_a.update(pred_policy_a, batch_policy_a, sample_weights)
            
            # Update metrics for policy B
            pred_policy_b_labels = pred_policy_b.argmax(dim=1)
            policy_acc_b.update(pred_policy_b_labels, batch_policy_b, sample_weights)
            policy_loss_b.update(pred_policy_b, batch_policy_b, sample_weights)
            
            total_samples += len(batch_x)
            
            if max_samples and total_samples >= max_samples:
                break
        
        if max_samples and total_samples >= max_samples:
            break
    
    # Compute final metrics
    results = {
        'total_samples': total_samples,
        'value_mse': value_mse.compute(),
        'value_acc_sign': value_acc_sign.compute(),
        'policy_a_accuracy': policy_acc_a.compute(),
        'policy_a_loss': policy_loss_a.compute(),
        'policy_b_accuracy': policy_acc_b.compute(),
        'policy_b_loss': policy_loss_b.compute(),
    }
    
    # Average policy metrics
    results['policy_accuracy_avg'] = (results['policy_a_accuracy'] + results['policy_b_accuracy']) / 2
    results['policy_loss_avg'] = (results['policy_a_loss'] + results['policy_b_loss']) / 2
    
    return results


def print_results(results):
    """Pretty print evaluation results."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total samples evaluated: {results['total_samples']:,}")
    print()
    print("VALUE HEAD:")
    print(f"  MSE Loss:      {results['value_mse']:.6f}")
    print(f"  Sign Accuracy: {results['value_acc_sign']:.4f} ({results['value_acc_sign']*100:.2f}%)")
    print()
    print("POLICY HEAD (Board A):")
    print(f"  Accuracy:      {results['policy_a_accuracy']:.4f} ({results['policy_a_accuracy']*100:.2f}%)")
    print(f"  Cross Entropy: {results['policy_a_loss']:.6f}")
    print()
    print("POLICY HEAD (Board B):")
    print(f"  Accuracy:      {results['policy_b_accuracy']:.4f} ({results['policy_b_accuracy']*100:.2f}%)")
    print(f"  Cross Entropy: {results['policy_b_loss']:.6f}")
    print()
    print("POLICY HEAD (Average):")
    print(f"  Accuracy:      {results['policy_accuracy_avg']:.4f} ({results['policy_accuracy_avg']*100:.2f}%)")
    print(f"  Cross Entropy: {results['policy_loss_avg']:.6f}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate supervised model on RL self-play data"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/home/ben/hivemind/src/training/weights/model-0.97878-0.683-0224-v3.0.onnx",
        help="Path to ONNX model file"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/home/ben/hivemind/engine/selfplay_games/training_data_parquet",
        help="Path to directory containing parquet files"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--augment-flip",
        action="store_true",
        help="Also evaluate on board-swapped augmentations (doubles evaluation samples)"
    )
    
    args = parser.parse_args()
    
    # Verify paths exist
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Data directory not found: {args.data}")
    
    # Load model
    session = load_onnx_model(args.model)
    
    # Evaluate
    results = evaluate_model_on_data(
        session, 
        args.data, 
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        augment_flip=args.augment_flip
    )
    
    # Print results
    print_results(results)


if __name__ == "__main__":
    main()
