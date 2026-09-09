import torch
import numpy as np
from pathlib import Path
import chess

from hivemind.architectures.model_config import create_model, model_args_from_state_dict
from hivemind.network import DEFAULT_CHECKPOINT_PATH
from hivemind.domain.board import BughouseBoard
from hivemind.domain.board2planes import board2planes

def load_model_from_checkpoint(model_path: str, device='cuda'):
    """
    Load the trained RiseV3 model from checkpoint
    """
    device_obj = torch.device(device)
    if device_obj.type == 'cuda' and not torch.cuda.is_available():
        device_obj = torch.device('cpu')
    checkpoint = torch.load(model_path, map_location=device_obj)
    state_dict = checkpoint['model_state_dict']
    model = create_model(model_args_from_state_dict(state_dict))

    # Load the model state
    model.load_state_dict(state_dict)
    model = model.to(device_obj)
    model.eval()
    
    return model, device_obj

def perform_inference(model, board_tensor, device):
    """
    Perform inference on the board position
    """
    with torch.no_grad():
        # Ensure input is on correct device and has batch dimension
        if board_tensor.dim() == 3:
            board_tensor = board_tensor.unsqueeze(0)  # Add batch dimension
        
        board_tensor = board_tensor.to(device)
        
        # Forward pass
        outputs = model(board_tensor)
        value_out, policy_out = outputs[:2]
        moves_left_out = outputs[4] if len(outputs) >= 5 else None
        
        # Extract outputs
        value = value_out.squeeze().cpu().numpy()
        policy_a = torch.softmax(policy_out[0], dim=1).squeeze().cpu().numpy()
        policy_b = torch.softmax(policy_out[1], dim=1).squeeze().cpu().numpy()
        
        moves_left = None
        if moves_left_out is not None:
            moves_left = float(moves_left_out.squeeze().cpu()) * 100.0

        return value, policy_a, policy_b, moves_left

def get_starting_position_planes():
    """
    Convert the starting chess position to input planes for the model
    """
    # Create starting position
    board = BughouseBoard()
    
    planes = board2planes(board, chess.WHITE)
    # Convert to tensor
    board_tensor = torch.from_numpy(planes).float()
    
    return board_tensor, board

def main():
    """
    Main inference function
    """
    # Path to your trained model
    import argparse

    parser = argparse.ArgumentParser(description="Run checkpoint inference on the starting position")
    parser.add_argument("--model", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model_path = args.model

    # Check if model file exists
    if not Path(model_path).exists():
        parser.error(f"Model file {model_path} not found; run python tools/fetch_network.py --variant checkpoint")
    
    print("Loading RiseV3 model...")
    
    # Load the model
    try:
        model, device = load_model_from_checkpoint(model_path, args.device)
        print(f"Model loaded successfully on device: {device}")
    except Exception as e:
        raise SystemExit(f"Error loading model: {e}") from e
    
    # Get starting position
    print("Converting starting position to input planes...")
    board_tensor, board = get_starting_position_planes()
    
    print("Starting position FEN:", board.fen())
    print("Input tensor shape:", board_tensor.shape)
    
    # Perform inference
    print("Performing inference...")
    try:
        value, policy_a, policy_b, moves_left = perform_inference(model, board_tensor, device)
        
        print("\n=== INFERENCE RESULTS ===")
        print(f"Position value: {value:.6f}")
        print(f"Policy A shape: {policy_a.shape}")
        print(f"Policy B shape: {policy_b.shape}")
        if moves_left is not None:
            print(f"Predicted plies to end: {moves_left:.1f}")
        
        # Show top policy moves for both heads
        print("\nTop 5 Policy A moves:")
        top_indices_a = np.argsort(policy_a)[-5:][::-1]
        for i, idx in enumerate(top_indices_a):
            print(f"  {i+1}. Index {idx}: {policy_a[idx]:.6f}")
            
        print("\nTop 5 Policy B moves:")
        top_indices_b = np.argsort(policy_b)[-5:][::-1]
        for i, idx in enumerate(top_indices_b):
            print(f"  {i+1}. Index {idx}: {policy_b[idx]:.6f}")
        
        # Additional analysis
        print(f"\nValue interpretation: {get_value_interpretation(value)}")
        
    except Exception as e:
        raise SystemExit(f"Error during inference: {e}") from e

def get_value_interpretation(value):
    """
    Interpret the value output
    """
    if value > 0.1:
        return f"White advantage ({value:.3f})"
    elif value < -0.1:
        return f"Black advantage ({value:.3f})"
    else:
        return f"Roughly equal ({value:.3f})"

if __name__ == "__main__":
    main()