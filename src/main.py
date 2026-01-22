import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import chess
import chess.engine

# Import your model architecture
from src.architectures.rise_mobile_v3 import get_rise_v33_model
from src.domain.board import BughouseBoard
from src.domain.board2planes import board2planes
from configs.main_config import main_config

def load_model_from_checkpoint(model_path: str, device='cuda'):
    """
    Load the trained RiseV3 model from checkpoint
    """
    class Args:
        def __init__(self):
            # Model type, e.g., "risev33" (specific architecture or version of the model)
            self.model_type = "risev33"

            # Input version, e.g., "1.0" (version of the input data or model configuration)
            self.input_version = "1.0"

            # Directory where the model checkpoints will be exported or saved
            self.export_dir = "../../checkpoints"

            # Device ID for running the model (e.g., GPU device ID)
            self.device_id = 0

            # Context in which the model will run, e.g., "gpu" or "cpu"
            self.context = "gpu"

            # Input shape of the model, represented as (channels, height, width)
            # Example: 64 channels, 8 rows (height), and 8 columns (width) for an 8x8 board
            self.input_shape = (64, 8, 8)

            # Number of labels for the policy head (output layer for move predictions)
            # Example: 4672 possible moves or actions in the policy output
            self.n_labels = 0

            # Number of channels in the policy head (output layer for move predictions)
            # Example: 75 channels in the policy head
            self.channels_policy_head = 73

            # Whether to select the policy directly from the plane (spatial output)
            # If True, the policy is derived from the spatial dimensions of the output
            self.select_policy_from_plane = True

            # Whether to use a Win/Draw/Loss (WDL) head
            # If True, the model will predict the game outcome (win, draw, or loss)
            self.use_wdl = False

            # Whether to use a "plys to end" head
            # If True, the model will predict the number of plies (half-moves) remaining until the end of the game
            self.use_plys_to_end = False

            # Whether to use a Multi-Layer Perceptron (MLP) for the WDL and "plys to end" heads
            # If True, an MLP will be used to process these outputs instead of a simpler method
            self.use_mlp_wdl_ply = False


    args = Args()

    # Initialize the model architecture
    model = get_rise_v33_model(args)
    
    # Load the checkpoint
    if device == 'cuda' and torch.cuda.is_available():
        device_obj = torch.device('cuda')
        checkpoint = torch.load(model_path, map_location='cuda')
    else:
        device_obj = torch.device('cpu')
        checkpoint = torch.load(model_path, map_location='cpu')
    
    # Load the model state
    model.load_state_dict(checkpoint['model_state_dict'])
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
        value_out, policy_out = model(board_tensor)
        
        # Extract outputs
        value = value_out.squeeze().cpu().numpy()
        policy_a = torch.softmax(policy_out[0], dim=1).squeeze().cpu().numpy()
        policy_b = torch.softmax(policy_out[1], dim=1).squeeze().cpu().numpy()
        
        return value, policy_a, policy_b

def get_starting_position_planes():
    """
    Convert the starting chess position to input planes for the model
    """
    # Create starting position
    board = BughouseBoard()
    
    # Convert board to planes (you'll need to implement this based on your encoding)
    # This is a placeholder - you'll need to use your actual board2planes function
    planes = board2planes(board, chess.WHITE)
    print("Printing all plane values:")
    print("=" * 50)

    # Get the shape of planes
    num_channels, height, width = planes.shape
    print(f"Planes shape: {planes.shape}")
    print()

    # Print each channel
    for channel in range(num_channels):
        print(f"Channel {channel}:")
        # Flatten the 8x8 plane and print values in rows
        channel_data = planes[channel].flatten()

        # Print values with spaces between them
        values_str = ' '.join([str(int(val)) for val in channel_data])
        print(values_str)
        print()  # Empty line between channels

    # Convert to tensor
    board_tensor = torch.from_numpy(planes).float()
    
    return board_tensor, board

def main():
    """
    Main inference function
    """
    # Path to your trained model
    model_path = "/home/ben/hivemind/src/training/weights/model-0.89792-0.714-0082.tar"

    # Check if model file exists
    if not Path(model_path).exists():
        print(f"Model file {model_path} not found!")
        return
    
    print("Loading RiseV3 model...")
    
    # Load the model
    try:
        model, device = load_model_from_checkpoint(model_path)
        print(f"Model loaded successfully on device: {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Get starting position
    print("Converting starting position to input planes...")
    board_tensor, board = get_starting_position_planes()
    
    print("Starting position FEN:", board.fen())
    print("Input tensor shape:", board_tensor.shape)
    
    # Perform inference
    print("Performing inference...")
    try:
        value, policy_a, policy_b = perform_inference(model, board_tensor, device)
        
        print("\n=== INFERENCE RESULTS ===")
        print(f"Position value: {value:.6f}")
        print(f"Policy A shape: {policy_a.shape}")
        print(f"Policy B shape: {policy_b.shape}")
        
        # Show top policy moves for both heads
        print(f"\nTop 5 Policy A moves:")
        top_indices_a = np.argsort(policy_a)[-5:][::-1]
        for i, idx in enumerate(top_indices_a):
            print(f"  {i+1}. Index {idx}: {policy_a[idx]:.6f}")
            
        print(f"\nTop 5 Policy B moves:")
        top_indices_b = np.argsort(policy_b)[-5:][::-1]
        for i, idx in enumerate(top_indices_b):
            print(f"  {i+1}. Index {idx}: {policy_b[idx]:.6f}")
        
        # Additional analysis
        print(f"\nValue interpretation: {get_value_interpretation(value)}")
        
    except Exception as e:
        print(f"Error during inference: {e}")

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