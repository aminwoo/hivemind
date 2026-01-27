#!/usr/bin/env python3
"""
Perform model inference on a bughouse position given FEN strings.

Usage:
    # Provide FEN for both boards
    python infer_from_fen.py --fen-a "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1" --fen-b "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    # Use starting position
    python infer_from_fen.py --starting
"""

import argparse
import numpy as np
import torch
import onnxruntime as ort
import chess

from src.domain.board import BughouseBoard
from src.domain.board2planes import board2planes
from src.domain.move2planes import make_map


def load_onnx_model(model_path):
    """Load ONNX model and return inference session."""
    print(f"Loading ONNX model from: {model_path}")
    
    # Configure ONNX Runtime for GPU if available
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    
    print(f"Execution providers: {session.get_providers()}")
    return session


def infer_from_fens(session, fen_a, fen_b, team_side=chess.WHITE, top_k=10):
    """
    Run inference on a bughouse position given two FEN strings.
    
    Args:
        session: ONNX Runtime inference session
        fen_a: FEN string for Board A
        fen_b: FEN string for Board B
        team_side: Which team's perspective (chess.WHITE or chess.BLACK)
        top_k: Number of top moves to display
    
    Returns:
        Dictionary with predictions
    """
    # Create bughouse board
    board = BughouseBoard()
    board.set_fen(f"{fen_a} | {fen_b}")
    
    # Convert to planes
    planes = board2planes(board, team_side, flip=False)
    
    # Convert to tensor and add batch dimension
    x = torch.from_numpy(planes).float().unsqueeze(0)
    
    # Run inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: x.numpy()})
    
    # Parse outputs
    # outputs[0] = value, outputs[1] = policy_a, outputs[2] = policy_b
    value = outputs[0][0][0]
    policy_a_logits = outputs[1][0]
    policy_b_logits = outputs[2][0]
    
    # Apply softmax to convert logits to probabilities
    def softmax(x):
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / exp_x.sum()
    
    policy_a = softmax(policy_a_logits)
    policy_b = softmax(policy_b_logits)
    
    # Get move labels
    labels = make_map()
    
    # Get top moves for Board A
    top_indices_a = np.argsort(policy_a)[-top_k:][::-1]
    top_moves_a = [(labels[idx], policy_a[idx]) for idx in top_indices_a]
    
    # Get top moves for Board B
    top_indices_b = np.argsort(policy_b)[-top_k:][::-1]
    top_moves_b = [(labels[idx], policy_b[idx]) for idx in top_indices_b]
    
    return {
        'value': value,
        'policy_a': policy_a,
        'policy_b': policy_b,
        'top_moves_a': top_moves_a,
        'top_moves_b': top_moves_b,
        'board': board
    }


def print_results(results):
    """Pretty print inference results."""
    board = results['board']
    
    print("\n" + "="*80)
    print("BUGHOUSE POSITION")
    print("="*80)
    
    # Print boards
    print("\nBoard A:")
    print(board.boards[0])
    print(f"Turn: {'White' if board.boards[0].turn else 'Black'}")
    print(f"Pockets: White={board.boards[0].pockets[chess.WHITE]}, Black={board.boards[0].pockets[chess.BLACK]}")
    
    print("\nBoard B:")
    print(board.boards[1])
    print(f"Turn: {'White' if board.boards[1].turn else 'Black'}")
    print(f"Pockets: White={board.boards[1].pockets[chess.WHITE]}, Black={board.boards[1].pockets[chess.BLACK]}")
    
    # Print value prediction
    print("\n" + "="*80)
    print("MODEL PREDICTIONS")
    print("="*80)
    
    value = results['value']
    print(f"\nValue (from team perspective): {value:.4f}")
    if value > 0.5:
        print(f"  → Team 0 winning ({value*100:.1f}%)")
    elif value < -0.5:
        print(f"  → Team 1 winning ({abs(value)*100:.1f}%)")
    else:
        print(f"  → Roughly equal position")
    
    # Print top moves for Board A
    print(f"\nBoard A - Top {len(results['top_moves_a'])} Moves:")
    for i, (move, prob) in enumerate(results['top_moves_a'], 1):
        if prob > 0.001:  # Only show moves with >0.1% probability
            print(f"  {i}. {move:10s} - {prob:.4f} ({prob*100:.2f}%)")
    
    # Print top moves for Board B
    print(f"\nBoard B - Top {len(results['top_moves_b'])} Moves:")
    for i, (move, prob) in enumerate(results['top_moves_b'], 1):
        if prob > 0.001:  # Only show moves with >0.1% probability
            print(f"  {i}. {move:10s} - {prob:.4f} ({prob*100:.2f}%)")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Perform inference on a bughouse position"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/home/ben/hivemind/src/training/weights/model-0.97878-0.683-0224-v3.0.onnx",
        help="Path to ONNX model file"
    )
    parser.add_argument(
        "--fen-a",
        type=str,
        help="FEN string for Board A"
    )
    parser.add_argument(
        "--fen-b",
        type=str,
        help="FEN string for Board B"
    )
    parser.add_argument(
        "--starting",
        action="store_true",
        help="Use starting position for both boards"
    )
    parser.add_argument(
        "--team",
        type=str,
        default="white",
        choices=["white", "black"],
        help="Team perspective (white or black)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top moves to display"
    )
    
    args = parser.parse_args()
    
    # Determine FENs to use
    if args.starting:
        fen_a = chess.STARTING_FEN
        fen_b = chess.STARTING_FEN
        print("Using starting position for both boards")
    elif args.fen_a and args.fen_b:
        fen_a = args.fen_a
        fen_b = args.fen_b
    else:
        parser.error("Must provide either --starting or both --fen-a and --fen-b")
    
    # Determine team side
    team_side = chess.WHITE if args.team == "white" else chess.BLACK
    
    # Load model
    session = load_onnx_model(args.model)
    
    # Run inference
    results = infer_from_fens(session, fen_a, fen_b, team_side, args.top_k)
    
    # Print results
    print_results(results)


if __name__ == "__main__":
    main()
