import os
import chess
import numpy as np
import polars as pl
from tqdm import tqdm
import uuid

from src.domain.board import BughouseBoard
from src.domain.board2planes import board2planes
from src.domain.move2planes import mirrorMoveUCI, make_map
from src.utils.game_reader import TrainingGameReader, process_parquet_file


class ShardWriter:
    def __init__(self, output_dir, samples_per_shard=2 ** 16):
        self.output_dir = output_dir
        self.samples_per_shard = samples_per_shard
        self.buffer = []
        os.makedirs(output_dir, exist_ok=True)

    def add_sample(self, x, policy_idx, value):
        self.buffer.append({
            "x": x.astype(np.uint8).tobytes(),
            "y_policy_idx": (int(policy_idx[0]), int(policy_idx[1])),
            "y_value": float(value)
        })
        if len(self.buffer) >= self.samples_per_shard:
            self.write_shard()

    def write_shard(self):
        if not self.buffer: return
        shard_id = uuid.uuid4().hex[:8]
        save_path = os.path.join(self.output_dir, f"shard_{shard_id}.parquet")
        pl.DataFrame(self.buffer).write_parquet(save_path, compression="zstd")
        print(f"Saved {len(self.buffer)} samples to {save_path}")
        self.buffer = []


def generate_planes(samples_per_shard=2 ** 16):
    labels = make_map()
    data_dir = 'data'
    games_path = os.path.join(data_dir, 'games.parquet')
    output_dir = os.path.join(data_dir, 'planes', 'train')
    writer = ShardWriter(output_dir, samples_per_shard)

    game_gen = process_parquet_file(games_path, min_rating=2200)
    print("Starting plane generation...")

    for reader in tqdm(game_gen, desc='Processing games'):
        if reader.time_control == -1:
            continue

        try:
            board = BughouseBoard(reader.time_control)

            # This holds the state and moves for the "current" team's turn
            current_action = {
                "team": None,  # 0 or 1
                "planes": None,  # Snapshot of board before any team moves
                "moves": [None, None]  # [board_0_move, board_1_move]
            }

            for board_num, move, time_left, move_time in reader.moves:
                board.update_time(board_num, time_left, move_time)
                side = board.boards[board_num].turn

                # Identify Team: Team 0 is (B0-White/B1-Black), Team 1 is (B0-Black/B1-White)
                # This works because partners always have opposite colors.
                moving_team = 0 if (board_num == 0 and side == chess.WHITE) or \
                                   (board_num == 1 and side == chess.BLACK) else 1

                # If the team changed, the previous team's opportunity to move is over. Flush it.
                if current_action["team"] is not None and moving_team != current_action["team"]:
                    save_team_action(writer, current_action, labels, reader.result)
                    current_action = {"team": None, "planes": None, "moves": [None, None]}

                # If this is the start of a team turn, snapshot the planes
                if current_action["planes"] is None:
                    # 'perspective_side' represents the perspective for board2planes
                    perspective_side = chess.WHITE if moving_team == 0 else chess.BLACK
                    current_action["planes"] = (board2planes(board, perspective_side), board2planes(board, not perspective_side))
                    current_action["team"] = moving_team

                # Canonicalize the move
                move_uci = move.uci()
                if side == chess.BLACK:
                    move_uci = mirrorMoveUCI(move_uci)
                if len(move_uci) == 5 and move_uci[-1] != 'n':
                    move_uci = move_uci[:-1]

                # Store the move in the buffer for the correct board
                current_action["moves"][board_num] = move_uci

                # Advance board state
                board.push(board_num, move)

            # Flush the final moves of the game
            if current_action["team"] is not None:
                save_team_action(writer, current_action, labels, reader.result)

        except Exception as e:
            print(f'Error processing game: {e}')

    writer.write_shard()


def save_team_action(writer, action, labels, game_result):
    """Utility to format and write the buffered team moves."""
    # Convert moves to labels, defaulting to 'pass' if a board didn't move
    m0 = action["moves"][0] if action["moves"][0] else 'pass'
    m1 = action["moves"][1] if action["moves"][1] else 'pass'

    assert m0 != "pass" or m1 != "pass", "Both boards didn't move!"

    policy_idx = (labels.index(m0), labels.index(m1))

    # Calculate value: If Team 0 moved, they represent the "Board 0 White" perspective
    # If Team 1 moved, they represent the "Board 0 Black" perspective
    value = game_result if action["team"] == 0 else -game_result

    has_time_advantage = action["planes"][0][31, 0, 0] > 0.5  # Check if time advantage plane is 1.0

    # Check if boards are on turn (channels 25 and 57)
    board_a_on_turn = action["planes"][0][25, 0, 0] > 0.5  # Board A turn plane
    board_b_on_turn = action["planes"][0][57, 0, 0] > 0.5  # Board B turn plane (channel 57 = 25 + 32)

    # Skip sample if team is down on time and passes on a board that's on turn
    if not has_time_advantage:
        if (m0 == 'pass' and board_a_on_turn) or (m1 == 'pass' and board_b_on_turn):
            return  # Don't add this sample

    writer.add_sample(action["planes"][0], policy_idx, value)

    # Create full pass move for other team to teach network how to sit and use time
    # From the other's team perspective it could be that both board are not on turn so both have to pass
    # Or the case we care more about: only one board is on turn and both boards still pass
    if 'pass' in [m0, m1]:
        # For the other team's sample, check their time advantage (same channel since it's duplicated)
        other_has_time_advantage = action["planes"][1][31, 0, 0] > 0.5

        # For other team's perspective, the turn channels are different
        other_board_a_on_turn = action["planes"][1][25, 0, 0] > 0.5
        other_board_b_on_turn = action["planes"][1][57, 0, 0] > 0.5

        # Skip other team's sample if they're down on time and would pass on a board that's on turn
        if not other_has_time_advantage and (other_board_a_on_turn or other_board_b_on_turn):
            return  # Don't add the other team's sample either
        # Skip if both boards are on turn since it doesn't make any sense to double sit even if up time
        if other_board_a_on_turn and other_board_b_on_turn:
            return

        writer.add_sample(action["planes"][1], (labels.index('pass'), labels.index('pass')), -value)

if __name__ == '__main__':
    generate_planes()
