import os
import chess
import numpy as np
import polars as pl
from tqdm import tqdm
import uuid

from src.domain.board import BughouseBoard
from src.domain.board2planes import board2planes
from src.domain.move2planes import mirrorMoveUCI, make_map
from src.constants import NUM_BUGHOUSE_CHANNELS, NUM_BUGHOUSE_CHANNELS_PER_BOARD
from src.utils.game_reader import TrainingGameReader, process_parquet_file


class ShardWriter:
    def __init__(
        self,
        output_dir,
        samples_per_shard=2 ** 16,
        augment_board_swap=True,
        max_samples=None,
    ):
        self.output_dir = output_dir
        self.samples_per_shard = samples_per_shard
        self.augment_board_swap = augment_board_swap
        self.max_samples = max_samples
        self.samples_written = 0
        self.buffer = []
        self.written_paths = []
        os.makedirs(output_dir, exist_ok=True)

    @property
    def is_full(self):
        return (
            self.max_samples is not None
            and self.samples_written + len(self.buffer) >= self.max_samples
        )

    @staticmethod
    def _encode_planes(x: np.ndarray) -> bytes:
        offset = NUM_BUGHOUSE_CHANNELS_PER_BOARD
        encoded_x = x.copy()
        encoded_x[12:22] *= 16.0
        encoded_x[offset + 12:offset + 22] *= 16.0
        return np.rint(encoded_x).astype(np.uint8).tobytes()

    def _append_encoded_sample(self, encoded_x, policy_idx, value, plys_to_end):
        if self.is_full:
            return
        self.buffer.append({
            "x": encoded_x,
            "y_policy_idx": (int(policy_idx[0]), int(policy_idx[1])),
            "y_value": float(value),
            "y_plys_to_end": int(plys_to_end),
        })

    def add_sample(self, x, policy_idx, value, plys_to_end):
        self._append_encoded_sample(
            encoded_x=self._encode_planes(x),
            policy_idx=policy_idx,
            value=value,
            plys_to_end=plys_to_end,
        )

        if self.augment_board_swap:
            offset = NUM_BUGHOUSE_CHANNELS // 2
            swapped_x = x.copy()
            swapped_x[:offset] = x[offset:]
            swapped_x[offset:] = x[:offset]
            swapped_policy_idx = (policy_idx[1], policy_idx[0])
            self._append_encoded_sample(
                encoded_x=self._encode_planes(swapped_x),
                policy_idx=swapped_policy_idx,
                value=value,
                plys_to_end=plys_to_end,
            )

        if len(self.buffer) >= self.samples_per_shard:
            self.write_shard()

    def write_shard(self):
        if not self.buffer: return
        shard_id = uuid.uuid4().hex[:8]
        save_path = os.path.join(self.output_dir, f"shard_{shard_id}.parquet")
        temporary_path = f"{save_path}.tmp"
        pl.DataFrame(self.buffer).write_parquet(temporary_path, compression="zstd")
        os.replace(temporary_path, save_path)
        print(f"Saved {len(self.buffer)} samples to {save_path}")
        self.samples_written += len(self.buffer)
        self.written_paths.append(save_path)
        self.buffer = []


def generate_planes(
    samples_per_shard=2 ** 16,
    games_path=None,
    output_dir=None,
    min_rating=2200,
    augment_board_swap=True,
    split=None,
    val_fraction=0.02,
    seed=42,
    max_samples=None,
):
    label_to_index = {label: index for index, label in enumerate(make_map())}
    data_dir = 'data'
    if games_path is None:
        games_path = os.path.join(data_dir, 'games.parquet')
    if output_dir is None:
        output_dir = os.path.join(data_dir, 'planes', 'train')
    writer = ShardWriter(
        output_dir,
        samples_per_shard,
        augment_board_swap=augment_board_swap,
        max_samples=max_samples,
    )

    game_gen = process_parquet_file(
        games_path,
        min_rating=min_rating,
        split=split,
        val_fraction=val_fraction,
        seed=seed,
    )
    print("Starting plane generation...")

    for reader in tqdm(game_gen, desc='Processing games'):
        if writer.is_full:
            break
        if reader.time_control == -1:
            continue

        try:
            board = BughouseBoard(reader.time_control)
            moves = list(reader.moves)

            # This holds the state and moves for the "current" team's turn
            current_action = {
                "team": None,  # 0 or 1
                "planes": None,  # Snapshot of board before any team moves
                "moves": [None, None],  # [board_0_move, board_1_move]
                "ply_index": None,
            }

            for ply_index, (board_num, move, time_left, move_time) in enumerate(moves):
                board.update_time(board_num, time_left, move_time)
                side = board.boards[board_num].turn

                # Identify Team: Team 0 is (B0-White/B1-Black), Team 1 is (B0-Black/B1-White)
                # This works because partners always have opposite colors.
                moving_team = 0 if (board_num == 0 and side == chess.WHITE) or \
                                   (board_num == 1 and side == chess.BLACK) else 1

                # If the team changed, the previous team's opportunity to move is over. Flush it.
                if current_action["team"] is not None and moving_team != current_action["team"]:
                    save_team_action(writer, current_action, label_to_index, reader.result, len(moves))
                    current_action = {
                        "team": None,
                        "planes": None,
                        "moves": [None, None],
                        "ply_index": None,
                    }

                # If this is the start of a team turn, snapshot the planes
                if current_action["planes"] is None:
                    # 'perspective_side' represents the perspective for board2planes
                    perspective_side = chess.WHITE if moving_team == 0 else chess.BLACK
                    current_action["planes"] = (board2planes(board, perspective_side), board2planes(board, not perspective_side))
                    current_action["team"] = moving_team
                    current_action["ply_index"] = ply_index

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
                save_team_action(writer, current_action, label_to_index, reader.result, len(moves))

        except Exception as e:
            print(f'Error processing game: {e}')

    writer.write_shard()
    return writer.written_paths


def save_team_action(writer, action, label_to_index, game_result, total_plys):
    """Utility to format and write the buffered team moves."""
    # Convert moves to labels, defaulting to 'pass' if a board didn't move
    m0 = action["moves"][0] if action["moves"][0] else 'pass'
    m1 = action["moves"][1] if action["moves"][1] else 'pass'

    assert m0 != "pass" or m1 != "pass", "Both boards didn't move!"

    policy_idx = (label_to_index[m0], label_to_index[m1])

    # Calculate value: If Team 0 moved, they represent the "Board 0 White" perspective
    # If Team 1 moved, they represent the "Board 0 Black" perspective
    value = game_result if action["team"] == 0 else -game_result
    plys_to_end = total_plys - action["ply_index"]

    has_time_advantage = action["planes"][0][31, 0, 0] > 0.5  # Check if time advantage plane is 1.0
    offset = NUM_BUGHOUSE_CHANNELS_PER_BOARD

    # Check if boards are on turn (channels 25 and 57)
    board_a_on_turn = action["planes"][0][25, 0, 0] > 0.5  # Board A turn plane
    board_b_on_turn = action["planes"][0][offset + 25, 0, 0] > 0.5

    # Skip sample if team is down on time and passes on a board that's on turn
    if not has_time_advantage:
        if (m0 == 'pass' and board_a_on_turn) or (m1 == 'pass' and board_b_on_turn):
            return  # Don't add this sample

    writer.add_sample(action["planes"][0], policy_idx, value, plys_to_end)

    # Create full pass move for other team to teach network how to sit and use time
    # From the other's team perspective it could be that both board are not on turn so both have to pass
    # Or the case we care more about: only one board is on turn and both boards still pass
    if 'pass' in [m0, m1]:
        # For the other team's sample, check their time advantage (same channel since it's duplicated)
        other_has_time_advantage = action["planes"][1][31, 0, 0] > 0.5

        # For other team's perspective, the turn channels are different
        other_board_a_on_turn = action["planes"][1][25, 0, 0] > 0.5
        other_board_b_on_turn = action["planes"][1][offset + 25, 0, 0] > 0.5

        # Skip other team's sample if they're down on time and would pass on a board that's on turn
        if not other_has_time_advantage and (other_board_a_on_turn or other_board_b_on_turn):
            return  # Don't add the other team's sample either
        # Skip if both boards are on turn since it doesn't make any sense to double sit even if up time
        if other_board_a_on_turn and other_board_b_on_turn:
            return

        writer.add_sample(
            action["planes"][1],
            (label_to_index['pass'], label_to_index['pass']),
            -value,
            plys_to_end,
        )

if __name__ == '__main__':
    generate_planes()
