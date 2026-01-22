import chess
import numpy as np
import polars as pl

from src.domain.board import BughouseBoard
from src.constants import (BOARD_A, BOARD_B)
from src.utils.tcn import tcn_decode


class TrainingGameReader:
    def __init__(self, row):
        """
        :param row: A dictionary representing a row from the joined Polars dataframe.
        """
        # 1. Basic Metadata (Time control in deciseconds)
        try:
            self.time_control = int(row["time_control"]) * 10
        except (ValueError, KeyError, TypeError):
            self.time_control = -1
            return

        # 2. Result relative to Board 0
        self.result = self.get_game_result(row["winner"])

        # 3. Ratings (Board 0 and Partner Board 1)
        self.ratings = [
            [row["white_rating"], row["black_rating"]],
            [row["partner_white_rating"], row["partner_black_rating"]]
        ]

        # 4. Decode Moves and Timestamps for both boards
        moves = [
            tcn_decode(row["tcn"]),
            tcn_decode(row["partner_tcn"])
        ]

        timestamps = [
            self._parse_timestamps(row["timestamps"]),
            self._parse_timestamps(row["partner_timestamps"])
        ]

        # 5. Calculate Time Deltas
        deltas = [
            self.get_board_deltas(timestamps[0]),
            self.get_board_deltas(timestamps[1])
        ]

        # 6. Reconstruct Global Move Order (Interleaving)
        move_order = self.get_naive_move_order(deltas)

        # 7. Verify and Finalize Moves (Handling pockets/drops)
        self.moves = self.verify_move_order(move_order, moves, timestamps, deltas)

    def _parse_timestamps(self, ts_str):
        if not ts_str or ts_str == "":
            return []
        return [int(t) for t in str(ts_str).split(",")]

    def get_game_result(self, winner_str):
        w = str(winner_str).lower()
        if w == "white": return 1
        if w == "black": return -1
        return 0

    def get_board_deltas(self, board_times):
        if not board_times:
            return {chess.WHITE: [], chess.BLACK: []}

        white_times = board_times[::2]
        black_times = board_times[1::2]

        # Prepend time control to calculate first move delta
        white_times.insert(0, self.time_control)
        black_times.insert(0, self.time_control)

        white_deltas = [a - b for a, b in zip(white_times[:-1], white_times[1:])]
        black_deltas = [a - b for a, b in zip(black_times[:-1], black_times[1:])]

        return {chess.WHITE: white_deltas, chess.BLACK: black_deltas}

    @staticmethod
    def get_naive_move_order(deltas):
        def get_board_cum_times(board_deltas):
            w = np.array(board_deltas[chess.WHITE])
            b = np.array(board_deltas[chess.BLACK])
            combined = np.empty(w.size + b.size)
            combined[0::2] = w
            combined[1::2] = b
            return np.cumsum(combined)

        a_times = get_board_cum_times(deltas[BOARD_A])
        b_times = get_board_cum_times(deltas[BOARD_B])

        all_times = np.concatenate((a_times, b_times))
        all_indices = np.argsort(all_times)
        move_order = np.digitize(all_indices, [0, a_times.shape[0]]) - 1
        return move_order

    @staticmethod
    def verify_move_order(move_order, moves, times, deltas):
        board = BughouseBoard()
        new_moves = []
        q = list(range(len(move_order)))

        moves = [list(m) for m in moves]
        times = [list(t) for t in times]
        deltas = [{k: list(v) for k, v in d.items()} for d in deltas]

        while q:
            stuck_board = -1
            for i in q:
                board_num = move_order[i]
                if not moves[board_num]:
                    q.remove(i)
                    break

                move = moves[board_num][0]
                turn = board.boards[board_num].turn

                if board_num == stuck_board or (
                        move.drop and board.boards[board_num].pockets[turn].count(move.drop) <= 0
                ):
                    stuck_board = board_num
                    continue
                else:
                    time_left = times[board_num][0]
                    move_time = deltas[board_num][turn][0]

                    new_moves.append((board_num, move, time_left, move_time))
                    board.push(board_num, move)

                    moves[board_num].pop(0)
                    times[board_num].pop(0)
                    deltas[board_num][turn].pop(0)
                    q.remove(i)
                    break
        return new_moves


def process_parquet_file(path, min_rating=2200):
    df = pl.read_parquet(path)
    df = df.filter(
        (pl.col("white_rating") >= min_rating) &
        (pl.col("black_rating") >= min_rating)
    )

    # Self-join with consistent, long-form aliases
    joined_df = df.join(
        df.select([
            pl.col("game_id").alias("partner_game_id"),
            pl.col("tcn").alias("partner_tcn"),
            pl.col("timestamps").alias("partner_timestamps"),
            pl.col("white_rating").alias("partner_white_rating"),
            pl.col("black_rating").alias("partner_black_rating"),
        ]),
        on="partner_game_id",
        how="left"
    ).filter(
        (pl.col("partner_white_rating") >= min_rating) &
        (pl.col("partner_black_rating") >= min_rating)
    )

    game_id_list = joined_df.get_column("game_id").to_list()
    print(f"{len(game_id_list)} unique game ids", game_id_list[:10])
    for row in joined_df.to_dicts():
        yield TrainingGameReader(row)


if __name__ == "__main__":
    parquet_path = "../../data/games.parquet"
    game_gen = process_parquet_file(parquet_path)

    for reader in game_gen:
        result = reader.result
        white_rating = reader.ratings[0][0]

        print(f"Game Result: {result} | Primary White Rating: {white_rating}")

        for board_num, move, time_left, move_time in reader.moves:
            print(f"Board {board_num} Move: {move.uci()} | Clock: {time_left}")

        break