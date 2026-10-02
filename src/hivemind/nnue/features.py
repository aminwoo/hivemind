"""NNUE feature layout; mirrors engine/src/nnue/features.h."""

import numpy as np

PIECE_TYPES = 6
HAND_SLOTS = 30

PIECE_OFFSET = 0
PIECE_FEATURES = 2 * 2 * PIECE_TYPES * 64
HAND_OFFSET = PIECE_OFFSET + PIECE_FEATURES
HAND_FEATURES = 2 * 2 * HAND_SLOTS
PROMOTED_OFFSET = HAND_OFFSET + HAND_FEATURES
PROMOTED_FEATURES = 2 * 2 * 64
ON_TURN_OFFSET = PROMOTED_OFFSET + PROMOTED_FEATURES
TIME_ADV_OFFSET = ON_TURN_OFFSET + 4
CASTLE_OFFSET = TIME_ADV_OFFSET + 2
NUM_FEATURES = CASTLE_OFFSET + 8


def mirror_feature(f: int) -> int:
    """The same feature seen by the other team."""
    if f < HAND_OFFSET:
        square, piece, board_rel = f % 64, (f // 64) % PIECE_TYPES, f // (64 * PIECE_TYPES)
        return ((board_rel ^ 1) * PIECE_TYPES + piece) * 64 + (square ^ 56)
    if f < PROMOTED_OFFSET:
        local = f - HAND_OFFSET
        return HAND_OFFSET + ((local // HAND_SLOTS) ^ 1) * HAND_SLOTS + local % HAND_SLOTS
    if f < ON_TURN_OFFSET:
        local = f - PROMOTED_OFFSET
        return PROMOTED_OFFSET + ((local // 64) ^ 1) * 64 + ((local % 64) ^ 56)
    if f < TIME_ADV_OFFSET:
        return ON_TURN_OFFSET + ((f - ON_TURN_OFFSET) ^ 1)
    if f < CASTLE_OFFSET:
        return TIME_ADV_OFFSET + ((f - TIME_ADV_OFFSET) ^ 1)
    local = f - CASTLE_OFFSET
    return CASTLE_OFFSET + ((local // 2) ^ 1) * 2 + local % 2


def swap_boards_feature(f: int) -> int:
    """The same feature after swapping boards A and B (see features.h)."""
    if f < HAND_OFFSET:
        stride = 2 * PIECE_TYPES * 64
        return f + stride if f < stride else f - stride
    if f < PROMOTED_OFFSET:
        local = f - HAND_OFFSET
        return HAND_OFFSET + (local + 2 * HAND_SLOTS if local < 2 * HAND_SLOTS else local - 2 * HAND_SLOTS)
    if f < ON_TURN_OFFSET:
        return PROMOTED_OFFSET + ((f - PROMOTED_OFFSET) ^ 128)
    if f < TIME_ADV_OFFSET:
        return ON_TURN_OFFSET + ((f - ON_TURN_OFFSET) ^ 2)
    if f < CASTLE_OFFSET:
        return f
    return CASTLE_OFFSET + ((f - CASTLE_OFFSET) ^ 4)


SWAP_TABLE = np.array([swap_boards_feature(f) for f in range(NUM_FEATURES)], dtype=np.int64)
MIRROR_TABLE = np.array([mirror_feature(f) for f in range(NUM_FEATURES)], dtype=np.int64)


# King buckets: piece features of each board are split by where the
# perspective's own king stands on that board (squares oriented so the team's
# pieces start at the bottom). Hand, promotion and flag features are shared.
PIECE_FEATURES_PER_BOARD = 2 * PIECE_TYPES * 64
KING_PIECE = 5


def king_bucket_table(scheme: str) -> np.ndarray:
    table = np.zeros(64, dtype=np.int64)
    for square in range(64):
        rank, file = divmod(square, 8)
        if scheme == "none":
            table[square] = 0
        elif scheme == "k4":
            # queenside, centre, kingside at home; anything advanced
            table[square] = 3 if rank >= 2 else (0 if file <= 2 else (2 if file >= 5 else 1))
        elif scheme == "k8":
            home = 0 if file <= 2 else (2 if file >= 5 else 1)
            table[square] = home if rank == 0 else (3 + home if rank == 1 else (6 if rank <= 3 else 7))
        else:
            raise ValueError(f"unknown king bucket scheme {scheme}")
    return table


def bucketed_feature_count(buckets: int) -> int:
    return 2 * buckets * PIECE_FEATURES_PER_BOARD + (NUM_FEATURES - PIECE_FEATURES)
