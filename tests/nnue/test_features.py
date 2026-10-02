from hivemind.nnue.features import MIRROR_TABLE, NUM_FEATURES, SWAP_TABLE


def test_mirror_and_swap_are_involutions():
    identity = list(range(NUM_FEATURES))
    assert list(MIRROR_TABLE[MIRROR_TABLE]) == identity
    assert list(SWAP_TABLE[SWAP_TABLE]) == identity


def test_mirror_and_swap_commute():
    assert list(MIRROR_TABLE[SWAP_TABLE]) == list(SWAP_TABLE[MIRROR_TABLE])


def test_king_buckets_reindex_piece_features_by_own_king():
    import torch

    from hivemind.nnue.features import HAND_OFFSET, PIECE_FEATURES_PER_BOARD
    from hivemind.nnue.model import NnueModel

    model = NnueModel(hidden=16, l1=0, l2=0, king_buckets="k4")
    # One position: own king on e1 of board A (bucket 1, centre), on g5 of
    # board B (bucket 3, advanced); a knight on each board; one hand feature.
    king_a = 5 * 64 + 4
    king_b = PIECE_FEATURES_PER_BOARD + 5 * 64 + 38
    knight_a = 1 * 64 + 10
    knight_b = PIECE_FEATURES_PER_BOARD + 1 * 64 + 10
    flat = torch.tensor([king_a, knight_a, king_b, knight_b, HAND_OFFSET])
    rows = torch.zeros(5, dtype=torch.long)
    out = model.bucket_features(flat, rows, 1).tolist()
    block = PIECE_FEATURES_PER_BOARD
    assert out[0] == (0 * 4 + 1) * block + king_a
    assert out[1] == (0 * 4 + 1) * block + knight_a
    assert out[2] == (1 * 4 + 3) * block + (king_b - block)
    assert out[3] == (1 * 4 + 3) * block + (knight_b - block)
    assert out[4] == 2 * 4 * block
    assert max(out) < model.num_features
