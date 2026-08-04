import numpy as np

from src.constants import NUM_BUGHOUSE_CHANNELS
from src.preprocessing.generate_planes import ShardWriter


def test_supervised_board_swap_doubling_in_writer(tmp_path):
    writer = ShardWriter(
        output_dir=str(tmp_path),
        samples_per_shard=1024,
        augment_board_swap=True,
    )

    offset = NUM_BUGHOUSE_CHANNELS // 2

    x = np.zeros((NUM_BUGHOUSE_CHANNELS, 8, 8), dtype=np.float32)
    x[0, 0, 0] = 1.0
    x[offset, 7, 7] = 1.0
    x[12, 0, 0] = 2.0 / 16.0
    x[offset + 12, 0, 0] = 3.0 / 16.0

    policy_idx = (11, 29)
    writer.add_sample(x, policy_idx, value=1.0, plys_to_end=14)

    assert len(writer.buffer) == 2

    first = writer.buffer[0]
    second = writer.buffer[1]

    x_first = np.frombuffer(first["x"], dtype=np.uint8).reshape(NUM_BUGHOUSE_CHANNELS, 8, 8)
    x_second = np.frombuffer(second["x"], dtype=np.uint8).reshape(NUM_BUGHOUSE_CHANNELS, 8, 8)

    # Board blocks are swapped in the doubled sample.
    assert np.array_equal(x_second[:offset], x_first[offset:])
    assert np.array_equal(x_second[offset:], x_first[:offset])

    # Policy targets are swapped in the doubled sample.
    assert first["y_policy_idx"] == (11, 29)
    assert second["y_policy_idx"] == (29, 11)


def test_supervised_board_swap_can_be_disabled(tmp_path):
    writer = ShardWriter(
        output_dir=str(tmp_path),
        samples_per_shard=1024,
        augment_board_swap=False,
    )

    x = np.zeros((NUM_BUGHOUSE_CHANNELS, 8, 8), dtype=np.float32)
    writer.add_sample(x, (1, 2), value=0.0, plys_to_end=3)

    assert len(writer.buffer) == 1
