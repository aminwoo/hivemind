import pytest
import torch

from src.architectures.rise_mobile_v3 import (
    CrossBoardRiseV3,
    DualStreamMemoryRiseV3,
)


def _build_model(attention_layers=1, joint_policy_rank=0):
    return CrossBoardRiseV3(
        nb_input_channels=74,
        board_height=8,
        board_width=8,
        channels=64,
        channels_operating_init=64,
        channel_expansion=0,
        act_types=['relu'],
        channels_value_head=4,
        channels_policy_head=73,
        value_fc_size=32,
        select_policy_from_plane=True,
        kernels=[3],
        n_labels=0,
        se_types=[None],
        use_wdl=True,
        use_plys_to_end=True,
        use_mlp_wdl_ply=False,
        attention_dim=32,
        attention_heads=4,
        attention_layers=attention_layers,
        joint_policy_rank=joint_policy_rank,
    )


def _build_dual_stream_model():
    return DualStreamMemoryRiseV3(
        nb_input_channels=74,
        board_height=8,
        board_width=8,
        channels=32,
        channels_operating_init=32,
        channel_expansion=0,
        act_types=['relu'] * 3,
        channels_value_head=4,
        channels_policy_head=73,
        value_fc_size=32,
        select_policy_from_plane=True,
        kernels=[3] * 3,
        n_labels=0,
        se_types=[None] * 3,
        use_wdl=True,
        use_plys_to_end=True,
        use_mlp_wdl_ply=False,
        attention_dim=32,
        attention_heads=4,
        memory_tokens=8,
        stage_sizes=(1, 1, 1),
    )


def test_cross_board_model_preserves_five_output_contract():
    model = _build_model()
    inputs = torch.rand(2, 74, 8, 8, requires_grad=True)

    value, policies, auxiliary, wdl, moves_left = model(inputs)

    assert value.shape == (2, 1)
    assert policies[0].shape == policies[1].shape == (2, 4672)
    assert auxiliary.shape == (2, 4)
    assert wdl.shape == (2, 3)
    assert moves_left.shape == (2, 1)

    (value.mean() + policies[0].mean() + policies[1].mean()).backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_cross_board_joint_policy_head_adds_ranked_move_and_pass_factors():
    model = _build_model(joint_policy_rank=4)
    outputs = model(torch.rand(2, 74, 8, 8))

    assert len(outputs) == 7
    assert outputs[5].shape == outputs[6].shape == (2, 4 * 4673)
    assert torch.isfinite(outputs[5]).all()


@pytest.mark.parametrize(
    ("channel", "context_index"),
    [
        (12, 0),
        (17, 1),
        (37 + 12, 2),
        (37 + 17, 3),
        (25, 4),
        (37 + 25, 5),
        (31, 6),
    ],
)
def test_context_tokens_use_explicit_bughouse_planes(channel, context_index):
    model = _build_model()
    inputs = torch.zeros(1, 74, 8, 8)
    baseline = model._context_tokens(inputs)

    inputs[:, channel] = 1.0
    changed = model._context_tokens(inputs)
    token_change = (changed - baseline).abs().sum(dim=2).squeeze(0)

    assert token_change[context_index] > 0
    unchanged_indices = [index for index in range(7) if index != context_index]
    assert torch.equal(
        token_change[unchanged_indices],
        torch.zeros(len(unchanged_indices)),
    )


def test_cross_board_model_accepts_one_or_two_attention_layers():
    assert len(_build_model(attention_layers=1).cross_board_blocks) == 1
    assert len(_build_model(attention_layers=2).cross_board_blocks) == 2

    with pytest.raises(ValueError, match="one or two attention layers"):
        _build_model(attention_layers=3)


def test_dual_stream_memory_model_preserves_output_and_gradient_contract():
    model = _build_dual_stream_model()
    inputs = torch.rand(2, 74, 8, 8, requires_grad=True)

    value, policies, auxiliary, wdl, moves_left = model(inputs)

    assert value.shape == (2, 1)
    assert policies[0].shape == policies[1].shape == (2, 4672)
    assert auxiliary.shape == (2, 4)
    assert wdl.shape == (2, 3)
    assert moves_left.shape == (2, 1)

    (value.mean() + policies[0].mean() + policies[1].mean()).backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert inputs.grad[:, :37].abs().sum() > 0
    assert inputs.grad[:, 37:].abs().sum() > 0


def test_dual_stream_memory_persists_between_exchange_stages():
    model = _build_dual_stream_model()
    first_memory = []
    second_memory = []

    def capture_first_output(_module, _inputs, output):
        first_memory.append(output[2])

    def capture_second_input(_module, inputs):
        second_memory.append(inputs[2])

    first_hook = model.memory_exchanges[0].register_forward_hook(
        capture_first_output
    )
    second_hook = model.memory_exchanges[1].register_forward_pre_hook(
        capture_second_input
    )
    model(torch.rand(1, 74, 8, 8))
    first_hook.remove()
    second_hook.remove()

    assert first_memory[0] is second_memory[0]
