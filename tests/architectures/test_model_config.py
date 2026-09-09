from types import SimpleNamespace

import pytest

from hivemind.architectures.model_config import ModelArgs, create_model, model_args_from_state_dict


def tensor(*shape):
    return SimpleNamespace(shape=shape)


def test_checkpoint_options_include_joint_and_auxiliary_heads():
    args = model_args_from_state_dict({
        "cross_board_blocks.0.weight": tensor(192, 192),
        "position_embedding": tensor(1, 64, 192),
        "value_head.body_wdl.0.weight": tensor(3, 1024),
        "value_head.body_plys.0.weight": tensor(1, 1024),
        "value_head.body_final.0.weight": tensor(8, 4),
        "joint_policy_heads.0.pass_factors.weight": tensor(4, 192),
    })
    assert args.model_type == "crossboard-risev33"
    assert args.attention_layers == 1
    assert args.use_wdl and args.use_plys_to_end and args.use_mlp_wdl_ply
    assert args.joint_policy_rank == 4


def test_checkpoint_detects_wdl_independently_from_moves_left():
    args = model_args_from_state_dict({"value_head.body_wdl.0.weight": tensor(3, 1024)})
    assert args.use_wdl
    assert not args.use_plys_to_end


def test_unknown_architecture_is_actionable():
    with pytest.raises(ValueError, match="Unknown architecture"):
        create_model(ModelArgs(model_type="unknown"))


def test_joint_policy_inference_preserves_moves_left_prediction():
    import torch
    from hivemind.inference.checkpoint import perform_inference

    class Model:
        def __call__(self, planes):
            return (
                torch.tensor([[0.25]]),
                (torch.zeros(1, 3), torch.zeros(1, 3)),
                torch.zeros(1, 4),
                torch.zeros(1, 3),
                torch.tensor([[0.5]]),
                torch.zeros(1, 12),
                torch.zeros(1, 12),
            )

    value, policy_a, policy_b, moves_left = perform_inference(
        Model(), torch.zeros(74, 8, 8), torch.device("cpu"),
    )
    assert value == 0.25
    assert moves_left == 50.0
    assert policy_a.sum() == pytest.approx(1.0)
    assert policy_b.sum() == pytest.approx(1.0)
