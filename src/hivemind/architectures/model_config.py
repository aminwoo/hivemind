"""Shared architecture configuration for training and checkpoint inference."""

from dataclasses import dataclass

from hivemind.constants import NUM_BUGHOUSE_CHANNELS
from hivemind.architectures.rise_mobile_v3 import (
    get_cross_board_rise_v33_model,
    get_dual_stream_memory_rise_v33_model,
    get_rise_v33_model,
)

MODEL_FACTORIES = {
    "risev33": get_rise_v33_model,
    "crossboard-risev33": get_cross_board_rise_v33_model,
    "dualstream-memory-risev33": get_dual_stream_memory_rise_v33_model,
}


@dataclass
class ModelArgs:
    model_type: str = "risev33"
    input_version: str = "1.0"
    export_dir: str = "../../checkpoints"
    device_id: int = 0
    context: str = "gpu"
    input_shape: tuple[int, int, int] = (NUM_BUGHOUSE_CHANNELS, 8, 8)
    n_labels: int = 0
    channels_policy_head: int = 73
    select_policy_from_plane: bool = True
    use_wdl: bool = False
    use_plys_to_end: bool = False
    use_mlp_wdl_ply: bool = False
    shared_policy_trunk: bool = False
    joint_policy_rank: int = 0
    attention_dim: int = 192
    attention_heads: int = 6
    attention_layers: int = 2
    memory_tokens: int = 8


def get_model_args(train_config=None, architecture="risev33") -> ModelArgs:
    args = ModelArgs(model_type=architecture)
    if train_config is not None:
        args.use_wdl = bool(train_config.use_wdl)
        args.use_plys_to_end = bool(train_config.use_plys_to_end)
        args.use_mlp_wdl_ply = bool(train_config.use_mlp_wdl_ply)
        args.shared_policy_trunk = args.use_wdl
        args.joint_policy_rank = train_config.joint_policy_rank
    return args


def create_model(args):
    try:
        factory = MODEL_FACTORIES[args.model_type]
    except KeyError:
        raise ValueError(f"Unknown architecture: {args.model_type}") from None
    return factory(args)


def model_args_from_state_dict(state_dict) -> ModelArgs:
    """Recover supported architecture options from checkpoint parameter shapes.

    Attention head count is not encoded in tensor shapes; released models use six.
    """
    args = ModelArgs()
    if any(key.startswith("memory_exchanges.") for key in state_dict):
        args.model_type = "dualstream-memory-risev33"
        args.memory_tokens = state_dict["initial_memory"].shape[1]
    elif any(key.startswith("cross_board_blocks.") for key in state_dict):
        args.model_type = "crossboard-risev33"
        args.attention_layers = len({
            key.split(".")[1] for key in state_dict
            if key.startswith("cross_board_blocks.")
        })
    if args.model_type != "risev33":
        args.attention_dim = state_dict["position_embedding"].shape[-1]
    args.use_wdl = any(key.startswith("value_head.body_wdl.") for key in state_dict)
    args.use_plys_to_end = any(key.startswith("value_head.body_plys.") for key in state_dict)
    args.shared_policy_trunk = any(key.startswith("policy_heads.shared_body.") for key in state_dict)
    final_weight = state_dict.get("value_head.body_final.0.weight")
    args.use_mlp_wdl_ply = (
        args.use_wdl and args.use_plys_to_end
        and final_weight is not None and final_weight.shape[1] == 4
    )
    joint_weight = state_dict.get("joint_policy_heads.0.pass_factors.weight")
    if joint_weight is not None:
        args.joint_policy_rank = joint_weight.shape[0]
    return args
