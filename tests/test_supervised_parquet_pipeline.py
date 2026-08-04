from types import SimpleNamespace

import polars as pl
import torch

from src.architectures.rise_mobile_v3 import get_rise_v33_model
from src.constants import NUM_BUGHOUSE_CHANNELS
from src.training.data_loaders import (
    NB_INPUT_VALUES,
    NB_POLICY_VALUES,
    load_parquet_shard,
)


def test_supervised_parquet_pipeline_with_aux_heads(tmp_path):
    x_bytes = bytes([0] * NB_INPUT_VALUES)

    shard_path = tmp_path / "sample.parquet"
    pl.DataFrame(
        {
            "x": [x_bytes],
            "y_policy_idx": [(0, 0)],
            "y_value": [1.0],
            "y_plys_to_end": [12],
        }
    ).write_parquet(shard_path)

    x, y_value, y_policy, wdl, plys = load_parquet_shard(
        str(shard_path), include_auxiliary=True
    )

    assert x.shape == (1, NUM_BUGHOUSE_CHANNELS, 8, 8)
    assert y_policy.shape == (1, 2)
    assert wdl.tolist() == [2]
    assert plys.shape == (1,)

    args = SimpleNamespace(
        input_shape=(NUM_BUGHOUSE_CHANNELS, 8, 8),
        channels_policy_head=73,
        select_policy_from_plane=True,
        n_labels=0,
        use_wdl=True,
        use_plys_to_end=True,
        use_mlp_wdl_ply=False,
        shared_policy_trunk=True,
    )

    model = get_rise_v33_model(args)
    model.eval()

    with torch.no_grad():
        value_out, policy_out, aux_out, wdl_out, plys_out = model(x)

    assert value_out.shape == (1, 1)
    assert policy_out[0].shape == (1, NB_POLICY_VALUES)
    assert policy_out[1].shape == (1, NB_POLICY_VALUES)
    assert aux_out.shape == (1, 4)
    assert wdl_out.shape == (1, 3)
    assert plys_out.shape == (1, 1)

    ce_loss = torch.nn.CrossEntropyLoss()(wdl_out, wdl)
    mse_loss = torch.nn.MSELoss()(torch.flatten(plys_out), plys)
    assert torch.isfinite(ce_loss)
    assert torch.isfinite(mse_loss)
