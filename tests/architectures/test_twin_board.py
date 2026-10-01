import pytest
import torch

from hivemind.architectures.twin_board import TWIN_BOARD_SIZES, get_twin_board_model
from hivemind.constants import NUM_BUGHOUSE_CHANNELS_PER_BOARD


def _swap(x):
    a, b = x.split(NUM_BUGHOUSE_CHANNELS_PER_BOARD, dim=1)
    return torch.cat((b, a), dim=1)


def test_outputs_match_the_engine_contract():
    model = get_twin_board_model("twin-s").eval()
    value, (pi_a, pi_b), auxiliary, wdl, plys = model(torch.rand(3, 74, 8, 8))
    assert value.shape == (3, 1) and wdl.shape == (3, 3) and plys.shape == (3, 1)
    assert pi_a.shape == pi_b.shape == (3, 4672)
    assert auxiliary.shape == (3, 4)


VARIANTS = [
    {},
    {"mid_attention": True},
    {"value_diff": True},
    {"mid_attention": True, "value_diff": True},
    {"attention_style": "lite", "se_every": 0},
    {"block": "dense", "attention_style": "lite", "se_every": 0},
    {"block": "dense", "attention_style": "lite", "se_every": 0, "mid_attention": True, "value_diff": True},
    {"block": "dense", "channels": 96, "blocks": 6, "attention_dim": 0, "se_every": 0},
]


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: ",".join(f"{k}={x}" for k, x in v.items()) or "default")
def test_swapping_boards_swaps_policies_and_keeps_the_value(variant):
    torch.manual_seed(0)
    model = get_twin_board_model("twin-s", **variant).eval()
    # Exchanges start as the identity; perturb them so the test sees them work.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
    x = torch.rand(4, 74, 8, 8)
    with torch.no_grad():
        value, (pi_a, pi_b), _, wdl, plys = model(x)
        value_s, (pi_a_s, pi_b_s), _, wdl_s, plys_s = model(_swap(x))
    torch.testing.assert_close(pi_a_s, pi_b, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(pi_b_s, pi_a, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(value_s, value, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(wdl_s, wdl, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(plys_s, plys, atol=1e-5, rtol=1e-4)


def test_all_sizes_build():
    for size in TWIN_BOARD_SIZES:
        get_twin_board_model(size)


def test_variants_keep_the_engine_contract():
    model = get_twin_board_model("twin-s", mid_attention=True, value_diff=True).eval()
    value, (pi_a, pi_b), auxiliary, wdl, plys = model(torch.rand(2, 74, 8, 8))
    assert value.shape == (2, 1) and pi_a.shape == pi_b.shape == (2, 4672) and auxiliary.shape == (2, 4)
    assert model.mid_attention_after == 3  # after the 4th of twin-s's 8 blocks
