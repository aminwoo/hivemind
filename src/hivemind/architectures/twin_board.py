"""Twin-board network: one shared trunk per bughouse board.

The RISEv3 family stacks both boards' planes as channels of a single 8x8
image, so every convolution mixes board A's squares with board B's, although
squares on different boards have no spatial relationship. Here each board's
37 planes go through the same trunk as their own image, and the boards talk
through explicit exchange steps instead.

The input encoding makes this exact rather than approximate: each board's
block of planes is written from the team's point of view (its own pieces,
pockets, turn and castling first), and swapping the boards is a symmetry of
the game. With shared weights and symmetric exchanges, swapping the two input
blocks swaps pi_a and pi_b and leaves value, WDL and moves-left unchanged, so
the network cannot spend capacity on the two boards separately and every
training position teaches both boards.

Inputs and outputs match the RISEv3 models, so the engine and the ONNX export
treat it as a drop-in replacement.
"""
import torch
from torch import nn
from torch.nn import BatchNorm2d, Conv2d, Module, Sequential

from hivemind.architectures.builder_util import (
    _BottlekneckResidualBlock,
    _PolicyHead,
    _Stem,
    _ValueHead,
)
from hivemind.constants import NUM_BUGHOUSE_CHANNELS_PER_BOARD


def _swap_boards(x, batch_size):
    """Board A rows are x[:batch], board B rows x[batch:]; exchange them."""
    return torch.cat((x[batch_size:], x[:batch_size]), dim=0)


class _BoardExchange(Module):
    """Per-channel scale and shift for each board from both boards' summaries.

    A board's update depends on (its own pooled features, the other board's
    pooled features) through the same weights for both boards, so the step is
    swap-equivariant. Zero-initialised output makes it start as the identity.
    """

    def __init__(self, channels, hidden):
        super().__init__()
        self.message = Sequential(
            nn.Linear(2 * channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * channels),
        )
        nn.init.zeros_(self.message[-1].weight)
        nn.init.zeros_(self.message[-1].bias)

    def forward(self, x, batch_size):
        pooled = x.mean(dim=(2, 3))
        scale, shift = self.message(
            torch.cat((pooled, _swap_boards(pooled, batch_size)), dim=1)
        ).chunk(2, dim=1)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class _BoardAttention(Module):
    """Every square of one board attends to every square of the other."""

    def __init__(self, channels, embedding_dim, heads, mlp_ratio=2):
        super().__init__()
        self.to_tokens = Sequential(
            Conv2d(channels, embedding_dim, kernel_size=1, bias=False),
            BatchNorm2d(embedding_dim),
            nn.ReLU(),
        )
        self.position = nn.Parameter(torch.zeros(1, 64, embedding_dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.query_norm = nn.LayerNorm(embedding_dim)
        self.memory_norm = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(embedding_dim, heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(embedding_dim)
        self.mlp = Sequential(
            nn.Linear(embedding_dim, embedding_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embedding_dim * mlp_ratio, embedding_dim),
        )
        self.to_channels = Sequential(
            Conv2d(embedding_dim, channels, kernel_size=1, bias=False),
            BatchNorm2d(channels),
        )
        nn.init.zeros_(self.to_channels[1].weight)

    def forward(self, x, batch_size):
        tokens = self.to_tokens(x).flatten(2).transpose(1, 2) + self.position
        other = _swap_boards(tokens, batch_size)
        memory = self.memory_norm(other)
        attended, _ = self.attention(self.query_norm(tokens), memory, memory, need_weights=False)
        tokens = tokens + attended
        tokens = tokens + self.mlp(self.mlp_norm(tokens))
        update = tokens.transpose(1, 2).reshape(x.shape[0], -1, 8, 8)
        return x + self.to_channels(update)


class _BoardAttentionLite(Module):
    """Cross-board attention without LayerNorms or an MLP.

    Same symmetric square-to-square exchange as _BoardAttention, in far fewer
    (and more fusable) kernels: at batch 8 each kernel costs a few
    microseconds whatever its size, and the full block is about a fifth of
    twin-s's inference time.
    """

    def __init__(self, channels, embedding_dim, heads):
        super().__init__()
        self.to_tokens = Sequential(
            Conv2d(channels, embedding_dim, kernel_size=1, bias=False),
            BatchNorm2d(embedding_dim),
        )
        self.position = nn.Parameter(torch.zeros(1, 64, embedding_dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.attention = nn.MultiheadAttention(embedding_dim, heads, batch_first=True)
        self.to_channels = Sequential(
            Conv2d(embedding_dim, channels, kernel_size=1, bias=False),
            BatchNorm2d(channels),
        )
        nn.init.zeros_(self.to_channels[1].weight)

    def forward(self, x, batch_size):
        tokens = self.to_tokens(x).flatten(2).transpose(1, 2) + self.position
        other = _swap_boards(tokens, batch_size)
        attended, _ = self.attention(tokens, other, other, need_weights=False)
        update = attended.transpose(1, 2).reshape(x.shape[0], -1, 8, 8)
        return x + self.to_channels(update)


class _DenseResidualBlock(Module):
    """Two dense 3x3 convolutions and a residual (two fused kernels in TensorRT,
    against three for the depthwise bottleneck)."""

    def __init__(self, channels, kernel=3):
        super().__init__()
        self.body = Sequential(
            Conv2d(channels, channels, kernel_size=kernel, padding=kernel // 2, bias=False),
            BatchNorm2d(channels),
            nn.ReLU(),
            Conv2d(channels, channels, kernel_size=kernel, padding=kernel // 2, bias=False),
            BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.body(x))


class TwinBoardNet(Module):
    """Shared per-board trunk with cross-board exchanges and symmetric heads.

    :param channels: trunk width per board
    :param channels_operating: expanded width inside each bottleneck block
    :param kernels: kernel size of each residual block (its length is the depth)
    :param exchange_every: a board exchange follows every this many blocks
    :param attention_dim: width of the final cross-board attention (0 = none)
    :param mid_attention: also run a cross-board attention halfway through the
        trunk, so later blocks see specific squares of the other board rather
        than only its pooled summaries
    :param value_diff: feed the value head |A - B| next to A + B, so "one
        board critical, one safe" differs from "both slightly worse" (both
        terms are swap-invariant)
    :param block: "bottleneck" (depthwise inverted residual) or "dense" (two
        dense convolutions; channels_operating is unused)
    :param attention_style: "full" (LayerNorms and MLP) or "lite"
    """

    def __init__(self, channels=192, channels_operating=384, kernels=(3,) * 10,
                 exchange_every=3, exchange_hidden=128, attention_dim=128,
                 attention_heads=4, channels_policy_head=73, channels_value_head=16,
                 value_fc_size=256, se_every=4, mid_attention=False, value_diff=False,
                 block="bottleneck", attention_style="full"):
        super().__init__()
        planes = NUM_BUGHOUSE_CHANNELS_PER_BOARD
        self.stem = _Stem(channels=channels, nb_input_channels=planes)
        if block == "bottleneck":
            self.blocks = nn.ModuleList([
                _BottlekneckResidualBlock(
                    channels=channels, channels_operating=channels_operating,
                    use_depthwise_conv=True, kernel=kernel,
                    se_type="eca_se" if se_every and index % se_every == se_every - 1 else None)
                for index, kernel in enumerate(kernels)
            ])
        elif block == "dense":
            self.blocks = nn.ModuleList([_DenseResidualBlock(channels, kernel) for kernel in kernels])
        else:
            raise ValueError(f"unknown block type {block!r}")
        attention_type = {"full": _BoardAttention, "lite": _BoardAttentionLite}[attention_style]
        self.exchange_after = {
            index for index in range(len(kernels))
            if exchange_every and index % exchange_every == exchange_every - 1
        }
        self.exchanges = nn.ModuleDict({
            str(index): _BoardExchange(channels, exchange_hidden) for index in self.exchange_after
        })
        self.attention = (attention_type(channels, attention_dim, attention_heads)
                          if attention_dim else None)
        # After the block that ends the first half of the trunk.
        self.mid_attention_after = len(kernels) // 2 - 1 if mid_attention else None
        self.mid_attention = (attention_type(channels, attention_dim or 128, attention_heads)
                              if mid_attention else None)
        self.value_diff = value_diff
        self.policy_head = _PolicyHead(8, 8, channels, channels_policy_head, 0, "relu", True)
        self.value_head = _ValueHead(
            8, 8, 2 * channels if value_diff else channels, channels_value_head, value_fc_size,
            "relu", False, planes,
            use_wdl=True, use_plys_to_end=True)
        # The exporter reads these.
        self.joint_policy_rank = 0

    def forward(self, x):
        batch_size = x.shape[0]
        board_a, board_b = x.split(NUM_BUGHOUSE_CHANNELS_PER_BOARD, dim=1)
        out = self.stem(torch.cat((board_a, board_b), dim=0))
        for index, block in enumerate(self.blocks):
            out = block(out)
            if index in self.exchange_after:
                out = self.exchanges[str(index)](out, batch_size)
            if index == self.mid_attention_after:
                out = self.mid_attention(out, batch_size)
        if self.attention is not None:
            out = self.attention(out, batch_size)

        policy = self.policy_head(out)
        policy_a, policy_b = policy[:batch_size], policy[batch_size:]
        # Summing the boards' maps (and their absolute difference) is
        # invariant to swapping them.
        board_a_features, board_b_features = out[:batch_size], out[batch_size:]
        value_input = board_a_features + board_b_features
        if self.value_diff:
            value_input = torch.cat((value_input, (board_a_features - board_b_features).abs()), dim=1)
        value, wdl, plys = self.value_head(value_input)
        auxiliary = torch.cat((wdl, plys), dim=1)
        return value, (policy_a, policy_b), auxiliary, wdl, plys


TWIN_BOARD_SIZES = {
    # name: (channels, channels_operating, blocks, attention_dim)
    "twin-s": (128, 256, 8, 96),
    "twin-m": (192, 384, 10, 128),
    "twin-l": (256, 512, 12, 160),
}


def get_twin_board_model(size="twin-m", mid_attention=False, value_diff=False,
                         block="bottleneck", attention_style="full", se_every=4,
                         channels=None, blocks=None, attention_dim=None):
    """A named size, optionally overriding its shape (channels, blocks,
    attention_dim) and design options. Dense blocks use 3x3 kernels only."""
    base_channels, operating, base_blocks, base_attention = TWIN_BOARD_SIZES[size]
    channels = channels or base_channels
    blocks = blocks or base_blocks
    attention = base_attention if attention_dim is None else attention_dim
    kernels = [3] * blocks
    if block == "bottleneck":
        for index in range(blocks // 2, blocks, 3):
            kernels[index] = 5
    return TwinBoardNet(channels=channels, channels_operating=operating * channels // base_channels,
                        kernels=kernels, attention_dim=attention,
                        attention_heads=max(1, attention // 32) if attention else 1,
                        mid_attention=mid_attention, value_diff=value_diff, se_every=se_every,
                        block=block, attention_style=attention_style)
