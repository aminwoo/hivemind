"""Student network and its engine export format.

The feature transformer maps sparse features to an accumulator of width H.
Both teams share it: the side to move's accumulator and the other team's
(built from mirrored features) are concatenated, passed through SCReLU
(clamp to [0, 1], then square), and fed to an output head. The head is either
a single linear layer or a small MLP. The output is a logit whose tanh
estimates the teacher's value.

The engine stores the transformer as int16 scaled by FT_SCALE and runs the head
in float, so only the transformer weights need a quantization-friendly range.
"""

import struct

import numpy as np
import torch
from torch import nn

from hivemind.nnue.features import (KING_PIECE, NUM_FEATURES, PIECE_FEATURES, PIECE_FEATURES_PER_BOARD,
                                    bucketed_feature_count, king_bucket_table)

FT_SCALE = 255
FT_WEIGHT_LIMIT = 127 / 64  # keeps any realistic accumulator inside int16
EXPORT_MAGIC = b"HMNNUE02"


def screlu(x):
    return x.clamp(0.0, 1.0).square()


class NnueModel(nn.Module):
    def __init__(self, hidden=512, l1=16, l2=32, king_buckets="none"):
        super().__init__()
        self.hidden, self.l1, self.l2 = hidden, l1, l2
        self.king_buckets = king_buckets
        self.register_buffer("bucket_table", torch.from_numpy(king_bucket_table(king_buckets)), persistent=False)
        self.num_buckets = int(self.bucket_table.max()) + 1
        self.num_features = bucketed_feature_count(self.num_buckets) if king_buckets != "none" else NUM_FEATURES
        self.ft_weight = nn.Parameter(torch.randn(self.num_features, hidden) * 0.02)
        self.ft_bias = nn.Parameter(torch.zeros(hidden))
        if l1 > 0:
            self.fc1 = nn.Linear(2 * hidden, l1)
            self.fc2 = nn.Linear(l1, l2)
            self.out = nn.Linear(l2, 1)
        else:
            self.out = nn.Linear(2 * hidden, 1)

    def accumulate(self, flat, rows, batch_size):
        # A dense one-hot product beats embedding_bag here: with only ~2k
        # features its backward pass is a plain matmul instead of millions of
        # atomic adds into the same few rows.
        if self.king_buckets != "none":
            flat = self.bucket_features(flat, rows, batch_size)
        onehot = torch.zeros(batch_size, self.num_features, device=flat.device)
        onehot[rows, flat] = 1.0
        return onehot @ self.ft_weight + self.ft_bias

    def bucket_features(self, flat, rows, batch_size):
        """Re-index each board's piece features by the own king's bucket there."""
        kings = torch.zeros(batch_size, 2, dtype=torch.long, device=flat.device)
        for board in (0, 1):
            base = board * PIECE_FEATURES_PER_BOARD + KING_PIECE * 64
            mask = (flat >= base) & (flat < base + 64)
            kings[rows[mask], board] = flat[mask] - base
        bucket = self.bucket_table[kings]
        board = (flat // PIECE_FEATURES_PER_BOARD).clamp(max=1)
        piece = (board * self.num_buckets + bucket[rows, board]) * PIECE_FEATURES_PER_BOARD \
            + flat % PIECE_FEATURES_PER_BOARD
        other = 2 * self.num_buckets * PIECE_FEATURES_PER_BOARD + (flat - PIECE_FEATURES)
        return torch.where(flat < PIECE_FEATURES, piece, other)

    def forward(self, stm_flat, nstm_flat, rows, batch_size):
        x = torch.cat([screlu(self.accumulate(stm_flat, rows, batch_size)),
                       screlu(self.accumulate(nstm_flat, rows, batch_size))], dim=1)
        if self.l1 > 0:
            x = screlu(self.fc1(x))
            x = screlu(self.fc2(x))
        return self.out(x).squeeze(1)

    @torch.no_grad()
    def clip_weights(self):
        self.ft_weight.clamp_(-FT_WEIGHT_LIMIT, FT_WEIGHT_LIMIT)
        self.ft_bias.clamp_(-FT_WEIGHT_LIMIT, FT_WEIGHT_LIMIT)

    def config(self):
        return {"hidden": self.hidden, "l1": self.l1, "l2": self.l2, "king_buckets": self.king_buckets}


def _quantize_ft(tensor):
    return np.round(tensor.detach().cpu().numpy() * FT_SCALE).astype(np.int16)


def _floats(tensor):
    return tensor.detach().cpu().numpy().astype("<f4").tobytes()


def export_network(model: NnueModel, path):
    """Writes the engine format (little endian).

    magic "HMNNUE02", u32 features, hidden, l1, l2, ft_scale, king buckets;
    u8 king bucket table[64];
    i16 ft_weight[features][hidden]; i16 ft_bias[hidden];
    then f32 head: fc1 w[l1][2H], b[l1], fc2 w[l2][l1], b[l2], out w[l2], b[1]
    (or out w[2H], b[1] when l1 == 0).
    """
    with open(path, "wb") as stream:
        stream.write(EXPORT_MAGIC)
        stream.write(struct.pack("<6I", model.num_features, model.hidden, model.l1, model.l2, FT_SCALE,
                                 model.num_buckets))
        stream.write(model.bucket_table.cpu().numpy().astype(np.uint8).tobytes())
        stream.write(_quantize_ft(model.ft_weight).astype("<i2").tobytes())
        stream.write(_quantize_ft(model.ft_bias).astype("<i2").tobytes())
        layers = [model.fc1, model.fc2, model.out] if model.l1 > 0 else [model.out]
        for layer in layers:
            stream.write(_floats(layer.weight))
            stream.write(_floats(layer.bias))


def quantized_copy(model: NnueModel) -> NnueModel:
    """The model as the engine sees it: transformer rounded to the int16 grid."""
    copy = NnueModel(**model.config()).to(model.ft_weight.device)
    copy.load_state_dict(model.state_dict())
    with torch.no_grad():
        copy.ft_weight.copy_(torch.round(model.ft_weight * FT_SCALE) / FT_SCALE)
        copy.ft_bias.copy_(torch.round(model.ft_bias * FT_SCALE) / FT_SCALE)
    return copy
