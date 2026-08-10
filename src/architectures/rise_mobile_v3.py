"""
Upgrade of RISE architecture using mixed depthwise convolutions, preactivation residuals and dropout
 proposed by Johannes Czech:

Influenced by the following papers:
    * MixConv: Mixed Depthwise Convolutional Kernels, Mingxing Tan, Quoc V. Le, https://arxiv.org/abs/1907.09595
    * ProxylessNas: Direct Neural Architecture Search on Target Task and Hardware, Han Cai, Ligeng Zhu, Song Han.
     https://arxiv.org/abs/1812.00332
    * MnasNet: Platform-Aware Neural Architecture Search for Mobile,
     Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew Howard, Quoc V. Le
     http://openaccess.thecvf.com/content_CVPR_2019/html/Tan_MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile_CVPR_2019_paper.html
    * FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search,
    Bichen Wu, Xiaoliang Dai, Peizhao Zhang, Yanghan Wang, Fei Sun, Yiming Wu, Yuandong Tian, Peter Vajda, Yangqing Jia, Kurt Keutzer,
    http://openaccess.thecvf.com/content_CVPR_2019/html/Wu_FBNet_Hardware-Aware_Efficient_ConvNet_Design_via_Differentiable_Neural_Architecture_Search_CVPR_2019_paper.html
    * MobileNetV3: Searching for MobileNetV3,
    Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam.
    https://arxiv.org/abs/1905.02244
    * Convolutional Block Attention Module (CBAM),
    Sanghyun Woo, Jongchan Park, Joon-Young Lee, In So Kweon
    https://arxiv.org/pdf/1807.06521.pdf

"""
import logging

import torch
from torch import nn
from torch.nn import Sequential, Conv2d, BatchNorm2d, Module
from timm.layers import DropPath

from src.architectures.builder_util import get_act, _ValueHead, _PolicyHead,\
    _Stem, get_se, process_value_policy_head, _BottlekneckResidualBlock, ClassicalResidualBlock, round_to_next_multiple_of_32
from src.architectures.next_vit_official_modules import NCB
from configs.train_config import TrainConfig
from src.architectures.next_vit_official_modules import NTB
from src.constants import NUM_BUGHOUSE_CHANNELS_PER_BOARD


class _SharedPolicyHeads(Module):
    def __init__(self, channels, policy_channels, act_type):
        super().__init__()
        self.nb_flatten = policy_channels * 8 * 8
        self.shared_body = Sequential(
            Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(channels),
            get_act(act_type),
        )
        self.board_projections = nn.ModuleList([
            Conv2d(channels, policy_channels, kernel_size=3, padding=1, bias=False),
            Conv2d(channels, policy_channels, kernel_size=3, padding=1, bias=False),
        ])

    def forward(self, x):
        shared = self.shared_body(x)
        return tuple(
            projection(shared).view(-1, self.nb_flatten)
            for projection in self.board_projections
        )


def _get_res_blocks(act_types, channels, channels_operating_init, channel_expansion, kernels, se_types, use_transformers, path_dropout_rates, conv_block, kernel_5_channel_ratio, round_channels_to_next_32):
    """Helper function which generates the residual blocks for Risev3"""

    channels_operating = channels_operating_init
    res_blocks = []

    for idx, kernel in enumerate(kernels):
        if kernel == 5:
            if kernel_5_channel_ratio is None:
                channels_operating_active = channels_operating - 32 * (idx // 2)
            else:
                channels_operating_active = int(channels_operating * kernel_5_channel_ratio + 0.5) # 0.68 95 #- 32 * (idx // 2)
        else:
            channels_operating_active = channels_operating

        if round_channels_to_next_32:
            channels_operating_active = round_to_next_multiple_of_32(channels_operating_active)
            channels = round_to_next_multiple_of_32(channels)

        if use_transformers[idx]:
            res_blocks.append(NTB(channels, channels, path_dropout=path_dropout_rates[idx]))
        elif conv_block == "mobile_bottlekneck_res_block":
            res_blocks.append(_BottlekneckResidualBlock(channels=channels,
                                                        channels_operating=channels_operating_active,
                                                        use_depthwise_conv=True,
                                                        kernel=kernel, act_type=act_types[idx],
                                                        se_type=se_types[idx],
                                                        path_dropout=path_dropout_rates[idx]))
        elif conv_block == "bottlekneck_res_block":
            res_blocks.append(_BottlekneckResidualBlock(channels=channels,
                                                        channels_operating=channels_operating_active,
                                                        use_depthwise_conv=False,
                                                        kernel=kernel, act_type=act_types[idx],
                                                        se_type=se_types[idx],
                                                        path_dropout=path_dropout_rates[idx]))
        elif conv_block == "classical_res_block":
            res_blocks.append(ClassicalResidualBlock(channels, act_types[idx], se_type=se_types[idx], path_dropout=path_dropout_rates[idx]))
        elif conv_block == "next_conv_block":
            res_blocks.append(NCB(channels, channels, stride=1, se_type=se_types[idx], path_dropout=path_dropout_rates[idx]))

        channels_operating += channel_expansion

    return res_blocks


class RiseV3(Module):

    def __init__(self, nb_input_channels, board_height, board_width,
                 channels=256, channels_operating_init=224, channel_expansion=32, act_types=None,
                 channels_value_head=8, channels_policy_head=81, value_fc_size=256, dropout_rate=0.15,
                 select_policy_from_plane=True, kernels=None, n_labels=4992,
                 se_types=None, use_avg_features=False, use_wdl=False, use_plys_to_end=False,
                 use_mlp_wdl_ply=False,
                 use_transformers=None, path_dropout=0, conv_block="mobile_bottlekneck_res_block",
                 kernel_5_channel_ratio=None, round_channels_to_next_32=False,
                 shared_policy_trunk=False,
                 ):
        """
        RISEv3 architecture
        :param channels: Main number of channels
        :param channels_operating: Initial number of channels at the start of the net for the depthwise convolution
        :param channel_expansion: Number of channels to add after each residual block
        :param act_types: Activation type to use as a list of layers.
        :param channels_value_head: Number of channels for the value head
        :param value_fc_size: Number of units in the fully connected layer of the value head
        :param channels_policy_head: Number of channels for the policy head
        :param dropout_rate: Dropout factor to use. If 0, no dropout will be applied. Value must be in [0,1]
        :param select_policy_from_plane: True, if policy head type shall be used
        :param kernels: List of kernel sizes used for the residual blocks. The length of the list corresponds to the number
        of residual blocks.
        :param n_labels: Number of policy target labels (used for select_policy_from_plane=False)
        :param se_types: List of squeeze excitation modules to use for each residual layer.
         The length of this list must be the same as len(kernels). Available types:
        - "se": Squeeze excitation block - Hu et al. - https://arxiv.org/abs/1709.01507
        - "cbam": Convolutional Block Attention Module (CBAM) - Woo et al. - https://arxiv.org/pdf/1807.06521.pdf
        - "ca_se": Same as "se"
        - "cm_se": Squeeze excitation with max operator
        - "sa_se": Spatial excitation with average operator
        - "sm_se": Spatial excitation with max operator
         the spatial dimensionality and the number of channels will be doubled.
        Later the spatial and scalar embeddings will be merged again.
        :param use_wdl: If a win draw loss head shall be used
        :param use_plys_to_end: If a plys to end prediction head shall be used
        :param use_mlp_wdl_ply: If a small mlp with value output for the wdl and ply head shall be used
        :param path_dropout: Path dropout for stochastic depth
        :param conv_block: Base convolutional block ["mobile_bottlekneck_res_block", "bottlekneck_res_block", "classical_res_block", "next_conv_block"]
        :param kernel_5_channel_ratio: Downscale factor for channels_operating in case of 5x5 kernels
        :param round_channels_to_next_32: Rounds all number of channels within the network to the closest multiple of 32
        :return: symbol
        """
        super(RiseV3, self).__init__()
        self.nb_input_channels = nb_input_channels
        self.use_plys_to_end = use_plys_to_end
        self.use_wdl = use_wdl
        self.shared_policy_trunk = shared_policy_trunk

        if round_channels_to_next_32:
            channels = round_to_next_multiple_of_32(channels)
        self.channels = channels

        if se_types is None:
            se_types = [None] * len(kernels)
        if use_transformers is None:
            use_transformers = [None] * len(kernels)
        if act_types is None:
            act_types = ['relu'] * len(kernels)

        if len(kernels) != len(se_types):
            raise Exception(f'The length of "kernels": {len(kernels)} must be the same as'
                            f' the length of "se_types": {len(se_types)}')

        valid_se_types = [None, "se", "cbam", "eca_se", "ca_se", "cm_se", "sa_se", "sm_se"]
        for se_type in se_types:
            if se_type not in valid_se_types:
                raise Exception(f"Unavailable se_type: {se_type}. Available se_types include {se_types}")

        path_dropout_rates = [x.item() for x in torch.linspace(0, path_dropout, len(kernels))]  # stochastic depth decay rule
        self.res_blocks = _get_res_blocks(act_types, channels, channels_operating_init, channel_expansion, kernels, se_types, use_transformers, path_dropout_rates, conv_block, kernel_5_channel_ratio, round_channels_to_next_32)

        self.body_spatial = Sequential(
            _Stem(channels=channels, act_type=act_types[0], nb_input_channels=nb_input_channels),
            *self.res_blocks,
        )
        self.nb_body_spatial_out = channels * board_height * board_width

        # create the three heads which will be used in the hybrid fwd pass
        self.value_head = _ValueHead(board_height, board_width, channels, channels_value_head, value_fc_size,
                                     act_types[-1], False, nb_input_channels,
                                     use_wdl, use_plys_to_end, use_mlp_wdl_ply)

        if shared_policy_trunk:
            if not select_policy_from_plane or board_height != 8 or board_width != 8:
                raise ValueError(
                    "The shared policy trunk requires 8x8 plane policy output"
                )
            self.policy_heads = _SharedPolicyHeads(
                channels, channels_policy_head, act_types[-1]
            )
        else:
            self.policy_heads = nn.ModuleList([
                _PolicyHead(board_height, board_width, channels, channels_policy_head, n_labels,
                            act_types[-1], select_policy_from_plane),
                _PolicyHead(board_height, board_width, channels, channels_policy_head, n_labels,
                            act_types[-1], select_policy_from_plane)
            ])

    def forward(self, x):
        """
        Implementation of the forward pass of the full network
        Uses a broadcast add operation for the shortcut and the output of the residual block
        :param x: Input to the ResidualBlock
        :return: Value & Policy Output
        """
        out = self.body_spatial(x)

        if self.shared_policy_trunk:
            value_head_out = self.value_head(out)
            policy_out = self.policy_heads(out)
            if self.use_plys_to_end and self.use_wdl:
                value_out, wdl_out, plys_to_end_out = value_head_out
                auxiliary_out = torch.cat((wdl_out, plys_to_end_out), dim=1)
                return (
                    value_out,
                    policy_out,
                    auxiliary_out,
                    wdl_out,
                    plys_to_end_out,
                )
            return value_head_out, policy_out

        return process_value_policy_head(out, self.value_head, self.policy_heads, self.use_plys_to_end, self.use_wdl)

    def merge_bn(self):
        """
        Calls the merge_bn() function for the NTB blocks
        """
        for res_block in self.res_blocks:
            if isinstance(res_block, NTB):
                res_block.merge_bn()
                logging.info("Called merge_bn()")


class _CrossBoardAttentionBlock(Module):
    def __init__(self, embedding_dim, num_heads, mlp_ratio=2):
        super().__init__()
        self.query_norm = nn.LayerNorm(embedding_dim)
        self.memory_norm = nn.LayerNorm(embedding_dim)
        self.cross_attention = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(embedding_dim)
        hidden_dim = embedding_dim * mlp_ratio
        self.mlp = Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def _update(self, board_tokens, other_board_tokens, context_tokens):
        memory = torch.cat((other_board_tokens, context_tokens), dim=1)
        attended, _ = self.cross_attention(
            self.query_norm(board_tokens),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        board_tokens = board_tokens + attended
        return board_tokens + self.mlp(self.mlp_norm(board_tokens))

    def forward(self, board_a_tokens, board_b_tokens, context_tokens):
        source_a = board_a_tokens
        source_b = board_b_tokens
        return (
            self._update(source_a, source_b, context_tokens),
            self._update(source_b, source_a, context_tokens),
        )


class CrossBoardRiseV3(RiseV3):
    """RISEv3 with explicit board, pocket, turn, and clock coordination tokens."""

    def __init__(self, *args, attention_dim=192, attention_heads=6,
                 attention_layers=2, **kwargs):
        if kwargs.get("shared_policy_trunk", False):
            raise ValueError(
                "CrossBoardRiseV3 uses separate post-attention policy heads"
            )
        super().__init__(*args, shared_policy_trunk=False, **kwargs)

        if self.nb_input_channels != 2 * NUM_BUGHOUSE_CHANNELS_PER_BOARD:
            raise ValueError(
                "CrossBoardRiseV3 requires two 37-plane bughouse boards"
            )
        if attention_layers not in (1, 2):
            raise ValueError("CrossBoardRiseV3 supports one or two attention layers")

        channels = self.channels
        board_channels = NUM_BUGHOUSE_CHANNELS_PER_BOARD
        self.board_token_projections = nn.ModuleList([
            Sequential(
                Conv2d(channels + board_channels, attention_dim, kernel_size=1,
                       bias=False),
                BatchNorm2d(attention_dim),
                nn.ReLU(),
            )
            for _ in range(2)
        ])
        self.position_embedding = nn.Parameter(
            torch.zeros(1, 64, attention_dim)
        )
        self.board_embedding = nn.Parameter(torch.zeros(2, 1, attention_dim))

        self.pocket_projection = nn.Linear(5, attention_dim)
        self.scalar_projection = nn.Linear(1, attention_dim)
        self.context_role_embedding = nn.Parameter(
            torch.zeros(1, 7, attention_dim)
        )
        self.cross_board_blocks = nn.ModuleList([
            _CrossBoardAttentionBlock(attention_dim, attention_heads)
            for _ in range(attention_layers)
        ])

        self.value_fusion = Sequential(
            Conv2d(2 * attention_dim, channels, kernel_size=1, bias=False),
            BatchNorm2d(channels),
            nn.ReLU(),
        )
        policy_channels = kwargs["channels_policy_head"]
        board_height = kwargs["board_height"]
        board_width = kwargs["board_width"]
        n_labels = kwargs["n_labels"]
        act_type = kwargs["act_types"][-1]
        select_policy_from_plane = kwargs["select_policy_from_plane"]
        self.policy_heads = nn.ModuleList([
            _PolicyHead(
                board_height,
                board_width,
                attention_dim,
                policy_channels,
                n_labels,
                act_type,
                select_policy_from_plane,
            )
            for _ in range(2)
        ])

        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.board_embedding, std=0.02)
        nn.init.trunc_normal_(self.context_role_embedding, std=0.02)

    def _context_tokens(self, x):
        offset = NUM_BUGHOUSE_CHANNELS_PER_BOARD
        pocket_values = (
            x[:, 12:17].mean(dim=(2, 3)),
            x[:, 17:22].mean(dim=(2, 3)),
            x[:, offset + 12:offset + 17].mean(dim=(2, 3)),
            x[:, offset + 17:offset + 22].mean(dim=(2, 3)),
        )
        pocket_tokens = [self.pocket_projection(values) for values in pocket_values]
        scalar_values = (
            x[:, 25:26].mean(dim=(2, 3)),
            x[:, offset + 25:offset + 26].mean(dim=(2, 3)),
            x[:, 31:32].mean(dim=(2, 3)),
        )
        scalar_tokens = [self.scalar_projection(values) for values in scalar_values]
        return torch.stack((*pocket_tokens, *scalar_tokens), dim=1) \
            + self.context_role_embedding

    def forward(self, x):
        shared_spatial = self.body_spatial(x)
        board_inputs = x.split(NUM_BUGHOUSE_CHANNELS_PER_BOARD, dim=1)
        board_maps = [
            projection(torch.cat((shared_spatial, board_input), dim=1))
            for projection, board_input in zip(
                self.board_token_projections, board_inputs
            )
        ]
        board_tokens = [
            board_map.flatten(2).transpose(1, 2)
            + self.position_embedding
            + self.board_embedding[index]
            for index, board_map in enumerate(board_maps)
        ]
        context_tokens = self._context_tokens(x)
        board_a_tokens, board_b_tokens = board_tokens
        for block in self.cross_board_blocks:
            board_a_tokens, board_b_tokens = block(
                board_a_tokens,
                board_b_tokens,
                context_tokens,
            )

        batch_size = x.shape[0]
        board_a_map = board_a_tokens.transpose(1, 2).reshape(
            batch_size, -1, 8, 8
        )
        board_b_map = board_b_tokens.transpose(1, 2).reshape(
            batch_size, -1, 8, 8
        )
        value_input = self.value_fusion(
            torch.cat((board_a_map, board_b_map), dim=1)
        )
        return process_value_policy_head(
            value_input,
            self.value_head,
            self.policy_heads,
            self.use_plys_to_end,
            self.use_wdl,
            policy_inputs=(board_a_map, board_b_map),
        )


class _AttentionResidual(Module):
    def __init__(self, embedding_dim, num_heads, mlp_ratio=2):
        super().__init__()
        self.query_norm = nn.LayerNorm(embedding_dim)
        self.source_norm = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(embedding_dim)
        hidden_dim = embedding_dim * mlp_ratio
        self.mlp = Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, query, source):
        normalized_source = self.source_norm(source)
        attended, _ = self.attention(
            self.query_norm(query),
            normalized_source,
            normalized_source,
            need_weights=False,
        )
        updated = query + attended
        return attended + self.mlp(self.mlp_norm(updated))


class _StateDependentResidualGate(Module):
    def __init__(self, embedding_dim):
        super().__init__()
        hidden_dim = max(embedding_dim // 2, 16)
        self.body = Sequential(
            nn.LayerNorm(3 * embedding_dim),
            nn.Linear(3 * embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.constant_(self.body[-1].bias, -1.38629436112)

    def forward(self, query, source, context_tokens):
        state = torch.cat(
            (
                query.mean(dim=1),
                source.mean(dim=1),
                context_tokens.mean(dim=1),
            ),
            dim=1,
        )
        return torch.sigmoid(self.body(state)).unsqueeze(1)


class _PersistentMemoryExchange(Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.memory_write = _AttentionResidual(embedding_dim, num_heads)
        self.memory_self_attention = _AttentionResidual(
            embedding_dim, num_heads
        )
        self.memory_read = _AttentionResidual(embedding_dim, num_heads)
        self.read_gate = _StateDependentResidualGate(embedding_dim)

    def forward(
        self,
        board_a_tokens,
        board_b_tokens,
        memory_tokens,
        context_tokens,
    ):
        write_source = torch.cat(
            (board_a_tokens, board_b_tokens, context_tokens), dim=1
        )
        memory_tokens = memory_tokens + self.memory_write(
            memory_tokens, write_source
        )
        memory_tokens = memory_tokens + self.memory_self_attention(
            memory_tokens, memory_tokens
        )

        paired_boards = torch.cat((board_a_tokens, board_b_tokens), dim=0)
        paired_memory = torch.cat((memory_tokens, memory_tokens), dim=0)
        paired_context = torch.cat((context_tokens, context_tokens), dim=0)
        paired_delta = self.memory_read(paired_boards, paired_memory)
        paired_gate = self.read_gate(
            paired_boards, paired_memory, paired_context
        )
        board_a_delta, board_b_delta = paired_delta.chunk(2, dim=0)
        board_a_gate, board_b_gate = paired_gate.chunk(2, dim=0)
        return (
            board_a_delta * board_a_gate,
            board_b_delta * board_b_gate,
            memory_tokens,
        )


class _GatedDirectCrossBoardAttention(Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.direct_attention = _AttentionResidual(embedding_dim, num_heads)
        self.direct_gate = _StateDependentResidualGate(embedding_dim)

    def forward(self, board_a_tokens, board_b_tokens, context_tokens):
        paired_queries = torch.cat((board_a_tokens, board_b_tokens), dim=0)
        paired_sources = torch.cat((board_b_tokens, board_a_tokens), dim=0)
        paired_context = torch.cat((context_tokens, context_tokens), dim=0)
        paired_delta = self.direct_attention(paired_queries, paired_sources)
        paired_gate = self.direct_gate(
            paired_queries, paired_sources, paired_context
        )
        board_a_delta, board_b_delta = paired_delta.chunk(2, dim=0)
        board_a_gate, board_b_gate = paired_gate.chunk(2, dim=0)
        return board_a_delta * board_a_gate, board_b_delta * board_b_gate


class DualStreamMemoryRiseV3(Module):
    """Shared-weight board streams with persistent latent communication."""

    def __init__(
        self,
        nb_input_channels,
        board_height,
        board_width,
        channels=384,
        channels_operating_init=256,
        channel_expansion=64,
        act_types=None,
        channels_value_head=16,
        channels_policy_head=73,
        value_fc_size=512,
        select_policy_from_plane=True,
        kernels=None,
        n_labels=0,
        se_types=None,
        use_wdl=False,
        use_plys_to_end=False,
        use_mlp_wdl_ply=False,
        use_transformers=None,
        path_dropout=0,
        conv_block="mobile_bottlekneck_res_block",
        kernel_5_channel_ratio=None,
        round_channels_to_next_32=False,
        attention_dim=192,
        attention_heads=6,
        memory_tokens=8,
        stage_sizes=(5, 5, 5),
    ):
        super().__init__()
        if nb_input_channels != 2 * NUM_BUGHOUSE_CHANNELS_PER_BOARD:
            raise ValueError(
                "DualStreamMemoryRiseV3 requires two 37-plane bughouse boards"
            )
        if board_height * board_width != 64:
            raise ValueError("DualStreamMemoryRiseV3 requires 8x8 boards")
        if sum(stage_sizes) != len(kernels) or len(stage_sizes) != 3:
            raise ValueError(
                "stage_sizes must contain three stages covering all blocks"
            )
        if attention_dim % attention_heads != 0:
            raise ValueError("attention_dim must be divisible by attention_heads")
        if memory_tokens <= 0:
            raise ValueError("memory_tokens must be positive")

        if round_channels_to_next_32:
            channels = round_to_next_multiple_of_32(channels)
        if se_types is None:
            se_types = [None] * len(kernels)
        if use_transformers is None:
            use_transformers = [None] * len(kernels)
        if act_types is None:
            act_types = ['relu'] * len(kernels)

        self.nb_input_channels = nb_input_channels
        self.board_height = board_height
        self.board_width = board_width
        self.channels = channels
        self.attention_dim = attention_dim
        self.use_wdl = use_wdl
        self.use_plys_to_end = use_plys_to_end
        self.shared_policy_trunk = False

        path_dropout_rates = [
            value.item()
            for value in torch.linspace(0, path_dropout, len(kernels))
        ]
        residual_blocks = _get_res_blocks(
            act_types,
            channels,
            channels_operating_init,
            channel_expansion,
            kernels,
            se_types,
            use_transformers,
            path_dropout_rates,
            conv_block,
            kernel_5_channel_ratio,
            round_channels_to_next_32,
        )
        self.shared_stem = _Stem(
            channels=channels,
            act_type=act_types[0],
            nb_input_channels=NUM_BUGHOUSE_CHANNELS_PER_BOARD,
        )
        stage_boundaries = (
            0,
            stage_sizes[0],
            stage_sizes[0] + stage_sizes[1],
            sum(stage_sizes),
        )
        self.shared_stages = nn.ModuleList([
            Sequential(*residual_blocks[start:end])
            for start, end in zip(stage_boundaries, stage_boundaries[1:])
        ])

        self.position_embedding = nn.Parameter(
            torch.zeros(1, board_height * board_width, attention_dim)
        )
        self.board_embedding = nn.Parameter(torch.zeros(2, 1, attention_dim))
        self.initial_memory = nn.Parameter(
            torch.zeros(1, memory_tokens, attention_dim)
        )
        self.context_role_embedding = nn.Parameter(
            torch.zeros(1, 7, attention_dim)
        )
        self.pocket_projection = nn.Linear(5, attention_dim)
        self.scalar_projection = nn.Linear(1, attention_dim)

        self.exchange_input_projections = nn.ModuleList([
            Sequential(
                Conv2d(channels, attention_dim, kernel_size=1, bias=False),
                BatchNorm2d(attention_dim),
                nn.ReLU(),
            )
            for _ in range(2)
        ])
        self.exchange_output_projections = nn.ModuleList([
            Conv2d(attention_dim, channels, kernel_size=1, bias=False)
            for _ in range(2)
        ])
        self.memory_exchanges = nn.ModuleList([
            _PersistentMemoryExchange(attention_dim, attention_heads)
            for _ in range(2)
        ])
        self.direct_input_projection = Sequential(
            Conv2d(channels, attention_dim, kernel_size=1, bias=False),
            BatchNorm2d(attention_dim),
            nn.ReLU(),
        )
        self.direct_output_projection = Conv2d(
            attention_dim, channels, kernel_size=1, bias=False
        )
        self.direct_exchange = _GatedDirectCrossBoardAttention(
            attention_dim, attention_heads
        )

        self.value_fusion = Sequential(
            Conv2d(2 * channels, channels, kernel_size=1, bias=False),
            BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.value_head = _ValueHead(
            board_height,
            board_width,
            channels,
            channels_value_head,
            value_fc_size,
            act_types[-1],
            False,
            nb_input_channels,
            use_wdl,
            use_plys_to_end,
            use_mlp_wdl_ply,
        )
        self.policy_heads = nn.ModuleList([
            _PolicyHead(
                board_height,
                board_width,
                channels,
                channels_policy_head,
                n_labels,
                act_types[-1],
                select_policy_from_plane,
            )
            for _ in range(2)
        ])

        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.board_embedding, std=0.02)
        nn.init.trunc_normal_(self.initial_memory, std=0.02)
        nn.init.trunc_normal_(self.context_role_embedding, std=0.02)

    def _context_tokens(self, x):
        offset = NUM_BUGHOUSE_CHANNELS_PER_BOARD
        pocket_values = (
            x[:, 12:17].mean(dim=(2, 3)),
            x[:, 17:22].mean(dim=(2, 3)),
            x[:, offset + 12:offset + 17].mean(dim=(2, 3)),
            x[:, offset + 17:offset + 22].mean(dim=(2, 3)),
        )
        pocket_tokens = [
            self.pocket_projection(values) for values in pocket_values
        ]
        scalar_values = (
            x[:, 25:26].mean(dim=(2, 3)),
            x[:, offset + 25:offset + 26].mean(dim=(2, 3)),
            x[:, 31:32].mean(dim=(2, 3)),
        )
        scalar_tokens = [
            self.scalar_projection(values) for values in scalar_values
        ]
        return (
            torch.stack((*pocket_tokens, *scalar_tokens), dim=1)
            + self.context_role_embedding
        )

    def _board_tokens(self, paired_maps, projection, batch_size):
        projected = projection(paired_maps)
        board_a_map, board_b_map = projected.split(batch_size, dim=0)
        board_a_tokens = (
            board_a_map.flatten(2).transpose(1, 2)
            + self.position_embedding
            + self.board_embedding[0]
        )
        board_b_tokens = (
            board_b_map.flatten(2).transpose(1, 2)
            + self.position_embedding
            + self.board_embedding[1]
        )
        return board_a_tokens, board_b_tokens

    def _tokens_to_paired_maps(self, board_a_tokens, board_b_tokens):
        paired_tokens = torch.cat((board_a_tokens, board_b_tokens), dim=0)
        return paired_tokens.transpose(1, 2).reshape(
            -1,
            self.attention_dim,
            self.board_height,
            self.board_width,
        )

    def forward(self, x):
        batch_size = x.shape[0]
        board_a_input, board_b_input = x.split(
            NUM_BUGHOUSE_CHANNELS_PER_BOARD, dim=1
        )
        paired_maps = self.shared_stem(
            torch.cat((board_a_input, board_b_input), dim=0)
        )
        context_tokens = self._context_tokens(x)
        memory_tokens = self.initial_memory.expand(batch_size, -1, -1)

        for stage_index, memory_exchange in enumerate(self.memory_exchanges):
            paired_maps = self.shared_stages[stage_index](paired_maps)
            board_a_tokens, board_b_tokens = self._board_tokens(
                paired_maps,
                self.exchange_input_projections[stage_index],
                batch_size,
            )
            board_a_delta, board_b_delta, memory_tokens = memory_exchange(
                board_a_tokens,
                board_b_tokens,
                memory_tokens,
                context_tokens,
            )
            paired_delta = self._tokens_to_paired_maps(
                board_a_delta, board_b_delta
            )
            paired_maps = paired_maps + self.exchange_output_projections[
                stage_index
            ](paired_delta)

        paired_maps = self.shared_stages[-1](paired_maps)
        board_a_tokens, board_b_tokens = self._board_tokens(
            paired_maps, self.direct_input_projection, batch_size
        )
        board_a_delta, board_b_delta = self.direct_exchange(
            board_a_tokens, board_b_tokens, context_tokens
        )
        paired_maps = paired_maps + self.direct_output_projection(
            self._tokens_to_paired_maps(board_a_delta, board_b_delta)
        )
        board_a_map, board_b_map = paired_maps.split(batch_size, dim=0)

        value_input = self.value_fusion(
            torch.cat((board_a_map, board_b_map), dim=1)
        )
        return process_value_policy_head(
            value_input,
            self.value_head,
            self.policy_heads,
            self.use_plys_to_end,
            self.use_wdl,
            policy_inputs=(board_a_map, board_b_map),
        )

    def merge_bn(self):
        for stage in self.shared_stages:
            for residual_block in stage:
                if isinstance(residual_block, NTB):
                    residual_block.merge_bn()
                    logging.info("Called merge_bn()")


def get_rise_v33_model(args):
    """
    Wrapper definition for RISEv3.3.
    :return: pytorch model object
    """
    kernels = [3] * 15
    kernels[7] = 5
    kernels[11] = 5
    kernels[12] = 5
    kernels[13] = 5

    se_types = [None] * len(kernels)
    se_types[5] = "eca_se"
    se_types[8] = "eca_se"
    se_types[12] = "eca_se"
    se_types[13] = "eca_se"
    se_types[14] = "eca_se"

    act_types = ['relu'] * len(kernels)

    model = RiseV3(nb_input_channels=args.input_shape[0], board_height=args.input_shape[1], board_width=args.input_shape[2],
                   channels=384, channels_operating_init=256, channel_expansion=64, act_types=act_types,
                   channels_value_head=16, value_fc_size=512,
                   channels_policy_head=args.channels_policy_head,
                   dropout_rate=0, select_policy_from_plane=args.select_policy_from_plane,
                   kernels=kernels, se_types=se_types, use_avg_features=False, n_labels=args.n_labels,
                   use_wdl=args.use_wdl, use_plys_to_end=args.use_plys_to_end, use_mlp_wdl_ply=args.use_mlp_wdl_ply,
                   shared_policy_trunk=getattr(args, "shared_policy_trunk", False),
                   )
    return model


def get_cross_board_rise_v33_model(args):
    """Build RISEv3.3 with explicit post-convolution board coordination."""
    kernels = [3] * 15
    kernels[7] = 5
    kernels[11] = 5
    kernels[12] = 5
    kernels[13] = 5

    se_types = [None] * len(kernels)
    for index in (5, 8, 12, 13, 14):
        se_types[index] = "eca_se"

    act_types = ['relu'] * len(kernels)
    return CrossBoardRiseV3(
        nb_input_channels=args.input_shape[0],
        board_height=args.input_shape[1],
        board_width=args.input_shape[2],
        channels=384,
        channels_operating_init=256,
        channel_expansion=64,
        act_types=act_types,
        channels_value_head=16,
        value_fc_size=512,
        channels_policy_head=args.channels_policy_head,
        dropout_rate=0,
        select_policy_from_plane=args.select_policy_from_plane,
        kernels=kernels,
        se_types=se_types,
        use_avg_features=False,
        n_labels=args.n_labels,
        use_wdl=args.use_wdl,
        use_plys_to_end=args.use_plys_to_end,
        use_mlp_wdl_ply=args.use_mlp_wdl_ply,
        attention_dim=getattr(args, "attention_dim", 192),
        attention_heads=getattr(args, "attention_heads", 6),
        attention_layers=getattr(args, "attention_layers", 2),
    )


def get_dual_stream_memory_rise_v33_model(args):
    """Build the staged dual-stream RISEv3 with persistent latent memory."""
    kernels = [3] * 15
    for index in (7, 11, 12, 13):
        kernels[index] = 5

    se_types = [None] * len(kernels)
    for index in (5, 8, 12, 13, 14):
        se_types[index] = "eca_se"

    act_types = ['relu'] * len(kernels)
    return DualStreamMemoryRiseV3(
        nb_input_channels=args.input_shape[0],
        board_height=args.input_shape[1],
        board_width=args.input_shape[2],
        channels=384,
        channels_operating_init=256,
        channel_expansion=64,
        act_types=act_types,
        channels_value_head=16,
        value_fc_size=512,
        channels_policy_head=args.channels_policy_head,
        select_policy_from_plane=args.select_policy_from_plane,
        kernels=kernels,
        se_types=se_types,
        n_labels=args.n_labels,
        use_wdl=args.use_wdl,
        use_plys_to_end=args.use_plys_to_end,
        use_mlp_wdl_ply=args.use_mlp_wdl_ply,
        attention_dim=getattr(args, "attention_dim", 192),
        attention_heads=getattr(args, "attention_heads", 6),
        memory_tokens=getattr(args, "memory_tokens", 8),
    )


def get_rise_v33_large_model(args):
    """
    Wrapper definition for RISEv3.3 large.
    The model has 17,879,540 parameters, compared to 5,755,348 parameters for the standard RISEv3 model.
    :return: pytorch model object
    """
    kernels = [3] * 36
    kernels[7] = 5
    kernels[11] = 5
    kernels[12] = 5
    kernels[13] = 5

    kernels[7+15] = 5
    kernels[11+15] = 5
    kernels[12+15] = 5
    kernels[13+15] = 5

    se_types = [None] * len(kernels)
    se_types[5] = "eca_se"
    se_types[8] = "eca_se"
    se_types[12] = "eca_se"
    se_types[13] = "eca_se"
    se_types[14] = "eca_se"

    se_types[5+15] = "eca_se"
    se_types[8+15] = "eca_se"
    se_types[12+15] = "eca_se"
    se_types[13+15] = "eca_se"
    se_types[14+15] = "eca_se"

    act_types = ['relu'] * len(kernels)

    model = RiseV3(nb_input_channels=args.input_shape[0],
                   board_height=args.input_shape[1], board_width=args.input_shape[2],
                   channels=384, channels_operating_init=256,
                   channel_expansion=64, act_types=act_types, channels_value_head=16,
                   value_fc_size=512,
                   channels_policy_head=args.channels_policy_head, dropout_rate=0,
                   select_policy_from_plane=args.select_policy_from_plane,
                   kernels=kernels, se_types=se_types, use_avg_features=False,
                   n_labels=args.n_labels, use_wdl=args.use_wdl,
                   use_plys_to_end=args.use_plys_to_end,
                   use_mlp_wdl_ply=args.use_mlp_wdl_ply
                   )
    return model


def get_rise_v2_model(args):
    """
    Wrapper definition for RISEv2.0
    :return: pytorch model object
    """
    kernels = [3] * 13

    se_types = [None] * len(kernels)
    se_types[8] = "ca_se"
    se_types[9] = "ca_se"
    se_types[10] = "ca_se"
    se_types[11] = "ca_se"
    se_types[12] = "ca_se"

    act_types = ['relu'] * len(kernels)

    model = RiseV3(nb_input_channels=args.input_shape[0], board_height=args.input_shape[1], board_width=args.input_shape[2],
                   channels=256, channels_operating_init=128, channel_expansion=64, act_types=act_types,
                   channels_value_head=8, value_fc_size=256,
                   channels_policy_head=args.channels_policy_head,
                   dropout_rate=0, select_policy_from_plane=args.select_policy_from_plane,
                   kernels=kernels, se_types=se_types, use_avg_features=False, n_labels=args.n_labels,
                   use_wdl=args.use_wdl, use_plys_to_end=args.use_plys_to_end, use_mlp_wdl_ply=args.use_mlp_wdl_ply,
                   )
    return model


if __name__ == "__main__":

    class Args:
        def __init__(self):
            # Model type, e.g., "risev33" (specific architecture or version of the model)
            self.model_type = "risev33"

            # Input version, e.g., "1.0" (version of the input data or model configuration)
            self.input_version = "1.0"

            # Directory where the model checkpoints will be exported or saved
            self.export_dir = "../../checkpoints"

            # Device ID for running the model (e.g., GPU device ID)
            self.device_id = 0

            # Context in which the model will run, e.g., "gpu" or "cpu"
            self.context = "gpu"

            # Input shape of the model, represented as (channels, height, width)
            self.input_shape = (74, 8, 8)

            # Number of labels for the policy head (output layer for move predictions)
            # Example: 9600 possible moves or actions in the policy output
            self.n_labels = 0

            # Number of channels in the policy head (output layer for move predictions)
            # Example: 73 channels in the policy head
            self.channels_policy_head = 73

            # Whether to select the policy directly from the plane (spatial output)
            # If True, the policy is derived from the spatial dimensions of the output
            self.select_policy_from_plane = True

            # Whether to use a Win/Draw/Loss (WDL) head
            # If True, the model will predict the game outcome (win, draw, or loss)
            self.use_wdl = False

            # Whether to use a "plys to end" head
            # If True, the model will predict the number of plies (half-moves) remaining until the end of the game
            self.use_plys_to_end = False

            # Whether to use a Multi-Layer Perceptron (MLP) for the WDL and "plys to end" heads
            # If True, an MLP will be used to process these outputs instead of a simpler method
            self.use_mlp_wdl_ply = False


    args = Args()
    model = get_rise_v33_model(args)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Params: {total_params / 1e6:.2f}M")

    model.eval()

    device = torch.device(f"cuda:{args.device_id}" if args.context == "gpu" else "cpu")
    model.to(device)

    batch_size = 1
    dummy_input = torch.randn(batch_size, *args.input_shape).to(device)

    with torch.no_grad():
        value_out, policy_out = model(dummy_input)

    print(value_out.shape, policy_out[0].shape)

    import torch
    import time


    def benchmark_inference(model, input_shape, batch_size=64, repetitions=100):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()

        # Generate dummy Bughouse input data
        dummy_input = torch.randn(batch_size, *input_shape).to(device)

        # --- WARM-UP ---
        print("Warming up...")
        with torch.inference_mode():
            for _ in range(20):
                _ = model(dummy_input)

        # --- BENCHMARK ---
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        print(f"Benchmarking with batch size {batch_size}...")
        start_event.record()

        with torch.inference_mode():
            for _ in range(repetitions):
                _ = model(dummy_input)

        end_event.record()
        torch.cuda.synchronize()  # Wait for GPU to finish

        # Calculate results
        total_time_ms = start_event.elapsed_time(end_event)
        avg_latency_ms = total_time_ms / repetitions
        sps = (batch_size * repetitions) / (total_time_ms / 1000.0)

        print(f"Average Latency: {avg_latency_ms:.2f} ms")
        print(f"Throughput (SPS): {sps:.2f} positions/sec")


    # Run it
    benchmark_inference(model, input_shape=(74, 8, 8))