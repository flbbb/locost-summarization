# coding=utf-8
# Copyright 2022 Google LLC., LongT5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LongT5 model."""


import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import LongT5Config
from transformers.activations import ACT2FN
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


# TODO: Update before the merge
LONGT5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/long-t5-local-base",
    "google/long-t5-local-large",
    "google/long-t5-tglobal-base",
    "google/long-t5-tglobal-large",
]


def _pad_to_multiple(
    x: torch.Tensor, block_len: int, dim: int, pad_value: int = 0
) -> torch.Tensor:
    """Pad a tensor so that a sequence length will be a multiple of `block_len`"""
    pad_len = -x.shape[dim] % block_len
    # Handle cases when an empty input sequence is given
    if not all(x.shape):
        new_shape = list(x.shape)
        new_shape[dim] += pad_len
        return torch.zeros(new_shape, dtype=x.dtype)

    pad = [(0, 0)] * x.ndim
    pad[dim] = (0, pad_len)
    pad = sum(pad[::-1], ())
    x = nn.functional.pad(x, pad=pad, mode="constant", value=pad_value)
    return x


def _split_into_blocks(x: torch.Tensor, block_len: int, dim: int) -> torch.Tensor:
    """Split an input tensor into blocks of a given `block_len` along the given `dim`. If the dimension length
    is not a multiple of `block_len`, it will be padded first with selected `pad_value`.
    """
    # pad tensor to multiple of block_len
    if x.shape[dim] % block_len != 0:
        x = _pad_to_multiple(x, block_len, dim, pad_value=0)
    num_blocks = x.shape[dim] // block_len
    output_shape = x.shape[:dim] + (num_blocks, block_len) + x.shape[(dim + 1) :]
    # If 0 is in output_shape, we cannot apply reshape because of incompatibility with ONNX conversion
    if 0 in output_shape:
        return torch.empty(output_shape, dtype=x.dtype, device=x.device)
    return x.reshape(output_shape)


def _concatenate_3_blocks(
    x: torch.Tensor, block_dim: int, sequence_dim: int, pad_value: int = 0
) -> torch.Tensor:
    """Concatenate three consecutive blocks for each input block for local attentiont.

    For more information, see: https://arxiv.org/pdf/2112.07916.pdf.
    """
    num_blocks = x.shape[block_dim]

    pad = [(0, 0)] * x.ndim
    pad[block_dim] = (1, 1)
    pad = sum(pad[::-1], ())
    # [batch_size, num_blocks, block_len] -> [batch_size, num_blocks + 2, block_len]
    x = nn.functional.pad(x, pad=pad, mode="constant", value=pad_value)

    blocks_list: List[torch.Tensor] = []
    for i in range(3):
        # We use indexing approach here:
        # https://numpy.org/doc/stable/user/basics.indexing.html#dealing-with-variable-numbers-of-indices-within-programs
        indices = [slice(0, None)] * x.ndim
        indices[block_dim] = slice(i, i + num_blocks)
        indices = tuple(indices)
        blocks_list.append(x[indices])
    # [batch_size, num_blocks, 3 * block_len, ...]
    return torch.cat(blocks_list, dim=sequence_dim)


def _make_3block_relative_position_ids(block_len: int) -> torch.Tensor:
    """Makes 3-blocked relative position ids for local attention."""
    position_ids = torch.arange(3 * block_len, dtype=torch.int32)
    center_position_ids = position_ids[block_len:-block_len]
    # [block_len, 3 * block_len]
    relative_position_ids = position_ids.unsqueeze(0) - center_position_ids.unsqueeze(1)
    return relative_position_ids


def _mask_local_attention_mask(
    local_attention_mask: torch.Tensor, block_len: int
) -> torch.Tensor:
    """Mask local attention mask to enforce that tokens are not allowed to attend tokens farther than ``local_radius."""
    relative_position_ids = _make_3block_relative_position_ids(block_len)
    locality_mask = torch.abs(relative_position_ids) < block_len
    locality_mask = locality_mask[None, None, :, :]
    locality_mask = locality_mask.to(local_attention_mask.device)
    return torch.logical_and(local_attention_mask, locality_mask)


def _get_local_attention_mask(
    attention_mask: torch.Tensor, block_len: int, device: torch.device
) -> torch.Tensor:
    """Prepare attention mask to be applied for a local attention."""
    # [batch_size, num_blocks, block_len]
    _blocked_attention_mask = _split_into_blocks(attention_mask, block_len, dim=1)
    # [batch_size, num_block, 3 * block_len]
    _3blocked_attention_mask = _concatenate_3_blocks(
        _blocked_attention_mask, block_dim=1, sequence_dim=2
    )

    _blocked_attention_mask = _blocked_attention_mask.unsqueeze(-1)
    _3blocked_attention_mask = _3blocked_attention_mask.unsqueeze(-2)
    # [batch_size, num_block, block_len, 3 * block_len]
    local_attention_mask = torch.logical_and(
        _blocked_attention_mask, _3blocked_attention_mask
    )
    local_attention_mask = _mask_local_attention_mask(local_attention_mask, block_len)
    # [batch_size, 1, num_block, block_len, 3 * block_len]
    return local_attention_mask.unsqueeze(1).to(device)


def _make_global_fixed_block_ids(
    attention_mask: torch.Tensor, global_block_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Obtain the "fixed block" global id corresponding to each input token.

    This implementation is a simlified version of the original Flaxformr implementation adopted from:
    https://github.com/google/flaxformer/blob/main/flaxformer/architectures/longt5/long_attention.py.

    In our scenario, as we use this strategy only for a decoder, orphan tokens, i.e. those tokens which do not make for
    the whole fixed block, are assigned to the preceding block.

    Padding tokens from the original sequence are represented by -1.
    """
    batch_size, seq_len = attention_mask.shape[:2]

    def handle_orphan_tokens(block_ids: torch.Tensor) -> torch.Tensor:
        block_ends = (
            torch.arange(seq_len) % global_block_size
        ) == global_block_size - 1
        block_ends = block_ends.to(block_ids.device)
        true_block_ends = torch.logical_and(block_ends, block_ids >= 0)
        full_blocks = true_block_ends.sum(-1).unsqueeze(-1).type(block_ids.dtype) - 1
        block_ids = torch.where(block_ids < full_blocks, block_ids, full_blocks)
        return block_ids

    fixed_block_mask = (
        torch.ones_like(attention_mask, device=attention_mask.device)
        / global_block_size
    )
    fixed_block_mask = torch.cumsum(fixed_block_mask, axis=1) - fixed_block_mask
    mask = torch.where(attention_mask != 0.0, 1.0, -1000.0).type(attention_mask.dtype)
    global_block_ids = torch.floor(mask + fixed_block_mask - 1.0).type(
        attention_mask.dtype
    )
    _global_block_ids_lower_bound = torch.tensor(
        -1, dtype=global_block_ids.dtype, device=global_block_ids.device
    )
    global_block_ids = torch.where(
        global_block_ids > _global_block_ids_lower_bound,
        global_block_ids,
        _global_block_ids_lower_bound,
    )
    # set padding tokens to -1
    global_block_ids = (global_block_ids * attention_mask) + (attention_mask - 1)
    # [batch_size, seq_len]
    global_block_ids = handle_orphan_tokens(global_block_ids)
    num_globals = seq_len // global_block_size
    # [batch_size, seq_len // global_block_size]
    if num_globals > 0:
        _sequence_block_ids_max = (
            torch.max(global_block_ids, dim=-1)
            .values.repeat(num_globals, 1)
            .transpose(0, 1)
        )
    else:
        _sequence_block_ids_max = torch.zeros(
            batch_size, 0, dtype=global_block_ids.dtype, device=global_block_ids.device
        )
    global_segment_ids = torch.cumsum(torch.ones(batch_size, num_globals), dim=-1) - 1
    global_segment_ids = global_segment_ids.to(attention_mask.device)
    global_segment_ids = torch.where(
        global_segment_ids <= _sequence_block_ids_max, 1, 0
    )
    return global_block_ids.type(torch.int), global_segment_ids.type(torch.int)


def _make_side_relative_position_ids(
    attention_mask: torch.Tensor, global_block_size: int
) -> torch.Tensor:
    """Create the relative position tensor for local -> global attention."""
    block_ids, global_segment_ids = _make_global_fixed_block_ids(
        attention_mask, global_block_size
    )
    global_seq_len = global_segment_ids.shape[-1]
    global_positions = torch.arange(global_seq_len, device=block_ids.device)
    side_relative_position = global_positions - block_ids[..., None]
    return side_relative_position.type(torch.int64)


def _create_global_aggregates(
    hidden_states: torch.Tensor, block_ids: torch.Tensor, global_seq_len: int
) -> torch.Tensor:
    """Compute individual block aggregates by summing over individual blocks."""
    # (batch..., seq_len, global_seq_len))
    block_ids = block_ids.where(
        block_ids >= 0,
        torch.tensor(global_seq_len, dtype=block_ids.dtype, device=block_ids.device),
    )
    one_hot_block_ids = nn.functional.one_hot(
        block_ids.type(torch.int64), global_seq_len + 1
    )[:, :, :-1]
    return torch.einsum(
        "...nd,...ng->...gd", hidden_states, one_hot_block_ids.type(hidden_states.dtype)
    )


# Copied from transformers.models.t5.modeling_t5.T5LayerNorm with T5->LongT5
class LongT5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the LongT5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # LongT5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


try:
    from apex.normalization import FusedRMSNorm

    LongT5LayerNorm = FusedRMSNorm  # noqa

    logger.info(
        "Discovered apex.normalization.FusedRMSNorm - will use it instead of LongT5LayerNorm"
    )
except ImportError:
    # using the normal LongT5LayerNorm
    pass
except Exception:
    logger.warning(
        "discovered ape.configuration_longt5x but it failed to load, falling back to LongT5LayerNorm"
    )
    pass

ALL_LAYERNORM_LAYERS.append(LongT5LayerNorm)


# Copied from transformers.models.t5.modeling_t5.T5DenseActDense with T5->LongT5
class LongT5DenseActDense(nn.Module):
    def __init__(self, config: LongT5Config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.config = config
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = F.dropout(
            hidden_states, p=self.config.dropout_rate, training=self.training
        )
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class LongT5DenseGatedActDense(nn.Module):
    def __init__(self, config: LongT5Config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.act = ACT2FN[config.dense_act_fn]
        self.config = config

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = F.dropout(
            hidden_states, p=self.config.dropout_rate, training=self.training
        )
        hidden_states = self.wo(hidden_states)
        return hidden_states


# Copied from transformers.models.t5.modeling_t5.T5LayerFF with T5->LongT5
class LongT5LayerFF(nn.Module):
    def __init__(self, config: LongT5Config):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = LongT5DenseGatedActDense(config)
        else:
            self.DenseReluDense = LongT5DenseActDense(config)

        self.layer_norm = LongT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.config = config

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + F.dropout(
            forwarded_states, p=self.config.dropout_rate, training=self.training
        )
        return hidden_states


class LongT5TransientGlobalAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias: bool = False) -> None:
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.local_radius = config.local_radius
        self.block_len = self.local_radius + 1
        self.global_block_size = config.global_block_size
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.config = config

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )
        self.pruned_heads = set()
        self.gradient_checkpointing = False

        # Relativen attention bias & Layer norm for global attention
        if self.has_relative_attention_bias:
            self.global_relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )
        self.global_input_layer_norm = LongT5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )

    # Copied from transformers.models.t5.modeling_t5.T5Attention.prune_heads
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    # Copied from transformers.models.t5.modeling_t5.T5Attention._relative_position_bucket
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position)
            )
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )

        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

    def compute_bias(self, block_length: int):
        """Compute binned relative position bias"""
        target_device = (
            self.relative_attention_bias.weight.device
            if self.relative_attention_bias.weight.device.type != "meta"
            else None
        )
        memory_position = torch.arange(
            3 * block_length, dtype=torch.long, device=target_device
        )
        context_position = memory_position[block_length:-block_length]

        # (block_length, 3 * block_length)
        relative_position = memory_position[None, :] - context_position[:, None]
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # (block_length, 3 * block_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # (block_length, 3 * block_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        # (1, 1, num_heads, block_length, 3 * block_length)
        values = values.permute([2, 0, 1]).unsqueeze(0).unsqueeze(0)
        return values

    def compute_side_bias(
        self, mask: torch.Tensor, global_segment_ids: torch.Tensor
    ) -> torch.Tensor:
        # (batch_size, 1, seq_len, global_seq_len)
        side_attention_mask = torch.eq(mask[..., None], global_segment_ids[:, None, :])[
            :, None, ...
        ]
        attention_side_bias = torch.where(side_attention_mask > 0, 0.0, -1e10)
        # (batch_size, seq_len, global_seq_len)
        side_relative_position = _make_side_relative_position_ids(
            mask, self.global_block_size
        )
        side_relative_position_bucket = self._relative_position_bucket(
            side_relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # (batch_size, seq_len, global_seq_len, num_heads)
        side_bias = self.global_relative_attention_bias(side_relative_position_bucket)

        # (batch_size, num_heads, seq_len, global_seq_len)
        side_bias = side_bias.permute([0, 3, 1, 2])
        # (batch_size, num_heads, seq_len, global_seq_len)
        attention_side_bias = attention_side_bias + side_bias
        return attention_side_bias

    def forward(
        self,
        hidden_states,
        mask=None,
        position_bias=None,
        layer_head_mask=None,
        output_attentions=False,
    ):
        batch_size, seq_length = hidden_states.shape[:2]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim)

        def unshape(states):
            """reshape"""
            return states.contiguous().view(batch_size, -1, self.inner_dim)

        # Prepare components for transient-global attention
        # Obtain block_ids and global_segment_ids
        # global_seq_len := seq_len // self.global_block_size
        # shapes: (batch_size, seq_len) & (batch_size, global_seq_len)
        block_ids, global_segment_ids = _make_global_fixed_block_ids(
            mask if mask is not None else torch.ones(hidden_states.shape[:-1]),
            self.global_block_size,
        )
        # Create global inputs
        _global_seq_len = global_segment_ids.shape[-1]
        global_inputs = _create_global_aggregates(
            hidden_states, block_ids, _global_seq_len
        )
        global_inputs = self.global_input_layer_norm(global_inputs)

        # get query states -> (batch_size, seq_length, n_heads, dim_per_head)
        query_states = shape(self.q(hidden_states))
        key_states = shape(self.k(hidden_states))
        value_states = shape(self.v(hidden_states))
        # Get global/side key/value states  shape: (batch_size, global_seq_len, n_heads, dim_per_head)
        side_key_states = shape(self.k(global_inputs))
        side_value_states = shape(self.v(global_inputs))

        # Split into blocks -> (batch_size, num_blocks, block_len, n_heads, dim_per_head)
        query_states = _split_into_blocks(query_states, self.block_len, dim=1)
        key_states = _split_into_blocks(key_states, self.block_len, dim=1)
        value_states = _split_into_blocks(value_states, self.block_len, dim=1)

        # Concatenate 3 blocks for keys and values -> (batch_size, num_blocks, 3 * block_len, n_heads, dim_per_head)
        key_states = _concatenate_3_blocks(key_states, block_dim=1, sequence_dim=2)
        value_states = _concatenate_3_blocks(value_states, block_dim=1, sequence_dim=2)

        # Tile side inputs across local key/value blocks
        # New shape: (batch_size, num_blocks, global_seq_len, n_heads, dim_per_head)
        reps = [1] * (side_key_states.ndim + 1)
        reps[1] = key_states.shape[1]
        side_key_states = side_key_states.unsqueeze(1).repeat(reps)
        side_value_states = side_value_states.unsqueeze(1).repeat(reps)

        # Concatenate "local" and "side"/"global" key/value states to allow each token to attend global aggregated ones
        # New shape: (batch_size, num_blocks, 3 * block_len + global_seq_len, n_heads, dim_per_head)
        key_states = torch.cat([key_states, side_key_states], dim=2)
        value_states = torch.cat([value_states, side_value_states], dim=2)

        # Compute scores -> (batch_size, num_block, n_heads, block_len, 3 * block_len + global_seq_len)
        scores = torch.einsum("...qhd,...khd->...hqk", query_states, key_states)

        if mask is not None:
            # We need to adjust position bias shape to be sum with mask
            local_attention_mask = _get_local_attention_mask(
                mask, self.block_len, hidden_states.device
            )
            # Replace masked positions with -10_000 (according to the original implementation)
            local_attention_mask = torch.where(local_attention_mask > 0, 0.0, -1e10)
        else:
            local_attention_mask = None

        if position_bias is None:
            # position_bias shape: # (1, 1, n_heads, block_len, 3 * block_len)
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, 1, self.n_heads, self.block_len, 3 * self.block_len),
                    device=scores.device,
                    dtype=scores.dtype,
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(self.block_len)

            if local_attention_mask is not None:
                # (batch_size, 1, n_heads, block_len, 3 * block_len)
                position_bias = position_bias + local_attention_mask.transpose(1, 2)
            position_bias = position_bias.type(scores.dtype)

            # Calculate global/side bias - shape: # (batch_size, num_heads, seq_len, global_seq_len)
            if mask is None:
                mask = torch.ones(batch_size, seq_length)
            # (batch_size, num_heads, seq_len, global_seq_len)
            side_position_bias = self.compute_side_bias(mask, global_segment_ids)
            # (batch_size, num_blocks, num_heads, block_len, global_seq_len)
            side_position_bias = _split_into_blocks(
                side_position_bias, self.block_len, dim=-2
            ).transpose(1, 2)
            side_position_bias = side_position_bias.type(scores.dtype).to(scores.device)
            # (batch_size, num_blocks, num_heads, block_len, 3 * block_len + global_seq_len)
            position_bias = torch.cat([position_bias, side_position_bias], dim=-1)

        scores += position_bias
        # (batch_size, num_blocks, n_heads, block_len, 3 * block_len + global_seq_len)
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.config.dropout_rate, training=self.training
        )

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask
        attn_weights = attn_weights.type(value_states.dtype)
        attn_output = unshape(
            torch.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        )
        attn_output = attn_output[:, :seq_length, :]
        attn_output = self.o(attn_output)

        present_key_value_state = None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class LongT5Attention(nn.Module):
    def __init__(self, config: LongT5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.config = config

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position)
            )
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )

        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[
            :, None
        ]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[
            None, :
        ]
        relative_position = (
            memory_position - context_position
        )  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(
            relative_position_bucket
        )  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += (
                past_key_value[0].shape[2] if query_length is None else query_length
            )

        key_length = (
            real_seq_length if key_value_states is None else key_value_states.shape[1]
        )

        def shape(states):
            """projection"""
            return states.view(
                batch_size, -1, self.n_heads, self.key_value_proj_dim
            ).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return (
                states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
            )

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(
            self.q(hidden_states)
        )  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states,
            self.k,
            key_value_states,
            past_key_value[0] if past_key_value is not None else None,
        )
        value_states = project(
            hidden_states,
            self.v,
            key_value_states,
            past_key_value[1] if past_key_value is not None else None,
        )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length),
                    device=scores.device,
                    dtype=scores.dtype,
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length, key_length, device=scores.device
                )

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = (
                    position_bias + mask
                )  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.config.dropout_rate, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(
            torch.matmul(attn_weights, value_states)
        )  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (
            (key_states, value_states) if (self.is_decoder and use_cache) else None
        )
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


# Copied from transformers.models.t5.modeling_t5.T5LayerSelfAttention with T5->LongT5
class LongT5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = LongT5Attention(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        self.layer_norm = LongT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.config = config

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        q=None,
        original_attention_mask=None,  # to absorb the kwarg
    ):
        if q is None:
            normed_hidden_states = self.layer_norm(hidden_states)
        else:
            normed_hidden_states = hidden_states

        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + F.dropout(
            attention_output[0], p=self.config.dropout_rate, training=self.training
        )
        outputs = (hidden_states,) + attention_output[
            1:
        ]  # add attentions if we output them
        return outputs


# Copied from transformers.models.t5.modeling_t5.T5LayerCrossAttention with T5->LongT5
class LongT5LayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = LongT5Attention(
            config, has_relative_attention_bias=False
        )
        self.layer_norm = LongT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.config = config

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + F.dropout(
            attention_output[0], p=self.config.dropout_rate, training=self.training
        )
        outputs = (layer_output,) + attention_output[
            1:
        ]  # add attentions if we output them
        return outputs
