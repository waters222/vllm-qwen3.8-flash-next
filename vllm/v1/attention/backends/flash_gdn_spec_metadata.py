# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused metadata staging for unpadded, uniform speculative GDN decode.

Writes the builder's existing persistent buffers, preserving graph addresses.
Mixed batches and partial speculative steps keep the native builder path.
"""

import os

import torch

from vllm.triton_utils import tl, triton

ENABLED = os.environ.get("VLLM_FLASH_GDN_SPEC_METADATA", "0") == "1"


@triton.jit
def _stage_spec(
    TABLE, SEQ_LENS, ACCEPTED, STATES, MASKS, TOKENS, QUERIES, ACCEPTED_OUT,
    N: tl.constexpr, WIDTH: tl.constexpr, TABLE_STRIDE: tl.constexpr,
    STATE_STRIDE: tl.constexpr, BLOCK_SIZE: tl.constexpr, ALIGN: tl.constexpr,
    TILE: tl.constexpr,
):
    token = tl.arange(0, TILE)
    row = token // WIDTH
    column = token % WIDTH
    start = tl.full((TILE,), 0, tl.int32)
    if ALIGN:
        length = tl.load(SEQ_LENS + row, row < N, other=1)
        start = tl.maximum((length - 1) // BLOCK_SIZE, 0)
    state = tl.load(TABLE + row * TABLE_STRIDE + start + column, row < N, other=0)
    tl.store(STATES + row * STATE_STRIDE + column, state, row < N)
    tl.store(TOKENS + token, token, token < N * WIDTH)
    tl.store(MASKS + token, True, token < N)
    count = tl.load(ACCEPTED + token, token < N, other=1)
    tl.store(ACCEPTED_OUT + token, count, token < N)
    tl.store(QUERIES + token, token * WIDTH, token <= N)


def try_build(builder, metadata, metadata_cls, accepted, draft_counts, fast_build):
    """Return None without writing buffers when the batch is unsupported."""
    if not ENABLED:
        return None
    n = metadata.num_reqs
    width = builder.num_spec + 1
    mode = builder.vllm_config.cache_config.mamba_cache_mode
    if (
        fast_build or not builder.use_spec_decode or not builder.use_full_cuda_graph
        or n not in (1, 2, 3, 4, 5, 6) or width != 5
        or mode not in ("none", "align")
        or metadata.max_query_len != width
        or metadata.num_actual_tokens != n * width
        or n * width > builder.decode_cudagraph_max_bs
        or accepted is None or draft_counts is None
        or builder.kv_cache_spec.num_speculative_blocks != builder.num_spec
    ):
        return None
    query_cpu = metadata.query_start_loc_cpu
    if (
        query_cpu.device.type != "cpu" or draft_counts.device.type != "cpu"
        or query_cpu.tolist() != list(range(0, (n + 1) * width, width))
        or draft_counts.tolist() != [builder.num_spec] * n
    ):
        return None
    table = metadata.block_table_tensor
    tensors = (
        table, metadata.seq_lens, accepted, builder.spec_state_indices_tensor,
        builder.spec_token_indx, builder.spec_query_start_loc,
        builder.num_accepted_tokens,
    )
    if (
        not all(t.is_cuda and t.device == table.device and t.dtype == torch.int32
                and t.is_contiguous() for t in tensors)
        or table.ndim != 2 or table.shape[0] != n or table.shape[1] < width
        or accepted.shape != (n,) or metadata.seq_lens.shape != (n,)
    ):
        return None
    _stage_spec[(1,)](
        table, metadata.seq_lens, accepted, builder.spec_state_indices_tensor,
        builder.spec_sequence_masks, builder.spec_token_indx,
        builder.spec_query_start_loc, builder.num_accepted_tokens,
        n, width, table.stride(0), builder.spec_state_indices_tensor.stride(0),
        builder.kv_cache_spec.block_size, mode == "align",
        triton.next_power_of_2(n * width), num_warps=1,
    )
    return metadata_cls(
        num_prefills=0, num_prefill_tokens=0, num_decodes=0, num_decode_tokens=0,
        num_spec_decodes=n, num_spec_decode_tokens=n * width,
        num_actual_tokens=n * width,
        spec_query_start_loc=builder.spec_query_start_loc[:n + 1],
        spec_state_indices_tensor=builder.spec_state_indices_tensor[:n],
        spec_sequence_masks=builder.spec_sequence_masks[:n],
        spec_token_indx=builder.spec_token_indx[:n * width],
        non_spec_token_indx=builder.non_spec_token_indx[:0],
        num_accepted_tokens=builder.num_accepted_tokens[:n],
    )
