# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native checkpoint exports for the opt-in Flash-Next prefill experiment."""

import os
from dataclasses import dataclass, replace

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import (
    get_mamba_prefill_checkpoint_position,
    is_mamba_prefill_checkpoint_valid,
)

CHECKPOINT_STATS = {"plans": 0, "histories": 0, "recurrent": 0}


@dataclass
class PrefillCheckpoint:
    request_indices: torch.Tensor
    token_indices: torch.Tensor
    query_start_loc: torch.Tensor
    chunk_indices: torch.Tensor
    chunk_offsets: torch.Tensor
    global_token_ends: torch.Tensor
    state_indices: torch.Tensor


def enable_checkpoint_spec(spec, config, backend):
    """Advertise native checkpoint slots only for the qualified experiment shape."""
    flag = os.environ.get("VLLM_FLASH_PREFILL_CHECKPOINT", "0")
    if flag == "0":
        return spec
    from vllm.platforms import current_platform

    speculative = config.speculative_config
    if not (
        flag == "1" and backend == "triton"
        and config.model_config.hf_text_config.model_type == "qwen4_exp_text"
        and config.parallel_config.tensor_parallel_size == 4
        and config.parallel_config.pipeline_parallel_size == 1
        and config.cache_config.mamba_cache_mode == "align"
        and config.scheduler_config.max_num_seqs <= 4
        and config.scheduler_config.max_num_batched_tokens <= 2048
        and speculative is not None and speculative.num_speculative_tokens == 4
        and current_platform.is_cuda() and current_platform.is_device_capability(86)
    ):
        raise ValueError("Prefill checkpoints require Flash-Next TP4/SM86/MTP4/align")
    return replace(spec, num_prefill_checkpoint_blocks=1,
                   prefill_checkpoint_alignment=1)


def build_prefill_checkpoint(common, spec, config, request_rows, query_offsets):
    """Use native boundary validation and native reserved checkpoint columns."""
    if not spec.num_prefill_checkpoint_blocks or not request_rows:
        return None
    from vllm.third_party.flash_linear_attention.ops.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )
    from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

    device = common.query_start_loc.device
    starts = common.query_start_loc_cpu.tolist()
    lengths = common.seq_lens_cpu_upper_bound.tolist()
    offsets = query_offsets.tolist()
    block = spec.block_size
    hash_block = config.cache_config.prefix_match_unit or block
    speculative = config.speculative_config
    drop = speculative is not None and speculative.use_eagle_block_drop()
    requests, tokens, ends, columns, selected_rows = [], [], [], [], []
    packed_offsets = [0]
    for local, row in enumerate(request_rows):
        query_len = starts[row + 1] - starts[row]
        seq_len = lengths[row]
        query_start = seq_len - query_len
        checkpoint = get_mamba_prefill_checkpoint_position(seq_len, hash_block, drop)
        if not is_mamba_prefill_checkpoint_valid(
            query_start, seq_len, checkpoint, hash_block, block,
            spec.prefill_checkpoint_alignment,
        ):
            continue
        count = checkpoint - query_start
        assert offsets[local + 1] - offsets[local] == query_len
        requests.append(local)
        tokens.extend(range(offsets[local], offsets[local] + count))
        ends.append(starts[row] + count)
        selected_rows.append(row)
        columns.append((seq_len + block - 1) // block - 2)
        packed_offsets.append(packed_offsets[-1] + count)
    if not requests:
        return None
    CHECKPOINT_STATS["plans"] += 1
    packed_cpu = torch.tensor(packed_offsets, dtype=torch.int32)

    def h2d(values, dtype=torch.int64):
        return async_tensor_h2d(values, dtype=dtype, device=device)

    states = common.block_table_tensor[h2d(selected_rows), h2d(columns)]
    return PrefillCheckpoint(
        h2d(requests), h2d(tokens), h2d(packed_offsets, torch.int32),
        h2d(prepare_chunk_indices(packed_cpu, FLA_CHUNK_SIZE)),
        h2d(prepare_chunk_offsets(packed_cpu, FLA_CHUNK_SIZE)),
        h2d(ends), states,
    )


@triton.jit
def _store_history(
    X, STATE, ENDS, IDS,
    X_STRIDE: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
    CHANNELS: tl.constexpr, WIDTH: tl.constexpr, NULL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    x = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    state = tl.load(IDS + request).to(tl.int64)
    end = tl.load(ENDS + request)
    channel, history = x // WIDTH, x % WIDTH
    valid = (channel < CHANNELS) & (state != NULL) & (end >= WIDTH)
    value = tl.load(X + (end - WIDTH + history) * X_STRIDE + channel,
                    mask=valid, other=0)
    tl.store(STATE + state * S0 + channel * S1 + history * S2, value, mask=valid)


def store_checkpoint_history(inputs, conv_state, checkpoint, width):
    """Export the meaningful history only; preserve speculative slack bytes."""
    if checkpoint is None:
        return
    assert inputs.ndim == 2 and inputs.stride(1) == 1
    assert conv_state.ndim == 3 and conv_state.shape[1] == inputs.shape[1]
    assert 0 < width <= conv_state.shape[2]
    CHECKPOINT_STATS["histories"] += checkpoint.state_indices.numel()
    _store_history[(checkpoint.state_indices.numel(),
                    triton.cdiv(inputs.shape[1] * width, 256))](
        inputs, conv_state, checkpoint.global_token_ends, checkpoint.state_indices,
        inputs.stride(0), *conv_state.stride(), inputs.shape[1], width,
        NULL_BLOCK_ID, 256,
    )


@triton.jit
def _store_recurrent(SOURCE, CACHE, IDS, SRC0: tl.constexpr, DST0: tl.constexpr,
                     WIDTH: tl.constexpr, NULL: tl.constexpr, BLOCK: tl.constexpr):
    request = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    state = tl.load(IDS + request).to(tl.int64)
    valid = (offset < WIDTH) & (state != NULL)
    value = tl.load(SOURCE + request * SRC0 + offset, mask=valid, other=0)
    tl.store(CACHE + state * DST0 + offset, value, mask=valid)


def export_gdn_checkpoint(kernel, checkpoint, q, k, v, g, beta,
                          initial_state, recurrent_cache):
    """Replay only the recurrent prefix, avoiding another backbone/MLP pass."""
    if checkpoint is None:
        return
    CHECKPOINT_STATS["recurrent"] += checkpoint.state_indices.numel()
    index = checkpoint.token_indices
    _, state = kernel(
        q=q.index_select(1, index), k=k.index_select(1, index),
        v=v.index_select(1, index), g=g.index_select(1, index),
        beta=beta.index_select(1, index),
        initial_state=initial_state.index_select(0, checkpoint.request_indices),
        output_final_state=True, cu_seqlens=checkpoint.query_start_loc,
        chunk_indices=checkpoint.chunk_indices, chunk_offsets=checkpoint.chunk_offsets,
        use_qk_l2norm_in_kernel=False,
    )
    state = state.contiguous().view(state.shape[0], -1)
    cache = recurrent_cache.view(recurrent_cache.shape[0], -1)
    assert state.shape[1] == cache.shape[1]
    assert state.stride(1) == cache.stride(1) == 1
    _store_recurrent[(state.shape[0], triton.cdiv(state.shape[1], 256))](
        state, cache, checkpoint.state_indices, state.stride(0), cache.stride(0),
        state.shape[1], NULL_BLOCK_ID, 256,
    )
