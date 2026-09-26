# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint state must equal an independently executed prefix, without mutation."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.flash_next_prefill_checkpoint import (
    build_prefill_checkpoint,
    export_gdn_checkpoint,
    store_checkpoint_history,
)
from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


def make_plan():
    device = torch.device("cuda", torch.cuda.current_device())
    offsets = torch.tensor([0, 2048, 4096], dtype=torch.int32)
    common = SimpleNamespace(
        query_start_loc_cpu=offsets,
        query_start_loc=offsets.to(device),
        seq_lens_cpu_upper_bound=torch.tensor([2048, 2048], dtype=torch.int32),
        block_table_tensor=torch.tensor([[1, 5, 2], [3, 8, 4]],
                                       device=device, dtype=torch.int32),
    )
    spec = SimpleNamespace(num_prefill_checkpoint_blocks=1, block_size=944,
                           prefill_checkpoint_alignment=1)
    config = SimpleNamespace(
        cache_config=SimpleNamespace(prefix_match_unit=None),
        speculative_config=SimpleNamespace(use_eagle_block_drop=lambda: True),
    )
    plan = build_prefill_checkpoint(common, spec, config, [0, 1], offsets)
    assert plan.state_indices.tolist() == [5, 8]
    assert plan.global_token_ends.tolist() == [944, 2992]
    assert plan.query_start_loc.tolist() == [0, 944, 1888]
    assert plan.token_indices.tolist() == list(range(944)) + list(range(2048, 2992))
    # A continuation starting at the checkpoint must not overwrite its input state.
    resumed_offsets = torch.tensor([0, 1104, 2208], dtype=torch.int32)
    common.query_start_loc_cpu = resumed_offsets
    common.query_start_loc = resumed_offsets.to(device)
    assert build_prefill_checkpoint(
        common, spec, config, [0, 1], resumed_offsets
    ) is None
    return device, plan


@pytest.mark.parametrize("width,channels", [(3, 2560), (9, 10240)])
@pytest.mark.parametrize("transposed", [False, True])
def test_checkpoint_history_preserves_neighbors_and_speculative_slack(width, channels,
                                                                     transposed):
    device, plan = make_plan()
    torch.manual_seed(1701)
    inputs = torch.randn(4096, channels + 8, device=device,
                         dtype=torch.bfloat16)[:, :channels]
    before_inputs = inputs.clone()
    backing = torch.full((11, 822272 // 2), -17, device=device, dtype=torch.bfloat16)
    state = backing.as_strided(
        (11, channels, width + 4),
        (backing.stride(0), 1 if transposed else width + 4,
         channels if transposed else 1),
    )
    expected_backing = backing.clone()
    expected = expected_backing.as_strided(state.shape, state.stride())
    for block, end in ((5, 944), (8, 2992)):
        expected[block, :, :width] = inputs[end-width:end].T
    store_checkpoint_history(inputs, state, plan, width)
    assert torch.equal(backing, expected_backing)
    assert torch.equal(inputs, before_inputs)
    plan = replace(
        plan, state_indices=torch.full_like(plan.state_indices, NULL_BLOCK_ID)
    )
    before = backing.clone()
    store_checkpoint_history(inputs, state, plan, width)
    assert torch.equal(backing, before)


@pytest.mark.parametrize("null_second", [False, True])
def test_gdn_checkpoint_matches_independent_prefix_and_preserves_inputs(null_second):
    device, plan = make_plan()
    torch.manual_seed(1729)
    q = torch.nn.functional.normalize(torch.randn(1, 4096, 4, 128, device=device),
                                     dim=-1).bfloat16()
    k = torch.nn.functional.normalize(torch.randn_like(q).float(), dim=-1).bfloat16()
    v = torch.randn(1, 4096, 12, 128, device=device, dtype=torch.bfloat16)
    g = -torch.rand(1, 4096, 12, device=device) * 0.1
    beta = torch.rand_like(g)
    initial = torch.randn(2, 12, 128, 128, device=device) * 0.01
    inputs = [q, k, v, g, beta, initial]
    snapshots = [x.clone() for x in inputs]
    backing = torch.full((11, 822272 // 4), -17., device=device)
    cache = backing[:, :12*128*128].view(11, 12, 128, 128)
    expected = backing.clone()
    expected_cache = expected[:, :12*128*128].view_as(cache)
    if null_second:
        plan.state_indices[1] = NULL_BLOCK_ID
    for request, start in enumerate((0, 2048)):
        if null_second and request == 1:
            continue
        _, state = chunk_gated_delta_rule(
            q=q[:, start:start+944], k=k[:, start:start+944],
            v=v[:, start:start+944], g=g[:, start:start+944],
            beta=beta[:, start:start+944], initial_state=initial[request:request+1],
            cu_seqlens=torch.tensor([0, 944], device=device, dtype=torch.int32),
            output_final_state=True, use_qk_l2norm_in_kernel=False,
        )
        expected_cache[(5, 8)[request]].copy_(state[0])
    export_gdn_checkpoint(
        chunk_gated_delta_rule, plan, q, k, v, g, beta, initial, cache
    )
    assert torch.isfinite(cache).all()
    assert torch.equal(backing, expected)
    assert all(torch.equal(x, old) for x, old in zip(inputs, snapshots))
