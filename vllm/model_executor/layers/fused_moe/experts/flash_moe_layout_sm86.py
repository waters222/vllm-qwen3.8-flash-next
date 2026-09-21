"""Default-off SM86 no-EP M1--4 expert alignment and combine specialization.

Alignment uses the validated fixed64/four-warp variant. Valid router IDs are
0..511; no routing or weight arithmetic changes. Other calls use native paths.
"""
import os

import torch
import triton
import triton.language as tl
import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size as _native_align,
)
from vllm.utils.torch_utils import direct_register_custom_op

_ENABLED = os.environ.get('VLLM_FLASH_MOE_LAYOUT_SM86', '0') == '1'


def supported_marlin_callback(callback, owner_type):
    owner = getattr(callback, '__self__', None)
    return (type(owner) is owner_type
            and getattr(callback, '__func__', None) is owner_type.moe_sum
            and owner._lora_context is None)


def alignment_eligible(ids, block_size, num_experts, expert_map,
                       pad_sorted_ids, ignore_invalid_experts):
    return (_ENABLED and not envs.VLLM_BATCH_INVARIANT
            and ids.ndim == 2 and 1 <= ids.shape[0] <= 4 and ids.shape[1] == 10
            and ids.dtype == torch.int32 and ids.is_cuda and ids.is_contiguous()
            and not ids.requires_grad
            and torch.cuda.get_device_capability(ids.device) == (8, 6)
            and block_size == 8 and num_experts == 512 and expert_map is None
            and pad_sorted_ids is False and ignore_invalid_experts is True)


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b)


@triton.jit
def _align_small(IDS, SORTED, EXPERTS, TOTAL, T: tl.constexpr, INIT: tl.constexpr):
    positions = tl.arange(0, 64)
    ids = tl.load(IDS + positions, positions < T, 512).to(tl.int32)
    # Normalize padding before packing: signed division/remainder of negative
    # keys can otherwise produce expert 0 with negative token indices.
    ids = tl.where((ids >= 0) & (ids < 512), ids, 512)
    keys = tl.sort(ids * 64 + positions, descending=False)
    experts = keys // 64
    tokens = keys % 64
    valid = (positions < T) & (experts >= 0) & (experts < 512)
    previous = tl.gather(experts, tl.maximum(positions - 1, 0), axis=0)
    starts = valid & ((positions == 0) | (experts != previous))
    first = tl.associative_scan(tl.where(starts, positions, 0), 0, _maximum)
    within = positions - first
    block_starts = valid & (within % 8 == 0)
    blocks = tl.cumsum(block_starts.to(tl.int32), axis=0)
    init = tl.arange(0, INIT)
    tl.store(SORTED + init, T, init < T * 8)
    tl.store(EXPERTS + positions, -1, positions < T)
    tl.debug_barrier()
    tl.store(SORTED + (blocks - 1) * 8 + within % 8, tokens, valid)
    tl.store(EXPERTS + blocks - 1, experts, block_starts)
    tl.store(TOTAL, tl.sum(block_starts.to(tl.int32), axis=0) * 8)


def _alignment(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not alignment_eligible(ids, 8, 512, None, False, True):
        raise RuntimeError('Alignment custom op requires enabled SM86 int32 M1--4 top10 no-EP scope')
    count = ids.numel()
    sorted_ids = torch.empty((count * 8,), dtype=torch.int32, device=ids.device)
    expert_ids = torch.empty((count,), dtype=torch.int32, device=ids.device)
    total = torch.empty((1,), dtype=torch.int32, device=ids.device)
    _align_small[(1,)](ids, sorted_ids, expert_ids, total, count,
                      triton.next_power_of_2(count * 8), num_warps=4)
    return sorted_ids, expert_ids, total


def _alignment_fake(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = ids.numel()
    return (torch.empty((count * 8,), dtype=torch.int32, device=ids.device),
            torch.empty((count,), dtype=torch.int32, device=ids.device),
            torch.empty((1,), dtype=torch.int32, device=ids.device))


direct_register_custom_op(op_name='flash_moe_layout_align_sm86', op_func=_alignment,
                          fake_impl=_alignment_fake)


def moe_align_block_size(ids, block_size, num_experts, expert_map=None,
                         pad_sorted_ids=False, ignore_invalid_experts=False):
    if alignment_eligible(ids, block_size, num_experts, expert_map,
                          pad_sorted_ids, ignore_invalid_experts):
        return torch.ops.vllm.flash_moe_layout_align_sm86(ids)
    return _native_align(ids, block_size, num_experts, expert_map,
                         pad_sorted_ids=pad_sorted_ids,
                         ignore_invalid_experts=ignore_invalid_experts)


def combine_eligible(x, out):
    return (_ENABLED and not envs.VLLM_BATCH_INVARIANT
            and x.ndim == 3 and 1 <= x.shape[0] <= 4
            and tuple(x.shape[1:]) == (10, 2560)
            and tuple(out.shape) == (x.shape[0], 2560)
            and all(t.dtype == torch.bfloat16 and t.is_cuda and t.is_contiguous()
                    and t.device == x.device and not t.requires_grad for t in (x, out))
            and torch.cuda.get_device_capability(x.device) == (8, 6)
            and x.untyped_storage().data_ptr() != out.untyped_storage().data_ptr())


@triton.jit
def _combine10(X, Y, IDS, USE_IDS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(1)
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for k in tl.static_range(10):
        valid = offsets < 2560
        if USE_IDS:
            expert = tl.load(IDS + row * 10 + k)
            valid = valid & (expert >= 0) & (expert < 512)
        value = tl.load(X + (row * 10 + k) * 2560 + offsets, valid, 0).to(tl.float32)
        acc = acc + value
    tl.store(Y + row * 2560 + offsets, acc, offsets < 2560)


def _combine(x: torch.Tensor, out: torch.Tensor,
             topk_ids: torch.Tensor | None = None) -> None:
    # Storage alias checking stays inside the opaque op, outside Dynamo tracing.
    ids_supported = topk_ids is None or (
        topk_ids.shape == x.shape[:2] and topk_ids.dtype == torch.int32
        and topk_ids.device == x.device and topk_ids.is_contiguous())
    if combine_eligible(x, out) and ids_supported:
        _combine10[(20, x.shape[0])](x, out, topk_ids if topk_ids is not None else x,
                                   topk_ids is not None, 128,
                                   num_warps=4, enable_fp_fusion=False)
    else:
        ops.moe_sum(x, out, topk_ids)


def _combine_fake(x: torch.Tensor, out: torch.Tensor,
                  topk_ids: torch.Tensor | None = None) -> None:
    return None


direct_register_custom_op(op_name='flash_moe_layout_combine_sm86', op_func=_combine,
                          mutates_args=['out'], fake_impl=_combine_fake)


def moe_sum(x, out, topk_ids=None):
    if not _ENABLED:
        return ops.moe_sum(x, out, topk_ids)
    return torch.ops.vllm.flash_moe_layout_combine_sm86(x, out, topk_ids)
