"""Default-off SM86 C1 EP combine; native fallback preserves other inputs.

This module alone does not install a serving hook. Inputs are already weighted.
"""
import os

import torch
import triton
import triton.language as tl
import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.utils.torch_utils import direct_register_custom_op

_ENABLED = os.environ.get('VLLM_FLASH_MOE_SUM_C1', '0') == '1'


def eligible(input, output, topk_ids, expert_map):
    if not _ENABLED or envs.VLLM_BATCH_INVARIANT:
        return False
    if topk_ids is None or expert_map is None:
        return False
    tensors = (input, output, topk_ids, expert_map)
    if (tuple(input.shape) != (1, 10, 2560)
            or tuple(output.shape) != (1, 2560)
            or tuple(topk_ids.shape) != (1, 10)
            or tuple(expert_map.shape) != (512,)
            or input.dtype != torch.bfloat16 or output.dtype != torch.bfloat16
            or topk_ids.dtype != torch.int32 or expert_map.dtype != torch.int32
            or not all(t.is_cuda and t.device == input.device
                       and t.is_contiguous() for t in tensors)
            or any(t.requires_grad for t in tensors)
            or torch.cuda.get_device_capability(input.device) != (8, 6)):
        return False
    # Conservative storage-level alias rejection, including disjoint views.
    # This executes inside the opaque custom op, not in Dynamo's tracing path.
    out_storage = output.untyped_storage().data_ptr()
    return all(t.untyped_storage().data_ptr() != out_storage
               for t in (input, topk_ids, expert_map))


@triton.jit
def _combine10(X, IDS, MAP, Y, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for k in tl.static_range(10):
        expert = tl.load(IDS + k)
        local = tl.load(MAP + tl.maximum(expert, 0), expert >= 0, other=-1)
        valid = (expert >= 0) & (local >= 0)
        value = tl.load(X + k * 2560 + offsets,
                        (offsets < 2560) & valid, other=0).to(tl.float32)
        acc = tl.where(valid, acc + value, acc)
    tl.store(Y + offsets, acc, offsets < 2560)


def _native(input, output, topk_ids, expert_map):
    # Match MarlinExperts.moe_sum exactly: without an expert map the original
    # method deliberately does not pass topk_ids to the native operator.
    if expert_map is not None:
        return ops.moe_sum(input, output, topk_ids, expert_map)
    return ops.moe_sum(input, output)


def _dispatch(input: torch.Tensor, output: torch.Tensor,
              topk_ids: torch.Tensor | None, expert_map: torch.Tensor | None) -> None:
    if eligible(input, output, topk_ids, expert_map):
        _combine10[(20,)](input, topk_ids, expert_map, output, 128,
                          num_warps=4, enable_fp_fusion=False)
    else:
        _native(input, output, topk_ids, expert_map)


def _fake(input: torch.Tensor, output: torch.Tensor,
          topk_ids: torch.Tensor | None, expert_map: torch.Tensor | None) -> None:
    return None


direct_register_custom_op(op_name='flash_moe_sum_c1', op_func=_dispatch,
                          mutates_args=['output'], fake_impl=_fake)


def moe_sum(input, output, topk_ids, expert_map):
    if not _ENABLED:
        return _native(input, output, topk_ids, expert_map)
    return torch.ops.vllm.flash_moe_sum_c1(input, output, topk_ids, expert_map)
