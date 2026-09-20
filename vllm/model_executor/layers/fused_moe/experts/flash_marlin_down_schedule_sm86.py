"""Default-off SM86 M2 expert-down scheduling, with unchanged native fallbacks.

Private stage4 library changes tile ownership, not weights or precision.
The installed call site must verify normal no-EP/no-LoRA Marlin scope first.
Registration and library verification run once, never inside a decode call.
"""
import hashlib
import os
from pathlib import Path

import torch
import vllm
import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types
from vllm.utils.torch_utils import direct_register_custom_op

_ENABLED = os.environ.get('VLLM_FLASH_MARLIN_DOWN_SCHEDULE_SM86', '0') == '1'
LIBRARY_SHA = '78541fcf5b9f9fa3ce3bc5f7fcb8de7f558c0f58be91def817a864bf3d040912'
NATIVE_EXTENSION_SHA = 'df2ea61124a7bcbe8f27708c45705a0f2624277df14f4ca914242b99ac299e6c'
NATIVE_SCHEMA_SHA = '1b958b2ea16d3a920927a92f620e3d622fc8673e248c2699110c65815ee9b4e5'
PRIVATE_SCHEMA = (
    'gemm(Tensor! a, Tensor? c_or_none, Tensor! b_q_weight, Tensor? b_bias_or_none, '
    'Tensor! b_scales, Tensor? a_scales, Tensor? global_scale, Tensor? b_zeros_or_none, '
    'Tensor! workspace, Tensor sorted_token_ids, Tensor! expert_ids, Tensor! num_tokens_past_padded, '
    'Tensor! topk_weights, int moe_block_size, int top_k, bool mul_topk_weights, int b_type_id, '
    'int size_m, int size_n, int size_k, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float, '
    'int thread_k, int thread_n, int blocks_per_sm) -> Tensor')
_OWNER = None
_OPERATOR = None


def initialize():
    """Load the pinned library before graph capture; fail closed on mismatches."""
    global _OWNER, _OPERATOR
    if _OPERATOR is not None:
        return
    path = Path(__file__).with_name('flash_marlin_dp4.so').resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != LIBRARY_SHA:
        raise RuntimeError('Whole-tile Marlin library pin differs')
    native_path = Path(vllm.__file__).parent / '_moe_C_stable_libtorch.abi3.so'
    if hashlib.sha256(native_path.read_bytes()).hexdigest() != NATIVE_EXTENSION_SHA:
        raise RuntimeError('Native Marlin extension pin differs')
    native_schema = str(torch.ops._moe_C.moe_wna16_marlin_gemm.default._schema)
    if hashlib.sha256(native_schema.encode()).hexdigest() != NATIVE_SCHEMA_SHA:
        raise RuntimeError('Native Marlin schema pin differs')
    # Schema equality is checked against the private DEF below, including its
    # argument names, mutation annotations and launch-override parameters.
    if str(path) not in torch.ops.loaded_libraries:
        # A pre-existing namespace from a different path is not silently reused.
        _OWNER = torch.library.Library('flash_marlin_dp4', 'DEF')
        _OWNER.define(PRIVATE_SCHEMA)
        torch.ops.load_library(str(path))
    operator = torch.ops.flash_marlin_dp4.gemm.default
    if str(operator._schema).split('(', 1)[1] != native_schema.split('(', 1)[1]:
        raise RuntimeError('Private/native Marlin schema differs')
    _OPERATOR = operator


def supported_scope(callback, owner_type, expert_map, global_experts, local_experts, topk):
    owner = getattr(callback, '__self__', None)
    return (_ENABLED and not envs.VLLM_BATCH_INVARIANT
            and type(owner) is owner_type
            and getattr(callback, '__func__', None) is owner_type.moe_sum
            and owner._lora_context is None and expert_map is None
            and global_experts == 512 and local_experts == 512 and topk == 10)


def tensor_eligible(a, out, weight, scales, zeros, workspace, sorted_ids,
                    expert_ids, total, routes):
    if not _ENABLED or envs.VLLM_BATCH_INVARIANT or _OPERATOR is None:
        return False
    tensors = (a, out, weight, scales, zeros, workspace, sorted_ids, expert_ids, total, routes)
    if not all(isinstance(t, torch.Tensor) and t.is_cuda and t.is_contiguous()
               and not t.requires_grad and t.device == a.device for t in tensors):
        return False
    if torch.cuda.get_device_capability(a.device) != (8, 6):
        return False
    specifications = (
        (a, (20, 320), torch.bfloat16), (out, (20, 2560), torch.bfloat16),
        (weight, (512, 20, 5120), torch.int32), (scales, (512, 10, 2560), torch.bfloat16),
        (zeros, (512, 10, 320), torch.int32), (routes, (2, 10), torch.float32),
        (total, (1,), torch.int32))
    if not all(tuple(t.shape) == shape and t.dtype == dtype for t, shape, dtype in specifications):
        return False
    sm_count = torch.cuda.get_device_properties(a.device).multi_processor_count
    if not all(t.ndim == 1 and t.dtype == torch.int32 and t.numel() >= minimum
               for t, minimum in ((workspace, sm_count * 4), (sorted_ids, 160), (expert_ids, 20))):
        return False
    # No device reads or host synchronization. These are storage metadata checks
    # inside an opaque custom op, not operations traced by Dynamo.
    spans = [(t.untyped_storage().data_ptr() + t.storage_offset() * t.element_size(),
              t.numel() * t.element_size()) for t in tensors]
    for index in (1, 5):
        start, length = spans[index]
        if any(other != index and max(start, address) < min(start + length, address + size)
               for other, (address, size) in enumerate(spans)):
            return False
    return True


def _native(a, out, weight, scales, zeros, workspace, sorted_ids, expert_ids, total, routes, size_m):
    return ops.moe_wna16_marlin_gemm(
        a, out, weight, None, scales, None, None, zeros, workspace,
        sorted_ids, expert_ids, total, routes, moe_block_size=8, top_k=1,
        mul_topk_weights=True, b_q_type=scalar_types.uint4, size_m=size_m,
        size_n=2560, size_k=320, use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)


def _down(a: torch.Tensor, out: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor,
          zeros: torch.Tensor, workspace: torch.Tensor, sorted_ids: torch.Tensor,
          expert_ids: torch.Tensor, total: torch.Tensor, routes: torch.Tensor, size_m: int) -> None:
    tensors = (a, out, weight, scales, zeros, workspace, sorted_ids, expert_ids, total, routes)
    if size_m == 20 and tensor_eligible(*tensors):
        _OPERATOR(a, out, weight, None, scales, None, None, zeros, workspace,
                  sorted_ids, expert_ids, total, routes, 8, 1, True, scalar_types.uint4.id,
                  20, 2560, 320, False, True, False, -1, -1, -1)
    else:
        _native(*tensors, size_m)


def _down_fake(a: torch.Tensor, out: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor,
               zeros: torch.Tensor, workspace: torch.Tensor, sorted_ids: torch.Tensor,
               expert_ids: torch.Tensor, total: torch.Tensor, routes: torch.Tensor, size_m: int) -> None:
    return None


direct_register_custom_op(op_name='flash_marlin_down_schedule_sm86', op_func=_down,
                          mutates_args=['out', 'workspace'], fake_impl=_down_fake)


def gemm(*args, **kwargs):
    size_m = kwargs.get('size_m')
    expected = dict(moe_block_size=8, top_k=1, mul_topk_weights=True,
                    b_q_type=scalar_types.uint4, size_m=size_m, size_n=2560, size_k=320,
                    use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
    if (type(size_m) is not int or size_m not in (10, 20, 30, 40)
            or kwargs != expected or len(args) != 13
            or any(args[i] is not None for i in (3, 5, 6))
            or any(args[i] is None for i in (0, 1, 2, 4, 7, 8, 9, 10, 11, 12))):
        return ops.moe_wna16_marlin_gemm(*args, **kwargs)
    # Keep both selected and native M1--4 kernels behind one correctly declared
    # mutable-output boundary. The installed native fake implementation lacks
    # three launch arguments and cannot itself be traced fullgraph. Do not
    # replace that global fake/operator; fallback arithmetic remains native.
    torch.ops.vllm.flash_marlin_down_schedule_sm86(
        *(args[i] for i in (0, 1, 2, 4, 7, 8, 9, 10, 11, 12)), size_m)
    return args[1]


if _ENABLED:
    initialize()
