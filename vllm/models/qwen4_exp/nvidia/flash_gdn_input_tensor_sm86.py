"""Default-off SM86 TP2 GDN tensor-core M2--4 projection; preserve M1 exactly.

BF16 operands/checkpoint weights, FP32 accumulation, BF16 output. Arithmetic
matches the qualified BN32/BK128 prototype. Serving installation is a separate
operation; unsupported inputs always delegate to the existing method.
"""
import os

import torch
import triton
import triton.language as tl
import vllm.envs as envs
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, UnquantizedLinearMethod
from vllm.utils.torch_utils import direct_register_custom_op

from . import flash_gdn_input_sm86 as base
from . import flash_gdn_input_multirow_sm86 as multirow

_ENABLED = os.environ.get('VLLM_FLASH_GDN_INPUT_TENSOR_SM86', '0') == '1'
_TP4_ENABLED = os.environ.get('VLLM_FLASH_GDN_TP4_PROJECTION', '0') == '1'
_TP4_ROWS = {4096: (2, 4, 6, 8, 10, 16, 20),
             24: (2, 4, 6, 8, 10, 16)}


def tp4_eligible(x, weight):
    return (x.ndim == weight.ndim == 2 and x.shape[1] == 2560
            and weight.shape[1] == 2560
            and x.shape[0] in _TP4_ROWS.get(weight.shape[0], ())
            and x.dtype == weight.dtype == torch.bfloat16
            and x.is_cuda and weight.is_cuda and x.device == weight.device
            and x.is_contiguous() and weight.is_contiguous())


@triton.jit
def tp4_projection(X, W, Y, M: tl.constexpr, N: tl.constexpr,
                   BM: tl.constexpr):
    ms = tl.arange(0, BM)
    ns = tl.program_id(0) * 32 + tl.arange(0, 32)
    ks = tl.arange(0, 128)
    acc = tl.zeros((BM, 32), tl.float32)
    for block in range(20):
        kk = block * 128 + ks
        x = tl.load(X + ms[:, None] * 2560 + kk[None, :],
                    ms[:, None] < M, 0)
        w = tl.load(W + ns[None, :] * 2560 + kk[:, None],
                    ns[None, :] < N, 0)
        acc = tl.dot(x, w, acc)
    tl.store(Y + ms[:, None] * N + ns[None, :], acc,
             (ms[:, None] < M) & (ns[None, :] < N))


def _tp4_project(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if (not _TP4_ENABLED or envs.VLLM_BATCH_INVARIANT
            or not tp4_eligible(x, weight)
            or torch.cuda.get_device_capability(x.device) != (8, 6)):
        raise RuntimeError('TP4 projection called outside its guarded scope')
    output = x.new_empty((x.shape[0], weight.shape[0]))
    tp4_projection[(triton.cdiv(weight.shape[0], 32),)](
        x, weight, output, x.shape[0], weight.shape[0],
        max(16, triton.next_power_of_2(x.shape[0])), num_warps=4, num_stages=3)
    return output


def tensor_eligible(x, weight):
    return (x.ndim == 2 and tuple(x.shape) in ((2, 2560), (3, 2560), (4, 2560))
            and weight.ndim == 2 and tuple(weight.shape) == (8192, 2560)
            and x.dtype == weight.dtype == torch.bfloat16
            and x.is_cuda and weight.is_cuda and x.device == weight.device
            and x.is_contiguous() and weight.is_contiguous())


@triton.jit
def projection(X, W, Y, M: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    ms = tl.arange(0, 16)
    ns = tl.program_id(0)*BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.zeros((16, BN), tl.float32)
    for block in range(2560//BK):
        kk = block*BK + ks
        x = tl.load(X + ms[:, None]*2560 + kk[None, :], ms[:, None] < M, 0)
        weight = tl.load(W + ns[None, :]*2560 + kk[:, None])
        acc = tl.dot(x, weight, acc)
    tl.store(Y + ms[:, None]*8192 + ns[None, :], acc, ms[:, None] < M)


def _tensor_input(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if (not _ENABLED or not base._ENABLED or not multirow._ENABLED or envs.VLLM_BATCH_INVARIANT
            or not tensor_eligible(x, weight)
            or torch.cuda.get_device_capability(x.device) != (8, 6)):
        raise RuntimeError('GDN tensor custom op called outside its guarded scope')
    output = x.new_empty((x.shape[0], 8192))
    projection[(256,)](x, weight, output, x.shape[0], 32, 128, num_warps=4, num_stages=3)
    return output


def _fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(op_name='flash_gdn_input_tensor_sm86', op_func=_tensor_input, fake_impl=_fake)
direct_register_custom_op(op_name='flash_gdn_tp4_projection',
                          op_func=_tp4_project, fake_impl=_fake)


class GDNTp4ProjectionMethod(UnquantizedLinearMethod):
    def __init__(self, original):
        self.original = original
        self._gemm_impl = original._gemm_impl

    def create_weights(self, *args, **kwargs):
        return self.original.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, *args, **kwargs):
        return self.original.process_weights_after_loading(*args, **kwargs)

    def apply(self, layer, x, bias=None):
        if (_TP4_ENABLED and not envs.VLLM_BATCH_INVARIANT and bias is None
                and layer.tp_size == 4 and tp4_eligible(x, layer.weight)
                and torch.cuda.get_device_capability(x.device) == (8, 6)):
            return torch.ops.vllm.flash_gdn_tp4_projection(x, layer.weight)
        return self.original.apply(layer, x, bias)


def enable_tp4_projections(module, dtype):
    from .flash_gdn_ba_sm86 import ba_name

    if (not _TP4_ENABLED or envs.VLLM_BATCH_INVARIANT
            or dtype != torch.bfloat16
            or getattr(module, 'lora_config', None) is not None
            or getattr(getattr(module, 'vllm_config', None),
                       'lora_config', None) is not None):
        return
    for name, child in module.named_modules():
        if base.gdn_input_name(name):
            shape, outputs = (4096, 2560), [2048, 2048, 6144, 6144]
        elif ba_name(name):
            shape, outputs = (24, 2560), [48, 48]
        else:
            continue
        if (type(child) is not MergedColumnParallelLinear
                or type(child.quant_method) is not UnquantizedLinearMethod
                or child.tp_size != 4 or list(child.output_sizes) != outputs
                or getattr(child, 'bias', None) is not None
                or getattr(child, 'lora_config', None) is not None):
            continue
        weight = getattr(child, 'weight', None)
        if (weight is None or tuple(weight.shape) != shape
                or weight.dtype != torch.bfloat16 or not weight.is_cuda
                or not weight.is_contiguous()
                or torch.cuda.get_device_capability(weight.device) != (8, 6)):
            continue
        child.quant_method = GDNTp4ProjectionMethod(child.quant_method)


class GDNInputTensorMethod(UnquantizedLinearMethod):
    def __init__(self, original):
        self.original = original
        self._gemm_impl = original._gemm_impl

    def create_weights(self, *args, **kwargs):
        return self.original.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, *args, **kwargs):
        return self.original.process_weights_after_loading(*args, **kwargs)

    def apply(self, layer, x, bias=None):
        if (_ENABLED and base._ENABLED and multirow._ENABLED and not envs.VLLM_BATCH_INVARIANT
                and bias is None and tensor_eligible(x, layer.weight)
                and torch.cuda.get_device_capability(x.device) == (8, 6)):
            return torch.ops.vllm.flash_gdn_input_tensor_sm86(x, layer.weight)
        return self.original.apply(layer, x, bias)


def enable_gdn_input_tensor(module, dtype):
    enable_tp4_projections(module, dtype)
    if (not _ENABLED or not base._ENABLED or not multirow._ENABLED or envs.VLLM_BATCH_INVARIANT
            or dtype != torch.bfloat16 or getattr(module, 'lora_config', None) is not None
            or getattr(getattr(module, 'vllm_config', None), 'lora_config', None) is not None):
        return
    for name, child in module.named_modules():
        if (not base.gdn_input_name(name) or type(child) is not MergedColumnParallelLinear
                or type(child.quant_method) is not multirow.GDNInputMultirowMethod
                or getattr(child, 'bias', None) is not None or child.tp_size != 2
                or list(child.output_sizes) != [2048, 2048, 6144, 6144]
                or getattr(child, 'lora_config', None) is not None):
            continue
        weight = getattr(child, 'weight', None)
        if (weight is None or tuple(weight.shape) != (8192, 2560)
                or weight.dtype != torch.bfloat16 or not weight.is_cuda
                or not weight.is_contiguous()
                or torch.cuda.get_device_capability(weight.device) != (8, 6)):
            continue
        child.quant_method = GDNInputTensorMethod(child.quant_method)
