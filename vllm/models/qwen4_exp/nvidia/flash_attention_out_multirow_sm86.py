"""Default-off M2--4 attention-output extension; preserve M1 and TP reduction.

BF16 checkpoint weights and operands, FP32 accumulation, BF16 output. The
tensor-core arithmetic is tolerance-validated, not bitwise native-equivalent.
"""
import os

import torch
import triton
import triton.language as tl
import vllm.envs as envs
from vllm.model_executor.layers.linear import RowParallelLinear, UnquantizedLinearMethod
from vllm.utils.torch_utils import direct_register_custom_op

from . import flash_attention_out_sm86 as base

_ENABLED = os.environ.get('VLLM_FLASH_ATTN_OUT_MULTIROW_SM86', '0') == '1'


def tensor_eligible(x, weight):
    return (x.ndim == 2 and tuple(x.shape) in ((2, 3072), (3, 3072), (4, 3072))
            and weight.ndim == 2 and tuple(weight.shape) == (2560, 3072)
            and x.dtype == weight.dtype == torch.bfloat16
            and x.is_cuda and weight.is_cuda and x.device == weight.device
            and x.is_contiguous() and weight.is_contiguous())


@triton.jit
def projection(X, W, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               BN: tl.constexpr):
    ms = tl.arange(0, 16)
    ns = tl.program_id(0) * BN + tl.arange(0, BN)
    ks = tl.arange(0, 64)
    acc = tl.zeros((16, BN), tl.float32)
    for block in range(K // 64):
        kk = block * 64 + ks
        x = tl.load(X + ms[:, None] * K + kk[None, :],
                    (ms[:, None] < M) & (kk[None, :] < K), 0)
        weight = tl.load(W + ns[None, :] * K + kk[:, None],
                         (ns[None, :] < N) & (kk[:, None] < K), 0)
        acc = tl.dot(x, weight, acc)
    tl.store(Y + ms[:, None] * N + ns[None, :], acc,
             (ms[:, None] < M) & (ns[None, :] < N))


def _multirow_output(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if (not _ENABLED or not base._ENABLED or envs.VLLM_BATCH_INVARIANT
            or not tensor_eligible(x, weight)
            or torch.cuda.get_device_capability(x.device) != (8, 6)):
        raise RuntimeError('Attention-output multirow custom op called outside its guarded scope')
    output = x.new_empty((x.shape[0], 2560))
    projection[(80,)](x, weight, output, x.shape[0], 2560, 3072, 32,
                      num_warps=4, num_stages=3)
    return output


def _fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(op_name='flash_attention_out_multirow_sm86',
                         op_func=_multirow_output, fake_impl=_fake)


class AttentionOutputMultirowMethod(UnquantizedLinearMethod):
    def __init__(self, original):
        self.original = original
        self._gemm_impl = original._gemm_impl

    def create_weights(self, *args, **kwargs):
        return self.original.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, *args, **kwargs):
        return self.original.process_weights_after_loading(*args, **kwargs)

    def apply(self, layer, x, bias=None):
        if (_ENABLED and base._ENABLED and not envs.VLLM_BATCH_INVARIANT
                and bias is None and tensor_eligible(x, layer.weight)
                and torch.cuda.get_device_capability(x.device) == (8, 6)):
            return torch.ops.vllm.flash_attention_out_multirow_sm86(x, layer.weight)
        return self.original.apply(layer, x, bias)


def enable_attention_output_multirow(module, dtype):
    if (not _ENABLED or not base._ENABLED or envs.VLLM_BATCH_INVARIANT
            or dtype != torch.bfloat16 or getattr(module, 'lora_config', None) is not None
            or getattr(getattr(module, 'vllm_config', None), 'lora_config', None) is not None):
        return
    for name, child in module.named_modules():
        if (not base.attention_output_name(name) or type(child) is not RowParallelLinear
                or type(child.quant_method) is not base.AttentionOutputMethod
                or getattr(child, 'bias', None) is not None or child.tp_size != 2
                or getattr(child, 'lora_config', None) is not None):
            continue
        weight = getattr(child, 'weight', None)
        if (weight is None or tuple(weight.shape) != (2560, 3072)
                or weight.dtype != torch.bfloat16 or not weight.is_cuda
                or not weight.is_contiguous()
                or torch.cuda.get_device_capability(weight.device) != (8, 6)):
            continue
        child.quant_method = AttentionOutputMultirowMethod(child.quant_method)
