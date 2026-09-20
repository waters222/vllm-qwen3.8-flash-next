"""Opt-in TP2 attention-output M1 GEMV; leaves row-parallel reductions intact."""
import os
import re

import torch
import triton
import triton.language as tl
import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import RowParallelLinear, UnquantizedLinearMethod
from vllm.utils.torch_utils import direct_register_custom_op

_ENABLED = os.environ.get('VLLM_FLASH_ATTN_OUT_SM86_GEMV', '0') == '1'
_LOG = init_logger(__name__)


def attention_output_name(name):
    return ('mtp' not in name.split('.') and re.search(
        r'(?:^|\.)layers\.\d+\.(?:linear_attn\.out_proj|self_attn\.o_proj)$', name) is not None)


def tensor_eligible(x, weight):
    return (x.ndim == 2 and tuple(x.shape) == (1, 3072)
            and weight.ndim == 2 and tuple(weight.shape) == (2560, 3072)
            and x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
            and x.is_cuda and weight.is_cuda and x.device == weight.device
            and x.is_contiguous() and weight.is_contiguous())


@triton.jit
def _attention_out_kernel(X, W, Y):
    # Exact microbenchmark configuration: BN=1, split=1, warps=8.
    ns = tl.program_id(0) + tl.arange(0, 1)
    ks = tl.arange(0, 4096)
    x = tl.load(X + ks, ks < 3072, 0).to(tl.float32)
    w = tl.load(W + ns[:, None] * 3072 + ks[None, :],
                ks[None, :] < 3072, 0).to(tl.float32)
    value = tl.sum(w * x[None, :], axis=1)
    tl.store(Y + ns, value)


def _attention_out(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if (_ENABLED and not envs.VLLM_BATCH_INVARIANT and tensor_eligible(x, weight)
            and torch.cuda.get_device_capability(x.device) == (8, 6)):
        _LOG.info_once('Using opt-in SM86 attention-output kernel: M=1, N=2560, K=3072; FP32 accumulation')
        output = x.new_empty((1, 2560))
        _attention_out_kernel[(2560,)](x, weight, output, num_warps=8, enable_fp_fusion=False)
        return output
    return torch.nn.functional.linear(x, weight)


def _attention_out_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(op_name='flash_attention_out_sm86', op_func=_attention_out,
                         fake_impl=_attention_out_fake)


class AttentionOutputMethod(UnquantizedLinearMethod):
    def __init__(self, original):
        # Preserve the exact existing backend and method for fallback/weight hooks.
        self.original = original
        self._gemm_impl = original._gemm_impl

    def apply(self, layer, x, bias=None):
        if (_ENABLED and not envs.VLLM_BATCH_INVARIANT and bias is None
                and tensor_eligible(x, layer.weight)):
            return torch.ops.vllm.flash_attention_out_sm86(x, layer.weight)
        return self.original.apply(layer, x, bias)


def enable_attention_output_gemv(module, dtype):
    if (not _ENABLED or envs.VLLM_BATCH_INVARIANT or dtype != torch.bfloat16
            or torch.cuda.get_device_capability() != (8, 6)):
        return
    selected = 0
    for name, child in module.named_modules():
        if (not attention_output_name(name) or type(child) is not RowParallelLinear
                or type(child.quant_method) is not UnquantizedLinearMethod
                or getattr(child, 'bias', None) is not None or child.tp_size != 2):
            continue
        weight = getattr(child, 'weight', None)
        if (weight is None or tuple(weight.shape) != (2560, 3072)
                or weight.dtype != torch.bfloat16 or not weight.is_cuda
                or not weight.is_contiguous()):
            continue
        child.quant_method = AttentionOutputMethod(child.quant_method)
        selected += 1
    _LOG.info('Opt-in SM86 attention-output methods installed: %d eligible local modules', selected)
