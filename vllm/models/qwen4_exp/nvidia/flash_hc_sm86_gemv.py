"""Experimental HC-down-only SM86 BF16 GEMV. Disabled unless explicitly enabled."""
import os

import torch
import triton
import triton.language as tl

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.utils.torch_utils import direct_register_custom_op

_ENABLED = os.environ.get('VLLM_FLASH_HC_SM86_GEMV', '0') == '1'
_LOG = init_logger(__name__)


@triton.jit
def _hc_down_kernel(X, W, Y, K: tl.constexpr, BLOCK_K: tl.constexpr):
    # Validated microbenchmark configuration: BN=1, split=1, num_warps=4.
    row = tl.program_id(0) + tl.arange(0, 1)
    ks = tl.arange(0, BLOCK_K)
    x = tl.load(X + ks, ks < K, 0).to(tl.float32)
    w = tl.load(W + row[:, None] * K + ks[None, :],
                ks[None, :] < K, 0).to(tl.float32)
    value = tl.sum(x[None, :] * w, axis=1)
    tl.store(Y + row, value)


def _runtime_eligible(x, weight):
    return (x.ndim == 2 and weight.ndim == 2
            and tuple(x.shape) == (1, 10240)
            and tuple(weight.shape) == (320, 10240)
            and x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
            and x.is_cuda and weight.is_cuda and x.device == weight.device
            and x.is_contiguous() and weight.is_contiguous()
            and torch.cuda.get_device_capability(x.device) == (8, 6))


def _hc_down(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if _ENABLED and _runtime_eligible(x, weight):
        _LOG.info_once('Using opt-in SM86 HC-down kernel: M=1, N=320, K=10240; FP32 accumulation')
        output = x.new_empty((1, 320))
        _hc_down_kernel[(320,)](x, weight, output, 10240, 16384,
                               num_warps=4, enable_fp_fusion=False)
        return output
    return torch.nn.functional.linear(x, weight)


def _hc_down_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(op_name='flash_hc_sm86_gemv', op_func=_hc_down,
                         fake_impl=_hc_down_fake)


def hc_down_projection(layer, x):
    # Do not replace quantized methods, bias handling, subclasses or loaders.
    # M-dependent dispatch remains inside the opaque custom op for compilation.
    weight = getattr(layer, 'weight', None)
    if (_ENABLED and not envs.VLLM_BATCH_INVARIANT
            and type(layer.quant_method) is UnquantizedLinearMethod
            and getattr(layer, 'bias', None) is None
            and weight is not None and tuple(weight.shape) == (320, 10240)):
        return torch.ops.vllm.flash_hc_sm86_gemv(x, weight)
    return layer(x)
