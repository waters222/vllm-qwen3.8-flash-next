"""Standalone opt-in HC-down/SiLU probe; no serving hook or weight change."""
import os

import torch
import triton
import triton.language as tl

_ENABLED = os.environ.get('VLLM_FLASH_HC_DOWN_SILU_SM86', '0') == '1'


def eligible(x, weight, hc_count):
    return (_ENABLED and type(hc_count) is int and hc_count == 4
            and x.ndim == weight.ndim == 2
            and tuple(x.shape) == (1, 10240)
            and tuple(weight.shape) == (320, 10240)
            and x.dtype == weight.dtype == torch.bfloat16
            and x.is_cuda and weight.is_cuda and x.device == weight.device
            and x.is_contiguous() and weight.is_contiguous()
            and torch.cuda.get_device_capability(x.device) == (8, 6))


@triton.jit
def _down_silu(X, W, Y, K: tl.constexpr, BLOCK_K: tl.constexpr):
    # Keep the installed HC-down reduction layout and FP32 arithmetic unchanged.
    row = tl.program_id(0) + tl.arange(0, 1)
    ks = tl.arange(0, BLOCK_K)
    x = tl.load(X + ks, ks < K, 0).to(tl.float32)
    w = tl.load(W + row[:, None] * K + ks[None, :],
                ks[None, :] < K, 0).to(tl.float32)
    value = tl.sum(x[None, :] * w, axis=1)
    # The original projection materializes BF16 before the HC SiLU operation.
    rounded = value.to(tl.bfloat16).to(tl.float32)
    scaled = rounded / 4
    activated = scaled * tl.sigmoid(scaled)
    tl.store(Y + row, activated)


def candidate(x, weight, hc_count=4):
    if not eligible(x, weight, hc_count):
        raise ValueError('Require opt-in SM86 contiguous BF16 M1 HC4 exact shapes')
    output = x.new_empty((1, 320))
    _down_silu[(320,)](x, weight, output, 10240, 16384,
                       num_warps=4, enable_fp_fusion=False)
    return output
