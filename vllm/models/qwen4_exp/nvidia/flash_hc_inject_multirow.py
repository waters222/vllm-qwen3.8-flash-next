"""Experimental BF16 HC injection for M1/M2/M4, N4/K10240. No serving hook.

Split the long dot products across the GPU; accumulate in FP32 and materialize
BF16 once. Original weights and quantization are unchanged.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _partial(X, W, P, M: tl.constexpr, BK: tl.constexpr, SPLITS: tl.constexpr):
    split = tl.program_id(0)
    ks = split * BK + tl.arange(0, BK)
    ns = tl.arange(0, 4)
    weight = tl.load(W + ns[:, None] * 10240 + ks[None, :], ks[None, :] < 10240, 0).to(tl.float32)
    for row in tl.static_range(M):
        x = tl.load(X + row * 10240 + ks, ks < 10240, 0).to(tl.float32)
        value = tl.sum(weight * x[None, :], axis=1)
        tl.store(P + (row * 4 + ns) * SPLITS + split, value)


@triton.jit
def _finish(P, Y, M: tl.constexpr, SPLITS: tl.constexpr, BS: tl.constexpr):
    row = tl.program_id(0)
    ns = tl.arange(0, 4)
    split = tl.arange(0, BS)
    value = tl.load(P + (row * 4 + ns[:, None]) * SPLITS + split[None, :], split[None, :] < SPLITS, 0)
    tl.store(Y + row * 4 + ns, tl.sum(value, axis=1))


def candidate(x, weight, block_k=128):
    if not (tuple(x.shape) in ((1, 10240), (2, 10240), (4, 10240)) and tuple(weight.shape) == (4, 10240)
            and block_k in (128, 256, 512) and x.dtype == weight.dtype == torch.bfloat16
            and x.is_cuda and weight.is_cuda and x.device == weight.device
            and x.is_contiguous() and weight.is_contiguous()
            and torch.cuda.get_device_capability(x.device) == (8, 6)):
        raise ValueError('Require SM86 contiguous BF16 HC injection M1/M2/M4, N4/K10240')
    rows, splits = x.shape[0], triton.cdiv(10240, block_k)
    partial = torch.empty((rows, 4, splits), device=x.device, dtype=torch.float32)
    output = torch.empty((rows, 4), device=x.device, dtype=torch.bfloat16)
    _partial[(splits,)](x, weight, partial, rows, block_k, splits, num_warps=4, enable_fp_fusion=False)
    _finish[(rows,)](partial, output, rows, splits, triton.next_power_of_2(splits),
                      num_warps=4, enable_fp_fusion=False)
    return output


@triton.jit
def _partial_rows(X, W, P, BK: tl.constexpr, SPLITS: tl.constexpr):
    split, row = tl.program_id(0), tl.program_id(1)
    ks = split * BK + tl.arange(0, BK)
    ns = tl.arange(0, 4)
    weight = tl.load(
        W + ns[:, None] * 10240 + ks[None, :], ks[None, :] < 10240, 0
    ).to(tl.float32)
    x = tl.load(X + row * 10240 + ks, ks < 10240, 0).to(tl.float32)
    value = tl.sum(weight * x[None, :], axis=1)
    tl.store(P + (row * 4 + ns) * SPLITS + split, value)


def candidate_speculative(x, weight):
    """Cover the native M5--16 skinny-GEMM cliff without changing M1--4."""
    if not (
        x.ndim == 2
        and 5 <= x.shape[0] <= 16
        and x.shape[1] == 10240
        and tuple(weight.shape) == (4, 10240)
        and x.dtype == weight.dtype == torch.bfloat16
        and x.is_cuda and weight.is_cuda and x.device == weight.device
        and x.is_contiguous() and weight.is_contiguous()
        and torch.cuda.get_device_capability(x.device) == (8, 6)
    ):
        raise ValueError('Require SM86 contiguous BF16 HC injection M5--16, N4/K10240')
    rows, splits = x.shape[0], 80
    partial = torch.empty((rows, 4, splits), device=x.device, dtype=torch.float32)
    output = torch.empty((rows, 4), device=x.device, dtype=torch.bfloat16)
    _partial_rows[(splits, rows)](
        x, weight, partial, 128, splits, num_warps=4, enable_fp_fusion=False
    )
    _finish[(rows,)](
        partial, output, rows, splits, 128, num_warps=4, enable_fp_fusion=False
    )
    return output
