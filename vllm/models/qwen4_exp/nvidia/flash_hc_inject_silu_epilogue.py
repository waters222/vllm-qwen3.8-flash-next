"""One-layer prototype: injection plus SiLU of an unchanged native projection.

The caller computes native BF16 down GEMM first. Fuse only its pointwise SiLU
with the independent injection projection. Original weights remain unchanged.
"""
import torch
import triton
import triton.language as tl

COMPILED_RESOURCES = {}


@triton.jit
def _epilogue(X, W, D, Y, I, M: tl.constexpr):
    pid = tl.program_id(0)
    if pid < 4 * M:
        # Independent row/head CTAs avoid serializing all M dot products on
        # only four SMs. The read-only injection weights are just 80 KiB.
        row = pid // 4
        head = pid % 4
        ks = tl.arange(0, 16384)
        weight = tl.load(W + head * 10240 + ks, ks < 10240, 0).to(tl.float32)
        x = tl.load(X + row * 10240 + ks, ks < 10240, 0).to(tl.float32)
        tl.store(I + row * 4 + head, tl.sum(weight * x, axis=0))
    else:
        row = pid - 4 * M
        ns = tl.arange(0, 512)
        value = tl.load(D + row * 320 + ns, ns < 320, 0).to(tl.float32) / 4
        tl.store(Y + row * 320 + ns, value * tl.sigmoid(value), ns < 320)


def candidate(x, projected_down, inject_weight, num_warps=8):
    tensors = (x, projected_down, inject_weight)
    if not (tuple(x.shape) in ((2, 10240), (3, 10240), (4, 10240))
            and tuple(projected_down.shape) == (x.shape[0], 320)
            and tuple(inject_weight.shape) == (4, 10240)
            and all(t.dtype == torch.bfloat16 and t.is_cuda and t.is_contiguous()
                    and t.device == x.device for t in tensors)
            and torch.cuda.get_device_capability(x.device) == (8, 6)
            and type(num_warps) is int and num_warps in (4, 8, 16)):
        raise ValueError('Require SM86 contiguous BF16 M2/M3/M4 native down output and HC4 injection')
    rows = x.shape[0]
    down = torch.empty((rows, 320), device=x.device, dtype=x.dtype)
    injection = torch.empty((rows, 4), device=x.device, dtype=x.dtype)
    compiled = _epilogue[(5 * rows,)](x, inject_weight, projected_down, down, injection,
                                    rows, num_warps=num_warps, enable_fp_fusion=False)
    COMPILED_RESOURCES[f'{rows}:{num_warps}'] = dict(
        registers_per_thread=getattr(compiled, 'n_regs', None),
        spill_count=getattr(compiled, 'n_spills', None),
        shared_bytes=getattr(getattr(compiled, 'metadata', None), 'shared', None))
    return down, injection
