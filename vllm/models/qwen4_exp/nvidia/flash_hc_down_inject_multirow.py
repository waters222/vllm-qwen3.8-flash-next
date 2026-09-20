"""Unvalidated one-layer experiment: one launch for HC injection and down/SiLU.

No serving integration. Original weights stay separate and unchanged. M1/M2/M4
reuse each weight row across input rows. GPU correctness and timing are pending.
"""
import torch
import triton
import triton.language as tl

COMPILED_RESOURCES = {}


@triton.jit
def _combined(X, DOWN, INJECT, Y, I, M: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr):
    # Schedule the four injection rows first, not as a tail after the bulk GEMV.
    output_row = tl.program_id(0)
    first_row = tl.program_id(1) * M
    ks = tl.arange(0, BK)
    if output_row < 4:
        inject_weight = tl.load(INJECT + output_row * 10240 + ks, ks < 10240, 0).to(tl.float32)
        for local_row in tl.static_range(M):
            row = first_row + local_row
            ix = tl.load(X + row * 10240 + ks, ks < 10240, 0).to(tl.float32)
            projected_injection = tl.sum(inject_weight * ix, axis=0)
            tl.store(I + row * 4 + output_row, projected_injection)
    else:
        ns = (output_row - 4) * BN + tl.arange(0, BN)
        down_weight = tl.load(DOWN + ns[:, None] * 10240 + ks[None, :],
                              ks[None, :] < 10240, 0).to(tl.float32)
        for local_row in tl.static_range(M):
            row = first_row + local_row
            dx = tl.load(X + row * 10240 + ks, ks < 10240, 0).to(tl.float32)
            projected_down = tl.sum(down_weight * dx[None, :], axis=1)
            # Preserve the native BF16 projection boundary before HC4 SiLU.
            rounded = projected_down.to(tl.bfloat16).to(tl.float32)
            scaled = rounded / 4
            tl.store(Y + row * 320 + ns, scaled * tl.sigmoid(scaled))


def candidate(x, down_weight, inject_weight, num_warps=4, group_rows=None, block_n=1):
    tensors = (x, down_weight, inject_weight)
    if not (tuple(x.shape) in ((1, 10240), (2, 10240), (4, 10240))
            and tuple(down_weight.shape) == (320, 10240)
            and tuple(inject_weight.shape) == (4, 10240)
            and all(t.dtype == torch.bfloat16 and t.is_cuda and t.is_contiguous()
                    and t.device == x.device for t in tensors)
            and torch.cuda.get_device_capability(x.device) == (8, 6)):
        raise ValueError('Require SM86 contiguous BF16 M1/M2/M4, HC4 down/injection exact shapes')
    rows = x.shape[0]
    group_rows = rows if group_rows is None else group_rows
    if (type(num_warps) is not int or num_warps not in (4, 8, 16)
            or type(group_rows) is not int or group_rows not in (1, 2, 4)
            or group_rows > rows or rows % group_rows
            or type(block_n) is not int or block_n not in (1, 2, 4)):
        raise ValueError('Require 4/8/16 warps, exact 1/2/4-row partition and BN1/2/4')
    down = torch.empty((rows, 320), device=x.device, dtype=x.dtype)
    injection = torch.empty((rows, 4), device=x.device, dtype=x.dtype)
    compiled = _combined[(4 + 320 // block_n, rows // group_rows)](x, down_weight, inject_weight, down, injection,
                                 group_rows, 16384, block_n, num_warps=num_warps, enable_fp_fusion=False)
    COMPILED_RESOURCES[f'{rows}:{group_rows}:{num_warps}:{block_n}'] = dict(registers_per_thread=getattr(compiled, 'n_regs', None),
        spill_count=getattr(compiled, 'n_spills', None),
        shared_bytes=getattr(getattr(compiled, 'metadata', None), 'shared', None))
    return down, injection
