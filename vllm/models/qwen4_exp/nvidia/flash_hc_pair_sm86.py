"""Experimental paired HC kernels; preserve native multirow down GEMM."""
import torch

from . import flash_hc_down_inject_multirow as combined
from . import flash_hc_inject_silu_epilogue as epilogue


def candidate(x, down_weight, injection_weight):
    if x.shape[0] == 1:
        return combined.candidate(x, down_weight, injection_weight, num_warps=8, block_n=2)
    projected = torch.nn.functional.linear(x, down_weight)
    return epilogue.candidate(x, projected, injection_weight, num_warps=8)
