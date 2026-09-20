"""Default-off GDN input M1 kernel candidate. No automatic model integration.

CPU guard tests do not validate the GPU kernel. Actual-weight and serving
validation are required before enabling this in a serving model.
"""
import os
import re

import torch
import triton
import triton.language as tl
import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, UnquantizedLinearMethod
from vllm.utils.torch_utils import direct_register_custom_op

_ENABLED = os.environ.get('VLLM_FLASH_GDN_INPUT_SM86_GEMV', '0') == '1'
_LOG = init_logger(__name__)


def gdn_input_name(name):
    match = re.fullmatch(r'(?:model\.)?(?:language_model\.)?layers\.(\d+)\.linear_attn\.in_proj_qkvz', name)
    return match is not None and 0 <= int(match[1]) < 48 and int(match[1]) % 4 != 3


def tensor_eligible(x, weight):
    return (x.ndim == 2 and tuple(x.shape) == (1, 2560)
            and weight.ndim == 2 and tuple(weight.shape) == (8192, 2560)
            and x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
            and x.is_cuda and weight.is_cuda and x.device == weight.device
            and x.is_contiguous() and weight.is_contiguous())


@triton.jit
def _gdn_input_kernel(X, W, Y):
    # Exact reviewed candidate: BN4, split1, BK4096, warps4, FP32 sum.
    ns = tl.program_id(0) * 4 + tl.arange(0, 4)
    ks = tl.arange(0, 4096)
    x = tl.load(X + ks, ks < 2560, 0).to(tl.float32)
    w = tl.load(W + ns[:, None] * 2560 + ks[None, :],
                (ns[:, None] < 8192) & (ks[None, :] < 2560), 0).to(tl.float32)
    value = tl.sum(w * x[None, :], axis=1)
    tl.store(Y + ns, value, ns < 8192)


def _gdn_input(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # The custom op is internal: unsupported tensors must go through the original
    # method, not a substitute F.linear backend. Fail closed on direct misuse.
    if (not _ENABLED or envs.VLLM_BATCH_INVARIANT or not tensor_eligible(x, weight)
            or torch.cuda.get_device_capability(x.device) != (8, 6)):
        raise RuntimeError('GDN SM86 custom op called outside its guarded scope')
    output = x.new_empty((1, 8192))
    _gdn_input_kernel[(2048,)](x, weight, output, num_warps=4, enable_fp_fusion=False)
    return output


def _gdn_input_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(op_name='flash_gdn_input_sm86', op_func=_gdn_input,
                         fake_impl=_gdn_input_fake)


class GDNInputMethod(UnquantizedLinearMethod):
    def __init__(self, original):
        self.original = original
        self._gemm_impl = original._gemm_impl

    def create_weights(self, *args, **kwargs):
        return self.original.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, *args, **kwargs):
        return self.original.process_weights_after_loading(*args, **kwargs)

    def apply(self, layer, x, bias=None):
        if (_ENABLED and not envs.VLLM_BATCH_INVARIANT and bias is None
                and tensor_eligible(x, layer.weight)
                and torch.cuda.get_device_capability(x.device) == (8, 6)):
            return torch.ops.vllm.flash_gdn_input_sm86(x, layer.weight)
        return self.original.apply(layer, x, bias)


def enable_gdn_input_gemv(module, dtype):
    if (not _ENABLED or envs.VLLM_BATCH_INVARIANT or dtype != torch.bfloat16
            or getattr(module, 'lora_config', None) is not None
            or getattr(getattr(module, 'vllm_config', None), 'lora_config', None) is not None):
        return
    selected = 0
    for name, child in module.named_modules():
        if (not gdn_input_name(name) or type(child) is not MergedColumnParallelLinear
                or type(child.quant_method) is not UnquantizedLinearMethod
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
        child.quant_method = GDNInputMethod(child.quant_method)
        selected += 1
    _LOG.info('Opt-in SM86 GDN-input methods installed: %d eligible local modules', selected)
