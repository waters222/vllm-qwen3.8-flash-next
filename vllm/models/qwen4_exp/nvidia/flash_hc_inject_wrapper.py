"""Default-off HC injection integration. Preserve unsupported native calls.

This module is prepared for an isolated engine-path test, not deployed serving.
"""
import os

import torch
import vllm.envs as envs
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.utils.torch_utils import direct_register_custom_op

from . import flash_hc_inject_multirow as kernel

_ENABLED = os.environ.get('VLLM_FLASH_HC_INJECT_SM86', '0') == '1'
_MTP_ENABLED = os.environ.get('VLLM_FLASH_HC_INJECT_MTP', '0') == '1'


def eligible(x, weight):
    return (x.ndim == 2
            and (tuple(x.shape) in ((1, 10240), (2, 10240), (4, 10240))
                 or (_MTP_ENABLED and 5 <= x.shape[0] <= 16 and x.shape[1] == 10240))
            and tuple(weight.shape) == (4, 10240)
            and x.dtype == weight.dtype == torch.bfloat16
            and x.is_cuda and weight.is_cuda and x.device == weight.device
            and x.is_contiguous() and weight.is_contiguous()
            and torch.cuda.get_device_capability(x.device) == (8, 6))


def _injection(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if not _ENABLED or envs.VLLM_BATCH_INVARIANT:
        raise RuntimeError('HC injection requires its opt-in and ordinary execution mode')
    if _MTP_ENABLED and x.ndim == 2 and 5 <= x.shape[0] <= 16:
        return kernel.candidate_speculative(x, weight)
    return kernel.candidate(x, weight)


def _fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(op_name='flash_hc_inject_sm86', op_func=_injection, fake_impl=_fake)


def hc_inject(layer, x):
    if (_ENABLED and not envs.VLLM_BATCH_INVARIANT
            and type(layer) is ReplicatedLinear
            and type(layer.quant_method) is UnquantizedLinearMethod
            and getattr(layer, 'bias', None) is None
            and getattr(layer, 'lora_config', None) is None
            and getattr(layer, 'return_bias', True) is False
            and eligible(x, layer.weight)):
        return torch.ops.vllm.flash_hc_inject_sm86(x, layer.weight)
    return layer(x)
