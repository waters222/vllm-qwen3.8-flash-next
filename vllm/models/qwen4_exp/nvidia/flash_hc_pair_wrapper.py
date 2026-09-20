"""Default-off paired HC integration. No checkpoint changes or silent fallback changes."""
import os

import torch
import vllm.envs as envs
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.utils.torch_utils import direct_register_custom_op

from . import flash_hc_pair_sm86 as kernel
from . import flash_hc_sm86_gemv as down
from . import flash_hc_down_silu_wrapper as down_silu
from . import flash_hc_inject_wrapper as injection

_ENABLED = os.environ.get('VLLM_FLASH_HC_PAIR_SM86', '0') == '1'
_M3_ENABLED = os.environ.get('VLLM_FLASH_HC_PAIR_M3', '0') == '1'


def runtime_enabled():
    return (_ENABLED and not envs.VLLM_BATCH_INVARIANT and down._ENABLED
            and down_silu.kernel._ENABLED and injection._ENABLED)


def eligible(x, down_weight, injection_weight):
    tensors = (x, down_weight, injection_weight)
    return (x.ndim == 2
            and (tuple(x.shape) in ((1, 10240), (2, 10240), (4, 10240))
                 or (_M3_ENABLED and tuple(x.shape) == (3, 10240)))
            and tuple(down_weight.shape) == (320, 10240)
            and tuple(injection_weight.shape) == (4, 10240)
            and all(t.dtype == torch.bfloat16 and t.is_cuda and t.is_contiguous()
                    and t.device == x.device for t in tensors)
            and torch.cuda.get_device_capability(x.device) == (8, 6))


def supported_layer(layer):
    return (type(layer) is ReplicatedLinear and type(layer.quant_method) is UnquantizedLinearMethod
            and getattr(layer, 'bias', None) is None
            and getattr(layer, 'lora_config', None) is None
            and getattr(layer, 'return_bias', True) is False)


def _paired(x: torch.Tensor, down_weight: torch.Tensor,
            injection_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not runtime_enabled():
        raise RuntimeError('Paired HC requires ordinary mode and all existing HC opt-ins')
    if x.shape[0] == 3 and not _M3_ENABLED:
        raise RuntimeError('Paired HC M3 requires its explicit opt-in')
    return kernel.candidate(x, down_weight, injection_weight)


def _fake(x: torch.Tensor, down_weight: torch.Tensor,
          injection_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x.new_empty((*x.shape[:-1], down_weight.shape[0])), x.new_empty((*x.shape[:-1], injection_weight.shape[0]))


direct_register_custom_op(op_name='flash_hc_pair_sm86', op_func=_paired, fake_impl=_fake)


def hc_down_inject(down_layer, injection_layer, x, hc_count):
    if (runtime_enabled() and type(hc_count) is int and hc_count == 4
            and supported_layer(down_layer) and supported_layer(injection_layer)
            and eligible(x, down_layer.weight, injection_layer.weight)):
        return torch.ops.vllm.flash_hc_pair_sm86(x, down_layer.weight, injection_layer.weight)
    # Match the previous order and exact wrappers, including unsupported
    # shapes, missing injection, disabled flags and special execution modes.
    logits = injection.hc_inject(injection_layer, x) if injection_layer is not None else None
    return down_silu.hc_down_silu(down_layer, x, hc_count), logits
