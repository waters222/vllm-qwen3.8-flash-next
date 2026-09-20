"""Default-off HC-down/SiLU integration with the exact installed fallback."""
import torch
import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.utils.torch_utils import direct_register_custom_op

from . import flash_hc_down_silu_sm86 as kernel
from . import flash_hc_sm86_gemv as down
from .ops.hc import hc_silu

_LOG = init_logger(__name__)


def _fused(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if envs.VLLM_BATCH_INVARIANT or not down._ENABLED:
        raise RuntimeError('HC-down/SiLU requires ordinary mode and the HC-down opt-in')
    _LOG.info_once('Using opt-in HC-down/SiLU SM86 M1 fusion with BF16 intermediate rounding')
    return kernel.candidate(x, weight, 4)


def _fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(op_name='flash_hc_down_silu_sm86', op_func=_fused, fake_impl=_fake)


def hc_down_silu(layer, x, hc_count):
    if (kernel._ENABLED and down._ENABLED and not envs.VLLM_BATCH_INVARIANT
            and type(layer) is ReplicatedLinear
            and type(layer.quant_method) is UnquantizedLinearMethod
            and getattr(layer, 'bias', None) is None
            and getattr(layer, 'lora_config', None) is None
            and getattr(layer, 'return_bias', True) is False
            and kernel.eligible(x, layer.weight, hc_count)):
        return torch.ops.vllm.flash_hc_down_silu_sm86(x, layer.weight)
    return hc_silu(down.hc_down_projection(layer, x), hc_count)
