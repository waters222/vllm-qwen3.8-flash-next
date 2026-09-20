"""Default-off HC up/gate integration. Original two-op fallback is retained."""
import torch
import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.utils.torch_utils import direct_register_custom_op

from . import flash_hc_up_gate_sm86 as kernel
from .ops.hc import hc_gate_mix

_LOG=init_logger(__name__)


def _fused(activation: torch.Tensor, weight: torch.Tensor,
           residual: torch.Tensor) -> torch.Tensor:
    if envs.VLLM_BATCH_INVARIANT:
        raise RuntimeError('HC up/gate custom op is unavailable in batch-invariant mode')
    _LOG.info_once('Using opt-in HC up/gate SM86 M1 fused kernel, BN4')
    return kernel.candidate(activation,weight,residual,bn=4)


def _fake(activation: torch.Tensor, weight: torch.Tensor,
          residual: torch.Tensor) -> torch.Tensor:
    return residual.new_empty((activation.shape[0],2560))


direct_register_custom_op(op_name='flash_hc_up_gate_sm86',op_func=_fused,fake_impl=_fake)


def hc_up_gate(layer,activation,residual,hc_count):
    if (kernel._ENABLED and not envs.VLLM_BATCH_INVARIANT and hc_count==4
            and type(layer) is ReplicatedLinear
            and type(layer.quant_method) is UnquantizedLinearMethod
            and getattr(layer,'bias',None) is None
            and getattr(layer,'lora_config',None) is None
            and getattr(layer,'return_bias',True) is False
            and kernel.eligible(activation,layer.weight,residual)):
        return torch.ops.vllm.flash_hc_up_gate_sm86(activation,layer.weight,residual)
    # Do not replace the installed backend or its numerical policy on fallback.
    gate=layer(activation)
    return hc_gate_mix(residual,gate,hc_count)
