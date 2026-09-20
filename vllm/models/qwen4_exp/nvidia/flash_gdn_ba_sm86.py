"""Default-off, exact-shape SM86 BF16 BA projection for TP2 decode M1--4."""
import os
import re

import torch
import triton
import triton.language as tl
import vllm.envs as envs
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, UnquantizedLinearMethod
from vllm.utils.torch_utils import direct_register_custom_op

_ENABLED = os.environ.get('VLLM_FLASH_GDN_BA_SM86', '0') == '1'


def ba_name(name):
    match=re.fullmatch(r'(?:model\.)?(?:language_model\.)?layers\.(\d+)\.linear_attn\.in_proj_ba',name)
    return match is not None and 0<=int(match[1])<48 and int(match[1])%4!=3


def tensor_eligible(x,weight):
    return (x.ndim==2 and 1<=x.shape[0]<=4 and x.shape[1]==2560
            and weight.ndim==2 and tuple(weight.shape)==(48,2560)
            and x.dtype==weight.dtype==torch.bfloat16
            and x.is_cuda and weight.is_cuda and x.device==weight.device
            and x.is_contiguous() and weight.is_contiguous())


@triton.jit
def _ba(X, W, Y, BN: tl.constexpr):
    row = tl.program_id(1)
    ns = tl.program_id(0) * BN + tl.arange(0, BN)
    ks = tl.arange(0, 4096)
    x = tl.load(X + row*2560 + ks, ks < 2560, 0).to(tl.float32)
    weight = tl.load(W + ns[:, None]*2560 + ks[None, :],
                     (ns[:, None] < 48) & (ks[None, :] < 2560), 0).to(tl.float32)
    y = tl.sum(weight*x[None, :], axis=1)
    tl.store(Y + row*48 + ns, y, ns < 48)


def _project(x: torch.Tensor,weight: torch.Tensor)->torch.Tensor:
    if (not _ENABLED or envs.VLLM_BATCH_INVARIANT or not tensor_eligible(x,weight)
            or torch.cuda.get_device_capability(x.device)!=(8,6)):
        raise RuntimeError('BA custom op called outside reviewed SM86 BF16 TP2 scope')
    output=x.new_empty((x.shape[0],48))
    _ba[(24,x.shape[0])](x,weight,output,2,num_warps=4,enable_fp_fusion=False)
    return output


def _fake(x: torch.Tensor,weight: torch.Tensor)->torch.Tensor:
    return x.new_empty((*x.shape[:-1],weight.shape[0]))


direct_register_custom_op(op_name='flash_gdn_ba_sm86',op_func=_project,fake_impl=_fake)


class GDNBaMethod(UnquantizedLinearMethod):
    def __init__(self,original):
        self.original=original
        self._gemm_impl=original._gemm_impl

    def create_weights(self,*args,**kwargs):
        return self.original.create_weights(*args,**kwargs)

    def process_weights_after_loading(self,*args,**kwargs):
        return self.original.process_weights_after_loading(*args,**kwargs)

    def apply(self,layer,x,bias=None):
        if (_ENABLED and not envs.VLLM_BATCH_INVARIANT and bias is None
                and tensor_eligible(x,layer.weight)
                and torch.cuda.get_device_capability(x.device)==(8,6)):
            return torch.ops.vllm.flash_gdn_ba_sm86(x,layer.weight)
        return self.original.apply(layer,x,bias)


def enable_gdn_ba(module,dtype):
    if (not _ENABLED or envs.VLLM_BATCH_INVARIANT or dtype!=torch.bfloat16
            or getattr(module,'lora_config',None) is not None
            or getattr(getattr(module,'vllm_config',None),'lora_config',None) is not None):
        return
    for name,child in module.named_modules():
        if (not ba_name(name) or type(child) is not MergedColumnParallelLinear
                or type(child.quant_method) is not UnquantizedLinearMethod
                or getattr(child,'bias',None) is not None or child.tp_size!=2
                or list(child.output_sizes)!=[48,48]
                or getattr(child,'lora_config',None) is not None):
            continue
        weight=getattr(child,'weight',None)
        if (weight is None or tuple(weight.shape)!=(48,2560)
                or weight.dtype!=torch.bfloat16 or not weight.is_cuda
                or not weight.is_contiguous()
                or torch.cuda.get_device_capability(weight.device)!=(8,6)):
            continue
        child.quant_method=GDNBaMethod(child.quant_method)
