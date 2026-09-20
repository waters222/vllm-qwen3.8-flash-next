"""Default-off BF16 shared-expert gate/sigmoid/scale fusion for SM86 TP2 M1--4."""
import os
import re

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
import vllm.envs as envs
from vllm.model_executor.layers.linear import (
    ReplicatedLinear, MergedColumnParallelLinear, RowParallelLinear, UnquantizedLinearMethod,
)
from vllm.utils.torch_utils import direct_register_custom_op

_ENABLED=os.environ.get('VLLM_FLASH_SHARED_EPILOGUE_SM86','0')=='1'


def shared_name(name):
    match=re.fullmatch(r'(?:model\.)?(?:language_model\.)?layers\.(\d+)\.mlp\.shared_expert',name)
    return match is not None and 0<=int(match[1])<48


def tensor_eligible(x,weight,out):
    return (x.ndim==2 and 1<=x.shape[0]<=4 and x.shape[1]==2560
            and tuple(weight.shape)==(1,2560) and out.shape==x.shape
            and all(t.dtype==torch.bfloat16 and t.is_cuda and t.is_contiguous()
                    and t.device==x.device for t in (x,weight,out))
            and torch.cuda.get_device_capability(x.device)==(8,6))


@triton.jit
def fused(X,W,O,Y):
    row=tl.program_id(0)
    ks=tl.arange(0,4096)
    x=tl.load(X+row*2560+ks,ks<2560,0).to(tl.float32)
    w=tl.load(W+ks,ks<2560,0).to(tl.float32)
    gate=tl.sum(x*w,axis=0).to(tl.bfloat16).to(tl.float32)
    scale=tl.div_rn(1.,1.+libdevice.exp(-gate)).to(tl.bfloat16).to(tl.float32)
    out=tl.load(O+row*2560+ks,ks<2560,0).to(tl.float32)
    tl.store(Y+row*2560+ks,out*scale,ks<2560)


def _epilogue(x:torch.Tensor,weight:torch.Tensor,out:torch.Tensor)->torch.Tensor:
    if not _ENABLED or envs.VLLM_BATCH_INVARIANT or not tensor_eligible(x,weight,out):
        raise RuntimeError('Shared epilogue custom op called outside reviewed SM86 BF16 M1--4 scope')
    result=torch.empty_like(out)
    fused[(x.shape[0],)](x,weight,out,result,num_warps=4,enable_fp_fusion=False)
    return result


def _fake(x:torch.Tensor,weight:torch.Tensor,out:torch.Tensor)->torch.Tensor:
    return torch.empty_like(out)


direct_register_custom_op(op_name='flash_shared_epilogue_sm86',op_func=_epilogue,fake_impl=_fake)


def supported_gate(gate):
    return (type(gate) is ReplicatedLinear and type(gate.quant_method) is UnquantizedLinearMethod
            and getattr(gate,'bias',None) is None and getattr(gate,'lora_config',None) is None
            and getattr(gate,'return_bias',True) is True)


def shared_gate_scale(owner,x,out):
    gate=owner.expert_gate
    if (_ENABLED and not envs.VLLM_BATCH_INVARIANT
            and getattr(owner,'_flash_shared_epilogue_eligible',False)
            and supported_gate(gate) and tensor_eligible(x,gate.weight,out)):
        return torch.ops.vllm.flash_shared_epilogue_sm86(x,gate.weight,out)
    # Preserve the installed gate operation and both native BF16 intermediates.
    return torch.nn.functional.sigmoid(gate(x)[0])*out


def enable_shared_epilogue(module,dtype):
    if (not _ENABLED or envs.VLLM_BATCH_INVARIANT or dtype!=torch.bfloat16
            or getattr(module,'lora_config',None) is not None
            or getattr(getattr(module,'vllm_config',None),'lora_config',None) is not None):
        return
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
    for name,child in module.named_modules():
        if not shared_name(name) or type(child) is not Qwen2MoeMLP:
            continue
        up,down,gate=child.gate_up_proj,child.down_proj,child.expert_gate
        if (type(up) is not MergedColumnParallelLinear or type(down) is not RowParallelLinear
                or up.tp_size!=2 or down.tp_size!=2 or down.reduce_results
                or list(up.output_sizes)!=[640,640] or not supported_gate(gate)
                or getattr(child,'lora_config',None) is not None):
            continue
        if any(type(layer.quant_method) is not UnquantizedLinearMethod
               or getattr(layer,'bias',None) is not None or getattr(layer,'lora_config',None) is not None
               for layer in (up,down)):
            continue
        if any(tuple(layer.weight.shape)!=shape or layer.weight.dtype!=torch.bfloat16
               or not layer.weight.is_cuda or not layer.weight.is_contiguous()
               or layer.weight.device!=gate.weight.device
               for layer,shape in ((gate,(1,2560)),(up,(640,2560)),(down,(2560,320)))):
            continue
        if torch.cuda.get_device_capability(gate.weight.device)!=(8,6):
            continue
        child._flash_shared_epilogue_eligible=True
