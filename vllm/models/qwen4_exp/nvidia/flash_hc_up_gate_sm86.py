"""Standalone, default-off HC up + gate mix candidate; no serving integration."""
import os
import torch
import triton
import triton.language as tl

_ENABLED=os.environ.get('VLLM_FLASH_HC_UP_GATE_SM86','0')=='1'


def eligible(activation,weight,residual):
    return (_ENABLED and activation.ndim==weight.ndim==residual.ndim==2
        and tuple(activation.shape)==(1,320) and tuple(weight.shape)==(10240,320)
        and tuple(residual.shape)==(1,10240)
        and all(t.dtype==torch.bfloat16 and t.is_cuda and t.is_contiguous()
                for t in (activation,weight,residual))
        and activation.device==weight.device==residual.device
        and torch.cuda.get_device_capability(activation.device)==(8,6))


@triton.jit
def _up_gate(A,W,R,Y,BN:tl.constexpr):
    features=tl.program_id(0)*BN+tl.arange(0,BN)
    k=tl.arange(0,512)
    a=tl.load(A+k,k<320,0).to(tl.float32)
    acc=tl.zeros((BN,),tl.float32)
    # Keep exactly the HC kernel's sequential four-stream sum, not a tree sum.
    for stream in tl.static_range(4):
        rows=stream*2560+features
        w=tl.load(W+rows[:,None]*320+k[None,:],
                  (features[:,None]<2560)&(k[None,:]<320),0).to(tl.float32)
        projected=tl.sum(w*a[None,:],axis=1)
        # Materialized projection is BF16 in the original two-operation path.
        rounded=projected.to(tl.bfloat16).to(tl.float32)
        residual=tl.load(R+rows,features<2560,0).to(tl.float32)
        acc+=tl.sigmoid(rounded)*residual
    acc/=4
    tl.store(Y+features,acc,features<2560)


def candidate(activation,weight,residual,bn=4):
    if type(bn) is not int or bn not in (1,4):
        raise ValueError('Only bounded BN1 and BN4 candidates are supported')
    if not eligible(activation,weight,residual):
        raise ValueError('HC up-gate requires opt-in SM86 contiguous BF16 M1 exact shapes')
    output=activation.new_empty((1,2560))
    _up_gate[(triton.cdiv(2560,bn),)](activation,weight,residual,output,bn,
                                    num_warps=4,enable_fp_fusion=False)
    return output
