"""Default-off isolated GDN decode metadata candidate; no serving hook installed.

For exact unpadded M1/M2/M4 no-draft decode, fuse dynamic state-index copy and
constant query-offset refresh. Persistent output addresses remain unchanged.
Return None outside the tested contract so callers use the original builder.
"""
from functools import lru_cache
import os

import torch
import triton
import triton.language as tl

_ENABLED=os.environ.get('VLLM_FLASH_GDN_DECODE_METADATA','0')=='1'


@lru_cache(maxsize=16)
def _sm86(device):
    return torch.cuda.get_device_capability(device)==(8,6)


def eligible(builder,m,num_accepted_tokens=None,num_decode_draft_tokens_cpu=None,fast_build=False):
    n=m.num_reqs
    if (not _ENABLED or builder.use_spec_decode or not builder.use_full_cuda_graph
            or builder.vllm_config.cache_config.mamba_cache_mode!='none'
            or num_accepted_tokens is not None or num_decode_draft_tokens_cpu is not None
            or fast_build or n not in (1,2,4) or n>builder.decode_cudagraph_max_bs
            or m.max_query_len!=1 or m.num_actual_tokens!=n):
        return False
    cpu=m.query_start_loc_cpu
    prefilling=m.is_prefilling
    if (cpu.device.type!='cpu' or cpu.dtype!=torch.int32 or tuple(cpu.shape)!=(n+1,)
            or cpu.tolist()!=list(range(n+1)) or prefilling is None
            or prefilling.device.type!='cpu' or prefilling.dtype!=torch.bool
            or tuple(prefilling.shape)!=(n,) or any(prefilling.tolist())):
        return False
    table=m.block_table_tensor
    states=builder.non_spec_state_indices_tensor
    offsets=builder.non_spec_query_start_loc
    query=m.query_start_loc
    tensors=(table,states,offsets,query)
    if (len(table.shape)!=2 or table.shape[0]!=n or table.shape[1]<1
            or len(states.shape)!=1 or states.shape[0]<n
            or len(offsets.shape)!=1 or offsets.shape[0]<n+1
            or tuple(query.shape)!=(n+1,)
            or not all(t.is_cuda and t.device==table.device and t.dtype==torch.int32
                       and t.is_contiguous() and not t.requires_grad for t in tensors)
            or not _sm86(table.device)):
        return False
    addresses=[t.untyped_storage().data_ptr() for t in tensors]
    return len(set(addresses))==len(addresses)


@triton.jit
def _stage(TABLE,STATES,OFFSETS,N:tl.constexpr,STRIDE:tl.constexpr):
    i=tl.arange(0,32)
    state=tl.load(TABLE+i*STRIDE,i<N,other=0)
    tl.store(STATES+i,state,i<N)
    tl.store(OFFSETS+i,i,i<=N)


def try_build(builder,m,metadata_cls,num_accepted_tokens=None,num_decode_draft_tokens_cpu=None,fast_build=False):
    if not eligible(builder,m,num_accepted_tokens,num_decode_draft_tokens_cpu,fast_build):
        return None
    n=m.num_reqs
    _stage[(1,)](m.block_table_tensor,builder.non_spec_state_indices_tensor,
                  builder.non_spec_query_start_loc,n,m.block_table_tensor.stride(0),num_warps=1)
    return metadata_cls(num_prefills=0,num_prefill_tokens=0,num_decodes=n,
        num_decode_tokens=n,num_spec_decodes=0,num_spec_decode_tokens=0,num_actual_tokens=n,
        non_spec_query_start_loc=builder.non_spec_query_start_loc[:n+1],
        non_spec_state_indices_tensor=builder.non_spec_state_indices_tensor[:n])


@triton.jit
def _stage_many(TABLES,STATES,OFFSETS,N:tl.constexpr,STRIDES:tl.constexpr):
    group=tl.program_id(0)
    i=tl.arange(0,32)
    for g in tl.static_range(len(TABLES)):
        if group==g:
            state=tl.load(TABLES[g]+i*STRIDES[g],i<N,other=0)
            tl.store(STATES[g]+i,state,i<N)
            tl.store(OFFSETS[g]+i,i,i<=N)


def try_build_many(entries,metadata_cls):
    """Isolated batch prototype: validate every group before touching any output.

    No persistent Python metadata cache or device-pointer cache. Callers must
    supply only exact GDN builders. Unsupported input returns None for the
    entire batch; partial staging is forbidden. No engine hook is installed.
    """
    if not 1<=len(entries)<=24:
        return None
    n=entries[0][1].num_reqs
    if any(m.num_reqs!=n or not eligible(b,m) for b,m in entries):
        return None
    tables=tuple(m.block_table_tensor for b,m in entries)
    states=tuple(b.non_spec_state_indices_tensor for b,m in entries)
    offsets=tuple(b.non_spec_query_start_loc for b,m in entries)
    if any(t.device!=tables[0].device for t in (*tables,*states,*offsets)):
        return None
    outputs=[t.untyped_storage().data_ptr() for t in (*states,*offsets)]
    inputs={t.untyped_storage().data_ptr() for t in tables}
    inputs.update(m.query_start_loc.untyped_storage().data_ptr() for b,m in entries)
    if len(set(outputs))!=len(outputs) or inputs.intersection(outputs):
        return None
    _stage_many[(len(entries),)](tables,states,offsets,n,
                                tuple(t.stride(0) for t in tables),num_warps=1)
    return [metadata_cls(num_prefills=0,num_prefill_tokens=0,num_decodes=n,
        num_decode_tokens=n,num_spec_decodes=0,num_spec_decode_tokens=0,num_actual_tokens=n,
        non_spec_query_start_loc=b.non_spec_query_start_loc[:n+1],
        non_spec_state_indices_tensor=b.non_spec_state_indices_tensor[:n]) for b,m in entries]
