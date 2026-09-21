# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Default-off, exact-shape unpadded TP4 Flash-Next expert-down pilot."""

import hashlib
import os
from pathlib import Path

import torch

import vllm

from .flash_marlin_down_schedule_sm86 import (
    NATIVE_EXTENSION_SHA,
    NATIVE_SCHEMA_SHA,
    PRIVATE_SCHEMA,
)

LIBRARY_SHA = "1d488454923f7f9b80f60172513f286371a0ac40671ea0dacac3f00aaa897a61"
_ENABLED = os.environ.get("VLLM_FLASH_TP4_MARLIN_K32", "0") == "1"
_OWNER = None
_OPERATOR = None


def validate_config(config, environ=None):
    env = os.environ if environ is None else environ
    flag = env.get("VLLM_FLASH_TP4_MARLIN_K32", "0")
    if flag == "0":
        return False
    if flag != "1":
        raise ValueError("VLLM_FLASH_TP4_MARLIN_K32 must be 0 or 1")
    text = config.model_config.hf_text_config
    parallel = config.parallel_config
    if not (
        text.model_type == "qwen4_exp_text"
        and text.hidden_size == 2560
        and text.num_hidden_layers == 48
        and parallel.tensor_parallel_size == 4
        and parallel.pipeline_parallel_size == parallel.data_parallel_size == 1
        and not parallel.enable_expert_parallel
        and config.speculative_config is None
        and config.lora_config is None
        and config.scheduler_config.max_num_batched_tokens <= 2048
        and env.get("VLLM_FLASH_TP4_CHUNKED_REPACK", "0") == "0"
    ):
        raise ValueError("K32 pilot requires Flash-Next TP4/PP1/noEP/noMTP")
    return True


def initialize():
    """Load only the pinned private library, before graph capture."""
    global _OWNER, _OPERATOR
    if _OPERATOR is not None:
        return
    if not _ENABLED or torch.cuda.is_current_stream_capturing():
        raise RuntimeError("K32 must initialize explicitly before graph capture")
    if torch.cuda.get_device_capability() != (8, 6):
        raise RuntimeError("K32 pilot requires SM86")
    path = Path(__file__).with_name("flash_marlin_k32.so").resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != LIBRARY_SHA:
        raise RuntimeError("K32 library pin differs")
    native_path = Path(vllm.__file__).parent / "_moe_C_stable_libtorch.abi3.so"
    if hashlib.sha256(native_path.read_bytes()).hexdigest() != NATIVE_EXTENSION_SHA:
        raise RuntimeError("Native Marlin extension pin differs")
    native_schema = str(torch.ops._moe_C.moe_wna16_marlin_gemm.default._schema)
    if hashlib.sha256(native_schema.encode()).hexdigest() != NATIVE_SCHEMA_SHA:
        raise RuntimeError("Native Marlin schema pin differs")
    _OWNER = torch.library.Library("flash_marlin_k32", "DEF")
    _OWNER.define(PRIVATE_SCHEMA)
    torch.ops.load_library(str(path))
    operator = torch.ops.flash_marlin_k32.gemm.default
    if str(operator._schema).split("(", 1)[1] != native_schema.split("(", 1)[1]:
        raise RuntimeError("Private/native K32 schema differs")
    _OPERATOR = operator


def prepare_unpadded(layer, input_dtype, bits, pack, group, w13, w2):
    from vllm.config import get_current_vllm_config

    if not validate_config(get_current_vllm_config()):
        raise RuntimeError("Unpadded preparation requires the explicit K32 gate")
    config = layer.moe_config
    if not (
        config.tp_size == 4
        and config.ep_size == config.dp_size == 1
        and layer.intermediate_size_per_partition == 160
        and layer.params_dtype == torch.bfloat16
        and bits == 4
        and pack == 8
        and group == 32
        and input_dtype in (None, torch.bfloat16)
        and tuple(w13.shape) == (512, 320, 320)
        and tuple(w2.shape) == (512, 20, 2560)
        and w13.dtype == w2.dtype == torch.int32
        and w13.is_cuda
        and w2.device == w13.device
        and w13.is_contiguous()
        and w2.is_contiguous()
    ):
        raise ValueError("Unpadded preparation requires exact Flash-Next AWQ shards")
    initialize()
    return 160


def gemm(
    input,
    output,
    b_qweight,
    b_bias,
    b_scales,
    a_scales,
    global_scale,
    b_qzeros,
    workspace,
    sorted_token_ids,
    expert_ids,
    num_tokens_past_padded,
    topk_weights,
    moe_block_size,
    top_k,
    mul_topk_weights,
    b_q_type,
    size_m,
    size_n,
    size_k,
    use_atomic_add,
    use_fp32_reduce,
    is_zp_float,
):
    if not _ENABLED or _OPERATOR is None:
        raise RuntimeError("Unpadded expert weights require initialized K32 operator")
    return _OPERATOR(
        input,
        output,
        b_qweight,
        b_bias,
        b_scales,
        a_scales,
        global_scale,
        b_qzeros,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_past_padded,
        topk_weights,
        moe_block_size,
        top_k,
        mul_topk_weights,
        b_q_type.id,
        size_m,
        size_n,
        size_k,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
        32,
        256,
        1,
    )
