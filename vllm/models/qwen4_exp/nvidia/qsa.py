# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA QSA owner with Triton kernels."""

from __future__ import annotations

import os
from typing import ClassVar, cast

import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import (
    set_default_quant_scales,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding, get_rope
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)
from vllm.utils.torch_utils import (
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)

from ..common.qsa_cache import QSAForwardMetadata
from . import model
from .indexer_qsa import QSAIndexer


class Qwen4ExpQSAMetadataBuilder(FlashAttentionMetadataBuilder):
    """Flash metadata supporting uniform decode and target-verify graphs."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class Qwen4ExpQSAFlashAttentionBackend(FlashAttentionBackend):
    """FullAttentionSpec backend used by the merged QSA owner."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "QWEN4_EXP_QSA_TRITON"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # QSA consumes manager pages directly and does not use FA4 paged attention.
        return [MultipleOf(16)]

    @staticmethod
    def get_impl_cls() -> type[Qwen4ExpQSAFlashAttentionImpl]:
        return Qwen4ExpQSAFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[Qwen4ExpQSAMetadataBuilder]:
        return Qwen4ExpQSAMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False


_QSA_STAGE_MIN_ROWS = 64


def _qsa_staged_prefill(
    layer,
    query,
    kv_cache,
    logical_indices,
    attn_metadata,
    token_to_req,
    num_tokens,
    use_prefill_config,
    output,
):
    """Serve prefill rows from host-resident K/V through a staging arena.

    Returns the ``(start, end)`` row range that was handled, or None to leave
    the whole batch to the direct path. Only the contiguous rows of a single
    request can be staged, so a mixed batch is served for its largest request
    and the remaining rows fall back to the direct path.
    """
    if not getattr(layer, "_qsa_kv_offload", False):
        return None
    min_rows = getattr(layer, "_qsa_stage_min_rows", 0)
    if min_rows <= 0 or num_tokens < min_rows:
        return None
    starts = getattr(attn_metadata, "query_start_loc", None)
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    if starts is None or seq_lens is None:
        return None
    num_requests = int(seq_lens.shape[0])
    if num_requests < 1 or starts.shape[0] < num_requests + 1:
        return None
    offsets = starts[: num_requests + 1].tolist()
    lengths = seq_lens[:num_requests].tolist()
    best = max(range(num_requests), key=lambda i: offsets[i + 1] - offsets[i])
    start, end = offsets[best], offsets[best + 1]
    if end - start < min_rows or end > num_tokens:
        return None

    from .ops.qsa import qsa_get_staging_arena, qsa_sparse_paged_attention_staged

    arena = qsa_get_staging_arena(
        kv_cache, getattr(layer, "_qsa_stage_arena_bytes", 0)
    )
    if arena is None:
        return None
    page_size = kv_cache.shape[2]
    max_logical_page = (max(int(lengths[best]), 1) - 1) // page_size
    if max_logical_page >= attn_metadata.block_table.shape[1]:
        return None
    rows = torch.arange(start, end, device=query.device, dtype=torch.int64)
    qsa_sparse_paged_attention_staged(
        query[:num_tokens],
        kv_cache,
        logical_indices,
        attn_metadata.block_table[best],
        rows,
        max_logical_page,
        arena,
        use_prefill_config,
        output[:num_tokens],
    )
    return start, end


class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
    """Run paged sparse GQA with the QSA Triton kernel."""

    supports_dcp: bool = False
    supports_pcp: bool = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not is_flash_attn_varlen_func_available():
            raise NotImplementedError("Qwen4Exp QSA requires FlashAttention")
        if self.dcp_world_size != 1:
            raise NotImplementedError(
                "Qwen4Exp QSA does not support decode context parallelism"
            )
        if self.kv_cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError("Qwen4Exp QSA requires a BF16 main KV cache")
        self.supports_quant_query_input = False

    def forward_qsa(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        token_to_req: torch.Tensor,
        use_prefill_config: bool,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del key, value
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("QSA does not support fused output quantization")
        if self.alibi_slopes is not None or self.sinks is not None:
            raise NotImplementedError("QSA does not support ALiBi or attention sinks")
        if self.sliding_window != (-1, -1):
            raise NotImplementedError("QSA does not support sliding-window attention")

        num_tokens = attn_metadata.num_actual_tokens
        output.zero_()
        if num_tokens == 0:
            return output

        topk_buffer = getattr(layer, "topk_indices_buffer", None)
        if topk_buffer is None:
            raise RuntimeError("QSA owner did not provide its top-k buffer")
        logical_indices = topk_buffer[:num_tokens]
        token_to_req = token_to_req[:num_tokens]
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        if key_cache.dtype != torch.bfloat16 or query.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA requires BF16 Q/K/V")

        from .ops.qsa import qsa_sparse_paged_attention

        staged = _qsa_staged_prefill(
            layer,
            query,
            kv_cache,
            logical_indices,
            attn_metadata,
            token_to_req,
            num_tokens,
            use_prefill_config,
            output,
        )
        if staged is None:
            qsa_sparse_paged_attention(
                query[:num_tokens],
                key_cache,
                value_cache,
                logical_indices,
                attn_metadata.block_table,
                token_to_req,
                use_prefill_config,
                output[:num_tokens],
            )
            return output
        staged_start, staged_end = staged
        for low, high in ((0, staged_start), (staged_end, num_tokens)):
            if high > low:
                qsa_sparse_paged_attention(
                    query[low:high],
                    key_cache,
                    value_cache,
                    logical_indices[low:high],
                    attn_metadata.block_table,
                    token_to_req[low:high],
                    use_prefill_config,
                    output[low:high],
                )
        return output


class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
    """Merged Qwen full-attention owner with a QSA index side branch."""

    supports_dcp = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        if cache_config is None:
            raise ValueError("Qwen4Exp QSA requires a paged KV cache")
        if model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA currently requires BF16")
        if cache_config.cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError("Qwen4Exp QSA requires a BF16 main KV cache")
        if getattr(quant_config, "kv_cache_scheme", None) is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support KV quantization")
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA does not support context parallelism"
            )
        if not getattr(config, "is_causal", True):
            raise NotImplementedError("Qwen4Exp QSA requires causal decoder attention")

        self.config = config
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA attention heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        # Decode/verify batches have at most 1 + num_spec query tokens per
        # request; use_prefill_config (max_query_len > this) steers the
        # config table. Shorter batches take the decode profile — harmless,
        # the difference is tile-shape tuning, not correctness.
        self._max_decode_query_len = 1 + vllm_config.num_speculative_tokens
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated QSA KV heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        if self.dual_chunk_attention_config is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support dual-chunk RoPE")
        # Qwen4Exp full-attention checkpoints always pack a sigmoid output
        # gate next to Q, even when an inherited config default says otherwise.
        self.attn_output_gate = True

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=model.without_modelopt_fp4(quant_config),
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        mm_config = model_config.multimodal_config
        text_only = mm_config is None or mm_config.language_model_only
        mrope_section = getattr(self.rotary_emb, "mrope_section", None)
        supports_mrope = bool(
            type(self.rotary_emb) is MRotaryEmbedding
            and mrope_section
            and len(mrope_section) == 3
            and sum(mrope_section) == self.rotary_emb.rotary_dim // 2
            and getattr(self.rotary_emb, "mrope_interleaved", False)
        )
        supports_dtype = getattr(self.rotary_emb, "dtype", None) in (
            torch.float16,
            torch.bfloat16,
        )
        self.use_fused_qk_norm_rope_gate = (
            self.attn_output_gate
            and getattr(self.rotary_emb, "is_neox_style", False)
            and current_platform.is_cuda()
            and supports_dtype
            and (text_only or supports_mrope)
        )

        self.layer_name = f"{prefix}.attn"
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = cache_config.cache_dtype
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, model_config
        )
        if self.kv_cache_torch_dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA requires BF16 cache storage")
        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        self._qsa_kv_offload = os.environ.get("VLLM_QSA_KV_OFFLOAD", "0") == "1"
        self._qsa_host_kv = None
        self._qsa_offload_layers = sum(
            kind == "full_attention" for kind in config.layer_types
        )
        if vllm_config.speculative_config is not None:
            # The MTP drafter carries full-attention layers of its own (mtp.py
            # builds every draft layer as "full_attention"), and each of them
            # binds a host pool through bind_kv_cache below. layer_types
            # describes the target, so the RAM estimate has to add them here or
            # it under-counts the pool it is about to allocate.
            self._qsa_offload_layers += max(
                1, int(getattr(config, "mtp_num_hidden_layers", 1) or 1)
            )
        if self._qsa_kv_offload:
            if tp_size not in (1, 2):
                raise NotImplementedError("QSA host KV validated layout requires TP=1 or TP=2")
            if vllm_config.parallel_config.data_parallel_size != 1:
                raise NotImplementedError("QSA host KV budget requires DP=1")
            if self.total_num_kv_heads % tp_size:
                raise NotImplementedError("QSA host KV requires evenly sharded KV heads")
            if self.num_kv_heads * tp_size != self.total_num_kv_heads:
                raise RuntimeError("QSA host KV local head count disagrees with TP layout")
            # Each rank owns only its local heads. Across both PP stages,
            # count every TP shard of every layer against the host budget.
            self._qsa_offload_tp_size = tp_size
            if self.num_heads % self.num_kv_heads:
                raise RuntimeError("QSA host KV query/KV head ratio is invalid")
            if vllm_config.num_speculative_tokens > 1:
                from vllm.logger import init_logger

                # The verify step runs 1 + num_speculative_tokens query rows
                # per request and every row selects its own budget of tokens,
                # so the per-step host read grows with the draft width. Width 1
                # is the only one that has been exercised against host K/V.
                init_logger(__name__).warning(
                    "QSA host KV with num_speculative_tokens=%d: the verify "
                    "step reads %dx the decode step's K/V over PCIe.",
                    vllm_config.num_speculative_tokens,
                    1 + vllm_config.num_speculative_tokens,
                )
        self._qsa_stage_min_rows = _QSA_STAGE_MIN_ROWS
        self._qsa_stage_arena_bytes = int(
            os.environ.get("VLLM_QSA_KVO_ARENA", str(192 * 1024 * 1024))
        )
        set_default_quant_scales(self, register_buffer=True)

        self.attn_backend = Qwen4ExpQSAFlashAttentionBackend
        self.impl = Qwen4ExpQSAFlashAttentionImpl(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            None,
            None,
            self.kv_cache_dtype,
            None,
            AttentionType.DECODER,
            None,
        )
        self.indexer = QSAIndexer(
            vllm_config=vllm_config,
            config=config,
            layer_id=layer_id,
            rotary_emb=self.rotary_emb,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        # PACKED selection buffer: the trailing column holds each row's
        # valid-entry count (written by the expand kernel) — never a token
        # index; the sparse attention kernel reads it as its loop bound.
        # MTP skip_topk steps reuse rows frozen from step 0; the count is
        # a row column, so compaction/reuse keep it paired with the content.
        self.register_buffer(
            "topk_indices_buffer",
            torch.empty(
                max_tokens,
                self.indexer.packed_output_width,
                dtype=torch.int32,
            ),
            persistent=False,
        )

        static_context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        static_context[self.layer_name] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
            num_head_slots=1 if self._qsa_kv_offload else None,
            state_content_bytes=2 if self._qsa_kv_offload else None,
        )

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        if not self._qsa_kv_offload:
            super().bind_kv_cache(kv_cache)
            return
        from vllm.logger import init_logger
        from vllm.utils.platform_utils import is_uva_available
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

        if not is_uva_available():
            raise RuntimeError("QSA host KV requires CUDA UVA")
        if kv_cache.ndim != 4 or kv_cache.shape[1] != 1 or kv_cache.shape[3] != 1:
            raise ValueError("QSA host KV expected [blocks, 1, tokens, 1] GPU slots")
        num_blocks, _, block_size, _ = kv_cache.shape
        shape = (num_blocks, self.num_kv_heads, block_size, 2 * self.head_dim)
        layer_bytes = (
            num_blocks * self.num_kv_heads * block_size * 2 * self.head_dim * 2
        )
        # Pinned allocations may be rounded to power-of-two allocator bins.
        # Budget conservatively for all TP ranks, not only this local shard.
        reserved_layer_bytes = 1 << (max(1, layer_bytes) - 1).bit_length()
        total_bytes = (reserved_layer_bytes * self._qsa_offload_layers
                       * self._qsa_offload_tp_size)
        budget = float(os.environ.get("VLLM_QSA_KV_OFFLOAD_MAX_GIB", "64")) * 2**30
        if total_bytes > budget:
            raise RuntimeError(
                f"QSA host KV pool would need {total_bytes / 2**30:.2f} GiB "
                f"across {self._qsa_offload_layers} layers, exceeding "
                f"VLLM_QSA_KV_OFFLOAD_MAX_GIB={budget / 2**30:g}. "
                "Reduce --kv-cache-memory-bytes or --num-gpu-blocks-override."
            )
        self._qsa_host_kv = torch.empty(
            shape, dtype=torch.bfloat16, device="cpu", pin_memory=True
        )
        self._qsa_gpu_slots = kv_cache
        self.kv_cache = get_accelerator_view_from_cpu_tensor(self._qsa_host_kv)
        init_logger(__name__).info(
            "QSA host KV %s: %d blocks x %d tokens, %.3f GiB pinned (UVA)",
            self.layer_name, num_blocks, block_size, layer_bytes / 2**30,
        )

    @eager_break_during_capture
    def _run_qsa(
        self,
        projected_qk: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            output.zero_()
            return
        main_metadata = cast(FlashAttentionMetadata, metadata[self.layer_name])
        if self.kv_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")

        num_tokens = main_metadata.num_actual_tokens
        side_metadata = cast(
            QSAForwardMetadata,
            metadata[self.indexer.raw_key_cache.prefix],
        )
        if side_metadata.num_actual_tokens != num_tokens:
            raise RuntimeError("QSA main and side metadata token counts disagree")
        selected = self.indexer(
            projected_qk,
            positions,
            self.topk_indices_buffer[:num_tokens],
        )
        if selected.shape != (num_tokens, self.indexer.packed_output_width):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        impl.do_kv_cache_update(
            self,
            key,
            value,
            self.kv_cache,
            main_metadata.slot_mapping,
        )
        impl.forward_qsa(
            self,
            query,
            key,
            value,
            self.kv_cache,
            main_metadata,
            output,
            token_to_req=side_metadata.token_to_req,
            use_prefill_config=main_metadata.max_query_len > self._max_decode_query_len,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        num_tokens = hidden_states.shape[0]
        query = q.view(num_tokens, self.num_heads, self.head_dim)
        key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        attn_output = torch.empty_like(query)
        # Keep the index projection outside the eager break.
        projected_qk, _ = self.indexer.index_qk_proj(hidden_states)
        self._run_qsa(
            projected_qk,
            positions,
            query,
            key,
            value,
            attn_output,
        )
        flat_output = attn_output.view(num_tokens, -1)
        if gate is not None:
            flat_output = flat_output * torch.sigmoid(gate)
        output, _ = self.o_proj(flat_output)
        return output


__all__ = [
    "QSAIndexer",
    "Qwen4ExpQSAAttention",
    "Qwen4ExpQSAFlashAttentionBackend",
    "Qwen4ExpQSAFlashAttentionImpl",
]
