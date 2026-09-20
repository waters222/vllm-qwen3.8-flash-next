# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp n-gram embeddings with device and pinned-host storage."""

import mmap
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config
from vllm.distributed import get_dp_group, get_etp_group, get_tp_group
from vllm.forward_context import DPMetadata, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
    ModelOptQuantConfigBase,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.model_executor.parameter import (
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from ..common.ple import PLEVocabParallelEmbedding
from .ops.ple import ple_ngram_ids

logger = init_logger(__name__)


def _ple_compressed_tensors_claims(
    quant_config: QuantizationConfig,
    names: list[str],
) -> bool:
    """Whether any compressed-tensors config group names one of ``names``."""
    targets = getattr(quant_config, "target_scheme_map", None) or {}
    for target in targets:
        if target.startswith("re:"):
            try:
                pattern = re.compile(target[3:])
            except re.error:
                continue
            if any(pattern.fullmatch(name) for name in names):
                return True
        elif any(name == target for name in names):
            return True
    return False


def _ple_method_for_compressed_tensors(
    quant_config: QuantizationConfig,
    prefix: str,
) -> "Qwen4ExpPLEEmbeddingMethod":
    """Resolve a PLE table against a compressed-tensors checkpoint."""
    names = [prefix, f"{prefix}.weight", f"{prefix}.shard_0"]
    if _ple_compressed_tensors_claims(quant_config, names):
        raise NotImplementedError(
            f"compressed-tensors claims to quantize the PLE table {prefix}, "
            "but vLLM has no gather-and-dequantize kernel for it"
        )
    return Qwen4ExpPLEUnquantizedEmbeddingMethod()


class Qwen4ExpPLEEmbedding(PLEVocabParallelEmbedding, ABC):
    """ETP-sharded PLE table shared by device and pinned-host backends."""

    supports_prefetch: ClassVar[bool] = False

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype,
        padding_size: int,
        prefix: str,
        embedding_method: "Qwen4ExpPLEEmbeddingMethod",
        num_ngram_heads: int = 1,
        max_total_tokens: int = 0,
        data_parallel_rank: int = 0,
    ) -> None:
        del num_ngram_heads, max_total_tokens
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
            quant_method=embedding_method,
            parallel_group=get_etp_group(),
        )
        self.embedding_method = embedding_method
        self.data_parallel_rank = data_parallel_rank
        tp_size = get_tp_group().world_size
        if self.tp_size % tp_size:
            raise ValueError(
                "ETP size must be divisible by TP size, but got "
                f"ETP={self.tp_size} and TP={tp_size}"
            )
        self.etp_data_parallel_size = self.tp_size // tp_size

    @abstractmethod
    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate storage for the complete embedding weight."""
        raise NotImplementedError

    def dequantize(
        self,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Delegate storage-format conversion to the embedding method."""
        return self.embedding_method.dequantize(self, embeddings, output_dtype)

    def _get_dp_gather_slot(self, local_num_tokens: int) -> tuple[int, int]:
        """Return the per-DP slot size and this rank's slot offset."""
        if self.etp_data_parallel_size == 1:
            return local_num_tokens, 0
        dp_metadata: DPMetadata | None = get_forward_context().dp_metadata
        if dp_metadata is None:
            raise RuntimeError("ETP spanning DP requires DP token metadata")
        group_start = (self.data_parallel_rank // self.etp_data_parallel_size) * (
            self.etp_data_parallel_size
        )
        group_end = group_start + self.etp_data_parallel_size
        token_counts = dp_metadata.num_tokens_across_dp_cpu.tolist()
        group_counts = token_counts[group_start:group_end]
        slot_size = max(group_counts)
        dp_rank = get_dp_group().rank_in_group
        return slot_size, dp_rank * slot_size

    def _gather_dp_ids(
        self,
        ngram_ids: torch.Tensor,
        slot_size: int,
    ) -> torch.Tensor:
        """Gather DP-local IDs that share one ETP-sharded PLE table."""
        if self.etp_data_parallel_size == 1:
            return ngram_ids
        if ngram_ids.shape[0] < slot_size:
            padding = ngram_ids.new_zeros(
                slot_size - ngram_ids.shape[0], ngram_ids.shape[1]
            )
            ngram_ids = torch.cat((ngram_ids, padding), dim=0)
        return get_dp_group().all_gather(ngram_ids, dim=0)

    def _select_embeddings(
        self,
        embeddings: torch.Tensor,
        local_num_tokens: int,
        slot_offset: int,
    ) -> torch.Tensor:
        """Select this DP rank's rows from the ETP-reduced embeddings."""
        if self.etp_data_parallel_size == 1:
            return embeddings
        return embeddings.narrow(0, slot_offset, local_num_tokens)

    @abstractmethod
    def start_prefetch(
        self,
        hidden_states: torch.Tensor,
        ngram_ids: torch.Tensor,
    ) -> None:
        """Start an asynchronous lookup when supported."""
        raise NotImplementedError


class Qwen4ExpPLEEmbeddingMethod(QuantizeMethodBase):
    """Quantization interface shared by resident and pinned PLE tables."""

    # PLE post-load processing only validates scales in their current storage.
    requires_device_loading: bool = False

    @staticmethod
    def from_quant_config(
        quant_config: QuantizationConfig | None,
        prefix: str,
        embedding_dtype: str | None = None,
    ) -> "Qwen4ExpPLEEmbeddingMethod":
        """Select the concrete PLE embedding format for a layer."""
        if embedding_dtype == "float8_e4m3fn":
            return Qwen4ExpPLEFp8EmbeddingMethod()
        if quant_config is None:
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if quant_config.get_name() == "compressed-tensors":
            return _ple_method_for_compressed_tensors(quant_config, prefix)
        if isinstance(quant_config, ModelOptMixedPrecisionConfig):
            if quant_config._resolve_quant_algo(prefix) == "FP8":
                return Qwen4ExpPLEFp8EmbeddingMethod()
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if isinstance(
            quant_config, ModelOptQuantConfigBase
        ) and quant_config.is_layer_excluded(prefix):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if not isinstance(quant_config, Fp8Config):
            raise NotImplementedError(
                "Qwen4Exp PLE embedding does not support quantization config "
                f"{type(quant_config).__name__}"
            )

        ignored_layers = quant_config.ignored_layers
        if is_layer_skipped(
            prefix,
            ignored_layers,
            quant_config.packed_modules_mapping,
            match_mode=quant_config.ignored_layers_match_mode,
        ):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        # PLE checkpoint shards form one runtime embedding parameter.
        shard_prefix = f"{prefix}.shard_"
        if any(name.startswith(shard_prefix) for name in ignored_layers):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if not quant_config.is_checkpoint_fp8_serialized:
            raise NotImplementedError(
                "Qwen4Exp PLE embedding only supports serialized FP8 checkpoints"
            )
        return Qwen4ExpPLEFp8EmbeddingMethod()

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("PLE weights only support embedding lookup")

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, layer.weight)

    @abstractmethod
    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert looked-up PLE rows to the activation dtype."""
        raise NotImplementedError


class Qwen4ExpPLEUnquantizedEmbeddingMethod(Qwen4ExpPLEEmbeddingMethod):
    """Unquantized PLE embedding storage and lookup semantics."""

    def create_weights(
        self,
        layer: Qwen4ExpPLEEmbedding,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        weight = nn.Parameter(
            layer.allocate_embedding_weight(
                sum(output_partition_sizes),
                input_size_per_partition,
                params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(weight, extra_weight_attrs)
        layer.register_parameter("weight", weight)

    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        del layer, output_dtype
        return embeddings


class Qwen4ExpPLEFp8EmbeddingMethod(Qwen4ExpPLEEmbeddingMethod):
    """FP8 PLE embedding with one global checkpoint scale."""

    def create_weights(
        self,
        layer: Qwen4ExpPLEEmbedding,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype
        weight_loader = extra_weight_attrs.get("weight_loader")
        weight = ModelWeightParameter(
            data=layer.allocate_embedding_weight(
                sum(output_partition_sizes),
                input_size_per_partition,
                torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = create_fp8_scale_parameter(
            PerTensorScaleParameter,
            output_partition_sizes,
            input_size_per_partition,
            None,
            weight_loader,
            scale_dtype=torch.float32,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Reject FP8 PLE checkpoints without a global scale."""
        sentinel = torch.finfo(torch.float32).min
        if torch.any(layer.weight_scale == sentinel):
            raise ValueError("FP8 PLE checkpoint is missing its global scale")

    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        weight_scale = getattr(layer, "weight_scale", None)
        if weight_scale is None:
            raise RuntimeError("FP8 PLE embedding is missing its global scale")
        if weight_scale.device != embeddings.device:
            raise RuntimeError("FP8 PLE embedding scale must be on the output device")
        return embeddings.to(output_dtype) * weight_scale.to(output_dtype)


class Qwen4ExpPLEDeviceEmbedding(Qwen4ExpPLEEmbedding):
    """PLE table allocated on the active model device."""

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate the complete PLE weight on the active device."""
        return torch.empty(num_embeddings, embedding_dim, dtype=dtype)

    def start_prefetch(
        self,
        hidden_states: torch.Tensor,
        ngram_ids: torch.Tensor,
    ) -> None:
        """Resident embedding prefetch is a no-op."""
        return None

    def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """Gather ETP inputs, look up embeddings, and select local rows."""
        slot_size, slot_offset = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
        embeddings = super().forward(gathered_ids)
        return self._select_embeddings(
            embeddings,
            ngram_ids.shape[0],
            slot_offset,
        )


@triton.jit
def _lookup_ple_embedding_from_pinned_kernel(
    weight_ptr,
    ids_ptr,
    output_ptr,
    embedding_dim,
    tp_vocab_start,
    tp_vocab_end,
    BLOCK_D: tl.constexpr,
):
    """Look up TP-owned PLE rows through a CUDA view of pinned host memory."""
    row_id = tl.program_id(0)
    global_idx = tl.load(ids_ptr + row_id)
    in_range = (global_idx >= tp_vocab_start) & (global_idx < tp_vocab_end)
    local_idx = tl.where(in_range, global_idx - tp_vocab_start, 0)
    offsets = tl.arange(0, BLOCK_D)
    store_mask = offsets < embedding_dim
    load_mask = store_mask & in_range
    values = tl.load(
        weight_ptr + local_idx * embedding_dim + offsets,
        mask=load_mask,
        other=0.0,
    )
    tl.store(
        output_ptr + row_id * embedding_dim + offsets,
        values,
        mask=store_mask,
    )


class Qwen4ExpPLEPinnedHostEmbedding(Qwen4ExpPLEEmbedding):
    """PLE table loaded into pinned CPU memory and looked up through UVA."""

    supports_prefetch: ClassVar[bool] = True

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype,
        padding_size: int,
        prefix: str,
        embedding_method: Qwen4ExpPLEEmbeddingMethod,
        num_ngram_heads: int = 1,
        max_total_tokens: int = 0,
        data_parallel_rank: int = 0,
    ) -> None:
        if not is_uva_available():
            raise RuntimeError("Engram CPU offload requires UVA support")
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
            embedding_method=embedding_method,
            num_ngram_heads=num_ngram_heads,
            max_total_tokens=max_total_tokens,
            data_parallel_rank=data_parallel_rank,
        )
        self._uva_weight = get_accelerator_view_from_cpu_tensor(self.weight)
        self._block_d = triton.next_power_of_2(self.embedding_dim)
        self._prefetch_stream = torch.cuda.Stream(device=self._uva_weight.device)
        self._prefetch_buffer = torch.empty(
            max_total_tokens * self.etp_data_parallel_size,
            num_ngram_heads,
            self.embedding_dim,
            dtype=self.weight.dtype,
            device=self._uva_weight.device,
        )
        self._output_dim = num_ngram_heads * self.embedding_dim

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate the complete PLE weight directly in pinned CPU memory."""
        return torch.empty(
            num_embeddings,
            embedding_dim,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )

    def _lookup(
        self,
        input_ids: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Look up local ETP rows while preserving the weight storage dtype."""
        expected_shape = (*input_ids.shape, self.embedding_dim)
        if output is None:
            output = torch.empty(
                expected_shape,
                dtype=self.weight.dtype,
                device=input_ids.device,
            )
        elif (
            tuple(output.shape) != expected_shape
            or output.dtype != self.weight.dtype
            or output.device != input_ids.device
        ):
            raise ValueError(
                "PLE prefetch output must match the input shape, weight dtype, "
                "and input device"
            )

        flat_ids = input_ids.reshape(-1).long()
        if flat_ids.numel():
            _lookup_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](
                self._uva_weight,
                flat_ids,
                output,
                self.embedding_dim,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                BLOCK_D=self._block_d,
            )
        return output

    def _reduce_etp_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Combine pinned lookup results owned by different ETP ranks."""
        if self.tp_size == 1:
            return embeddings
        assert self.parallel_group is not None
        if embeddings.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # Each vocabulary row has one owner, so reduce the raw FP8 bytes.
            reduced = self.parallel_group.all_reduce(embeddings.view(torch.int8))
            return reduced.view(embeddings.dtype)
        return self.parallel_group.all_reduce(embeddings)

    @eager_break_during_capture
    def start_prefetch(
        self,
        hidden_states: torch.Tensor,
        ngram_ids: torch.Tensor,
    ) -> None:
        """Gather ETP IDs and launch their UVA lookup on the side stream."""
        slot_size, _ = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
        active_output = self._prefetch_buffer[: gathered_ids.shape[0]]
        prefetch_stream = self._prefetch_stream
        prefetch_stream.wait_stream(torch.cuda.current_stream())
        gathered_ids.record_stream(prefetch_stream)
        with torch.cuda.stream(prefetch_stream):
            self._lookup(gathered_ids, output=active_output)

    @eager_break_during_capture
    def _finalize_prefetch(
        self,
        prefetch_output: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Join the side stream, reduce ETP shards, and select local rows."""
        torch.cuda.current_stream().wait_stream(self._prefetch_stream)
        slot_size, slot_offset = self._get_dp_gather_slot(output.shape[0])
        active_output = prefetch_output[: slot_size * self.etp_data_parallel_size]
        embeddings = self._reduce_etp_embeddings(active_output)
        embeddings = self._select_embeddings(
            embeddings,
            output.shape[0],
            slot_offset,
        )
        output.copy_(embeddings.flatten(-2))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Finish the pinned lookup into graph-owned output storage."""
        output = self._prefetch_buffer.new_empty(
            (hidden_states.shape[0], self._output_dim)
        )
        self._finalize_prefetch(self._prefetch_buffer, output)
        return output


_PLE_MMAP_PATH_ENV = "VLLM_PLE_MMAP_PATH"
_PLE_MMAP_REBUILD_ENV = "VLLM_PLE_MMAP_REBUILD"
_PLE_HOST_GATHER_ENV = "VLLM_PLE_HOST_GATHER"

# numpy has no bfloat16, so rows are gathered through a same-width int view.
_PLE_STORAGE_NUMPY_DTYPES = {1: np.int8, 2: np.int16, 4: np.int32}
_PLE_STORAGE_TORCH_DTYPES = {1: torch.int8, 2: torch.int16, 4: torch.int32}


class Qwen4ExpPLEMmapHostEmbedding(Qwen4ExpPLEPinnedHostEmbedding):
    """PLE table left on disk and gathered on the host through a mmap.

    The file is a raw, C-contiguous ``[num_embeddings_padded, embedding_dim]``
    array in the checkpoint's storage dtype -- the ``shard_0 .. shard_N`` PLE
    tensors concatenated in index order. ``VLLM_PLE_MMAP_PATH`` points at it;
    a missing file is created and filled while the checkpoint loads, and an
    existing one of the expected size is mapped copy-on-write and left alone.
    ``VLLM_PLE_MMAP_REBUILD=1`` forces a refill.

    Requires ETP=1, checked here.

    The lookup is a host-side gather and cannot run inside a captured CUDA
    graph -- so it does not run there. ``Qwen4ExpModelState.prepare_inputs``
    calls :meth:`host_gather` before the forward begins, and :meth:`forward`
    only reads the buffer it filled. Every cudagraph mode therefore works,
    ``FULL_DECODE_ONLY`` included; a PIECEWISE mode still needs
    ``VLLM_USE_BREAKABLE_CUDAGRAPH=1`` for the eager breaks the GDN, QSA and
    short-conv layers rely on.

    ``VLLM_PLE_HOST_GATHER=0`` puts the gather back inside the forward, which
    is the behaviour of earlier releases. It restores the old restriction that
    FULL cudagraph modes are rejected, and it costs roughly 45% of single
    stream decode throughput. It exists to A/B the change on one build.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype,
        padding_size: int,
        prefix: str,
        embedding_method: "Qwen4ExpPLEEmbeddingMethod",
        num_ngram_heads: int = 1,
        max_total_tokens: int = 0,
        data_parallel_rank: int = 0,
    ) -> None:
        self._mmap_path = os.environ.get(_PLE_MMAP_PATH_ENV, "").strip()
        if not self._mmap_path:
            raise RuntimeError(
                f"{_PLE_MMAP_PATH_ENV} must point at the raw PLE table file"
            )
        self._mmap_array: "np.memmap | None" = None
        self._mmap_prefilled = False
        # Skips Qwen4ExpPLEPinnedHostEmbedding.__init__, which requires UVA and
        # builds an accelerator view of pinned memory. Everything else it
        # defines (start_prefetch / _finalize_prefetch / forward) is reused.
        Qwen4ExpPLEEmbedding.__init__(
            self,
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
            embedding_method=embedding_method,
            num_ngram_heads=num_ngram_heads,
            max_total_tokens=max_total_tokens,
            data_parallel_rank=data_parallel_rank,
        )
        if self.tp_size != 1:
            raise RuntimeError(
                "The mmap PLE backend requires ETP=1 because the mapped file "
                f"holds the whole table; got ETP={self.tp_size}"
            )

        self.prefetch_from_model_state = (
            os.environ.get(_PLE_HOST_GATHER_ENV, "1").strip() not in ("", "0")
        )

        from vllm.compilation.breakable_cudagraph import (
            is_breakable_cudagraph_enabled,
        )

        cg_mode = get_current_vllm_config().compilation_config.cudagraph_mode
        # FULL is allowed because host_gather() runs before the forward, which
        # leaves forward() a plain device read at a fixed address. With the
        # gather back inside the forward it cannot be captured at all.
        if not self.prefetch_from_model_state and cg_mode.has_full_cudagraphs():
            raise RuntimeError(
                f"{_PLE_HOST_GATHER_ENV}=0 keeps the host gather inside the "
                f"forward, where it cannot be captured; cudagraph_mode "
                f"{cg_mode.name} requires it hoisted out"
            )
        if cg_mode.has_piecewise_cudagraphs() and not is_breakable_cudagraph_enabled():
            raise RuntimeError(
                f"mmap PLE with cudagraph_mode {cg_mode.name} requires "
                "VLLM_USE_BREAKABLE_CUDAGRAPH=1"
            )

        device = torch.device("cuda", torch.cuda.current_device())
        self._block_d = triton.next_power_of_2(self.embedding_dim)
        self._prefetch_stream = torch.cuda.Stream(device=device)
        # host_gather() fills this and forward() returns a view of it, so it is
        # allocated once and never reallocated: a cudagraph replay reads the
        # same address every time. Zeroed rather than uninitialised so that a
        # forward running before the first gather sees zeros.
        self._prefetch_buffer = torch.zeros(
            max_total_tokens * self.etp_data_parallel_size,
            num_ngram_heads,
            self.embedding_dim,
            dtype=self.weight.dtype,
            device=device,
        )
        self._host_stage = torch.empty(
            max_total_tokens * self.etp_data_parallel_size * num_ngram_heads,
            self.embedding_dim,
            dtype=self.weight.dtype,
            device="cpu",
            pin_memory=True,
        )
        storage_dtype = _PLE_STORAGE_TORCH_DTYPES[self.weight.dtype.itemsize]
        self._host_stage_np = self._host_stage.view(storage_dtype).numpy()
        self._output_dim = num_ngram_heads * self.embedding_dim

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Map the complete PLE weight from the raw file on disk."""
        np_dtype = _PLE_STORAGE_NUMPY_DTYPES.get(dtype.itemsize)
        if np_dtype is None:
            raise RuntimeError(
                f"mmap PLE backend cannot represent a {dtype.itemsize}-byte "
                f"storage dtype ({dtype})"
            )
        path = self._mmap_path
        nbytes = num_embeddings * embedding_dim * dtype.itemsize
        exists = os.path.exists(path)
        if exists:
            actual = os.path.getsize(path)
            if actual != nbytes:
                raise RuntimeError(
                    f"PLE mmap table {path} is {actual} bytes but this model "
                    f"needs {nbytes} for [{num_embeddings}, {embedding_dim}] "
                    f"{dtype}. Point {_PLE_MMAP_PATH_ENV} elsewhere or delete "
                    "the stale file."
                )
        rebuild = os.environ.get(_PLE_MMAP_REBUILD_ENV, "0").strip() not in ("", "0")
        self._mmap_prefilled = exists and not rebuild
        if not exists:
            parent = os.path.dirname(os.path.abspath(path))
            os.makedirs(parent, exist_ok=True)
            with open(path, "wb") as handle:
                handle.truncate(nbytes)
        # Copy-on-write, so a stray write cannot corrupt a shared cache file.
        mode = "c" if self._mmap_prefilled else "r+"
        self._mmap_array = np.memmap(
            path,
            dtype=np_dtype,
            mode=mode,
            shape=(num_embeddings, embedding_dim),
        )
        # The lookup is a scattered row gather, so readahead is pure waste.
        raw_map = getattr(self._mmap_array, "_mmap", None)
        if raw_map is not None and hasattr(raw_map, "madvise"):
            try:
                raw_map.madvise(mmap.MADV_RANDOM)
            except OSError as err:
                logger.warning("MADV_RANDOM on the PLE table failed: %s", err)
        logger.info(
            "PLE table mapped from %s (%.2f GiB, mode=%s, prefilled=%s)",
            path,
            nbytes / (1 << 30),
            mode,
            self._mmap_prefilled,
        )
        return torch.from_numpy(self._mmap_array).view(dtype)

    def weight_loader(
        self,
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        checkpoint_start: int | None = None,
    ) -> None:
        """Skip the copy when the mapped file already holds the table."""
        if self._mmap_prefilled:
            return
        super().weight_loader(param, loaded_weight, checkpoint_start)

    def finalize_mmap_cache(self) -> None:
        """Flush a freshly filled table so the next run can skip loading."""
        if self._mmap_prefilled or self._mmap_array is None:
            return
        self._mmap_array.flush()
        self._mmap_prefilled = True
        logger.info("PLE table written to %s", self._mmap_path)

    def _lookup(
        self,
        input_ids: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gather the requested rows on the host and copy them to the device."""
        expected_shape = (*input_ids.shape, self.embedding_dim)
        if output is None:
            output = torch.empty(
                expected_shape,
                dtype=self.weight.dtype,
                device=input_ids.device,
            )
        elif (
            tuple(output.shape) != expected_shape
            or output.dtype != self.weight.dtype
            or output.device != input_ids.device
        ):
            raise ValueError(
                "PLE prefetch output must match the input shape, weight dtype, "
                "and input device"
            )

        flat_ids = input_ids.reshape(-1)
        count = flat_ids.numel()
        if count:
            if count > self._host_stage.shape[0]:
                raise RuntimeError(
                    f"PLE host staging buffer holds {self._host_stage.shape[0]} "
                    f"rows but {count} were requested"
                )
            # Host gather: needs the ids on the CPU, hence a device sync.
            ids = flat_ids.to(device="cpu", dtype=torch.int64).numpy()
            vocab_start = self.shard_indices.org_vocab_start_index
            vocab_end = self.shard_indices.org_vocab_end_index
            in_range = (ids >= vocab_start) & (ids < vocab_end)
            local_ids = np.where(in_range, ids - vocab_start, 0)
            stage = self._host_stage_np[:count]
            np.take(self._mmap_array, local_ids, axis=0, out=stage)
            if not in_range.all():
                stage[~in_range] = 0
            output.view(-1, self.embedding_dim).copy_(
                self._host_stage[:count], non_blocking=True
            )
        return output

    prefetch_from_model_state: bool      # set from the environment in __init__

    # The gather runs before the forward, not inside it.
    #
    # _lookup() stops the host until the ids reach the CPU. Done inside the
    # forward, that sync every step destroys the run-ahead that vLLM V2 plus
    # async scheduling is built on: the host never gets far enough ahead of the
    # GPU, and the per-layer eager islands (12 GDN + 4 QSA + 1 PLE short conv
    # per rank) surface as GPU gaps instead of hiding behind the compute. On
    # 3 x RTX 3090 that capped single stream decode at 45 tok/s against a
    # 12.6 ms GPU-busy floor; hoisting the gather and enabling FULL_DECODE_ONLY
    # measured 80 tok/s.
    #
    # Note that hoisting alone changes nothing (measured 45.1 tok/s): under
    # PIECEWISE the eager islands still follow the sync. The value of the hoist
    # is that it makes a FULL cudagraph mode legal, and that is what pays.

    def host_gather(self, ngram_ids: torch.Tensor) -> None:
        """Fill the lookup buffer from the host. Runs OUTSIDE the forward.

        The same work start_prefetch() used to do, minus the side stream: the
        H2D copy lands on the calling stream, which is the stream the forward
        (or its cudagraph replay) runs on immediately afterwards, so ordering
        is implicit and no event is needed.
        """
        slot_size, _ = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
        self._lookup(
            gathered_ids,
            output=self._prefetch_buffer[: gathered_ids.shape[0]],
        )

    def start_prefetch(
        self,
        hidden_states: torch.Tensor,
        ngram_ids: torch.Tensor,
    ) -> None:
        """No-op once host_gather() owns the lookup; the old path otherwise."""
        if self.prefetch_from_model_state:
            return
        super().start_prefetch(hidden_states, ngram_ids)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return the rows host_gather() staged for this batch.

        A view, so the address is fixed and the read can be captured. ETP is 1
        for this backend (__init__ enforces it), which makes the ETP all-reduce
        and the DP row selection that _finalize_prefetch performs both
        identities -- all that remains of it is the flatten(-2).
        """
        if not self.prefetch_from_model_state:
            return super().forward(hidden_states)
        return self._prefetch_buffer[: hidden_states.shape[0]].flatten(-2)


class Qwen4ExpNGramEmbedding(nn.Module):
    _MASK64 = (1 << 64) - 1
    _SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
    _SPLITMIX_M1 = 0xBF58476D1CE4E5B9
    _SPLITMIX_M2 = 0x94D049BB133111EB
    _PLE_LAYER_PRIME = 10007

    @classmethod
    def _splitmix64(cls, value: int) -> int:
        """Mix an integer into a deterministic unsigned 64-bit value."""
        value = (value + cls._SPLITMIX_GAMMA) & cls._MASK64
        value = ((value ^ (value >> 30)) * cls._SPLITMIX_M1) & cls._MASK64
        value = ((value ^ (value >> 27)) * cls._SPLITMIX_M2) & cls._MASK64
        return (value ^ (value >> 31)) & cls._MASK64

    @staticmethod
    def _is_prime_64(value: int) -> bool:
        """Return whether a 64-bit integer is prime."""
        if value < 2:
            return False
        for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if value % prime == 0:
                return value == prime
        exponent = value - 1
        shifts = 0
        while exponent % 2 == 0:
            exponent //= 2
            shifts += 1
        for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
            if base % value == 0:
                continue
            witness = pow(base, exponent, value)
            if witness in (1, value - 1):
                continue
            for _ in range(shifts - 1):
                witness = pow(witness, 2, value)
                if witness == value - 1:
                    break
            else:
                return False
        return True

    @classmethod
    def _nth_prime_after(cls, start: int, count: int) -> int:
        """Return the ``count``-th prime strictly greater than ``start``."""
        prime = int(start)
        for _ in range(count):
            candidate = prime + 1
            if candidate <= 2:
                prime = 2
                continue
            if candidate % 2 == 0:
                candidate += 1
            while not cls._is_prime_64(candidate):
                candidate += 2
            prime = candidate
        return prime

    @classmethod
    def _make_layer_multipliers(
        cls,
        *,
        ngram_size: int,
        unigram_vocab_size: int,
        seed: int,
        ple_dense_layer_id: int,
    ) -> list[int]:
        """Build deterministic hash multipliers for one PLE layer."""
        max_multiplier = ((1 << 63) - 1) // unigram_vocab_size
        half_bound = max(1, max_multiplier // 2)
        base_seed = seed + cls._PLE_LAYER_PRIME * ple_dense_layer_id
        multipliers = []
        for index in range(ngram_size):
            value = base_seed + cls._SPLITMIX_GAMMA * (index + 1)
            multipliers.append(2 * (cls._splitmix64(value) % half_bound) + 1)
        return multipliers

    @classmethod
    def _make_vocab_layout(
        cls,
        *,
        ngram_vocab_size_base: int,
        ngram_heads: int,
        ple_dense_layer_id: int,
    ) -> tuple[list[int], list[int], int]:
        """Build per-head vocabulary sizes, offsets, and total row count."""
        sizes: list[int] = []
        offsets: list[int] = []
        offset = 0
        for local_head in range(ngram_heads):
            global_head = ple_dense_layer_id * ngram_heads + local_head
            size = cls._nth_prime_after(ngram_vocab_size_base - 1, global_head + 1)
            sizes.append(size)
            offsets.append(offset)
            offset += size
        return sizes, offsets, offset

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        embedding_dim: int,
        ple_dense_layer_id: int,
        max_total_tokens: int,
        *,
        data_parallel_rank: int,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError(f"heads_per_ngram must be > 0, got {self.heads_per_ngram}")
        if embedding_dim % self.ngram_heads:
            raise ValueError(
                "ple_embed_dim must be divisible by total ngram heads: "
                f"{embedding_dim} % {self.ngram_heads} != 0"
            )
        self.head_dim = embedding_dim // self.ngram_heads
        self.eos_token_id = int(config.eos_token_id)
        self.unigram_vocab_size = int(config.vocab_size)
        self.split_ngram_parts = int(getattr(config, "split_ngram_parts", 512))
        if self.split_ngram_parts <= 0:
            raise ValueError("split_ngram_parts must be positive")

        multipliers = self._make_layer_multipliers(
            ngram_size=self.ngram_size,
            unigram_vocab_size=self.unigram_vocab_size,
            seed=int(getattr(config, "seed", 1234)),
            ple_dense_layer_id=ple_dense_layer_id,
        )
        self.register_buffer(
            "layer_multipliers",
            torch.tensor(multipliers, dtype=torch.long),
            persistent=True,
        )

        sizes, offsets, total_vocab_size = self._make_vocab_layout(
            ngram_vocab_size_base=int(config.ngram_vocab_size_base),
            ngram_heads=self.ngram_heads,
            ple_dense_layer_id=ple_dense_layer_id,
        )
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(sizes, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(offsets, dtype=torch.long),
            persistent=True,
        )
        divisor = int(config.make_ngram_vocab_size_divisible_by)
        padded_vocab_size = ((total_vocab_size + divisor - 1) // divisor) * divisor
        embedding_prefix = f"{prefix}.ngram_embedding"
        embedding_quant_method = Qwen4ExpPLEEmbeddingMethod.from_quant_config(
            quant_config,
            embedding_prefix,
            getattr(config, "ple_embedding_dtype", None),
        )
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        engram_config = get_current_vllm_config().engram_config
        if os.environ.get(_PLE_MMAP_PATH_ENV, "").strip():
            embedding_cls = Qwen4ExpPLEMmapHostEmbedding
        elif engram_config is not None and engram_config.cpu_offload:
            embedding_cls = Qwen4ExpPLEPinnedHostEmbedding
        else:
            embedding_cls = Qwen4ExpPLEDeviceEmbedding
        self.ngram_embedding = embedding_cls(
            padded_vocab_size,
            self.head_dim,
            params_dtype=params_dtype,
            padding_size=divisor,
            prefix=embedding_prefix,
            embedding_method=embedding_quant_method,
            num_ngram_heads=self.ngram_heads,
            max_total_tokens=max_total_tokens,
            data_parallel_rank=data_parallel_rank,
        )
        weight = self.ngram_embedding.weight
        logger.info(
            "Initialized PLE embedding %s: quantization_method=%s, "
            "weight_dtype=%s, weight_device=%s, pinned=%s",
            embedding_prefix,
            type(embedding_quant_method).__name__,
            weight.dtype,
            weight.device,
            weight.is_pinned(),
        )

    @staticmethod
    def _shift_precompute(
        tokens: torch.Tensor, eos_token_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.dim() != 2:
            raise ValueError("tokens must be a 2D tensor")
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device, dtype=torch.int64)
        eos_positions = torch.where(tokens == eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [
                eos_positions.new_full((batch_size, 1), -1),
                previous_eos_inclusive[:, :-1],
            ],
            dim=1,
        )
        return positions, positions.unsqueeze(0) - previous_eos - 1

    @staticmethod
    def _shift_apply(
        tokens: torch.Tensor,
        positions: torch.Tensor,
        position_in_segment: torch.Tensor,
        shift: int,
        eos_token_id: int,
    ) -> torch.Tensor:
        if shift == 0:
            return tokens
        source = positions - shift
        gather_indices = source.clamp_min(0).unsqueeze(0).expand(tokens.shape[0], -1)
        shifted = tokens.gather(1, gather_indices)
        valid = (source.unsqueeze(0) >= 0) & (position_in_segment >= shift)
        return torch.where(valid, shifted, tokens.new_full((), eos_token_id))

    def compute_ngram_ids(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute n-gram embedding indices for the current request layout."""
        input_ids = input_ids.reshape(-1)
        num_reqs = query_start_loc.numel() - 1
        num_tokens = input_ids.shape[0]

        if input_ids.is_cuda:
            return ple_ngram_ids(
                input_ids=input_ids,
                query_start_loc=query_start_loc,
                ngram_context=ngram_context,
                layer_multipliers=self.layer_multipliers,
                ngram_heads_vocab_sizes=self.ngram_heads_vocab_sizes,
                ngram_heads_offsets=self.ngram_heads_offsets,
                eos_token_id=self.eos_token_id,
                heads_per_ngram=self.heads_per_ngram,
                output=output,
            )
        input_ids = input_ids.long()
        query_start_loc = query_start_loc.long()
        positions = torch.arange(num_tokens, device=input_ids.device, dtype=torch.int64)
        packed = torch.full(
            (num_reqs, num_tokens),
            self.eos_token_id,
            device=input_ids.device,
            dtype=torch.int64,
        )
        request_indices = torch.searchsorted(query_start_loc, positions, right=True) - 1
        request_indices.clamp_(max=num_reqs - 1)
        columns = (positions - query_start_loc[request_indices]).clamp(
            0, packed.shape[1] - 1
        )
        packed[request_indices, columns] = input_ids
        ngram_context = ngram_context[:num_reqs].to(
            device=input_ids.device, dtype=torch.long
        )

        context = torch.cat([ngram_context, packed], dim=-1)
        positions_2d, position_in_segment = self._shift_precompute(
            context, self.eos_token_id
        )
        shifted = [context]
        for shift in range(1, self.ngram_size):
            shifted.append(
                self._shift_apply(
                    context,
                    positions_2d,
                    position_in_segment,
                    shift,
                    self.eos_token_id,
                )
            )
        adjusted_columns = columns + self.ngram_size - 1
        id_blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for index in range(1, ngram):
                mixed = torch.bitwise_xor(
                    mixed, shifted[index] * self.layer_multipliers[index]
                )
            sizes = self.ngram_heads_vocab_sizes[start:end]
            offsets = self.ngram_heads_offsets[start:end]
            ids = torch.remainder(mixed.unsqueeze(-1), sizes) + offsets
            id_blocks.append(ids[request_indices, adjusted_columns])
        return torch.cat(id_blocks, dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.ngram_embedding
        if embedding.supports_prefetch:
            return embedding(hidden_states)
        ngram_ids = self.compute_ngram_ids(input_ids, query_start_loc, ngram_context)
        return self.ngram_embedding(ngram_ids).flatten(-2)

    def host_gather(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> None:
        """Fill the PLE lookup buffer from the host, before the forward.

        Called by Qwen4ExpModelState.prepare_inputs. Only the mmap backend
        wants it: its gather reads the ids on the CPU, and doing that inside
        the forward synchronises the host with the device on every step. A
        no-op for the device and pinned backends, which prefetch from inside
        the forward.
        """
        embedding = self.ngram_embedding
        if not getattr(embedding, "prefetch_from_model_state", False):
            return
        ngram_ids = self.compute_ngram_ids(
            input_ids,
            query_start_loc,
            ngram_context,
        )
        embedding.host_gather(ngram_ids)

    def start_prefetch(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> None:
        """Start the pinned lookup while the preceding decoder layer runs."""
        embedding = self.ngram_embedding
        if not embedding.supports_prefetch:
            return
        if getattr(embedding, "prefetch_from_model_state", False):
            # host_gather() already ran. Recomputing the ids here would put a
            # kernel back into every forward, and so into every graph replay.
            return
        ngram_ids = self.compute_ngram_ids(
            input_ids,
            query_start_loc,
            ngram_context,
        )
        embedding.start_prefetch(hidden_states, ngram_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load hash buffers and checkpoint-split embedding rows."""

        persistent_buffers = {
            "layer_multipliers": self.layer_multipliers,
            "ngram_heads_offsets": self.ngram_heads_offsets,
            "ngram_heads_vocab_sizes": self.ngram_heads_vocab_sizes,
        }
        loaded: set[str] = set()
        regular_weights: list[tuple[str, torch.Tensor]] = []
        shard_prefix = "ngram_embedding.shard_"

        for name, loaded_weight in weights:
            leaf_name = name.rsplit(".", 1)[-1]
            if leaf_name.startswith("hashstats_") or leaf_name == "token_lookup":
                continue
            if name in persistent_buffers:
                buffer = persistent_buffers[name]
                if buffer.shape != loaded_weight.shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: expected "
                        f"{tuple(buffer.shape)}, got {tuple(loaded_weight.shape)}"
                    )
                buffer.copy_(loaded_weight.to(device=buffer.device, dtype=buffer.dtype))
                loaded.add(name)
                continue
            if name.startswith(shard_prefix) and name.endswith(".weight"):
                shard_text = name[len(shard_prefix) : -len(".weight")]
                if not shard_text.isdigit():
                    regular_weights.append((name, loaded_weight))
                    continue
                shard_index = int(shard_text)
                if shard_index >= self.split_ngram_parts:
                    raise ValueError(
                        f"PLE embedding shard index {shard_index} exceeds "
                        f"split_ngram_parts={self.split_ngram_parts}"
                    )
                embedding = self.ngram_embedding
                shard_size = (
                    embedding.org_vocab_size + self.split_ngram_parts - 1
                ) // self.split_ngram_parts
                checkpoint_start = shard_index * shard_size
                expected_rows = max(
                    0,
                    min(shard_size, embedding.org_vocab_size - checkpoint_start),
                )
                expected_shape = (expected_rows, embedding.embedding_dim)
                if tuple(loaded_weight.shape) != expected_shape:
                    raise ValueError(
                        f"Shape mismatch for PLE embedding shard {shard_index}: "
                        f"expected {expected_shape}, got "
                        f"{tuple(loaded_weight.shape)}"
                    )
                embedding.weight.weight_loader(
                    embedding.weight,
                    loaded_weight,
                    checkpoint_start=checkpoint_start,
                )
                loaded.add("ngram_embedding.weight")
                continue
            regular_weights.append((name, loaded_weight))

        if regular_weights:
            loaded.update(AutoWeightsLoader(self).load_weights(regular_weights))
        finalize = getattr(self.ngram_embedding, "finalize_mmap_cache", None)
        if finalize is not None:
            finalize()
        return loaded


__all__ = [
    "Qwen4ExpNGramEmbedding",
]
