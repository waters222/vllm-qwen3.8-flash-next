# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-runner state for Qwen4Exp PLE inputs."""

from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.states import RequestState


class Qwen4ExpModelState(MambaHybridModelState):
    """Add rollback-safe PLE n-gram context to the model inputs."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        config = self.model_config.hf_text_config
        self.uses_ngram_embedding = bool(config.ple_layer_ids)
        self._ple_ngram: nn.Module | None = None
        if not self.uses_ngram_embedding:
            self.ngram_context_len = 0
            self.ngram_eos_token_id = 0
            return

        pp_size = vllm_config.parallel_config.pipeline_parallel_size
        if pp_size > 1:
            from vllm.distributed.utils import get_pp_indices

            num_layers = int(config.num_hidden_layers)
            first_start, first_end = get_pp_indices(num_layers, 0, pp_size)
            stranded = sorted(
                abs_id - 1
                for abs_id in config.ple_layer_ids
                if not (first_start <= abs_id - 1 < first_end)
            )
            if stranded:
                raise RuntimeError(
                    "N-gram PLE embedding requires every PLE layer to live on "
                    "the first pipeline rank, because later ranks do not "
                    "receive the raw input_ids that PLE consumes. "
                    f"pipeline_parallel_size={pp_size} strands decoder "
                    f"layer(s) {stranded} outside the first rank's range "
                    f"[{first_start}, {first_end}). Run with PP=1, or "
                    "repartition with VLLM_PP_LAYER_PARTITION so that the PLE "
                    "layers stay on rank 0."
                )

        self.ngram_context_len = int(config.ngram_size) - 1
        if self.ngram_context_len <= 0:
            raise ValueError("N-gram embedding requires context length >= 1.")
        self.ngram_eos_token_id = int(config.eos_token_id)
        # PLE runs inside captured regions, so these buffers keep a fixed shape
        # and address as the active request count changes between replays.
        self.ngram_context = torch.full(
            (self.max_num_reqs, self.ngram_context_len),
            self.ngram_eos_token_id,
            dtype=torch.int32,
            device=self.device,
        )
        self.ngram_context_offsets = torch.arange(
            -self.ngram_context_len,
            0,
            dtype=torch.int64,
            device=self.device,
        )
        self.ple_query_start_loc = torch.zeros(
            self.max_num_reqs + 1,
            dtype=torch.int32,
            device=self.device,
        )

        # The NVMe-backed PLE table gathers its rows on the host, which means
        # pulling the n-gram ids down from the device. Inside the forward that
        # sync runs every step and costs the host its run-ahead, and it also
        # keeps the PLE lookup out of every CUDA graph. Hoist it here instead:
        # prepare_inputs already runs on the host immediately before the
        # forward, and by then input_ids, query_start_loc and the n-gram
        # context are all final.
        #
        # Only the mmap backend opts in, via prefetch_from_model_state; the
        # device and pinned backends keep prefetching inside the forward. The
        # PLE layer lives on the first pipeline rank only, so every other rank
        # finds nothing here and skips the call.
        for _, module in model.named_modules():
            embedding = getattr(module, "ngram_embedding", None)
            if embedding is not None and getattr(
                embedding, "prefetch_from_model_state", False
            ):
                self._ple_ngram = module
                break
        if self._ple_ngram is not None:
            # prepare_dummy_inputs() is given token counts, not tokens.
            # Profiling and graph capture only need the buffer written, not
            # meaningful contents, so feed the gather a fixed run of zeros.
            self._ple_dummy_input_ids = torch.zeros(
                self.max_num_tokens,
                dtype=torch.int32,
                device=self.device,
            )

    def _prepare_ngram_context(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        context = self.ngram_context
        context.fill_(self.ngram_eos_token_id)
        if num_reqs == 0:
            return context

        request_indices = input_batch.idx_mapping[:num_reqs].long()
        context_end = req_states.num_computed_tokens.gpu[request_indices].long()
        token_indices = context_end.unsqueeze(1) + self.ngram_context_offsets
        valid_tokens = token_indices >= 0
        token_indices.clamp_min_(0)
        context_tokens = req_states.all_token_ids.gpu[
            request_indices.unsqueeze(1), token_indices
        ]
        context[:num_reqs].copy_(
            torch.where(
                valid_tokens,
                context_tokens,
                context_tokens.new_full((), self.ngram_eos_token_id),
            )
        )
        return context

    def prepare_inputs(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        if not self.uses_ngram_embedding:
            return model_inputs

        num_reqs_padded = input_batch.num_reqs_after_padding
        query_start_loc = self.ple_query_start_loc
        query_start_loc[: num_reqs_padded + 1].copy_(input_batch.query_start_loc)
        # Represent unused capacity as trailing zero-length requests.
        query_start_loc[num_reqs_padded + 1 :].copy_(input_batch.query_start_loc[-1])
        ngram_context = self._prepare_ngram_context(input_batch, req_states)
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
        )
        if self._ple_ngram is not None:
            # Exactly what Qwen4ExpModel._start_layer_ple_prefetch used to do
            # at the top of the forward, one host step earlier. input_ids is
            # the padded row count the forward will see, so the buffer is
            # filled for every row the graph reads back.
            self._ple_ngram.host_gather(
                input_batch.input_ids,
                query_start_loc,
                ngram_context,
            )
        return model_inputs

    def prepare_dummy_inputs(
        self,
        num_reqs: int,
        num_tokens: int,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if not self.uses_ngram_embedding:
            return model_inputs

        query_start_loc = self.ple_query_start_loc
        query_start_loc[0] = 0
        tokens_per_req, num_extra_tokens = divmod(num_tokens, num_reqs)
        query_lens = torch.full(
            (num_reqs,),
            tokens_per_req,
            dtype=query_start_loc.dtype,
            device=query_start_loc.device,
        )
        if num_extra_tokens > 0:
            query_lens[-num_extra_tokens:] += 1
        torch.cumsum(query_lens, dim=0, out=query_start_loc[1 : num_reqs + 1])
        query_start_loc[num_reqs + 1 :].fill_(num_tokens)

        ngram_context = self.ngram_context
        ngram_context.fill_(self.ngram_eos_token_id)
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
        )
        if self._ple_ngram is not None:
            # Must run for capture too: forward() reads the buffer, so it has
            # to hold something the PLE projection can consume while the graph
            # is being recorded. This runs outside the capture, because
            # cudagraph_utils calls prepare_dummy_inputs() before entering
            # torch.cuda.graph().
            self._ple_ngram.host_gather(
                self._ple_dummy_input_ids[:num_tokens],
                query_start_loc,
                ngram_context,
            )
        return model_inputs


__all__ = ["Qwen4ExpModelState"]
