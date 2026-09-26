# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker half of Flash-Next's opt-in, quiescent session-swap protocol.

Never call the transaction methods from a running decode iteration. The engine
coordinator must drain scheduling on every rank and hold it drained until the
transaction completes. This module does not decide scheduler block ownership.
The separate streaming-update hook runs before re-adding an existing slot;
its request must have no unsettled forward in the scheduler.
"""

import json
import os
import struct
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from vllm.v1.worker.gpu.flash_session_swap import (
    SessionColdStore,
    SessionKey,
    SwapBoundary,
    flash_cache_regions,
)


def _view(value: Any, slot: int, axis: int = 0) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    if not isinstance(value, torch.Tensor) or not 0 <= slot < value.shape[axis]:
        raise ValueError("invalid worker slot state")
    return value.narrow(axis, slot, 1)


def materialize_mamba_boundary(regions, accepted: int, *, conv_dim_first: bool) -> int:
    """Move accepted speculative state to the base slot before non-spec reuse.

    Regions are (kind, tensor, physical block IDs) triples for one request.
    All validation precedes writes. Caller must settle PP output/acceptance
    updates and keep this request unscheduled until the copies complete.
    Layout semantics match native get_conv_copy_spec/get_temporal_copy_spec;
    clones make overlapping conv-window shifts safe in both SD and DS layouts.
    """
    if type(accepted) is not int or accepted < 1:
        raise ValueError("invalid accepted-state boundary")
    offset = accepted - 1
    copies = []
    for kind, state, blocks in regions:
        if (
            not isinstance(state, torch.Tensor)
            or state.ndim < 2
            or len(blocks) < accepted
            or any(type(b) is not int or b <= 0 or b >= len(state) for b in blocks)
            or len(set(blocks)) != len(blocks)
        ):
            raise ValueError("invalid Mamba boundary cache or block table")
        base = state[blocks[0]]
        if kind == "conv":
            if state.ndim != 3:
                raise ValueError("unsupported convolutional-state rank")
            width = base.shape[-1] if conv_dim_first else base.shape[0]
            if offset >= width:
                raise ValueError("accepted convolutional offset exceeds state width")
            if conv_dim_first:
                source, target = base[:, offset:], base[:, : width - offset]
            else:
                source, target = base[offset:], base[: width - offset]
            needs_clone = True
        elif kind == "temporal":
            source, target = state[blocks[offset]], base
            needs_clone = False
        else:
            raise ValueError("unsupported Mamba boundary state kind")
        if offset:
            copies.append((source, target, needs_clone))
    for source, target, needs_clone in copies:
        target.copy_(source.clone() if needs_clone else source)
    return sum(target.numel() * target.element_size() for _, target, _ in copies)


def slot_fields(runner: Any, slot: int, token_count: int) -> dict[str, torch.Tensor]:
    """Enumerate persistent dynamic state; static parameters are re-added natively.

    The coordinator retains NewRequestData/SamplingParams and token history.
    add_requests() reconstructs static sampling parameters, RoPE and block
    tables before these dynamic fields are restored into the new slot.
    """
    if not 0 <= token_count <= runner.req_states.max_model_len:
        raise ValueError("invalid retained token count")
    model_type = type(runner.model_state)
    if model_type.__name__ not in ("MambaHybridModelState", "Qwen4ExpModelState") or (
        "MambaHybridModelState" not in {base.__name__ for base in model_type.__mro__}
    ):
        raise ValueError("session swap requires native hybrid model state")
    model = runner.model_state
    if model.recoverssm is not None:
        raise ValueError("RecoverSSM session state is not yet supported")
    if runner.adaptive_verification is not None:
        raise ValueError("adaptive verification session state is not yet supported")
    if runner.pooling_runner is not None:
        raise ValueError("session swap currently requires text-only generation")
    if runner.encoder_cache is not None:
        request_id = next(
            key
            for key, index in runner.req_states.req_id_to_index.items()
            if index == slot
        )
        features = runner.encoder_cache.mm_features.get(request_id)
        if features is None or features:
            raise ValueError("session swap requires a registered text-only request")
    # Qwen4Exp's PLE n-gram context/query-start buffers are per-batch scratch.
    # prepare_inputs rebuilds them from the restored token history and position;
    # copying a slot row would be incorrect because scratch rows use batch order.
    if getattr(model, "prompt_embeds_state", None) is not None:
        raise ValueError("prompt-embedding session state is not yet supported")
    fields: dict[str, torch.Tensor] = {}

    def add(path: str, axis: int = 0):
        value = runner
        for attr in path.split("."):
            value = getattr(value, attr)
        fields[path] = _view(value, slot, axis)

    for path in (
        "req_states.prompt_len.np",
        "req_states.prefill_len.np",
        "req_states.num_computed_prefill_tokens",
        "req_states.num_computed_tokens_np",
        "req_states.max_seq_len",
        "req_states.total_len.gpu",
        "req_states.num_computed_tokens.gpu",
        "req_states.last_sampled_tokens",
        "req_states.draft_tokens",
        "model_state.num_accepted_tokens_gpu",
    ):
        add(path)
    add("req_states.next_prefill_tokens", axis=1)
    fields["req_states.all_token_ids.gpu"] = runner.req_states.all_token_ids.gpu[
        slot : slot + 1, :token_count
    ]
    if model._align_mode:
        for name in (
            "_mamba_state_idx_gpu",
            "_mamba_src_col_gpu",
            "_mamba_src_off_gpu",
        ):
            add(f"model_state.{name}")
    sampler = runner.sampler
    if sampler is not None:
        if (
            sampler.__class__.__name__ != "Sampler"
            or sampler.trace_replay_state is not None
        ):
            raise ValueError("custom/watermark/replay sampler state is not supported")
        for path in (
            "sampler.sampling_states.seeds.np",
            "sampler.penalties_state.prompt_bin_mask",
            "sampler.penalties_state.output_bin_counts",
        ):
            add(path)
        if sampler.thinking_budget_state.enabled:
            for name in ("cached_last_start", "cached_last_end", "cached_scan_pos"):
                add(f"sampler.thinking_budget_state.{name}")
    speculator = runner.speculator
    if speculator is not None:
        if getattr(speculator, "draft_watermarker", None) is not None:
            raise ValueError("draft watermarker state is not supported")
        if speculator.draft_logits is not None:
            add("speculator.draft_logits")
    return fields


def pack_fields(fields: Mapping[str, torch.Tensor], token_count: int) -> bytes:
    """Serialize only bytes and a schema, never pickle live worker objects."""
    descriptors = []
    payloads = []
    for name, tensor in fields.items():
        raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        descriptors.append([name, str(tensor.dtype), list(tensor.shape), len(raw)])
        payloads.append(raw)
    header = json.dumps(
        {"version": 1, "token_count": token_count, "fields": descriptors},
        separators=(",", ":"),
    ).encode()
    return struct.pack("<I", len(header)) + header + b"".join(payloads)


def field_header(data: bytes) -> dict[str, Any]:
    if len(data) < 4:
        raise ValueError("truncated session worker metadata")
    (size,) = struct.unpack_from("<I", data)
    if size > len(data) - 4:
        raise ValueError("truncated session worker header")
    header = json.loads(data[4 : 4 + size])
    if header.get("version") != 1 or type(header.get("token_count")) is not int:
        raise ValueError("unsupported session worker metadata schema")
    return header


def unpack_fields(data: bytes, fields: Mapping[str, torch.Tensor]) -> None:
    """Validate every descriptor before writing any destination state."""
    header = field_header(data)
    (header_size,) = struct.unpack_from("<I", data)
    offset = 4 + header_size
    copies = []
    names = []
    for name, dtype, shape, size in header["fields"]:
        if name not in fields:
            raise ValueError("session worker state schema changed")
        target = fields[name]
        if (
            dtype != str(target.dtype)
            or shape != list(target.shape)
            or type(size) is not int
            or size != target.numel() * target.element_size()
            or offset + size > len(data)
        ):
            raise ValueError("session worker state shape or payload mismatch")
        names.append(name)
        if size:
            source = torch.frombuffer(
                bytearray(data[offset : offset + size]), dtype=target.dtype
            )
            copies.append((target, source.reshape(target.shape)))
        offset += size
    if (
        len(names) != len(set(names))
        or set(names) != set(fields)
        or offset != len(data)
    ):
        raise ValueError("incomplete or duplicate session worker state")
    for target, source in copies:
        target.copy_(source, non_blocking=False)


def _mamba_layer_specs(groups):
    from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

    layers = []
    for group_id, group in enumerate(groups):
        for name in group.layer_names:
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[name]
            if isinstance(spec, MambaSpec):
                layers.append((group_id, name, spec))
    return layers


class FlashSessionWorker:
    """Prepare/commit worker operations; scheduler ownership remains external."""

    def __init__(self, runner: Any, store: SessionColdStore):
        self.runner = runner
        self.store = store
        self.phases: dict[SessionKey, str] = {}
        self.captured_blocks: dict[SessionKey, tuple[tuple[int, ...], ...]] = {}

    def _settle_stream_output(self, slot: int) -> None:
        """Apply queued PP results through this slot before invalidating it.

        Consume only the necessary FIFO prefix; keep later unrelated receives
        queued. The native handler preserves ring length and orders its GPU
        postprocess after the receive event. Stale slot generations do not count.
        """
        pp = self.runner.pp_handler
        if pp is None:
            return
        prefix = 0
        for index, pending in enumerate(pp.queue):
            if pending is not None and np.any(
                (pending.idx_mapping_np == slot)
                & pending.need_sampled_mask
                & (pending.gen_at_receive_np == pp.req_idx_gen_np[slot])
            ):
                prefix = index + 1
        for _ in range(prefix):
            self.runner.update_pp_decode_requests()

    def prepare_streaming_update(self, new_request_data) -> int:
        """Materialize accepted Mamba state before native add_request resets it."""
        runner = self.runner
        slot = runner.req_states.req_id_to_index.get(new_request_data.req_id)
        if slot is None or new_request_data.num_computed_tokens == 0:
            return 0
        self._settle_stream_output(slot)
        computed = int(runner.req_states.num_computed_tokens.gpu[slot].item())
        if computed != new_request_data.num_computed_tokens:
            raise ValueError("streaming scheduler/worker token boundaries disagree")
        accepted = int(runner.model_state.num_accepted_tokens_gpu[slot].item())
        if accepted == 1:
            return 0
        if runner.vllm_config.cache_config.mamba_cache_mode != "none":
            raise ValueError("streaming Mamba materialization requires none mode")

        from vllm.model_executor.layers.mamba.mamba_utils import (
            get_conv_copy_spec,
            get_temporal_copy_spec,
            is_conv_state_dim_first,
        )

        groups = runner.kv_cache_config.kv_cache_groups
        if len(new_request_data.block_ids) != len(groups):
            raise ValueError("streaming update lacks complete block tables")
        mamba_layers = _mamba_layer_specs(groups)
        funcs = runner.model.get_mamba_state_copy_funcs(
            {spec.mamba_type for _, _, spec in mamba_layers}
        )
        kinds = {get_conv_copy_spec: "conv", get_temporal_copy_spec: "temporal"}
        context = runner.compilation_config.static_forward_context
        regions = []
        for group_id, name, spec in mamba_layers:
            if not 1 <= accepted <= 1 + spec.num_speculative_blocks:
                raise ValueError("streaming acceptance exceeds speculative cache")
            copies = funcs[spec.mamba_type]
            states = context[name].kv_cache
            if len(states) != len(copies) or any(f not in kinds for f in copies):
                raise ValueError("unsupported streaming Mamba state layout")
            regions.extend(
                (kinds[copy], state, new_request_data.block_ids[group_id])
                for state, copy in zip(states, copies, strict=True)
            )
        if not regions:
            raise ValueError("streaming acceptance has no local Mamba state")
        copied = materialize_mamba_boundary(
            regions, accepted, conv_dim_first=is_conv_state_dim_first()
        )
        runner.model_state.num_accepted_tokens_gpu[slot].fill_(1)
        return copied

    def _fence(self) -> None:
        # PP non-last ranks can have completed receives whose state updates
        # normally execute at the next forward. Apply them before taking a
        # checkpoint, preserving the ring length and per-slot generation guard.
        pp = self.runner.pp_handler
        if pp is not None:
            for _ in range(len(pp.queue)):
                self.runner.update_pp_decode_requests()
        if self.runner.device.type == "cuda":
            torch.cuda.synchronize(self.runner.device)

    def capture(self, key: SessionKey, blocks, computed_tokens: int) -> dict[str, Any]:
        if key in self.phases or any(
            k.request_id == key.request_id for k in self.phases
        ):
            raise ValueError("request already has a swap transaction")
        self._fence()
        slot = self.runner.req_states.req_id_to_index[key.request_id]
        state = self.runner.req_states
        if int(state.num_computed_tokens.gpu[slot].item()) != computed_tokens:
            raise ValueError("scheduler and worker token boundaries disagree")
        total = int(state.total_len.gpu[slot].item())
        metadata = pack_fields(slot_fields(self.runner, slot, total), total)
        size = self.store.capture(
            key, blocks, SwapBoundary(computed_tokens, 0), metadata
        )
        self.phases[key] = "captured"
        self.captured_blocks[key] = tuple(tuple(ids) for ids in blocks)
        return {"phase": "captured", "bytes": size}

    def commit_cold(self, key: SessionKey) -> dict[str, Any]:
        if self.phases.get(key) == "cold":
            return {"phase": "cold"}
        if self.phases.get(key) != "captured":
            raise ValueError("cold commit requires a completed checkpoint")
        self._fence()
        metadata = self.store.verify_hot(key, self.captured_blocks[key])
        slot = self.runner.req_states.req_id_to_index[key.request_id]
        total = int(self.runner.req_states.total_len.gpu[slot].item())
        if pack_fields(slot_fields(self.runner, slot, total), total) != metadata:
            raise RuntimeError("hot worker state changed before cold commit")
        if not self.runner._remove_request(key.request_id):
            raise ValueError("hot worker slot disappeared before cold commit")
        self.phases[key] = "cold"
        del self.captured_blocks[key]
        return {"phase": "cold"}

    def restore(self, key: SessionKey, new_request_data) -> dict[str, Any]:
        if self.phases.get(key) != "cold":
            raise ValueError("restore requires a cold session")
        if (
            new_request_data.req_id != key.request_id
            or key.request_id in self.runner.req_states.req_id_to_index
        ):
            raise ValueError("restore request identity or destination slot mismatch")
        self._fence()
        metadata = self.store.metadata(key)
        header = field_header(metadata)
        if not self.runner.req_states.free_indices:
            raise MemoryError("no hot worker slot available")
        try:
            # Native initialization rebuilds static parameters and sampling
            # constraints. Dynamic checkpoint state overwrites only afterwards.
            self.runner.add_requests(
                SimpleNamespace(scheduled_new_reqs=[new_request_data])
            )
            slot = self.runner.req_states.req_id_to_index[key.request_id]
            fields = slot_fields(self.runner, slot, header["token_count"])
            boundary, _ = self.store.restore(key, new_request_data.block_ids)
            if boundary.computed_tokens != new_request_data.num_computed_tokens:
                raise ValueError("restored scheduler token boundary changed")
            unpack_fields(metadata, fields)
            for name in ("prompt_len", "prefill_len"):
                getattr(self.runner.req_states, name).copy_to_uva()
            if self.runner.sampler is not None:
                self.runner.sampler.sampling_states.seeds.copy_to_uva()
            self.runner.block_tables.apply_staged_writes()
            self._fence()
            if pack_fields(fields, header["token_count"]) != metadata:
                raise RuntimeError("restored worker metadata verification failed")
        except Exception:
            # Destinations may be partly written. Keep cold data and remove the
            # worker slot; the coordinator must release its reserved hot blocks.
            self.runner._remove_request(key.request_id)
            raise
        self.phases[key] = "restored"
        return {"phase": "restored"}

    def retire(self, key: SessionKey, expected_phase: str) -> dict[str, Any]:
        if key not in self.phases and not self.store.contains(key):
            # A lost acknowledgement must be retryable after a successful drop.
            # This does not authorize any transfer or release of hot ownership.
            return {"phase": "retired"}
        if self.phases.get(key) != expected_phase:
            raise ValueError("checkpoint retirement phase mismatch")
        self.store.drop(key)
        del self.phases[key]
        self.captured_blocks.pop(key, None)
        return {"phase": "retired"}

    def abort_capture(self, key: SessionKey) -> dict[str, Any]:
        """Release partial all-rank preparation without touching hot ownership."""
        if key not in self.phases:
            self.store.abort_capture(key)
            return {"phase": "absent"}
        return self.retire(key, "captured")

    def rollback_restore(self, key: SessionKey) -> dict[str, Any]:
        """Keep the cold source when another rank could not restore."""
        self.store.quiesce()
        phase = self.phases.get(key)
        if phase == "cold":
            return {"phase": "cold"}
        if phase != "restored":
            raise ValueError("restore rollback requires a retained cold checkpoint")
        self._fence()
        self.runner._remove_request(key.request_id)
        self.phases[key] = "cold"
        return {"phase": "cold"}

    def dispatch(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Internal collective RPC; never a public user-supplied object loader."""
        key = SessionKey(**payload["key"])
        if operation == "capture":
            return self.capture(key, payload["block_ids"], payload["computed_tokens"])
        if operation == "commit_cold":
            return self.commit_cold(key)
        if operation == "restore":
            return self.restore(key, payload["new_request_data"])
        if operation == "retire":
            return self.retire(key, payload["expected_phase"])
        if operation == "abort_capture":
            return self.abort_capture(key)
        if operation == "rollback_restore":
            return self.rollback_restore(key)
        raise ValueError("unknown session swap operation")


def init_flash_session_worker(runner: Any, kv_caches) -> FlashSessionWorker | None:
    """Default off; reject unqualified combinations before allocating cold RAM."""
    capacity = int(os.environ.get("VLLM_FLASH_SESSION_SWAP_BYTES", "0"))
    if capacity == 0:
        return None
    if getattr(runner.kv_cache_config, "direct_host_num_blocks", None) is not None:
        raise ValueError(
            "Direct host KV requires native prefix caching, not session swap"
        )
    max_sessions = int(os.environ.get("VLLM_FLASH_SESSION_SWAP_MAX_SESSIONS", "100"))
    config = runner.vllm_config
    if (
        config.cache_config.enable_prefix_caching
        or config.cache_config.mamba_cache_mode not in ("none", "align")
        or config.kv_transfer_config is not None
        or config.lora_config is not None
        or config.parallel_config.data_parallel_size != 1
        or (
            config.speculative_config is not None
            and config.speculative_config.method != "mtp"
        )
    ):
        raise ValueError("unsupported configuration for Flash session swap")
    regions, empty = flash_cache_regions(
        runner.kv_cache_config,
        kv_caches,
        runner.compilation_config.static_forward_context,
    )
    if not any(region.name.startswith("host/") for region in regions):
        raise ValueError("Flash session swap requires existing QSA host KV")
    backend = os.environ.get("VLLM_FLASH_SESSION_SWAP_BACKEND", "copy")
    if backend == "native":
        from vllm.v1.worker.gpu.flash_session_swap import NativeSessionColdStore

        store_type = NativeSessionColdStore
    elif backend == "copy":
        store_type = SessionColdStore
    else:
        raise ValueError("unknown Flash session swap backend")
    store = store_type(
        regions,
        len(runner.kv_cache_config.kv_cache_groups),
        capacity,
        max_sessions,
        empty_groups=empty,
    )
    return FlashSessionWorker(runner, store)
