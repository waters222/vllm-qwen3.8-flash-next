"""Bounded first-prefill tracing for isolated session-correctness diagnostics.

CPU readback changes timing. A stable traced run cannot establish that an
untraced run is race-free. No weights or full tensor payloads are returned.
"""

import os
import re


def checked_host_registration(register, region, peek_error, emit):
    """Distinguish an inherited runtime error without clearing CUDA status."""
    before = int(peek_error())
    emit(
        dict(
            rank=region.rank,
            phase="before",
            cuda_error=before,
            bytes=region.total_size_bytes,
        )
    )
    if before:
        raise RuntimeError(f"Pre-existing CUDA error {before} before host registration")
    try:
        register(region)
    except Exception:
        emit(dict(rank=region.rank, phase="failed", cuda_error=int(peek_error())))
        raise
    emit(dict(rank=region.rank, phase="registered", cuda_error=int(peek_error())))


def install_host_registration_trace():
    import ctypes
    import importlib
    import json

    worker = importlib.import_module("vllm.v1.kv_offload.cpu.gpu_worker")
    runtime = ctypes.CDLL("libcudart.so.13")  # Exact diagnostic image uses CUDA 13.
    peek = runtime.cudaPeekAtLastError
    peek.argtypes = []
    peek.restype = ctypes.c_int
    original = worker.pin_mmap_region

    def traced(region):
        checked_host_registration(
            original,
            region,
            peek,
            lambda record: print(
                "COLD_REGISTER_TRACE " + json.dumps(record), flush=True
            ),
        )

    worker.pin_mmap_region = traced


if os.environ.get("FLASH_COLD_REGISTER_TRACE") == "1":
    install_host_registration_trace()


def manager_block_ids(kernel_ids, ratio):
    """Decode native split-block tables without confusing pool-local page IDs."""
    if ratio < 1 or len(kernel_ids) % ratio:
        raise ValueError("Invalid split-block table geometry")
    result = []
    for start in range(0, len(kernel_ids), ratio):
        chunk = kernel_ids[start : start + ratio]
        base = chunk[0] // ratio
        if chunk != list(range(base * ratio, (base + 1) * ratio)):
            raise ValueError("Noncanonical split-block table")
        result.append(base)
    return result


def compare_prefix_pages(reference, candidate):
    """Compare logical table entries; do not equate physical IDs with content."""
    if reference["boundary"] != candidate["boundary"]:
        raise ValueError("Cache-page snapshots use different token boundaries")

    def entries(snapshot):
        result = {}
        for group in snapshot["groups"]:
            for name, parts in group["layers"].items():
                for part, rows in enumerate(parts):
                    for row in rows:
                        key = (group["group_id"], name, part, row["logical_index"])
                        if key in result:
                            raise ValueError("Duplicate logical cache-page entry")
                        result[key] = (group, row)
        return result

    left, right = entries(reference), entries(candidate)
    summaries, mismatches = {}, []
    for key in sorted(left.keys() | right.keys()):
        a, b = left.get(key), right.get(key)
        group, row = a or b
        category = "before_boundary" if row["before_boundary"] else "working_or_suffix"
        label = f"{group['group_id']}:{key[1]}:part{key[2]}:{category}"
        summary = summaries.setdefault(
            label,
            dict(
                host_resident=group["host_resident"],
                spec_type=group["spec_type"],
                shared_entries=0,
                same_content=0,
                different_content=0,
                only_reference=0,
                only_candidate=0,
                changed_physical_ids=0,
            ),
        )
        if a is None or b is None:
            summary["only_candidate" if a is None else "only_reference"] += 1
            continue
        ag, ar = a
        bg, br = b
        layout_equal = all(
            ag[field] == bg[field]
            for field in ("host_resident", "spec_type", "block_size")
        ) and all(
            ar[field] == br[field]
            for field in ("shape", "dtype", "bytes", "before_boundary")
        )
        content_equal = layout_equal and ar["sha256"] == br["sha256"]
        summary["shared_entries"] += 1
        summary["same_content" if content_equal else "different_content"] += 1
        summary["changed_physical_ids"] += int(ar["physical_id"] != br["physical_id"])
        ac, bc = ar.get("prefill_convolution"), br.get("prefill_convolution")
        if ac is not None or bc is not None:
            conv = summary.setdefault(
                "prefill_convolution", dict(equal=0, different=0, missing=0)
            )
            if ac is None or bc is None:
                conv["missing"] += 1
            else:
                conv["equal" if layout_equal and ac == bc else "different"] += 1
        if not content_equal:
            mismatches.append(
                dict(
                    key=list(key),
                    category=category,
                    layout_equal=layout_equal,
                    reference_sha256=ar["sha256"],
                    candidate_sha256=br["sha256"],
                )
            )
    return dict(
        reference=reference["request_id"],
        candidate=candidate["request_id"],
        boundary=reference["boundary"],
        summaries=summaries,
        mismatches=mismatches,
        scope="Table entries only; missing/working pages need semantic interpretation",
    )


def index_prefix_page_requests(snapshots, expected, boundary=None):
    """Match public diagnostic IDs to native IDs without disabling randomization."""
    import re

    result = {}
    for snapshot in snapshots:
        if snapshot["bytes_hashed"] <= 0:
            raise ValueError("Prefix-page snapshot contains no hashed bytes")
        for request in snapshot["requests"]:
            if boundary is not None and request["boundary"] != boundary:
                continue
            internal = request["request_id"]
            matches = [
                name
                for name in expected
                if internal == name
                or re.fullmatch(re.escape(name) + r"-[0-9a-f]{8}", internal)
            ]
            if len(matches) != 1 or matches[0] in result:
                raise ValueError("Unexpected, ambiguous or duplicate prefix-page ID")
            result[matches[0]] = request
    if result.keys() != set(expected):
        raise ValueError("Prefix-page snapshot missed required requests")
    return result


def hash_page_chunks(view, max_chunk_bytes=8 * 1024**2):
    """Hash logical row-major bytes with bounded CPU copy/contiguous buffers."""
    import hashlib

    import torch

    if view.ndim == 0:
        view = view.reshape(1)
    row_bytes = view[0].numel() * view.element_size() if len(view) else 0
    if row_bytes <= 0 or row_bytes > max_chunk_bytes:
        raise ValueError("Prefix-page row exceeds chunk bound or is empty")
    digest = hashlib.sha256()
    chunk_rows = max_chunk_bytes // row_bytes
    for start in range(0, len(view), chunk_rows):
        data = (
            view[start : start + chunk_rows]
            .detach()
            .cpu()
            .contiguous()
            .view(torch.uint8)
        )
        digest.update(data.numpy())
        del data
    return digest.hexdigest()


def prefill_convolution_window(view, history_tokens, dim_first, capacity=None):
    """Canonicalize the GDN/PLE prefill history slice, excluding draft scratch."""
    if view.ndim != 2 or history_tokens <= 0:
        raise ValueError("Invalid prefill convolution window")
    canonical = view if dim_first else view.transpose(-1, -2)
    if capacity is not None:
        if not history_tokens <= capacity <= canonical.shape[-1]:
            raise ValueError("Invalid prefill convolution capacity")
        canonical = canonical[..., -capacity:]
    if history_tokens > canonical.shape[-1]:
        raise ValueError("Prefill history exceeds convolution storage")
    return canonical[..., :history_tokens]


def describe_request_batch(batch, positions):
    """Record actual GPU row boundaries/inputs, not optimistic CPU token counts."""
    offsets = batch.query_start_loc[: len(batch.req_ids) + 1].cpu().tolist()
    if (
        len(offsets) != len(batch.req_ids) + 1
        or offsets[0] != 0
        or any(a >= b for a, b in zip(offsets, offsets[1:]))
        or offsets[-1] > positions.numel()
        or positions.numel() > 16384
    ):
        raise ValueError("Invalid diagnostic request row boundaries")
    result = []
    for i, name in enumerate(batch.req_ids):
        start, end = offsets[i : i + 2]
        values = positions[start:end].detach().cpu().tolist()
        ranges = []
        for value in values:
            if ranges and ranges[-1][1] == value:
                ranges[-1][1] += 1
            else:
                ranges.append([value, value + 1])
        result.append(
            dict(
                request_id=name,
                row_start=start,
                row_end=end,
                position_ranges=ranges,
                input_ids_sha256=hash_page_chunks(batch.input_ids[start:end]),
                is_prefilling=bool(batch.is_prefilling_np[i]),
            )
        )
    return dict(requests=result, padding_rows=positions.numel() - offsets[-1])


def prefix_page_readback_budget(boundary, request_count):
    """Bound total observed bytes; hashing still copies at most 8MiB at once."""
    if type(request_count) is not int or not 1 <= request_count <= 4:
        raise ValueError("Prefix-page readback requires one to four requests")
    return (2 * max(2, request_count) if boundary >= 8192 else 2) * 1024**3


def snapshot_prefix_pages(
    runner, batch, boundary, max_bytes=2 * 1024**3, *, active_request_ids=None
):
    """Hash live allocated pages before target forward; never change cache state.

    Includes working/suffix pages, explicitly labeled. Equality of all allocated
    pages is not required: unused state may differ. No tensor bytes are returned.
    """
    import torch

    if type(boundary) is not int or not 0 < boundary < 32768 or max_bytes <= 0:
        raise ValueError("Invalid prefix-page diagnostic bounds")
    if active_request_ids is not None and (
        not active_request_ids or not set(active_request_ids) <= set(batch.req_ids)
    ):
        raise ValueError("Active cache audit requires requests in the current batch")
    if batch.positions.is_cuda:
        # CPU aliases of pinned RAM do not synchronize GPU writers on .cpu().
        # This intentionally perturbs timing and cannot prove race freedom.
        torch.cuda.synchronize(batch.positions.device)
    config = runner.kv_cache_config
    context = runner.compilation_config.static_forward_context
    tables = runner.block_tables
    records, pending, total_bytes = [], [], 0
    for index, request_id in enumerate(batch.req_ids):
        if active_request_ids is None:
            if (
                not batch.is_prefilling_np[index]
                or batch.num_computed_tokens_np[index] != boundary
            ):
                continue
        elif request_id not in active_request_ids:
            continue
        start = int(batch.query_start_loc_np[index])
        position = int(batch.positions[start].item())
        if active_request_ids is not None:
            if batch.is_prefilling_np[index] or position < boundary:
                raise ValueError("Active cache audit requires post-prefix decode")
        elif position != boundary:
            raise ValueError("Prefix-page CPU/GPU positions disagree")
        slot = int(batch.idx_mapping_np[index])
        if runner.req_states.req_id_to_index[request_id] != slot:
            raise ValueError("Prefix-page request slot disagrees with native runner")
        groups = []
        for group_id, group in enumerate(config.kv_cache_groups):
            count = int(tables.num_blocks.np[group_id, slot])
            if count > 4096:
                raise ValueError("Prefix-page table exceeds diagnostic bound")
            ids = manager_block_ids(
                tables.block_tables[group_id].gpu[slot, :count].cpu().tolist(),
                tables.blocks_per_kv_block[group_id],
            )
            num_blocks = (
                config.direct_host_num_blocks
                if group.host_resident
                else config.num_blocks
            )
            if any(page < 0 or page >= num_blocks for page in ids):
                raise ValueError("Prefix-page ID is outside its owning pool")
            layers = {}
            for name in group.layer_names:
                owner = context[name]
                value = getattr(owner, "_qsa_host_kv", None)
                if value is None:
                    value = owner.kv_cache
                tensors = value if isinstance(value, (list, tuple)) else (value,)
                parts = []
                for part_index, tensor in enumerate(tensors):
                    if not isinstance(tensor, torch.Tensor) or not tensor.numel():
                        raise ValueError(
                            "Prefix-page trace requires allocated layer caches"
                        )
                    if tensor.shape[0] % num_blocks:
                        raise ValueError(
                            "Prefix-page cache cannot be grouped into manager pages"
                        )
                    pages = (
                        tensor
                        if tensor.shape[0] == num_blocks
                        else tensor.unflatten(0, (num_blocks, -1))
                    )
                    rows = []
                    for logical_index, page in enumerate(ids):
                        if page == 0:
                            continue  # Native null page: not owned by this request.
                        view = pages[page]
                        size = view.numel() * view.element_size()
                        total_bytes += size
                        rows.append(
                            dict(
                                logical_index=logical_index,
                                physical_id=page,
                                bytes=size,
                                shape=list(view.shape),
                                dtype=str(view.dtype),
                                before_boundary=(logical_index + 1)
                                * tables.block_sizes[group_id]
                                <= boundary,
                            )
                        )
                        pending.append((view, rows[-1]))
                        kind = owner.__class__.__name__
                        if part_index == 0 and kind in (
                            "QwenGatedDeltaNetAttention",
                            "Qwen4ExpPLELayer",
                        ):
                            from vllm.model_executor.layers.mamba.mamba_utils import (
                                is_conv_state_dim_first,
                            )

                            is_ple = kind == "Qwen4ExpPLELayer"
                            history = (
                                owner.conv_state_len
                                if is_ple
                                else owner.conv_kernel_size - 1
                            )
                            capacity = (
                                history + owner.num_spec_tokens if is_ple else None
                            )
                            window = prefill_convolution_window(
                                view, history, is_conv_state_dim_first(), capacity
                            )
                            window_record = dict(
                                history_tokens=history,
                                shape=list(window.shape),
                                dtype=str(window.dtype),
                                bytes=window.numel() * window.element_size(),
                            )
                            rows[-1]["prefill_convolution"] = window_record
                            pending.append((window, window_record))
                            total_bytes += window_record["bytes"]
                    parts.append(rows)
                layers[name] = parts
            groups.append(
                dict(
                    group_id=group_id,
                    host_resident=group.host_resident,
                    spec_type=type(group.kv_cache_spec).__name__,
                    block_size=tables.block_sizes[group_id],
                    block_ids=ids,
                    layers=layers,
                )
            )
        running = getattr(
            getattr(runner, "model_state", None), "_mamba_state_idx_gpu", None
        )
        records.append(
            dict(
                request_id=request_id,
                slot=slot,
                boundary=boundary,
                groups=groups,
                mamba_running_column=int(running[slot].item())
                if running is not None
                else None,
            )
        )
    if total_bytes > max_bytes:
        raise ValueError(
            f"Prefix-page readback needs {total_bytes} bytes; limit is {max_bytes}"
        )
    for view, row in pending:
        row["sha256"] = hash_page_chunks(view)
    return dict(
        requests=records,
        bytes_hashed=total_bytes,
        max_bytes=max_bytes,
        max_chunk_bytes=8 * 1024**2,
        scope=(
            "Allocated pages at native pause; includes working/suffix pages"
            if active_request_ids is not None
            else "Allocated pages before target forward; includes working/suffix pages"
        ),
    )


class PrefillTraceWorkerExtension:
    """Named diagnostic RPCs; no callable/pickle payloads cross the engine API."""

    def set_prefill_cpu_probe(self, label, cpu=None):
        """Observe launches and bind only this disposable worker's main thread."""
        import json
        import resource
        import threading
        import time
        from pathlib import Path

        tid = threading.get_native_id()
        if not hasattr(self, "_prefill_cpu_probe"):
            original = self.model_runner.execute_model
            probe = dict(affinity=sorted(os.sched_getaffinity(0)), tid=tid, label=label)
            self._prefill_cpu_probe = probe

            def snapshot():
                usage = resource.getrusage(resource.RUSAGE_THREAD)
                fields = (
                    Path("/proc/thread-self/stat").read_text().rsplit(")", 1)[1].split()
                )
                return dict(
                    wall=time.perf_counter(), cpu_seconds=time.thread_time(),
                    cpu=int(fields[36]), voluntary=usage.ru_nvcsw,
                    involuntary=usage.ru_nivcsw,
                )

            def observed(*args, **kwargs):
                assert threading.get_native_id() == probe["tid"]
                before = snapshot()
                result = original(*args, **kwargs)
                after = snapshot()
                scheduler = args[0] if args else kwargs["scheduler_output"]
                row = dict(
                    label=probe["label"], rank=self.rank, before=before, after=after,
                    scheduled_tokens=dict(scheduler.num_scheduled_tokens),
                    affinity=sorted(os.sched_getaffinity(0)),
                )
                path = Path(f"/results/cpu-launch-rank{self.rank}.jsonl")
                with path.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                return result

            self.model_runner.execute_model = observed
        probe = self._prefill_cpu_probe
        assert tid == probe["tid"]
        if cpu is not None and (type(cpu) is not int or cpu not in probe["affinity"]):
            raise ValueError("Diagnostic CPU must be inside original thread affinity")
        os.sched_setaffinity(0, probe["affinity"] if cpu is None else {cpu})
        probe["label"] = label
        return dict(rank=self.rank, tid=tid, affinity=sorted(os.sched_getaffinity(0)),
                    original_affinity=probe["affinity"], label=label)

    def set_prefill_cpu_control(self, label, bound):
        """Use four distinct physical cores on the authorized development host."""
        return self.set_prefill_cpu_probe(
            label, (16 + 4 * self.rank) if bound else None
        )

    def set_prefill_utilization_hint(self, enabled):
        """Change only the disposable execution thread's utilization floor."""
        import threading

        from flash_worker_util_hint import UtilizationHint

        if not hasattr(self, "_prefill_util_hint"):
            hint = self._prefill_util_hint = UtilizationHint()
            execute = self.model_runner.execute_model

            def checked_execute(*args, **kwargs):
                if threading.get_native_id() != hint.tid:
                    raise RuntimeError("Utilization hint is not on execution thread")
                return execute(*args, **kwargs)

            self.model_runner.execute_model = checked_execute
        return dict(rank=self.rank, **self._prefill_util_hint.set(enabled))

    def prefill_checkpoint_status(self):
        """Read native checkpoint activation without changing cache state."""
        from vllm.model_executor.layers.mamba.ops.flash_next_prefill_checkpoint import (
            CHECKPOINT_STATS,
        )

        return dict(rank=getattr(self, "rank", None), **CHECKPOINT_STATS)

    def qsa_prefill_metadata_status(self):
        """Read eager metadata-reuse counters without touching GPU state."""
        import sys

        module = sys.modules.get("vllm.models.qwen4_exp.nvidia.qsa")
        return dict(
            rank=getattr(self, "rank", None),
            **getattr(module, "_QSA_PREFILL_METADATA_STATS", {}),
        )

    def set_qsa_prefill_metadata_cache(self, enabled):
        """Toggle only eager QSA metadata reuse for a disposable paired control."""
        models = [self.model_runner.model]
        speculator = getattr(self.model_runner, "speculator", None)
        if speculator is not None:
            models.append(speculator.model)
        layers = {}
        for model in models:
            for layer in model.modules():
                if hasattr(layer, "_qsa_prefill_metadata_cache"):
                    layers[id(layer)] = layer
        for layer in layers.values():
            layer._qsa_prefill_metadata_cache = bool(enabled)
        return dict(
            layers=len(layers),
            enabled=bool(enabled),
            counters=self.qsa_prefill_metadata_status(),
        )

    def qsa_verification_staging_status(self):
        """Read capture-time dispatch receipts without touching GPU state."""
        import sys

        module = sys.modules.get("vllm.models.qwen4_exp.nvidia.ops.qsa_verify_staging")
        reported = getattr(module, "_REPORTED", ())
        return dict(
            rank=getattr(self, "rank", None),
            active_shapes=sorted(
                [list(item[1:]) for item in reported if item[0] == "active_shape"]
            ),
        )

    def snapshot_cuda_memory(self):
        """Read allocator counters without resetting peaks or emptying caches."""
        import torch

        device = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(device)
        stats = torch.cuda.memory_stats(device)
        keys = (
            "allocated_bytes.all.current",
            "allocated_bytes.all.peak",
            "reserved_bytes.all.current",
            "reserved_bytes.all.peak",
            "inactive_split_bytes.all.current",
            "num_alloc_retries",
            "num_ooms",
        )
        return dict(
            rank=self.rank,
            device=device,
            free_bytes=free,
            total_bytes=total,
            allocator={key: stats.get(key) for key in keys},
            scope="Endpoint device free memory and process-lifetime allocator peaks; "
            "not minimum free memory or a production headroom guarantee. "
            "No peak reset, synchronization or cache emptying requested.",
        )

    def arm_native_load_failure(self):
        from pathlib import Path

        from session_transfer_audit import install_native_load_failure

        from vllm.distributed.kv_transfer import get_kv_transfer_group
        from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker

        if self.model_runner.req_states.req_id_to_index:
            raise ValueError("Failure injection requires an idle diagnostic worker")
        connector = get_kv_transfer_group().connector_worker
        if not isinstance(connector.worker, CPUOffloadingWorker):
            raise ValueError("Failure injection requires native CPU offload")
        return install_native_load_failure(
            connector, self.rank, Path("/results/native-load-failure-rank3.json")
        )

    def set_qsa_prefill_staging(self, enabled):
        """Toggle only idle diagnostic modules; None restores original flags."""
        import torch

        if os.environ.get("FLASH_PREFILL_STAGING_AB") != "1":
            raise ValueError("In-process staging toggle requires the isolated AB gate")
        if enabled is not None and type(enabled) is not bool:
            raise ValueError("Staging choice must be a boolean or restoration")
        runner = self.model_runner
        if runner.req_states.req_id_to_index:
            raise ValueError("Staging toggle requires no resident worker requests")
        torch.cuda.synchronize()
        saved = getattr(self, "_qsa_staging_ab", None)
        if enabled is None:
            if saved is None:
                raise ValueError("No staging experiment to restore")
            for module, original in saved:
                module._qsa_stage_all_prefill = original
            del self._qsa_staging_ab
            return dict(rank=self.rank, restored=True, modules=len(saved))
        models = (self.get_model(), runner.get_draft_model())
        groups = [
            [
                m
                for m in model.modules()
                if m.__class__.__name__ == "Qwen4ExpQSAAttention"
            ]
            if model is not None
            else []
            for model in models
        ]
        modules = [m for group in groups for m in group]
        if (
            list(map(len, groups)) != [12, 1]
            or len({id(m) for m in modules}) != 13
            or any(
                not m._qsa_kv_offload or type(m._qsa_stage_all_prefill) is not bool
                for m in modules
            )
        ):
            raise ValueError(
                "Staging AB requires twelve target and one draft QSA owner"
            )
        if saved is None:
            saved = [(m, m._qsa_stage_all_prefill) for m in modules]
            self._qsa_staging_ab = saved
        elif [id(m) for m, _ in saved] != [id(m) for m in modules]:
            raise ValueError("Staging AB module identities changed")
        for module in modules:
            module._qsa_stage_all_prefill = enabled
        return dict(rank=self.rank, enabled=enabled, target=12, draft=1)

    def install_native_copy_audit(self):
        from session_transfer_audit import NativeCopyAudit

        from vllm.distributed.kv_transfer import get_kv_transfer_group
        from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker

        if hasattr(self, "_native_copy_audit"):
            raise ValueError("Native copy audit is already installed")
        worker = get_kv_transfer_group().connector_worker.worker
        if not isinstance(worker, CPUOffloadingWorker):
            raise ValueError("Native copy audit requires CPU offloading")
        audit = NativeCopyAudit()
        result = audit.install(worker)
        self._native_copy_audit = audit
        return result

    def collect_native_copy_audit(self):
        result = self._native_copy_audit.remove()
        del self._native_copy_audit
        result["rank"] = self.rank
        return result

    def install_active_cache_audit(self):
        runner = self.model_runner
        if hasattr(self, "_active_cache_audit") or hasattr(self, "_prefix_shape_trace"):
            raise ValueError("Active cache audit must not overlap another batch hook")
        original = runner.prepare_inputs
        state = dict(
            original=original, shadowed="prepare_inputs" in vars(runner), batch=None
        )

        def prepare(*args, **kwargs):
            batch = original(*args, **kwargs)
            state["batch"] = batch
            return batch

        runner.prepare_inputs = prepare
        self._active_cache_audit = state
        return dict(installed=True, timing_perturbed=True)

    def snapshot_active_cache_audit(self, external_request_id):
        state = self._active_cache_audit
        batch = state["batch"]
        if batch is None:
            raise ValueError("Active cache audit has no native batch")
        names = {
            name
            for name in batch.req_ids
            if re.fullmatch(re.escape(external_request_id) + r"(?:-[0-9a-f]{8})?", name)
        }
        if len(names) != 1:
            raise ValueError("Active cache audit requires one unambiguous request")
        snapshot = snapshot_prefix_pages(
            self.model_runner, batch, 5664, active_request_ids=names
        )
        snapshot["rank"] = self.rank
        return snapshot

    def remove_active_cache_audit(self):
        state = self._active_cache_audit
        runner = self.model_runner
        if state["shadowed"]:
            runner.prepare_inputs = state["original"]
        else:
            del runner.prepare_inputs
        del self._active_cache_audit
        return dict(removed=True)

    def install_prefix_shape_trace(self, prompt_tokens, trace_pages=False):
        """Record bounded packed position ranges, not activation tensors or IDs."""
        import torch

        if (
            type(prompt_tokens) is not int
            or not 1 <= prompt_tokens <= 32768
            or (trace_pages and prompt_tokens <= 5664)
            or hasattr(self, "_prefix_shape_trace")
        ):
            raise ValueError("Invalid or already-installed prefix shape trace")
        layer = next(
            (
                module
                for module in self.get_model().modules()
                if module.__class__.__name__ == "Qwen4ExpDecoderLayer"
            ),
            None,
        )
        if layer is None:
            raise ValueError("Prefix shape trace requires a local Flash decoder")
        state = dict(
            phase=None,
            records={},
            pages={},
            seen_pages=set(),
            batch=None,
            prompt_tokens=prompt_tokens,
            trace_pages=trace_pages,
            page_boundaries=(5664,),
        )
        if trace_pages:
            runner = self.model_runner
            if not hasattr(runner, "req_states") or not hasattr(runner, "block_tables"):
                raise ValueError("Prefix-page trace requires native V2 runner")
            original = runner.prepare_inputs
            state["prepare_inputs_restore"] = (
                runner,
                original,
                "prepare_inputs" in runner.__dict__,
            )

            def prepare(*args, **kwargs):
                batch = original(*args, **kwargs)
                state["batch"] = batch
                return batch

            runner.prepare_inputs = prepare

        def before(module, args, kwargs):
            positions = kwargs.get("positions")
            if state["phase"] is None or positions is None:
                return
            if positions.is_cuda and torch.cuda.is_current_stream_capturing():
                raise ValueError("Prefix shape trace requires eager execution")
            if positions.ndim == 2:
                positions = positions[0]
            if positions.ndim != 1 or positions.numel() > 16384:
                raise ValueError("Prefix shape trace exceeds bounded position layout")
            values = positions.detach().cpu().tolist()
            values = [value for value in values if 0 <= value < prompt_tokens]
            if not values and not trace_pages:
                return
            for boundary in state["page_boundaries"] if trace_pages else ():
                if boundary not in values:
                    continue
                snapshot = snapshot_prefix_pages(
                    self.model_runner,
                    state["batch"],
                    boundary,
                    max_bytes=prefix_page_readback_budget(
                        boundary, len(state["batch"].req_ids)
                    ),
                )
                for request in snapshot["requests"]:
                    key = (state["phase"], request["request_id"], boundary)
                    if key in state["seen_pages"]:
                        raise ValueError(
                            "Repeated prefix-page boundary for one request"
                        )
                    state["seen_pages"].add(key)
                state["pages"].setdefault(state["phase"], []).append(snapshot)
            ranges = []
            for value in values:
                if ranges and value == ranges[-1][1]:
                    ranges[-1][1] += 1
                else:
                    ranges.append([value, value + 1])
            rows = state["records"][state["phase"]]
            max_steps = 512 if trace_pages else 64
            if len(rows) >= max_steps:
                raise ValueError(
                    f"Prefix shape trace exceeds{max_steps} steps per phase"
                )
            rows.append(
                dict(
                    ranges=ranges,
                    prefill_rows=len(values),
                    total_rows=positions.numel(),
                )
            )
            if trace_pages:
                rows[-1]["request_batch"] = describe_request_batch(
                    state["batch"], positions
                )

        state["handle"] = layer.register_forward_pre_hook(before, with_kwargs=True)
        self._prefix_shape_trace = state
        return dict(
            installed=True,
            max_steps=512 if trace_pages else 64,
            includes_decode=trace_pages,
            timing_perturbed=True,
        )

    def begin_prefix_shape_trace(self, phase, page_boundary=None):
        state = self._prefix_shape_trace
        boundaries = (
            (5664,)
            if page_boundary is None
            else (page_boundary,)
            if type(page_boundary) is int
            else page_boundary
        )
        if page_boundary is not None and (
            not state["trace_pages"]
            or phase is None
            or not isinstance(boundaries, (tuple, list))
            or not 1 <= len(boundaries) <= 3
            or any(
                type(boundary) is not int or not 0 < boundary < state["prompt_tokens"]
                for boundary in boundaries
            )
            or list(boundaries) != sorted(set(boundaries))
        ):
            raise ValueError("Invalid explicit prefix-page boundary")
        if phase is not None:
            if not isinstance(phase, str) or not phase or len(phase) > 64:
                raise ValueError("Invalid prefix trace phase")
            if phase in state["records"] or len(state["records"]) >= 32:
                raise ValueError("Duplicate or excessive prefix trace phase")
            state["records"][phase] = []
        state["phase"] = phase
        state["page_boundaries"] = tuple(boundaries)
        return dict(phase=phase)

    def collect_prefix_shape_trace(self):
        state = self._prefix_shape_trace
        state["handle"].remove()
        if "prepare_inputs_restore" in state:
            runner, original, shadowed = state["prepare_inputs_restore"]
            if shadowed:
                runner.prepare_inputs = original
            else:
                del runner.prepare_inputs
        del self._prefix_shape_trace
        return dict(
            rank=getattr(self, "rank", None),
            phases=state["records"],
            pages=state["pages"],
            timing_perturbed=True,
        )

    def install_stable_qsa_selection(self, stable_ties=False):
        """Canonicalize block order; optionally choose deterministic score ties."""
        import importlib

        module = importlib.import_module("vllm.models.qwen4_exp.nvidia.ops.qsa_indexer")
        if hasattr(self, "_stable_qsa_selection"):
            raise ValueError("Stable QSA selection is already installed")
        state = dict(native=module._topk, calls=0)

        def ordered(logits, visible, token_topk, ratio, indices, workspace):
            state["native"](logits, visible, token_topk, ratio, indices, workspace)
            if stable_ties:
                indices.copy_(stable_qsa_topk(logits, visible, indices.shape[1]))
            indices.copy_(canonical_qsa_selection(indices))
            state["calls"] += 1

        self._stable_qsa_selection = state
        module._topk = ordered
        return dict(installed=True, diagnostic_only=True, stable_ties=stable_ties)

    def remove_stable_qsa_selection(self):
        import importlib

        module = importlib.import_module("vllm.models.qwen4_exp.nvidia.ops.qsa_indexer")
        state = self._stable_qsa_selection
        module._topk = state["native"]
        del self._stable_qsa_selection
        return dict(calls=state["calls"], removed=True)

    def install_session_prefill_trace(self, prompt_tokens, repeats):
        return install_prefill_trace(self.get_model(), prompt_tokens, repeats)

    def install_prefix_prefill_trace(self, repeats):
        return install_prefill_trace(self.get_model(), 32, repeats, source_tokens=2048)

    def install_prompt_chunk_trace(self, prompt_tokens, repeats):
        return install_prompt_chunk_trace(self.get_model(), prompt_tokens, repeats)

    def begin_prompt_chunk_trace(self, run):
        state = self.get_model()._flash_session_prefill_trace
        if state.get("mode") != "prompt_chunks" or not 0 <= run < state["repeats"]:
            raise ValueError("Invalid prompt trace run")
        if run != state["run"] + 1:
            raise ValueError("Prompt trace runs must advance exactly once")
        state["run"] = run
        return dict(run=run)

    def collect_session_prefill_trace(self):
        return collect_prefill_trace(self.get_model())

    def install_stable_expert_layout(self):
        """Diagnostic only: canonicalize native no-EP sorting, not routing."""
        import importlib

        marlin = importlib.import_module(
            "vllm.model_executor.layers.fused_moe.experts.marlin_moe"
        )
        if hasattr(self, "_stable_expert_layout"):
            raise ValueError("Stable expert layout is already installed")
        if marlin.flash_moe_layout_sm86._ENABLED:
            raise ValueError("Stable layout diagnostic requires native layout kernels")
        native = marlin.moe_align_block_size
        wrapper = marlin.flash_moe_layout_sm86.moe_align_block_size
        state = dict(calls=0, native=native, wrapper=wrapper)

        def stable_align(topk_ids, block_size, num_experts, expert_map=None, **kwargs):
            if expert_map is not None:
                raise ValueError("Stable layout diagnostic does not support EP")
            layout = native(topk_ids, block_size, num_experts, expert_map, **kwargs)
            result = canonical_expert_layout(
                layout, block_size, topk_ids.numel(), num_experts
            )
            state["calls"] += 1
            return result

        self._stable_expert_layout = state
        marlin.moe_align_block_size = stable_align
        marlin.flash_moe_layout_sm86.moe_align_block_size = stable_align
        return dict(installed=True, diagnostic_only=True)

    def install_mlp_numerical_control(self):
        from session_numerical_control import install

        if hasattr(self, "_mlp_numerical_remove"):
            raise ValueError("Numerical control already installed")
        self._mlp_numerical_remove = install()
        return dict(installed=True, diagnostic_only=True)

    def remove_mlp_numerical_control(self):
        result = self._mlp_numerical_remove()
        del self._mlp_numerical_remove
        return result

    def remove_stable_expert_layout(self):
        import importlib

        marlin = importlib.import_module(
            "vllm.model_executor.layers.fused_moe.experts.marlin_moe"
        )
        state = self._stable_expert_layout
        marlin.moe_align_block_size = state["native"]
        marlin.flash_moe_layout_sm86.moe_align_block_size = state["wrapper"]
        del self._stable_expert_layout
        return dict(calls=state["calls"], removed=True)


def canonical_qsa_selection(indices):
    """Diagnostic integer ordering with invalid slots kept after valid blocks."""
    import torch

    sentinel = torch.iinfo(indices.dtype).max
    ordered = torch.where(indices >= 0, indices, sentinel).sort(dim=-1).values
    return torch.where(ordered == sentinel, -1, ordered)


def stable_qsa_topk(logits, visible, count):
    """Diagnostic full sort: select score ties by ascending logical block index."""
    import torch

    columns = torch.arange(logits.shape[1], device=logits.device)
    masked = torch.where(columns < visible[:, None], logits, -torch.inf)
    selected = masked.argsort(dim=-1, descending=True, stable=True)[:, :count]
    valid = torch.arange(selected.shape[1], device=logits.device) < visible[:, None]
    selected = torch.where(valid, selected, -1).to(torch.int32)
    return torch.nn.functional.pad(selected, (0, count - selected.shape[1]), value=-1)


def canonical_expert_layout(layout, block_size, num_routes, num_experts):
    """Sort token indices within native expert groups, preserving padding/counts.

    Only used by the no-EP correctness diagnostic. It allocates/sorts on the
    input device and is not a serving or performance candidate.
    """
    import torch

    ids, experts, total = layout
    if block_size <= 0 or num_routes <= 0 or num_experts <= 0:
        raise ValueError("Invalid native expert layout geometry")
    positions = torch.arange(ids.numel(), device=ids.device)
    active = positions < total
    groups = experts.to(torch.int64).repeat_interleave(block_size)[: ids.numel()]
    base = num_routes + 1
    keys = torch.where(active, groups * base + ids.to(torch.int64), num_experts * base)
    ordered = torch.sort(keys).values.remainder(base)
    ordered = torch.where(active, ordered, num_routes).to(ids.dtype)
    return ordered, experts, total


def install_prompt_chunk_trace(model, prompt_tokens, repeats):
    """Hash every target prefill chunk, keyed by positions and explicit request.

    This diagnostic reads activations back to CPU and changes execution timing.
    Different chunk schedules are reported as unmatched, never compared as if
    they were equivalent. No activation tensors are retained between forwards.
    """
    import hashlib
    from functools import partial

    import torch

    if not 1 <= prompt_tokens <= 8192 or not 2 <= repeats <= 16:
        raise ValueError("Prompt trace exceeds diagnostic bounds")
    if hasattr(model, "_flash_session_prefill_trace"):
        raise ValueError("prefill trace already installed")
    layers = [
        (name, layer)
        for name, layer in model.named_modules()
        if layer.__class__.__name__ == "Qwen4ExpDecoderLayer"
    ]
    if not layers:
        raise ValueError("prefill trace requires local Flash decoder layers")
    state = dict(
        mode="prompt_chunks",
        run=-1,
        repeats=repeats,
        handles=[],
        method_restores=[],
        baseline_bytes=0,
        records={},
        components={},
        active={},
        signatures={},
        seen=set(),
    )
    model._flash_session_prefill_trace = state

    def summarize(name, key, phase, tensors):
        result = []
        for index, tensor in enumerate(tensors):
            signature = None
            row = dict(present=tensor is not None)
            if tensor is not None:
                value = tensor.detach().to("cpu").contiguous()
                digest = hashlib.sha256(value.view(torch.uint8).numpy().tobytes())
                signature = (tuple(value.shape), str(value.dtype), digest.hexdigest())
                row.update(
                    shape=list(value.shape),
                    dtype=str(value.dtype),
                    sha256=signature[-1],
                    finite=bool(torch.isfinite(value).all()),
                )
            baseline_key = (name, key, phase, index)
            if state["run"] == 0:
                state["signatures"][baseline_key] = signature
            matched = baseline_key in state["signatures"]
            row.update(
                matched_baseline=matched,
                exact=(signature == state["signatures"][baseline_key])
                if matched
                else None,
            )
            result.append(row)
        return result

    def before(name, module, args, kwargs):
        state["active"].pop(name, None)
        positions = kwargs.get("positions")
        if state["run"] < 0 or positions is None or positions.ndim not in (1, 2):
            return
        if positions.is_cuda and torch.cuda.is_current_stream_capturing():
            raise ValueError("Prompt trace requires eager execution")
        values = positions.detach().to("cpu").contiguous()
        if not values.numel() or values.min() < 0 or values.max() >= prompt_tokens:
            return  # Exclude decode/speculative verification beyond the prompt.
        key = hashlib.sha256(values.numpy().tobytes()).hexdigest()
        identity = (name, state["run"], key)
        if identity in state["seen"]:
            raise ValueError("Repeated prompt positions in one diagnostic request")
        records = state["records"].setdefault(name, [])
        if sum(row["run"] == state["run"] for row in records) >= 16:
            raise ValueError("Prompt trace exceeds 16 chunks per layer/request")
        state["seen"].add(identity)
        row = dict(
            run=state["run"],
            position_start=int(values.min()),
            position_end=int(values.max()),
            position_shape=list(values.shape),
            position_sha256=key,
            inputs=summarize(
                name,
                key,
                "input",
                [
                    kwargs.get(k)
                    for k in ("hidden_states", "prev_block_output", "prev_injection")
                ],
            ),
        )
        records.append(row)
        state["active"][name] = (key, row)

    def after(name, module, args, kwargs, output):
        active = state["active"].pop(name, None)
        if active is not None:
            key, row = active
            tensors = [output] if isinstance(output, torch.Tensor) else output
            row["outputs"] = summarize(name, key, "output", tensors)

    for name, layer in layers:
        state["handles"].extend(
            [
                layer.register_forward_pre_hook(
                    partial(before, name), with_kwargs=True
                ),
                layer.register_forward_hook(partial(after, name), with_kwargs=True),
            ]
        )
    return dict(layers=[name for name, _ in layers], max_chunks=16, repeats=repeats)


def install_prefill_trace(model, prompt_tokens, repeats, *, source_tokens=None):
    import hashlib

    import torch

    if not 1 <= prompt_tokens <= 512 or not 2 <= repeats <= 16:
        raise ValueError("prefill trace exceeds diagnostic bounds")
    if source_tokens is not None and not prompt_tokens <= source_tokens <= 2048:
        raise ValueError("prefill source trace exceeds diagnostic bounds")
    if hasattr(model, "_flash_session_prefill_trace"):
        raise ValueError("prefill trace already installed")
    layers = [
        (name, module)
        for name, module in model.named_modules()
        if module.__class__.__name__ == "Qwen4ExpDecoderLayer"
    ]
    if not layers:
        raise ValueError("prefill trace requires local Flash decoder layers")
    state = dict(
        handles=[],
        records={},
        baselines={},
        baseline_bytes=0,
        active={},
        components={},
        component_active={},
        method_restores=[],
        source_hashes={},
    )
    model._flash_session_prefill_trace = state

    def summarize(name, phase, tensors):
        baseline_key = (name, phase)
        snapshots = [
            None
            if t is None
            else t[:prompt_tokens].detach().to("cpu").contiguous().clone()
            for t in tensors
        ]
        if baseline_key not in state["baselines"]:
            size = sum(t.nbytes for t in snapshots if t is not None)
            if size + state["baseline_bytes"] > 64 * 1024**2:
                raise ValueError("prefill trace baseline exceeds 64 MiB per rank")
            state["baselines"][baseline_key] = snapshots
            state["baseline_bytes"] += size
        previous = state["baselines"][baseline_key]
        rows = []
        for index, (old, current, source) in enumerate(
            zip(previous, snapshots, tensors, strict=True)
        ):
            if current is None or old is None:
                rows.append(dict(exact=old is None and current is None))
                continue
            raw = current.view(torch.uint8).numpy().tobytes()
            same_layout = old.shape == current.shape and old.dtype == current.dtype
            finite = bool(torch.isfinite(current).all())
            rows.append(
                dict(
                    shape=list(current.shape),
                    source_shape=list(source.shape),
                    source_stride=list(source.stride()),
                    dtype=str(current.dtype),
                    sha256=hashlib.sha256(raw).hexdigest(),
                    finite=finite,
                    exact=same_layout
                    and torch.equal(old.view(torch.uint8), current.view(torch.uint8)),
                    max_abs_difference=(
                        float((old.float() - current.float()).abs().max())
                        if same_layout and finite and bool(torch.isfinite(old).all())
                        else None
                    ),
                )
            )
            if source_tokens is not None:
                # Retain only the digest and bounded sample, not full activations.
                full = source[:source_tokens].detach().to("cpu").contiguous()
                digest = hashlib.sha256(
                    full.view(torch.uint8).numpy().tobytes()
                ).hexdigest()
                signature = (tuple(full.shape), str(full.dtype), digest)
                key = (name, phase, index)
                reference = state["source_hashes"].setdefault(key, signature)
                rows[-1].update(
                    sample_exact=rows[-1]["exact"],
                    exact=signature == reference,
                    source_sha256=digest,
                    hashed_shape=list(full.shape),
                    max_abs_difference_scope="leading sampled rows only",
                    finite=bool(torch.isfinite(full).all()),
                )
        return rows

    def before(name, module, args, kwargs):
        state["active"][name] = False
        if len(state["records"].get(name, [])) >= repeats:
            return
        positions = kwargs.get("positions")
        if positions is None or positions.ndim not in (1, 2):
            return
        if positions.ndim == 2 and positions.shape[0] != 3:
            return
        limit = source_tokens or 512
        if not prompt_tokens <= positions.shape[-1] <= limit:
            return
        if source_tokens is not None and positions.shape[-1] != source_tokens:
            return
        if positions.is_cuda and torch.cuda.is_current_stream_capturing():
            return
        prefix = positions[..., :prompt_tokens]
        if not torch.equal(
            prefix,
            torch.arange(prompt_tokens, device=positions.device).expand_as(prefix),
        ):
            return
        record = dict(
            inputs=summarize(
                name,
                "input",
                [
                    kwargs.get(k)
                    for k in ("hidden_states", "prev_block_output", "prev_injection")
                ],
            )
        )
        state["records"].setdefault(name, []).append(record)
        state["active"][name] = True

    def after(name, module, args, kwargs, output):
        if state["active"].pop(name, False):
            state["records"][name][-1]["outputs"] = summarize(name, "output", output)

    from functools import partial

    def tensors_in(value):
        if isinstance(value, torch.Tensor):
            return [value]
        if isinstance(value, dict):
            return [t for item in value.values() for t in tensors_in(item)]
        if isinstance(value, (tuple, list)):
            return [t for item in value for t in tensors_in(item)]
        return []

    def component_before(parent, name, module, args, kwargs):
        active = state["active"].get(parent, False)
        state["component_active"][name] = active
        if active:
            state["components"].setdefault(name, []).append(
                dict(inputs=summarize(name, "input", tensors_in((args, kwargs))))
            )

    def component_after(name, module, args, kwargs, output):
        if state["component_active"].pop(name, False):
            state["components"][name][-1]["outputs"] = summarize(
                name, "output", tensors_in(output)
            )

    component_names = []

    def trace_method(parent, key, owner, method_name):
        method = getattr(owner, method_name)
        had_override = method_name in vars(owner)
        previous_override = vars(owner).get(method_name)

        def traced(*args, **kwargs):
            if not state["active"].get(parent, False):
                return method(*args, **kwargs)
            record = dict(inputs=summarize(key, "input", tensors_in((args, kwargs))))
            state["components"].setdefault(key, []).append(record)
            output = method(*args, **kwargs)
            record["outputs"] = summarize(key, "output", tensors_in(output))
            return output

        setattr(owner, method_name, traced)
        state["method_restores"].append(
            (owner, method_name, had_override, previous_override)
        )
        component_names.append(key)

    for name, layer in layers:
        state["handles"].append(
            layer.register_forward_pre_hook(partial(before, name), with_kwargs=True)
        )
        state["handles"].append(
            layer.register_forward_hook(partial(after, name), with_kwargs=True)
        )
        # The first observed divergence is layer 1's MLP. Keep fine tracing
        # local to this block; do not multiply full-model readback overhead.
        if name.rsplit(".", 1)[-1] == "1" and hasattr(layer, "mlp"):
            for suffix, component in layer.mlp.named_modules():
                if "." in suffix and suffix not in {
                    "shared_expert.gate_up_proj",
                    "shared_expert.down_proj",
                    "shared_expert.act_fn",
                }:
                    continue
                key = name + ".mlp" + ("." + suffix if suffix else "")
                component_names.append(key)
                state["handles"].extend(
                    [
                        component.register_forward_pre_hook(
                            partial(component_before, name, key), with_kwargs=True
                        ),
                        component.register_forward_hook(
                            partial(component_after, key), with_kwargs=True
                        ),
                    ]
                )
            experts = getattr(layer.mlp, "experts", None)
            if experts is not None:
                for owner, method_name, suffix in (
                    (getattr(experts, "router", None), "select_experts", "router"),
                    (
                        getattr(experts, "routed_experts", None),
                        "forward_modular",
                        "routed_experts",
                    ),
                    (experts, "_maybe_reduce_final_output", "final_reduce"),
                ):
                    if owner is not None and hasattr(owner, method_name):
                        trace_method(name, name + ".mlp." + suffix, owner, method_name)
    return dict(
        layers=[name for name, _ in layers],
        components=component_names,
        max_repeats=repeats,
        sample_tokens=prompt_tokens,
        source_tokens=source_tokens,
    )


def collect_prefill_trace(model):
    state = model._flash_session_prefill_trace
    for handle in state["handles"]:
        handle.remove()
    for owner, name, had_override, previous in reversed(state["method_restores"]):
        if had_override:
            setattr(owner, name, previous)
        else:
            delattr(owner, name)
    del model._flash_session_prefill_trace
    return dict(
        baseline_bytes=state["baseline_bytes"],
        layers=state["records"],
        components=state["components"],
        timing_perturbed=True,
    )
