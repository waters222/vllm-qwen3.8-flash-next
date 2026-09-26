"""Bounded, timing-perturbing audit of native worker-private offload copies."""

import hashlib
import json
import os

from session_prefill_trace import hash_page_chunks

MAX_COPY_RECORDS = 32768


def validate_native_load_failure(args):
    mode = getattr(args, "native_load_failure", None)
    if mode is None:
        return
    if (
        mode not in ("inject", "recovery")
        or not all(
            getattr(args, flag, False)
            for flag in (
                "native_prefix",
                "tp4",
                "serving_decode",
                "private_cold_buffers",
                "balanced_prefix_prefill",
                "untraced_balanced_prefix",
            )
        )
        or getattr(args, "prefix_concurrency", 1) != 2
        or getattr(args, "prefix_prefill_budget", None) != 2048
        or any(
            getattr(args, flag, False)
            for flag in (
                "no_mtp",
                "performance_repeats",
                "performance_iteration_details",
                "profile_prefix_prefill",
                "prefill_staging_ab",
                "continuations",
                "active_cancel",
                "audit_active_cancel",
                "expire_active_cache",
                "idle_expiry",
                "pending_expiry",
                "paired_cold_loads",
                "audit_native_copies",
                "trace_prefix_shapes",
                "trace_prefix_pages",
                "trace_prefix_prefill",
                "trace_prefill",
                "trace_prompt_chunks",
                "trace_host_registration",
                "stable_expert_layout",
                "stable_qsa_selection",
                "stable_qsa_ties",
                "native_kernels",
                "optimized_canonical",
                "cuda_launch_blocking",
                "streaming",
                "diagnose",
                "divergent_prefixes",
            )
        )
    ):
        raise ValueError(
            "Failure/recovery requires isolated untraced TP4/MTP4 "
            "private native offload, budget2048 and no other fixture"
        )


def install_native_load_failure(connector, rank, output):
    """Fail one completed load on rank 3; only for a disposable engine test."""
    if os.environ.get("FLASH_NATIVE_LOAD_FAILURE") != "1":
        raise ValueError("Native load failure requires the isolated diagnostic gate")
    if type(rank) is not int or rank not in range(4):
        raise ValueError("Native load failure requires a TP4 rank")
    if rank != 3:
        return dict(rank=rank, armed=False)
    backend = connector.worker
    if output.exists() or getattr(backend, "_native_load_failure_armed", False):
        raise ValueError("Native load failure may only be armed once with fresh output")
    native = backend.get_finished
    triggered = False

    def get_finished():
        nonlocal triggered
        if triggered:
            raise RuntimeError("Diagnostic native cold-load failure remains fatal")
        results = native()
        for result in results:
            request_id = connector._load_jobs.get(result.job_id)
            if request_id is None:
                continue
            if not (
                request_id == "failure-cold" or request_id.startswith("failure-cold-")
            ):
                raise ValueError(
                    "Unexpected load request after failure diagnostic armed"
                )
            if (
                result.success is not True
                or type(result.transfer_size) is not int
                or result.transfer_size <= 0
                or result.job_id in connector._connector_worker_meta.completed_jobs
            ):
                raise ValueError(
                    "Failure injection lacks a completed unacknowledged load"
                )
            record = dict(
                rank=rank,
                job_id=result.job_id,
                request_id=request_id,
                backend_load_completed=True,
                transfer_bytes=result.transfer_size,
                completion_already_acknowledged=False,
                scope="Diagnostic exception after a real completed CPU-to-GPU load; "
                "not a physical DMA fault or partially copied destination.",
            )
            with output.open("x") as stream:
                json.dump(record, stream, allow_nan=False)
                stream.write("\n")
            triggered = True
            raise RuntimeError("Injected native cold-load completion failure")
        return results

    backend.get_finished = get_finished
    backend._native_load_failure_armed = True
    return dict(rank=rank, armed=True)


async def native_load_failure_screen(
    engine, generate, mode, output, dead_error, evidence, save
):
    """Full-engine failure observation or fresh-process recovery; never both."""
    if mode not in ("inject", "recovery"):
        raise ValueError("Select injection or fresh-process recovery")
    evidence.update(
        mode=mode, passed=False, failed_request_emitted_tokens=0, retry_emitted_tokens=0
    )
    save()
    salt = "native-load-failure-recovery"
    prime = await generate("failure-prime", cache_salt=salt)
    hot = await generate("failure-hot", cache_salt=salt)
    if (
        prime["cached_tokens"] != 0
        or hot["cached_tokens"] != 5664
        or prime["prompt_sha256"] != hot["prompt_sha256"]
        or any(
            row["transfer_delta"].get("vllm:kv_offload_load_bytes", 0) != 0
            for row in (prime, hot)
        )
    ):
        raise ValueError("Failure fixture lacks fresh-process and hot-prefix controls")
    await engine.check_health()
    evidence.update(
        prompt_sha256=prime["prompt_sha256"],
        cache_salt=salt,
        fresh_cached_tokens=0,
        hot_cached_tokens=5664,
        initial_health_passed=True,
    )
    save()
    if mode == "recovery":
        evidence.update(
            passed=True,
            scope="Fresh engine recomputation and hot reuse; "
            "requires an external join to the failed engine's identity "
            "and prompt. Not recovery of volatile RAM contents.",
        )
        save()
        return
    if output.exists():
        raise ValueError("Failure artifact must not exist before arming")
    for index in range(1, 33):
        await generate(
            f"failure-pressure-{index}",
            index=index,
            tokens=8,
            cache_salt=f"failure-pressure-{index}",
        )
    await engine.pause_generation(mode="wait", clear_cache=False)
    armed = await engine.collective_rpc("arm_native_load_failure")
    if sorted(row["rank"] for row in armed) != list(range(4)) or any(
        row.get("armed") is not (row["rank"] == 3) for row in armed
    ):
        raise ValueError("Failure fixture lacks exactly one armed TP4 rank")
    evidence["armed_workers"] = armed
    save()
    await engine.resume_generation()

    async def observe(name, result):
        key = (
            "failed_request_emitted_tokens"
            if name == "failure-cold"
            else "retry_emitted_tokens"
        )
        evidence[key] = max(evidence[key], len(result.outputs[0].token_ids))
        save()

    try:
        await generate("failure-cold", cache_salt=salt, observe=observe)
    except dead_error as error:
        evidence["cold_error_type"] = type(error).__name__
    else:
        raise AssertionError("Injected load did not terminate the engine request")
    with output.open() as stream:
        record = json.load(stream)
    if not (
        record.get("rank") == 3
        and type(record.get("job_id")) is int
        and isinstance(record.get("request_id"), str)
        and (
            record["request_id"] == "failure-cold"
            or record["request_id"].startswith("failure-cold-")
        )
        and record.get("backend_load_completed") is True
        and record.get("completion_already_acknowledged") is False
        and type(record.get("transfer_bytes")) is int
        and record["transfer_bytes"] > 0
    ):
        raise ValueError(
            "Failure evidence lacks a completed unacknowledged native load"
        )
    evidence["worker_failure"] = record
    save()
    try:
        await engine.check_health()
    except dead_error:
        evidence["dead_health_rejected"] = True
    else:
        raise AssertionError("Failed engine still reports healthy")
    try:
        await generate("failure-retry", cache_salt=salt, tokens=1, observe=observe)
    except dead_error:
        evidence["dead_retry_rejected"] = True
    else:
        raise AssertionError("Failed engine accepted a new generation")
    evidence["passed"] = not (
        evidence["failed_request_emitted_tokens"] or evidence["retry_emitted_tokens"]
    )
    evidence["scope"] = (
        "Injected failure after completed native load; engine death, "
        "zero output and retry rejection. Not physical DMA failure or restart proof."
    )
    save()
    if not evidence["passed"]:
        raise AssertionError("Failed or retried request emitted tokens")


def copy_audit_verdict(ranks):
    """Require both real copy directions on every TP4 rank; no vacuous pass."""
    checks = dict(
        all_four_ranks=[row.get("rank") for row in ranks] == [0, 1, 2, 3],
        observed_both_directions=bool(ranks)
        and all(
            {row.get("direction") for row in rank.get("records", [])}
            == {"store", "load"}
            for rank in ranks
        ),
        exact_observed_copies=bool(ranks)
        and all(
            rank.get("records")
            and all(
                row.get("exact") is True
                and row.get("source_unchanged") is True
                and row.get("bytes", 0) > 0
                and row.get("descriptors", 0) > 0
                and (
                    row.get("direction") != "load"
                    or row.get("stored_version_verified") is True
                )
                for row in rank["records"]
            )
            for rank in ranks
        ),
    )
    return dict(
        checks=checks,
        passed=all(checks.values()),
        scope="Observed native CPU/GPU copy bytes and CPU store versions only",
    )


def locate_fragment(tensors, pointer, size):
    """Resolve only bytes inside a known tensor row; never dereference a pointer."""
    if type(pointer) is not int or type(size) is not int or size <= 0:
        raise ValueError("Invalid native copy descriptor")
    matches = []
    for index, tensor in enumerate(tensors):
        if (
            tensor.ndim != 2
            or min(tensor.shape) <= 0
            or str(tensor.dtype) != "torch.int8"
            or tensor.stride(1) != 1
            or tensor.stride(0) < tensor.shape[1]
            or tensor.element_size() != 1
        ):
            raise ValueError(
                "Audit requires native byte rows without overlapping strides"
            )
        offset = pointer - tensor.data_ptr()
        if offset < 0:
            continue
        row, start = divmod(offset, tensor.stride(0))
        if row < tensor.shape[0] and start + size <= tensor.shape[1]:
            matches.append(
                ((index, row, start, size), tensor[row, start : start + size])
            )
    if len(matches) != 1:
        raise ValueError("Descriptor is outside or ambiguous in its owning tensor pool")
    return matches[0]


class NativeCopyAudit:
    """Compare actual copy bytes, including the last observed CPU store version.

    This does not prove scheduler hash ownership, untouched non-destinations,
    resident RAM KV integrity, race freedom or unobserved lifecycle paths.
    """

    def __init__(self, *, digest=hash_page_chunks, synchronize=None):
        if synchronize is None:
            import torch

            synchronize = lambda: torch.cuda.current_stream().synchronize()
        self.digest = digest
        self.synchronize = synchronize
        self.versions = {}
        self.row_versions = {}
        self.records = []
        self.restores = []

    def copy(self, handler, native, src, dst, sizes, **kwargs):
        sources, destinations, lengths = src.tolist(), dst.tolist(), sizes.tolist()
        if len(self.records) >= MAX_COPY_RECORDS:
            raise ValueError(
                "Native copy audit record limit: "
                f"{len(self.records)}/{MAX_COPY_RECORDS}"
            )
        if (
            not 0 < len(lengths) <= 65536
            or len(sources) != len(lengths)
            or len(destinations) != len(lengths)
            or any(type(size) is not int or size <= 0 for size in lengths)
            or sum(lengths) > 2 * 1024**3
        ):
            raise ValueError(
                "Native transfer descriptor/payload bounds exceeded: "
                f"src={len(sources)}, dst={len(destinations)}, "
                f"sizes={len(lengths)}, bytes={sum(lengths)}"
            )
        fragments = [
            (
                locate_fragment(handler.src_tensors, a, size),
                locate_fragment(handler.dst_tensors, b, size),
            )
            for a, b, size in zip(sources, destinations, lengths, strict=True)
        ]
        # Wait on the native transfer stream's compute/previous-copy dependencies.
        self.synchronize()
        before = [self.digest(source[1]) for source, _ in fragments]
        cpu_keys = [
            destination[0] if handler.gpu_to_cpu else source[0]
            for source, destination in fragments
        ]
        if len(set(cpu_keys)) != len(cpu_keys):
            raise ValueError("Audit does not accept duplicate CPU fragments in a copy")
        ordered = sorted(cpu_keys)
        if any(
            left[:2] == right[:2] and left[2] + left[3] > right[2]
            for left, right in zip(ordered, ordered[1:])
        ):
            raise ValueError(
                "Audit does not accept overlapping CPU fragments in a copy"
            )
        if not handler.gpu_to_cpu:
            for key, value in zip(cpu_keys, before, strict=True):
                if self.versions.get(key) != value:
                    raise ValueError(
                        "CPU load source lacks its exact observed store version"
                    )
        native(src, dst, sizes, **kwargs)
        self.synchronize()
        after = [self.digest(destination[1]) for _, destination in fragments]
        source_unchanged = [self.digest(source[1]) for source, _ in fragments] == before
        exact = before == after
        record = dict(
            direction="store" if handler.gpu_to_cpu else "load",
            copy_index=len(self.records),
            descriptors=len(lengths),
            bytes=sum(lengths),
            exact=exact,
            source_unchanged=source_unchanged,
            stored_version_verified=not handler.gpu_to_cpu,
            payload_sha256=hashlib.sha256(
                json.dumps(list(zip(lengths, before))).encode()
            ).hexdigest(),
        )
        self.records.append(record)
        if not exact or not source_unchanged:
            raise ValueError(
                "Native offload copy changed source or restored different bytes"
            )
        if handler.gpu_to_cpu:
            for key, value in zip(cpu_keys, before, strict=True):
                index, row, start, size = key
                row_keys = self.row_versions.setdefault((index, row), set())
                overlapping = [
                    old
                    for old in row_keys
                    if max(old[2], start) < min(old[2] + old[3], start + size)
                ]
                for old in overlapping:
                    del self.versions[old]
                    row_keys.remove(old)
                self.versions[key] = value
                row_keys.add(key)
            if len(self.versions) > 65536:
                raise ValueError("Observed CPU versions exceed audit bound")

    def install(self, worker):
        handlers = (worker._store_handler, worker._load_handler)
        if (
            self.restores
            or self.records
            or self.versions
            or getattr(worker, "_mmap_region", None) is not None
        ):
            raise ValueError("Audit requires uninstrumented worker-private CPU buffers")
        if any(
            handler._transfers or handler._canonical_copy_plans is not None
            for handler in handlers
        ):
            raise ValueError("Audit requires drained direct-layout handlers")
        if (
            not handlers[0].gpu_to_cpu
            or handlers[1].gpu_to_cpu
            or any(
                len(first) != len(second)
                or any(a is not b for a, b in zip(first, second))
                for first, second in (
                    (handlers[0].dst_tensors, handlers[1].src_tensors),
                    (handlers[0].src_tensors, handlers[1].dst_tensors),
                )
            )
        ):
            raise ValueError("Store and load must own the same CPU/GPU tensor objects")
        for handler in handlers:
            native = handler._swap_blocks_batch

            def audited(src, dst, sizes, *, _handler=handler, _native=native, **kwargs):
                return self.copy(_handler, _native, src, dst, sizes, **kwargs)

            self.restores.append((handler, native))
            handler._swap_blocks_batch = audited
        return dict(installed=True, timing_perturbed=True)

    def remove(self):
        for handler, native in self.restores:
            handler._swap_blocks_batch = native
        self.restores.clear()
        return dict(
            records=self.records,
            retained_versions=len(self.versions),
            timing_perturbed=True,
            scope=self.__doc__,
        )
