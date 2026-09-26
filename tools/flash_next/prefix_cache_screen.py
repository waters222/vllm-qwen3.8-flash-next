"""Ordinary-request hot/cold screen; never invokes the session-ID prototype."""

import asyncio
import copy
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import asdict

from prefix_cancellation import ActiveCancellation
from session_prefill_trace import index_prefix_page_requests


async def admitted_batch(engine, calls, timeout=30):
    """Use native pause/admission so the diagnostic batch starts together."""
    if not calls:
        raise ValueError("Admission control requires a nonempty batch")
    native = engine.add_request
    had_override = "add_request" in vars(engine)
    admitted = asyncio.Event()
    count = 0

    async def observe(*args, **kwargs):
        nonlocal count
        result = await native(*args, **kwargs)
        count += 1
        if count == len(calls):
            admitted.set()
        return result

    tasks, group, waiter = [], None, None
    await engine.pause_generation(mode="keep", clear_cache=False)
    try:
        engine.add_request = observe
        try:
            tasks = [asyncio.create_task(call()) for call in calls]
            group = asyncio.gather(*tasks)
            waiter = asyncio.create_task(admitted.wait())
            done, _ = await asyncio.wait(
                (group, waiter), timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
            if waiter not in done:
                if group in done:
                    await group
                raise TimeoutError("Diagnostic batch did not finish native admission")
        finally:
            if had_override:
                engine.add_request = native
            else:
                del engine.add_request
            await engine.resume_generation()
        return await group
    finally:
        if waiter is not None:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
        for task in tasks:
            if not task.done():
                task.cancel()
        if group is not None:
            await asyncio.gather(group, return_exceptions=True)
        await asyncio.gather(*tasks, return_exceptions=True)


def validate_balanced_observation(args):
    """Require either shape evidence or an explicitly untraced follow-up."""
    balanced = getattr(args, "balanced_prefix_prefill", False)
    untraced = getattr(args, "untraced_balanced_prefix", False)
    shapes = getattr(args, "trace_prefix_shapes", False)
    copies = getattr(args, "audit_native_copies", False)
    if untraced and (
        not balanced
        or copies
        or any(
            getattr(args, name, False)
            for name in (
                "trace_prefix_shapes",
                "trace_prefix_pages",
                "trace_prefix_prefill",
                "trace_prompt_chunks",
                "trace_prefill",
                "trace_host_registration",
            )
        )
    ):
        raise ValueError(
            "Untraced balanced diagnostic cannot enable tensor/registration traces"
        )
    if balanced and (
        getattr(args, "prefix_concurrency", 1) != 2
        or not (shapes or untraced or copies)
    ):
        raise ValueError(
            "Balanced prefix diagnostic requires C2 and shape tracing "
            "or native-copy auditing or explicit untraced mode"
        )


def validate_continuation_observation(args):
    context = getattr(args, "continuation_context", 8192)
    continuation_seed_tokens(context)
    concurrency = getattr(args, "continuation_concurrency", 1)
    pages = getattr(args, "trace_prefix_pages", False) and getattr(
        args, "trace_prefix_shapes", False
    )
    if (
        getattr(args, "continuations", False)
        and pages
        and (
            context != 32768
            or concurrency not in (2, 4)
            or not getattr(args, "tp4", False)
            or not getattr(args, "balanced_prefix_prefill", False)
        )
    ):
        raise ValueError("Continuation page audit requires balanced TP4/C2 or C4 at 32768")
    if concurrency not in (1, 2, 4) or (
        concurrency > 1
        and (
            not getattr(args, "continuations", False)
            or not getattr(args, "tp4", False)
            or not getattr(args, "balanced_prefix_prefill", False)
            or getattr(args, "prefix_concurrency", 1) != 2
        )
    ):
        raise ValueError(
            "Concurrent continuations require balanced TP4 with C2 prefix controls"
        )
    if concurrency == 4 and context != 32768:
        raise ValueError("C4 continuations require the 32768 context fixture")
    c4_cancel = (getattr(args, "active_cancel", False)
                 and getattr(args, "cancellation_concurrency", 2) == 4)
    if context != 8192 and not (getattr(args, "continuations", False) or c4_cancel
                                or getattr(args, "sustained_decode", False)):
        raise ValueError("Extended context requires continuations or C4 cancellation")
    if getattr(args, "continuations", False) and (
        not getattr(args, "native_prefix", False)
        or not (
            getattr(args, "untraced_balanced_prefix", False)
            or getattr(args, "audit_native_copies", False)
            or pages
        )
        or getattr(args, "idle_expiry", False)
        or getattr(args, "pending_expiry", False)
        or not 1 <= getattr(args, "tokens", 128) <= 128
    ):
        raise ValueError(
            "Continuations require untraced, copy- or page-audited native prefixes, "
            "no expiry, and <=128 output tokens"
        )


def validate_cancellation_observation(args):
    concurrency = getattr(args, "cancellation_concurrency", 2)
    if concurrency not in (2, 4) or (concurrency == 4 and (
        not all(getattr(args, key, False) for key in (
            "active_cancel", "tp4", "balanced_prefix_prefill"
        ))
        or not (getattr(args, "serving_decode", False)
                or getattr(args, "audit_active_cancel", False))
        or getattr(args, "continuation_context", 8192) != 32768
        or getattr(args, "prefix_prefill_budget", None) != 2048
        or getattr(args, "no_mtp", False)
    )):
        raise ValueError(
            "C4 cancellation requires TP4/MTP4 at32768 with graphs or audit"
        )
    if getattr(args, "expire_active_cache", False) and not getattr(
        args, "audit_active_cancel", False
    ):
        raise ValueError("Active expiry requires the paused active-cancel audit")
    if getattr(args, "audit_active_cancel", False) and (
        not getattr(args, "active_cancel", False)
        or not getattr(args, "paired_cold_loads", False)
    ):
        raise ValueError(
            "Cancellation audit requires active cancel and paired scheduler"
        )
    if getattr(args, "active_cancel", False) and (
        not getattr(args, "native_prefix", False)
        or not getattr(args, "untraced_balanced_prefix", False)
        or not getattr(args, "divergent_prefixes", False)
        or getattr(args, "prefix_concurrency", 1) != 2
        or getattr(args, "tokens", 128) != 128
        or any(
            getattr(args, flag, False)
            for flag in ("idle_expiry", "pending_expiry", "continuations")
        )
    ):
        raise ValueError(
            "Active cancellation requires untraced divergent C2 native prefixes, "
            "128 output tokens, and no other lifecycle fixture"
        )


def validate_serving_decode(args):
    if not getattr(args, "serving_decode", False):
        return
    byte_audit = getattr(args, "trace_prefix_shapes", False) and getattr(
        args, "trace_prefix_pages", False
    )
    if (
        not getattr(args, "native_prefix", False)
        or not getattr(args, "tp4", False)
        or not (
            getattr(args, "untraced_balanced_prefix", False)
            or byte_audit
            or getattr(args, "audit_native_copies", False)
        )
        or any(
            getattr(args, name, False)
            for name in (
                "audit_active_cancel",
                "stable_expert_layout",
                "stable_qsa_selection",
                "stable_qsa_ties",
                "native_kernels",
                "optimized_canonical",
            )
        )
    ):
        raise ValueError(
            "Graph decode requires TP4 native-prefix mode with either untraced "
            "observation, a complete prefix-page audit or native-copy auditing, "
            "without ordering controls, kernel fallbacks or paused cache audits"
        )


def validate_copy_audit(args):
    if getattr(args, "audit_native_copies", False) and (
        not getattr(args, "native_prefix", False)
        or not getattr(args, "tp4", False)
        or not getattr(args, "private_cold_buffers", False)
        or not getattr(args, "balanced_prefix_prefill", False)
        or getattr(args, "prefix_concurrency", 1) != 2
        or getattr(args, "untraced_balanced_prefix", False)
        or any(
            getattr(args, name, False)
            for name in (
                "idle_expiry",
                "pending_expiry",
                "active_cancel",
                "trace_prefix_pages",
                "trace_prefix_shapes",
                "trace_prefix_prefill",
                "trace_prompt_chunks",
                "trace_prefill",
                "trace_host_registration",
            )
        )
    ):
        raise ValueError(
            "Native copy audit requires private native-prefix buffers "
            "without another state/expiry/cancellation audit"
        )


def compare_active_cache_audit(audit, survivors=None):
    """Compare every allocated survivor page while native execution is paused."""
    if survivors is not None:
        expected = tuple(survivors)
        recorded = audit.get("survivors", {})
        complete = (len(expected) == len(set(expected)) == 3
                    and set(recorded) == set(expected))
        results = {}
        for name in expected:
            item = recorded.get(name, {})
            assessment = compare_active_cache_audit(item)
            identity = all(
                len(item.get(side, [])) == 4
                and [snapshot.get("rank") for snapshot in item[side]] == list(range(4))
                and all(
                    len(snapshot.get("requests", [])) == 1
                    and re.fullmatch(
                        re.escape(name) + r"(?:-[0-9a-f]{8})?",
                        snapshot["requests"][0].get("request_id", ""),
                    ) is not None
                    for snapshot in item[side]
                )
                for side in ("before", "after")
            )
            results[name] = dict(assessment, identity_verified=identity)
            complete &= identity and assessment["passed"]
        return dict(survivors=results, passed=bool(complete),
                    scope="All three TP4 survivors across paused native peer abort")
    before, after = audit.get("before", []), audit.get("after", [])
    ranks = []
    for left, right in zip(before, after):
        requests = left.get("requests", [])
        groups = requests[0].get("groups", []) if len(requests) == 1 else []
        names = [name for group in groups for name in group["layers"]]
        pages = [
            row
            for group in groups
            for parts in group["layers"].values()
            for rows in parts
            for row in rows
        ]
        checks = dict(
            allocated_pages=bool(pages)
            and left.get("bytes_hashed", 0) > 0
            and all(row.get("sha256") and row.get("bytes", 0) > 0 for row in pages),
            complete_groups=len(groups) == 41
            and {group["group_id"] for group in groups} == set(range(41)),
            complete_layer_parts=bool(groups)
            and all(
                len(group["layers"])
                == {37: 13, 38: 13, 39: 12, 40: 1}.get(group["group_id"], 1)
                and all(
                    len(parts) == (2 if group["group_id"] < 36 else 1)
                    and all(rows for rows in parts)
                    for parts in group["layers"].values()
                )
                for group in groups
            ),
            target_and_draft=any("mtp" in name for name in names)
            and any(".linear_attn" in name for name in names)
            and any("compressed_key_cache" in name for name in names)
            and sum(group["host_resident"] for group in groups) == 2,
            same_survivor_bytes_and_mapping=left == right,
        )
        ranks.append(
            dict(
                checks=checks,
                pages=len(pages),
                bytes_hashed=left.get("bytes_hashed", 0),
                passed=all(checks.values()),
            )
        )
    return dict(
        scope="TP4 survivor allocated pages across paused native peer abort",
        ranks=ranks,
        passed=len(before) == len(after) == 4 and all(rank["passed"] for rank in ranks),
    )


def cancellation_verdict(control):
    """A lifecycle gate, not proof of byte equality or numerical equivalence."""
    concurrency = control.get("concurrency", 2)
    rows = control["requests"]
    if concurrency not in (2, 4) or len(rows) != concurrency:
        return dict(checks=dict(expected_participants=False), passed=False)
    victim, *survivors = rows
    expected = ["cancel-victim", *[f"cancel-survivor-{i}" for i in range(3)]]
    checks = dict(
        native_abort=control["abort_acknowledged"]
        and bool(control["at_abort"])
        and all(
            row["tokens"] >= 8 and not row["finished"]
            for row in control["at_abort"].values()
        ),
        partial_abort=victim.get("finish_reason") == "abort"
        and 0 < len(victim["tokens"]) < victim["requested_output_tokens"]
        and victim["output_completeness"]["passed"],
        survivor_complete=all(
            row["output_completeness"]["passed"] for row in survivors
        ),
        concurrent=control["max_running"] >= concurrency,
        real_cold_reuse=control["load_bytes"] > 0
        and all(row["cached_tokens"] > 0 for row in rows),
        subsequent_reuse=all(
            row["cached_tokens"] > 0 and row["output_completeness"]["passed"]
            for row in control["reuse"]
        )
        and len(control["reuse"]) == concurrency,
        frontend_drained=control["frontend_drained"],
    )
    if concurrency == 4:
        checks["long_context"] = control.get("context_tokens") == 32768 and all(
            row.get("prompt_tokens", 0) == continuation_seed_tokens(32768)
            for row in rows + control["reuse"]
        )
        checks["same_replay_inputs"] = (
            len({row.get("prompt_sha256") for row in rows}) == 4
            and len(control["reuse"]) == 4
            and all(
                row.get("prompt_sha256") and row.get("cache_salt")
                and (row["prompt_sha256"], row["cache_salt"])
                == (reuse.get("prompt_sha256"), reuse.get("cache_salt"))
                for row, reuse in zip(rows, control["reuse"])
            )
        )
        checks["execution_mode"] = (
            control.get("serving_decode") is True and any(
                graph.get("runtime_mode") == "FULL"
                and graph.get("num_unpadded_tokens")
                == graph.get("num_padded_tokens") == 20
                and graph.get("steps", 0) > 0
                for row in rows for graph in row.get("native_cudagraph_window", [])
            )
        ) or (control.get("serving_decode") is False
              and control.get("audit_required") is True)
        checks["expected_participants"] = (
            [row.get("name") for row in rows] == expected
            and set(control.get("at_abort") or {}) == set(expected)
            and [row.get("name") for row in control["reuse"]]
            == [f"cancel-reuse-{i}" for i in range(4)]
        )
        checks["complete_raw_outputs"] = all(
            row.get("requested_output_tokens") == len(row["tokens"]) == 128
            and row.get("finish_reason") == "length"
            for row in survivors + control["reuse"]
        )
    if control.get("audit_required"):
        checks["survivor_cache_bytes"] = compare_active_cache_audit(
            control.get("cache_audit") or {},
            expected[1:] if concurrency == 4 else None,
        )["passed"]
        checks["survivor_ownership"] = control.get("ownership_audit", {}).get(
            "passed", False
        )
        if concurrency == 4:
            ownership = control.get("ownership_audit", {})
            peers = ownership.get("survivors", {})
            checks["survivor_ownership"] &= (
                ownership.get("victim_removed_from_running") is True
                and set(peers) == set(expected[1:])
                and all(
                    row.get("passed") is True and row.get("survivor_running") is True
                    and row.get("blocks", 0) > 0
                    and all(row.get("checks", {}).get(key) is True for key in (
                        "complete_owned_table", "same_pages_and_hashes",
                        "live_references",
                    ))
                    and re.fullmatch(re.escape(name) + r"(?:-[0-9a-f]{8})?",
                                     row.get("survivor", "")) is not None
                    for name, row in peers.items()
                )
            )
    if control.get("active_expiry_required"):
        expiry = control.get("ownership_audit", {}).get("active_expiry", {})
        checks["active_expiry"] = (
            control.get("audit_required") is True
            and expiry.get("active_requests") == concurrency
            and expiry.get("protected_host_blocks", 0) > 0
            and expiry.get("expired_host_blocks", 0) > 0
            and expiry.get("expired_cpu_chunks", 0) > 0
            and expiry.get("configured_ttl_seconds") == 3600
            and expiry.get("advanced_seconds") == 3601
            and expiry.get("clocks_restored") is True
            and expiry.get("ownership_preserved") is True
        )
    return dict(checks=checks, passed=all(checks.values()))


def continuation_seed_tokens(context):
    """Keep the original prefill remainder and room for three turns plus MTP."""
    if context not in (8192, 32768):
        raise ValueError("Continuation context must be 8192 or 32768")
    return ((context - 3 * 944 - 128 - 4 - 492) // 944) * 944 + 492


def continuation_page_boundaries(context, concurrency=2):
    """Observe reused checkpoints and earlier duplicate physical producers."""
    if context != 32768 or concurrency not in (2, 4):
        raise ValueError("Continuation page capture requires C2/C4 at 32768")
    seed = continuation_seed_tokens(context)
    reused = [(seed // 944 + turn - 2) * 944 for turn in range(1, 4)]
    # Lookup can select an earlier cached version instead of the last request's.
    boundaries = (reused[:2], reused, reused[1:], [reused[2]])
    return {
        f"turn{concurrency}-{mode}-{turn}": list(values)
        for mode in ("hot", "cold")
        for turn, values in enumerate(boundaries)
    }


def validate_gpu_cache_budget(args):
    budget = getattr(args, "prefix_gpu_cache_mib", None)
    long_context = getattr(args, "continuation_context", 8192) == 32768 or (
        getattr(args, "prefill_staging_ab", False)
        and getattr(args, "prefill_staging_concurrency", 2) == 4
    )
    if budget is not None and (
        budget not in (512, 768, 896, 1024) or not long_context
        or not getattr(args, "native_prefix", False)
        or not getattr(args, "tp4", False)
    ):
        raise ValueError("GPU cache budget requires native TP4/32k and512/768/896/1024MiB")


def prefix_engine_config(
    source,
    no_mtp=False,
    private_cold_buffers=False,
    tp4=False,
    prefill_budget=None,
    balanced_prefill=False,
    continuation_context=8192,
    serving_decode=False,
    continuation_concurrency=1,
    gpu_cache_mib=None,
):
    continuation_seed_tokens(continuation_context)
    if gpu_cache_mib is not None and (
        gpu_cache_mib not in (512, 768, 896, 1024) or not tp4
        or continuation_context != 32768
    ):
        raise ValueError("GPU cache budget requires TP4/32k and512/768/896/1024MiB")
    if continuation_concurrency not in (1, 2, 4) or (
        continuation_concurrency == 4
        and (not tp4 or continuation_context != 32768 or not balanced_prefill)
    ):
        raise ValueError("C4 engine slots require balanced TP4/32768 continuations")
    config = copy.deepcopy(source)
    if config.get("worker_extension_cls"):
        raise ValueError("Worker extension already configured")
    config.update(
        worker_extension_cls="session_prefill_trace.PrefillTraceWorkerExtension",
        max_model_len=continuation_context,
        per_request_spec_decode_metrics="summary",
        max_num_seqs=max(2, continuation_concurrency),
        enable_prefix_caching=True,
        enforce_eager=True,
        compilation_config={"mode": 0, "cudagraph_mode": "NONE"},
        mamba_cache_mode="align",
        additional_config={
            "flash_next_direct_host_kv": {
                "num_blocks": 512,
                "max_bytes": 128 * 1024**3,
            }
        },
        kv_transfer_config={
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "cpu_bytes_to_use": 8 * 1024**3,
                "idle_ttl_seconds": 3600,
                "eviction_policy": "lru",
            },
        },
    )
    if continuation_context == 32768:
        config["kv_cache_memory_bytes"] = (gpu_cache_mib or 1024) * 1024**2
        # Keep the cold prefix beyond the 32-request GPU eviction sequence.
        config["additional_config"]["flash_next_direct_host_kv"]["num_blocks"] = 1024
    if tp4:
        config.update(tensor_parallel_size=4, pipeline_parallel_size=1)
        config["speculative_config"] = {"method": "mtp", "num_speculative_tokens": 4}
    if no_mtp:
        config["speculative_config"] = None
    if private_cold_buffers:
        config["kv_transfer_config"]["kv_connector_extra_config"][
            "use_shared_memory"
        ] = False
    if prefill_budget is not None:
        if prefill_budget not in (1024, 2048):
            raise ValueError("Prefix diagnostic prefill budget must be1024 or2048")
        config["max_num_batched_tokens"] = prefill_budget
        config["max_num_scheduled_tokens"] = prefill_budget
    if balanced_prefill:
        if not tp4 or prefill_budget != 2048:
            raise ValueError("Balanced prefix diagnostic requires TP4/budget2048")
        config.update(
            enable_chunked_prefill=True,
            # Fit four initial chunks before any branch can populate the shared
            # prefix. Native sub-block prefill still checkpoints at 944 tokens.
            long_prefill_token_threshold=472 if continuation_concurrency == 4 else 944,
        )
    if serving_decode:
        if not tp4:
            raise ValueError("Serving decode diagnostic requires TP4")
        gpu_budget = config.get("kv_cache_memory_bytes")
        if type(gpu_budget) is not int or gpu_budget <= 0:
            raise ValueError("Graph diagnostic requires explicit GPU KV byte budget")
        config["additional_config"]["flash_next_direct_host_kv"]["allow_cudagraph"] = (
            True
        )
        config.update(
            enforce_eager=False,
            cudagraph_metrics=True,
            compilation_config={
                "mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1, 2]
                if no_mtp
                else [1, 2, 3, 4, 5, 6, 8, 10],
            },
        )
        if continuation_concurrency == 4:
            config["compilation_config"]["cudagraph_capture_sizes"] = (
                [1, 2, 3, 4] if no_mtp else [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
            )
    return config


def prefix_verdict(report):
    return bool(
        report.get("baseline_repeatable")
        and report.get("hot_exact")
        and report.get("cold_exact")
        and report.get("hot_cached_tokens", 0) > 0
        and report.get("cold_cached_tokens", 0) > 0
        and report.get("cold_load_bytes", 0) > 0
        and report.get("store_bytes", 0) > 0
        and ("concurrent" not in report or report["concurrent"].get("passed", False))
        and (
            not report.get("performance_repeats")
            or report.get("performance_control", {}).get("passed", False)
        )
        and (
            not report.get("audit_native_copies")
            or report.get("native_copy_audit_verdict", {}).get("passed", False)
        )
        and (
            not report.get("paired_cold_loads")
            or report.get("load_barrier_exercised", False)
        )
        and (
            not report.get("continuation_load_gate")
            or report.get("continuation_load_gate_exercised", False)
        )
        and (
            not report.get("idle_expiry")
            or report.get("expiry_control", {}).get("passed", False)
        )
        and (
            not report.get("pending_expiry")
            or report.get("pending_expiry_control", {}).get("passed", False)
        )
        and (
            not report.get("continuations")
            or report.get("continuation_control", {}).get("passed", False)
        )
        and (
            not report.get("active_cancel")
            or report.get("cancellation_control", {}).get("passed", False)
        )
    )


def compare_idle_expiry(evidence, expired, reference, refreshed, hot):
    """Require actual native expiry, a fresh miss, and renewed prefix reuse."""
    rows = (expired, reference, refreshed, hot)
    checks = dict(
        native_expiry=evidence.get("expired_host_blocks", 0) > 0
        and evidence.get("expired_cpu_chunks", 0) > 0,
        safe_clock_control=evidence.get("configured_ttl_seconds") == 3600
        and evidence.get("advanced_seconds") == 3601
        and evidence.get("quiescent") is True
        and evidence.get("clocks_restored") is True,
        same_prompt=bool(expired.get("prompt_sha256"))
        and len({row.get("prompt_sha256") for row in rows}) == 1,
        expired_miss=expired.get("cached_tokens") == 0
        and expired.get("skip_cache") is False
        and expired.get("transfer_delta", {}).get("vllm:kv_offload_load_bytes", 0) == 0,
        fresh_reference=reference.get("cached_tokens") == 0
        and reference.get("skip_cache") is True,
        exact_recomputation=bool(expired.get("tokens"))
        and expired["tokens"] == reference.get("tokens")
        and bool(expired.get("logprobs"))
        and expired["logprobs"] == reference.get("logprobs"),
        renewed_reuse=refreshed.get("cached_tokens", 0) > 0
        and refreshed.get("skip_cache") is False
        and refreshed.get("transfer_delta", {}).get("vllm:kv_offload_load_bytes", 0)
        == 0,
        exact_hot_continuation=bool(refreshed.get("tokens"))
        and refreshed["tokens"] == hot.get("tokens")
        and bool(refreshed.get("logprobs"))
        and refreshed["logprobs"] == hot.get("logprobs"),
    )
    return dict(**checks, passed=all(checks.values()))


def compare_pending_expiry(evidence, concurrent):
    """Require actual protected loads and exact subsequent C2 hot/cold output."""
    comparisons = concurrent.get("hot_cold_comparisons", [])
    checks = dict(
        both_requests_acquired=evidence.get("acquired_cold_requests") == 2,
        real_load_ownership=all(
            evidence.get(key, 0) > 0
            for key in (
                "pending_load_jobs",
                "protected_host_blocks",
                "protected_cpu_chunks",
                "owned_destination_blocks",
            )
        ),
        real_idle_expiry=evidence.get("expired_host_blocks", 0) > 0
        and evidence.get("expired_cpu_chunks", 0) > 0,
        ownership_preserved=evidence.get("ownership_preserved") is True,
        clocks_restored=evidence.get("clocks_restored") is True
        and evidence.get("advanced_seconds") == 3601
        and evidence.get("configured_ttl_seconds") == 3600,
        actual_cold_load=concurrent.get("cold", {}).get("load_bytes", 0) > 0,
        concurrent_execution=concurrent.get("cold", {}).get("max_running", 0) == 2,
        exact_hot_cold=len(comparisons) == 2
        and all(
            row.get("tokens", {}).get("equal") is True
            and row.get("logprobs_exact") is True
            for row in comparisons
        ),
    )
    return dict(**checks, passed=all(checks.values()))


def append_continuation(prompt, answer, tail, max_output_tokens, context=8192):
    """Retain the actual generated answer and append a fixed-width user tail."""
    if not prompt or not tail or not 1 <= len(answer) == max_output_tokens <= 128:
        raise ValueError(
            "Continuation requires complete output and nonempty prompt/tail"
        )
    count = 944 - len(answer)
    extended = [*prompt, *answer, *(tail * (count // len(tail) + 1))[:count]]
    continuation_seed_tokens(context)
    if len(extended) + max_output_tokens + 4 > context:
        raise ValueError("Continuation exceeds the diagnostic model context")
    return extended


def compare_continuation_requests(hot, cold, first, second, previous_length):
    rows = (hot, cold, first, second)

    def loads(row):
        return row.get("transfer_delta", {}).get("vllm:kv_offload_load_bytes", 0)

    checks = dict(
        same_prompt=bool(hot.get("prompt_sha256"))
        and len({row.get("prompt_sha256") for row in rows}) == 1
        and len({row.get("prompt_tokens") for row in rows}) == 1,
        same_sampling=hot.get("repetition_penalty") is not None
        and all(
            row.get("repetition_penalty") == hot["repetition_penalty"] for row in rows
        ),
        appended_turn=hot.get("prompt_tokens", 0) > previous_length > 0,
        isolated_namespaces=all(row.get("cache_salt") for row in rows)
        and len({row.get("cache_salt") for row in rows}) == 4,
        matched_reused_prefix=0
        < hot.get("cached_tokens", 0)
        == cold.get("cached_tokens", 0)
        <= previous_length,
        real_hot_and_cold=loads(hot) == 0
        and loads(cold) > 0
        and all(row.get("skip_cache") is False for row in (hot, cold)),
        fresh_references=all(
            row.get("cached_tokens") == 0
            and row.get("skip_cache") is True
            and loads(row) == 0
            for row in (first, second)
        ),
        exact_tokens=bool(first.get("tokens"))
        and all(row.get("tokens") == first["tokens"] for row in rows),
        exact_logprobs=bool(first.get("logprobs"))
        and all(row.get("logprobs") == first["logprobs"] for row in rows),
    )
    return dict(**checks, passed=all(checks.values()))


async def continuation_screen(
    generate, seed, tail, prompt_factory, output_tokens, pressure_rounds, context=8192
):
    """Run isolated warm/cold chains before either reference namespace writes."""
    if len(seed) != continuation_seed_tokens(context) or not 1 <= pressure_rounds <= 32:
        raise ValueError("Continuation fixture requires its bounded seed and pressure")
    penalties = (1.0, 1.05, 1.1, 1.0)
    conversations = [list(seed)]
    hot = [await generate("turn-hot-0", prompt_ids=seed, cache_salt="turn-hot")]
    for turn in range(1, 4):
        conversation = append_continuation(
            conversations[-1], hot[-1]["tokens"], tail, output_tokens, context
        )
        conversations.append(conversation)
        hot.append(
            await generate(
                f"turn-hot-{turn}",
                prompt_ids=conversation,
                cache_salt="turn-hot",
                repetition_penalty=penalties[turn],
            )
        )
    cold = [await generate("turn-cold-0", prompt_ids=seed, cache_salt="turn-cold")]
    for turn in range(1, 4):
        for pressure in range(pressure_rounds):
            await generate(
                f"turn-pressure-{turn}-{pressure}",
                tokens=8,
                prompt_ids=prompt_factory(5000 + turn * 100 + pressure),
                cache_salt="turn-pressure",
            )
        cold.append(
            await generate(
                f"turn-cold-{turn}",
                prompt_ids=conversations[turn],
                cache_salt="turn-cold",
                repetition_penalty=penalties[turn],
            )
        )
    turns = []
    for turn in range(1, 4):
        references = [
            await generate(
                f"turn-reference-{repeat}-{turn}",
                prompt_ids=conversations[turn],
                skip_cache=True,
                cache_salt=f"turn-reference-{repeat}",
                repetition_penalty=penalties[turn],
            )
            for repeat in ("a", "b")
        ]
        checks = compare_continuation_requests(
            hot[turn],
            cold[turn],
            *references,
            len(conversations[turn - 1]) + output_tokens,
        )
        turns.append(
            dict(
                turn=turn,
                prompt_tokens=len(conversations[turn]),
                repetition_penalty=penalties[turn],
                cached_tokens=cold[turn]["cached_tokens"],
                load_bytes=cold[turn]["transfer_delta"].get(
                    "vllm:kv_offload_load_bytes", 0
                ),
                **checks,
            )
        )
    seed_exact = (
        hot[0]["tokens"] == cold[0]["tokens"]
        and hot[0]["logprobs"] == cold[0]["logprobs"]
    )
    seed_fresh = all(
        row.get("cached_tokens") == 0
        and row.get("transfer_delta", {}).get("vllm:kv_offload_load_bytes", 0) == 0
        for row in (hot[0], cold[0])
    )
    growing_reuse = all(
        right["cached_tokens"] > left["cached_tokens"]
        for left, right in zip(turns, turns[1:])
    )
    changed_sampling = all(
        tuple(row.get("repetition_penalty") for row in chain) == penalties
        for chain in (hot, cold)
    )
    return dict(
        scope="Full-model C1 continuations; native prefix matching with cache salts",
        seed_tokens=len(seed),
        context_tokens=context,
        seed_exact=seed_exact,
        seed_fresh=seed_fresh,
        growing_reuse=growing_reuse,
        changed_sampling=changed_sampling,
        turns=turns,
        passed=seed_exact
        and seed_fresh
        and growing_reuse
        and changed_sampling
        and all(row["passed"] for row in turns),
    )


async def concurrent_continuation_screen(
    generate,
    batch,
    seeds,
    tail,
    prompt_factory,
    output_tokens,
    pressure_rounds,
    context=8192,
):
    """Exercise growing branches; report lifecycle and numerics separately."""
    concurrency = len(seeds)
    if (
        concurrency not in (2, 4)
        or (concurrency == 4 and context != 32768)
        or any(len(seed) != continuation_seed_tokens(context) for seed in seeds)
        or len({tuple(seed) for seed in seeds}) != concurrency
        or any(seed[:944] != seeds[0][:944] for seed in seeds)
        or not 1 <= pressure_rounds <= 32
    ):
        raise ValueError(
            "Concurrent fixture requires two or four shared-prefix private branches"
        )
    namespace = f"turn{concurrency}"
    penalties = (1.0, 1.05, 1.1, 1.0)
    conversations = [[list(seed) for seed in seeds]]

    async def run(mode, turn, skip=False):
        return await batch(
            f"{namespace}-{mode}-{turn}",
            conversations[turn],
            cache_salt=f"{namespace}-{mode}",
            skip_cache=skip,
            repetition_penalty=penalties[turn],
        )

    hot = [await run("hot", 0)]
    for turn in range(1, 4):
        conversations.append(
            [
                append_continuation(
                    prompt_ids, row["tokens"], tail, output_tokens, context
                )
                for prompt_ids, row in zip(
                    conversations[-1], hot[-1]["requests"], strict=True
                )
            ]
        )
        hot.append(await run("hot", turn))
    cold = [await run("cold", 0)]
    for turn in range(1, 4):
        for pressure in range(pressure_rounds):
            await generate(
                f"{namespace}-pressure-{turn}-{pressure}",
                tokens=8,
                prompt_ids=prompt_factory(6000 + turn * 100 + pressure),
                cache_salt=f"{namespace}-pressure",
            )
        cold.append(await run("cold", turn))
    references, turns = [], []
    for turn in range(1, 4):
        fresh = [await run(f"reference-{repeat}", turn, True) for repeat in ("a", "b")]
        references.extend(fresh)
        branches = []
        for branch in range(concurrency):
            rows = [
                phase["requests"][branch] for phase in (hot[turn], cold[turn], *fresh)
            ]
            comparison = compare_continuation_requests(
                *rows, len(conversations[turn - 1][branch]) + output_tokens
            )
            lifecycle = {
                key: value
                for key, value in comparison.items()
                if key not in ("exact_tokens", "exact_logprobs", "passed")
            }
            lifecycle["complete_finite_outputs"] = all(
                complete_request_output(row, output_tokens)["passed"] for row in rows
            )
            branches.append(
                dict(
                    branch=branch,
                    checks=lifecycle,
                    lifecycle_passed=all(lifecycle.values()),
                    numerical_consistency={
                        key: comparison[key]
                        for key in ("exact_tokens", "exact_logprobs")
                    },
                    prompt_tokens=len(conversations[turn][branch]),
                    cached_tokens=cold[turn]["requests"][branch]["cached_tokens"],
                )
            )
        turns.append(dict(turn=turn, branches=branches))
    phases = hot + cold + references
    checks = dict(
        complete_finite_outputs=all(
            complete_request_output(row, output_tokens)["passed"]
            for phase in phases
            for row in phase["requests"]
        ),
        concurrent_execution=all(
            len(phase["requests"]) == phase["max_running"] == concurrency
            for phase in phases
        ),
        fresh_seeds=all(
            phase["load_bytes"] == 0
            and all(row["cached_tokens"] == 0 for row in phase["requests"])
            for phase in (hot[0], cold[0])
        ),
        phase_transfer_evidence=all(
            phase["load_bytes"] == 0 for phase in hot + references
        )
        and all(phase["load_bytes"] > 0 for phase in cold[1:]),
        shared_namespace_private_branches=all(
            len({row["cache_salt"] for row in phase["requests"]}) == 1
            and len({row["prompt_sha256"] for row in phase["requests"]}) == concurrency
            for phase in phases
        ),
        growing_reuse=all(
            right["branches"][branch]["cached_tokens"]
            > left["branches"][branch]["cached_tokens"]
            for branch in range(concurrency)
            for left, right in zip(turns, turns[1:])
        ),
        changed_sampling=all(
            tuple(phase["requests"][branch]["repetition_penalty"] for phase in chain)
            == penalties
            for chain in (hot, cold)
            for branch in range(concurrency)
        ),
        turn_lifecycles=all(
            branch["lifecycle_passed"] for turn in turns for branch in turn["branches"]
        ),
    )
    numerical = all(
        all(branch["numerical_consistency"].values())
        for turn in turns
        for branch in turn["branches"]
    ) and all(
        hot[0]["requests"][branch][field] == cold[0]["requests"][branch][field]
        for branch in range(concurrency)
        for field in ("tokens", "logprobs")
    )
    return dict(
        scope=f"C{concurrency} growing-prefix lifecycle; "
        "no byte-integrity or quality verdict",
        context_tokens=context,
        seed_tokens=len(seeds[0]),
        concurrency=concurrency,
        checks=checks,
        lifecycle_passed=all(checks.values()),
        numerical_consistency_passed=numerical,
        passed=all(checks.values()) and numerical,
        hot=hot,
        cold=cold,
        references=references,
        turns=turns,
        transfer_scope="Phase totals; overlapping per-request deltas are not additive",
    )


def complete_request_output(row, expected_tokens):
    """Reject equal-but-truncated output or incomplete probability evidence."""
    tokens = row.get("tokens", [])
    logprobs = row.get("logprobs", [])
    checks = dict(
        requested_tokens=expected_tokens > 0 and len(tokens) == expected_tokens,
        probability_coverage=len(logprobs) == len(tokens) > 0
        and all(str(token) in values for token, values in zip(tokens, logprobs)),
        finite_probabilities=not row.get("nonfinite_logprobs")
        and bool(logprobs)
        and all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for values in logprobs
            for value in values.values()
        ),
    )
    return dict(**checks, passed=all(checks.values()))


def stream_timing(first_seconds, last_seconds, first_tokens, final_tokens):
    """Client-observed latency; exclude the whole initial speculative burst."""
    if (
        first_seconds is None
        or last_seconds is None
        or not math.isfinite(first_seconds)
        or not math.isfinite(last_seconds)
        or not 0 <= first_seconds <= last_seconds
        or not 0 < first_tokens <= final_tokens
    ):
        raise ValueError("Stream timing requires monotonic, nonempty output")
    elapsed = last_seconds - first_seconds
    remaining = final_tokens - first_tokens
    return dict(
        first_output_seconds=first_seconds,
        last_output_seconds=last_seconds,
        first_output_tokens=first_tokens,
        post_first_output_tokens=remaining,
        post_first_output_tokens_per_second=remaining / elapsed
        if elapsed > 0 and remaining > 0
        else None,
        scope="Client-observed stream timing; not an isolated prefill/kernel benchmark",
    )


def finite_logprobs(rows):
    """Keep a JSON-safe failure record instead of losing NaN evidence."""
    invalid = []
    encoded = []
    for position, values in enumerate(rows):
        encoded_row = {}
        for token, value in values.items():
            probability = value.logprob
            if math.isfinite(probability):
                encoded_row[str(token)] = probability
            else:
                encoded_row[str(token)] = None
                invalid.append(
                    dict(position=position, token=token, value=str(probability))
                )
        encoded.append(encoded_row)
    return encoded, invalid


def compare_common_history_logprobs(reference, candidate):
    """Measure recorded top-k differences only while input histories match."""
    count = min(len(reference["tokens"]), len(candidate["tokens"]))
    first_token_difference = next(
        (i for i in range(count) if reference["tokens"][i] != candidate["tokens"][i]),
        None,
    )
    # The first divergent output still has the same input history. Later outputs
    # do not, so comparing their probabilities would mix cause and consequence.
    wanted = count if first_token_difference is None else first_token_difference + 1
    positions = min(wanted, len(reference["logprobs"]), len(candidate["logprobs"]))
    differences, intersections, first_logprob_difference = [], [], None
    for i in range(positions):
        left, right = reference["logprobs"][i], candidate["logprobs"][i]
        if any(
            value is None or not math.isfinite(value)
            for row in (left, right)
            for value in row.values()
        ):
            raise ValueError("Cannot compare missing or nonfinite log probabilities")
        if left != right and first_logprob_difference is None:
            first_logprob_difference = i
        shared = left.keys() & right.keys()
        intersections.append(len(shared))
        differences.extend(abs(left[token] - right[token]) for token in shared)
    return dict(
        first_token_difference=first_token_difference,
        first_logprob_difference=first_logprob_difference,
        compared_positions=positions,
        complete_common_history=positions == wanted and wanted > 0,
        min_topk_intersection=min(intersections, default=0),
        max_common_token_logprob_delta=max(differences, default=None),
        mean_common_token_logprob_delta=(
            sum(differences) / len(differences) if differences else None
        ),
        scope="Top-k intersection only; not a full-distribution metric or pass gate",
    )


def compare_prefix_shapes(reference, candidate, start_token):
    """Compare observed packed suffix steps, not request identity or numerics."""
    if start_token < 0:
        raise ValueError("Suffix comparison requires a nonnegative token boundary")

    def suffix(steps):
        # Keep whole steps: clipping would conceal a chunk crossing the boundary
        # or unrelated rows sharing the same computation shape.
        return [
            {key: step[key] for key in ("ranges", "prefill_rows", "total_rows")}
            for step in steps
            if any(end > start_token for _, end in step["ranges"])
        ]

    left, right = suffix(reference), suffix(candidate)
    return dict(
        start_token=start_token,
        observed=bool(left and right),
        same_packed_steps=bool(left and right) and left == right,
        reference_steps=left,
        candidate_steps=right,
        scope="position ranges and row counts only; not request IDs or equality proof",
    )


def compare_request_execution(
    reference, reference_id, candidate, candidate_id, start_token
):
    """Distinguish own-query changes from changes in the surrounding packed batch."""
    from itertools import zip_longest

    if start_token < 0:
        raise ValueError("Request comparison requires a nonnegative token boundary")

    def selected(steps, request_id):
        result = []
        for step in steps:
            batch = step.get("request_batch")
            if batch is None:
                continue
            own = [q for q in batch["requests"] if q["request_id"] == request_id]
            if len(own) > 1:
                raise ValueError("Duplicate request in a packed execution step")
            if not own or not any(
                end > start_token for _, end in own[0]["position_ranges"]
            ):
                continue
            result.append(
                dict(
                    own_query={
                        key: own[0][key]
                        for key in ("position_ranges", "input_ids_sha256")
                    },
                    total_rows=step["total_rows"],
                    padding_rows=batch["padding_rows"],
                    packed_requests=[
                        {key: value for key, value in q.items() if key != "request_id"}
                        for q in batch["requests"]
                    ],
                )
            )
        return result

    left = selected(reference, reference_id)
    right = selected(candidate, candidate_id)

    def first_difference(own_only):
        for i, (a, b) in enumerate(zip_longest(left, right)):
            if a is None or b is None:
                return i
            compared_a = a["own_query"] if own_only else a
            compared_b = b["own_query"] if own_only else b
            if compared_a != compared_b:
                return i
        return None

    return dict(
        observed=bool(left and right),
        start_token=start_token,
        first_own_query_difference=first_difference(True),
        first_packed_query_difference=first_difference(False),
        reference_steps=left,
        candidate_steps=right,
        scope="Whole observed steps; not state equality or a correctness verdict",
    )


def divergent_prefix_prompts(base, suffixes, min_shared_tokens):
    """Keep a real common prefix while giving each branch a private prompt tail."""
    if not 0 < min_shared_tokens < len(base) or len(suffixes) < 2:
        raise ValueError("Divergent-prefix control requires multiple bounded branches")
    if any(not tail or len(tail) > len(base) - min_shared_tokens for tail in suffixes):
        raise ValueError("Branch suffix would remove the required common prefix")
    prompts = [list(base[: -len(tail)]) + list(tail) for tail in suffixes]
    if len({tuple(prompt) for prompt in prompts}) != len(prompts):
        raise ValueError("Branch prompts must be distinct")
    shared = next(
        (
            i
            for i, tokens in enumerate(zip(*prompts, strict=True))
            if len(set(tokens)) > 1
        ),
        len(base),
    )
    return prompts, shared


async def prefix_screen(args, report, save):
    from prefix_cache_benchmark import (
        NativeIterationCapture,
        performance_screen,
        prefill_profile_screen,
        prefill_profiler_config,
        prefill_staging_ab_screen,
        validate_performance,
    )
    from qualify_session_swap import compare_tokens
    from session_transfer_audit import (
        native_load_failure_screen,
        validate_native_load_failure,
    )
    from transformers import AutoTokenizer

    from vllm import SamplingParams
    from vllm.compilation.cuda_graph import CUDAGraphLogging
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
        OffloadingConnectorStats,
    )
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.metrics.loggers import StatLoggerBase

    validate_gpu_cache_budget(args)
    from session_sustained_decode import run as sustained_run
    from session_sustained_decode import validate as validate_sustained

    validate_sustained(args)
    sustained = getattr(args, "sustained_decode", False)
    staging_c4 = (getattr(args, "prefill_staging_ab", False)
                  and getattr(args, "prefill_staging_concurrency", 2) == 4)
    config = prefix_engine_config(
        json.loads(args.engine_args.read_text()),
        args.no_mtp,
        getattr(args, "private_cold_buffers", False),
        getattr(args, "tp4", False),
        getattr(args, "prefix_prefill_budget", None),
        getattr(args, "balanced_prefix_prefill", False),
        32768 if staging_c4 else getattr(args, "continuation_context", 8192),
        getattr(args, "serving_decode", False),
        max(4 if staging_c4 or sustained else 1,
            getattr(args, "continuation_concurrency", 1),
            getattr(args, "cancellation_concurrency", 2)
            if getattr(args, "active_cancel", False) else 1),
        gpu_cache_mib=getattr(args, "prefix_gpu_cache_mib", None),
    )
    real_idle = getattr(args, "sustained_real_idle_expiry", False)
    idle_ttl = getattr(args, "sustained_idle_ttl", 3600)
    if real_idle:
        config["kv_transfer_config"]["kv_connector_extra_config"][
            "idle_ttl_seconds"] = idle_ttl
    balanced = getattr(args, "balanced_prefix_prefill", False)
    validate_balanced_observation(args)
    validate_continuation_observation(args)
    validate_cancellation_observation(args)
    validate_serving_decode(args)
    validate_copy_audit(args)
    validate_performance(args)
    validate_native_load_failure(args)
    failure_mode = getattr(args, "native_load_failure", None)
    if failure_mode and (
        os.environ.get("FLASH_NATIVE_LOAD_FAILURE", "0")
        != ("1" if failure_mode == "inject" else "0")
        or config.get("profiler_config")
        or config.get("enable_logging_iteration_details")
    ):
        raise ValueError("Failure diagnostic gate or instrumentation does not match")
    profile_prefill = getattr(args, "profile_prefix_prefill", False)
    staging_ab = getattr(args, "prefill_staging_ab", False)
    if staging_ab and (
        config.get("profiler_config")
        or config.get("enable_logging_iteration_details")
        or os.environ.get("FLASH_PREFILL_STAGING_AB") != "1"
    ):
        raise ValueError("Staging AB requires its explicit gate and no instrumentation")
    profile_directory = args.output.parent / "prefill-profile"
    if profile_prefill:
        if config.get("profiler_config") or profile_directory.exists():
            raise ValueError(
                "Prefill profile requires a fresh output and no profiler override"
            )
        config["profiler_config"] = prefill_profiler_config(profile_directory)
    iteration_details = (
        getattr(args, "performance_iteration_details", False) or profile_prefill
    )
    if iteration_details:
        config["enable_logging_iteration_details"] = True
    iteration_capture = NativeIterationCapture() if iteration_details else None
    audit_native_copies = getattr(args, "audit_native_copies", False)
    continuations = getattr(args, "continuations", False)
    active_cancel = getattr(args, "active_cancel", False)
    cancellation_concurrency = getattr(args, "cancellation_concurrency", 2)
    cancel_names = ["cancel-victim", *(
        [f"cancel-survivor-{i}" for i in range(3)]
        if cancellation_concurrency == 4 else ["cancel-survivor"]
    )]
    audit_active_cancel = getattr(args, "audit_active_cancel", False)
    paired_loads = getattr(args, "paired_cold_loads", False)
    continuation_load_gate = getattr(args, "continuation_load_gate", False)
    if continuation_load_gate and not (
        paired_loads and args.continuations and args.continuation_concurrency == 4
    ):
        raise ValueError("Continuation load gate requires paired C4 continuations")
    idle_expiry = getattr(args, "idle_expiry", False)
    pending_expiry = getattr(args, "pending_expiry", False)
    if (idle_expiry or pending_expiry) and not paired_loads:
        raise ValueError(
            "Expiry diagnostics require the isolated paired-load scheduler"
        )
    if paired_loads and not balanced:
        raise ValueError("Paired cold loads require the balanced C2 diagnostic")
    stable_layout = getattr(args, "stable_expert_layout", False)
    numerical_control = getattr(args, "mlp_numerical_control", False)
    if numerical_control and (not stable_layout or args.serving_decode
                              or config["tensor_parallel_size"] != 4):
        raise ValueError("MLP numerical control requires eager TP4 stable layout")
    divergent = getattr(args, "divergent_prefixes", False)
    if divergent and getattr(args, "prefix_concurrency", 1) < 2:
        raise ValueError("Divergent-prefix control requires concurrent requests")
    stable_qsa = getattr(args, "stable_qsa_selection", False)
    stable_ties = getattr(args, "stable_qsa_ties", False)
    if stable_ties and not stable_qsa:
        raise ValueError("Stable QSA ties require the selection diagnostic")
    trace_prefill = getattr(args, "trace_prefix_prefill", False)
    trace_chunks = getattr(args, "trace_prompt_chunks", False)
    trace_shapes = getattr(args, "trace_prefix_shapes", False)
    trace_pages = getattr(args, "trace_prefix_pages", False)
    if trace_pages and not trace_shapes:
        raise ValueError("Prefix-page tracing requires the phase/shape trace")
    report.update(
        scope=(
            "ordinary independent requests; "
            f"{'CUDA-graph decode' if args.serving_decode else 'eager'} "
            f"TP{config['tensor_parallel_size']}"
            f"/PP{config['pipeline_parallel_size']}; "
            f"{config['max_model_len']}-token screen"
        ),
        config=config,
        requests=[],
        complete=False,
        screen_passed=False,
        session_integration_qualified=False,
        stable_expert_layout=stable_layout,
        mlp_numerical_control=numerical_control,
        sustained_decode=sustained,
        sustained_real_idle_expiry=real_idle,
        sustained_uniform_outputs=getattr(args, "sustained_uniform_outputs", False),
        sustained_observe_lookups=getattr(args, "sustained_observe_lookups", False),
        sustained_idle_ttl=idle_ttl,
        sustained_output_tokens=getattr(args, "sustained_output_tokens", 1024),
        divergent_prefixes=divergent,
        stable_qsa_selection=stable_qsa,
        stable_qsa_ties=stable_ties,
        trace_prefix_prefill=trace_prefill,
        trace_prompt_chunks=trace_chunks,
        trace_prefix_shapes=trace_shapes,
        trace_prefix_pages=trace_pages,
        balanced_prefix_prefill=balanced,
        paired_cold_loads=paired_loads,
        continuation_load_gate=continuation_load_gate,
        idle_expiry=idle_expiry,
        pending_expiry=pending_expiry,
        continuations=continuations,
        continuation_concurrency=getattr(args, "continuation_concurrency", 1),
        active_cancel=active_cancel,
        cancellation_concurrency=cancellation_concurrency,
        audit_active_cancel=audit_active_cancel,
        expire_active_cache=getattr(args, "expire_active_cache", False),
        performance_repeats=getattr(args, "performance_repeats", 0),
        performance_iteration_details=iteration_details,
        profile_prefix_prefill=profile_prefill,
        prefill_staging_ab=staging_ab,
        prefill_staging_concurrency=getattr(args, "prefill_staging_concurrency", 2),
        native_load_failure_mode=failure_mode,
        qsa_stage_all_prefill=os.environ.get("VLLM_QSA_STAGE_ALL_PREFILL", "0") == "1",
        timing_perturbed_by_audit=bool(
            trace_shapes
            or audit_active_cancel
            or audit_native_copies
            or profile_prefill
            or failure_mode
        ),
        audit_native_copies=audit_native_copies,
        serving_decode=getattr(args, "serving_decode", False),
        untraced_balanced_prefix=getattr(args, "untraced_balanced_prefix", False),
    )
    save()
    counters = OffloadingConnectorStats()
    cache_hits = {"local": 0, "connector": 0}
    phase_running = {"max": 0}
    graph_logging = None

    class Recorder(StatLoggerBase):
        def __init__(self, vllm_config, engine_index=0):
            nonlocal graph_logging
            if vllm_config.observability_config.cudagraph_metrics:
                graph_logging = CUDAGraphLogging(
                    vllm_config.compilation_config.cudagraph_mode,
                    vllm_config.compilation_config.cudagraph_capture_sizes,
                )

        def log_engine_initialized(self):
            pass

        def record(
            self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0
        ):
            if scheduler_stats is None:
                return
            if iteration_capture is not None:
                iteration_capture.record(scheduler_stats.iteration_details)
            if (
                graph_logging is not None
                and scheduler_stats.cudagraph_stats is not None
            ):
                graph_logging.observe(scheduler_stats.cudagraph_stats)
            phase_running["max"] = max(
                phase_running["max"], scheduler_stats.num_running_reqs
            )
            if scheduler_stats.kv_connector_stats:
                counters.aggregate(
                    OffloadingConnectorStats(data=scheduler_stats.kv_connector_stats)
                )
            for name, stats in (
                ("local", scheduler_stats.prefix_cache_stats),
                ("connector", scheduler_stats.connector_prefix_cache_stats),
            ):
                if stats is not None:
                    cache_hits[name] += stats.hits

    tokenizer = AutoTokenizer.from_pretrained(config["model"], local_files_only=True)
    text = (
        "A pelican rides a bicycle beside a quiet lake. The bicycle has two wheels. "
        "The pelican balances carefully and rings its bell. "
    )
    body = tokenizer.encode(text, add_special_tokens=False)
    tail = tokenizer.encode(
        "\nExplain the scene in a detailed short story.\n", add_special_tokens=False
    )

    def prompt(index, length=7100):
        start = tokenizer.encode(f"Document {index}:\n", add_special_tokens=False)
        count = length - len(start) - len(tail)
        return start + (body * (count // len(body) + 1))[:count] + tail

    if getattr(args, "sustained_observe_lookups", False):
        engine_args = AsyncEngineArgs(**config)
        resolved = engine_args.create_engine_config()
        if resolved.scheduler_config.scheduler_cls is not None:
            raise ValueError("Observer cannot replace an existing custom scheduler")
        native = resolved.scheduler_config.get_scheduler_cls()
        scheduler_name = ("CacheLookupAsyncScheduler"
                          if resolved.scheduler_config.async_scheduling
                          else "CacheLookupScheduler")
        resolved.scheduler_config.scheduler_cls = f"prefix_load_barrier.{scheduler_name}"
        if not issubclass(resolved.scheduler_config.get_scheduler_cls(), native):
            raise AssertionError("Observer changed native scheduler inheritance")
        resolved.additional_config["flash_next_diagnostic_lookup_observer"] = str(
            args.output.with_name("cache-lookups.jsonl"))
        report["timing_perturbed_by_audit"] = True
        report["resolved_scheduler"] = dict(native_class=native.__name__,
                                            diagnostic_class=scheduler_name)
        save()
        engine = AsyncLLM.from_vllm_config(
            resolved, stat_loggers=[Recorder],
            enable_log_requests=engine_args.enable_log_requests,
            disable_log_stats=engine_args.disable_log_stats)
    elif paired_loads:
        engine_args = AsyncEngineArgs(**config)
        resolved = engine_args.create_engine_config()
        if resolved.scheduler_config.scheduler_cls is not None:
            raise ValueError("A diagnostic cannot replace an existing custom scheduler")
        native = resolved.scheduler_config.get_scheduler_cls()
        scheduler_name = (
            "PairedLoadAsyncScheduler"
            if resolved.scheduler_config.async_scheduling
            else "PairedLoadScheduler"
        )
        resolved.scheduler_config.scheduler_cls = (
            f"prefix_load_barrier.{scheduler_name}"
        )
        if not issubclass(resolved.scheduler_config.get_scheduler_cls(), native):
            raise AssertionError("Diagnostic changed the native scheduler inheritance")
        barrier_path = args.output.with_name("prefix-load-barrier.json")
        resolved.additional_config["flash_next_diagnostic_load_barrier"] = str(
            barrier_path
        )
        resolved.additional_config["flash_next_diagnostic_c4_load_gate"] = (
            continuation_load_gate
        )
        if continuations and trace_pages:
            ownership_path = args.output.with_name("prefix-resident-ownership.json")
            resolved.additional_config["flash_next_diagnostic_resident_ownership"] = (
                str(ownership_path)
            )
            resolved.additional_config["flash_next_diagnostic_resident_boundaries"] = (
                continuation_page_boundaries(
                    config["max_model_len"], args.continuation_concurrency
                )
            )
            report["resident_ownership_requested"] = True
        if idle_expiry:
            expiry_path = args.output.with_name("prefix-idle-expiry.json")
            resolved.additional_config["flash_next_diagnostic_idle_expiry"] = str(
                expiry_path
            )
        if pending_expiry:
            pending_expiry_path = args.output.with_name("prefix-pending-expiry.json")
            resolved.additional_config["flash_next_diagnostic_pending_expiry"] = str(
                pending_expiry_path
            )
        if audit_active_cancel:
            active_abort_path = args.output.with_name("prefix-active-abort.json")
            resolved.additional_config["flash_next_diagnostic_active_abort"] = str(
                active_abort_path
            )
            resolved.additional_config["flash_next_diagnostic_active_survivors"] = (
                cancel_names[1:]
            )
            resolved.additional_config["flash_next_diagnostic_active_expiry"] = bool(
                getattr(args, "expire_active_cache", False)
            )
        report["resolved_scheduler"] = dict(
            native_class=native.__name__,
            diagnostic_class=scheduler_name,
            async_scheduling=resolved.scheduler_config.async_scheduling,
        )
        save()
        engine = AsyncLLM.from_vllm_config(
            resolved,
            stat_loggers=[Recorder],
            enable_log_requests=engine_args.enable_log_requests,
            disable_log_stats=engine_args.disable_log_stats,
        )
    else:
        engine = AsyncLLM.from_engine_args(
            AsyncEngineArgs(**config), stat_loggers=[Recorder]
        )
    report["engine_started"] = True
    save()

    async def generate(
        name,
        index=0,
        skip_cache=False,
        tokens=None,
        prompt_ids=None,
        cache_salt=None,
        repetition_penalty=1.0,
        observe=None,
        expect_abort=False,
    ):
        if (
            trace_shapes
            and not name.startswith(("batch-", "turn2-", "turn4-"))
            and "pressure" not in name
        ):
            await engine.collective_rpc("begin_prefix_shape_trace", args=(name,))
        before = counters.reduce()
        graph_start = len(graph_logging.stats) if graph_logging is not None else 0
        started = time.monotonic()
        output = None
        first_seconds = last_seconds = None
        first_tokens = 0
        prompt_ids = prompt(index) if prompt_ids is None else prompt_ids
        inputs = {"prompt_token_ids": prompt_ids}
        if cache_salt is not None:
            inputs["cache_salt"] = cache_salt
        async for output in engine.generate(
            prompt=inputs,
            sampling_params=SamplingParams(
                temperature=0,
                seed=args.seed,
                repetition_penalty=repetition_penalty,
                max_tokens=tokens or args.tokens,
                ignore_eos=True,
                logprobs=5,
                skip_reading_prefix_cache=skip_cache,
            ),
            request_id=name,
        ):
            if output.outputs[0].token_ids:
                last_seconds = time.monotonic() - started
                if first_seconds is None:
                    first_seconds = last_seconds
                    first_tokens = len(output.outputs[0].token_ids)
            if observe is not None:
                await observe(name, output)
        if output is None or not output.finished:
            raise AssertionError("ordinary request did not finish")
        # Allow frontend stats already queued by the final engine step to drain.
        await asyncio.sleep(0)
        logprobs, invalid = finite_logprobs(output.outputs[0].logprobs or [])
        row = dict(
            name=name,
            prompt_index=index,
            prompt_tokens=len(prompt_ids),
            prompt_sha256=hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest(),
            skip_cache=skip_cache,
            cache_salt=cache_salt,
            repetition_penalty=repetition_penalty,
            seconds=time.monotonic() - started,
            started_monotonic=started,
            tokens=list(output.outputs[0].token_ids),
            cached_tokens=output.num_cached_tokens or 0,
            transfer_delta={
                key: value - before.get(key, 0)
                for key, value in counters.reduce().items()
                if key in ("vllm:kv_offload_load_bytes", "vllm:kv_offload_store_bytes")
            },
            logprobs=logprobs,
            nonfinite_logprobs=invalid,
            requested_output_tokens=tokens or args.tokens,
            finish_reason=output.outputs[0].finish_reason,
            expected_abort=expect_abort,
            native_cudagraph_window=(
                [
                    dict(asdict(stat), steps=count)
                    for stat, count in Counter(
                        graph_logging.stats[graph_start:]
                    ).items()
                ]
                if graph_logging is not None
                else None
            ),
            stream_timing=stream_timing(
                first_seconds,
                last_seconds,
                first_tokens,
                len(output.outputs[0].token_ids),
            ),
            speculative_decoding=output.outputs[0].spec_decode_metrics.to_dict()
            if output.outputs[0].spec_decode_metrics is not None
            else None,
        )
        row["output_completeness"] = complete_request_output(
            row, len(row["tokens"]) if expect_abort else row["requested_output_tokens"]
        )
        if expect_abort:
            row["output_completeness"]["intentional_partial_abort"] = (
                row["finish_reason"] == "abort"
                and 0 < len(row["tokens"]) < row["requested_output_tokens"]
            )
            row["output_completeness"]["passed"] &= row["output_completeness"][
                "intentional_partial_abort"
            ]
        report["requests"].append(row)
        report["native_counters"] = counters.reduce()
        report["prefix_hit_tokens"] = dict(cache_hits)
        if graph_logging is not None:
            report["native_cudagraph_metrics"] = dict(
                table=graph_logging.generate_metric_table(),
                scope="Native target-step dispatch; concurrent request windows overlap",
            )
        save()
        if invalid:
            raise AssertionError("nonfinite logprobs in ordinary prefix screen")
        if not row["output_completeness"]["passed"]:
            raise AssertionError("incomplete output in ordinary prefix screen")
        return row

    async def performance_batch(name, prompts, **kwargs):
        phase_running["max"] = 0
        before = counters.reduce().get("vllm:kv_offload_load_bytes", 0)
        started = time.monotonic()
        if iteration_capture is not None:
            iteration_capture.start(name)
        rows = await admitted_batch(
            engine,
            [
                lambda i=i, ids=ids: generate(f"{name}-{i}", prompt_ids=ids, **kwargs)
                for i, ids in enumerate(prompts)
            ],
        )
        phase = dict(
            requests=rows,
            max_running=phase_running["max"],
            started_monotonic=started,
            ended_monotonic=time.monotonic(),
            load_bytes=counters.reduce().get("vllm:kv_offload_load_bytes", 0) - before,
        )
        if iteration_capture is not None:
            phase["native_iterations"] = iteration_capture.finish()
        return phase

    async def record_memory(phase):
        try:
            rows = await engine.collective_rpc("snapshot_cuda_memory")
            if sorted(row["rank"] for row in rows) != list(range(config["tensor_parallel_size"])):
                raise ValueError("Memory snapshot lacks every TP rank")
            report.setdefault("cuda_memory", {})[phase] = rows
        except Exception as error:
            report.setdefault("cuda_memory_errors", {})[phase] = (
                f"{type(error).__name__}: {error}")
        save()

    try:
        await record_memory("initialized")
        if failure_mode:
            from vllm.v1.engine.exceptions import EngineDeadError

            report["native_load_failure_control"] = {}
            await native_load_failure_screen(
                engine,
                generate,
                failure_mode,
                args.output.with_name("native-load-failure-rank3.json"),
                EngineDeadError,
                report["native_load_failure_control"],
                save,
            )
            report["complete"] = True
            save()
            return
        if profile_prefill or staging_ab:
            prompts, _ = divergent_prefix_prompts(
                prompt(2000, 29756 if staging_c4 else 7100),
                [
                    tokenizer.encode(
                        f"\nBranch {i}: tell the pelican story from viewpoint {i}.\n",
                        add_special_tokens=False,
                    )
                    for i in range(4 if staging_c4 else 2)
                ],
                min_shared_tokens=29264 if staging_c4 else 6608,
            )
            if staging_ab:
                report["staging_ab"] = await prefill_staging_ab_screen(
                    engine, performance_batch, prompts
                )
            else:
                report["prefill_profile"] = await prefill_profile_screen(
                    engine, performance_batch, prompts, profile_directory
                )
            report["complete"] = True
            save()
            return
        if audit_native_copies:
            report["native_copy_audit_install"] = await engine.collective_rpc(
                "install_native_copy_audit"
            )
            save()
        if trace_shapes:
            report["prefix_shape_install"] = await engine.collective_rpc(
                "install_prefix_shape_trace",
                args=(config["max_model_len"] if continuations else 7100, trace_pages),
            )
            save()
        if stable_qsa:
            report["stable_qsa_install"] = await engine.collective_rpc(
                "install_stable_qsa_selection", args=(stable_ties,)
            )
            save()
        if stable_layout:
            report["stable_layout_install"] = await engine.collective_rpc(
                "install_stable_expert_layout"
            )
            save()
        if numerical_control:
            report["mlp_numerical_install"] = await engine.collective_rpc(
                "install_mlp_numerical_control"
            )
            save()
        if trace_chunks:
            report["prefill_trace_install"] = await engine.collective_rpc(
                "install_prompt_chunk_trace", args=(7100, args.reference_runs)
            )
            save()
        elif trace_prefill:
            report["prefill_trace_install"] = await engine.collective_rpc(
                "install_prefix_prefill_trace", args=(args.reference_runs,)
            )
            save()
        references = []
        for i in range(args.reference_runs):
            if trace_chunks:
                await engine.collective_rpc("begin_prompt_chunk_trace", args=(i,))
            references.append(await generate(f"reference-{i}", skip_cache=True))
        if trace_prefill or trace_chunks:
            report["prefill_trace"] = await engine.collective_rpc(
                "collect_session_prefill_trace"
            )
            save()
        report["baseline_comparisons"] = [
            compare_tokens(references[0]["tokens"], row["tokens"])
            for row in references[1:]
        ]
        report["baseline_repeatable"] = all(
            row["equal"] for row in report["baseline_comparisons"]
        )
        hot = await generate("hot")
        report["hot_comparison"] = compare_tokens(
            references[0]["tokens"], hot["tokens"]
        )
        report["hot_exact"] = report["hot_comparison"]["equal"]
        report["hot_cached_tokens"] = hot["cached_tokens"]
        if trace_shapes:
            await engine.collective_rpc("begin_prefix_shape_trace", args=(None,))
        for index in range(1, args.reuse_rounds + 1):
            await generate(f"pressure-{index}", index=index, tokens=8)
        cold = await generate("cold")
        report["cold_comparison"] = compare_tokens(
            references[0]["tokens"], cold["tokens"]
        )
        report["cold_exact"] = report["cold_comparison"]["equal"]
        report["hot_cold_comparison"] = dict(
            tokens=compare_tokens(hot["tokens"], cold["tokens"]),
            logprobs_exact=hot["logprobs"] == cold["logprobs"],
        )
        report["logprob_diagnostics"] = {
            name: compare_common_history_logprobs(references[0], row)
            for name, row in (("hot", hot), ("cold", cold))
        }
        report["cold_cached_tokens"] = cold["cached_tokens"]
        report["cold_load_bytes"] = cold["transfer_delta"].get(
            "vllm:kv_offload_load_bytes", 0
        )
        report["store_bytes"] = counters.reduce().get("vllm:kv_offload_store_bytes", 0)
        concurrency = getattr(args, "prefix_concurrency", 1)
        if concurrency > 1:
            # Keep the balanced C2 fixture out of the preceding C1 prefix namespace.
            batch_prompt_index = 2000 if balanced else 0
            batch_prompts = [prompt(batch_prompt_index) for _ in range(concurrency)]
            shared_tokens = len(batch_prompts[0])
            if divergent:
                batch_prompts, shared_tokens = divergent_prefix_prompts(
                    batch_prompts[0],
                    [
                        tokenizer.encode(
                            f"\nBranch {i}: tell the pelican story "
                            f"from viewpoint {i}.\n",
                            add_special_tokens=False,
                        )
                        for i in range(concurrency)
                    ],
                    min_shared_tokens=6608,
                )

            async def batch(name, skip_cache=False):
                if trace_shapes:
                    await engine.collective_rpc(
                        "begin_prefix_shape_trace", args=(name,)
                    )
                phase_running["max"] = 0
                before = counters.reduce().get("vllm:kv_offload_load_bytes", 0)
                calls = [
                    lambda i=i: generate(
                        f"{name}-{i}",
                        skip_cache=skip_cache,
                        prompt_ids=batch_prompts[i],
                    )
                    for i in range(concurrency)
                ]
                rows = (
                    await admitted_batch(engine, calls)
                    if balanced
                    else await asyncio.gather(*(call() for call in calls))
                )
                return dict(
                    requests=rows,
                    max_running=phase_running["max"],
                    load_bytes=counters.reduce().get("vllm:kv_offload_load_bytes", 0)
                    - before,
                )

            concurrent = report["concurrent"] = {
                "passed": False,
                "concurrency": concurrency,
                "divergent_prefixes": divergent,
                "shared_prompt_tokens": shared_tokens,
                "prompt_index": batch_prompt_index,
            }
            concurrent["reference_a"] = await batch("batch-reference-a", True)
            concurrent["reference_b"] = await batch("batch-reference-b", True)
            concurrent["hot"] = await batch("batch-hot")
            if trace_shapes:
                await engine.collective_rpc("begin_prefix_shape_trace", args=(None,))
            for index in range(1, args.reuse_rounds + 1):
                await generate(f"batch-pressure-{index}", index=1000 + index, tokens=8)
            concurrent["cold"] = await batch("batch-cold")
            concurrent["comparisons"] = {
                name: [
                    compare_tokens(a["tokens"], b["tokens"])
                    for a, b in zip(
                        concurrent["reference_a"]["requests"],
                        concurrent[name]["requests"],
                        strict=True,
                    )
                ]
                for name in ("reference_b", "hot", "cold")
            }
            concurrent["hot_cold_comparisons"] = [
                dict(
                    tokens=compare_tokens(a["tokens"], b["tokens"]),
                    logprobs_exact=a["logprobs"] == b["logprobs"],
                )
                for a, b in zip(
                    concurrent["hot"]["requests"],
                    concurrent["cold"]["requests"],
                    strict=True,
                )
            ]
            concurrent["logprob_diagnostics"] = {
                name: [
                    compare_common_history_logprobs(a, b)
                    for a, b in zip(
                        concurrent["reference_a"]["requests"],
                        concurrent[name]["requests"],
                        strict=True,
                    )
                ]
                for name in ("reference_b", "hot", "cold")
            }
            concurrent["passed"] = (
                all(
                    row["equal"]
                    for rows in concurrent["comparisons"].values()
                    for row in rows
                )
                and all(
                    concurrent[name]["max_running"] >= concurrency
                    for name in ("reference_a", "reference_b", "hot", "cold")
                )
                and all(
                    row["cached_tokens"] > 0
                    for name in ("hot", "cold")
                    for row in concurrent[name]["requests"]
                )
                and concurrent["cold"]["load_bytes"] > 0
            )
            save()
        if paired_loads:
            report["load_barrier"] = json.loads(barrier_path.read_text())
            barrier = report["load_barrier"]
            report["load_barrier_exercised"] = bool(
                barrier.get("released")
                and len(barrier.get("promoted", [])) == 2
                and set(barrier["promoted"])
                == set(barrier.get("ready_request_ids", []))
            )
        if pending_expiry:
            pending_evidence = json.loads(pending_expiry_path.read_text())
            report["pending_expiry_control"] = dict(
                evidence=pending_evidence,
                **compare_pending_expiry(pending_evidence, report["concurrent"]),
            )
            save()
        if idle_expiry:
            expired = await generate("expiry-trigger")
            expiry_evidence = json.loads(expiry_path.read_text())
            reference = await generate("expiry-reference", skip_cache=True)
            refreshed = await generate("expiry-hot")
            hot = next(row for row in report["requests"] if row["name"] == "hot")
            report["expiry_control"] = dict(
                evidence=expiry_evidence,
                **compare_idle_expiry(
                    expiry_evidence, expired, reference, refreshed, hot
                ),
            )
            save()
        if active_cancel:
            salt = "active-cancel-shared-prefix"
            cancel_prompts = batch_prompts
            if cancellation_concurrency == 4:
                seed_length = continuation_seed_tokens(config["max_model_len"])
                cancel_prompts, _ = divergent_prefix_prompts(
                    prompt(8000, seed_length + 512)[:seed_length],
                    [tokenizer.encode(f"\nCancellation branch {i}.\n",
                                      add_special_tokens=False) for i in range(4)],
                    min_shared_tokens=seed_length - 492,
                )
            for i in range(cancellation_concurrency):
                await generate(
                    f"cancel-warm-{i}", prompt_ids=cancel_prompts[i], cache_salt=salt
                )
            for index in range(1, args.reuse_rounds + 1):
                await generate(f"cancel-pressure-{index}", index=6000 + index, tokens=8)
            if audit_active_cancel:
                report["active_cache_audit_install"] = await engine.collective_rpc(
                    "install_active_cache_audit"
                )
                save()
            controller = ActiveCancellation(
                engine, cancel_names[0], cancel_names[1:], audit=audit_active_cancel
            )
            phase_running["max"] = 0
            before = counters.reduce().get("vllm:kv_offload_load_bytes", 0)
            rows = await admitted_batch(
                engine,
                [
                    lambda i=i, name=name: generate(
                        name,
                        prompt_ids=cancel_prompts[i],
                        cache_salt=salt,
                        tokens=512 if i == 0 else 128,
                        expect_abort=i == 0,
                        observe=controller.observe,
                    )
                    for i, name in enumerate(cancel_names)
                ],
            )
            control = report["cancellation_control"] = dict(
                scope=(
                    "Native active abort after cold reuse; paused cache-byte audit"
                    if audit_active_cancel
                    else "Native active abort after cold reuse; no tensor-byte trace"
                ),
                requests=rows,
                concurrency=cancellation_concurrency,
                context_tokens=config["max_model_len"],
                serving_decode=bool(getattr(args, "serving_decode", False)),
                at_abort=controller.at_abort,
                abort_acknowledged=controller.acknowledged,
                max_running=phase_running["max"],
                load_bytes=counters.reduce().get("vllm:kv_offload_load_bytes", 0)
                - before,
                frontend_drained=not engine.output_processor.has_unfinished_requests(),
                reuse=[],
                passed=False,
                audit_required=audit_active_cancel,
                active_expiry_required=getattr(args, "expire_active_cache", False),
            )
            if audit_active_cancel:
                control["cache_audit"] = controller.cache_audit
                control["ownership_audit"] = json.loads(active_abort_path.read_text())
                control["cache_audit_assessment"] = compare_active_cache_audit(
                    controller.cache_audit or {},
                    cancel_names[1:] if cancellation_concurrency == 4 else None,
                )
                report["active_cache_audit_remove"] = await engine.collective_rpc(
                    "remove_active_cache_audit"
                )
            save()
            for i in range(cancellation_concurrency):
                control["reuse"].append(
                    await generate(
                        f"cancel-reuse-{i}",
                        prompt_ids=cancel_prompts[i],
                        cache_salt=salt,
                    )
                )
            control["numerical_consistency"] = {
                "survivor_vs_reuse": compare_tokens(
                    rows[1]["tokens"], control["reuse"][1]["tokens"]
                ),
                "survivor_logprobs_exact": rows[1]["logprobs"]
                == control["reuse"][1]["logprobs"],
            }
            if cancellation_concurrency == 4:
                control["numerical_consistency"]["all_survivors"] = [
                    dict(name=rows[i]["name"],
                         tokens=compare_tokens(rows[i]["tokens"],
                                               control["reuse"][i]["tokens"]),
                         logprobs_exact=rows[i]["logprobs"]
                         == control["reuse"][i]["logprobs"])
                    for i in range(1, 4)
                ]
            control.update(cancellation_verdict(control))
            save()
        if sustained:
            base = prompt(9200, 32768)[:32764]
            markers = [tokenizer.encode(f"\nSustained branch {i}.\n",
                                        add_special_tokens=False) for i in range(4)]
            sustained_seeds = [base[:944] + marker + base[944 + len(marker):]
                               for marker in markers]

            async def sustained_batch(name, prompts, output_tokens, **kwargs):
                phase_running["max"] = 0
                before = counters.reduce().get("vllm:kv_offload_load_bytes", 0)
                rows = await admitted_batch(engine, [
                    lambda i=i: generate(f"{name}-{i}", prompt_ids=prompts[i],
                                         tokens=output_tokens[i], **kwargs)
                    for i in range(4)
                ])
                return dict(name=name, requests=rows,
                            max_running=phase_running["max"],
                            load_bytes=counters.reduce().get(
                                "vllm:kv_offload_load_bytes", 0) - before)

            def idle_progress(evidence):
                report["sustained_idle_progress"] = evidence
                save()

            report["sustained_control"] = await sustained_run(
                generate, sustained_batch, sustained_seeds, prompt,
                getattr(args, "sustained_output_tokens", 1024), args.reuse_rounds,
                real_idle_expiry=real_idle, idle_progress=idle_progress,
                idle_ttl=idle_ttl,
                uniform_outputs=getattr(args, "sustained_uniform_outputs", False),
            )
            save()
        if continuations:
            context = config["max_model_len"]
            continuation_concurrency = getattr(args, "continuation_concurrency", 1)
            page_plan = (
                continuation_page_boundaries(context, continuation_concurrency)
                if trace_pages else {}
            )
            report["continuation_page_boundaries"] = page_plan
            seed_length = continuation_seed_tokens(context)
            seed = prompt(9000, max(7100, seed_length + 512))[:seed_length]
            if continuation_concurrency > 1:
                seeds, _ = divergent_prefix_prompts(
                    seed,
                    [
                        tokenizer.encode(
                            f"\nConversation branch {i}.\n", add_special_tokens=False
                        )
                        for i in range(continuation_concurrency)
                    ],
                    min_shared_tokens=seed_length - 492,
                )

                async def continuation_batch(name, prompts, **kwargs):
                    boundaries = page_plan.get(name)
                    if trace_pages:
                        await engine.collective_rpc(
                            "begin_prefix_shape_trace",
                            args=(name, boundaries) if boundaries else (None,),
                        )
                    phase_running["max"] = 0
                    before = counters.reduce().get("vllm:kv_offload_load_bytes", 0)
                    rows = await admitted_batch(
                        engine,
                        [
                            lambda i=i: generate(
                                f"{name}-{i}", prompt_ids=prompts[i], **kwargs
                            )
                            for i in range(continuation_concurrency)
                        ],
                    )
                    if trace_pages:
                        await engine.collective_rpc(
                            "begin_prefix_shape_trace", args=(None,)
                        )
                    return dict(
                        name=name,
                        requests=rows,
                        page_audit_inputs={
                            str(boundary): [
                                hashlib.sha256(
                                    json.dumps(ids[:boundary]).encode()
                                ).hexdigest()
                                for ids in prompts
                            ]
                            for boundary in boundaries or ()
                        },
                        max_running=phase_running["max"],
                        load_bytes=counters.reduce().get(
                            "vllm:kv_offload_load_bytes", 0
                        )
                        - before,
                    )

                report["continuation_control"] = await concurrent_continuation_screen(
                    generate,
                    continuation_batch,
                    seeds,
                    tail,
                    prompt,
                    args.tokens,
                    args.reuse_rounds,
                    context,
                )
            else:
                report["continuation_control"] = await continuation_screen(
                    generate,
                    seed,
                    tail,
                    prompt,
                    args.tokens,
                    args.reuse_rounds,
                    context,
                )
            save()
        if getattr(args, "performance_repeats", 0):
            report["performance_control"] = await performance_screen(
                generate,
                performance_batch,
                batch_prompts,
                prompt,
                args.tokens,
                args.performance_repeats,
                args.reuse_rounds,
            )
            save()
            if not report["performance_control"]["passed"]:
                raise AssertionError(
                    "Performance fixture lacks matched cache-tier evidence"
                )
        if report.get("native_copy_audit_install"):
            from session_transfer_audit import copy_audit_verdict

            report["native_copy_audit"] = await engine.collective_rpc(
                "collect_native_copy_audit"
            )
            report["native_copy_audit_verdict"] = copy_audit_verdict(
                report["native_copy_audit"]
            )
            save()
        report["complete"] = True
        if continuation_load_gate:
            report["load_barrier"] = json.loads(barrier_path.read_text())
            turns = report["load_barrier"].get("continuation_turns", {})
            report["continuation_load_gate_exercised"] = (
                set(turns) == {"1", "2", "3"}
                and all(
                    row.get("released")
                    and len(row.get("promoted", [])) == 4
                    and len(set(row["promoted"])) == 4
                    and set(row["promoted"]) == set(row.get("ready_request_ids", []))
                    for row in turns.values()
                )
            )
        report["screen_passed"] = prefix_verdict(report)
        if sustained:
            report["screen_passed"] &= (
                report["sustained_control"]["lifecycle_passed"]
                and report["sustained_control"]["numerical_passed"]
            )
        save()
        if not report["screen_passed"]:
            raise AssertionError(
                "native prefix screen lacks exactness or real cold-hit evidence"
            )
    finally:
        try:
            if (
                report.get("native_copy_audit_install")
                and "native_copy_audit" not in report
            ):
                try:
                    report["native_copy_audit"] = await engine.collective_rpc(
                        "collect_native_copy_audit"
                    )
                except Exception as error:
                    report["native_copy_audit_collection_error"] = (
                        f"{type(error).__name__}: {error}"
                    )
                save()
            if paired_loads and barrier_path.exists():
                report["load_barrier"] = json.loads(barrier_path.read_text())
                save()
            if report.get("resident_ownership_requested"):
                if ownership_path.exists():
                    report["resident_ownership"] = json.loads(
                        ownership_path.read_text()
                    )
                else:
                    report["resident_ownership_collection_error"] = "Missing audit file"
                save()
            if report.get("prefix_shape_install"):
                report["prefix_shapes"] = await engine.collective_rpc(
                    "collect_prefix_shape_trace"
                )
                # Retain raw evidence even if coverage indexing rejects the trace.
                save()
                phases = [f"reference-{i}" for i in range(args.reference_runs)] + [
                    "hot",
                    "cold",
                ]
                if getattr(args, "prefix_concurrency", 1) > 1:
                    phases += [
                        "batch-reference-a",
                        "batch-reference-b",
                        "batch-hot",
                        "batch-cold",
                    ]
                report["prefix_shapes_exercised"] = all(
                    all(row["phases"].get(phase) for phase in phases)
                    for row in report["prefix_shapes"]
                ) and bool(report["prefix_shapes"])
                pairs = [
                    ("reference-0", f"reference-{i}", 0)
                    for i in range(1, args.reference_runs)
                ]
                pairs += [
                    ("reference-0", name, report[f"{name}_cached_tokens"])
                    for name in ("hot", "cold")
                    if f"{name}_cached_tokens" in report
                ]
                concurrent = report.get("concurrent", {})
                for name, phase in (
                    ("reference_b", "batch-reference-b"),
                    ("hot", "batch-hot"),
                    ("cold", "batch-cold"),
                ):
                    rows = concurrent.get(name, {}).get("requests", [])
                    if rows:
                        pairs.append(
                            (
                                "batch-reference-a",
                                phase,
                                min(row["cached_tokens"] for row in rows),
                            )
                        )
                report["prefix_shape_comparisons"] = [
                    {
                        f"{left}/{right}": compare_prefix_shapes(
                            row["phases"].get(left, []),
                            row["phases"].get(right, []),
                            start,
                        )
                        for left, right, start in pairs
                    }
                    for row in report["prefix_shapes"]
                ]
                if trace_pages:
                    expected = {
                        phase: (
                            {f"{phase}-{i}" for i in range(args.prefix_concurrency)}
                            if phase.startswith("batch-")
                            else {phase}
                        )
                        for phase in phases
                    }
                    report["prefix_pages_exercised"] = False
                    if continuations:
                        report["continuation_pages_exercised"] = False
                    save()
                    report["prefix_page_request_mapping"] = [
                        {
                            phase: {
                                name: request["request_id"]
                                for name, request in index_prefix_page_requests(
                                    row["pages"].get(phase, []), names
                                ).items()
                            }
                            for phase, names in expected.items()
                        }
                        for row in report["prefix_shapes"]
                    ]
                    report["prefix_pages_exercised"] = bool(report["prefix_shapes"])
                    if continuations:
                        report["continuation_pages_exercised"] = False
                        report["continuation_page_request_mapping"] = [
                            {
                                phase: {
                                    str(boundary): {
                                        name: request["request_id"]
                                        for name, request in index_prefix_page_requests(
                                            rank["pages"].get(phase, []),
                                            {
                                                f"{phase}-{i}"
                                                for i in range(args.continuation_concurrency)
                                            },
                                            boundary=boundary,
                                        ).items()
                                    }
                                    for boundary in boundaries
                                }
                                for phase, boundaries in report[
                                    "continuation_page_boundaries"
                                ].items()
                            }
                            for rank in report["prefix_shapes"]
                        ]
                        report["continuation_pages_exercised"] = (
                            len(report["prefix_shapes"]) == 4
                        )
                save()
                if (
                    report.get("complete")
                    and trace_pages
                    and not report["prefix_pages_exercised"]
                ):
                    report["screen_passed"] = False
                    save()
                    raise AssertionError("Prefix-page trace missed required requests")
                if (
                    report.get("complete")
                    and continuations
                    and trace_pages
                    and not report.get("continuation_pages_exercised")
                ):
                    report["screen_passed"] = False
                    save()
                    raise AssertionError("Continuation page trace missed TP4 coverage")
                if report.get("complete") and not report["prefix_shapes_exercised"]:
                    report["screen_passed"] = False
                    save()
                    raise AssertionError("Prefix shape trace missed required phases")
            if report.get("prefill_trace_install") and "prefill_trace" not in report:
                report["prefill_trace"] = await engine.collective_rpc(
                    "collect_session_prefill_trace"
                )
                save()
            if numerical_control and report.get("mlp_numerical_install"):
                report["mlp_numerical_remove"] = await engine.collective_rpc(
                    "remove_mlp_numerical_control"
                )
                report["mlp_numerical_exercised"] = (
                    len(report["mlp_numerical_remove"]) == 4
                    and all(row["exercised"] and row["hooks_restored"]
                            for row in report["mlp_numerical_remove"])
                )
                if report.get("complete") and not report["mlp_numerical_exercised"]:
                    report["screen_passed"] = False
                    save()
                    raise AssertionError("MLP numerical control missed TP4 coverage")
                save()
            if stable_layout and report.get("stable_layout_install"):
                report["stable_layout_remove"] = await engine.collective_rpc(
                    "remove_stable_expert_layout"
                )
                report["stable_layout_exercised"] = all(
                    row["calls"] > 0 for row in report["stable_layout_remove"]
                )
                if report.get("complete") and not report["stable_layout_exercised"]:
                    report["screen_passed"] = False
                    save()
                    raise AssertionError("Stable layout diagnostic was not exercised")
                save()
        finally:
            try:
                if stable_qsa and report.get("stable_qsa_install"):
                    report["stable_qsa_remove"] = await engine.collective_rpc(
                        "remove_stable_qsa_selection"
                    )
                    report["stable_qsa_exercised"] = all(
                        row["calls"] > 0 for row in report["stable_qsa_remove"]
                    )
                    if report.get("complete") and not report["stable_qsa_exercised"]:
                        report["screen_passed"] = False
                        save()
                        raise AssertionError("Stable QSA diagnostic was not exercised")
                    save()
            finally:
                try:
                    if report.get("complete") and not failure_mode:
                        await record_memory("completed")
                finally:
                    engine.shutdown()
