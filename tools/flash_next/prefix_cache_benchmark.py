"""Warmed native-prefix latency/throughput trials; no cache implementation."""

import hashlib
import math
import re
import statistics


class NativeIterationCapture:
    """Bound frontend copies of vLLM iteration stats; no worker hooks or sync."""

    def __init__(self):
        self.phase = None
        self.records = []
        self.error = None

    def start(self, phase):
        if self.phase is not None or not phase:
            raise ValueError("Iteration capture requires one nonempty phase")
        self.phase, self.records, self.error = phase, [], None

    def record(self, details):
        if self.phase is None or details is None or self.error:
            return
        from dataclasses import asdict

        row = asdict(details)
        counts = (
            "iteration_index",
            "num_ctx_requests",
            "num_ctx_tokens",
            "num_generation_requests",
            "num_generation_tokens",
            "num_encoder_inputs",
            "num_encoder_output_tokens",
        )
        if (
            len(self.records) >= 1024
            or not all(type(row.get(k)) is int and row[k] >= 0 for k in counts)
            or not isinstance(row.get("is_dummy"), bool)
            or type(row.get("elapsed_ms")) not in (int, float)
            or not math.isfinite(row.get("elapsed_ms", float("nan")))
            or row["elapsed_ms"] < 0
            or (
                self.records
                and row["iteration_index"] <= self.records[-1]["iteration_index"]
            )
        ):
            self.error = "Invalid, duplicate or excess native iteration details"
            return
        self.records.append(row)

    def finish(self):
        result = dict(
            phase=self.phase,
            records=self.records,
            error=self.error,
            complete=bool(self.records) and self.error is None,
            scope="Native engine-core iteration counts and elapsed CPU wall time; "
            "not CUDA kernel time. Frontend phase attribution may include "
            "late stats; verify token totals before attributing latency. "
            "Collection/logging adds overhead; not an uninstrumented baseline.",
        )
        self.phase, self.records, self.error = None, [], None
        return result


def validate_performance(args):
    repeats = getattr(args, "performance_repeats", 0)
    profile = getattr(args, "profile_prefix_prefill", False)
    staging_ab = getattr(args, "prefill_staging_ab", False)
    staging_c = getattr(args, "prefill_staging_concurrency", 2)
    if staging_c not in (2, 4) or (staging_c == 4 and (
        not staging_ab or getattr(args, "prefix_prefill_budget", None) != 2048
    )):
        raise ValueError("C4 staging requires the staging AB fixture and budget2048")
    if staging_ab and profile:
        raise ValueError("Staging AB must remain unprofiled")
    if staging_ab and any(
        getattr(args, flag, False)
        for flag in (
            "paired_cold_loads",
            "trace_prompt_chunks",
            "trace_prefix_pages",
            "stable_expert_layout",
            "stable_qsa_selection",
            "stable_qsa_ties",
            "native_kernels",
            "optimized_canonical",
            "cuda_launch_blocking",
            "streaming",
            "diagnose",
            "audit_active_cancel",
            "expire_active_cache",
        )
    ):
        raise ValueError("Staging AB cannot alter kernels or add diagnostic controls")
    short = profile or staging_ab
    if short and (repeats or not getattr(args, "private_cold_buffers", False)):
        raise ValueError(
            "Short prefill trials need private buffers and no benchmark repeats"
        )
    if getattr(args, "performance_iteration_details", False) and repeats == 0:
        raise ValueError("Iteration details require the warmed performance fixture")
    if repeats == 0 and not short:
        return
    if (
        (not short and repeats not in (3, 5))
        or not all(
            getattr(args, flag, False)
            for flag in (
                "native_prefix",
                "tp4",
                "serving_decode",
                "untraced_balanced_prefix",
                "balanced_prefix_prefill",
                "divergent_prefixes",
            )
        )
        or getattr(args, "prefix_concurrency", 1) != 2
        or any(
            getattr(args, flag, False)
            for flag in (
                "continuations",
                "active_cancel",
                "idle_expiry",
                "pending_expiry",
                "audit_native_copies",
                "no_mtp",
                "trace_prefix_shapes",
                "trace_prefix_prefill",
                "trace_prefill",
                "trace_host_registration",
            )
        )
    ):
        raise ValueError(
            "Performance trials require untraced TP4/MTP4 graph C1/C2, "
            "three or five repeats, and no other lifecycle fixture"
        )


def prefill_profiler_config(directory):
    """Use native bounded profiling without stacks or retained tensor shapes."""
    return dict(
        profiler="torch",
        torch_profiler_dir=str(directory),
        torch_profiler_with_stack=False,
        torch_profiler_record_shapes=False,
        torch_profiler_with_memory=False,
        torch_profiler_use_gzip=True,
        torch_profiler_dump_cuda_time_total=True,
        ignore_frontend=True,
        max_iterations=32,
    )


async def prefill_profile_screen(engine, batch, prompts, directory):
    """Warm native prefixes, then capture only C1/C2 first-token requests."""
    if len(prompts) != 2 or any(len(p) != 7100 for p in prompts):
        raise ValueError("Prefill profile requires the existing two 7100-token prompts")
    directory.mkdir(exist_ok=False)
    phases = {}

    async def run(concurrency, mode):
        name = f"profile-c{concurrency}-{mode}"
        phase = await batch(
            name,
            prompts[:concurrency],
            tokens=1,
            cache_salt=f"profile-c{concurrency}",
        )
        rows = phase["requests"]
        if (
            len(rows) != concurrency
            or phase["max_running"] != concurrency
            or phase["load_bytes"] != 0
            or any(
                row["cached_tokens"] != (0 if mode == "prime" else 5664)
                or row["prompt_tokens"] != 7100
                or row["requested_output_tokens"] != 1
                or len(row["tokens"]) != 1
                or row["finish_reason"] != "length"
                or row["nonfinite_logprobs"]
                or row["skip_cache"]
                for row in rows
            )
        ):
            raise ValueError("Prefill profile did not exercise the requested hot path")
        if mode != "prime" and any(
            row["prompt_sha256"] != reference["prompt_sha256"]
            for row, reference in zip(
                rows, phases[f"profile-c{concurrency}-prime"]["requests"], strict=True
            )
        ):
            raise ValueError("Prefill profile inputs changed")
        phases[name] = phase

    for concurrency in (1, 2):
        await run(concurrency, "prime")
        await run(concurrency, "warm")
    try:
        await engine.start_profile(profile_prefix="prefix-prefill")
        for concurrency in (1, 2):
            await run(concurrency, "capture")
    finally:
        await engine.stop_profile()
    artifacts = []
    for path in sorted(directory.glob("*.pt.trace.json.gz")):
        match = re.search(r"_rank([0-3])\.", path.name)
        if not match or not 0 < path.stat().st_size <= 256 * 1024**2:
            raise ValueError("Unexpected or oversized native profiler artifact")
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        artifacts.append(
            dict(
                name=path.name,
                rank=int(match[1]),
                bytes=path.stat().st_size,
                sha256=digest,
            )
        )
    if sorted(row["rank"] for row in artifacts) != [0, 1, 2, 3]:
        raise ValueError("Native profiler did not produce exactly four rank traces")
    return dict(
        phases=phases,
        artifacts=artifacts,
        native_artifacts_recorded=True,
        scope="Nine one-output-token requests: six unprofiled priming/warmup and "
        "three profiled hot requests. Trace contents still require independent "
        "CUDA-event/phase coverage analysis. Not a throughput or integrity gate.",
    )


async def prefill_staging_ab_screen(engine, batch, prompts):
    """Four alternating pairs per C in one engine; only one output per request."""
    c4 = len(prompts) == 4
    prompt_length, cached_length = (29756, 28320) if c4 else (7100, 5664)
    concurrencies = (4,) if c4 else (1, 2)
    if len(prompts) not in (2, 4) or any(len(p) != prompt_length for p in prompts):
        raise ValueError("Staging AB requires C2/7100 or C4/29756 prompts")
    phases, changes, references = {}, [], {}

    async def change(enabled):
        await engine.pause_generation(mode="wait", clear_cache=False)
        rows = await engine.collective_rpc("set_qsa_prefill_staging", args=(enabled,))
        if (
            len(rows) != 4
            or sorted(r["rank"] for r in rows) != [0, 1, 2, 3]
            or any(
                (r.get("restored") is not True or r.get("modules") != 13)
                if enabled is None
                else (
                    r.get("enabled") is not enabled
                    or (r.get("target"), r.get("draft")) != (12, 1)
                )
                for r in rows
            )
        ):
            raise ValueError(
                "Staging AB did not receive four matching acknowledgements"
            )
        changes.append(rows)
        await engine.resume_generation()

    async def run(c, mode, enabled):
        name = f"staging-ab-c{c}-{mode}-{'on' if enabled else 'off'}"
        phase = await batch(name, prompts[:c], tokens=1, cache_salt=f"staging-ab-c{c}")
        rows = phase["requests"]
        if (
            len(rows) != phase["max_running"]
            or len(rows) != c
            or phase["load_bytes"] != 0
            or any(
                r["prompt_tokens"] != prompt_length
                or r["cached_tokens"] != (0 if mode == "prime" else cached_length)
                or r["requested_output_tokens"] != 1
                or len(r["tokens"]) != 1
                or r["finish_reason"] != "length"
                or r["nonfinite_logprobs"]
                or r["skip_cache"]
                for r in rows
            )
        ):
            raise ValueError("Staging AB changed the requested hot-cache path")
        hashes = [r["prompt_sha256"] for r in rows]
        if mode == "prime":
            references[c] = hashes
        elif hashes != references[c]:
            raise ValueError("Staging AB prompt identities changed")
        phases[name] = dict(enabled=enabled, concurrency=c, mode=mode, **phase)

    try:
        for c in concurrencies:
            await change(False)
            await run(c, "prime", False)
            for enabled in (False, True):
                await change(enabled)
                await run(c, "warm", enabled)
            for repeat in range(4):
                for enabled in (False, True) if repeat % 2 == 0 else (True, False):
                    await change(enabled)
                    await run(c, f"r{repeat}", enabled)
    finally:
        await change(None)
    return dict(
        phases=phases,
        worker_changes=changes,
        restored=True,
        concurrencies=list(concurrencies), prompt_tokens=prompt_length,
        expected_requests=11 * sum(concurrencies),
        scope="Four alternating measured pairs per C, with separate prime/warm requests. "
        "Native pause/synchronization and flag changes occur outside request "
        "timing. No profiler, decode-throughput or cache-integrity claim.",
    )


def phase_measurements(phase):
    """Use a common frontend clock; do not sum overlapping request rates."""
    rows = phase["requests"]
    first, last, post = [], [], 0
    accepted = drafts = steps = 0
    for row in rows:
        timing = row["stream_timing"]
        start = row["started_monotonic"]
        a, b = timing["first_output_seconds"], timing["last_output_seconds"]
        burst = timing["first_output_tokens"]
        spec = row["speculative_decoding"]
        if (
            not all(math.isfinite(value) for value in (start, a, b))
            or not phase["started_monotonic"] <= start
            or not 0 <= a < b
            or not 0 < burst < len(row["tokens"])
            or spec["num_spec_tokens"] != 4
            or spec["num_spec_steps"] <= 0
            or spec["num_draft_tokens"] != 4 * spec["num_spec_steps"]
            or not 0 <= spec["num_accepted_draft_tokens"] <= spec["num_draft_tokens"]
        ):
            raise ValueError("Performance evidence lacks valid stream/MTP measurements")
        first.append(start + a)
        last.append(start + b)
        post += len(row["tokens"]) - burst
        accepted += spec["num_accepted_draft_tokens"]
        drafts += spec["num_draft_tokens"]
        steps += spec["num_spec_steps"]
    if (
        not rows
        or not phase["started_monotonic"] < max(last) <= phase["ended_monotonic"]
    ):
        raise ValueError("Performance phase lacks a complete frontend time interval")
    return dict(
        ttft_seconds=[row["stream_timing"]["first_output_seconds"] for row in rows],
        aggregate_post_first_tokens_per_second=post / (max(last) - min(first)),
        aggregate_e2e_output_tokens_per_second=sum(len(row["tokens"]) for row in rows)
        / (max(last) - phase["started_monotonic"]),
        accepted_draft_tokens=accepted,
        draft_tokens=drafts,
        spec_steps=steps,
        draft_acceptance_rate=accepted / drafts,
        mean_acceptance_length=1 + accepted / steps,
        scope="Frontend stream timing, initial bursts excluded from decode rate; "
        "not isolated prefill/kernel timing or byte-integrity evidence",
    )


async def performance_screen(
    generate,
    batch,
    prompts,
    prompt_factory,
    output_tokens,
    repeats=3,
    pressure_rounds=32,
):
    """One discarded cycle per C, then matched hot/cold/fresh repetitions."""
    from prefix_cache_screen import complete_request_output

    if (
        repeats not in (3, 5)
        or len(prompts) != 2
        or prompts[0] == prompts[1]
        or not 1 <= pressure_rounds <= 32
        or output_tokens != 128
    ):
        raise ValueError(
            "Performance fixture needs two distinct prompts and 128 outputs"
        )
    cycles = []
    for concurrency in (1, 2):
        selected = prompts[:concurrency]
        for repeat in range(repeats + 1):
            salt = f"perf-c{concurrency}-r{repeat}"
            phases = {}

            async def run(mode, *, phases=phases, salt=salt, selected=selected):
                phases[mode] = await batch(
                    f"{salt}-{mode}",
                    selected,
                    cache_salt=salt + ("-fresh" if mode == "fresh" else ""),
                    skip_cache=mode == "fresh",
                )

            if repeat % 2:
                await run("fresh")
            await run("prime")
            await run("hot")
            for index in range(pressure_rounds):
                await generate(
                    f"{salt}-pressure-{index}",
                    tokens=8,
                    prompt_ids=prompt_factory(
                        12000 + concurrency * 1000 + repeat * 100 + index
                    ),
                    cache_salt=salt + "-pressure",
                )
            await run("cold")
            if not repeat % 2:
                await run("fresh")
            checks = dict(
                cache_modes=all(
                    row["skip_cache"] == (mode == "fresh")
                    for mode, phase in phases.items()
                    for row in phase["requests"]
                ),
                complete_finite_outputs=all(
                    complete_request_output(row, output_tokens)["passed"]
                    for phase in phases.values()
                    for row in phase["requests"]
                ),
                actual_concurrency=all(
                    len(phase["requests"]) == concurrency
                    and phase["max_running"] == concurrency
                    for phase in phases.values()
                ),
                actual_tiers=all(
                    phases[mode]["load_bytes"] == 0
                    for mode in ("prime", "hot", "fresh")
                )
                and phases["cold"]["load_bytes"] > 0,
                fresh_misses=all(
                    row["cached_tokens"] == 0
                    for mode in ("prime", "fresh")
                    for row in phases[mode]["requests"]
                ),
                same_inputs_and_hit_lengths=all(
                    len(
                        {
                            (
                                phases[mode]["requests"][i]["prompt_sha256"],
                                phases[mode]["requests"][i]["repetition_penalty"],
                            )
                            for mode in phases
                        }
                    )
                    == 1
                    and phases["hot"]["requests"][i]["cached_tokens"]
                    == phases["cold"]["requests"][i]["cached_tokens"]
                    > 0
                    for i in range(concurrency)
                ),
            )
            if any("native_iterations" in phase for phase in phases.values()):
                checks["native_iteration_coverage"] = all(
                    phase.get("native_iterations", {}).get("complete") is True
                    for phase in phases.values()
                )
            measurements = {
                mode: phase_measurements(phases[mode])
                for mode in ("hot", "cold", "fresh")
            }
            cycles.append(
                dict(
                    concurrency=concurrency,
                    repeat=repeat,
                    warmup=repeat == 0,
                    phases=phases,
                    checks=checks,
                    passed=all(checks.values()),
                    measurements=measurements,
                )
            )
    summaries = []
    for concurrency in (1, 2):
        for mode in ("hot", "cold", "fresh"):
            samples = [
                row["measurements"][mode]
                for row in cycles
                if row["concurrency"] == concurrency and not row["warmup"]
            ]
            timings = [v for row in samples for v in row["ttft_seconds"]]
            accepted = sum(row["accepted_draft_tokens"] for row in samples)
            drafts = sum(row["draft_tokens"] for row in samples)
            steps = sum(row["spec_steps"] for row in samples)
            summaries.append(
                dict(
                    concurrency=concurrency,
                    mode=mode,
                    measured_cycles=len(samples),
                    ttft_seconds=dict(
                        median=statistics.median(timings),
                        minimum=min(timings),
                        maximum=max(timings),
                    ),
                    median_aggregate_decode_tokens_per_second=statistics.median(
                        row["aggregate_post_first_tokens_per_second"] for row in samples
                    ),
                    median_aggregate_e2e_output_tokens_per_second=statistics.median(
                        row["aggregate_e2e_output_tokens_per_second"] for row in samples
                    ),
                    draft_acceptance_rate=accepted / drafts,
                    mean_acceptance_length=1 + accepted / steps,
                )
            )
    return dict(
        scope="Warmed C1/C2 native-prefix benchmark; not numerical/state qualification",
        warmup_cycles_per_concurrency=1,
        measured_cycles_per_concurrency=repeats,
        pressure_rounds=pressure_rounds,
        cycles=cycles,
        summaries=summaries,
        passed=all(row["passed"] for row in cycles),
        caveat="Outputs may differ; report MTP acceptance beside throughput. "
        "Warmup is discarded; medians from few repeats are not confidence bounds.",
    )
