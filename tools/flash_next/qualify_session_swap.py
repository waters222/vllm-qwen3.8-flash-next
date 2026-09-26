"""Offline greedy C1 swap screen. Requires an explicitly provisioned GPU trial.

This tool does not launch containers, change serving, or download weights. It
loads the supplied local engine configuration and writes a new evidence report.
A pass is limited to this prompt/topology/depth, not full session qualification.
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path


def compare_tokens(reference, candidate):
    for index, (expected, actual) in enumerate(zip(reference, candidate)):
        if expected != actual:
            return {
                "equal": False,
                "index": index,
                "expected": expected,
                "actual": actual,
            }
    if len(reference) != len(candidate):
        return {
            "equal": False,
            "index": min(len(reference), len(candidate)),
            "reference_length": len(reference),
            "candidate_length": len(candidate),
        }
    return {"equal": True, "tokens": len(reference)}


def block_sets(description):
    null = description["null_block_id"]
    return [set(ids) - {null} for ids in description["block_ids"]]


def screen_verdict(report):
    return bool(
        report["baseline_repeatable"]
        and report["exact_token_match"]
        and report["full_reuse_proven"]
    )


class Generation:
    def __init__(self, engine, prompt, params, name):
        self.engine = engine
        self.prompt = prompt
        self.params = params
        self.name = name
        self.tokens = []
        self.logprobs = []
        self.finished = False
        self.changed = asyncio.Condition()
        self.task = asyncio.create_task(self.collect())

    async def collect(self):
        try:
            async for output in self.engine.generate(
                prompt=self.prompt, sampling_params=self.params, request_id=self.name
            ):
                completion = output.outputs[0]
                async with self.changed:
                    self.tokens = list(completion.token_ids)
                    self.logprobs = [
                        {str(token): value.logprob for token, value in row.items()}
                        for row in completion.logprobs or []
                    ]
                    self.finished = output.finished
                    self.changed.notify_all()
        finally:
            async with self.changed:
                self.finished = True
                self.changed.notify_all()

    async def wait_tokens(self, count):
        async with self.changed:
            await self.changed.wait_for(
                lambda: len(self.tokens) >= count or self.finished
            )
        if self.finished:
            await self.task
            raise RuntimeError("generation finished before the requested swap boundary")

    def internal_id(self):
        ids = self.engine.output_processor.external_req_ids.get(self.name, [])
        if len(ids) != 1:
            raise RuntimeError(
                "expected exactly one retained internal request identity"
            )
        return ids[0]


async def screen(args, report, save):
    if args.native_prefix:
        from prefix_cache_screen import prefix_screen

        return await prefix_screen(args, report, save)
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    if int(os.environ.get("VLLM_FLASH_SESSION_SWAP_BYTES", "0")) <= 0:
        raise ValueError(
            "the isolated trial must explicitly enable bounded session RAM"
        )
    config = json.loads(args.engine_args.read_text())
    if args.trace_prefill:
        if config.get("worker_extension_cls"):
            raise ValueError("refuse to replace an existing worker extension")
        config["worker_extension_cls"] = (
            "session_prefill_trace.PrefillTraceWorkerExtension"
        )
    if args.no_mtp:
        config["speculative_config"] = None
    if not Path(config["model"]).is_dir():
        raise ValueError("qualification requires a local read-only model mount")
    if config.get("enable_prefix_caching") is not False:
        raise ValueError("qualification requires explicit enable_prefix_caching=false")
    config["seed"] = args.seed
    config["disable_log_stats"] = True
    report["engine"] = {
        name: config.get(name)
        for name in (
            "model",
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "speculative_config",
            "max_model_len",
            "kv_cache_memory_bytes",
            "seed",
            "worker_extension_cls",
        )
    }
    report["settings"] = dict(
        storage_backend=os.environ.get("VLLM_FLASH_SESSION_SWAP_BACKEND", "copy"),
        temperature=0,
        seed=args.seed,
        tokens=args.tokens,
        cycles=args.cycles,
        reuse_rounds=args.reuse_rounds,
    )
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**config))
    report["engine"]["effective_mamba_cache_mode"] = (
        engine.vllm_config.cache_config.mamba_cache_mode
    )
    tasks = []
    try:
        text = (
            args.prompt.read_text()
            if args.prompt
            else (
                "Explain why prime numbers are infinite. Give a detailed proof and "
                "then demonstrate it with several small numerical examples."
            )
        )
        prompt_ids = engine.get_tokenizer().encode(text)
        prompt = {"prompt_token_ids": prompt_ids}
        params = SamplingParams(
            temperature=0,
            seed=args.seed,
            max_tokens=args.tokens,
            ignore_eos=True,
            logprobs=5,
        )
        report["prompt_token_ids"] = prompt_ids
        save()
        if args.streaming:
            from session_stream_screen import run_stream_screen

            await run_stream_screen(engine, args, prompt_ids, params, report, save)
            return
        reference = Generation(engine, prompt, params, "swap-reference")
        tasks.append(reference.task)
        await reference.task
        if len(reference.tokens) != args.tokens:
            raise RuntimeError("reference did not generate the requested token count")
        report["reference"] = dict(
            token_ids=reference.tokens, top_logprobs=reference.logprobs
        )
        save()
        repeat = Generation(engine, prompt, params, "swap-reference-repeat")
        tasks.append(repeat.task)
        await repeat.task
        report["reference_repeat"] = dict(
            token_ids=repeat.tokens,
            top_logprobs=repeat.logprobs,
            comparison=compare_tokens(reference.tokens, repeat.tokens),
        )
        report["reference_runs"] = [report["reference"], report["reference_repeat"]]
        report["baseline_repeatable"] = report["reference_repeat"]["comparison"][
            "equal"
        ]
        for index in range(2, args.reference_runs):
            extra = Generation(engine, prompt, params, f"swap-reference-{index}")
            tasks.append(extra.task)
            await extra.task
            row = dict(
                token_ids=extra.tokens,
                top_logprobs=extra.logprobs,
                comparison=compare_tokens(reference.tokens, extra.tokens),
                previous_comparison=compare_tokens(
                    report["reference_runs"][-1]["token_ids"], extra.tokens
                ),
            )
            report["reference_runs"].append(row)
            report["baseline_repeatable"] &= row["comparison"]["equal"]
            save()
        save()
        if not report["baseline_repeatable"] and not args.diagnose:
            raise RuntimeError("uninterrupted greedy reference is not repeatable")
        print(
            f"Reference repeatable={report['baseline_repeatable']}; "
            "starting swap candidate (a failed baseline still fails qualification)",
            flush=True,
        )

        candidate = Generation(engine, prompt, params, "swap-candidate")
        tasks.append(candidate.task)
        client_index = engine.engine_core.client_index

        async def utility(operation, request_id="", generation=None):
            return await engine.engine_core.call_utility_async(
                "flash_session_swap", operation, request_id, client_index, generation
            )

        threshold = args.pause_after
        for cycle in range(args.cycles):
            await candidate.wait_tokens(threshold)
            await engine.pause_generation(mode="keep", clear_cache=False)
            internal = candidate.internal_id()
            before = await utility("describe", internal)
            if before["phase"] != "hot" or candidate.finished:
                raise RuntimeError("candidate is not hot at the drained boundary")
            source = block_sets(before)
            start = time.monotonic()
            cold = await utility("suspend", internal)
            cold_ms = (time.monotonic() - start) * 1000
            cold_description = await utility("describe", internal)
            if (
                cold_description["phase"] != "cold"
                or cold_description["owns_hot_blocks"]
            ):
                raise RuntimeError(
                    "cold request did not relinquish hot block ownership"
                )
            row = dict(
                cycle=cycle,
                boundary=len(candidate.tokens),
                before=before,
                cold=cold_description,
                capture_ms=cold_ms,
                noise=[],
            )
            report["swaps"].append(row)
            save()
            await engine.resume_generation()
            reused = [set() for _ in source]
            for noise_index in range(args.reuse_rounds):
                noise = Generation(
                    engine, prompt, params, f"noise-{cycle}-{noise_index}"
                )
                tasks.append(noise.task)
                await noise.wait_tokens(1)
                await engine.pause_generation(mode="keep", clear_cache=False)
                description = await utility("describe", noise.internal_id())
                if description["phase"] != "hot":
                    raise RuntimeError(
                        "noise allocation was not retained for inspection"
                    )
                for seen, old, current in zip(
                    reused, source, block_sets(description), strict=True
                ):
                    seen.update(old & current)
                row["noise"].append(description)
                await engine.resume_generation()
                await noise.task
                if all(old <= seen for old, seen in zip(source, reused, strict=True)):
                    break
            row["reused_pages_by_group"] = [len(ids) for ids in reused]
            row["source_pages_by_group"] = [len(ids) for ids in source]
            row["all_source_pages_reused_in_same_group"] = all(
                old <= seen for old, seen in zip(source, reused, strict=True)
            )
            await engine.pause_generation(mode="keep", clear_cache=False)
            start = time.monotonic()
            restored = await utility("restore", internal, cold["key"]["generation"])
            row["restore_ms"] = (time.monotonic() - start) * 1000
            if restored.get("cache") != "cold_hit":
                raise RuntimeError("restore did not report a retained RAM hit")
            row["after"] = await utility("describe", internal)
            save()
            threshold = len(candidate.tokens) + args.pause_after
            if cycle + 1 < args.cycles and threshold >= args.tokens - 8:
                raise RuntimeError(
                    "insufficient remaining output for another swap cycle"
                )
            await engine.resume_generation()
            print(f"Swap cycle {cycle + 1}: restored; reuse proof recorded", flush=True)
        await candidate.task
        report["candidate"] = dict(
            token_ids=candidate.tokens, top_logprobs=candidate.logprobs
        )
        report["comparison"] = compare_tokens(reference.tokens, candidate.tokens)
        report["stats"] = await utility("stats")
        report["exact_token_match"] = report["comparison"]["equal"]
        report["full_reuse_proven"] = all(
            row["all_source_pages_reused_in_same_group"] for row in report["swaps"]
        )
        report["screen_passed"] = screen_verdict(report)
        report["complete"] = True
        save()
        if not report["screen_passed"]:
            raise AssertionError(
                "swap screen failed baseline repeatability, token equality or reuse"
            )
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        engine.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-args", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", type=Path)
    parser.add_argument("--no-mtp", action="store_true")
    parser.add_argument("--native-prefix", action="store_true")
    parser.add_argument("--tp4", action="store_true")
    parser.add_argument("--serving-decode", action="store_true")
    parser.add_argument("--prefix-concurrency", type=int, choices=(1, 2), default=1)
    parser.add_argument("--prefix-prefill-budget", type=int, choices=(1024, 2048))
    parser.add_argument("--prefix-gpu-cache-mib", type=int, choices=(512, 768, 896, 1024))
    parser.add_argument("--balanced-prefix-prefill", action="store_true")
    parser.add_argument("--paired-cold-loads", action="store_true")
    parser.add_argument("--continuation-load-gate", action="store_true")
    parser.add_argument("--idle-expiry", action="store_true")
    parser.add_argument("--pending-expiry", action="store_true")
    parser.add_argument("--continuations", action="store_true")
    parser.add_argument(
        "--continuation-concurrency", type=int, choices=(1, 2, 4), default=1
    )
    parser.add_argument("--active-cancel", action="store_true")
    parser.add_argument(
        "--cancellation-concurrency", type=int, choices=(2, 4), default=2
    )
    parser.add_argument("--audit-active-cancel", action="store_true")
    parser.add_argument("--expire-active-cache", action="store_true")
    parser.add_argument("--performance-repeats", type=int, choices=(0, 3, 5), default=0)
    parser.add_argument("--performance-iteration-details", action="store_true")
    parser.add_argument("--profile-prefix-prefill", action="store_true")
    parser.add_argument("--prefill-staging-ab", action="store_true")
    parser.add_argument("--prefill-staging-concurrency", type=int, choices=(2, 4), default=2)
    parser.add_argument("--native-load-failure", choices=("inject", "recovery"))
    parser.add_argument(
        "--continuation-context", type=int, choices=(8192, 32768), default=8192
    )
    parser.add_argument("--untraced-balanced-prefix", action="store_true")
    parser.add_argument("--divergent-prefixes", action="store_true")
    parser.add_argument("--stable-expert-layout", action="store_true")
    parser.add_argument("--mlp-numerical-control", action="store_true")
    parser.add_argument("--sustained-decode", action="store_true")
    parser.add_argument("--sustained-real-idle-expiry", action="store_true")
    parser.add_argument("--sustained-uniform-outputs", action="store_true")
    parser.add_argument("--sustained-observe-lookups", action="store_true")
    parser.add_argument("--sustained-idle-ttl", type=int, choices=(300, 3600), default=3600)
    parser.add_argument("--sustained-output-tokens", type=int, choices=(1024, 4096),
                        default=1024)
    parser.add_argument("--stable-qsa-selection", action="store_true")
    parser.add_argument("--stable-qsa-ties", action="store_true")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--trace-prefill", action="store_true")
    parser.add_argument("--trace-prefix-prefill", action="store_true")
    parser.add_argument("--trace-prompt-chunks", action="store_true")
    parser.add_argument("--trace-prefix-shapes", action="store_true")
    parser.add_argument("--trace-prefix-pages", action="store_true")
    parser.add_argument("--private-cold-buffers", action="store_true")
    parser.add_argument("--audit-native-copies", action="store_true")
    parser.add_argument("--reference-runs", type=int, default=2)
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Continue transactions after baseline failure; never waive that failure",
    )
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--pause-after", type=int, default=16)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--reuse-rounds", type=int, default=16)
    parser.add_argument("--seed", type=int, default=173)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    from prefix_cache_benchmark import validate_performance
    from prefix_cache_screen import (
        validate_balanced_observation,
        validate_cancellation_observation,
        validate_continuation_observation,
        validate_serving_decode,
    )
    from session_transfer_audit import validate_native_load_failure

    try:
        validate_native_load_failure(args)
        validate_performance(args)
        validate_balanced_observation(args)
        validate_continuation_observation(args)
        validate_cancellation_observation(args)
        validate_serving_decode(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.paired_cold_loads and not args.balanced_prefix_prefill:
        parser.error("paired cold loads require balanced prefix prefill")
    if (args.idle_expiry or args.pending_expiry) and not args.paired_cold_loads:
        parser.error("expiry diagnostics require the isolated paired-load scheduler")
    if args.trace_prefix_pages and not args.trace_prefix_shapes:
        parser.error("prefix page tracing requires the phase/shape trace")
    if (args.tp4 or args.prefix_concurrency > 1) and not args.native_prefix:
        parser.error("TP4/concurrency screening requires native-prefix mode")
    if args.divergent_prefixes and args.prefix_concurrency < 2:
        parser.error("divergent prefixes require concurrent native-prefix screening")
    if args.prefix_prefill_budget is not None and not args.native_prefix:
        parser.error("prefill budget diagnostic requires native-prefix mode")
    from prefix_cache_screen import validate_gpu_cache_budget

    validate_gpu_cache_budget(args)
    if args.private_cold_buffers and not args.native_prefix:
        parser.error("private cold buffers require native-prefix screening")
    if args.stable_qsa_selection and not args.native_prefix:
        parser.error("stable QSA diagnostic requires native-prefix screening")
    if args.stable_qsa_ties and not args.stable_qsa_selection:
        parser.error("stable QSA ties require the selection diagnostic")
    if (
        args.trace_prefix_prefill
        or args.trace_prompt_chunks
        or args.trace_prefix_shapes
    ) and not args.native_prefix:
        parser.error("prefix prefill tracing requires ordinary native-prefix screening")
    if args.trace_prefix_prefill and args.trace_prompt_chunks:
        parser.error("select only one prefix tracing mode")
    if args.native_prefix and (args.streaming or args.trace_prefill):
        parser.error("ordinary prefix screening cannot use streaming prototype flags")
    if args.trace_prefill and not args.streaming:
        parser.error("prefill tracing requires the streaming diagnostic")
    if args.reference_runs < 2:
        parser.error("at least two reference runs are required")
    if min(args.cycles, args.pause_after, args.reuse_rounds, args.tokens) < 1:
        parser.error("token/cycle/reuse limits must be positive")
    if args.pause_after * args.cycles + 16 >= args.tokens:
        parser.error("output budget is too small for the requested swap boundaries")
    with args.output.open("x"):
        pass
    report = {
        "complete": False,
        "screen_passed": False,
        "swaps": [],
        "scope": (
            "C1 greedy token and same-group dirty-block-reuse screen; "
            "not full qualification"
        ),
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        asyncio.run(asyncio.wait_for(screen(args, report, save), timeout=args.timeout))
    except BaseException as exc:
        report["error_type"] = type(exc).__name__
        save()
        raise


if __name__ == "__main__":
    main()
