"""Bounded C4 long-output workload; numerical failures remain separate."""

import hashlib
import json
import asyncio
import time


def validate(args):
    ttl = getattr(args, "sustained_idle_ttl", 3600)
    if ttl not in (300, 3600) or (ttl != 3600 and not getattr(
            args, "sustained_real_idle_expiry", False)):
        raise ValueError("Short test TTL requires real idle expiry")
    if not getattr(args, "sustained_decode", False):
        if getattr(args, "sustained_observe_lookups", False):
            raise ValueError("Lookup observer requires sustained decode")
        if getattr(args, "sustained_uniform_outputs", False):
            raise ValueError("Uniform outputs require sustained decode")
        if getattr(args, "sustained_real_idle_expiry", False):
            raise ValueError("Real idle expiry requires sustained decode")
        return
    required = ("native_prefix", "tp4", "serving_decode", "balanced_prefix_prefill",
                "untraced_balanced_prefix", "private_cold_buffers")
    excluded = ("continuations", "active_cancel", "paired_cold_loads",
                "audit_native_copies", "prefill_staging_ab", "stable_expert_layout",
                "mlp_numerical_control", "idle_expiry", "pending_expiry",
                "native_load_failure", "profile_prefix_prefill",
                "performance_iteration_details", "no_mtp")
    if (not all(getattr(args, k, False) for k in required)
            or any(getattr(args, k, False) for k in excluded)
            or getattr(args, "continuation_context", 8192) != 32768
            or getattr(args, "sustained_output_tokens", 1024) not in (1024, 4096)):
        raise ValueError("Sustained decode requires untraced native TP4/MTP C4 "
                         "graph mode, 32k context and no competing fixture")


def workload(seeds, max_output=1024, context=32768, uniform_outputs=False):
    if context != 32768 or max_output not in (1024, 4096):
        raise ValueError("Require 32k context and 1024 or 4096 output budget")
    if len(seeds) != 4:
        raise ValueError("Require four private seed prompts")
    lengths = [context - max_output - 4 - 944 * (3 - i) for i in range(4)]
    if any(len(seed) < size for seed, size in zip(seeds, lengths)):
        raise ValueError("Seed too short for sustained workload")
    prompts = [list(seed[:size]) for seed, size in zip(seeds, lengths)]
    shortest = min(lengths)
    if len({tuple(p[:shortest]) for p in prompts}) != 4 or any(
        p[:944] != prompts[0][:944] for p in prompts
    ):
        raise ValueError("Require common prefix and distinct branch prompts")
    outputs = ([max_output] * 4 if uniform_outputs else
               [max_output * (i + 1) // 4 for i in range(4)])
    return prompts, outputs


async def run(generate, batch, seeds, prompt_factory, max_output=1024,
              pressure_rounds=32, real_idle_expiry=False, idle_progress=None,
              idle_ttl=3600, uniform_outputs=False):
    """Callbacks record real engine outputs; batch accepts per-branch budgets."""
    from prefix_cache_screen import complete_request_output

    if type(pressure_rounds) is not int or not 1 <= pressure_rounds <= 32:
        raise ValueError("Pressure rounds must be bounded")
    if idle_ttl not in (300, 3600):
        raise ValueError("Require 300 or 3600 second test TTL")
    wait_seconds = idle_ttl + 5
    prompts, outputs = workload(seeds, max_output, uniform_outputs=uniform_outputs)
    phases = []

    async def phase(name, skip=False):
        row = await batch(
            name, prompts, output_tokens=outputs, cache_salt="sustained-c4",
            skip_cache=skip,
        )
        phases.append(row)
        return row

    await phase("sustained-seed")
    for cycle in range(2):
        await phase(f"sustained-hot-{cycle}")
        for pressure in range(pressure_rounds):
            await generate(
                f"sustained-pressure-{cycle}-{pressure}", tokens=8,
                prompt_ids=prompt_factory(8000 + cycle * 100 + pressure),
                cache_salt=f"sustained-pressure-{cycle}",
            )
        await phase(f"sustained-cold-{cycle}")
    idle = None
    if real_idle_expiry:
        started = time.monotonic()
        idle = dict(started_monotonic=started, required_seconds=wait_seconds,
                    elapsed_seconds=0, clocks_modified=False)
        while idle["elapsed_seconds"] < wait_seconds:
            if idle_progress is not None:
                idle_progress(dict(idle))
            await asyncio.sleep(min(30, wait_seconds - idle["elapsed_seconds"]))
            idle["elapsed_seconds"] = time.monotonic() - started
        idle["finished_monotonic"] = time.monotonic()
        if idle_progress is not None:
            idle_progress(dict(idle))
        await phase("sustained-expired")
        await phase("sustained-renewed")
    await phase("sustained-reference-a", True)
    await phase("sustained-reference-b", True)
    checks = dict(
        complete_outputs=all(
            len(p["requests"]) == 4 and all(
                complete_request_output(r, n)["passed"]
                for r, n in zip(p["requests"], outputs)
            ) for p in phases
        ),
        concurrent_execution=all(p["max_running"] == 4 for p in phases),
        prompt_identity=all(
            len(p["requests"]) == 4 and all(
                r["prompt_tokens"] == len(ids)
                and r["prompt_sha256"] == hashlib.sha256(
                    json.dumps(ids).encode()).hexdigest()
                for r, ids in zip(p["requests"], prompts)
            ) for p in phases
        ),
        fresh_seed=all(r["cached_tokens"] == 0 for r in phases[0]["requests"]),
        reuse=all(r["cached_tokens"] > 0 for p in phases[1:5]
                  for r in p["requests"]),
        cold_loads=all(p["load_bytes"] > 0 for p in (phases[2], phases[4])),
        no_other_loads=all(p["load_bytes"] == 0
                           for p in (phases[0], phases[1], phases[3], *phases[5:])),
        uncached_references=all(r["cached_tokens"] == 0 for p in phases[-2:]
                                for r in p["requests"]),
    )
    if real_idle_expiry:
        checks["real_idle_elapsed"] = idle["elapsed_seconds"] >= wait_seconds
        checks["expired_miss"] = all(r["cached_tokens"] == 0
                                      for r in phases[5]["requests"])
        checks["renewed_reuse"] = all(r["cached_tokens"] > 0
                                       for r in phases[6]["requests"])
    comparisons = [
        dict(phase=p["name"], branch=i,
             tokens_equal=r["tokens"] == phases[-2]["requests"][i]["tokens"],
             logprobs_equal=r["logprobs"]
             == phases[-2]["requests"][i]["logprobs"])
        for p in phases[1:-2] + phases[-1:]
        for i, r in enumerate(p["requests"])
    ]
    return dict(
        context=32768, output_tokens=outputs, pressure_rounds=pressure_rounds,
        uniform_outputs=uniform_outputs,
        real_idle_expiry=real_idle_expiry, idle_wait=idle,
        phases=phases, checks=checks, lifecycle_passed=all(checks.values()),
        comparisons=comparisons,
        numerical_passed=all(r["tokens_equal"] and r["logprobs_equal"]
                             for r in comparisons),
        scope="C4 sustained generation and reuse lifecycle; not byte integrity, "
              "universal race freedom or model quality.",
    )
