"""Actual-weight TP4 GDN projection row sweep; diagnostic only, no integration."""

import statistics
import os

import torch
import triton
import triton.language as tl

ROWS = (1, 2, 4, 6, 8, 10, 16, 20, 24, 30, 32)


@triton.jit
def projection(X, W, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    ms = tl.arange(0, BM)
    ns = tl.program_id(0) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + ks
        x = tl.load(X + ms[:, None] * K + kk[None, :],
                    (ms[:, None] < M) & (kk[None, :] < K), 0)
        w = tl.load(W + ns[None, :] * K + kk[:, None],
                    (ns[None, :] < N) & (kk[:, None] < K), 0)
        acc = tl.dot(x, w, acc)
    tl.store(Y + ms[:, None] * N + ns[None, :], acc,
             (ms[:, None] < M) & (ns[None, :] < N))


def measure(call, repeats=5, iterations=100):
    for _ in range(10):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iterations):
            call()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
    return dict(samples_us=samples, median_us=statistics.median(samples),
                timing_mode="CUDA graph, 100 repeated projections per replay")


def measure_pair(native, installed, repeats=8, iterations=100):
    """Share capture allocation placement and alternate replay order."""
    pool = torch.cuda.graph_pool_handle()
    graphs = {}
    for name, call in (("native", native), ("installed", installed)):
        for _ in range(10):
            call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            for _ in range(iterations):
                call()
        graphs[name] = graph
    samples = {name: [] for name in graphs}
    for repeat in range(repeats):
        order = ("native", "installed") if repeat % 2 == 0 else ("installed", "native")
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / iterations)
    return {name: dict(samples_us=values, median_us=statistics.median(values),
                       timing_mode="Shared-pool CUDA graphs, alternating order, "
                                   "100 projections per replay")
            for name, values in samples.items()}


def run(context, parameters):
    if parameters:
        raise ValueError("Fixed bounded row sweep takes no parameters")
    if os.environ.get("VLLM_FLASH_GDN_TP4_PROJECTION") == "1":
        result = run_integrated(context)
        if os.environ.get("FLASH_REDUCED_TP4_CACHE_CHECK") == "1":
            import experiment_flash_layer0_mtp_cold as lifecycle

            context["validate_native"]()
            cache = lifecycle.run(context, {})
            return dict(projections=result, cache=cache,
                        screen_passed=result["screen_passed"] and cache["screen_passed"],
                        scope="Installed projection checks plus reduced C2 target/MTP "
                              "cache lifecycle. Not C4/C6 or full-model qualification.")
        return result
    layer = context["layer"].linear_attn
    weights = [("gdn_input", layer.in_proj_qkvz.weight, (4096, 2560)),
               ("gdn_ba", layer.in_proj_ba.weight, (24, 2560))]
    records = []
    generator = torch.Generator(device=context["device"]).manual_seed(20260923)
    for name, weight, shape in weights:
        if (tuple(weight.shape) != shape or weight.dtype != torch.bfloat16
                or not weight.is_contiguous() or not weight.is_cuda
                or torch.cuda.get_device_capability(weight.device) != (8, 6)):
            raise ValueError("Unexpected actual TP4 weight layout: " + name)
        n, k = shape
        for rows in ROWS:
            x = torch.randn((rows, k), device=weight.device, dtype=weight.dtype,
                            generator=generator)
            original = x.clone()
            native = torch.empty((rows, n), device=x.device, dtype=x.dtype)
            torch.mm(x, weight.T, out=native)
            record = dict(operator=name, rows=rows, weight_shape=list(shape),
                          candidates=[], native=None)
            functions = [("native", lambda: torch.mm(x, weight.T, out=native))]
            for bn, bk in ((32, 128), (64, 64)):
                out = torch.empty_like(native)

                def candidate(out=out, bn=bn, bk=bk):
                    projection[(triton.cdiv(n, bn),)](
                        x, weight, out, rows, n, k, max(16, triton.next_power_of_2(rows)),
                        bn, bk, num_warps=4, num_stages=3)

                candidate()
                torch.cuda.synchronize()
                delta = (out.float() - native.float()).abs()
                finite = bool(torch.isfinite(out).all())
                record["candidates"].append(dict(
                    name=f"bn{bn}_bk{bk}", finite=finite,
                    exact=torch.equal(out, native),
                    max_abs=float(delta.max()),
                    rms_error=float(delta.square().mean().sqrt()),
                    tolerance_passed=finite and torch.allclose(out, native, rtol=0.02, atol=0.02),
                ))
                functions.append((f"bn{bn}_bk{bk}", candidate))
            # Alternate variant order across row sizes to reduce fixed-order bias.
            timings = {}
            for label, function in (functions if ROWS.index(rows) % 2 == 0 else functions[::-1]):
                timings[label] = measure(function)
            record["native"] = timings["native"]
            for candidate in record["candidates"]:
                timing = timings[candidate["name"]]
                candidate.update(timing)
                candidate["native_over_candidate"] = timings["native"]["median_us"] / timing["median_us"]
                candidate["rows_per_second"] = rows * 1e6 / timing["median_us"]
            record["input_unchanged"] = torch.equal(x, original)
            records.append(record)
    return dict(
        rank=context["rank"], records=records,
        screen_passed=all(r["input_unchanged"] and all(c["tolerance_passed"] for c in r["candidates"])
                          for r in records),
        scope="Actual TP4 weights, controlled seeded BF16 inputs; standalone projection CUDA-event timings. "
              "rtol=0.02/atol=0.02 operator screen, not bitwise equality, model quality, "
              "full verification throughput, collective latency or long-context KV performance. "
              "No engine hooks or serving changes.",
    )


def run_integrated(context):
    from vllm.models.qwen4_exp.nvidia import flash_gdn_input_tensor_sm86 as hook
    from vllm.models.qwen4_exp.nvidia.low_latency_gemm import (
        enable_qwen4_exp_low_latency_gemm,
    )

    # The reduced harness's structural root has an extra namespace. Exercise
    # the real model installer using its supported layers namespace.
    holder = torch.nn.Module()
    holder.layers = torch.nn.ModuleDict({"0": context["layer"]})
    enable_qwen4_exp_low_latency_gemm(holder, torch.bfloat16)
    records = []
    generator = torch.Generator(device=context["device"]).manual_seed(20260923)
    attn = context["layer"].linear_attn
    for name, layer in (("gdn_input", attn.in_proj_qkvz),
                        ("gdn_ba", attn.in_proj_ba)):
        method = layer.quant_method
        assert type(method) is hook.GDNTp4ProjectionMethod, name
        # Repeated installation must preserve the original native fallback.
        enable_qwen4_exp_low_latency_gemm(holder, torch.bfloat16)
        assert layer.quant_method is method
        x = torch.randn((4, 2560), device=context["device"],
                        dtype=torch.bfloat16, generator=generator)
        bias = torch.randn((layer.weight.shape[0],), device=x.device,
                           dtype=x.dtype, generator=generator)
        assert torch.equal(method.apply(layer, x, bias),
                           method.original.apply(layer, x, bias))
        enabled = hook._TP4_ENABLED
        invariant = hook.envs.VLLM_BATCH_INVARIANT
        try:
            for disabled, batch_invariant in ((True, False), (False, True)):
                hook._TP4_ENABLED = not disabled
                hook.envs.VLLM_BATCH_INVARIANT = batch_invariant
                assert torch.equal(method.apply(layer, x),
                                   method.original.apply(layer, x))
                try:
                    torch.ops.vllm.flash_gdn_tp4_projection(x, layer.weight)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("Disabled direct custom op must reject")
        finally:
            hook._TP4_ENABLED = enabled
            hook.envs.VLLM_BATCH_INVARIANT = invariant
        compiled = torch.compile(lambda x: method.apply(layer, x),
                                 backend="eager", fullgraph=True)
        assert torch.equal(compiled(x), method.apply(layer, x))
        for rows in (*ROWS, 3, 31, 33):
            for strided in (False, True):
                x = torch.randn((rows, 5120 if strided else 2560),
                                device=context["device"], dtype=torch.bfloat16,
                                generator=generator)
                if strided:
                    x = x[:, ::2]
                before = x.clone()
                expected = method.original.apply(layer, x)
                original_apply = method.original.apply
                calls = []

                def counted(*args, **kwargs):
                    calls.append(1)
                    return original_apply(*args, **kwargs)

                method.original.apply = counted
                try:
                    actual = method.apply(layer, x)
                finally:
                    method.original.apply = original_apply
                optimized = hook.tp4_eligible(x, layer.weight)
                fallback_correct = len(calls) == (0 if optimized else 1)
                exact = torch.equal(actual, expected)
                passed = (torch.isfinite(actual).all().item()
                          and torch.allclose(actual, expected, rtol=0.02, atol=0.02)
                          and fallback_correct and torch.equal(x, before)
                          and (optimized or exact))
                record = dict(operator=name, rows=rows, strided=strided,
                              optimized=optimized, fallback_correct=fallback_correct,
                              exact=exact, passed=passed,
                              max_abs=float((actual.float()-expected.float()).abs().max()))
                if not strided and rows in ROWS:
                    record.update(measure_pair(lambda: original_apply(layer, x),
                                               lambda: method.apply(layer, x)))
                records.append(record)
    return dict(rank=context["rank"], records=records,
                screen_passed=all(r["passed"] for r in records),
                scope="Actual layer weights; model installer and native fallback, "
                      "seeded BF16 inputs, CUDA graph replay; rtol/atol 0.02. "
                      "Not full layer output, cache or model quality qualification.")
