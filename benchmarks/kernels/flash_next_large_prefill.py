"""Reduced real-weight TP4 MLP row-capacity and chunking control."""

import json
from pathlib import Path
import time


def run(context, parameters):
    if parameters:
        raise ValueError("Fixed row-capacity experiment accepts no overrides")
    import torch
    from vllm.forward_context import set_forward_context

    layer = context["layer"]
    captured = []
    states = [value.clone() for value in context["states"]]
    hook = layer.mlp.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone()))
    try:
        context["validate_native"]()
    finally:
        hook.remove()
        for original, saved in zip(context["states"], states):
            original.copy_(saved)
    assert captured
    bank = torch.cat(captured)

    def forward(x):
        with set_forward_context({}, context["config"], num_tokens=len(x)):
            return layer.mlp(x).clone()

    results = []
    for rows in (2048, 4096, 8192):
        for kind in ("captured", "seeded"):
            if kind == "captured":
                x = bank.repeat((rows + len(bank) - 1) // len(bank), 1)[:rows].contiguous()
            else:
                generator = torch.Generator(device=context["device"]).manual_seed(17)
                x = torch.randn((rows, bank.shape[1]), device=bank.device,
                                dtype=bank.dtype, generator=generator)
                x.mul_(bank.float().std().to(bank.dtype))
            saved = x.clone()
            reference = torch.cat([forward(part) for part in x.split(2048)])
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            actual = forward(x)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            repeat = forward(x)
            finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
            error = float((actual.float() - reference.float()).abs().max())
            scale = float(reference.float().abs().max())
            # Diagnostic tolerance, not an exact-text or model-quality verdict.
            close = finite and torch.allclose(actual, reference, atol=1e-3, rtol=0.02)
            result = dict(rows=rows, inputs=kind, finite=finite,
                          chunk_reference_exact=torch.equal(actual, reference),
                          chunk_reference_close=bool(close), max_abs_error=error,
                          reference_max_abs=scale, repeat_exact=torch.equal(actual, repeat),
                          input_unchanged=torch.equal(x, saved), seconds=seconds,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated())
            results.append(result)
            Path(f"/results/large-prefill-rank{context['rank']}.json").write_text(
                json.dumps(results, indent=2))
            assert finite and close and result["input_unchanged"], result
    return dict(screen_passed=True, checks=results,
                scope="Selected real-weight MLP row capacity and numerical tolerance; "
                      "not full-model memory, exact output or cache qualification")
