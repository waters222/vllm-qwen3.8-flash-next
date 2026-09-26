"""Diagnostic whole-K Marlin gate/up and FP32 MLP TP control; eager only."""

import importlib
import inspect
import math


def install():
    import torch

    marlin = importlib.import_module(
        "vllm.model_executor.layers.fused_moe.experts.marlin_moe")
    runner = importlib.import_module(
        "vllm.model_executor.layers.fused_moe.runner.moe_runner")
    gemm = marlin.ops.moe_wna16_marlin_gemm
    signature = inspect.signature(gemm)
    reduce = runner.tensor_model_parallel_all_reduce
    counts = dict(gate_up=0, tp=0)
    shapes = set()

    def controlled_gemm(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        if a["top_k"] <= 1:
            return gemm(*args, **kwargs)
        if (a["size_k"] % 128 or a["size_n"] % 64
                or a["input"].dtype != torch.bfloat16
                or a["use_atomic_add"] or not a["use_fp32_reduce"]):
            raise AssertionError("Unsupported numerical-control gate/up scope")
        block = a["moe_block_size"]
        padded = int(a["num_tokens_past_padded"].item())
        grid = torch.cuda.get_device_properties(a["input"].device).multi_processor_count
        multiple = grid // math.gcd(grid, a["size_n"] // 64)
        old_blocks = padded // block
        blocks = ((old_blocks + multiple - 1) // multiple) * multiple
        ids = torch.full((blocks * block,), a["size_m"] * a["top_k"],
                         dtype=a["sorted_token_ids"].dtype, device=a["input"].device)
        ids[:padded].copy_(a["sorted_token_ids"][:padded])
        experts = torch.zeros((blocks,), dtype=a["expert_ids"].dtype,
                              device=a["input"].device)
        experts[:old_blocks].copy_(a["expert_ids"][:old_blocks])
        a.update(sorted_token_ids=ids, expert_ids=experts,
                 num_tokens_past_padded=torch.full_like(
                     a["num_tokens_past_padded"], blocks * block),
                 thread_k=128, thread_n=64, blocks_per_sm=1)
        counts["gate_up"] += 1
        shapes.add((a["size_m"], a["size_n"], a["size_k"]))
        return gemm(*bound.args, **bound.kwargs)

    def controlled_reduce(states, *args, **kwargs):
        counts["tp"] += 1
        return reduce(states.float(), *args, **kwargs).to(states.dtype)

    marlin.ops.moe_wna16_marlin_gemm = controlled_gemm
    runner.tensor_model_parallel_all_reduce = controlled_reduce

    def remove():
        marlin.ops.moe_wna16_marlin_gemm = gemm
        runner.tensor_model_parallel_all_reduce = reduce
        return dict(counts=counts, gate_up_shapes=sorted(shapes), hooks_restored=True,
                    exercised=all(counts.values()), diagnostic_only=True)

    return remove
