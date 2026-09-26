"""Isolated TP4 QSA verification test of the optional engine helper."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention
from vllm.models.qwen4_exp.nvidia.ops.qsa_verify_staging import try_verify_staging
from vllm.triton_utils import triton
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
from vllm.v1.worker.block_table import get_block_table_width

if os.environ.get("FLASH_QSA_PACKED_TEST") == "1":
    from packed import try_verify_staging


def run_case(rank, concurrency, length, shared_groups):
    production_layout = os.environ.get("FLASH_QSA_PRODUCTION_LAYOUT") == "1"
    page, dim, heads, width = (944 if production_layout else 472), 256, 6, 2051
    pages = triton.cdiv(length, page)
    if production_layout:
        pages = get_block_table_width(pages, page)
    capacity = pages * page
    rows = concurrency * 5
    torch.manual_seed(17 + rank)
    pool_layers = 3 if os.environ.get("FLASH_QSA_STRIDED_TEST") == "1" else 1
    if production_layout:
        pool_layers = 12
    host_pool = torch.full(
        (concurrency * pages + 1, pool_layers, page, 1, 2 * dim),
        float("nan"),
        dtype=torch.bfloat16,
        pin_memory=True,
    )
    backing = host_pool[:, pool_layers // 2].transpose(1, 2)
    backing.copy_(torch.randn(backing.shape, dtype=torch.bfloat16))
    backing[0].fill_(float("nan"))
    cache = get_accelerator_view_from_cpu_tensor(backing)
    table = torch.randperm(concurrency * pages).reshape(concurrency, pages) + 1
    table = table.to(device="cuda", dtype=torch.int32)
    query = torch.randn((rows, heads, dim), dtype=torch.bfloat16, device="cuda")
    selected = torch.full((rows, width + 1), -1, dtype=torch.int32)
    group_count = min(512, (length - 8) // 4)
    shared_groups = min(shared_groups, group_count)
    valid_tokens = group_count * 4 + 3
    for request in range(concurrency):
        pool = torch.randperm((length - 8) // 4)
        common = pool[:shared_groups]
        for offset in range(5):
            extras = pool[shared_groups:].roll(offset * (group_count - shared_groups))[
                : group_count - shared_groups
            ]
            groups = torch.cat((common, extras))
            tokens = (groups[:, None] * 4 + torch.arange(4)).flatten()
            row = request * 5 + offset
            selected[row, : tokens.numel()] = tokens
            selected[row, tokens.numel() : valid_tokens] = torch.arange(
                length - 3, length
            )
            selected[row, -1] = valid_tokens
    union_tokens = [
        int(torch.unique(selected[r * 5 : (r + 1) * 5, :valid_tokens]).numel())
        for r in range(concurrency)
    ]
    selected = selected.cuda()
    full_selection = selected.clone()
    req_ids = torch.arange(
        concurrency, device="cuda", dtype=torch.int32
    ).repeat_interleave(5)
    keys, values = cache.transpose(1, 2).split(dim, dim=-1)
    native_out, staged_out = torch.empty_like(query), torch.empty_like(query)
    layer = SimpleNamespace(
        _qsa_verify_staging=True,
        _qsa_kv_offload=True,
        _qsa_stage_arena_bytes=96 * 1024 * 1024,
    )
    metadata = SimpleNamespace(
        num_actual_tokens=rows,
        max_query_len=5,
        block_table=table,
        seq_lens=torch.full((concurrency,), length, device="cuda"),
        query_start_loc=torch.arange(concurrency + 1, device="cuda") * 5,
    )

    def native():
        qsa_sparse_paged_attention(
            query, keys, values, selected, table, req_ids, False, native_out
        )

    def staged():
        assert try_verify_staging(layer, query, cache, selected, metadata, staged_out)

    for _ in range(3):
        native()
        staged()
    torch.cuda.synchronize()
    graphs = []
    for fn in (native, staged):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        graphs.append(graph)
    checks = []
    full_table = table.clone()
    for change in range(6):
        if change:
            query.mul_(0.9)
            backing[1:].mul_(0.9)
            table.copy_(table.roll(1, dims=1))
            rotate_width = min(2048, valid_tokens)
            selected[:, :rotate_width].copy_(selected[:, :rotate_width].roll(4, dims=1))
            # Change valid lengths and leave unused selections explicitly invalid.
            if change == 2:
                selected[:, min(1024, valid_tokens) : width] = -1
                selected[:, -1] = min(1024, valid_tokens)
            elif change == 3:
                selected[:, :width] = -1
                selected[:, -1] = 0
            elif change == 4:
                selected.copy_(full_selection)
                table[:, ::3] = -1
                table[:, 1::3] = cache.shape[0] + 1
            elif change == 5:
                table.copy_(full_table)
                selected[:, :4] = capacity + 100
        padding = torch.arange(width, device="cuda")[None, :] >= selected[:, -1:]
        assert bool((selected[:, :width][padding] == -1).all()), (
            "Invalid packed padding"
        )
        graphs[0].replay()
        graphs[1].replay()
        torch.cuda.synchronize()
        exact = torch.equal(native_out, staged_out)
        checks.append(
            dict(
                change=change,
                exact=exact,
                finite=bool(torch.isfinite(staged_out).all()),
                max_abs_error=float((native_out - staged_out).abs().max()),
            )
        )
        assert exact and checks[-1]["finite"], checks[-1]
    selected.copy_(full_selection)
    table.copy_(full_table)
    # Time the full selections, including both high and low overlap cases.
    samples = [[], []]
    for iteration in range(6):
        for index in (0, 1) if iteration % 2 == 0 else (1, 0):
            dist.barrier()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(20):
                graphs[index].replay()
            end.record()
            end.synchronize()
            samples[index].append(start.elapsed_time(end) / 20)
    return dict(
        concurrency=concurrency,
        context=length,
        shared_groups=shared_groups,
        initial_union_tokens=union_tokens,
        timed_valid_tokens=valid_tokens,
        checks=checks,
        native_ms=samples[0],
        staged_ms=samples[1],
        arena_bytes=layer._qsa_stage_arena_bytes,
        cache_strides=list(cache.stride()),
        host_pool_layers=pool_layers,
    )


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 4 and torch.cuda.get_device_capability() == (8, 6)
    results = []
    path = Path(f"/results/qsa-verify-staging-rank{rank}.json")
    concurrencies = (
        (2, 4, 6) if os.environ.get("FLASH_QSA_PACKED_TEST") == "1" else (2, 4)
    )
    for concurrency in concurrencies:
        for length in (128, 512, 8192, 32768):
            for shared in (0, 480):
                results.append(run_case(rank, concurrency, length, shared))
                path.write_text(
                    json.dumps(dict(complete=False, cases=results), indent=2)
                )
    path.write_text(
        json.dumps(
            dict(
                complete=True,
                cases=results,
                scope="Synthetic QSA component; no engine/cache qualification",
            ),
            indent=2,
        )
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
