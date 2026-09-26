"""Weight-free staged-prefill warp sweep; no engine dispatch changes."""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from vllm.models.qwen4_exp.nvidia.ops import qsa
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor


def run_case(rank, rows, length):
    page, heads, dim, width = 944, 6, 256, 2051
    pages = (length + page - 1) // page
    torch.manual_seed(901 + rank)
    pool = torch.zeros((pages + 1, 12, page, 1, 512),
                       dtype=torch.bfloat16, pin_memory=True)
    backing = pool[:, 6].transpose(1, 2)
    backing.copy_(torch.randn(backing.shape, dtype=torch.bfloat16))
    cache = get_accelerator_view_from_cpu_tensor(backing)
    table = (torch.randperm(pages) + 1).to(device='cuda', dtype=torch.int32)
    query = torch.randn((rows, heads, dim), device='cuda', dtype=torch.bfloat16)
    permutation = torch.randperm((length - rows - 4) // 4)
    groups = permutation[(torch.arange(512)[None, :] +
                          torch.arange(rows)[:, None] * 13) % len(permutation)]
    selected = torch.empty((rows, width + 1), dtype=torch.int32)
    selected[:, :2048] = (groups[:, :, None] * 4 + torch.arange(4)).flatten(1)
    selected[:, 2048:2051] = (
        length - rows + torch.arange(rows)[:, None] - torch.arange(3)
    )
    selected[:, -1] = width
    selected = selected.cuda()
    row_ids = torch.arange(rows, device='cuda', dtype=torch.int64)
    arena = qsa.qsa_get_staging_arena(cache, 96 * 1024 * 1024)
    assert arena is not None
    native_config = qsa._select_config(rows, 1, True, width)
    original = qsa._select_config
    graphs, outputs = {}, {}
    try:
        for warps in (1, 2, 4, 8):
            qsa._select_config = lambda *args, w=warps: (
                native_config[0], w, native_config[2], native_config[3])
            output = outputs[warps] = torch.empty_like(query)

            def execute(output=output):
                qsa.qsa_sparse_paged_attention_staged(
                    query, cache, selected, table, row_ids, pages - 1,
                    arena, True, output)

            for _ in range(3):
                execute()
            torch.cuda.synchronize()
            graph = graphs[warps] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                execute()
    finally:
        qsa._select_config = original
    checks = []
    for change in range(3):
        if change:
            query.mul_(0.9)
            backing.mul_(0.9)
            table.copy_(table.roll(1))
        saved = backing.clone()
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(saved, backing), 'RAM KV changed during read-only attention'
        reference = outputs[native_config[1]]
        checks.extend({'warps': w, 'change': change,
                       'exact': torch.equal(reference, out),
                       'finite': bool(torch.isfinite(out).all()),
                       'max_abs_error': float((reference - out).abs().max())}
                      for w, out in outputs.items())
    samples = {w: [] for w in graphs}
    for repeat in range(5):
        order = (1, 2, 4, 8) if repeat % 2 == 0 else (8, 4, 2, 1)
        for warps in order:
            dist.barrier()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(20):
                graphs[warps].replay()
            end.record()
            end.synchronize()
            samples[warps].append(start.elapsed_time(end) / 20)
    return {'rows': rows, 'context': length, 'native_config': native_config,
            'cache_strides': list(cache.stride()), 'checks': checks,
            'milliseconds': samples, 'ram_bytes_unchanged': True}


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    assert dist.get_world_size() == 4
    assert torch.cuda.get_device_capability() == (8, 6)
    path = Path(f'/results/qsa-prefill-tuning-rank{rank}.json')
    report = {
        'complete': False, 'cases': [],
        'scope': 'Synthetic staged-prefill component only; no engine or TG changes',
    }
    try:
        for length in (16384, 32768):
            for rows in (160, 944, 1104):
                report['cases'].append(run_case(rank, rows, length))
                path.write_text(json.dumps(report, indent=2))
        report['complete'] = True
    finally:
        path.write_text(json.dumps(report, indent=2))
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
