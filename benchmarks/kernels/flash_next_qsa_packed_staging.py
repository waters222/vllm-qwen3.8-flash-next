"""Experimental packed staging; benchmark-only until independently qualified."""

import torch

from vllm.models.qwen4_exp.nvidia.ops.qsa import (
    qsa_get_staging_arena,
    qsa_sparse_paged_attention,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _clear(FLAGS, MAP, COUNTS, CAPACITY: tl.constexpr, TILE: tl.constexpr):
    token = tl.program_id(0) * TILE + tl.arange(0, TILE)
    request = tl.program_id(1)
    tl.store(FLAGS + request * CAPACITY + token, 0, token < CAPACITY)
    tl.store(MAP + request * CAPACITY + token, -1, token < CAPACITY)
    if tl.program_id(0) == 0:
        tl.store(COUNTS + request, 0)


@triton.jit
def _pack(
    CACHE,
    TABLE,
    INDICES,
    FLAGS,
    MAP,
    COUNTS,
    ARENA,
    PAGE: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCKS: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    MAX_UNION: tl.constexpr,
):
    row = tl.program_id(1)
    request = row // 5
    count = tl.load(INDICES + row * INDEX_STRIDE + WIDTH)
    if tl.program_id(0) * 4 < count:
        column = tl.program_id(0) * 4 + tl.arange(0, 4)
        token = tl.load(INDICES + row * INDEX_STRIDE + column, column < WIDTH, other=-1)
        valid = (column < WIDTH) & (column < count)
        valid &= (token >= 0) & (token < CAPACITY)
        safe = tl.maximum(token, 0)
        physical = tl.load(
            TABLE + request * TABLE_STRIDE + safe // PAGE, valid, other=-1
        )
        valid &= (physical >= 0) & (physical < BLOCKS)
        old = tl.atomic_or(FLAGS + request * CAPACITY + safe, 1, valid, sem="relaxed")
        winner = valid & (old == 0)
        offset = tl.atomic_add(
            COUNTS + request + tl.full((4,), 0, tl.int32),
            tl.full((4,), 1, tl.int32),
            winner,
            sem="relaxed",
        )
        packed = request * MAX_UNION + offset
        dim = tl.arange(0, 512)
        source = (physical.to(tl.int64) * CACHE_STRIDE + safe % PAGE * 512)[:, None]
        value = tl.load(CACHE + source + dim[None, :], winner[:, None], other=0)
        tl.store(ARENA + packed[:, None] * 512 + dim[None, :], value, winner[:, None])
        tl.store(MAP + request * CAPACITY + safe, packed, winner)


@triton.jit
def _remap(
    INDICES,
    MAP,
    REMAPPED,
    CAPACITY: tl.constexpr,
    STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(1)
    column = tl.program_id(0) * TILE + tl.arange(0, TILE)
    token = tl.load(INDICES + row * STRIDE + column, column <= WIDTH, other=-1)
    valid = (column < WIDTH) & (token >= 0) & (token < CAPACITY)
    mapped = tl.load(MAP + row // 5 * CAPACITY + tl.maximum(token, 0), valid, other=-1)
    tl.store(
        REMAPPED + row * (WIDTH + 1) + column,
        tl.where(column == WIDTH, token, mapped),
        column <= WIDTH,
    )


_SCRATCH = {}


def try_verify_staging(layer, query, cache, indices, metadata, output):
    rows, page = query.shape[0], cache.shape[2]
    requests = rows // 5
    table = metadata.block_table
    capacity = table.shape[1] * page
    width = indices.shape[1] - 1
    max_union = width * 5
    arena = qsa_get_staging_arena(cache, layer._qsa_stage_arena_bytes)
    pages = triton.cdiv(requests * max_union, page)
    assert arena is not None and pages <= arena.shape[0]
    key = (cache.device, requests, capacity, page, width)
    if key not in _SCRATCH:
        _SCRATCH[key] = (
            torch.empty((requests, capacity), device=cache.device, dtype=torch.int32),
            torch.empty((requests, capacity), device=cache.device, dtype=torch.int32),
            torch.empty(requests, device=cache.device, dtype=torch.int32),
            torch.empty((rows, width + 1), device=cache.device, dtype=torch.int32),
            torch.arange(pages, device=cache.device, dtype=torch.int32)[None, :],
            torch.zeros(rows, device=cache.device, dtype=torch.int32),
        )
    flags, mapping, counts, remapped, local_table, local_ids = _SCRATCH[key]
    _clear[(triton.cdiv(capacity, 256), requests)](
        flags, mapping, counts, capacity, 256
    )
    _pack[(triton.cdiv(width, 4), rows)](
        cache,
        table,
        indices,
        flags,
        mapping,
        counts,
        arena,
        page,
        capacity,
        cache.shape[0],
        cache.stride(0),
        table.stride(0),
        indices.stride(0),
        width,
        max_union,
        num_warps=4,
    )
    _remap[(triton.cdiv(width + 1, 256), rows)](
        indices, mapping, remapped, capacity, indices.stride(0), width, 256
    )
    keys, values = arena[:pages].transpose(1, 2).split(256, dim=-1)
    qsa_sparse_paged_attention(
        query, keys, values, remapped, local_table, local_ids, False, output
    )
    return True
