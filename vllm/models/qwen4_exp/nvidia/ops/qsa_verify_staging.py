# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional packed host-KV staging for uniform TP4 MTP4 verification."""

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.block_table import get_block_table_width

from .qsa import qsa_get_staging_arena, qsa_sparse_paged_attention


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


_SCRATCH: dict[tuple, tuple[torch.Tensor, ...]] = {}
_REPORTED: set[tuple] = set()
_LOGGER = init_logger(__name__)


def try_verify_staging(layer, query, cache, indices, metadata, output) -> bool:
    """Write uniform five-row verification output, or leave it untouched.

    The arena and scratch are shared across sequential layers on the model's
    compute stream. No cache contents or ownership metadata are modified.
    """
    if not getattr(layer, "_qsa_verify_staging", False):
        return False
    if not getattr(layer, "_qsa_kv_offload", False):
        return False
    rows = metadata.num_actual_tokens
    table = metadata.block_table
    requests = metadata.seq_lens.shape[0]
    if metadata.max_query_len == 5:
        signature = (
            tuple(query.shape),
            tuple(cache.shape),
            tuple(cache.stride()),
            tuple(indices.shape),
            tuple(table.shape),
            tuple(metadata.query_start_loc.shape),
            rows,
            requests,
        )
        if signature not in _REPORTED:
            _REPORTED.add(signature)
            _LOGGER.info(
                "QSA verify staging geometry (query/cache/stride/indices/"
                "table/starts/rows/requests): %s",
                signature,
            )
    # Sum(query lengths) == requests * max_query_len proves uniformity.
    if (
        metadata.max_query_len != 5
        or not 1 <= requests <= 4
        or rows != requests * 5
        or table.shape[0] < requests
        or metadata.query_start_loc.shape[0] < requests + 1
        or query.shape != (rows, 6, 256)
        or cache.ndim != 4
        or cache.shape[1] != 1
        or cache.shape[3] != 512
        or cache.stride(3) != 1
        or cache.stride(2) != 512
        or cache.stride(0) < cache.shape[2] * 512
        or cache.dtype != torch.bfloat16
        or query.dtype != torch.bfloat16
        or not cache.is_cuda
        or indices.shape != (rows, 2052)
        or indices.stride(1) != 1
        or table.stride(1) != 1
        or torch.cuda.get_device_capability(cache.device) != (8, 6)
    ):
        return False
    logical_pages, page = table.shape[1], cache.shape[2]
    capacity = logical_pages * page
    width = indices.shape[1] - 1
    max_union = width * 5
    pages = triton.cdiv(requests * max_union, page)
    arena_bytes = getattr(layer, "_qsa_stage_arena_bytes", 0)
    if (
        logical_pages < 1
        or logical_pages > get_block_table_width(triton.cdiv(32768, page), page)
        or pages * page * 1024 > arena_bytes
    ):
        return False
    arena = qsa_get_staging_arena(cache, arena_bytes)
    if arena is None:
        return False
    _REPORTED.add(("active_shape", rows, requests))
    if ("active",) not in _REPORTED:
        _REPORTED.add(("active",))
        _LOGGER.info("QSA verification staging active; reusing existing arena")
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
