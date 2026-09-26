# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact block-table relocation for drained Flash-Next session transactions.

Only the native private-block managers used by the qualified no-prefix-cache
Flash layout are supported. Worker transfers and scheduler queue transitions
are the coordinator's responsibility. No cache is recomputed here.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MambaAlignLayout:
    allocated: bool
    last_state_index: int | None
    retired_count: int | None
    checkpoint: tuple[int, int] | None


@dataclass(frozen=True)
class SessionBlockLayout:
    block_ids: tuple[tuple[int, ...], ...]
    cached_counts: tuple[int | None, ...]
    null_block_id: int
    align_states: tuple[MambaAlignLayout | None, ...]

    @property
    def private_blocks(self) -> int:
        return sum(
            block_id != self.null_block_id
            for group in self.block_ids
            for block_id in group
        )


class FlashSessionAllocator:
    """Capture, release and reserve exact logical tables using native block pools."""

    def __init__(self, kv_cache_manager: Any):
        self.manager = kv_cache_manager
        self.pool = kv_cache_manager.block_pool
        self.groups = tuple(kv_cache_manager.coordinator.single_type_managers)
        if kv_cache_manager.enable_caching or not self.groups:
            raise ValueError("session swapping requires private cache groups")
        for group in self.groups:
            if (
                group.__class__.__name__
                not in ("FullAttentionManager", "CircularBufferManager", "MambaManager")
                or group.block_pool is not self.pool
                or getattr(group, "mamba_cache_mode", "none") not in ("none", "align")
            ):
                raise ValueError("unsupported cache manager for session swapping")

    def capture_layout(self, request_id: str) -> SessionBlockLayout:
        tables = []
        cached = []
        align = []
        seen = set()
        for group in self.groups:
            if (
                request_id in group._partial_hit_reqs
                or group._pending_cow_copies
                or group.new_block_ids
                or any(
                    entry[0] == request_id
                    for entry in group._pending_boundary_state_offloads
                )
            ):
                raise ValueError("session cache manager still has pending work")
            blocks = group.req_to_blocks.get(request_id)
            if blocks is None:
                raise ValueError("session is missing an allocated cache group")
            for block in blocks:
                if block.is_null:
                    continue
                if block.ref_cnt != 1 or block.block_id in seen:
                    raise ValueError("session cache blocks must be exclusively owned")
                seen.add(block.block_id)
            tables.append(tuple(block.block_id for block in blocks))
            cached.append(group.num_cached_block.get(request_id))
            state = None
            if getattr(group, "mamba_cache_mode", "none") == "align":
                if request_id in group._producer_partial_tail_reqs:
                    raise ValueError("session has a pending Mamba partial tail")
                # These are logical table columns, not physical block IDs.
                # Keep null positions and columns unchanged during relocation.
                state = MambaAlignLayout(
                    request_id in group._allocated_block_reqs,
                    group.last_state_block_idx.get(request_id),
                    group._num_retired_blocks.get(request_id),
                    group._checkpoints.get(request_id),
                )
            align.append(state)
        return SessionBlockLayout(
            tuple(tables), tuple(cached), self.pool.null_block.block_id, tuple(align)
        )

    def release(self, request_id: str, expected: SessionBlockLayout) -> None:
        """Release only after all workers acknowledge a complete cold snapshot."""
        if self.capture_layout(request_id) != expected:
            raise ValueError("hot cache ownership changed during session capture")
        for group in self.groups:
            group.free(request_id)

    def reserve(
        self, request_id: str, layout: SessionBlockLayout
    ) -> SessionBlockLayout:
        """Reserve an exact destination, never exposing it to scheduling here.

        Restored bytes must not be placed on the new-block zeroing queue: doing
        so would erase the checkpoint at the next forward. The worker adapter
        writes whole physical pages before the coordinator makes them runnable.
        """
        if (
            len(layout.block_ids) != len(self.groups)
            or len(layout.cached_counts) != len(self.groups)
            or len(layout.align_states) != len(self.groups)
            or layout.null_block_id != self.pool.null_block.block_id
        ):
            raise ValueError("cold session cache layout changed")
        for group, state in zip(self.groups, layout.align_states, strict=True):
            if (getattr(group, "mamba_cache_mode", "none") == "align") != (
                state is not None
            ):
                raise ValueError("cold session Mamba mode changed")
        if any(request_id in group.req_to_blocks for group in self.groups):
            raise ValueError("request already owns hot cache blocks")
        if layout.private_blocks > self.pool.get_num_free_blocks():
            raise MemoryError("insufficient hot blocks to restore session")
        allocated = self.pool.get_new_blocks(layout.private_blocks)
        blocks = iter(allocated)
        touched = []
        try:
            for group, source, cached, state in zip(
                self.groups,
                layout.block_ids,
                layout.cached_counts,
                layout.align_states,
                strict=True,
            ):
                table = [
                    self.pool.null_block if i == layout.null_block_id else next(blocks)
                    for i in source
                ]
                group.req_to_blocks[request_id] = table
                touched.append(group)
                if cached is not None:
                    group.num_cached_block[request_id] = cached
                if state is not None:
                    if state.allocated:
                        group._allocated_block_reqs.add(request_id)
                    for mapping, value in (
                        (group.last_state_block_idx, state.last_state_index),
                        (group._num_retired_blocks, state.retired_count),
                        (group._checkpoints, state.checkpoint),
                    ):
                        if value is not None:
                            mapping[request_id] = value
            return self.capture_layout(request_id)
        except Exception:
            for group in touched:
                # Clear native mode-specific maps without releasing these pages
                # twice; the complete allocation is returned below.
                group.pop_blocks_for_free(request_id)
            self.pool.free_blocks(reversed(allocated))
            raise
