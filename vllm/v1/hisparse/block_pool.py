# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from math import isfinite
from time import monotonic

from vllm.distributed.kv_events import MEDIUM_CPU, KVCacheEvent
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, BlockHashWithGroupId, KVCacheBlock


class SharedEventQueueBlockPool(BlockPool):
    """A pool that publishes into another pool's live KV event queue.

    The owner rebinds its queue on every drain, so this reads it through the
    owner rather than holding a reference.
    """

    def __init__(self, *args, event_owner: BlockPool, **kwargs) -> None:
        self._event_owner = event_owner
        super().__init__(*args, **kwargs)

    @property
    def kv_event_queue(self) -> list[KVCacheEvent]:
        return self._event_owner.kv_event_queue

    @kv_event_queue.setter
    def kv_event_queue(self, events: list[KVCacheEvent]) -> None:
        # ``BlockPool.__init__`` seeds an empty queue and ``take_events``
        # swaps in a fresh one; both belong to the owner, which drains it.
        assert not events


class IdleExpiringHostBlockPool(SharedEventQueueBlockPool):
    """Resident host pages with native ownership/LRU and lazy idle expiry.

    Expiry removes prefix hashes, not tensor bytes. Idle pages are already
    free; actual reuse still runs the native pool's watcher and allocation path.
    """

    def __init__(self, *args, idle_ttl_seconds: float | None = 3600, **kwargs):
        if idle_ttl_seconds is not None and (
            type(idle_ttl_seconds) not in (int, float)
            or not isfinite(idle_ttl_seconds)
            or idle_ttl_seconds <= 0
        ):
            raise ValueError("idle_ttl_seconds must be a finite positive number")
        self.idle_ttl_seconds = idle_ttl_seconds
        self._idle_deadlines: OrderedDict[int, float] = OrderedDict()
        super().__init__(*args, **kwargs)
        if self.medium != MEDIUM_CPU:
            raise ValueError("IdleExpiringHostBlockPool requires CPU storage")

    def _mark_idle(self, blocks: Iterable[KVCacheBlock]) -> None:
        if self.idle_ttl_seconds is None:
            return
        deadline = monotonic() + self.idle_ttl_seconds
        for block in blocks:
            if (
                block.ref_cnt == 0
                and not block.is_null
                and block.block_hash is not None
            ):
                self._idle_deadlines[block.block_id] = deadline
                self._idle_deadlines.move_to_end(block.block_id)

    def expire_idle(self) -> int:
        now = monotonic()
        expired = []
        while self._idle_deadlines:
            block_id, deadline = next(iter(self._idle_deadlines.items()))
            if deadline > now:
                break
            block = self.blocks[block_id]
            if block.ref_cnt != 0 or block.is_null:
                raise RuntimeError("Host cache expiry found a referenced idle page")
            self._maybe_evict_cached_block(block)
            self.free_block_queue.remove(block)
            expired.append(block)
        self.free_block_queue.prepend_n(expired)
        return len(expired)

    def get_cached_block(self, block_hash: BlockHash, kv_cache_group_ids: list[int]):
        self.expire_idle()
        return super().get_cached_block(block_hash, kv_cache_group_ids)

    def _remove_cached_block_hashes(self, block: KVCacheBlock):
        self._idle_deadlines.pop(block.block_id, None)
        return super()._remove_cached_block_hashes(block)

    def _insert_block_hash(
        self,
        block_hash_with_group_id: BlockHashWithGroupId,
        block: KVCacheBlock,
        num_tokens: int | None,
    ) -> None:
        super()._insert_block_hash(block_hash_with_group_id, block, num_tokens)
        self._mark_idle((block,))

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        for block in blocks:
            self._idle_deadlines.pop(block.block_id, None)
        super().touch(blocks)

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        blocks = list(ordered_blocks)
        super().free_blocks(blocks)
        self._mark_idle(block for block in blocks if block.pool is self)

    def unpin_blocks(
        self,
        blocks: Iterable[KVCacheBlock],
        on_reuse: Callable[[KVCacheBlock], None],
    ) -> None:
        released = list(blocks)
        super().unpin_blocks(released, on_reuse)
        self._mark_idle(released)

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        self.expire_idle()
        return super().get_new_blocks(num_blocks)

    def reset_prefix_cache(self) -> bool:
        reset = super().reset_prefix_cache()
        if reset:
            self._idle_deadlines.clear()
        return reset
