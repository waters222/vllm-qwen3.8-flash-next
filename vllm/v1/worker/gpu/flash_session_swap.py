# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, rank-local cold storage for complete Flash-Next request state.

This is a transfer primitive, not a scheduler or prefix-cache connector. The
coordinator must quiesce a request on every rank, supply its complete block
tables and worker metadata, and obtain every rank's acknowledgement *before*
releasing hot blocks. Restore destinations must be exclusively reserved and
must not become runnable until every rank acknowledges. A failed restore may
have written some destination bytes; its cold source is deliberately retained.

RAM-backed QSA pages must be registered too: they use hot block IDs, so freeing
only the GPU allocation would allow another request to overwrite their KV.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class SessionKey:
    """Engine-local identity; never accept an unscoped client session name."""

    client_index: int
    request_id: str
    generation: int

    def __post_init__(self) -> None:
        if self.client_index < 0 or self.generation < 0 or not self.request_id:
            raise ValueError("invalid scoped session identity")


@dataclass(frozen=True)
class SwapBoundary:
    """Scheduler/worker agreement at a drained iteration boundary.

    All speculative state slots and acceptance metadata are preserved; a
    drained boundary does not imply that deferred recurrent rollback has run.
    """

    computed_tokens: int
    inflight_steps: int

    def validate(self) -> None:
        if self.computed_tokens < 0 or self.inflight_steps != 0:
            raise ValueError("session swap requires a drained, valid token boundary")


@dataclass(frozen=True)
class BlockRegion:
    """Byte-exact block view of GPU state or authoritative host QSA backing.

    The first dimension is the scheduler's physical block ID. Byte columns
    may be strided; no dtype conversion or contiguous-layout assumption is
    made. Separate regions can represent different parts of the same group.
    """

    name: str
    group_id: int
    blocks: torch.Tensor = field(repr=False)

    def __post_init__(self) -> None:
        if not self.name or self.group_id < 0:
            raise ValueError("invalid session-cache region identity")
        if (
            self.blocks.ndim != 2
            or self.blocks.dtype not in (torch.uint8, torch.int8)
            or self.blocks.shape[1] == 0
            or self.blocks.device.type not in ("cpu", "cuda")
        ):
            raise ValueError("session-cache region must be a CPU/CUDA byte matrix")


def flash_cache_regions(
    kv_cache_config: Any,
    kv_caches: Mapping[str, torch.Tensor],
    forward_context: Mapping[str, Any],
) -> tuple[tuple[BlockRegion, ...], tuple[int, ...]]:
    """Bind the fork's block-outermost shared pool and actual QSA host pools.

    Copies entire physical pages, including recurrent speculative slots and
    indexer data. Layouts with layer-outermost storage or separate host block
    IDs require a different adapter and are rejected. This does not allocate
    or replace any live cache tensor; captured CUDA graph addresses stay fixed.
    """
    config = kv_cache_config
    if config.kv_cache_layout != "BLNHC" or config.num_blocks <= 0:
        raise ValueError("session swap requires the qualified BLNHC cache layout")
    if not config.kv_cache_tensors:
        raise ValueError("session swap requires an allocated physical cache")
    stride, remainder = divmod(config.kv_cache_tensors[0].size, config.num_blocks)
    if remainder or stride <= 0:
        raise ValueError("cache allocation is not a whole number of physical pages")
    if any(
        t.host_resident
        or t.size != config.num_blocks * stride
        or t.block_stride != stride
        for t in config.kv_cache_tensors
    ):
        raise ValueError("unsupported session-cache placement or host block pool")

    groups = config.kv_cache_groups
    local_names = {name for group in groups for name in group.layer_names}
    if not local_names or any(
        name not in kv_caches or name not in forward_context for name in local_names
    ):
        raise ValueError("missing local cache tensors in session-cache registration")
    tensors = [kv_caches[name] for name in sorted(local_names)]
    storage = tensors[0].untyped_storage()
    if storage.nbytes() != config.num_blocks * stride or any(
        t.device != tensors[0].device
        or t.untyped_storage().data_ptr() != storage.data_ptr()
        for t in tensors
    ):
        raise ValueError("session swap requires one shared hot-cache allocation")
    raw = torch.empty(0, dtype=torch.uint8, device=tensors[0].device).set_(storage)
    pages = raw.view(config.num_blocks, stride)
    regions = []
    empty = []
    for group_id, group in enumerate(groups):
        if group.host_resident:
            raise ValueError("separate host block pools are not supported")
        if not group.layer_names:
            empty.append(group_id)
            continue
        regions.append(BlockRegion(f"hot/{group_id}", group_id, pages))
        for name in group.layer_names:
            layer = forward_context.get(name)
            host = getattr(layer, "_qsa_host_kv", None)
            if getattr(layer, "_qsa_kv_offload", False) and host is None:
                raise ValueError("QSA host backing has not been bound")
            if host is not None:
                if (
                    host.device.type != "cpu"
                    or not host.is_contiguous()
                    or host.shape[0] != config.num_blocks
                ):
                    raise ValueError("unsupported QSA host backing layout")
                regions.append(
                    BlockRegion(
                        f"host/{name}",
                        group_id,
                        host.view(torch.uint8).view(config.num_blocks, -1),
                    )
                )
    return tuple(regions), tuple(empty)


def native_group_cache_layouts(
    regions: Sequence[BlockRegion],
    num_groups: int,
    *,
    empty_groups: Sequence[int] = (),
):
    """Adapt complete Flash regions to native offload, one pool per group.

    No cache allocation or copy occurs here. Keep QSA host owners alive through
    the regions; native transfers hold accelerator aliases of pinned RAM.
    Separate group layouts avoid reserving QSA sidecars for state-only chunks.
    Caller must budget the sum of all pools, not apply its budget to each pool.
    This does not enable the prefix connector or implement session ownership.
    """
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    from vllm.v1.kv_offload.base import (
        CanonicalKVCacheRef,
        CanonicalKVCaches,
        CanonicalKVCacheTensor,
    )

    if num_groups <= 0:
        raise ValueError("native session layout needs positive group count")
    names = [region.name for region in regions]
    present = {region.group_id for region in regions}
    empty = set(empty_groups)
    if (
        len(names) != len(set(names))
        or len(empty) != len(empty_groups)
        or present & empty
        or present | empty != set(range(num_groups))
    ):
        raise ValueError("native session layout needs unique, complete group coverage")
    devices = {region.blocks.device for region in regions if region.blocks.is_cuda}
    if len(devices) != 1:
        raise ValueError("native session layout needs exactly one CUDA device")
    counts = {len(region.blocks) for region in regions}
    if len(counts) != 1 or next(iter(counts)) <= 0:
        raise ValueError("native session regions must share physical block count")
    # Native handlers reshape physical pages; silently materializing a strided
    # view would disconnect transfers from the live cache.
    for region in regions:
        if not region.blocks.is_contiguous():
            raise ValueError("native session layout requires contiguous byte pages")
        if not region.blocks.is_cuda and not region.blocks.is_pinned():
            raise ValueError("native session host pages must be pinned")

    layouts = []
    with torch.cuda.device(next(iter(devices))):
        for group_id in range(num_groups):
            if group_id in empty:
                layouts.append(None)
                continue
            tensors, refs, spans = [], [], []
            for region in regions:
                if region.group_id != group_id:
                    continue
                blocks = region.blocks
                begin, end = blocks.data_ptr(), blocks.data_ptr() + blocks.numel()
                if any(begin < stop and start < end for start, stop in spans):
                    raise ValueError("native session group has overlapping regions")
                spans.append((begin, end))
                tensor = blocks.view(torch.int8)
                if not tensor.is_cuda:
                    tensor = get_accelerator_view_from_cpu_tensor(tensor)
                page_bytes = blocks.shape[1]
                refs.append(CanonicalKVCacheRef(len(tensors), page_bytes))
                tensors.append(CanonicalKVCacheTensor(tensor, page_bytes))
            layouts.append(CanonicalKVCaches(tensors, [refs]))
    return tuple(layouts)


@dataclass(frozen=True)
class _RegionCopy:
    region_name: str
    group_id: int
    logical_positions: tuple[int, ...]
    data: torch.Tensor = field(repr=False)


@dataclass(frozen=True)
class _Checkpoint:
    boundary: SwapBoundary
    block_counts: tuple[int, ...]
    null_positions: tuple[tuple[int, ...], ...]
    metadata: bytes = field(repr=False)
    regions: tuple[_RegionCopy, ...] = field(repr=False)
    size_bytes: int
    digest: str


class SessionColdStore:
    """Synchronous copy/ack primitive with explicit admission and retirement.

    No automatic eviction: discarding a cold checkpoint can destroy the only
    resumable state. RAM admission failure leaves the hot request untouched.
    The caller owns CUDA stream ordering and all-rank transaction coordination.
    This object is used only from the worker's serialized control thread.
    """

    def __init__(
        self,
        regions: Sequence[BlockRegion],
        num_groups: int,
        capacity_bytes: int,
        max_sessions: int,
        *,
        empty_groups: Sequence[int] = (),
        null_block_id: int = 0,
    ) -> None:
        if num_groups <= 0 or capacity_bytes <= 0 or max_sessions <= 0:
            raise ValueError("session-cache limits must be positive")
        if null_block_id < 0:
            raise ValueError("null block ID must be nonnegative")
        self.regions = tuple(regions)
        self.num_groups = num_groups
        self.capacity_bytes = capacity_bytes
        self.max_sessions = max_sessions
        self.null_block_id = null_block_id
        names = [region.name for region in self.regions]
        if len(set(names)) != len(names):
            raise ValueError("duplicate session-cache region name")
        present = {region.group_id for region in self.regions}
        empty = set(empty_groups)
        if present & empty or present | empty != set(range(num_groups)):
            raise ValueError("every local cache group must have explicit coverage")
        self._checkpoints: dict[SessionKey, _Checkpoint] = {}
        self.used_bytes = 0

    def _synchronize_devices(self) -> None:
        # Host QSA tensors can also have outstanding CUDA UVA writes. A
        # current-stream fence alone is insufficient for another compute stream.
        devices = {r.blocks.device for r in self.regions if r.blocks.is_cuda}
        for device in devices:
            torch.cuda.synchronize(device)

    def quiesce(self) -> None:
        """Fence reads/writes before coordinator rollback releases hot blocks."""
        self._synchronize_devices()

    @property
    def session_count(self) -> int:
        return len(self._checkpoints)

    def contains(self, key: SessionKey) -> bool:
        return key in self._checkpoints

    def metadata(self, key: SessionKey) -> bytes:
        """Read internal worker metadata before reserving a destination slot."""
        checkpoint = self._verified_checkpoint(key)
        return checkpoint.metadata

    @staticmethod
    def _digest(key, boundary, block_counts, null_positions, metadata, regions) -> str:
        descriptor = json.dumps(
            [
                [key.client_index, key.request_id, key.generation],
                boundary.computed_tokens,
                boundary.inflight_steps,
                block_counts,
                null_positions,
                len(metadata),
                [
                    (
                        r.region_name,
                        r.group_id,
                        r.logical_positions,
                        str(r.data.dtype),
                        list(r.data.shape),
                    )
                    for r in regions
                ],
            ],
            separators=(",", ":"),
        ).encode()
        digest = hashlib.sha256(descriptor)
        digest.update(metadata)
        for region in regions:
            # Copies are contiguous CPU byte matrices. Hash the storage view,
            # avoiding another large temporary allocation just for a checksum.
            if region.data.numel():
                digest.update(memoryview(region.data.numpy()).cast("B"))
        return digest.hexdigest()

    def _verified_checkpoint(self, key: SessionKey) -> _Checkpoint:
        checkpoint = self._checkpoints[key]
        digest = self._digest(
            key,
            checkpoint.boundary,
            checkpoint.block_counts,
            checkpoint.null_positions,
            checkpoint.metadata,
            checkpoint.regions,
        )
        if digest != checkpoint.digest:
            raise RuntimeError("cold checkpoint integrity check failed; do not resume")
        return checkpoint

    def _validate_tables(
        self, block_ids: Sequence[Sequence[int]]
    ) -> tuple[tuple[int, ...], ...]:
        tables = tuple(tuple(ids) for ids in block_ids)
        if len(tables) != self.num_groups:
            raise ValueError("session checkpoint requires every cache group")
        seen: set[int] = set()
        for ids in tables:
            for block_id in ids:
                if type(block_id) is not int or block_id < 0:
                    raise ValueError("invalid session-cache block ID")
                if block_id == self.null_block_id:
                    continue
                if block_id in seen:
                    raise ValueError("shared hot blocks are not supported")
                seen.add(block_id)
        for region in self.regions:
            if any(i >= len(region.blocks) for i in tables[region.group_id]):
                raise ValueError("session-cache block ID exceeds region bounds")
        return tables

    def _positions(self, ids: Sequence[int], *, null: bool) -> tuple[int, ...]:
        return tuple(
            i
            for i, block_id in enumerate(ids)
            if (block_id == self.null_block_id) == null
        )

    def _copy_out(self, region: BlockRegion, ids: Sequence[int]) -> torch.Tensor:
        # Allocate host output directly; avoid a gathered GPU temporary that
        # could OOM precisely when swapping is needed to recover GPU capacity.
        data = torch.empty(
            (len(ids), region.blocks.shape[1]),
            dtype=region.blocks.dtype,
            device="cpu",
        )
        for row, block_id in enumerate(ids):
            data[row].copy_(region.blocks[block_id], non_blocking=False)
        return data

    def _capture_regions(self, key, tables, positions):
        copies = []
        for region in self.regions:
            pos = positions[region.group_id]
            ids = [tables[region.group_id][i] for i in pos]
            copies.append(
                _RegionCopy(
                    region.name, region.group_id, pos, self._copy_out(region, ids)
                )
            )
        return copies

    def _restore_regions(self, key, tables, copies):
        for region, saved in zip(self.regions, copies, strict=True):
            for row, position in enumerate(saved.logical_positions):
                block_id = tables[region.group_id][position]
                region.blocks[block_id].copy_(saved.data[row], non_blocking=False)

    def _release_storage(self, key):
        """Optional backend cleanup; must complete before ownership is retired."""

    def abort_capture(self, key):
        if key in self._checkpoints:
            raise ValueError("committed checkpoint requires explicit retirement")
        self._release_storage(key)

    def _verify_copies(self, tables, copies, phase: str) -> None:
        # Re-read one page at a time: verification must not duplicate an entire
        # long-context checkpoint in RAM or allocate a temporary on the GPU.
        self._synchronize_devices()
        for region, saved in zip(self.regions, copies, strict=True):
            actual = torch.empty(region.blocks.shape[1], dtype=region.blocks.dtype)
            for row, position in enumerate(saved.logical_positions):
                actual.copy_(
                    region.blocks[tables[region.group_id][position]],
                    non_blocking=False,
                )
                if not torch.equal(actual, saved.data[row]):
                    raise RuntimeError(
                        f"{phase} cache verification failed; do not resume"
                    )

    def capture(
        self,
        key: SessionKey,
        block_ids: Sequence[Sequence[int]],
        boundary: SwapBoundary,
        metadata: bytes,
    ) -> int:
        """Preserve a hot request, without freeing or modifying any hot bytes.

        Returns the admitted payload size. Metadata must include any state
        outside the cache tensors (acceptance count, pending draft state,
        sampler state, and token-position information). Its schema and coverage
        are the worker adapter's responsibility, not inferred by this store.
        """
        boundary.validate()
        if not isinstance(metadata, bytes):
            raise TypeError("worker metadata must be immutable bytes")
        if key in self._checkpoints:
            raise ValueError("session checkpoint already exists")
        tables = self._validate_tables(block_ids)
        positions = tuple(self._positions(ids, null=False) for ids in tables)
        size = len(metadata) + sum(
            len(positions[region.group_id]) * region.blocks.shape[1]
            for region in self.regions
        )
        if (
            len(self._checkpoints) >= self.max_sessions
            or size > self.capacity_bytes - self.used_bytes
        ):
            raise MemoryError("cold session budget exhausted; retain hot state")

        self._synchronize_devices()
        try:
            copies = self._capture_regions(key, tables, positions)
            counts = tuple(map(len, tables))
            nulls = tuple(self._positions(ids, null=True) for ids in tables)
            self._verify_copies(tables, copies, "captured")
            checkpoint = _Checkpoint(
                boundary,
                counts,
                nulls,
                metadata,
                tuple(copies),
                size,
                self._digest(key, boundary, counts, nulls, metadata, copies),
            )
        except Exception:
            self._release_storage(key)
            raise
        self._checkpoints[key] = checkpoint
        self.used_bytes += size
        return size

    def restore(
        self, key: SessionKey, block_ids: Sequence[Sequence[int]]
    ) -> tuple[SwapBoundary, bytes]:
        """Restore into reserved hot blocks; keep cold data until explicit drop.

        All structural checks precede writes. Copy failure retains the source
        but may partially modify the destination; the coordinator must not
        schedule those blocks and must retry or free them. A successful return
        acknowledges completed copies on this rank, not global readiness.
        """
        checkpoint = self._verified_checkpoint(key)
        tables = self._validate_tables(block_ids)
        if tuple(map(len, tables)) != checkpoint.block_counts:
            raise ValueError("restore block-table lengths differ from checkpoint")
        nulls = tuple(self._positions(ids, null=True) for ids in tables)
        if nulls != checkpoint.null_positions:
            raise ValueError("restore changed the logical null-block positions")

        self._synchronize_devices()
        self._restore_regions(key, tables, checkpoint.regions)
        # CPU-to-GPU copy ordering must be complete before the coordinator can
        # acknowledge this rank and make the request runnable on another stream.
        self._synchronize_devices()
        self._verify_copies(tables, checkpoint.regions, "restored")
        return checkpoint.boundary, checkpoint.metadata

    def verify_hot(self, key: SessionKey, block_ids: Sequence[Sequence[int]]) -> bytes:
        """Recheck original ownership bytes immediately before releasing them."""
        checkpoint = self._verified_checkpoint(key)
        tables = self._validate_tables(block_ids)
        if (
            tuple(map(len, tables)) != checkpoint.block_counts
            or tuple(self._positions(ids, null=True) for ids in tables)
            != checkpoint.null_positions
        ):
            raise ValueError("hot block layout changed before cold commit")
        self._verify_copies(tables, checkpoint.regions, "hot pre-commit")
        return checkpoint.metadata

    def drop(self, key: SessionKey) -> bool:
        """Retire cold data after global restore commit or explicit cancellation."""
        checkpoint = self._checkpoints.get(key)
        if checkpoint is None:
            return False
        self._release_storage(key)
        del self._checkpoints[key]
        self.used_bytes -= checkpoint.size_bytes
        return True


class NativeSessionColdStore(SessionColdStore):
    """Checkpoint-private native pools under the same aggregate byte budget.

    Each group's pool has exactly as many native chunks as this checkpoint
    needs. No other checkpoint can allocate or evict those chunks. This keeps
    whole-session ownership with the coordinator, without pretending that a
    long-lived checkpoint is an in-flight native load. Pools are released only
    on explicit retirement after native transfers drain. Pool churn is a known
    performance tradeoff; byte integrity and ownership precede pool sharing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._layouts = native_group_cache_layouts(
            self.regions,
            self.num_groups,
            empty_groups=kwargs.get("empty_groups", ()),
        )
        self._native_pools = {}
        self._storage_failed = False

    def _ensure_healthy(self):
        if self._storage_failed:
            raise RuntimeError("native session transfer unresolved; do not resume")

    def _verified_checkpoint(self, key):
        self._ensure_healthy()
        return super()._verified_checkpoint(key)

    @staticmethod
    def _finish(worker, job_id):
        worker.wait({job_id})
        results = worker.get_finished()
        if len(results) != 1 or results[0].job_id != job_id or not results[0].success:
            raise RuntimeError("native session transfer did not acknowledge completion")

    def capture(self, *args, **kwargs):
        self._ensure_healthy()
        return super().capture(*args, **kwargs)

    def _capture_regions(self, key, tables, positions):
        from vllm.v1.kv_offload.base import (
            GPULoadStoreSpec,
            ReqContext,
            make_offload_key,
        )
        from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
        from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

        pools = self._native_pools[key] = {}
        saved = {}
        for group_id, layout in enumerate(self._layouts):
            group_regions = [r for r in self.regions if r.group_id == group_id]
            pos = positions[group_id]
            if not pos or layout is None:
                for region in group_regions:
                    saved[region.name] = torch.empty(
                        (0, region.blocks.shape[1]), dtype=region.blocks.dtype
                    )
                continue
            context = ReqContext(str(key))
            keys = [make_offload_key(str(i).encode(), group_id) for i in pos]
            manager = CPUOffloadingManager(len(pos))
            worker = CPUOffloadingWorker(layout, 1, len(pos))
            pools[group_id] = (worker, manager, keys, context)
            pending = manager.prepare_store(keys, context)
            if pending is None or pending.keys_to_store != keys or pending.evicted_keys:
                raise RuntimeError("private native checkpoint allocation failed")
            if pending.store_spec.chunk_ids.tolist() != list(range(len(pos))):
                raise RuntimeError(
                    "private native checkpoint chunks are not contiguous"
                )
            ids = [tables[group_id][i] for i in pos]
            if not worker.submit_store(
                0, GPULoadStoreSpec(ids, [len(ids)], [0]), pending.store_spec
            ):
                raise RuntimeError("native session store submission failed")
            self._finish(worker, 0)
            manager.complete_store(keys, context)
            # Version-pinned adapter to the unmodified native worker. These
            # views are its authoritative CPU storage, not a second copy.
            buffers = worker._store_handler.dst_tensors
            if len(buffers) != len(group_regions):
                raise RuntimeError("native CPU layout differs from registered regions")
            for region, data in zip(group_regions, buffers, strict=True):
                if (
                    data.device.type != "cpu"
                    or not data.is_pinned()
                    or not data.is_contiguous()
                    or tuple(data.shape) != (len(pos), region.blocks.shape[1])
                ):
                    raise RuntimeError("native CPU page storage is incompatible")
                saved[region.name] = data.view(region.blocks.dtype)
        return [
            _RegionCopy(r.name, r.group_id, positions[r.group_id], saved[r.name])
            for r in self.regions
        ]

    def _restore_regions(self, key, tables, copies):
        from vllm.v1.kv_offload.base import GPULoadStoreSpec

        by_group = {copy.group_id: copy.logical_positions for copy in copies}
        try:
            for group_id, (worker, manager, keys, context) in self._native_pools[
                key
            ].items():
                ids = [tables[group_id][i] for i in by_group[group_id]]
                lease = manager.prepare_load(keys, context)
                try:
                    if not worker.submit_load(
                        1, lease, GPULoadStoreSpec(ids, [len(ids)], [0])
                    ):
                        raise RuntimeError("native session load submission failed")
                    self._finish(worker, 1)
                except Exception:
                    # An acknowledgement can fail after DMA was submitted.
                    # Drain before releasing its native read lease or allowing
                    # coordinator rollback to recycle destination blocks.
                    self._storage_failed = True
                    self.quiesce()
                    worker.get_finished()
                    manager.complete_load(keys, context)
                    self._storage_failed = False
                    raise
                manager.complete_load(keys, context)
        except Exception:
            # Cold storage is retained whether recovery fencing succeeded or
            # failed; only an unresolved fence poisons subsequent operations.
            raise

    def _release_storage(self, key):
        pools = self._native_pools.get(key, {})
        try:
            if pools:
                self.quiesce()
            for group_id, (worker, manager, _, _) in list(pools.items()):
                worker.shutdown()
                manager.reset_cache()
                del pools[group_id]
        except Exception:
            self._storage_failed = True
            raise
        self._native_pools.pop(key, None)
