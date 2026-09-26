# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for contiguous KV cache packing.

Every cache group packs its layers densely into one block; groups overlay each other
(a block ID is owned by one group at a time), so the packed block stride is the
largest group's packing. The layout decides whether the layer dim sits outside the
block dim (a contiguous region per layer) or inside it (all layers' pages within each
block); the allocation is the same either way.
"""

from collections import deque
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import CacheConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
    get_offloading_group_ids,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    get_sliding_window_size_in_chunks,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    _get_kv_cache_bytes_per_block,
    _get_packed_kv_cache_groups,
    _pool_bytes_per_block,
    _project_kv_cache_groups_to_worker,
    generate_scheduler_kv_cache_config,
    get_direct_host_kv_cache_configs,
    get_kv_cache_config_from_groups,
    get_kv_cache_configs,
    get_kv_cache_groups,
    get_request_block_hasher,
    init_none_hash,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    DirectHostAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupRole,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
    get_direct_host_cache_options,
    iter_layer_specs,
    replace_as,
)
from vllm.v1.kv_offload.base import LookupResult, ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.request import Request
from vllm.v1.worker.utils import _allocate_kv_cache, allocate_kv_cache

MEMORY = 8 * 1024 * 1024


def _mla(head_size: int) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=64, num_kv_heads=1, head_size=head_size, dtype=torch.uint8
    )


def _full() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float16
    )


def _uniform_group(specs: dict) -> KVCacheGroupSpec:
    return KVCacheGroupSpec(
        list(specs),
        UniformTypeKVCacheSpecs(block_size=64, kv_cache_specs=specs),
    )


def _mixed_page_groups(n_mla=3, n_idx=3, n_swa=5):
    """A DeepSeek-V4-style hybrid: two groups with different page mixes."""
    g1 = {f"mla.{i}": _mla(512) for i in range(n_mla)}
    g1.update({f"idx.{i}": _mla(128) for i in range(n_idx)})
    g2 = {f"swa.{i}": _mla(512) for i in range(n_swa)}
    return [_uniform_group(g1), _uniform_group(g2)], g1, g2


def _mock_vllm_config(layout: str | None):
    config = MagicMock()
    config.cache_config = CacheConfig()
    config.cache_config.num_gpu_blocks_override = None
    config.cache_config.kv_cache_layout = layout
    config.attention_config.hisparse_config = None
    config.kv_transfer_config = None
    return config


def _pages(groups) -> dict[str, int]:
    return {
        name: group.kv_cache_spec.kv_cache_specs[name].page_size_bytes
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        else group.kv_cache_spec.page_size_bytes
        for group in groups
        for name in group.layer_names
    }


def _expected_bytes_per_block(groups) -> int:
    pages = _pages(groups)
    return max(sum(pages[n] for n in g.layer_names) for g in groups)


def _bind(config, layout: str):
    return allocate_kv_cache(config, torch.device("cpu"), KVCacheLayout[layout], None)


MAIN_KV_PAGE_BYTES = 2_048
COMPRESSED_PAGE_BYTES = 128
NUM_CACHE_TUPLES = 3
BYTES_PER_BLOCK = NUM_CACHE_TUPLES * (MAIN_KV_PAGE_BYTES + COMPRESSED_PAGE_BYTES)


def _main_kv_name(layer_index: int) -> str:
    return f"model.layers.{layer_index}.self_attn"


def _compressed_name(layer_index: int) -> str:
    return f"model.layers.{layer_index}.self_attn.indexer.compressed_key_cache"


def _compressor_state_name(layer_index: int) -> str:
    return f"model.layers.{layer_index}.self_attn.indexer.raw_key_cache"


def _make_csa_linear_specs(
    num_mamba: int = 7,
    num_tuples: int = NUM_CACHE_TUPLES,
) -> dict[str, KVCacheSpec]:
    specs: dict[str, KVCacheSpec] = {}
    for layer_index in range(num_mamba):
        specs[f"model.layers.{layer_index}.linear_attn"] = MambaSpec(
            block_size=16,
            shapes=((32,),),
            dtypes=(torch.bfloat16,),
        )
    specs["model.layers.0.ple"] = MambaSpec(
        block_size=16,
        shapes=((24,),),
        dtypes=(torch.bfloat16,),
        tp_replicated=True,
    )
    for layer_index in range(num_tuples):
        specs[_main_kv_name(layer_index)] = FullAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=16,
            head_size_v=16,
            dtype=torch.bfloat16,
        )
        specs[_compressed_name(layer_index)] = MLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=16,
            dtype=torch.bfloat16,
            tokens_per_state=4,
        )
        specs[_compressor_state_name(layer_index)] = CircularBufferSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=8,
            head_size_v=0,
            dtype=torch.bfloat16,
        )
    return specs


def _shared_layout_config():
    config = _mock_vllm_config("BLNHC")
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    config.speculative_config = None
    config.model_config.max_model_len = 16
    config.model_config.get_total_num_hidden_layers.return_value = 64
    config.model_config.get_total_num_kv_heads.return_value = 2
    config.model_config.get_num_kv_heads.return_value = 2
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 1
    config.cache_config.block_size = 16
    config.cache_config.enable_prefix_caching = False
    config.cache_config.prefix_match_unit = None
    config.cache_config.mamba_cache_mode = "none"
    return config


class TestDirectHostPipelinePacking:
    """Keep shared logical IDs but allocate only each stage's owned pages."""

    def test_deferred_free_keeps_host_owner_and_starts_ttl_after_fence(
        self, monkeypatch
    ):
        """A flattened deferred list must not put RAM pages in the GPU free queue."""
        now = [0.0]
        monkeypatch.setattr("vllm.v1.hisparse.block_pool.monotonic", lambda: now[0])
        init_none_hash(sha256)
        config = KVCacheConfig(
            num_blocks=5,
            direct_host_num_blocks=7,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(["device"], _full()),
                KVCacheGroupSpec(
                    ["host"],
                    _full(),
                    host_resident=True,
                    role=KVCacheGroupRole.DIRECT_HOST,
                ),
            ],
        )
        manager = KVCacheManager(
            config, max_model_len=64, scheduler_block_size=16, hash_block_size=16
        )
        host = manager.coordinator.single_type_managers[1].block_pool
        pools = (manager.block_pool, host)
        capacity = [pool.get_num_free_blocks() for pool in pools]
        request = Request(
            "inflight",
            list(range(49)),
            SamplingParams(max_tokens=1),
            None,
            block_hasher=get_request_block_hasher(16, sha256),
        )
        assert manager.allocate_slots(request, 32) is not None
        request.num_computed_tokens = 32
        manager.cache_blocks(request, 32)
        pages = manager.get_blocks(request.request_id).blocks
        assert [page.block_id for page in pages[0]] == [
            page.block_id for page in pages[1]
        ]
        request.last_sched_seq = 2
        scheduler = object.__new__(Scheduler)
        scheduler.kv_cache_manager = manager
        scheduler.defer_block_free = True
        scheduler.sched_step_seq = 2
        scheduler.processed_step_seq = 0
        scheduler.deferred_frees = deque()
        scheduler._free_request_blocks(request)
        held_capacity = [pool.get_num_free_blocks() for pool in pools]
        now[0] = 7200
        scheduler.processed_step_seq = 1
        scheduler._drain_deferred_frees()
        assert len(scheduler.deferred_frees) == 1
        assert host.expire_idle() == 0
        assert all(page.ref_cnt == 1 for group in pages for page in group)
        assert [pool.get_num_free_blocks() for pool in pools] == held_capacity
        scheduler.processed_step_seq = 2
        scheduler._drain_deferred_frees()
        assert not scheduler.deferred_frees
        assert [pool.get_num_free_blocks() for pool in pools] == capacity
        assert all(page.ref_cnt == 0 for group in pages for page in group)
        now[0] = 10799
        assert host.expire_idle() == 0
        now[0] = 10800
        assert host.expire_idle() == 2
        for pool in pools:
            allocated = pool.get_new_blocks(pool.get_num_free_blocks())
            assert all(page.pool is pool for page in allocated)
            assert all(pool.blocks[page.block_id] is page for page in allocated)
            pool.free_blocks(allocated)

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_native_cold_capacity_counts_packed_gpu_blocks_not_aliased_layers(
        self, device
    ):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("Native DMA needs CUDA")
        config, _, _, small = self.setup()
        stride = 2 * small.page_size_bytes
        state = MambaSpec(
            block_size=16,
            shapes=((4, 64), (32, 64)),
            dtypes=(torch.bfloat16, torch.float32),
            page_size_padded=stride,
        )
        groups = (
            [KVCacheGroupSpec(["a", "b"], small)]
            + [KVCacheGroupSpec([f"mamba.{i}"], state) for i in range(3)]
            + [
                KVCacheGroupSpec(
                    ["scratch"],
                    CircularBufferSpec(
                        block_size=4,
                        num_kv_heads=1,
                        head_size=8,
                        dtype=torch.bfloat16,
                        replay_alignment=4,
                    ),
                ),
                KVCacheGroupSpec(
                    ["host"],
                    small,
                    host_resident=True,
                    enable_kv_transfer=False,
                    role=KVCacheGroupRole.DIRECT_HOST,
                ),
            ]
        )
        specs = {
            name: group.kv_cache_spec for group in groups for name in group.layer_names
        }
        plan = get_direct_host_kv_cache_configs(
            config,
            groups,
            [specs],
            [MEMORY],
            host_num_blocks=7,
            host_max_bytes=MEMORY,
        )[0]
        assert plan.direct_host_offload_bytes_per_block == stride

        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
            OffloadingConnectorWorker,
        )

        config.parallel_config.tensor_parallel_size = 1
        config.parallel_config.cp_kv_cache_interleave_size = 1
        spec = MagicMock()
        spec.replicated_layout = False
        worker = OffloadingConnectorWorker(spec, config, plan)
        # No pinned allocation is needed to verify device-only registration.
        device_plan = replace(
            plan,
            kv_cache_tensors=[t for t in plan.kv_cache_tensors if not t.host_resident],
        )
        caches = _allocate_kv_cache(
            device_plan, torch.device(device), KVCacheLayout.BLNHC
        )
        caches["host"] = torch.full((7, 16, 2, 128), 17, dtype=torch.bfloat16)
        worker.register_kv_caches(caches)
        canonical = spec.get_worker.call_args.args[0]
        assert len(canonical.tensors) == 1
        assert canonical.tensors[0].page_size_bytes == stride
        assert len(canonical.group_data_refs) == 4
        assert all(
            len(refs) == 1 and refs[0].page_size_bytes == stride
            for refs in canonical.group_data_refs
        )
        spec.get_worker.reset_mock()
        with pytest.raises(ValueError, match="do not share"):
            worker.register_kv_caches({**caches, "b": caches["b"].clone()})
        spec.get_worker.assert_not_called()

        if device == "cuda":
            from vllm.v1.kv_offload.base import GPULoadStoreSpec
            from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
            from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker

            backing = canonical.tensors[0].tensor
            generator = torch.Generator(device=device).manual_seed(173)
            backing.copy_(
                torch.randint(
                    0,
                    127,
                    backing.shape,
                    device=device,
                    dtype=torch.int8,
                    generator=generator,
                )
            )
            expected = backing.clone()
            transfer = CPUOffloadingWorker(
                canonical, blocks_per_chunk=1, num_cpu_chunks=16
            )
            try:
                source = GPULoadStoreSpec(
                    [1, 3, 5, 7], group_sizes=(1, 1, 1, 1), block_indices=(0, 0, 0, 0)
                )
                cold = CPULoadStoreSpec([2, 4, 6, 8])
                assert transfer.submit_store(1, source, cold)
                transfer.wait({1})
                target = GPULoadStoreSpec(
                    [2, 4, 6, 8], group_sizes=(1, 1, 1, 1), block_indices=(0, 0, 0, 0)
                )
                assert transfer.submit_load(2, cold, target)
                transfer.wait({2})
                expected[[2, 4, 6, 8]] = expected[[1, 3, 5, 7]]
                assert torch.equal(backing, expected)
                assert bool((caches["host"] == 17).all())
            finally:
                transfer.shutdown()

        config.cache_config.kv_cache_layout = "LBNHC"
        groups = [
            replace(group, kv_cache_spec=small)
            for group in groups
            if "scratch" not in group.layer_names
        ]
        specs = {name: small for name in specs if name != "scratch"}
        fallback = get_direct_host_kv_cache_configs(
            config,
            groups,
            [specs],
            [MEMORY],
            host_num_blocks=7,
            host_max_bytes=MEMORY,
        )[0]
        assert fallback.direct_host_offload_packed_stride is None
        assert fallback.direct_host_offload_bytes_per_block == 5 * small.page_size_bytes

    def test_uneven_pp_workers_share_one_native_cpu_chunk_geometry(self):
        config, _, _, small = self.setup()
        config.kv_transfer_config = SimpleNamespace(
            engine_id="pp-geometry",
            kv_connector_extra_config={
                "cpu_bytes_to_use": MEMORY,
            },
        )
        config.kv_events_config = None
        config.parallel_config.world_size = 4
        config.parallel_config.tensor_parallel_size = 2
        config.parallel_config.pipeline_parallel_size = 2
        config.parallel_config.prefill_context_parallel_size = 1
        config.model_config.dtype = torch.float16
        groups = [
            KVCacheGroupSpec(["gpu.0", "gpu.1", "gpu.2"], small),
            KVCacheGroupSpec(
                ["host.0", "host.1"],
                small,
                host_resident=True,
                enable_kv_transfer=False,
                role=KVCacheGroupRole.DIRECT_HOST,
            ),
        ]
        stage0 = {"gpu.0": small, "host.0": small}
        stage1 = {"gpu.1": small, "gpu.2": small, "host.1": small}
        plans = get_direct_host_kv_cache_configs(
            config,
            groups,
            [stage0, stage0, stage1, stage1],
            [MEMORY] * 4,
            host_num_blocks=7,
            host_max_bytes=MEMORY,
        )
        scheduler = generate_scheduler_kv_cache_config(plans)
        specs = [
            CPUOffloadingSpec(build_offloading_config(config, plan))
            for plan in [*plans, scheduler]
        ]
        geometry = {
            (spec.num_chunks, spec.kv_bytes_per_chunk, spec.cpu_page_size_per_worker)
            for spec in specs
        }
        assert len(geometry) == 1, geometry
        plans[-1].direct_host_offload_bytes_per_block = small.page_size_bytes
        with pytest.raises(ValueError, match="cannot fit"):
            build_offloading_config(config, plans[-1])
        with pytest.raises(AssertionError):
            generate_scheduler_kv_cache_config(plans)

    @pytest.mark.parametrize("host_first", [False, True])
    @pytest.mark.parametrize("draft", [False, True])
    def test_cold_device_hit_reuses_resident_host_pages(self, host_first, draft):
        """A device miss must not hide RAM pages needed by an external hit."""
        init_none_hash(sha256)
        device = KVCacheGroupSpec(["device"], replace(_mla(128), block_size=16))
        host = KVCacheGroupSpec(
            ["host"],
            _full(),
            is_eagle_group=draft,
            host_resident=True,
            role=KVCacheGroupRole.DIRECT_HOST,
        )
        config = KVCacheConfig(
            num_blocks=16,
            direct_host_num_blocks=16,
            kv_cache_tensors=[],
            kv_cache_groups=[host, device] if host_first else [device, host],
        )
        host_id = 0 if host_first else 1
        manager = KVCacheManager(
            config, max_model_len=64, scheduler_block_size=16, hash_block_size=16
        )
        hasher = get_request_block_hasher(16, sha256)
        request = Request(
            "source",
            list(range(49)),
            SamplingParams(max_tokens=1),
            None,
            block_hasher=hasher,
        )
        assert manager.allocate_slots(request, 48) is not None
        original = manager.get_block_ids(request.request_id)[host_id][:]
        request.num_computed_tokens = 48
        manager.cache_blocks(request, 48)
        manager.free(request)
        pool = manager.block_pool
        evicted = pool.get_new_blocks(pool.get_num_free_blocks())
        pool.free_blocks(evicted)
        repeat = Request(
            "repeat",
            list(range(49)),
            SamplingParams(max_tokens=1),
            None,
            block_hasher=hasher,
        )
        connector = object.__new__(OffloadingConnector)
        connector._kv_cache_config = config
        connector._kv_cache_manager = manager
        boundary = 32 if draft else 48
        assert connector._max_loadable_tokens(repeat, 0) == boundary
        hits, local, _ = manager.get_computed_blocks(repeat)
        assert local == 0
        assert [b.block_id for b in hits.blocks[host_id]] == original[: boundary // 16]
        assert (
            manager.allocate_slots(
                repeat,
                0,
                new_computed_blocks=hits,
                num_external_computed_tokens=boundary,
            )
            is not None
        )
        assert (
            manager.get_block_ids(repeat.request_id)[host_id]
            == original[: boundary // 16]
        )
        manager.free(repeat)

    @pytest.mark.parametrize("host_capacity", [4, 8])
    @pytest.mark.parametrize("restored_tokens", [0, 16, 32])
    def test_host_hits_beyond_restored_prefix_are_not_writable(
        self, host_capacity, restored_tokens
    ):
        """Recomputed suffixes need private pages, including admission accounting."""
        init_none_hash(sha256)
        config = KVCacheConfig(
            num_blocks=16,
            direct_host_num_blocks=host_capacity,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(["device"], replace(_mla(128), block_size=16)),
                KVCacheGroupSpec(
                    ["host"],
                    _full(),
                    host_resident=True,
                    role=KVCacheGroupRole.DIRECT_HOST,
                ),
            ],
        )
        manager = KVCacheManager(
            config, max_model_len=64, scheduler_block_size=16, hash_block_size=16
        )
        hasher = get_request_block_hasher(16, sha256)
        source = Request(
            "source",
            list(range(49)),
            SamplingParams(max_tokens=1),
            None,
            block_hasher=hasher,
        )
        assert manager.allocate_slots(source, 48) is not None
        source.num_computed_tokens = 48
        manager.cache_blocks(source, 48)
        host = manager.coordinator.single_type_managers[1]
        retained = list(host.req_to_blocks[source.request_id])
        # Pin the source pages independently, as another reader would do.
        host.block_pool.touch(retained)
        manager.free(source)
        evicted = manager.block_pool.get_new_blocks(
            manager.block_pool.get_num_free_blocks()
        )
        manager.block_pool.free_blocks(evicted)
        repeat = Request(
            "repeat",
            list(range(49)),
            SamplingParams(max_tokens=1),
            None,
            block_hasher=hasher,
        )
        hits, local, _ = manager.get_computed_blocks(repeat)
        assert local == 0
        assert len(hits.blocks[1]) == 3
        allocated = manager.allocate_slots(
            repeat,
            16,
            new_computed_blocks=hits,
            num_external_computed_tokens=restored_tokens,
        )
        if host_capacity == 4:
            assert allocated is None
        else:
            assert allocated is not None
            ids = manager.get_block_ids(repeat.request_id)[1]
            boundary = restored_tokens // 16
            assert ids[:boundary] == [b.block_id for b in retained[:boundary]]
            assert len(ids) == boundary + 1
            assert ids[-1] not in [b.block_id for b in retained]
            assert retained[boundary].ref_cnt == 1
        manager.free(repeat)
        host.block_pool.free_blocks(retained)

    def test_expired_host_prefix_bounds_an_otherwise_live_cold_copy(self, monkeypatch):
        now = [0.0]
        monkeypatch.setattr("vllm.v1.hisparse.block_pool.monotonic", lambda: now[0])
        monkeypatch.setattr("vllm.v1.kv_offload.cpu.manager.monotonic", lambda: now[0])
        init_none_hash(sha256)
        config = KVCacheConfig(
            num_blocks=4,
            kv_cache_tensors=[],
            direct_host_num_blocks=4,
            kv_cache_groups=[
                KVCacheGroupSpec(["device"], _full()),
                KVCacheGroupSpec(
                    ["host"],
                    _full(),
                    host_resident=True,
                    role=KVCacheGroupRole.DIRECT_HOST,
                ),
            ],
        )
        manager = KVCacheManager(
            config, max_model_len=64, scheduler_block_size=16, hash_block_size=16
        )
        request = Request(
            "prefix",
            list(range(33)),
            SamplingParams(max_tokens=4),
            None,
            block_hasher=get_request_block_hasher(16, sha256),
        )
        assert manager.allocate_slots(request, 32) is not None
        request.num_computed_tokens = 32
        manager.cache_blocks(request, 32)
        manager.free(request)
        # Exercise the connector's actual RAM-coverage bound, not a mock of it.
        connector = object.__new__(OffloadingConnector)
        connector._kv_cache_config = config
        connector._kv_cache_manager = manager
        assert connector._max_loadable_tokens(request, 0) == 32
        ctx = ReqContext(req_id="prefix")
        cold = CPUOffloadingManager(2, idle_ttl_seconds=7200)
        keys = [make_offload_key(h, 0) for h in request.block_hashes]
        assert cold.prepare_store(keys, ctx) is not None
        cold.complete_store(keys, ctx)
        now[0] = 3600
        assert all(cold.lookup(key, ctx) == LookupResult.HIT for key in keys)
        assert connector._max_loadable_tokens(request, 0) == 0
        assert manager.get_computed_blocks(request)[1] == 0
        assert (
            manager.coordinator.single_type_managers[1].block_pool.get_num_free_blocks()
            == 3
        )

    @pytest.mark.parametrize("branches", [2, 6])
    def test_shared_host_prefix_branches_release_and_expire_independently(
        self, monkeypatch, branches
    ):
        """Live branches pin shared pages; idle TTL cannot retire another reader."""
        now = [0.0]
        monkeypatch.setattr("vllm.v1.hisparse.block_pool.monotonic", lambda: now[0])
        init_none_hash(sha256)
        config = KVCacheConfig(
            num_blocks=32,
            direct_host_num_blocks=branches + 3,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(["device"], _full()),
                KVCacheGroupSpec(
                    ["host"],
                    _full(),
                    host_resident=True,
                    role=KVCacheGroupRole.DIRECT_HOST,
                ),
            ],
        )
        manager = KVCacheManager(
            config, max_model_len=64, scheduler_block_size=16, hash_block_size=16
        )
        hasher = get_request_block_hasher(16, sha256)

        def request(name, suffix):
            return Request(
                name,
                list(range(32)) + [suffix] * 17,
                SamplingParams(max_tokens=1),
                None,
                block_hasher=hasher,
            )

        source = request("source", 100)
        assert manager.allocate_slots(source, 32) is not None
        source.num_computed_tokens = 32
        manager.cache_blocks(source, 32)
        host = manager.coordinator.single_type_managers[1]
        shared = list(host.req_to_blocks[source.request_id])
        manager.free(source)
        device_pages = manager.block_pool.get_new_blocks(
            manager.block_pool.get_num_free_blocks()
        )
        manager.block_pool.free_blocks(device_pages)
        active, suffix_ids = [], set()
        for i in range(branches):
            branch = request(str(i), 200 + i)
            hits, local, _ = manager.get_computed_blocks(branch)
            assert (
                manager.allocate_slots(
                    branch,
                    16,
                    num_new_computed_tokens=local,
                    new_computed_blocks=hits,
                    num_external_computed_tokens=32 - local,
                )
                is not None
            )
            pages = host.req_to_blocks[branch.request_id]
            assert pages[:2] == shared
            assert pages[2].block_id not in suffix_ids
            suffix_ids.add(pages[2].block_id)
            branch.num_computed_tokens = 48
            manager.cache_blocks(branch, 48)
            active.append(branch)
        assert all(page.ref_cnt == branches for page in shared)
        assert host.block_pool.get_num_free_blocks() == 0
        overflow = request("overflow", 300)
        hits, local, _ = manager.get_computed_blocks(overflow)
        assert (
            manager.allocate_slots(
                overflow,
                16,
                num_new_computed_tokens=local,
                new_computed_blocks=hits,
                num_external_computed_tokens=32 - local,
            )
            is None
        )
        manager.free(overflow)
        now[0] = 3601
        assert host.block_pool.expire_idle() == 0
        retired = min(2, branches - 1)
        for branch in active[:retired]:
            manager.free(branch)
        now[0] += 3600
        assert host.block_pool.expire_idle() == retired
        assert all(page.ref_cnt == branches - retired for page in shared)
        assert all(page.block_hash is not None for page in shared)
        for branch in active[retired:]:
            manager.free(branch)
        assert host.block_pool.get_num_free_blocks() == branches + 2
        now[0] += 3600
        assert host.block_pool.expire_idle() == branches - retired + 2
        assert all(page.ref_cnt == 0 and page.block_hash is None for page in shared)
        assert manager.get_computed_blocks(request("expired", 400))[1] == 0

    @staticmethod
    def setup():
        config = _shared_layout_config()
        config.model_config.original_max_model_len = 32
        config.model_config.max_model_len = 32
        small = _full()
        large = replace(small, head_size=128, head_size_v=128)
        groups = [
            KVCacheGroupSpec(["gpu.0", "gpu.1"], small),
            KVCacheGroupSpec(
                ["host.0", "host.1", "host.2"],
                UniformTypeKVCacheSpecs(
                    block_size=16,
                    kv_cache_specs={"host.0": small, "host.1": large, "host.2": small},
                ),
                is_eagle_group=True,
                host_resident=True,
                enable_kv_transfer=False,
                role=KVCacheGroupRole.DIRECT_HOST,
            ),
        ]
        workers = [
            {"gpu.0": small, "host.0": small},
            {"gpu.1": small, "host.1": large, "host.2": small},
        ]
        return config, groups, workers, small

    @staticmethod
    def enable_startup(config, blocks=7, budget=MEMORY):
        config.additional_config = {
            "flash_next_direct_host_kv": {"num_blocks": blocks, "max_bytes": budget}
        }
        config.model_config.enforce_eager = True
        config.parallel_config.data_parallel_size = 1

    def test_model_specs_select_native_planner_and_keep_target_draft_groups_separate(
        self,
    ):
        config, _, workers, small = self.setup()
        self.enable_startup(config)
        for worker in workers:
            for name, spec in list(worker.items()):
                if name.startswith("host."):
                    worker[name] = replace_as(
                        spec, DirectHostAttentionSpec, is_mtp_draft=name == "host.2"
                    )
        plans = get_kv_cache_configs(config, workers, [MEMORY, MEMORY // 2])
        scheduler = generate_scheduler_kv_cache_config(plans)
        host_groups = [g for g in scheduler.kv_cache_groups if g.host_resident]
        assert len(host_groups) == 2
        assert [g.is_eagle_group for g in host_groups] == [False, True]
        assert all(g.role is KVCacheGroupRole.DIRECT_HOST for g in host_groups)
        assert all(type(g.kv_cache_spec) is FullAttentionSpec for g in host_groups)
        assert scheduler.host_num_blocks == 7
        assert resolve_kv_cache_block_sizes(scheduler, config) == (16, 16)
        assert all(
            plan.num_blocks == MEMORY // (2 * small.page_size_bytes) for plan in plans
        )

    def test_qsa_publishes_real_host_geometry_only_with_explicit_opt_in(self):
        # Import through model first: model/qsa have an existing circular import.
        from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpQSAAttention

        config, _, _, _ = self.setup()
        config.additional_config = {}
        config.cache_config.block_size = 3504
        layer = SimpleNamespace(
            _qsa_kv_offload=True,
            num_kv_heads=1,
            head_dim=256,
            kv_cache_torch_dtype=torch.bfloat16,
            kv_cache_dtype="auto",
            _qsa_is_mtp_draft=True,
        )
        legacy = Qwen4ExpQSAAttention.get_kv_cache_spec(layer, config)
        assert type(legacy) is FullAttentionSpec
        assert legacy.page_size_bytes == 7008
        self.enable_startup(config)
        real = Qwen4ExpQSAAttention.get_kv_cache_spec(layer, config)
        assert isinstance(real, DirectHostAttentionSpec)
        assert real.is_mtp_draft
        assert real.block_size == 3504
        assert real.page_size_bytes == 3588096
        assert real.num_head_slots is None and real.state_content_bytes is None
        layer._qsa_kv_offload = False
        with pytest.raises(ValueError, match="VLLM_QSA_KV_OFFLOAD"):
            Qwen4ExpQSAAttention.get_kv_cache_spec(layer, config)

    @pytest.mark.parametrize("invalid", ["bool", "extra", "eager", "dp", "hybrid"])
    def test_startup_rejects_ambiguous_or_unsupported_opt_in(self, invalid):
        config, _, _, _ = self.setup()
        self.enable_startup(config)
        options = config.additional_config["flash_next_direct_host_kv"]
        if invalid == "bool":
            options["num_blocks"] = True
        elif invalid == "extra":
            options["max_byte"] = MEMORY
        elif invalid == "eager":
            config.model_config.enforce_eager = False
        elif invalid == "dp":
            config.parallel_config.data_parallel_size = 2
        else:
            config.scheduler_config.disable_hybrid_kv_cache_manager = True
        with pytest.raises(ValueError):
            get_direct_host_cache_options(config)

    def test_host_groups_cannot_silently_enter_gpu_only_profiling_allocator(self):
        config, groups, _, _ = self.setup()
        with pytest.raises(ValueError, match="independent host/device planner"):
            get_kv_cache_config_from_groups(config, groups, MEMORY)

    def test_opted_in_graphs_keep_the_independent_pool_plan(self):
        config, _, workers, _ = self.setup()
        self.enable_startup(config)
        for worker in workers:
            for name, spec in list(worker.items()):
                if name.startswith("host."):
                    worker[name] = replace_as(spec, DirectHostAttentionSpec)
        eager = get_kv_cache_configs(config, workers, [MEMORY, MEMORY // 2])
        config.model_config.enforce_eager = False
        config.cache_config.kv_cache_memory_bytes = MEMORY
        config.compilation_config = SimpleNamespace(
            mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY
        )
        config.additional_config["flash_next_direct_host_kv"]["allow_cudagraph"] = True
        graph = get_kv_cache_configs(config, workers, [MEMORY, MEMORY // 2])
        assert graph == eager

    @pytest.mark.parametrize(
        "invalid",
        [
            "auto",
            "zero",
            "negative",
            "bool_budget",
            "override",
            "compile",
            "piecewise",
            "full",
            "no_graph",
            "nonbool_opt_in",
        ],
    )
    def test_graph_opt_in_cannot_enter_unsupported_profiling_or_compile_paths(
        self, invalid
    ):
        config, _, _, _ = self.setup()
        self.enable_startup(config)
        config.model_config.enforce_eager = False
        config.cache_config.kv_cache_memory_bytes = MEMORY
        config.compilation_config = SimpleNamespace(
            mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY
        )
        options = config.additional_config["flash_next_direct_host_kv"]
        options["allow_cudagraph"] = True
        if invalid in ("auto", "zero", "negative", "bool_budget", "override"):
            config.cache_config.kv_cache_memory_bytes = {
                "auto": None,
                "zero": 0,
                "negative": -1,
                "bool_budget": True,
                "override": None,
            }[invalid]
            config.cache_config.num_gpu_blocks_override = 32
        elif invalid == "compile":
            config.compilation_config.mode = CompilationMode.VLLM_COMPILE
        elif invalid == "nonbool_opt_in":
            options["allow_cudagraph"] = 1
        else:
            config.compilation_config.cudagraph_mode = {
                "piecewise": CUDAGraphMode.PIECEWISE,
                "full": CUDAGraphMode.FULL,
                "no_graph": CUDAGraphMode.NONE,
            }[invalid]
        with pytest.raises(ValueError):
            get_direct_host_cache_options(config)

    def test_aligned_ring_replay_is_excluded_from_native_cpu_offload(self):
        from vllm.models.qwen4_exp.common.qsa_cache import QSAKeyStateCache

        config, _, _, small = self.setup()
        self.enable_startup(config)
        config.num_speculative_tokens = 2
        config.attention_config.resolve_indexer_kv_dtype.return_value = "bf16"
        config.cache_config.enable_prefix_caching = True
        config.model_config.use_mla = False
        config.use_v2_model_runner = True
        config.kv_transfer_config = SimpleNamespace(
            engine_id="ring-replay-test", kv_connector_extra_config={}
        )
        owner = SimpleNamespace(
            compress_ratio=4,
            cache_config=config.cache_config,
            head_size=128,
            dtype=torch.bfloat16,
        )
        ring = QSAKeyStateCache.get_kv_cache_spec(owner, config)
        assert ring.replay_alignment == 4 and ring.block_size == 8
        assert not ring.prefix_cacheable
        specs = {
            "qsa": replace_as(small, DirectHostAttentionSpec),
            "compressed": MLAAttentionSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=128,
                dtype=torch.bfloat16,
                tokens_per_state=4,
            ),
            "ring": ring,
            "gdn": MambaSpec(
                block_size=16,
                shapes=((2, 4), (2, 2, 4)),
                dtypes=(torch.bfloat16, torch.float32),
                mamba_cache_mode="align",
                num_speculative_blocks=2,
            ),
        }
        plans = get_kv_cache_configs(config, [specs], [MEMORY])
        scheduler = generate_scheduler_kv_cache_config(plans)
        selected = get_offloading_group_ids(scheduler)
        names = {
            name for i in selected for name in scheduler.kv_cache_groups[i].layer_names
        }
        assert names == {"compressed", "gdn"}
        offload = build_offloading_config(config, scheduler)
        assert {group.group_id for group in offload.groups} == set(selected)
        assert offload.cache.tokens_per_hash == 16
        assert offload.extra_config["idle_ttl_seconds"] == 3600
        original_extra = config.kv_transfer_config.kv_connector_extra_config
        assert "idle_ttl_seconds" not in original_extra
        original_extra["idle_ttl_seconds"] = 120
        assert (
            build_offloading_config(config, scheduler).extra_config["idle_ttl_seconds"]
            == 120
        )
        for i in selected:
            get_sliding_window_size_in_chunks(
                scheduler.kv_cache_groups[i].kv_cache_spec, 16
            )
        # Fine-grained Mamba hits must not expose a mid-compression prefix.
        # Remove compressed-K so this exercises the ring's own guard, not the
        # existing tokens_per_state guard of the compressed cache.
        ring_only_guard = replace(
            scheduler,
            kv_cache_groups=[
                group
                for group in scheduler.kv_cache_groups
                if "compressed" not in group.layer_names
            ],
        )
        config.cache_config.prefix_match_unit = 2
        with pytest.raises(ValueError, match="per-state compression"):
            resolve_kv_cache_block_sizes(ring_only_guard, config)

    def test_unqualified_ring_remains_on_legacy_path_and_cannot_merge_with_replay_ring(
        self,
    ):
        _, groups, _, small = self.setup()
        ordinary = CircularBufferSpec(
            block_size=8, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
        )
        replay = replace(ordinary, replay_alignment=4)
        assert (
            UniformTypeKVCacheSpecs.from_specs({"old": ordinary, "new": replay}) is None
        )
        assert (
            UniformTypeKVCacheSpecs.from_specs({"new": replay, "old": ordinary}) is None
        )
        # Unqualified circular state must not be silently omitted from transfers.
        from vllm.v1.kv_cache_interface import KVCacheConfig

        cache = KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(["ordinary"], ordinary),
                KVCacheGroupSpec(["replay"], replay),
                KVCacheGroupSpec(["gpu"], small),
                groups[1],
            ],
        )
        assert get_offloading_group_ids(cache) == (0, 2)

    @pytest.mark.parametrize("alignment", [0, 3, True])
    def test_invalid_ring_replay_alignment_is_rejected(self, alignment):
        with pytest.raises(ValueError, match="Ring replay alignment"):
            CircularBufferSpec(
                block_size=8,
                num_kv_heads=1,
                head_size=128,
                dtype=torch.bfloat16,
                replay_alignment=alignment,
            )

    def test_projection_preserves_pool_and_transfer_roles_even_on_empty_stage(self):
        _, groups, workers, small = self.setup()
        for owned in (*workers, {"gpu.0": small}):
            projected = _project_kv_cache_groups_to_worker(groups, owned)
            host = projected[1]
            assert host.host_resident
            assert host.role is KVCacheGroupRole.DIRECT_HOST
            assert not host.enable_kv_transfer
            assert host.is_eagle_group == bool(host.layer_names)
        assert groups[1].layer_names == ["host.0", "host.1", "host.2"]

    def test_tp2_pp2_normalizes_device_capacity_without_shrinking_host_pages(self):
        config, groups, stages, small = self.setup()
        page = small.page_size_bytes
        # Both TP ranks of each PP stage consume their own pinned backing.
        workers = [stages[0], stages[0], stages[1], stages[1]]
        plans = get_direct_host_kv_cache_configs(
            config,
            groups,
            workers,
            [10 * page, 9 * page, 6 * page, 5 * page],
            host_num_blocks=7,
            host_max_bytes=80 * page,
        )
        assert [plan.num_blocks for plan in plans] == [5] * 4
        assert [plan.direct_host_num_blocks for plan in plans] == [7] * 4
        assert [plan.direct_host_idle_ttl_seconds for plan in plans] == [3600] * 4
        assert [plan.direct_host_max_bytes for plan in plans] == [
            8 * page,
            8 * page,
            32 * page,
            32 * page,
        ]
        for plan, owned in zip(plans, workers):
            assert {name for t in plan.kv_cache_tensors for name in t.layers} == set(
                owned
            )
            assert {t.size for t in plan.kv_cache_tensors if not t.host_resident} == {
                5 * page
            }
            host_pages = sum(
                spec.page_size_bytes
                for name, spec in owned.items()
                if name.startswith("host.")
            )
            assert {t.size for t in plan.kv_cache_tensors if t.host_resident} == {
                7 * host_pages
            }
            host_tensors = [t for t in plan.kv_cache_tensors if t.host_resident]
            assert all(t.block_stride == host_pages for t in host_tensors)
        scheduler = generate_scheduler_kv_cache_config(plans)
        assert scheduler.host_num_blocks == 7
        assert scheduler.kv_cache_groups[1].host_resident

    @pytest.mark.parametrize("ttl", [None, 120.5, 3600])
    def test_host_idle_lifetime_reaches_every_worker_and_native_pool(self, ttl):
        config, groups, stages, small = self.setup()
        config.kv_transfer_config = SimpleNamespace(
            kv_connector_extra_config={"idle_ttl_seconds": ttl}
        )
        plans = get_direct_host_kv_cache_configs(
            config,
            groups,
            stages,
            [MEMORY] * 2,
            host_num_blocks=7,
            host_max_bytes=MEMORY,
        )
        assert all(plan.direct_host_idle_ttl_seconds == ttl for plan in plans)
        scheduler = generate_scheduler_kv_cache_config(plans)
        manager = KVCacheManager(
            scheduler,
            max_model_len=32,
            enable_caching=True,
            hash_block_size=16,
            scheduler_block_size=16,
        )
        host = manager.coordinator.single_type_managers[1].block_pool
        assert host.idle_ttl_seconds == ttl
        assert host is not manager.block_pool

    def test_total_budget_counts_tp_copies_and_pinned_allocator_rounding(self):
        config, groups, workers, small = self.setup()
        page = small.page_size_bytes
        # Logical size is 28 pages, but rounded reservations need 40 pages.
        with pytest.raises(ValueError, match="total RAM budget"):
            get_direct_host_kv_cache_configs(
                config,
                groups,
                workers,
                [10 * page] * 2,
                host_num_blocks=7,
                host_max_bytes=40 * page - 1,
            )

    def test_tp4_target_draft_geometry_fits_non_power_of_two_host_capacity(self):
        """Plan measured QSA page geometry without allocating large tensors."""
        config, _, _, small = self.setup()
        config.model_config.original_max_model_len = 240000
        config.model_config.max_model_len = 240000
        page = FullAttentionSpec(
            block_size=944, num_kv_heads=1, head_size=256, dtype=torch.bfloat16
        )
        assert page.page_size_bytes == 966656
        groups = [KVCacheGroupSpec(["gpu"], small)]
        workers = {"gpu": small}
        for name, layers in (("target", 12), ("draft", 1)):
            names = [f"{name}.{i}" for i in range(layers)]
            workers.update(dict.fromkeys(names, page))
            groups.append(
                KVCacheGroupSpec(
                    names,
                    page,
                    is_eagle_group=name == "draft",
                    host_resident=True,
                    enable_kv_transfer=False,
                    role=KVCacheGroupRole.DIRECT_HOST,
                )
            )
        plans = get_direct_host_kv_cache_configs(
            config,
            groups,
            [workers] * 4,
            [1024**3] * 4,
            host_num_blocks=2560,
            host_max_bytes=128 * 1024**3,
        )
        assert [plan.direct_host_max_bytes for plan in plans] == [32 * 1024**3] * 4
        for plan in plans:
            host = [t for t in plan.kv_cache_tensors if t.host_resident]
            assert {t.size for t in host} == {2560 * 12 * page.page_size_bytes}
            assert {t.block_stride for t in host} == {12 * page.page_size_bytes}
        with pytest.raises(ValueError, match="total RAM budget"):
            get_direct_host_kv_cache_configs(
                config,
                groups,
                [workers] * 4,
                [1024**3] * 4,
                host_num_blocks=4096,
                host_max_bytes=128 * 1024**3,
            )

    def test_planned_host_views_do_not_alias_owned_layers_or_device_backing(self):
        config, groups, workers, _ = self.setup()
        plan = get_direct_host_kv_cache_configs(
            config,
            groups,
            workers,
            [MEMORY] * 2,
            host_num_blocks=7,
            host_max_bytes=MEMORY,
        )[1]
        # CPU layout test only; pinned UVA binding has a separate GPU probe.
        device_views, host_views = [
            _allocate_kv_cache(
                replace(
                    plan,
                    kv_cache_tensors=[
                        t for t in plan.kv_cache_tensors if t.host_resident == host
                    ],
                ),
                torch.device("cpu"),
                KVCacheLayout.BLNHC,
            )
            for host in (False, True)
        ]
        assert set(host_views) == {"host.1", "host.2"}
        device_views["gpu.1"].fill_(11)
        host_views["host.1"].fill_(22)
        host_views["host.2"].fill_(33)
        for name, value in (("host.1", 22), ("host.2", 33)):
            assert host_views[name].shape[0] == 7
            assert torch.all(host_views[name] == value)
        assert torch.all(device_views["gpu.1"] == 11)
        assert host_views["host.1"].untyped_storage().data_ptr() == (
            host_views["host.2"].untyped_storage().data_ptr()
        )
        assert host_views["host.1"].untyped_storage().data_ptr() != (
            device_views["gpu.1"].untyped_storage().data_ptr()
        )

    def test_empty_host_or_device_stage_keeps_shared_ids_without_unused_allocations(
        self,
    ):
        config, groups, workers, small = self.setup()
        workers = [{"gpu.0": small}, {"host.0": small}]
        plans = get_direct_host_kv_cache_configs(
            config,
            groups,
            workers,
            [8 * small.page_size_bytes, 0],
            host_num_blocks=7,
            host_max_bytes=8 * small.page_size_bytes,
        )
        assert [plan.num_blocks for plan in plans] == [8, 8]
        assert plans[0].direct_host_max_bytes == 0
        assert all(not t.host_resident for t in plans[0].kv_cache_tensors)
        assert all(t.host_resident for t in plans[1].kv_cache_tensors)
        assert plans[0].kv_cache_groups[1].role is KVCacheGroupRole.DIRECT_HOST
        assert plans[0].kv_cache_groups[1].layer_names == []
        scheduler = generate_scheduler_kv_cache_config(plans)
        assert scheduler.kv_cache_groups[1].is_eagle_group
        assert not plans[0].kv_cache_groups[1].is_eagle_group

    @pytest.mark.parametrize(
        "failure", ["capacity", "placeholder", "coverage", "roles"]
    )
    def test_invalid_host_plan_is_rejected_before_any_allocation(self, failure):
        config, groups, workers, small = self.setup()
        blocks = 7
        match = ""
        if failure == "capacity":
            blocks, match = 2, "null block"
        elif failure == "placeholder":
            groups[1].kv_cache_spec = replace(
                small, num_head_slots=1, state_content_bytes=2
            )
            match = "real full-attention"
        elif failure == "coverage":
            workers[0]["unassigned"] = small
            match = "exactly one"
        else:
            groups[1].role = KVCacheGroupRole.DEFAULT
            match = "consistent host group roles"
        with pytest.raises(ValueError, match=match):
            get_direct_host_kv_cache_configs(
                config,
                groups,
                workers,
                [MEMORY] * 2,
                host_num_blocks=blocks,
                host_max_bytes=MEMORY,
            )


class TestCSALinearGrouping:
    """A CSA + linear-attention model (sparse attention with a compressor ring,
    plus sharded GDN and one TP-replicated PLE state) goes through the generic
    packed-group path; no model-specific branch is involved."""

    @staticmethod
    def _mamba_groups(groups):
        """(group, per-layer mamba spec) for every group holding mamba state."""
        out = []
        for group in groups:
            spec = next(iter(iter_layer_specs(group.kv_cache_spec)))
            if isinstance(spec, MambaSpec):
                out.append((group, spec))
        return out

    def test_replicated_state_gets_its_own_group(self):
        """The PLE state is TP-replicated and a different size, so it must not
        share a manager group with the sharded GDN states (the NIXL worker
        requires exactly one single-layer replicated group)."""
        groups = get_kv_cache_groups(_shared_layout_config(), _make_csa_linear_specs())

        mamba_groups = self._mamba_groups(groups)
        replicated = [g for g, spec in mamba_groups if spec.tp_replicated]
        assert len(replicated) == 1
        assert replicated[0].layer_names == ["model.layers.0.ple"]
        sharded = [g for g, spec in mamba_groups if not spec.tp_replicated]
        assert sharded, "GDN states must keep their own groups"
        assert sorted(n for g in sharded for n in g.layer_names) == sorted(
            f"model.layers.{i}.linear_attn" for i in range(7)
        )

    def test_roles_land_in_separate_groups(self):
        groups = get_kv_cache_groups(_shared_layout_config(), _make_csa_linear_specs())

        owner = next(g for g in groups if _main_kv_name(0) in g.layer_names)
        assert sorted(owner.layer_names) == sorted(
            [
                *(_main_kv_name(i) for i in range(NUM_CACHE_TUPLES)),
                *(_compressed_name(i) for i in range(NUM_CACHE_TUPLES)),
            ]
        )
        assert owner.kv_cache_spec.prefix_cacheable

        scratch = next(g for g in groups if _compressor_state_name(0) in g.layer_names)
        assert scratch.layer_names == [
            _compressor_state_name(i) for i in range(NUM_CACHE_TUPLES)
        ]
        assert not scratch.kv_cache_spec.prefix_cacheable

    def test_every_group_fits_one_packed_block(self):
        config = _shared_layout_config()
        groups = get_kv_cache_groups(config, _make_csa_linear_specs())
        bytes_per_block = _get_kv_cache_bytes_per_block(groups)

        pages = _pages(groups)
        for group in groups:
            assert sum(pages[n] for n in group.layer_names) <= bytes_per_block

        kv_cache_config = get_kv_cache_config_from_groups(
            config, groups, available_memory=bytes_per_block * 32
        )
        assert kv_cache_config.num_blocks == 32
        # Groups overlay from byte 0 of each block, so a layer never addresses
        # past the block it belongs to.
        for tensor in kv_cache_config.kv_cache_tensors:
            assert tensor.block_stride == bytes_per_block
            assert tensor.offset < bytes_per_block

    def test_scratch_group_survives_computed_block_truncation(self):
        """The scratch group contributes no computed blocks, so truncating a
        lookup result must skip it: its block size is the ring capacity, which
        neither divides the hit length nor bounds the (empty) block list."""
        config = _shared_layout_config()
        config.cache_config.enable_prefix_caching = True
        groups = get_kv_cache_groups(config, _make_csa_linear_specs())
        kv_cache_config = get_kv_cache_config_from_groups(
            config, groups, available_memory=BYTES_PER_BLOCK * 64
        )
        scheduler_config = generate_scheduler_kv_cache_config([kv_cache_config])
        manager = KVCacheManager(
            scheduler_config,
            max_model_len=8192,
            enable_caching=True,
            hash_block_size=16,
            scheduler_block_size=16,
        )
        blocks = manager.create_kv_cache_blocks(
            tuple(
                manager.block_pool.get_new_blocks(3)
                if group.kv_cache_spec.prefix_cacheable
                else []
                for group in scheduler_config.kv_cache_groups
            )
        )
        truncated = manager.truncate_computed_blocks(blocks, 48)
        scratch_index = next(
            i
            for i, group in enumerate(scheduler_config.kv_cache_groups)
            if not group.kv_cache_spec.prefix_cacheable
        )
        assert truncated.blocks[scratch_index] == []

    def test_prefix_hits_respect_compression_alignment(self):
        config = _shared_layout_config()
        config.cache_config.enable_prefix_caching = True
        config.cache_config.prefix_match_unit = 16
        config.cache_config.mamba_cache_mode = "align"
        specs = {
            name: replace(spec, mamba_cache_mode="align")
            if isinstance(spec, MambaSpec)
            else spec
            for name, spec in _make_csa_linear_specs(num_tuples=1).items()
        }
        groups = get_kv_cache_groups(config, specs)
        kv_cache_config = get_kv_cache_config_from_groups(
            config, groups, available_memory=8 * MAIN_KV_PAGE_BYTES
        )
        assert resolve_kv_cache_block_sizes(kv_cache_config, config) == (16, 16)

        # 2 tokens is not a multiple of the compression ratio 4: a prefix hit
        # could land inside a partially filled compressed state.
        config.cache_config.prefix_match_unit = 2
        with pytest.raises(ValueError, match="per-state compression"):
            resolve_kv_cache_block_sizes(kv_cache_config, config)

    def test_scratch_ring_does_not_drag_hash_granularity(self):
        config = _shared_layout_config()
        config.cache_config.enable_prefix_caching = True
        config.cache_config.mamba_cache_mode = "align"
        specs = {
            name: replace(spec, mamba_cache_mode="align")
            if isinstance(spec, MambaSpec)
            else spec
            for name, spec in _make_csa_linear_specs(num_tuples=1).items()
        }
        groups = get_kv_cache_groups(config, specs)
        kv_cache_config = get_kv_cache_config_from_groups(
            config, groups, available_memory=8 * MAIN_KV_PAGE_BYTES
        )
        # The hash granularity is the GCD over prefix-cacheable groups only;
        # the 4-token scratch ring is excluded (it would drag it to 4).
        assert resolve_kv_cache_block_sizes(kv_cache_config, config) == (16, 16)

    def test_compressed_attention_hashes_can_be_finer_than_cache_hits(self):
        config = _shared_layout_config()
        config.cache_config.enable_prefix_caching = True
        specs = {
            "compressed.4": MLAAttentionSpec(
                block_size=256,
                num_kv_heads=1,
                head_size=16,
                dtype=torch.bfloat16,
                tokens_per_state=4,
            ),
            "compressed.128": MLAAttentionSpec(
                block_size=256,
                num_kv_heads=1,
                head_size=16,
                dtype=torch.bfloat16,
                tokens_per_state=128,
            ),
            "compressor_state.4": SlidingWindowMLASpec(
                block_size=4,
                num_kv_heads=1,
                head_size=8,
                head_size_v=0,
                dtype=torch.bfloat16,
                sliding_window=8,
            ),
        }
        groups = get_kv_cache_groups(config, specs)
        kv_cache_config = get_kv_cache_config_from_groups(
            config, groups, available_memory=8 * MAIN_KV_PAGE_BYTES
        )

        # Hashes are computed every 4 tokens, but without an align-mode Mamba
        # group cache hits remain on the 256-token scheduler boundary.
        assert resolve_kv_cache_block_sizes(kv_cache_config, config) == (256, 4)

    @pytest.mark.parametrize("wide", ["unbalanced_attention", "unsplittable_state"])
    def test_mamba_split_measures_the_block_the_other_groups_already_force(self, wide):
        """Whatever fixes the block stride -- a bucket with unequal layer counts
        per page size, which is emitted whole, or a single-layer state that
        cannot be split at all -- the mamba layers must be sized against it.
        Sizing them against a narrower bucket splits them past what the block
        already fits, spending a pool block per extra group for no saving."""
        config = _mock_vllm_config("BLNHC")
        config.speculative_config = None
        specs = {}
        if wide == "unbalanced_attention":
            for i in range(3):
                specs[f"wide.{i}"] = _mla(1024)
            specs["wide.odd"] = _mla(800)
        else:
            specs["wide.state"] = MambaSpec(
                block_size=16,
                shapes=((51_200,),),
                dtypes=(torch.bfloat16,),
                tp_replicated=True,
            )
        # Mixed and balanced, but far narrower than the bucket above.
        for i in range(10):
            specs[f"narrow.a.{i}"] = FullAttentionSpec(
                block_size=16, num_kv_heads=1, head_size=8, dtype=torch.uint8
            )
            specs[f"narrow.b.{i}"] = FullAttentionSpec(
                block_size=16, num_kv_heads=1, head_size=16, dtype=torch.uint8
            )
        for i in range(100):
            specs[f"gdn.{i}"] = MambaSpec(
                block_size=16, shapes=((512,),), dtypes=(torch.bfloat16,)
            )

        groups = _get_packed_kv_cache_groups(config, specs)
        gdn = [g for g in groups if g.layer_names[0].startswith("gdn.")]

        assert _get_kv_cache_bytes_per_block(groups) == sum(
            specs[name].page_size_bytes for name in specs if name.startswith("wide.")
        )
        # That block holds every GDN state at once, so the repeat pattern alone
        # decides the split; the cap must not add groups on top of it.
        assert len(gdn) == 10


class TestSlidingWindowBucketCap:
    def test_sliding_window_bucket_is_capped_at_the_main_page(self):
        """DeepSeek-V4.1 shape: 43 SlidingWindowMLASpec SWA caches beside an
        unbalanced 8-layer paged MLA bucket. Left whole, the SWA bucket would
        set the block width and the MLA group would fill under a third of
        every block; capped, it splits into groups no wider than the MLA
        page."""
        config = _shared_layout_config()
        specs: dict[str, KVCacheSpec] = {}
        for layer, ratio in ((2, 2), (8, 2), (14, 2), (20, 1)):
            specs[f"layers.{layer}.attn"] = MLAAttentionSpec(
                block_size=128,
                num_kv_heads=1,
                head_size=584,
                dtype=torch.uint8,
                tokens_per_state=ratio,
                alignment=576,
            )
            specs[f"layers.{layer}.idx"] = MLAAttentionSpec(
                block_size=128,
                num_kv_heads=1,
                head_size=132,
                dtype=torch.uint8,
                tokens_per_state=ratio,
                alignment=576,
            )
        for layer in range(43):
            specs[f"layers.{layer}.swa"] = SlidingWindowMLASpec(
                block_size=64,
                num_kv_heads=1,
                head_size=584,
                dtype=torch.uint8,
                sliding_window=128,
                alignment=576,
            )
        groups = get_kv_cache_groups(config, specs)
        pages = _pages(groups)
        main = next(g for g in groups if "layers.2.attn" in g.layer_names)
        main_bytes = sum(pages[n] for n in main.layer_names)
        assert _get_kv_cache_bytes_per_block(groups) == main_bytes
        swa_groups = [g for g in groups if g.layer_names[0].endswith(".swa")]
        per_group = main_bytes // pages["layers.0.swa"]
        assert len(swa_groups) == -(-43 // per_group)
        for g in swa_groups:
            assert sum(pages[n] for n in g.layer_names) <= main_bytes
        assert sorted(n for g in swa_groups for n in g.layer_names) == sorted(
            f"layers.{i}.swa" for i in range(43)
        )


class TestDensePacking:
    def test_bytes_per_block_is_largest_group(self):
        groups, g1, g2 = _mixed_page_groups()
        assert _get_kv_cache_bytes_per_block(groups) == _expected_bytes_per_block(
            groups
        )

        config = get_kv_cache_config_from_groups(
            _mock_vllm_config("BLHNC"), groups, MEMORY
        )
        # Groups overlay: both start at offset 0.
        assert [tensor.offset for tensor in config.kv_cache_tensors].count(0) == 2
        assert [tensor.layers for tensor in config.kv_cache_tensors] == [
            list(g1)[:3],
            list(g1)[3:],
            list(g2),
        ]

    def test_layers_within_a_group_are_dense(self):
        groups, _, _ = _mixed_page_groups()
        pages = _pages(groups)
        config = get_kv_cache_config_from_groups(
            _mock_vllm_config("BLHNC"), groups, MEMORY
        )
        offsets = {
            name: tensor.offset + i * tensor.layer_stride
            for tensor in config.kv_cache_tensors
            for i, name in enumerate(tensor.layers)
        }
        for group in groups:
            expected = 0
            for name in group.layer_names:
                assert offsets[name] == expected
                expected += pages[name]

    @pytest.mark.parametrize("layout", ["LBNHC", "BLHNC"])
    def test_allocation_is_layout_invariant(self, layout):
        specs = {f"l.{i}": _full() for i in range(4)}
        groups = [KVCacheGroupSpec(list(specs), _full())]
        config = get_kv_cache_config_from_groups(
            _mock_vllm_config(layout), groups, MEMORY
        )
        (tensor,) = config.kv_cache_tensors
        page = _full().page_size_bytes
        assert tensor.layers == list(specs)
        assert config.num_blocks == MEMORY // (4 * page)
        assert tensor.size == 4 * page * config.num_blocks
        if layout == "LBNHC":
            assert (tensor.layer_stride, tensor.block_stride) == (
                page * config.num_blocks,
                page,
            )
        else:
            assert (tensor.layer_stride, tensor.block_stride) == (page, 4 * page)

    @pytest.mark.parametrize("layout", ["LBNHC", "BLHNC"])
    def test_single_group_mixed_pages_follows_layout(self, layout):
        specs = {"mla.0": _mla(512), "mla.1": _mla(512), "idx.0": _mla(128)}
        groups = [_uniform_group(specs)]
        config = get_kv_cache_config_from_groups(
            _mock_vllm_config(layout), groups, MEMORY
        )
        block_stride = _expected_bytes_per_block(groups)
        mla_tensor, idx_tensor = config.kv_cache_tensors
        assert mla_tensor.layers == ["mla.0", "mla.1"]
        assert idx_tensor.layers == ["idx.0"]
        assert {t.size for t in config.kv_cache_tensors} == {
            block_stride * config.num_blocks
        }
        if layout == "LBNHC":
            assert (
                idx_tensor.offset == 2 * _mla(512).page_size_bytes * config.num_blocks
            )
            assert mla_tensor.block_stride == _mla(512).page_size_bytes
        else:
            assert idx_tensor.offset == 2 * _mla(512).page_size_bytes
            assert mla_tensor.block_stride == block_stride

    def test_overlaid_groups_alias_and_stay_isolated(self):
        groups, g1, g2 = _mixed_page_groups()
        # Overlay models resolve to a block-outer layout at backend selection (the
        # model's backend declares it); mirror that here.
        config = get_kv_cache_config_from_groups(
            _mock_vllm_config("BLNHC"), groups, MEMORY
        )
        assert config.num_blocks == MEMORY // _expected_bytes_per_block(groups)
        assert _pool_bytes_per_block(groups) == _expected_bytes_per_block(groups)

        views = _bind(config, "BLNHC")
        assert set(views) == set(g1) | set(g2)
        assert views["swa.0"].data_ptr() == views["mla.0"].data_ptr()

        # A block is owned by one group at a time: writes to group-owned
        # blocks never disturb the other group's blocks.
        for i, name in enumerate(g1):
            views[name][0].fill_(i + 1)
            views[name][2].fill_(i + 1)
        for i, name in enumerate(g2):
            views[name][1].fill_(100 + i)
            views[name][3].fill_(100 + i)
        for i, name in enumerate(g1):
            assert (views[name][0].to(torch.int32) == i + 1).all()
            assert (views[name][2].to(torch.int32) == i + 1).all()
        for i, name in enumerate(g2):
            assert (views[name][1].to(torch.int32) == 100 + i).all()
            assert (views[name][3].to(torch.int32) == 100 + i).all()
        # Layers within a group are disjoint.
        views["mla.0"][0].fill_(77)
        for i, name in enumerate(list(g1)[1:], start=1):
            assert (views[name][0].to(torch.int32) == i + 1).all()

    def test_layer_compact_layout_rejected_for_overlaid_groups(self):
        # The layout has a single writer (backend-selection resolution); a layer-
        # compact layout reaching an overlay model's allocation is an error, not a
        # silent flip.
        groups, _, _ = _mixed_page_groups()
        with pytest.raises(
            ValueError, match="cannot express this model's mixed page sizes"
        ):
            get_kv_cache_config_from_groups(_mock_vllm_config("LBNHC"), groups, MEMORY)

    def test_unresolved_layout_rejected(self):
        groups, _, _ = _mixed_page_groups()
        with pytest.raises(ValueError, match="has not been resolved"):
            get_kv_cache_config_from_groups(_mock_vllm_config(None), groups, MEMORY)

    def test_head_outer_layout_rejected_for_mixed_pages(self):
        groups, _, _ = _mixed_page_groups()
        with pytest.raises(
            ValueError, match="cannot express this model's mixed page sizes"
        ):
            get_kv_cache_config_from_groups(_mock_vllm_config("LHBNC"), groups, MEMORY)

    @pytest.mark.parametrize("layout", ["LBNHC", "BLHNC"])
    def test_bound_views_round_trip(self, layout):
        specs = {"mla.0": _mla(512), "mla.1": _mla(512), "idx.0": _mla(128)}
        groups = [_uniform_group(specs)]
        config = get_kv_cache_config_from_groups(
            _mock_vllm_config(layout), groups, MEMORY
        )
        views = _bind(config, layout)
        for i, name in enumerate(specs):
            views[name].fill_(i + 1)
        for i, name in enumerate(specs):
            assert (views[name].to(torch.int32) == i + 1).all(), name
            assert views[name].shape[0] == config.num_blocks


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestCompressorRingGroup:
    def test_ring_beside_paged_groups_keeps_the_kv_block_as_scheduler_block(self):
        """DeepSeek-V4.1 compressor state: a CircularBufferSpec ring (capacity
        8) beside 128-token paged MLA groups and 64-token SWA groups. The ring
        claims one block per request, never hashes, and does not disturb the
        scheduler/hash block sizes the paged groups imply."""
        config = _shared_layout_config()
        config.cache_config.enable_prefix_caching = True
        specs: dict[str, KVCacheSpec] = {
            "layers.2.attn": MLAAttentionSpec(
                block_size=128,
                num_kv_heads=1,
                head_size=584,
                dtype=torch.uint8,
                tokens_per_state=2,
                alignment=576,
            ),
            "layers.2.attn.compressor.state_cache": CircularBufferSpec(
                block_size=8,
                num_kv_heads=1,
                head_size=1024,
                head_size_v=0,
                dtype=torch.float32,
            ),
        }
        for layer in range(4):
            specs[f"layers.{layer}.attn.swa_cache"] = SlidingWindowMLASpec(
                block_size=64,
                num_kv_heads=1,
                head_size=584,
                dtype=torch.uint8,
                sliding_window=128,
                alignment=576,
            )
        groups = get_kv_cache_groups(config, specs)
        ring_groups = [g for g in groups if "state_cache" in g.layer_names[0]]
        assert len(ring_groups) == 1
        assert not ring_groups[0].kv_cache_spec.prefix_cacheable
        kv_cache_config = get_kv_cache_config_from_groups(
            config, groups, available_memory=64 * _get_kv_cache_bytes_per_block(groups)
        )
        assert resolve_kv_cache_block_sizes(kv_cache_config, config) == (128, 64)
        manager = KVCacheManager(
            generate_scheduler_kv_cache_config([kv_cache_config]),
            max_model_len=8192,
            enable_caching=True,
            hash_block_size=64,
            scheduler_block_size=128,
        )
        assert len(manager.coordinator.single_type_managers) == len(groups)
