# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for native offloading specs and their factory."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingHistogramMetadata,
    OffloadingManager,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec


@pytest.fixture(autouse=True)
def restore_registry():
    original = dict(OffloadingSpecFactory._registry)
    yield
    OffloadingSpecFactory._registry = original


def _make_offloading_config(
    *,
    spec_name: str | None = "CPUOffloadingSpec",
    cpu_bytes_to_use: int | None = 65536,
    worker_kv_bytes_per_block: int = 8,
    groups: tuple[OffloadingGroupConfig, ...] | None = None,
    tokens_per_hash: int = 16,
    blocks_per_chunk: int = 1,
    rank: int = 0,
    world_size: int = 1,
    tp_size: int | None = None,
    pp_size: int = 1,
    pcp_size: int = 1,
    dcp_size: int = 1,
    data_parallel_index: int = 0,
    data_parallel_size: int = 1,
    data_parallel_rank_local: int | None = None,
    is_parallelism_agnostic: bool = False,
    replicated_layout: bool = False,
    extra_config: dict[str, Any] | None = None,
) -> OffloadingConfig:
    normalized_extra_config = dict(extra_config or {})
    if spec_name is not None:
        normalized_extra_config["spec_name"] = spec_name
    if cpu_bytes_to_use is not None:
        normalized_extra_config["cpu_bytes_to_use"] = cpu_bytes_to_use

    if groups is None:
        groups = (OffloadingGroupConfig(16, ("layer",), 0),)

    return OffloadingConfig(
        groups=groups,
        worker_kv_bytes_per_block=worker_kv_bytes_per_block,
        enable_kv_cache_events=False,
        extra_config=normalized_extra_config,
        engine_id="test-engine",
        model=OffloadingModelConfig(name="test-model", dtype="float16"),
        cache=OffloadingCacheConfig(
            tokens_per_hash=tokens_per_hash,
            blocks_per_chunk=blocks_per_chunk,
        ),
        parallel=OffloadingParallelConfig(
            rank=rank,
            world_size=world_size,
            tp_size=world_size if tp_size is None else tp_size,
            pp_size=pp_size,
            pcp_size=pcp_size,
            dcp_size=dcp_size,
            data_parallel_index=data_parallel_index,
            data_parallel_size=data_parallel_size,
            data_parallel_rank_local=data_parallel_rank_local,
            is_parallelism_agnostic=is_parallelism_agnostic,
        ),
        replicated_layout=replicated_layout,
    )


def _create_spec(**kwargs: Any) -> OffloadingSpec:
    return OffloadingSpecFactory.create_spec(_make_offloading_config(**kwargs))


@pytest.mark.parametrize("ttl", [None, 3600, 120.5])
def test_cpu_spec_passes_idle_expiry_to_native_manager(ttl):
    spec = _create_spec(extra_config={"idle_ttl_seconds": ttl})
    assert spec.get_manager().idle_ttl_seconds == ttl


def test_expiry_counter_registered_even_when_ttl_default_is_injected_later():
    definitions = CPUOffloadingSpec.build_metric_definitions({})
    assert CPUOffloadingMetrics.CPU_CACHE_EXPIRED_CHUNKS in definitions


@pytest.mark.parametrize("matches", [True, False])
def test_pp_mmap_geometry_is_checked_across_workers(monkeypatch, matches):
    import vllm.v1.kv_offload.cpu.spec as module

    group = MagicMock()
    monkeypatch.setattr(module, "_all_workers_group", lambda: group)
    monkeypatch.setattr(module.torch.distributed, "get_world_size", lambda g: 4)

    def gather(layouts, geometry, *, group):
        layouts[:] = [geometry] * 4
        if not matches:
            layouts[2] = (geometry[0], geometry[1] * 2, geometry[2] * 2)

    monkeypatch.setattr(module.torch.distributed, "all_gather_object", gather)
    if matches:
        module._validate_pp_mmap_geometry((128, 65536, 16384))
    else:
        with pytest.raises(ValueError, match="disagree"):
            module._validate_pp_mmap_geometry((128, 65536, 16384))


def test_pp_mmap_mismatch_fails_before_any_region_is_opened(monkeypatch):
    import vllm.v1.kv_offload.cpu.spec as module

    spec = _create_spec(world_size=4, tp_size=2, pp_size=2)
    monkeypatch.setattr(spec, "_uses_shared_region", lambda: True)
    validate = MagicMock(side_effect=ValueError("disagree"))
    region = MagicMock()
    monkeypatch.setattr(module, "_validate_pp_mmap_geometry", validate)
    monkeypatch.setattr(module, "SharedOffloadRegion", region)
    with pytest.raises(ValueError, match="disagree"):
        spec.create_worker(MagicMock())
    validate.assert_called_once()
    region.assert_not_called()


class SingleArgExternalOffloadingSpec(OffloadingSpec):
    def get_manager(self) -> OffloadingManager:
        raise NotImplementedError

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        raise NotImplementedError


def test_pre_registered_specs_can_be_imported():
    for name in OffloadingSpecFactory._registry:
        cls = OffloadingSpecFactory._registry[name]()
        assert issubclass(cls, OffloadingSpec)


def test_cpu_spec_registered():
    cls = OffloadingSpecFactory._registry["CPUOffloadingSpec"]()
    assert cls is CPUOffloadingSpec


def test_tiering_spec_registered():
    cls = OffloadingSpecFactory._registry["TieringOffloadingSpec"]()
    assert cls is TieringOffloadingSpec


def test_get_spec_cls_returns_registered_class():
    spec_cls = OffloadingSpecFactory.get_spec_cls(
        _make_offloading_config().extra_config
    )
    assert spec_cls is CPUOffloadingSpec


def test_get_spec_cls_defaults_to_cpu():
    spec_cls = OffloadingSpecFactory.get_spec_cls(
        _make_offloading_config(spec_name=None).extra_config
    )
    assert spec_cls is CPUOffloadingSpec


def test_create_cpu_offloading_spec():
    spec = _create_spec()
    assert isinstance(spec, CPUOffloadingSpec)
    assert spec.num_chunks > 0


def test_cpu_spec_sizes_normalized_worker_layout():
    # The CPU spec now rounds the offloaded row up to the mmap page size
    # (matching the shared region), so kv_bytes_per_chunk picks up padding
    # while cpu_page_size_per_worker stays the un-padded per-worker slot.
    alignment = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    spec = _create_spec(
        cpu_bytes_to_use=alignment * 3,
        worker_kv_bytes_per_block=16,
        blocks_per_chunk=2,
        world_size=6,
        tp_size=3,
        pp_size=2,
    )

    assert isinstance(spec, CPUOffloadingSpec)
    assert spec.cpu_page_size_per_worker == 32
    assert spec.kv_bytes_per_chunk == alignment
    assert spec.num_chunks == 3


def test_cpu_spec_zero_worker_bytes_produces_empty_cache():
    spec = _create_spec(worker_kv_bytes_per_block=0, world_size=4)

    assert isinstance(spec, CPUOffloadingSpec)
    assert spec.cpu_page_size_per_worker == 0
    assert spec.kv_bytes_per_chunk == 0
    assert spec.num_chunks == 0


def test_tiering_spec_aligns_row_size():
    alignment = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    spec = _create_spec(
        spec_name="TieringOffloadingSpec",
        cpu_bytes_to_use=alignment * 3,
        worker_kv_bytes_per_block=16,
        blocks_per_chunk=2,
        world_size=6,
        tp_size=3,
        pp_size=2,
    )

    assert isinstance(spec, TieringOffloadingSpec)
    assert spec.cpu_page_size_per_worker == 32
    assert spec.kv_bytes_per_chunk == alignment
    assert spec.num_chunks == 3


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_tiering_spec_replicated_sizing_removes_world_factor(world_size: int):
    worker_kv_bytes_per_block = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    spec = _create_spec(
        spec_name="TieringOffloadingSpec",
        cpu_bytes_to_use=worker_kv_bytes_per_block * 8,
        worker_kv_bytes_per_block=worker_kv_bytes_per_block,
        world_size=world_size,
        replicated_layout=True,
    )

    assert isinstance(spec, TieringOffloadingSpec)
    assert spec.replicated_layout is True
    assert spec.cpu_page_size_per_worker == worker_kv_bytes_per_block
    assert spec.kv_bytes_per_chunk == worker_kv_bytes_per_block
    assert spec.num_chunks == 8


def test_tiering_spec_create_worker_uses_single_slot_for_replicated_layout(monkeypatch):
    import vllm.v1.kv_offload.tiering.spec as tiering_spec_module

    worker_kv_bytes_per_block = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    spec = _create_spec(
        spec_name="TieringOffloadingSpec",
        cpu_bytes_to_use=worker_kv_bytes_per_block * 8,
        worker_kv_bytes_per_block=worker_kv_bytes_per_block,
        world_size=4,
        replicated_layout=True,
    )
    assert isinstance(spec, TieringOffloadingSpec)

    region = MagicMock()
    region_calls: list[dict[str, Any]] = []
    worker_calls: list[dict[str, Any]] = []

    def fake_region_ctor(**kwargs):
        region_calls.append(kwargs)
        return region

    def fake_worker_ctor(**kwargs):
        worker_calls.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(tiering_spec_module, "SharedOffloadRegion", fake_region_ctor)
    monkeypatch.setattr(tiering_spec_module, "CPUOffloadingWorker", fake_worker_ctor)
    monkeypatch.setattr(
        tiering_spec_module.torch.accelerator, "current_device_index", lambda: 5
    )

    kv_caches = MagicMock()
    spec.create_worker(kv_caches)

    assert region_calls[0]["rank"] == 0
    assert region_calls[0]["kv_bytes_per_chunk"] == worker_kv_bytes_per_block
    assert worker_calls[0]["kv_caches"] is kv_caches
    assert worker_calls[0]["mmap_region"] is region


def test_tiering_spec_create_worker_folds_device_index_for_sharded_layout(monkeypatch):
    import vllm.v1.kv_offload.tiering.spec as tiering_spec_module

    spec = _create_spec(
        spec_name="TieringOffloadingSpec",
        worker_kv_bytes_per_block=4096,
        world_size=4,
    )
    assert isinstance(spec, TieringOffloadingSpec)

    region_calls: list[dict[str, Any]] = []

    def fake_region_ctor(**kwargs):
        region_calls.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(tiering_spec_module, "SharedOffloadRegion", fake_region_ctor)
    monkeypatch.setattr(tiering_spec_module, "CPUOffloadingWorker", MagicMock())
    monkeypatch.setattr(
        tiering_spec_module.torch.accelerator,
        "current_device_index",
        lambda: 5,
    )

    spec.create_worker(MagicMock())

    assert region_calls[0]["rank"] == 1


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_cpu_spec_replicated_sizing_on_shared_region(monkeypatch, world_size: int):
    # On shared-region (CUDA-alike) platforms the default spec now honors
    # replicated layout: a single MLA copy (num_copies=1), matching tiering.
    import vllm.v1.kv_offload.cpu.spec as cpu_spec_module

    monkeypatch.setattr(cpu_spec_module.current_platform, "is_cuda_alike", lambda: True)
    worker_kv_bytes_per_block = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    spec = _create_spec(
        cpu_bytes_to_use=worker_kv_bytes_per_block * 8,
        worker_kv_bytes_per_block=worker_kv_bytes_per_block,
        world_size=world_size,
        replicated_layout=True,
    )

    assert isinstance(spec, CPUOffloadingSpec)
    assert spec.replicated_layout is True
    assert spec.cpu_page_size_per_worker == worker_kv_bytes_per_block
    assert spec.kv_bytes_per_chunk == worker_kv_bytes_per_block
    assert spec.num_chunks == 8


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_cpu_spec_replicated_disabled_without_shared_region(
    monkeypatch, world_size: int
):
    # Data-loss guard: non-CUDA-alike platforms keep a per-rank private pinned
    # tensor (no shared medium), so replicated layout MUST stay off. Otherwise
    # the rank-0 writer gate would ack rank>0 stores without writing, leaving
    # those private buffers empty and corrupting subsequent loads.
    import vllm.v1.kv_offload.cpu.spec as cpu_spec_module

    monkeypatch.setattr(
        cpu_spec_module.current_platform, "is_cuda_alike", lambda: False
    )
    worker_kv_bytes_per_block = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    spec = _create_spec(
        cpu_bytes_to_use=worker_kv_bytes_per_block * world_size * 2,
        worker_kv_bytes_per_block=worker_kv_bytes_per_block,
        world_size=world_size,
        replicated_layout=True,
    )

    assert isinstance(spec, CPUOffloadingSpec)
    assert spec.replicated_layout is False
    assert spec.cpu_page_size_per_worker == worker_kv_bytes_per_block
    assert spec.kv_bytes_per_chunk == worker_kv_bytes_per_block * world_size
    assert spec.num_chunks == 2


@pytest.mark.parametrize("config_replicated", [True, False])
@pytest.mark.parametrize("cuda_alike", [True, False])
@pytest.mark.parametrize("use_shared_memory", [True, False])
def test_cpu_spec_replicated_layout_truth_matrix(
    monkeypatch, cuda_alike: bool, config_replicated: bool, use_shared_memory: bool
):
    # replicated_layout is enabled iff the config gate passes AND the deployment
    # actually allocates on the shared region (CUDA-alike).
    import vllm.v1.kv_offload.cpu.spec as cpu_spec_module

    monkeypatch.setattr(
        cpu_spec_module.current_platform, "is_cuda_alike", lambda: cuda_alike
    )
    worker_kv_bytes_per_block = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    spec = _create_spec(
        cpu_bytes_to_use=worker_kv_bytes_per_block * 8,
        worker_kv_bytes_per_block=worker_kv_bytes_per_block,
        world_size=4,
        replicated_layout=config_replicated,
        extra_config={"use_shared_memory": use_shared_memory},
    )

    assert isinstance(spec, CPUOffloadingSpec)
    assert spec.replicated_layout is (
        cuda_alike and config_replicated and use_shared_memory
    )


@pytest.mark.parametrize("invalid", [None, "false", 0, 1])
def test_cpu_spec_rejects_ambiguous_shared_memory_selection(invalid):
    with pytest.raises(ValueError, match="use_shared_memory must be a boolean"):
        _create_spec(extra_config={"use_shared_memory": invalid})


def test_cpu_spec_create_worker_uses_mmap_on_cuda_alike(monkeypatch):
    import vllm.v1.kv_offload.cpu.spec as cpu_spec_module

    worker_kv_bytes_per_block = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    spec = _create_spec(
        cpu_bytes_to_use=worker_kv_bytes_per_block * 8,
        worker_kv_bytes_per_block=worker_kv_bytes_per_block,
        world_size=4,
    )
    assert isinstance(spec, CPUOffloadingSpec)

    region = MagicMock()
    region_calls: list[dict[str, Any]] = []
    worker_calls: list[dict[str, Any]] = []

    def fake_region_ctor(**kwargs):
        region_calls.append(kwargs)
        return region

    def fake_worker_ctor(**kwargs):
        worker_calls.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(cpu_spec_module.current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(cpu_spec_module, "SharedOffloadRegion", fake_region_ctor)
    monkeypatch.setattr(cpu_spec_module, "CPUOffloadingWorker", fake_worker_ctor)
    monkeypatch.setattr(
        cpu_spec_module.torch.accelerator, "current_device_index", lambda: 5
    )

    kv_caches = MagicMock()
    spec.create_worker(kv_caches)

    assert region_calls[0]["engine_id"] == "test-engine"
    assert region_calls[0]["kv_bytes_per_chunk"] == worker_kv_bytes_per_block * 4
    assert worker_calls[0]["kv_caches"] is kv_caches
    assert worker_calls[0]["mmap_region"] is region


@pytest.mark.parametrize("cuda_alike", [False, True])
def test_cpu_spec_create_worker_uses_existing_private_tensor_path(
    monkeypatch, cuda_alike
):
    import vllm.v1.kv_offload.cpu.spec as cpu_spec_module

    spec = _create_spec(
        worker_kv_bytes_per_block=4096,
        world_size=4,
        extra_config={"use_shared_memory": False},
    )
    assert isinstance(spec, CPUOffloadingSpec)

    region_calls: list[dict[str, Any]] = []
    worker_calls: list[dict[str, Any]] = []

    def fake_region_ctor(**kwargs):
        region_calls.append(kwargs)
        return MagicMock()

    def fake_worker_ctor(**kwargs):
        worker_calls.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(
        cpu_spec_module.current_platform, "is_cuda_alike", lambda: cuda_alike
    )
    monkeypatch.setattr(cpu_spec_module, "SharedOffloadRegion", fake_region_ctor)
    monkeypatch.setattr(cpu_spec_module, "CPUOffloadingWorker", fake_worker_ctor)

    spec.create_worker(MagicMock())

    # Explicit private allocation never constructs or registers a shared mmap.
    assert region_calls == []
    assert worker_calls[0]["mmap_region"] is None


def test_cpu_spec_create_worker_skips_mmap_for_empty_cache(monkeypatch):
    import vllm.v1.kv_offload.cpu.spec as cpu_spec_module

    # worker_kv_bytes_per_block=0 yields num_chunks=0; a zero-byte region cannot
    # be mmap'd, so even on CUDA-alike this must fall back to the tensor path.
    spec = _create_spec(worker_kv_bytes_per_block=0, world_size=4)
    assert isinstance(spec, CPUOffloadingSpec)
    assert spec.num_chunks == 0

    region_calls: list[dict[str, Any]] = []
    worker_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(cpu_spec_module.current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(
        cpu_spec_module,
        "SharedOffloadRegion",
        lambda **kwargs: region_calls.append(kwargs),
    )
    monkeypatch.setattr(
        cpu_spec_module,
        "CPUOffloadingWorker",
        lambda **kwargs: worker_calls.append(kwargs),
    )

    spec.create_worker(MagicMock())

    assert region_calls == []
    assert worker_calls[0]["mmap_region"] is None


@pytest.mark.parametrize(
    ("replicated_layout", "device_index", "world_size", "expected_rank"),
    [
        (True, 5, 4, 0),  # replicated: always slot 0
        (True, 0, 4, 0),  # replicated: slot 0 regardless of device
        (False, 5, 4, 1),  # non-replicated: 5 % 4 == 1
        (False, 7, 4, 3),  # non-replicated: 7 % 4 == 3
    ],
)
def test_cpu_spec_create_worker_rank_assignment(
    monkeypatch, replicated_layout, device_index, world_size, expected_rank
):
    import vllm.v1.kv_offload.cpu.spec as cpu_spec_module

    monkeypatch.setattr(cpu_spec_module.current_platform, "is_cuda_alike", lambda: True)
    worker_kv_bytes_per_block = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    spec = _create_spec(
        cpu_bytes_to_use=worker_kv_bytes_per_block * 8,
        worker_kv_bytes_per_block=worker_kv_bytes_per_block,
        world_size=world_size,
        replicated_layout=replicated_layout,
    )

    region_calls: list[dict[str, Any]] = []

    def fake_region_ctor(**kwargs):
        region_calls.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(cpu_spec_module, "SharedOffloadRegion", fake_region_ctor)
    monkeypatch.setattr(cpu_spec_module, "CPUOffloadingWorker", MagicMock())
    monkeypatch.setattr(
        cpu_spec_module.torch.accelerator, "current_device_index", lambda: device_index
    )

    spec.create_worker(MagicMock())

    assert region_calls[0]["rank"] == expected_rank


def test_offloading_spec_has_replicated_layout_default():
    spec = SingleArgExternalOffloadingSpec(_make_offloading_config())
    assert spec.replicated_layout is False


def test_offloading_spec_uses_normalized_chunk_geometry():
    groups = (
        OffloadingGroupConfig(12, ("full_layer",), 0),
        OffloadingGroupConfig(16, ("mla_layer",), 1),
    )
    spec = _create_spec(
        groups=groups,
        tokens_per_hash=4,
        blocks_per_chunk=2,
    )

    assert spec.tokens_per_block == (12, 16)
    assert spec.tokens_per_hash == 4
    assert spec.blocks_per_chunk == 2


def test_create_dynamic_spec_receives_config():
    config = _make_offloading_config(
        spec_name="SingleArgExternalOffloadingSpec",
        extra_config={"spec_module_path": "tests.v1.kv_offload.test_factory"},
    )

    spec = OffloadingSpecFactory.create_spec(config)

    assert isinstance(spec, SingleArgExternalOffloadingSpec)
    assert spec.config is config


def test_dynamic_load_via_spec_module_path():
    del OffloadingSpecFactory._registry["CPUOffloadingSpec"]
    config = _make_offloading_config(
        extra_config={"spec_module_path": "vllm.v1.kv_offload.cpu.spec"}
    )

    spec_cls = OffloadingSpecFactory.get_spec_cls(config.extra_config)

    assert spec_cls is CPUOffloadingSpec


def test_unregistered_spec_without_module_path_raises():
    config = _make_offloading_config(spec_name="NonexistentSpec")
    with pytest.raises(ValueError, match="Unsupported spec type"):
        OffloadingSpecFactory.get_spec_cls(config.extra_config)

    with pytest.raises(ValueError, match="Unsupported spec type"):
        OffloadingSpecFactory.create_spec(config)


def test_cpu_spec_missing_cpu_bytes_to_use_raises():
    with pytest.raises(Exception, match="cpu_bytes_to_use must be specified"):
        _create_spec(cpu_bytes_to_use=None)


def test_duplicate_registration_raises():
    with pytest.raises(ValueError, match="is already registered"):
        OffloadingSpecFactory.register_spec(
            "CPUOffloadingSpec", "some.module", "SomeClass"
        )


def test_build_metric_definitions_below_threshold():
    from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics

    extra_config = {"store_threshold": 1}
    spec_cls = OffloadingSpecFactory.get_spec_cls({"spec_name": "CPUOffloadingSpec"})
    metrics = spec_cls.build_metric_definitions(extra_config)

    assert CPUOffloadingMetrics.STORES_SKIPPED not in metrics
    assert CPUOffloadingMetrics.CPU_ALLOCATION_SIZE in metrics


def test_build_metric_definitions_allocation_size_histogram():
    from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics

    spec_cls = OffloadingSpecFactory.get_spec_cls({"spec_name": "CPUOffloadingSpec"})
    metrics = spec_cls.build_metric_definitions({})
    metadata = metrics[CPUOffloadingMetrics.CPU_ALLOCATION_SIZE]

    assert isinstance(metadata, OffloadingHistogramMetadata)
    assert metadata.buckets == (
        1,
        4,
        16,
        64,
        256,
        1024,
        4096,
        16384,
        65536,
        262144,
    )


def test_build_metric_definitions_returns_counter_at_threshold():
    from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics

    extra_config = {"store_threshold": 2}
    spec_cls = OffloadingSpecFactory.get_spec_cls({"spec_name": "CPUOffloadingSpec"})
    metrics = spec_cls.build_metric_definitions(extra_config)

    assert CPUOffloadingMetrics.STORES_SKIPPED in metrics
