"""Exercise native offload DMA with explicit GPU + QSA mapped-host pages.

This is a transfer-only compatibility probe, not model/session qualification.
Use one isolated development GPU; no model weights or service are loaded.
"""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    LookupResult,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.worker.gpu.flash_session_swap import (
    BlockRegion,
    NativeSessionColdStore,
    SessionKey,
    SwapBoundary,
    native_group_cache_layouts,
)


def finish(worker, job_id):
    worker.wait({job_id})
    results = worker.get_finished()
    assert len(results) == 1 and results[0].job_id == job_id
    assert results[0].success
    return asdict(results[0])


def probe(report):
    torch.cuda.set_device(0)
    generator = torch.Generator().manual_seed(173)
    # A representative packed raw-state page and the actual TP2 QSA host
    # payload geometry: 3504 tokens * 1 local head * 2 K/V * 256 * BF16.
    raw_bytes, host_bytes = 1624064, 3588096
    raw = torch.empty((4, raw_bytes), dtype=torch.int8, device="cuda")
    host = torch.empty((4, host_bytes), dtype=torch.int8, pin_memory=True)
    host_cuda = get_accelerator_view_from_cpu_tensor(host)
    assert host_cuda.is_cuda and host.is_pinned()
    caches = CanonicalKVCaches(
        tensors=[
            CanonicalKVCacheTensor(raw, raw_bytes),
            CanonicalKVCacheTensor(host_cuda, host_bytes),
        ],
        group_data_refs=[
            [CanonicalKVCacheRef(0, raw_bytes)],
            [CanonicalKVCacheRef(0, raw_bytes), CanonicalKVCacheRef(1, host_bytes)],
        ],
    )
    report.update(
        gpu=torch.cuda.get_device_name(0),
        seed=173,
        raw_bytes_per_block=raw_bytes,
        host_bytes_per_block=host_bytes,
        native_cpu_allocation_bytes=2 * (raw_bytes + host_bytes),
        cycles=[],
    )
    worker = CPUOffloadingWorker(caches, blocks_per_chunk=1, num_cpu_chunks=2)
    manager = CPUOffloadingManager(2)
    context = ReqContext("transfer-probe")
    try:
        for cycle in range(3):
            raw_expected = torch.randint(
                -128, 128, raw.shape, dtype=torch.int8, generator=generator
            )
            host.random_(-128, 128, generator=generator)
            host_expected = host.clone()
            raw.copy_(raw_expected)
            keys = [
                make_offload_key(f"cycle-{cycle}".encode(), group) for group in range(2)
            ]
            pending = manager.prepare_store(keys, context)
            assert pending is not None and list(pending.keys_to_store) == keys
            assert all(
                manager.lookup(key, context) is LookupResult.HIT_PENDING for key in keys
            )
            store_id = 2 * cycle
            start = time.monotonic()
            assert worker.submit_store(
                store_id, GPULoadStoreSpec([1, 3], [1, 1], [0, 0]), pending.store_spec
            )
            store_result = finish(worker, store_id)
            manager.complete_store(keys, context)
            capture_ms = (time.monotonic() - start) * 1000

            # Dirty every hot page, including the original source blocks.
            raw.fill_(23)
            torch.cuda.synchronize()
            host.fill_(-41)
            lease = manager.prepare_load(keys, context)
            start = time.monotonic()
            assert worker.submit_load(
                store_id + 1, lease, GPULoadStoreSpec([2, 0], [1, 1], [0, 0])
            )
            load_result = finish(worker, store_id + 1)
            manager.complete_load(keys, context)
            restore_ms = (time.monotonic() - start) * 1000
            restored = raw.cpu()
            assert torch.equal(restored[2], raw_expected[1])
            assert torch.equal(restored[0], raw_expected[3])
            assert torch.equal(host[0], host_expected[3])
            assert torch.all(restored[[1, 3]] == 23)
            assert torch.all(host[1:] == -41)
            expected_transfer_bytes = 2 * raw_bytes + host_bytes
            assert store_result["transfer_size"] == expected_transfer_bytes
            assert load_result["transfer_size"] == expected_transfer_bytes
            report["cycles"].append(
                dict(
                    cycle=cycle,
                    byte_exact=True,
                    all_hot_sources_overwritten=True,
                    non_destinations_unchanged=True,
                    source_blocks=[1, 3],
                    destination_blocks=[2, 0],
                    capture_wall_ms=capture_ms,
                    restore_wall_ms=restore_ms,
                    store=store_result,
                    load=load_result,
                )
            )
    finally:
        worker.shutdown()
    report["transfer_probe_passed"] = True


def probe_group_adapter(report):
    raw = torch.empty((6, 1624064), dtype=torch.uint8, device="cuda")
    host = torch.empty((6, 3588096), dtype=torch.uint8, pin_memory=True)
    regions = (
        BlockRegion("state", 0, raw),
        BlockRegion("qsa-raw", 1, raw),
        BlockRegion("qsa-host", 1, host),
    )
    layouts = native_group_cache_layouts(regions, 3, empty_groups=(2,))
    assert layouts[2] is None
    assert len(layouts[0].tensors) == 1 and len(layouts[1].tensors) == 2
    assert layouts[0].tensors[0].tensor.data_ptr() == raw.data_ptr()
    # The host sidecar is a CUDA UVA alias, not a detached device copy.
    assert layouts[1].tensors[1].tensor.data_ptr() == host.data_ptr()
    rejected = []
    invalid = {
        "missing_group": (regions, 3, ()),
        "unregistered_empty_group": (regions, 3, (1, 2)),
        "duplicate_region": (regions + (regions[0],), 3, (2,)),
        "overlapping_region": (regions + (BlockRegion("alias", 0, raw),), 3, (2,)),
        "strided_region": ((BlockRegion("strided", 0, raw[:, ::2]),), 1, ()),
        "unpinned_host": (
            (
                regions[0],
                BlockRegion("host", 0, torch.empty((6, 8), dtype=torch.uint8)),
            ),
            1,
            (),
        ),
    }
    for label, (bad_regions, count, empty) in invalid.items():
        try:
            native_group_cache_layouts(bad_regions, count, empty_groups=empty)
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError(f"unsafe layout accepted: {label}")
    workers = []
    try:
        for layout in layouts[:2]:
            workers.append(CPUOffloadingWorker(layout, 1, 1))
        from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

        torch.manual_seed(173)
        raw.random_(0, 256)
        torch.cuda.synchronize()
        host.random_(0, 256)
        raw_expected, host_expected = raw.cpu(), host[3].clone()
        for worker, block_id in zip(workers, (1, 3), strict=True):
            assert worker.submit_store(
                0, GPULoadStoreSpec([block_id], [1], [0]), CPULoadStoreSpec([0])
            )
        stored = [finish(worker, 0) for worker in workers]
        raw.fill_(23)
        torch.cuda.synchronize()
        host.fill_(41)
        for worker, block_id in zip(workers, (4, 5), strict=True):
            assert worker.submit_load(
                1, CPULoadStoreSpec([0]), GPULoadStoreSpec([block_id], [1], [0])
            )
        loaded = [finish(worker, 1) for worker in workers]
        actual = raw.cpu()
        assert torch.equal(actual[4], raw_expected[1])
        assert torch.equal(actual[5], raw_expected[3])
        assert torch.equal(host[5], host_expected)
        assert torch.all(actual[:4] == 23) and torch.all(host[:5] == 41)
        allocation = sum(t.page_size_bytes for g in layouts[:2] for t in g.tensors)
        assert allocation == 2 * 1624064 + 3588096
        report["group_adapter"] = dict(
            byte_exact=True,
            non_destinations_unchanged=True,
            empty_group_without_pool=True,
            rejected_layouts=rejected,
            native_cpu_allocation_bytes=allocation,
            stores=stored,
            loads=loaded,
        )
    finally:
        for worker in workers:
            worker.shutdown()


def probe_checkpoint_backend(report):
    raw = torch.empty((16, 1624064), dtype=torch.uint8, device="cuda")
    host = torch.empty((16, 3588096), dtype=torch.uint8, pin_memory=True)
    regions = (
        BlockRegion("state", 0, raw),
        BlockRegion("qsa-raw", 1, raw),
        BlockRegion("qsa-host", 1, host),
    )
    raw.random_(0, 256)
    torch.cuda.synchronize()
    host.random_(0, 256)
    expected_raw, expected_host = raw.cpu(), host.clone()
    metadata = b'{"computed":120,"pending_drafts":[1,2]}'
    boundary = SwapBoundary(120, 0)
    first, second, third = [SessionKey(0, f"session-{i}", 0) for i in range(3)]
    source = ((1, 2), (0, 3))
    destination = ((7, 8), (0, 9))
    payload = 3 * raw.shape[1] + host.shape[1] + len(metadata)
    store = NativeSessionColdStore(regions, 2, 2 * payload, 3)

    def must_fail(exc_type, action):
        try:
            action()
        except exc_type:
            return
        raise AssertionError(f"expected {exc_type.__name__}")

    try:
        assert store.capture(first, source, boundary, metadata) == payload
        assert store.capture(second, ((4, 5), (0, 6)), boundary, metadata) == payload
        assert store.used_bytes == 2 * payload
        must_fail(
            MemoryError,
            lambda: store.capture(third, ((10, 11), (0, 12)), boundary, metadata),
        )
        assert store.contains(first) and store.contains(second)
        store.verify_hot(first, source)
        raw.fill_(23)
        torch.cuda.synchronize()
        host.fill_(41)
        assert store.restore(first, destination) == (boundary, metadata)
        assert torch.equal(raw[7].cpu(), expected_raw[1])
        assert torch.equal(raw[8].cpu(), expected_raw[2])
        assert torch.equal(raw[9].cpu(), expected_raw[3])
        assert torch.equal(host[9], expected_host[3])
        assert torch.all(raw[0] == 23) and torch.all(host[0] == 41)
        assert store.restore(second, ((10, 11), (0, 12))) == (boundary, metadata)
        assert torch.equal(raw[10].cpu(), expected_raw[4])
        assert torch.equal(host[12], expected_host[6])
        # One cold byte is deliberately corrupted. Rejection must precede any
        # destination write, while explicit retirement must still be possible.
        store._checkpoints[second].regions[0].data[0, 0] ^= 1
        before = raw[10].clone()
        must_fail(RuntimeError, lambda: store.restore(second, ((10, 11), (0, 12))))
        assert torch.equal(before, raw[10])
        assert store.drop(second) and not store.drop(second)
        assert store.used_bytes == payload and len(store._native_pools) == 1
        # Fail after submission: native work must drain, and the checkpoint
        # must survive so a clean retry restores the same bytes.
        with patch.object(store, "_finish", side_effect=RuntimeError("injected ack")):
            must_fail(RuntimeError, lambda: store.restore(first, destination))
        assert store.contains(first) and not store._storage_failed
        assert store.restore(first, destination) == (boundary, metadata)
        assert store.drop(first)
        assert store.used_bytes == 0 and not store._native_pools
        with patch.object(store, "_finish", side_effect=RuntimeError("injected ack")):
            must_fail(
                RuntimeError, lambda: store.capture(third, source, boundary, metadata)
            )
        assert not store.contains(third) and not store._native_pools
        store.abort_capture(third)
        assert store.capture(third, source, boundary, metadata) == payload
        assert store.drop(third)
        report["checkpoint_backend"] = dict(
            passed=True,
            checks=[
                "two independent native checkpoints",
                "capacity refusal without eviction",
                "exact GPU and host bytes after destructive reuse",
                "metadata and null positions preserved",
                "corruption refused before destination writes",
                "idempotent retirement and zero remaining payload",
                "failed load acknowledgement drained; exact retry",
                "failed capture acknowledgement drained; no published checkpoint",
            ],
            payload_bytes_per_checkpoint=payload,
            final_used_bytes=store.used_bytes,
            final_native_pools=len(store._native_pools),
            scope="rank-local real CUDA storage; no model, scheduler, or TTL clock",
        )
    finally:
        for key in list(store._native_pools):
            if store.contains(key):
                store.drop(key)
            else:
                store.abort_capture(key)


def probe_direct_host_binding(report):
    """Use real native allocation, QSA writes/reads, zeroing and CPU offload."""
    from dataclasses import replace

    from vllm.config import CacheConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
        OffloadingConnectorWorker,
    )

    # Match normal model import order: qsa and model refer to each other.
    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpQSAAttention
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention
    from vllm.models.qwen4_exp.nvidia.qsa import (
        Qwen4ExpQSAFlashAttentionImpl,
    )
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy, get_kv_cache_configs
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheLayout,
    )
    from vllm.v1.worker.utils import (
        KVBlockZeroer,
        allocate_kv_cache,
        copy_kv_cache_blocks_inplace,
    )

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    block_size, host_blocks, gpu_blocks, head_dim = 3504, 7, 4, 256
    state_spec = FullAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=2, dtype=torch.bfloat16
    )
    cache_options = CacheConfig(block_size=block_size)
    cache_options.kv_cache_layout = "BLHNC"
    startup_config = NS(
        cache_config=cache_options,
        model_config=NS(
            max_model_len=block_size,
            original_max_model_len=block_size,
            enforce_eager=True,
        ),
        parallel_config=NS(data_parallel_size=1, decode_context_parallel_size=1),
        scheduler_config=NS(disable_hybrid_kv_cache_manager=False),
        attention_config=NS(hisparse_config=None),
        speculative_config=None,
        additional_config={
            "flash_next_direct_host_kv": {
                "num_blocks": host_blocks,
                "max_bytes": 64 * 2**20,
            }
        },
    )
    host_spec = Qwen4ExpQSAAttention.get_kv_cache_spec(
        NS(
            _qsa_kv_offload=True,
            num_kv_heads=1,
            head_dim=head_dim,
            kv_cache_torch_dtype=torch.bfloat16,
            kv_cache_dtype="auto",
        ),
        startup_config,
    )
    gpu_page, host_page = state_spec.page_size_bytes, host_spec.page_size_bytes
    config = get_kv_cache_configs(
        startup_config,
        [{"state": state_spec, "qsa0": host_spec, "qsa1": host_spec}],
        [gpu_blocks * gpu_page],
    )[0]
    assert config.num_blocks == gpu_blocks
    assert config.direct_host_num_blocks == host_blocks
    report["layout_source"] = "QSA real specs through native startup planner"
    # Budget refusals must happen before either pool is allocated.
    with patch("torch.zeros", side_effect=AssertionError("allocation before gate")):
        for budget in (None, 1):
            try:
                allocate_kv_cache(
                    replace(config, direct_host_max_bytes=budget),
                    device,
                    KVCacheLayout.BLHNC,
                    [block_size] * 2,
                )
            except ValueError:
                pass
            else:
                raise AssertionError("missing host allocation budget refusal")
    caches = allocate_kv_cache(config, device, KVCacheLayout.BLHNC, [block_size] * 2)
    host0, host1 = caches["qsa0"], caches["qsa1"]
    assert host0.is_pinned() and host1.is_pinned()
    assert host0.untyped_storage().data_ptr() == host1.untyped_storage().data_ptr()
    assert host0.shape == (host_blocks, 1, block_size, 2 * head_dim)
    assert host0.stride(0) == 2 * host_page // 2
    layer = object.__new__(Qwen4ExpQSAAttention)
    torch.nn.Module.__init__(layer)
    layer._qsa_kv_offload = True
    layer.num_kv_heads, layer.head_dim = 1, head_dim
    layer._k_scale = layer._v_scale = torch.ones((), device=device)
    layer.bind_kv_cache(host0)
    assert layer._qsa_host_kv is host0 and layer._qsa_gpu_slots is None
    assert layer.kv_cache.data_ptr() == host0.data_ptr()
    assert layer.kv_cache.stride() == host0.stride()
    host0.fill_(-3)
    host1.fill_(7)
    torch.manual_seed(173)
    key = torch.randn((16, 1, head_dim), dtype=torch.bfloat16, device=device)
    value = torch.randn_like(key)
    slots = torch.arange(17, 33, device=device, dtype=torch.int64) + 5 * block_size
    Qwen4ExpQSAFlashAttentionImpl.do_kv_cache_update(
        NS(attn_type=AttentionType.DECODER, head_size=head_dim, kv_cache_dtype="auto"),
        layer,
        key,
        value,
        layer.kv_cache,
        slots,
    )
    torch.cuda.synchronize()
    assert torch.equal(
        host0[5, 0, 17:33, :head_dim].view(torch.uint8),
        key[:, 0].cpu().view(torch.uint8),
    )
    assert torch.equal(
        host0[5, 0, 17:33, head_dim:].view(torch.uint8),
        value[:, 0].cpu().view(torch.uint8),
    )
    assert bool((host1 == 7).all())
    host_snapshots = [host0.clone(), host1.clone()]
    gpu = caches["state"]
    gpu.fill_(5)
    zeroer = KVBlockZeroer(
        device,
        attn_groups_iter=[
            NS(
                kv_cache_spec=g.kv_cache_spec,
                layer_names=g.layer_names,
                kv_cache_group_id=i,
            )
            for i, g in enumerate(config.kv_cache_groups)
        ],
        kernel_block_sizes=[block_size] * 2,
        static_forward_context={
            "state": NS(kv_cache=gpu),
            "qsa0": layer,
            "qsa1": NS(kv_cache=get_accelerator_view_from_cpu_tensor(host1)),
        },
        num_blocks=gpu_blocks,
        host_group_ids=config.host_group_ids,
    )
    zeroer.zero_block_ids([1])
    torch.cuda.synchronize()
    assert bool((gpu[1] == 0).all()) and bool((gpu[0] == 5).all())
    gpu[2].fill_(19)
    device_caches = [cache for cache in caches.values() if cache.device == device]
    copy_kv_cache_blocks_inplace(device_caches, gpu_blocks, [KVCacheBlockCopy(2, 1)])
    torch.cuda.synchronize()
    assert torch.equal(gpu[1], gpu[2])

    captured = []
    registration = object.__new__(OffloadingConnectorWorker)
    registration.kv_cache_config = config
    registration.vllm_config = NS(
        parallel_config=NS(
            world_size=4,
            tensor_parallel_size=2,
            prefill_context_parallel_size=1,
        )
    )
    registration._init_worker = captured.append
    registration.register_kv_caches(caches)
    native_layout = captured[0]
    assert len(native_layout.tensors) == 1
    assert native_layout.tensors[0].page_size_bytes == gpu_page
    worker = CPUOffloadingWorker(native_layout, blocks_per_chunk=1, num_cpu_chunks=1)
    manager = CPUOffloadingManager(1)
    context = ReqContext("direct-host-test")
    cache_key = make_offload_key(b"state", 0)
    try:
        pending = manager.prepare_store([cache_key], context)
        assert worker.submit_store(
            0, GPULoadStoreSpec([1], [1], [0]), pending.store_spec
        )
        stored = finish(worker, 0)
        manager.complete_store([cache_key], context)
        lease = manager.prepare_load([cache_key], context)
        gpu.fill_(0)
        assert worker.submit_load(1, lease, GPULoadStoreSpec([3], [1], [0]))
        restored = finish(worker, 1)
        manager.complete_load([cache_key], context)
        assert bool((gpu[3] == 19).all())
        assert stored["transfer_size"] == restored["transfer_size"] == gpu_page
    finally:
        worker.shutdown()
    torch.cuda.synchronize()
    for host, expected in zip((host0, host1), host_snapshots):
        assert torch.equal(host.view(torch.uint8), expected.view(torch.uint8))

    query = torch.randn((1, 32, head_dim), dtype=torch.bfloat16, device=device)
    indices = torch.tensor(
        [list(range(17, 33)) + [16]], dtype=torch.int32, device=device
    )
    table = torch.tensor([[5]], dtype=torch.int32, device=device)
    req_ids = torch.zeros(1, dtype=torch.int32, device=device)
    key_host, value_host = layer.kv_cache.transpose(1, 2).split(head_dim, dim=-1)
    reference = host0.to(device)
    key_gpu, value_gpu = reference.transpose(1, 2).split(head_dim, dim=-1)
    actual = qsa_sparse_paged_attention(
        query, key_host, value_host, indices, table, req_ids, False
    )
    expected = qsa_sparse_paged_attention(
        query, key_gpu, value_gpu, indices, table, req_ids, False
    )
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert bool(torch.isfinite(actual).all())
    report["direct_host_binding"] = dict(
        passed=True,
        host_blocks=host_blocks,
        gpu_blocks=gpu_blocks,
        head_dim=head_dim,
        block_size=block_size,
        seed=173,
        allocated_host_bytes=host0.untyped_storage().nbytes(),
        native_gpu_transfer_bytes=gpu_page,
        uva_alias_no_copy=True,
        raw_byte_comparisons=True,
        native_qsa_scatter_exact=True,
        gpu_zero_copy_offload_leave_host_unchanged=True,
        sparse_attention_matches_device_reference=True,
        scope="native allocation and one QSA cache path; no full model or scheduler",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--direct-host", action="store_true")
    args = parser.parse_args()
    report = dict(
        scope="native transfer-only; explicit layout, no model or scheduler",
        transfer_probe_passed=False,
        session_integration_qualified=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    with args.output.open("x") as output:
        try:
            if args.direct_host:
                probe_direct_host_binding(report)
            else:
                probe(report)
                probe_group_adapter(report)
                probe_checkpoint_backend(report)
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            json.dump(report, output, indent=2)
            output.write("\n")
            print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
