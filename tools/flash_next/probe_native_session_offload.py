"""CPU-only native-offload compatibility audit for the Flash session prototype.

Exercise the installed native classes, not replacement implementations. Tensor
registration uses a small CPU stand-in; it does not test DMA or model outputs.
Known incompatibilities are reported as gaps, never as qualification passes.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    get_sliding_window_size_in_chunks,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.kv_offload.base import LookupResult, ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager


def native_manager_probe():
    manager = CPUOffloadingManager(2, enable_events=True)
    context = ReqContext("audit")
    a, b, c = [make_offload_key(bytes([i]), 0) for i in range(3)]
    manager.prepare_store([a, b], context)
    assert manager.lookup(a, context) is LookupResult.HIT_PENDING
    manager.complete_store([a, b], context)
    lease = manager.prepare_load([a], context)
    replacement = manager.prepare_store([c], context)
    assert replacement is not None and replacement.evicted_keys == [b]
    assert manager.lookup(a, context) is LookupResult.HIT
    assert manager.lookup(b, context) is LookupResult.MISS
    assert manager.lookup(c, context) is LookupResult.HIT_PENDING
    # A is being read and C is being written: neither slot can be recycled.
    refusal = manager.prepare_store([b], context)
    assert refusal is None
    manager.complete_store([c], context, success=False)
    assert manager.lookup(c, context) is LookupResult.MISS
    manager.complete_load([a], context)
    return dict(
        pending_store_not_readable=True,
        in_flight_load_protected=True,
        idle_lru_eviction_reported=True,
        capacity_refused_while_both_slots_busy=True,
        failed_store_removed=True,
        held_chunk_ids=lease.chunk_ids.tolist(),
        scope="native manager bookkeeping; no tensor transfer",
    )


def flash_layout_probe(head_dim, local_kv_heads):
    parallel = NS(
        world_size=4,
        tensor_parallel_size=2,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
    )
    config = NS(
        parallel_config=parallel,
        cache_config=NS(enable_prefix_caching=False, block_size=3504),
        kv_transfer_config=NS(engine_id="native-audit", kv_connector_extra_config={}),
    )
    mamba = MambaSpec(
        block_size=240000,
        shapes=((2, 4), (2, 2, 4)),
        dtypes=(torch.bfloat16, torch.float32),
        num_speculative_blocks=2,
        mamba_cache_mode="none",
    )
    qsa = FullAttentionSpec(
        block_size=3504,
        num_kv_heads=local_kv_heads,
        head_size=head_dim,
        dtype=torch.bfloat16,
        num_head_slots=1,
        state_content_bytes=2,
    )
    ring = CircularBufferSpec(
        block_size=8, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
    )
    groups = [
        KVCacheGroupSpec([name], spec)
        for name, spec in (("gdn", mamba), ("qsa", qsa), ("raw", ring))
    ]
    kv_config = KVCacheConfig(num_blocks=2, kv_cache_tensors=[], kv_cache_groups=groups)
    report = dict(group_token_sizes=[g.kv_cache_spec.block_size for g in groups])
    try:
        build_offloading_config(config, kv_config)
    except AssertionError as exc:
        report["current_none_mode_registration_gap"] = str(exc)
    else:
        raise AssertionError("expected current mixed-block hashing incompatibility")

    assert get_sliding_window_size_in_chunks(mamba, 240000) == 1
    try:
        get_sliding_window_size_in_chunks(ring, 8)
    except AssertionError:
        report["raw_ring_window_gap"] = "CircularBufferSpec is not handled"
    else:
        raise AssertionError("native circular-buffer support changed; review adapter")

    # Same placeholder/host split as QSA bind_kv_cache, on CPU for this audit.
    gpu_placeholder = torch.zeros((2, 1, 3504, 1), dtype=torch.bfloat16)
    host_backing = torch.full(
        (2, local_kv_heads, 3504, 2 * head_dim), 17, dtype=torch.bfloat16
    )
    captured = []
    worker = object.__new__(OffloadingConnectorWorker)
    worker.vllm_config = config
    worker.kv_cache_config = KVCacheConfig(
        num_blocks=2, kv_cache_tensors=[], kv_cache_groups=[groups[1]]
    )
    worker._init_worker = captured.append
    worker.register_kv_caches({"qsa": gpu_placeholder})
    layout = captured[0]
    assert all(t.tensor.data_ptr() != host_backing.data_ptr() for t in layout.tensors)
    registered_bytes = sum(t.page_size_bytes for t in layout.tensors)
    real_bytes = host_backing[0].numel() * host_backing.element_size()
    assert registered_bytes == 3504 * 2
    assert real_bytes == local_kv_heads * 2 * head_dim * registered_bytes
    report["qsa_registration"] = dict(
        registered_bytes_per_block=registered_bytes,
        real_host_bytes_per_block=real_bytes,
        real_host_backing_registered=False,
        gpu_placeholder_registered=True,
        scope="one QSA group CPU stand-in; not full packed model registration",
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    args = parser.parse_args()
    assert not torch.cuda.is_initialized(), "CPU-only compatibility audit"
    model_bytes = args.model_config.read_bytes()
    model = json.loads(model_bytes)
    model = model.get("text_config", model)
    head_dim = model["head_dim"]
    total_kv_heads = model["num_key_value_heads"]
    assert total_kv_heads % 2 == 0, "this probe models the current TP2 layout"
    report = dict(
        native_manager=native_manager_probe(),
        flash_gaps=flash_layout_probe(head_dim, total_kv_heads // 2),
        model_geometry=dict(
            head_dim=head_dim,
            total_kv_heads=total_kv_heads,
            tensor_parallel_size=2,
            config_sha256=hashlib.sha256(model_bytes).hexdigest(),
        ),
        native_integration_qualified=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
