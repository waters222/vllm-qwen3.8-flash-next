# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native CPU allocator gate; requires the installed engine's dependencies.

No model weights or GPU allocations. This validates manager bookkeeping and
allocation reuse, not tensor contents or generated output equivalence.
"""

import importlib.util
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
        get_offloading_group_ids,
    )
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import (
        generate_scheduler_kv_cache_config,
        get_max_concurrency_for_kv_cache_config,
        get_request_block_hasher,
        init_none_hash,
    )
    from vllm.v1.core.single_type_kv_cache_manager import HiSparseSourceManager
    from vllm.v1.kv_cache_interface import (
        CircularBufferSpec,
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupRole,
        KVCacheGroupSpec,
        MambaSpec,
    )
    from vllm.v1.request import Request
except ImportError as exc:
    raise unittest.SkipTest("native engine dependencies required") from exc

PATH = Path(__file__).resolve().parents[2] / "vllm/v1/core/flash_session_allocator.py"
SPEC = importlib.util.spec_from_file_location("flash_session_allocator_tested", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class NativeSessionAllocatorTests(unittest.TestCase):
    @staticmethod
    def manager(depth, mode="none"):
        groups = [
            KVCacheGroupSpec(
                ["gdn"],
                MambaSpec(
                    block_size=16 if mode == "align" else 240000,
                    shapes=((2, 4), (2, 2, 4)),
                    dtypes=(torch.bfloat16, torch.float32),
                    num_speculative_blocks=depth,
                    mamba_cache_mode=mode,
                ),
            ),
            KVCacheGroupSpec(
                ["qsa"],
                FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=4,
                    dtype=torch.bfloat16,
                ),
            ),
            KVCacheGroupSpec(
                ["raw"],
                CircularBufferSpec(
                    block_size=4,
                    num_kv_heads=1,
                    head_size=4,
                    head_size_v=0,
                    dtype=torch.bfloat16,
                ),
            ),
        ]
        return KVCacheManager(
            KVCacheConfig(num_blocks=80, kv_cache_tensors=[], kv_cache_groups=groups),
            max_model_len=240000,
            scheduler_block_size=16 if mode == "align" else 240000,
            hash_block_size=16,
            enable_caching=False,
            max_in_flight_tokens=4096,
        )

    @staticmethod
    def request(name):
        return Request(name, [1] * 32, SamplingParams(max_tokens=512), None)

    def test_release_relocate_continue_and_reclaim_all_native_blocks(self):
        for depth in (0, 1, 2, 3, 4):
            with self.subTest(depth=depth):
                manager = self.manager(depth)
                allocator = MODULE.FlashSessionAllocator(manager)
                req = self.request("cold-a")
                self.assertIsNotNone(
                    manager.allocate_slots(req, 32, num_lookahead_tokens=depth)
                )
                manager.take_new_block_ids()
                req.num_computed_tokens = 32
                req.append_output_token_ids([1])
                layout = allocator.capture_layout(req.request_id)
                self.assertEqual(len(layout.block_ids[0]), depth + 1)
                old_ids = {i for ids in layout.block_ids for i in ids}
                allocator.release(req.request_id, layout)
                self.assertEqual(manager.block_pool.get_num_free_blocks(), 79)
                other = self.request("hot-b")
                self.assertIsNotNone(
                    manager.allocate_slots(other, 32, num_lookahead_tokens=depth)
                )
                manager.take_new_block_ids()
                occupied = {
                    i for ids in manager.get_block_ids(other.request_id) for i in ids
                }
                restored = allocator.reserve(req.request_id, layout)
                new_ids = {i for ids in restored.block_ids for i in ids}
                self.assertFalse(new_ids & occupied)
                self.assertNotEqual(old_ids, new_ids)
                self.assertEqual(
                    tuple(map(len, restored.block_ids)),
                    tuple(map(len, layout.block_ids)),
                )
                self.assertEqual(manager.take_new_block_ids(), [])
                for step in range(8):
                    self.assertIsNotNone(
                        manager.allocate_slots(
                            req, depth + 1, num_lookahead_tokens=depth
                        )
                    )
                    req.num_computed_tokens += step % (depth + 1) + 1
                    req.append_output_token_ids([1] * (step % (depth + 1) + 1))
                    manager.take_new_block_ids()
                manager.free(req)
                manager.free(other)
                self.assertEqual(manager.block_pool.get_num_free_blocks(), 79)

    def test_failed_reservation_does_not_consume_blocks_or_install_tables(self):
        manager = self.manager(4)
        allocator = MODULE.FlashSessionAllocator(manager)
        req = self.request("cold")
        self.assertIsNotNone(manager.allocate_slots(req, 32, num_lookahead_tokens=4))
        manager.take_new_block_ids()
        layout = allocator.capture_layout(req.request_id)
        allocator.release(req.request_id, layout)
        retained = manager.block_pool.get_new_blocks(78)
        before = manager.block_pool.get_num_free_blocks()
        with self.assertRaises(MemoryError):
            allocator.reserve(req.request_id, layout)
        self.assertEqual(manager.block_pool.get_num_free_blocks(), before)
        self.assertTrue(
            all(req.request_id not in g.req_to_blocks for g in allocator.groups)
        )
        manager.block_pool.free_blocks(retained)
        allocator.reserve(req.request_id, layout)
        manager.free(req)
        self.assertEqual(manager.block_pool.get_num_free_blocks(), 79)

    def test_align_relocation_preserves_next_allocation_and_retirement(self):
        """A swapped align request must evolve like an uninterrupted request."""
        for depth in range(5):
            with self.subTest(depth=depth):
                baseline = self.manager(depth, "align")
                swapped = self.manager(depth, "align")
                managers = (baseline, swapped)
                adapters = tuple(MODULE.FlashSessionAllocator(m) for m in managers)
                requests = tuple(self.request("a") for _ in managers)
                for manager, req in zip(managers, requests, strict=True):
                    self.assertIsNotNone(
                        manager.allocate_slots(req, 32, num_lookahead_tokens=depth)
                    )
                    manager.take_new_block_ids()
                    req.num_computed_tokens = 32
                    req.append_output_token_ids([1])
                for step in range(32):
                    if step % 4 == 0:
                        layout = adapters[1].capture_layout("a")
                        adapters[1].release("a", layout)
                        group = swapped.coordinator.single_type_managers[0]
                        self.assertNotIn("a", group._allocated_block_reqs)
                        self.assertNotIn("a", group.last_state_block_idx)
                        self.assertNotIn("a", group._num_retired_blocks)
                        occupied = swapped.block_pool.get_new_blocks(
                            layout.private_blocks
                        )
                        relocated = adapters[1].reserve("a", layout)
                        occupied_ids = {block.block_id for block in occupied}
                        self.assertFalse(
                            occupied_ids.intersection(
                                i for ids in relocated.block_ids for i in ids
                            )
                        )
                        self.assertEqual(relocated.align_states, layout.align_states)
                        swapped.block_pool.free_blocks(occupied)
                        self.assertEqual(swapped.take_new_block_ids(), [])
                    for manager, req in zip(managers, requests, strict=True):
                        self.assertIsNotNone(
                            manager.allocate_slots(
                                req, depth + 1, num_lookahead_tokens=depth
                            )
                        )
                        accepted = step % (depth + 1) + 1
                        req.num_computed_tokens += accepted
                        req.append_output_token_ids([1] * accepted)
                        manager.take_new_block_ids()
                    left, right = (a.capture_layout("a") for a in adapters)
                    self.assertEqual(left.align_states, right.align_states)
                    self.assertEqual(left.cached_counts, right.cached_counts)
                    self.assertEqual(
                        tuple(
                            tuple(i == left.null_block_id for i in t)
                            for t in left.block_ids
                        ),
                        tuple(
                            tuple(i == right.null_block_id for i in t)
                            for t in right.block_ids
                        ),
                    )
                    self.assertEqual(
                        baseline.block_pool.get_num_free_blocks(),
                        swapped.block_pool.get_num_free_blocks(),
                    )
                for manager, req in zip(managers, requests, strict=True):
                    manager.free(req)
                    self.assertEqual(manager.block_pool.get_num_free_blocks(), 79)


class NativeHostPoolAdmissionTests(unittest.TestCase):
    """Allocate native tables only when every owning pool has capacity.

    CPU manager tests catch partial allocation and wrong-pool accounting before
    worker binding exists. They do not claim RAM-KV integration or byte safety.
    """

    @staticmethod
    def manager(gpu_blocks=3, host_blocks=5, enable_caching=False):
        spec = FullAttentionSpec(
            block_size=16, num_kv_heads=1, head_size=4, dtype=torch.bfloat16
        )
        manager = KVCacheManager(
            KVCacheConfig(
                num_blocks=gpu_blocks,
                kv_cache_tensors=[],
                kv_cache_groups=[
                    KVCacheGroupSpec(["gpu"], spec),
                    *[
                        KVCacheGroupSpec(
                            [name],
                            spec,
                            host_resident=True,
                            role=KVCacheGroupRole.DIRECT_HOST,
                        )
                        for name in ("host-a", "host-b")
                    ],
                ],
                direct_host_num_blocks=host_blocks,
            ),
            max_model_len=256,
            scheduler_block_size=16,
            hash_block_size=16,
            enable_caching=enable_caching,
        )
        host_pool = manager.coordinator.single_type_managers[1].block_pool
        return manager, host_pool

    @staticmethod
    def request():
        return Request("host-admission", [1] * 32, SamplingParams(max_tokens=4), None)

    def test_host_pages_do_not_consume_device_admission_budget(self):
        manager, host = self.manager()
        request = self.request()
        self.assertIsNotNone(manager.allocate_slots(request, 32))
        self.assertEqual(manager.block_pool.get_num_free_blocks(), 0)
        self.assertEqual(host.get_num_free_blocks(), 0)
        gpu, host_a, host_b = manager.get_block_ids(request.request_id)
        self.assertEqual(tuple(map(len, (gpu, host_a, host_b))), (2, 2, 2))
        self.assertFalse(set(host_a) & set(host_b))
        manager.free(request)
        self.assertEqual(manager.block_pool.get_num_free_blocks(), 2)
        self.assertEqual(host.get_num_free_blocks(), 4)

    def test_shared_host_pool_exhaustion_refuses_before_any_allocation(self):
        for full_sequence in (False, True):
            with self.subTest(full_sequence=full_sequence):
                # Each host group fits separately; their sum must be checked.
                manager, host = self.manager(gpu_blocks=20, host_blocks=4)
                request = self.request()
                self.assertIsNone(
                    manager.allocate_slots(
                        request, 32, full_sequence_must_fit=full_sequence
                    )
                )
                self.assertEqual(manager.block_pool.get_num_free_blocks(), 19)
                self.assertEqual(host.get_num_free_blocks(), 3)
                self.assertEqual(
                    manager.get_block_ids(request.request_id), ([], [], [])
                )

    def test_device_exhaustion_does_not_acquire_host_pages(self):
        manager, host = self.manager(gpu_blocks=2)
        self.assertIsNone(manager.allocate_slots(self.request(), 32))
        self.assertEqual(manager.block_pool.get_num_free_blocks(), 1)
        self.assertEqual(host.get_num_free_blocks(), 4)

    def test_hisparse_reports_real_host_load_requirement(self):
        manager, host = self.manager()
        source = HiSparseSourceManager(
            manager.kv_cache_config.kv_cache_groups[1].kv_cache_spec,
            block_pool=host,
            enable_caching=False,
            kv_cache_group_id=1,
            scheduler_block_size=16,
        )
        self.assertEqual(
            source.get_num_blocks_to_allocate("load", 32, [], 32, 0, 32), 2
        )
        # GPU-computed HiSparse pages remain best effort, unchanged.
        self.assertEqual(
            source.get_num_blocks_to_allocate("prefill", 32, [], 0, 0, 32), 0
        )

    def test_native_deferred_free_returns_blocks_to_their_owning_pools(self):
        manager, host = self.manager()
        request = self.request()
        self.assertIsNotNone(manager.allocate_slots(request, 32))
        blocks = manager.pop_blocks_for_free(request)
        self.assertEqual(manager.get_block_ids(request.request_id), ([], [], []))
        self.assertEqual(manager.block_pool.get_num_free_blocks(), 0)
        self.assertEqual(host.get_num_free_blocks(), 0)
        self.assertTrue(all(block.ref_cnt == 1 for block in blocks))
        # This is the existing scheduler drain path, including native dispatch
        # by block.pool. Identical numeric IDs in RAM/GPU must remain distinct.
        manager.block_pool.free_blocks(reversed(blocks))
        self.assertEqual(manager.block_pool.get_num_free_blocks(), 2)
        self.assertEqual(host.get_num_free_blocks(), 4)
        self.assertTrue(all(block.ref_cnt == 0 for block in blocks))
        for pool in (host, manager.block_pool):
            allocated = pool.get_new_blocks(pool.get_num_free_blocks())
            self.assertTrue(all(block.pool is pool for block in allocated))
            pool.free_blocks(allocated)

    def test_direct_host_groups_do_not_enter_cpu_offload_transfer_set(self):
        manager, _ = self.manager()
        self.assertEqual(get_offloading_group_ids(manager.kv_cache_config), (0,))
        self.assertEqual(manager.kv_cache_config.prefix_cacheable_group_ids, (0, 1, 2))

    def test_direct_host_layout_rejects_missing_or_unusable_capacity(self):
        for capacity in (None, 0, 1):
            with (
                self.subTest(capacity=capacity),
                self.assertRaisesRegex(ValueError, "valid host layout"),
            ):
                self.manager(host_blocks=capacity)

    def test_host_prefix_survives_gpu_block_reuse_without_a_second_copy(self):
        init_none_hash(sha256)
        manager, host = self.manager(enable_caching=True)
        request = Request(
            "retained-prefix",
            list(range(33)),
            SamplingParams(max_tokens=4),
            None,
            block_hasher=get_request_block_hasher(16, sha256),
        )
        self.assertIsNotNone(manager.allocate_slots(request, 32))
        request.num_computed_tokens = 32
        manager.cache_blocks(request, 32)
        original = manager.get_block_ids(request.request_id)
        manager.free(request)
        reused_gpu = manager.block_pool.get_new_blocks(2)
        blocks, lengths = manager.coordinator.find_longest_cache_hit_per_group(
            request.block_hashes, 32
        )
        self.assertEqual(lengths, (0, 32, 32))
        for group in (1, 2):
            self.assertEqual([b.block_id for b in blocks[group]], original[group])
            self.assertTrue(all(b.pool is host for b in blocks[group]))
        manager.block_pool.free_blocks(reused_gpu)
        # A host eviction remains a miss: no stale pointer from the GPU pool
        # can resurrect the overwritten host prefix.
        overwritten_host = host.get_new_blocks(4)
        _, lengths = manager.coordinator.find_longest_cache_hit_per_group(
            request.block_hashes, 32
        )
        self.assertEqual(lengths, (0, 0, 0))
        host.free_blocks(overwritten_host)

    def test_unitary_host_prefix_hit_reuses_state_with_fresh_sampling_settings(self):
        init_none_hash(sha256)
        base, _ = self.manager()
        manager = KVCacheManager(
            replace(
                base.kv_cache_config,
                num_blocks=1,
                direct_host_num_blocks=4,
                kv_cache_groups=[base.kv_cache_config.kv_cache_groups[1]],
            ),
            max_model_len=256,
            scheduler_block_size=16,
            hash_block_size=16,
            enable_caching=True,
        )
        requests = [
            Request(
                str(seed),
                list(range(33)),
                SamplingParams(max_tokens=4, temperature=temperature, seed=seed),
                None,
                block_hasher=get_request_block_hasher(16, sha256),
            )
            for seed, temperature in ((173, 0), (11, 0.8))
        ]
        first, second = requests
        self.assertIsNotNone(manager.allocate_slots(first, 32))
        first.num_computed_tokens = 32
        manager.cache_blocks(first, 32)
        expected = manager.get_block_ids(first.request_id)[0]
        manager.free(first)
        computed, length, _ = manager.get_computed_blocks(second)
        self.assertEqual(length, 32)
        self.assertEqual(computed.get_block_ids()[0], expected)
        self.assertIsNotNone(manager.allocate_slots(second, 1, length, computed))
        self.assertEqual(second.sampling_params.seed, 11)
        self.assertEqual(second.sampling_params.temperature, 0.8)
        self.assertEqual(second.num_output_tokens, 0)
        manager.free(second)

    def test_offered_host_groups_are_pinned_before_deadline_crossing_allocation(self):
        """Native two-phase admission protects both host groups before allocating."""
        clock = [0.0]
        with patch("vllm.v1.hisparse.block_pool.monotonic", lambda: clock[0]):
            init_none_hash(sha256)
            manager, host = self.manager(
                gpu_blocks=4, host_blocks=7, enable_caching=True
            )
            requests = [
                Request(
                    name,
                    list(range(33)),
                    SamplingParams(max_tokens=4),
                    None,
                    block_hasher=get_request_block_hasher(16, sha256),
                )
                for name in ("producer", "consumer", "expired")
            ]
            first, second, expired = requests
            self.assertIsNotNone(manager.allocate_slots(first, 32))
            first.num_computed_tokens = 32
            manager.cache_blocks(first, 32)
            manager.free(first)
            clock[0] = 3599.99
            computed, length, _ = manager.get_computed_blocks(second)
            self.assertEqual(length, 32)
            offered = [
                (block, block.block_hash)
                for group in computed.blocks
                for block in group
            ]
            clock[0] = 3600.01
            self.assertIsNotNone(manager.allocate_slots(second, 1, length, computed))
            self.assertTrue(
                all(
                    block.ref_cnt == 1 and block.block_hash == key
                    for block, key in offered
                )
            )
            self.assertEqual(host.expire_idle(), 0)
            for index, group in enumerate(manager.get_blocks(second.request_id).blocks):
                self.assertEqual(group[:2], list(computed.blocks[index]))
                self.assertNotIn(group[2], computed.blocks[index])
            manager.free(second)
            clock[0] = 7200.02
            self.assertEqual(manager.get_computed_blocks(expired)[1], 0)
            self.assertEqual(host.get_num_free_blocks(), 6)
            self.assertEqual(manager.block_pool.get_num_free_blocks(), 3)

    def test_c4_ownership_capture_covers_all_branches_without_changing_ownership(self):
        from prefix_load_barrier import collect_scheduled_resident_ownership

        init_none_hash(sha256)
        manager, host = self.manager(gpu_blocks=20, host_blocks=20, enable_caching=True)
        requests = [Request(
            f"turn4-hot-0-{i}", list(range(33)), SamplingParams(max_tokens=4), None,
            block_hasher=get_request_block_hasher(16, sha256), cache_salt="shared",
        ) for i in range(4)]
        for request in requests:
            self.assertIsNotNone(manager.allocate_slots(request, 32))
            request.num_computed_tokens = 32
            manager.cache_blocks(request, 32)
            request.num_computed_tokens = 33
        scheduler = NS(kv_cache_manager=manager,
                       requests={r.request_id: r for r in requests})
        output = NS(num_scheduled_tokens={r.request_id: 1 for r in requests})
        before = [(b.block_id, b.ref_cnt, b.block_hash) for b in host.blocks]
        records = [dict(request_id=f"prior-{i}", boundary=32) for i in range(60)]
        self.assertEqual(collect_scheduled_resident_ownership(
            scheduler, output, {"turn4-hot-0": [32]}, records), 4)
        self.assertEqual({r["request_id"] for r in records[-4:]}, set(scheduler.requests))
        self.assertTrue(all(r["passed"] for r in records[-4:]))
        self.assertEqual(before, [(b.block_id, b.ref_cnt, b.block_hash) for b in host.blocks])
        with self.assertRaisesRegex(ValueError, "excess"):
            collect_scheduled_resident_ownership(
                scheduler, output, {"turn4-hot-0": [32]}, records)
        with self.assertRaisesRegex(ValueError, "C2/C4"):
            collect_scheduled_resident_ownership(
                scheduler, output, {"turn4-hot-0": [32], "turn2-hot-0": [32]}, [])

    def test_resident_ownership_audit_accepts_duplicate_versions_not_wrong_prefix(self):
        from prefix_load_barrier import (
            collect_scheduled_resident_ownership,
            snapshot_resident_prefix_ownership,
        )

        init_none_hash(sha256)
        manager, host = self.manager(gpu_blocks=20, host_blocks=20, enable_caching=True)
        requests = [
            Request(
                f"turn2-hot-0-{i}",
                list(range(33)),
                SamplingParams(max_tokens=4),
                None,
                block_hasher=get_request_block_hasher(16, sha256),
                cache_salt="same-prefix",
            )
            for i in range(2)
        ]
        # Two fresh computations of the same input legitimately have distinct pages.
        for request in requests:
            self.assertIsNotNone(manager.allocate_slots(request, 32))
            request.num_computed_tokens = 32
            manager.cache_blocks(request, 32)
        scheduler = NS(
            kv_cache_manager=manager, requests={r.request_id: r for r in requests}
        )
        before = [(b.block_id, b.ref_cnt, b.block_hash) for b in host.blocks]
        reports = [
            snapshot_resident_prefix_ownership(scheduler, r.request_id, 32)
            for r in requests
        ]
        self.assertTrue(all(r["passed"] for r in reports))
        self.assertEqual(
            before, [(b.block_id, b.ref_cnt, b.block_hash) for b in host.blocks]
        )
        self.assertNotEqual(
            reports[0]["groups"][0]["pages"][0]["physical_id"],
            reports[1]["groups"][0]["pages"][0]["physical_id"],
        )
        second = requests[1]
        saved = second.block_hashes
        other = Request(
            "other",
            list(range(33)),
            SamplingParams(max_tokens=4),
            None,
            block_hasher=get_request_block_hasher(16, sha256),
            cache_salt="other-prefix",
        )
        second.block_hashes = other.block_hashes
        self.assertFalse(
            snapshot_resident_prefix_ownership(scheduler, second.request_id, 32)[
                "passed"
            ]
        )
        second.block_hashes = saved
        block = manager.get_blocks(second.request_id).blocks[1][0]
        key = block.block_hash
        self.assertIs(host.cached_block_hash_to_block.pop(key, block.block_id), block)
        self.assertFalse(
            snapshot_resident_prefix_ownership(scheduler, second.request_id, 32)[
                "passed"
            ]
        )
        host.cached_block_hash_to_block.insert(key, block)
        records = []
        output = NS(num_scheduled_tokens={r.request_id: 1 for r in requests})
        for request in requests:
            request.num_computed_tokens = 33
        self.assertEqual(
            collect_scheduled_resident_ownership(
                scheduler, output, {"turn2-hot-0": [32]}, records
            ),
            2,
        )
        self.assertTrue(all(row["passed"] for row in records))
        self.assertEqual([row["boundary"] for row in records], [32, 32])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            collect_scheduled_resident_ownership(
                scheduler, output, {"turn2-hot-0": [32]}, records
            )
        self.assertEqual(
            collect_scheduled_resident_ownership(
                scheduler, output, {"turn2-cold-0": [32]}, []
            ),
            0,
        )
        bounded = [dict(request_id=f"prior-{i}", boundary=32) for i in range(31)]
        self.assertEqual(
            collect_scheduled_resident_ownership(
                scheduler,
                NS(num_scheduled_tokens={requests[0].request_id: 1}),
                {"turn2-hot-0": [32]},
                bounded,
            ),
            1,
        )
        with self.assertRaisesRegex(ValueError, "excess"):
            collect_scheduled_resident_ownership(
                scheduler,
                NS(num_scheduled_tokens={requests[1].request_id: 1}),
                {"turn2-hot-0": [32]},
                bounded,
            )
        self.assertEqual(len(bounded), 32)
        self.assertEqual(
            before, [(b.block_id, b.ref_cnt, b.block_hash) for b in host.blocks]
        )
        for boundary in (True, 0, 31, 32768):
            with self.assertRaises(ValueError):
                snapshot_resident_prefix_ownership(
                    scheduler, second.request_id, boundary
                )
        for request in requests:
            manager.free(request)

    def test_capacity_reporting_and_rank_merge_use_the_direct_host_budget(self):
        manager, _ = self.manager()
        config = manager.kv_cache_config
        vllm_config = NS(
            model_config=NS(max_model_len=256),
            parallel_config=NS(decode_context_parallel_size=1),
        )
        self.assertEqual(
            get_max_concurrency_for_kv_cache_config(vllm_config, config), 5 / 32
        )
        self.assertEqual(
            generate_scheduler_kv_cache_config([config, config]).host_num_blocks, 5
        )
        with self.assertRaises(AssertionError):
            generate_scheduler_kv_cache_config(
                [config, replace(config, direct_host_num_blocks=6)]
            )


if __name__ == "__main__":
    assert not torch.cuda.is_initialized(), "CPU-only allocator gate"
    unittest.main()
