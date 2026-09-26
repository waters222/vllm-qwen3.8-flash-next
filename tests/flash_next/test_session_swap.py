# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-tensor CPU tests for transactional cold storage, without engine imports.

Contract: preserve all registered bytes plus metadata across hot block reuse
and relocation. Cheap failure gates: no release on failed capture; no runnable
state on failed restore; bounded capacity and scoped request identity. These
tests do not qualify distributed scheduling, CUDA ordering, or model outputs.
"""

import importlib.util
import json
import struct
import sys
import unittest
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch

PATH = Path(__file__).resolve().parents[2] / "vllm/v1/worker/gpu/flash_session_swap.py"
SPEC = importlib.util.spec_from_file_location("flash_session_swap_tested", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
BlockRegion = MODULE.BlockRegion
SessionColdStore = MODULE.SessionColdStore
SessionKey = MODULE.SessionKey
SwapBoundary = MODULE.SwapBoundary
sys.modules["vllm.v1.worker.gpu.flash_session_swap"] = MODULE
WORKER_SPEC = importlib.util.spec_from_file_location(
    "flash_session_worker_tested", PATH.with_name("flash_session_worker.py")
)
WORKER = importlib.util.module_from_spec(WORKER_SPEC)
WORKER_SPEC.loader.exec_module(WORKER)
TX_SPEC = importlib.util.spec_from_file_location(
    "flash_session_transactions_tested",
    PATH.parents[2] / "core/flash_session_transactions.py",
)
TX = importlib.util.module_from_spec(TX_SPEC)
sys.modules[TX_SPEC.name] = TX
TX_SPEC.loader.exec_module(TX)


class SessionSwapTests(unittest.TestCase):
    def setUp(self):
        self.hot = torch.arange(16 * 24, dtype=torch.int64).to(torch.uint8).view(16, 24)
        self.ram = torch.arange(16 * 40, dtype=torch.int64).to(torch.uint8).view(16, 40)
        self.regions = (
            BlockRegion("states", 0, self.hot),
            BlockRegion("index", 1, self.hot),
            BlockRegion("host-kv", 1, self.ram),
        )
        self.store = SessionColdStore(self.regions, 2, 4096, 8)
        self.key = SessionKey(0, "request-a", 1)
        self.boundary = SwapBoundary(120, 0)
        self.source = ((1, 2, 3), (0, 4, 5))
        self.destination = ((6, 7, 8), (0, 9, 10))
        self.metadata = b'{"accepted":3,"computed":120,"pending_drafts":[1,2]}'

    def capture(self, key=None):
        return self.store.capture(
            key or self.key, self.source, self.boundary, self.metadata
        )

    def test_resume_after_reuse_preserves_gpu_pages_host_kv_and_metadata(self):
        hot, ram = self.hot.clone(), self.ram.clone()
        size = self.capture()
        self.assertEqual(size, 3 * 24 + 2 * (24 + 40) + len(self.metadata))
        self.hot.fill_(231)
        self.ram.fill_(117)  # Another request reused the original hot IDs.
        boundary, metadata = self.store.restore(self.key, self.destination)
        self.assertEqual((boundary, metadata), (self.boundary, self.metadata))
        for sources, targets in zip(self.source, self.destination):
            for src, dst in zip(sources, targets):
                if src:
                    torch.testing.assert_close(self.hot[dst], hot[src], rtol=0, atol=0)
        for src, dst in ((4, 9), (5, 10)):
            torch.testing.assert_close(self.ram[dst], ram[src], rtol=0, atol=0)
        self.assertTrue(bool((self.hot[0] == 231).all()))
        self.assertTrue(bool((self.ram[0] == 117).all()))
        self.assertEqual(self.store.used_bytes, size)  # Wait for all-rank commit.
        self.assertTrue(self.store.drop(self.key))
        self.assertEqual(self.store.used_bytes, 0)
        self.assertFalse(self.store.drop(self.key))

    def test_repeated_swaps_relocate_without_leaks(self):
        for cycle in range(20):
            self.capture()
            self.store.restore(self.key, self.destination)
            self.store.drop(self.key)
            self.source, self.destination = self.destination, self.source
            self.assertEqual(self.store.session_count, 0)
            self.assertEqual(self.store.used_bytes, 0)

    def test_backend_retirement_failure_retains_checkpoint_and_budget(self):
        size = self.capture()
        with (
            patch.object(
                self.store, "_release_storage", side_effect=RuntimeError("drain failed")
            ),
            self.assertRaises(RuntimeError),
        ):
            self.store.drop(self.key)
        self.assertTrue(self.store.contains(self.key))
        self.assertEqual(self.store.used_bytes, size)
        self.assertTrue(self.store.drop(self.key))
        self.assertEqual(self.store.used_bytes, 0)

    def test_failed_capture_releases_backend_before_leaving_hot_state(self):
        before = self.hot.clone()
        with (
            patch.object(
                self.store, "_capture_regions", side_effect=RuntimeError("copy failed")
            ),
            patch.object(self.store, "_release_storage") as release,
            self.assertRaises(RuntimeError),
        ):
            self.capture()
        release.assert_called_once_with(self.key)
        self.assertFalse(self.store.contains(self.key))
        self.assertEqual(self.store.used_bytes, 0)
        torch.testing.assert_close(self.hot, before, rtol=0, atol=0)

    def test_capacity_failure_preserves_hot_bytes_and_other_cold_sessions(self):
        size = self.capture()
        self.store.capacity_bytes = size
        before = self.hot.clone()
        with self.assertRaises(MemoryError):
            self.capture(SessionKey(0, "request-b", 1))
        self.assertEqual(self.store.session_count, 1)
        self.assertEqual(self.store.used_bytes, size)
        torch.testing.assert_close(self.hot, before, rtol=0, atol=0)
        self.store.restore(self.key, self.destination)

    def test_slot_limit_and_identity_do_not_evict_a_checkpoint(self):
        self.store.max_sessions = 1
        self.capture()
        with self.assertRaises(MemoryError):
            self.capture(SessionKey(1, "request-a", 1))
        with self.assertRaises(ValueError):
            self.capture()
        for wrong in (SessionKey(1, "request-a", 1), SessionKey(0, "request-a", 2)):
            with self.assertRaises(KeyError):
                self.store.restore(wrong, self.destination)
        self.assertTrue(self.store.contains(self.key))

    def test_inflight_boundary_rejected_before_any_copy(self):
        for boundary in (SwapBoundary(120, 1), SwapBoundary(-1, 0)):
            with self.assertRaises(ValueError):
                self.store.capture(self.key, self.source, boundary, self.metadata)
        self.assertEqual(self.store.used_bytes, 0)

    def test_partial_capture_failure_never_publishes_checkpoint(self):
        original = self.store._copy_out
        calls = 0

        def fail(region, ids):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("injected transfer failure")
            return original(region, ids)

        with (
            patch.object(self.store, "_copy_out", side_effect=fail),
            self.assertRaises(RuntimeError),
        ):
            self.capture()
        self.assertFalse(self.store.contains(self.key))
        self.assertEqual(self.store.used_bytes, 0)

    def test_restore_validation_precedes_all_destination_writes(self):
        self.capture()
        invalid = (
            ((6, 7), (0, 9, 10)),
            ((6, 7, 8), (9, 0, 10)),
            ((6, 7, 8), (0, 9, 16)),
            ((6, 7, 8), (0, 9, 6)),
            ((6, 7, 8),),
        )
        for tables in invalid:
            before = self.hot.clone()
            with self.assertRaises(ValueError):
                self.store.restore(self.key, tables)
            torch.testing.assert_close(self.hot, before, rtol=0, atol=0)
            self.assertTrue(self.store.contains(self.key))

    def test_partial_restore_can_retry_from_retained_cold_source(self):
        before = self.hot.clone()
        self.capture()
        original = torch.Tensor.copy_
        calls = 0

        def fail(destination, source, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected restore failure")
            return original(destination, source, **kwargs)

        with (
            patch.object(torch.Tensor, "copy_", new=fail),
            self.assertRaises(RuntimeError),
        ):
            self.store.restore(self.key, self.destination)
        self.assertTrue(self.store.contains(self.key))
        self.store.restore(self.key, self.destination)
        torch.testing.assert_close(self.hot[7], before[2], rtol=0, atol=0)

    def test_corrupted_cold_bytes_are_rejected_before_destination_writes(self):
        self.capture()
        checkpoint = self.store._checkpoints[self.key]
        checkpoint.regions[0].data[0, 0] ^= 1
        before = self.hot.clone()
        with self.assertRaisesRegex(RuntimeError, "integrity check"):
            self.store.restore(self.key, self.destination)
        torch.testing.assert_close(self.hot, before, rtol=0, atol=0)
        self.assertTrue(self.store.contains(self.key))

    def test_silent_capture_copy_corruption_retains_hot_state(self):
        original = self.store._copy_out
        before = self.hot.clone()

        def corrupt(region, ids):
            result = original(region, ids)
            result[0, 0] ^= 1
            return result

        with (
            patch.object(self.store, "_copy_out", side_effect=corrupt),
            self.assertRaisesRegex(RuntimeError, "captured cache verification"),
        ):
            self.capture()
        self.assertFalse(self.store.contains(self.key))
        self.assertEqual(self.store.used_bytes, 0)
        torch.testing.assert_close(self.hot, before, rtol=0, atol=0)

    def test_checkpoint_identity_cannot_be_substituted(self):
        self.capture()
        wrong = SessionKey(1, self.key.request_id, self.key.generation + 1)
        self.store._checkpoints[wrong] = self.store._checkpoints[self.key]
        before = self.hot.clone()
        with self.assertRaisesRegex(RuntimeError, "integrity check"):
            self.store.restore(wrong, self.destination)
        torch.testing.assert_close(self.hot, before, rtol=0, atol=0)

    def test_empty_and_null_only_groups_preserve_metadata_without_page_writes(self):
        before = self.hot.clone()
        tables = ((), (0, 0))
        self.store.capture(self.key, tables, self.boundary, self.metadata)
        self.assertEqual(self.store.used_bytes, len(self.metadata))
        self.assertEqual(
            self.store.restore(self.key, tables), (self.boundary, self.metadata)
        )
        torch.testing.assert_close(self.hot, before, rtol=0, atol=0)

    def test_silent_destination_copy_corruption_cannot_acknowledge_restore(self):
        self.capture()
        original = torch.Tensor.copy_

        def corrupt(destination, source, **kwargs):
            result = original(destination, source, **kwargs)
            if destination.data_ptr() == self.hot[6].data_ptr():
                destination[0] ^= 1
            return result

        with (
            patch.object(torch.Tensor, "copy_", new=corrupt),
            self.assertRaisesRegex(RuntimeError, "restored cache verification"),
        ):
            self.store.restore(self.key, self.destination)
        self.assertTrue(self.store.contains(self.key))
        self.store.restore(self.key, self.destination)

    def test_exact_capacity_boundary_and_cancel_reclaim_space(self):
        size = self.capture()
        self.store.drop(self.key)
        self.store.capacity_bytes = size - 1
        with self.assertRaises(MemoryError):
            self.capture()
        self.store.capacity_bytes = size
        self.capture()
        self.assertEqual(self.store.used_bytes, size)
        self.store.drop(self.key)
        self.assertEqual(self.store.used_bytes, 0)

    def test_null_pages_and_strided_byte_regions(self):
        raw = torch.arange(128, dtype=torch.uint8).view(8, 16)
        view = raw[:, ::2]
        store = SessionColdStore((BlockRegion("strided", 0, view),), 1, 128, 1)
        original = view[1].clone()
        store.capture(self.key, ((0, 1),), self.boundary, b"")
        raw[1].zero_()
        untouched = raw[2, 1::2].clone()
        store.restore(self.key, ((0, 2),))
        torch.testing.assert_close(view[2], original, rtol=0, atol=0)
        torch.testing.assert_close(raw[2, 1::2], untouched, rtol=0, atol=0)

    def test_registration_requires_explicit_group_coverage(self):
        with self.assertRaises(ValueError):
            SessionColdStore(self.regions, 3, 4096, 8)
        store = SessionColdStore(self.regions, 3, 4096, 8, empty_groups=(2,))
        store.capture(self.key, (*self.source, (11,)), self.boundary, b"")
        store.restore(self.key, (*self.destination, (12,)))

    def test_native_adapter_includes_host_backing_and_shared_physical_pages(self):
        config = NS(
            kv_cache_layout="BLNHC",
            num_blocks=16,
            kv_cache_tensors=[NS(size=384, block_stride=24, host_resident=False)],
            kv_cache_groups=[
                NS(layer_names=["gdn"], host_resident=False),
                NS(layer_names=["qsa"], host_resident=False),
                NS(layer_names=[], host_resident=False),
            ],
        )
        context = {
            "gdn": NS(),
            "qsa": NS(_qsa_host_kv=self.ram, _qsa_kv_offload=True),
        }
        regions, empty = MODULE.flash_cache_regions(
            config, {"gdn": self.hot, "qsa": self.hot[:, :2]}, context
        )
        self.assertEqual(empty, (2,))
        self.assertEqual(
            [(r.name, r.group_id) for r in regions],
            [("hot/0", 0), ("hot/1", 1), ("host/qsa", 1)],
        )
        self.assertEqual(regions[0].blocks.data_ptr(), self.hot.data_ptr())
        self.assertEqual(regions[-1].blocks.data_ptr(), self.ram.data_ptr())
        context["qsa"]._qsa_host_kv = None
        with self.assertRaises(ValueError):
            MODULE.flash_cache_regions(
                config, {"gdn": self.hot, "qsa": self.hot}, context
            )


class WorkerMetadataTests(unittest.TestCase):
    def test_streaming_materialization_finds_each_wrapped_layer_spec(self):
        class MambaSpec:
            pass

        class UniformTypeKVCacheSpecs:
            def __init__(self, specs):
                self.kv_cache_specs = specs

        first, second = MambaSpec(), MambaSpec()
        groups = [
            NS(layer_names=[], kv_cache_spec=UniformTypeKVCacheSpecs({})),
            NS(layer_names=["plain"], kv_cache_spec=first),
            NS(
                layer_names=["a", "b"],
                kv_cache_spec=UniformTypeKVCacheSpecs({"a": first, "b": second}),
            ),
            NS(layer_names=["attention"], kv_cache_spec=object()),
        ]
        module = NS(
            MambaSpec=MambaSpec, UniformTypeKVCacheSpecs=UniformTypeKVCacheSpecs
        )
        with patch.dict(sys.modules, {"vllm.v1.kv_cache_interface": module}):
            self.assertEqual(
                WORKER._mamba_layer_specs(groups),
                [(1, "plain", first), (2, "a", first), (2, "b", second)],
            )
            groups[2].kv_cache_spec.kv_cache_specs.pop("b")
            with self.assertRaises(KeyError):
                WORKER._mamba_layer_specs(groups)

    def test_streaming_pp_settle_consumes_only_through_live_target_generation(self):
        def pending(name, slot, generation, needed=True):
            return NS(
                name=name,
                idx_mapping_np=np.array([slot]),
                gen_at_receive_np=np.array([generation]),
                need_sampled_mask=np.array([needed]),
            )

        unrelated = pending("later", 1, 3)
        pp = NS(
            queue=deque(
                [None, pending("earlier", 1, 3), pending("target", 0, 11), unrelated]
            ),
            req_idx_gen_np=np.array([11, 3]),
        )
        consumed = []

        def update():
            entry = pp.queue.popleft()
            consumed.append(None if entry is None else entry.name)
            pp.queue.append(None)

        adapter = WORKER.FlashSessionWorker(
            NS(pp_handler=pp, update_pp_decode_requests=update), None
        )
        adapter._settle_stream_output(0)
        self.assertEqual(consumed, [None, "earlier", "target"])
        self.assertEqual(len(pp.queue), 4)
        self.assertIs(pp.queue[0], unrelated)
        pp.queue.extend([pending("stale", 0, 10), pending("no-output", 0, 11, False)])
        adapter._settle_stream_output(0)
        self.assertEqual(len(consumed), 3)

    def test_streaming_boundary_disagreement_does_not_reset_acceptance(self):
        runner = self.runner()
        adapter = WORKER.FlashSessionWorker(runner, None)
        with self.assertRaisesRegex(ValueError, "token boundaries disagree"):
            adapter.prepare_streaming_update(
                NS(req_id="request-a", num_computed_tokens=5)
            )
        self.assertEqual(int(runner.model_state.num_accepted_tokens_gpu[0]), 3)

    def test_mamba_boundary_materializes_accepted_state_and_overlap_safe_conv(self):
        for dim_first in (False, True):
            for accepted in (1, 2, 3):
                with self.subTest(dim_first=dim_first, accepted=accepted):
                    conv = torch.arange(5 * 6 * 8).reshape(5, 6, 8).float()
                    if dim_first:
                        conv = conv.transpose(1, 2)
                    temporal = torch.arange(5 * 12).reshape(5, 3, 4).float()
                    original_conv, original_temporal = conv.clone(), temporal.clone()
                    regions = [
                        ("conv", conv, [1, 3, 4]),
                        ("temporal", temporal, [1, 3, 4]),
                    ]
                    copied = WORKER.materialize_mamba_boundary(
                        regions, accepted, conv_dim_first=dim_first
                    )
                    offset = accepted - 1
                    torch.testing.assert_close(
                        temporal[1],
                        original_temporal[[1, 3, 4][offset]],
                        rtol=0,
                        atol=0,
                    )
                    if dim_first:
                        actual, expected = (
                            conv[1, :, : 6 - offset],
                            original_conv[1, :, offset:],
                        )
                    else:
                        actual, expected = (
                            conv[1, : 6 - offset],
                            original_conv[1, offset:],
                        )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(
                        conv[3:], original_conv[3:], rtol=0, atol=0
                    )
                    self.assertEqual(copied == 0, accepted == 1)

    def test_mamba_boundary_validates_all_regions_before_modifying_any(self):
        conv = torch.arange(5 * 6 * 8).reshape(5, 6, 8).float()
        before = conv.clone()
        with self.assertRaises(ValueError):
            WORKER.materialize_mamba_boundary(
                [("conv", conv, [1, 3, 4]), ("unknown", conv, [1, 3, 4])],
                2,
                conv_dim_first=False,
            )
        torch.testing.assert_close(conv, before, rtol=0, atol=0)

    def test_byte_exact_codec_and_schema_validation_before_writes(self):
        fields = {
            "bf16": torch.arange(12).to(torch.bfloat16).view(2, 6)[:, ::2],
            "int": torch.tensor([2**50], dtype=torch.int64),
            "empty": torch.empty(1, 0, dtype=torch.float32),
        }
        data = WORKER.pack_fields(fields, 12)
        restored = {name: torch.zeros_like(value) for name, value in fields.items()}
        WORKER.unpack_fields(data, restored)
        for name in fields:
            torch.testing.assert_close(restored[name], fields[name], rtol=0, atol=0)
        (header_size,) = struct.unpack_from("<I", data)
        header = WORKER.field_header(data)
        header["fields"][-1][0] = "unknown"
        raw_header = json.dumps(header).encode()
        corrupt = (
            struct.pack("<I", len(raw_header)) + raw_header + data[4 + header_size :]
        )
        untouched = {name: value.clone() for name, value in restored.items()}
        for malformed in (corrupt, data[:-1], data + b"extra"):
            with self.assertRaises(ValueError):
                WORKER.unpack_fields(malformed, restored)
            for name in restored:
                torch.testing.assert_close(
                    restored[name], untouched[name], rtol=0, atol=0
                )

    @staticmethod
    def runner():
        class Uva:
            def __init__(self, values):
                self.np = np.array(values, dtype=np.int32)
                self.gpu = torch.from_numpy(self.np.copy())

            def copy_to_uva(self):
                self.gpu.copy_(torch.from_numpy(self.np))

        state = NS(
            max_model_len=16,
            req_id_to_index={"request-a": 0},
            free_indices=[1],
            prompt_len=Uva([3, 0]),
            prefill_len=Uva([3, 0]),
            num_computed_prefill_tokens=np.array([3, 0], dtype=np.int32),
            num_computed_tokens_np=np.array([4, 0], dtype=np.int32),
            max_seq_len=np.array([12, 0], dtype=np.int32),
            total_len=NS(gpu=torch.tensor([5, 0], dtype=torch.int32)),
            num_computed_tokens=NS(gpu=torch.tensor([4, 0], dtype=torch.int32)),
            last_sampled_tokens=torch.tensor([[81], [0]], dtype=torch.int64),
            draft_tokens=torch.tensor([[82, 83, 84], [0, 0, 0]], dtype=torch.int64),
            next_prefill_tokens=torch.tensor([[11, 0]], dtype=torch.int32),
            all_token_ids=NS(gpu=torch.arange(32, dtype=torch.int32).reshape(2, 16)),
        )
        model = type("MambaHybridModelState", (), {})()
        model.num_accepted_tokens_gpu = torch.tensor([3, 1], dtype=torch.int32)
        model.recoverssm = None
        model._align_mode = False
        runner = NS(
            req_states=state,
            model_state=model,
            device=torch.device("cpu"),
            pp_handler=None,
            sampler=None,
            adaptive_verification=None,
            pooling_runner=None,
            encoder_cache=None,
            speculator=None,
            block_tables=NS(apply_staged_writes=lambda: None),
        )

        def remove(req_id):
            slot = state.req_id_to_index.pop(req_id, None)
            if slot is None:
                return False
            state.free_indices.append(slot)
            return True

        def add(output):
            for data in output.scheduled_new_reqs:
                slot = state.free_indices.pop()
                state.req_id_to_index[data.req_id] = slot
                for value in WORKER.slot_fields(runner, slot, 5).values():
                    value.zero_()

        runner._remove_request = remove
        runner.add_requests = add
        return runner

    def test_worker_swap_releases_slot_and_restores_acceptance_and_drafts(self):
        for align in (False, True):
            with self.subTest(align=align):
                self._check_worker_relocation(align)

    def _check_worker_relocation(self, align):
        runner = self.runner()
        if align:
            runner.model_state._align_mode = True
            runner.model_state._mamba_state_idx_gpu = torch.tensor([2, 0])
            runner.model_state._mamba_src_col_gpu = torch.tensor([1, -1])
            runner.model_state._mamba_src_off_gpu = torch.tensor([2, 0])
        hot = torch.arange(256, dtype=torch.uint8).reshape(16, 16)
        store = SessionColdStore((BlockRegion("hot", 0, hot),), 1, 10000, 8)
        worker = WORKER.FlashSessionWorker(runner, store)
        key = SessionKey(0, "request-a", 1)
        expected = {k: v.clone() for k, v in WORKER.slot_fields(runner, 0, 5).items()}
        worker.capture(key, ((1, 2, 3, 4),), 4)
        self.assertIn("request-a", runner.req_states.req_id_to_index)
        worker.commit_cold(key)
        self.assertNotIn("request-a", runner.req_states.req_id_to_index)
        # Reuse old slot for another request. Resume must use a different slot.
        runner.req_states.free_indices.remove(0)
        runner.req_states.req_id_to_index["request-b"] = 0
        for value in WORKER.slot_fields(runner, 0, 5).values():
            value.fill_(99)
        hot.zero_()
        request = NS(
            req_id="request-a", block_ids=((5, 6, 7, 8),), num_computed_tokens=4
        )
        worker.restore(key, request)
        self.assertEqual(runner.req_states.req_id_to_index["request-a"], 1)
        for name, actual in WORKER.slot_fields(runner, 1, 5).items():
            torch.testing.assert_close(actual, expected[name], rtol=0, atol=0)
        self.assertTrue(store.contains(key))
        worker.retire(key, "restored")
        self.assertEqual(store.used_bytes, 0)
        self.assertEqual(runner.req_states.req_id_to_index["request-b"], 0)

    def test_worker_boundary_failure_and_failed_restore_retain_ownership(self):
        runner = self.runner()
        store = SessionColdStore(
            (BlockRegion("hot", 0, torch.zeros(8, 16, dtype=torch.uint8)),), 1, 10000, 8
        )
        worker = WORKER.FlashSessionWorker(runner, store)
        key = SessionKey(0, "request-a", 1)
        with self.assertRaises(ValueError):
            worker.capture(key, ((1, 2),), 5)
        self.assertFalse(store.contains(key))
        self.assertEqual(runner.req_states.req_id_to_index["request-a"], 0)
        worker.capture(key, ((1, 2),), 4)
        worker.commit_cold(key)
        worker.commit_cold(key)  # Retry after a lost acknowledgement is idempotent.
        with self.assertRaises(ValueError):
            worker.restore(
                key, NS(req_id="request-a", block_ids=((3,),), num_computed_tokens=4)
            )
        self.assertTrue(store.contains(key))
        self.assertNotIn("request-a", runner.req_states.req_id_to_index)
        self.assertEqual(worker.phases[key], "cold")
        worker.restore(
            key, NS(req_id="request-a", block_ids=((3, 4),), num_computed_tokens=4)
        )

    def test_default_off_does_not_inspect_runner(self):
        with patch.dict("os.environ", {"VLLM_FLASH_SESSION_SWAP_BYTES": "0"}):
            self.assertIsNone(WORKER.init_flash_session_worker(object(), None))

    def test_independent_host_pool_rejects_legacy_session_copy_protocol(self):
        runner = NS(kv_cache_config=NS(direct_host_num_blocks=7))
        with (
            patch.dict("os.environ", {"VLLM_FLASH_SESSION_SWAP_BYTES": "4096"}),
            self.assertRaisesRegex(ValueError, "native prefix caching"),
        ):
            # The incompatible request-state protocol must fail before any
            # region discovery, allocation, or transfer can use GPU IDs for RAM.
            WORKER.init_flash_session_worker(runner, None)

    def test_qwen_subclass_accepts_text_in_a_multimodal_capable_runner(self):
        runner = self.runner()
        base = type(runner.model_state)
        runner.model_state.__class__ = type("Qwen4ExpModelState", (base,), {})
        runner.encoder_cache = NS(mm_features={"request-a": []})
        fields = WORKER.slot_fields(runner, 0, 5)
        self.assertIn("model_state.num_accepted_tokens_gpu", fields)
        runner.encoder_cache.mm_features["request-a"] = [object()]
        with self.assertRaisesRegex(ValueError, "text-only"):
            WORKER.slot_fields(runner, 0, 5)
        runner.encoder_cache.mm_features.clear()
        with self.assertRaisesRegex(ValueError, "text-only"):
            WORKER.slot_fields(runner, 0, 5)

    def test_changed_worker_metadata_cannot_release_hot_slot(self):
        runner = self.runner()
        store = SessionColdStore(
            (BlockRegion("hot", 0, torch.zeros(8, 16, dtype=torch.uint8)),), 1, 10000, 8
        )
        worker = WORKER.FlashSessionWorker(runner, store)
        key = SessionKey(0, "request-a", 1)
        worker.capture(key, ((1, 2),), 4)
        runner.model_state.num_accepted_tokens_gpu[0] += 1
        with self.assertRaisesRegex(RuntimeError, "state changed"):
            worker.commit_cold(key)
        self.assertEqual(runner.req_states.req_id_to_index["request-a"], 0)
        self.assertEqual(worker.phases[key], "captured")
        self.assertTrue(store.contains(key))

    def test_changed_hot_or_cold_cache_cannot_release_hot_slot(self):
        for corrupt_cold in (False, True):
            with self.subTest(corrupt_cold=corrupt_cold):
                runner = self.runner()
                hot = torch.zeros(8, 16, dtype=torch.uint8)
                store = SessionColdStore((BlockRegion("hot", 0, hot),), 1, 10000, 8)
                worker = WORKER.FlashSessionWorker(runner, store)
                key = SessionKey(0, "request-a", 1)
                worker.capture(key, ((1, 2),), 4)
                if corrupt_cold:
                    store._checkpoints[key].regions[0].data[0, 0] ^= 1
                else:
                    hot[1, 0] ^= 1
                with self.assertRaises(RuntimeError):
                    worker.commit_cold(key)
                self.assertEqual(runner.req_states.req_id_to_index["request-a"], 0)
                self.assertEqual(worker.phases[key], "captured")
                self.assertTrue(store.contains(key))


class TransactionTests(unittest.TestCase):
    def setUp(self):
        @dataclass
        class RequestData:
            req_id: str
            block_ids: tuple

            @classmethod
            def from_request(cls, request, block_ids, **kwargs):
                return cls(request.request_id, block_ids)

        class Queue(list):
            def remove_requests(self, requests):
                self[:] = [x for x in self if all(x is not y for y in requests)]

            def add_request(self, request):
                self.append(request)

        self.status = NS(
            RUNNING="running",
            WAITING_FOR_STREAMING_REQ="streaming",
            WAITING_FOR_COLD_SESSION="cold",
            FINISHED_ABORTED="aborted",
        )
        self.request = NS(
            request_id="a",
            client_index=0,
            status="running",
            num_in_flight_tokens=0,
            num_stale_output_tokens=0,
            num_output_placeholders=0,
            num_computed_tokens=32,
            num_prompt_tokens=16,
            mm_features=[],
            prompt_embeds=None,
            structured_output_request=None,
            pooling_params=None,
            lora_request=None,
            sampling_params=NS(prompt_logprobs=None),
            all_token_ids=[1] * 33,
            streaming_queue=deque(),
        )
        self.scheduler = NS(
            requests={"a": self.request},
            running=[self.request],
            waiting=Queue(),
            skipped_waiting=Queue(),
            pause_state="paused",
            processed_step_seq=10,
            sched_step_seq=10,
            num_waiting_for_streaming_input=0,
            max_num_running_reqs=4,
            kv_cache_manager=NS(
                usage=0.0,
                block_pool=NS(num_gpu_blocks=10, get_num_free_blocks=lambda: 7),
            ),
        )
        self.scheduler.set_pause_state = lambda state: setattr(
            self.scheduler, "pause_state", state
        )
        self.scheduler._enqueue_waiting_request = (
            self.scheduler.skipped_waiting.add_request
        )
        self.events = []
        self.fail = set()
        self.bad_reply = set()
        self.hot_owned = True
        self.layout = NS(block_ids=((1, 2),))

        def release(request_id, layout):
            self.events.append("release")
            self.assertTrue(self.hot_owned)
            self.hot_owned = False

        def reserve(request_id, layout):
            self.events.append("reserve")
            self.assertFalse(self.hot_owned)
            self.hot_owned = True
            return NS(block_ids=((3, 4),))

        phases = dict(
            capture="captured",
            commit_cold="cold",
            restore="restored",
            rollback_restore="cold",
            retire="retired",
            abort_capture="absent",
        )

        def rpc(method, args):
            operation, payload = args
            self.events.append(operation)
            if operation in self.fail:
                raise RuntimeError("injected rank/transport failure")
            replies = [
                dict(phase=phases[operation], key=dict(payload["key"]))
                for _ in range(4)
            ]
            if operation in self.bad_reply:
                replies.pop()
            return replies

        self.engine = NS(
            scheduler=self.scheduler,
            batch_queue=[],
            _idle_state_callbacks=[],
            collective_rpc=rpc,
            vllm_config=NS(parallel_config=NS(world_size=4)),
        )
        allocator = NS(
            capture_layout=lambda _: self.layout, release=release, reserve=reserve
        )
        self.tx = TX.FlashSessionTransactions(
            self.engine, allocator, self.status, "paused", RequestData
        )

    def suspend(self):
        return self.tx.suspend("a", 0)["key"]["generation"]

    def test_stats_expose_admission_pressure_without_running_worker_rpcs(self):
        stats = self.tx.stats()["scheduler"]
        self.assertEqual(stats["retained_requests"], 1)
        self.assertEqual(stats["running"], 1)
        self.assertEqual(stats["status_counts"], {"running": 1})
        self.assertEqual(stats["free_gpu_blocks"], 7)
        self.assertEqual(stats["usable_gpu_blocks"], 9)
        self.assertEqual(self.events, [])

    def test_all_rank_commit_precedes_release_and_restore_precedes_scheduling(self):
        generation = self.suspend()
        self.assertEqual(self.events, ["capture", "commit_cold", "release"])
        self.assertEqual(self.scheduler.running, [])
        self.assertEqual(self.request.status, "cold")
        self.assertFalse(self.hot_owned)
        self.tx.assert_can_resume()
        self.tx.restore("a", 0, generation)
        self.assertEqual(self.events[-3:], ["reserve", "restore", "retire"])
        self.assertEqual(self.scheduler.running, [self.request])
        self.assertEqual(self.request.status, "running")
        self.assertFalse(self.tx.records)

    def test_partial_capture_rolls_back_only_after_all_rank_abort_ack(self):
        self.bad_reply.add("capture")
        with self.assertRaises(RuntimeError):
            self.suspend()
        self.assertEqual(self.events, ["capture", "abort_capture"])
        self.assertTrue(self.hot_owned)
        self.assertEqual(self.scheduler.running, [self.request])
        self.assertFalse(self.tx.records)

    def test_ambiguous_commit_holds_blocks_pause_and_identity_until_retry(self):
        self.fail.add("commit_cold")
        with self.assertRaises(RuntimeError):
            self.suspend()
        self.assertTrue(self.hot_owned)
        self.assertEqual(self.scheduler.running, [])
        with self.assertRaises(RuntimeError):
            self.tx.assert_can_resume()
        with self.assertRaises(ValueError):
            self.tx.complete_suspend("a", 1, 1)
        self.fail.clear()
        self.tx.complete_suspend("a", 0, 1)
        self.assertEqual(self.events.count("capture"), 1)
        self.assertEqual(self.events.count("release"), 1)
        self.tx.assert_can_resume()

    def test_restore_failure_releases_only_after_all_rank_rollback(self):
        generation = self.suspend()
        self.fail.add("restore")
        with self.assertRaises(RuntimeError):
            self.tx.restore("a", 0, generation)
        self.assertEqual(
            self.events[-4:], ["reserve", "restore", "rollback_restore", "release"]
        )
        self.assertFalse(self.hot_owned)
        self.assertEqual(self.tx.records["a"].phase, "cold")
        self.assertEqual(self.scheduler.running, [])
        self.fail.clear()
        self.tx.restore("a", 0, generation)

    def test_failed_rollback_quarantines_destination_and_keeps_engine_paused(self):
        generation = self.suspend()
        self.fail.update(("restore", "rollback_restore"))
        with self.assertRaises(RuntimeError):
            self.tx.restore("a", 0, generation)
        self.assertTrue(self.hot_owned)
        self.assertEqual(self.events.count("release"), 1)
        with self.assertRaises(RuntimeError):
            self.tx.assert_can_resume()

    def test_lost_retirement_ack_does_not_schedule_until_retry(self):
        generation = self.suspend()
        self.fail.add("retire")
        with self.assertRaises(RuntimeError):
            self.tx.restore("a", 0, generation)
        self.assertEqual(self.scheduler.running, [])
        self.assertTrue(self.hot_owned)
        with self.assertRaises(RuntimeError):
            self.tx.assert_can_resume()
        self.fail.clear()
        self.tx.complete_restore("a", 0, generation)
        self.assertEqual(self.events.count("restore"), 1)
        self.assertEqual(self.scheduler.running, [self.request])

    def test_cancel_failure_cannot_be_restored_and_pauses_engine(self):
        generation = self.suspend()
        self.scheduler.pause_state = "unpaused"
        self.fail.add("retire")
        with self.assertRaises(RuntimeError):
            self.tx.cancel("a")
        self.assertEqual(self.scheduler.pause_state, "paused")
        with self.assertRaises(ValueError):
            self.tx.restore("a", 0, generation)
        self.fail.clear()
        self.tx.cancel("a")
        self.assertFalse(self.tx.records)

    def test_idle_streaming_session_frees_and_recovers_hot_slot_accounting(self):
        self.request.status = "streaming"
        self.scheduler.running.clear()
        self.scheduler.skipped_waiting.add_request(self.request)
        self.scheduler.num_waiting_for_streaming_input = 1
        generation = self.suspend()
        self.assertEqual(self.scheduler.num_waiting_for_streaming_input, 0)
        self.assertEqual(self.scheduler.skipped_waiting, [])
        self.tx.restore("a", 0, generation)
        self.assertEqual(self.scheduler.num_waiting_for_streaming_input, 1)
        self.assertEqual(self.scheduler.skipped_waiting, [self.request])

    def test_inflight_pause_or_hot_capacity_failure_does_not_transfer(self):
        self.engine.batch_queue.append(object())
        with self.assertRaises(ValueError):
            self.suspend()
        self.assertEqual(self.events, [])
        self.engine.batch_queue.clear()
        generation = self.suspend()
        self.scheduler.max_num_running_reqs = 0
        with self.assertRaises(MemoryError):
            self.tx.restore("a", 0, generation)
        self.assertEqual(self.events, ["capture", "commit_cold", "release"])
        self.assertEqual(self.tx.records["a"].phase, "cold")

    def make_idle(self):
        self.request.status = "streaming"
        self.scheduler.running.clear()
        self.scheduler.skipped_waiting.add_request(self.request)
        self.scheduler.num_waiting_for_streaming_input = 1

    def test_ttl_is_sixty_minutes_and_expired_restore_is_an_explicit_miss(self):
        self.make_idle()
        clock = [100.0]
        self.tx.clock = lambda: clock[0]

        def finish(request_id, status):
            request = self.scheduler.requests.pop(request_id)
            request.status = status
            return [request]

        self.scheduler.finish_requests = finish
        generation = self.suspend()
        clock[0] += 3599
        self.assertEqual(self.tx.lookup("a", 0, generation)["cache"], "cold_hit")
        clock[0] += 1
        result = self.tx.restore("a", 0, generation)
        self.assertEqual(result["reason"], "expired")
        self.assertTrue(result["requires_prefill"])
        self.assertFalse(self.tx.records)
        self.assertNotIn("reserve", self.events)
        self.assertEqual(self.tx.lookup("a", 0, generation)["reason"], "expired")
        self.assertEqual(self.tx.stats()["counters"]["retired_expired"], 1)

    def test_active_generation_snapshot_is_not_subject_to_idle_expiry(self):
        clock = [0.0]
        self.tx.clock = lambda: clock[0]
        generation = self.suspend()
        clock[0] = 7200.0
        self.assertEqual(self.tx.expire_idle()["sessions"], [])
        self.assertEqual(self.tx.lookup("a", 0, generation)["cache"], "cold_hit")

    def test_disabled_ttl_and_lookup_polling_do_not_change_retention_order(self):
        self.make_idle()
        clock = [0.0]
        self.tx.clock = lambda: clock[0]
        self.tx.observe_hot_sessions()
        before = dict(self.tx.hot_last_used)
        self.tx.lookup("a", 0)
        clock[0] = 50.0
        self.tx.lookup("a", 0)
        self.assertEqual(self.tx.hot_last_used, before)
        self.tx.ttl_seconds = 0
        generation = self.suspend()
        clock[0] = 100000.0
        self.assertEqual(self.tx.lookup("a", 0, generation)["cache"], "cold_hit")

    def test_lru_spills_oldest_idle_and_never_active_or_pending_input(self):
        self.make_idle()
        newer = NS(**vars(self.request))
        newer.request_id = "b"
        newer.streaming_queue = deque()
        active = NS(**vars(self.request))
        active.request_id = "c"
        active.status = "running"
        pending = NS(**vars(self.request))
        pending.request_id = "d"
        pending.streaming_queue = deque([object()])
        self.scheduler.requests.update(b=newer, c=active, d=pending)
        self.scheduler.running.append(active)
        self.scheduler.skipped_waiting.extend((newer, pending))
        self.scheduler.num_waiting_for_streaming_input = 3
        self.tx.hot_last_used.update(
            {("a", 0): 20, ("b", 0): 30, ("c", 0): 0, ("d", 0): 0}
        )
        result = self.tx.evict_lru()
        self.assertEqual(result["key"]["request_id"], "a")
        self.assertIn(active, self.scheduler.running)
        self.assertEqual(self.scheduler.num_waiting_for_streaming_input, 2)
        self.assertEqual(self.tx.stats()["counters"]["lru_spills"], 1)

    def test_stale_generation_and_foreign_identity_are_misses_not_hits(self):
        generation = self.suspend()
        self.assertEqual(self.tx.lookup("a", 1, generation)["cache"], "miss")
        self.tx.restore("a", 0, generation)
        self.assertEqual(self.tx.lookup("a", 0, generation)["cache"], "hot_hit")
        self.assertEqual(self.tx.lookup("a", 0, generation + 1)["cache"], "miss")
        # Retry after a lost restore acknowledgement does not copy or allocate again.
        self.assertEqual(self.tx.restore("a", 0, generation)["cache"], "hot_hit")
        self.assertEqual(self.events.count("restore"), 1)

    def test_automatic_lru_waits_for_pipeline_drain_then_releases_slot(self):
        self.make_idle()
        self.scheduler.max_num_running_reqs = 1
        self.scheduler.waiting.add_request(object())
        self.scheduler.pause_state = "unpaused"
        self.engine.batch_queue.append(object())
        self.tx.automatic_tick()
        self.assertEqual(self.scheduler.pause_state, "paused")
        self.assertTrue(self.tx.automatic_pause)
        self.assertEqual(self.events, [])
        self.engine.batch_queue.clear()
        self.tx.automatic_tick()
        self.assertEqual(self.events, ["capture", "commit_cold", "release"])
        self.assertEqual(self.scheduler.pause_state, "unpaused")
        self.assertEqual(self.scheduler.num_waiting_for_streaming_input, 0)

    def test_automatic_failure_keeps_engine_paused_and_does_not_retry(self):
        self.make_idle()
        self.scheduler.max_num_running_reqs = 1
        self.scheduler.waiting.add_request(object())
        self.scheduler.pause_state = "unpaused"
        self.fail.add("commit_cold")
        self.tx.automatic_tick()
        self.assertEqual(self.scheduler.pause_state, "paused")
        self.assertIsNotNone(self.tx.policy_error)
        self.assertTrue(self.hot_owned)
        before = list(self.events)
        self.tx.automatic_tick()
        self.assertEqual(self.events, before)
        with self.assertRaises(RuntimeError):
            self.tx.assert_can_resume()

    def test_automatic_policy_respects_manual_pause(self):
        self.make_idle()
        self.scheduler.max_num_running_reqs = 1
        self.scheduler.waiting.add_request(object())
        self.tx.automatic_tick()
        self.assertEqual(self.events, [])
        self.assertEqual(self.scheduler.pause_state, "paused")

    def test_automatic_ttl_retires_expired_cold_state(self):
        self.make_idle()
        clock = [0.0]
        self.tx.clock = lambda: clock[0]
        self.suspend()
        self.scheduler.finish_requests = lambda request_id, status: [
            self.scheduler.requests.pop(request_id)
        ]
        clock[0] = 3600
        self.scheduler.pause_state = "unpaused"
        self.tx.automatic_tick()
        self.assertFalse(self.tx.records)
        self.assertEqual(self.scheduler.pause_state, "unpaused")
        self.assertEqual(self.events[-1], "retire")

    def test_automatic_restore_processes_queued_streaming_input_after_state_restore(
        self,
    ):
        self.make_idle()
        self.suspend()
        update = object()
        self.request.streaming_queue.append(update)

        def append_input(request, actual_update):
            self.assertIs(actual_update, update)
            self.assertEqual(self.events[-1], "retire")
            self.scheduler.num_waiting_for_streaming_input -= 1
            request.status = "waiting"

        self.scheduler._update_request_as_session = append_input
        self.scheduler.pause_state = "unpaused"
        self.tx.automatic_tick()
        self.assertEqual(self.events[-3:], ["reserve", "restore", "retire"])
        self.assertEqual(self.request.status, "waiting")
        self.assertFalse(self.request.streaming_queue)
        self.assertFalse(self.tx.records)
        self.assertIsNone(self.tx.policy_error)


if __name__ == "__main__":
    unittest.main()
