# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native CPU scheduler lifecycle gate for the session-swap coordinator.

Use actual requests, scheduler queues, hybrid block managers and output updates.
Only model configuration and collective worker replies are stand-ins; these
tests cannot establish numerical, CUDA-stream or distributed correctness.
"""

import unittest
from collections import deque
from copy import deepcopy
from types import SimpleNamespace as NS

import torch

try:
    from vllm.config import CacheConfig, ObservabilityConfig, SchedulerConfig
    from vllm.config.ec_manager_config import EncoderCacheManagerConfig
    from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.flash_session_allocator import FlashSessionAllocator
    from vllm.v1.core.flash_session_transactions import FlashSessionTransactions
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.interface import PauseState
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.core import EngineCore, EngineCoreProc
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request, RequestStatus
except ImportError as exc:
    raise unittest.SkipTest("patched native engine dependencies required") from exc

from test_session_allocator_native import (
    NativeHostPoolAdmissionTests,
    NativeSessionAllocatorTests,
)


class NativeSessionSchedulerTests(unittest.TestCase):
    def test_four_half_block_prefills_fit_without_skipping_checkpoints(self):
        scheduler = NS(
            cache_config=NS(block_size=944), hash_block_size=16,
            max_num_scheduled_tokens=2048,
            scheduler_config=NS(long_prefill_token_threshold=472),
            use_eagle_block_drop=True, mamba_has_prefill_checkpoint_blocks=False,
            mamba_partial_cache_hit=False, mamba_fine_grained_prefix_cache=False,
        )
        request = NS(num_computed_tokens=0, num_prompt_tokens=29756,
                     num_tokens=29756, shared_prefix_boundary=0)
        budget = 2048
        for _ in range(4):
            scheduled = Scheduler._mamba_block_aligned_split(
                scheduler, request, min(472, budget))
            self.assertEqual(scheduled, 472)
            budget -= scheduled
        for start in (472, 28320 - 472, 29264 - 472, 30208 - 472):
            request.num_computed_tokens = start
            request.num_prompt_tokens = request.num_tokens = 32588
            scheduled = Scheduler._mamba_block_aligned_split(scheduler, request, 472)
            self.assertEqual(scheduled, 472)
            self.assertEqual((start + scheduled) % 944, 0)

    def test_native_fence_keeps_direct_host_pages_pinned_until_last_step(self):
        manager, host = NativeHostPoolAdmissionTests.manager()
        request = NativeHostPoolAdmissionTests.request()
        self.assertIsNotNone(manager.allocate_slots(request, 32))
        request.last_sched_seq = 3
        scheduler = object.__new__(Scheduler)
        scheduler.kv_cache_manager = manager
        scheduler.defer_block_free = True
        scheduler.sched_step_seq = 3
        scheduler.processed_step_seq = 1
        scheduler.deferred_frees = deque()
        scheduler._free_request_blocks(request)
        self.assertEqual(len(scheduler.deferred_frees), 1)
        for sequence in (1, 2):
            scheduler.processed_step_seq = sequence
            scheduler._drain_deferred_frees()
            self.assertEqual(manager.block_pool.get_num_free_blocks(), 0)
            self.assertEqual(host.get_num_free_blocks(), 0)
        scheduler.processed_step_seq = 3
        scheduler._drain_deferred_frees()
        self.assertFalse(scheduler.deferred_frees)
        self.assertEqual(manager.block_pool.get_num_free_blocks(), 2)
        self.assertEqual(host.get_num_free_blocks(), 4)

    def test_engine_mutations_cannot_invalidate_retained_session_state(self):
        engine = object.__new__(EngineCore)
        engine.flash_session_transactions = NS(records={"cold": object()})
        # No scheduler/executor exists: a forbidden call must fail before it
        # can reset caches, discard GPU state or send a weight-changing RPC.
        calls = (
            lambda: engine.reset_prefix_cache(True, True),
            engine.reset_mm_cache,
            engine.reset_encoder_cache,
            lambda: engine.sleep(level=1, mode="keep"),
            lambda: engine.pause_scheduler(mode="keep", clear_cache=True),
            lambda: EngineCoreProc.pause_scheduler(engine, "keep", True),
            lambda: engine.set_weight_version("new"),
            lambda: engine.collective_rpc("reload_weights"),
            lambda: engine.collective_rpc(lambda worker: None),
        )
        for call in calls:
            with self.assertRaisesRegex(RuntimeError, "retained sessions"):
                call()
        engine.model_executor = NS(collective_rpc=lambda *args: ["allowed"])
        self.assertEqual(engine.collective_rpc("flash_session_swap"), ["allowed"])
        self.assertEqual(engine.collective_rpc("synchronize_device"), ["allowed"])
        engine.flash_session_transactions.records.clear()
        self.assertEqual(engine.collective_rpc("reload_weights"), ["allowed"])
        engine.set_weight_version("new")
        self.assertEqual(engine.get_weight_version(), "new")

    def test_native_ple_context_rebuilds_from_relocated_token_history(self):
        model = object.__new__(Qwen4ExpModelState)
        model.ngram_eos_token_id = 0
        model.ngram_context = torch.zeros(2, 3, dtype=torch.int32)
        model.ngram_context_offsets = torch.arange(-3, 0)
        state = NS(
            num_computed_tokens=NS(gpu=torch.tensor([4, 3])),
            all_token_ids=NS(
                gpu=torch.tensor([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]])
            ),
        )
        batch = NS(num_reqs=2, idx_mapping=torch.tensor([1, 0]))
        reference = model._prepare_ngram_context(batch, state).clone()
        torch.testing.assert_close(
            reference, torch.tensor([[20, 21, 22], [11, 12, 13]], dtype=torch.int32)
        )
        state.all_token_ids.gpu = state.all_token_ids.gpu.flip(0)
        state.num_computed_tokens.gpu = state.num_computed_tokens.gpu.flip(0)
        model.ngram_context.fill_(999)
        batch.idx_mapping = torch.tensor([0, 1])
        restored = model._prepare_ngram_context(batch, state)
        torch.testing.assert_close(restored, reference, rtol=0, atol=0)

    @staticmethod
    def setup_engine(asynchronous=False, pp=1, depth=2, mode="align"):
        manager = NativeSessionAllocatorTests.manager(depth, mode)
        config = NS(
            scheduler_config=SchedulerConfig(
                max_num_seqs=2,
                max_num_batched_tokens=128,
                max_model_len=240000,
                enable_chunked_prefill=True,
                async_scheduling=asynchronous,
                is_encoder_decoder=False,
                watermark=0.0,
            ),
            cache_config=CacheConfig(
                block_size=16,
                enable_prefix_caching=False,
                mamba_cache_mode=mode,
            ),
            model_config=NS(
                uses_mrope=False,
                is_encoder_decoder=False,
                max_model_len=240000,
                is_diffusion=False,
                enable_return_routed_experts=False,
                return_sampling_mask=False,
            ),
            parallel_config=NS(
                world_size=2 * pp,
                tensor_parallel_size=2,
                pipeline_parallel_size=pp,
                data_parallel_size=1,
                data_parallel_index=0,
                decode_context_parallel_size=1,
                prefill_context_parallel_size=1,
            ),
            observability_config=ObservabilityConfig(),
            ec_manager_config=EncoderCacheManagerConfig(),
            is_mm_encoder_only=False,
            lora_config=None,
            kv_events_config=None,
            kv_transfer_config=None,
            ec_transfer_config=None,
            speculative_config=None,
            num_speculative_tokens=depth,
            num_lookahead_tokens=depth,
            max_in_flight_tokens=4096,
            max_concurrent_batches=pp + 1 if asynchronous else pp,
            use_v2_model_runner=True,
        )
        config.cache_config.num_gpu_blocks = 80
        scheduler_cls = AsyncScheduler if asynchronous else Scheduler
        scheduler = scheduler_cls(
            vllm_config=config,
            kv_cache_config=manager.kv_cache_config,
            structured_output_manager=NS(should_advance=lambda *args, **kwargs: False),
            block_size=16 if mode == "align" else 240000,
            hash_block_size=16,
            mm_registry=NS(supports_multimodal_inputs=lambda _: False),
        )
        calls = []

        def rpc(method, args):
            operation, payload = args
            calls.append((operation, payload))
            phase = {
                "capture": "captured",
                "commit_cold": "cold",
                "restore": "restored",
                "retire": "retired",
                "rollback_restore": "cold",
                "abort_capture": "absent",
            }[operation]
            return [dict(phase=phase, key=dict(payload["key"])) for _ in range(2 * pp)]

        engine = NS(
            scheduler=scheduler,
            vllm_config=config,
            batch_queue=[],
            _idle_state_callbacks=[],
            collective_rpc=rpc,
        )
        transactions = FlashSessionTransactions(
            engine,
            FlashSessionAllocator(scheduler.kv_cache_manager),
            RequestStatus,
            PauseState.PAUSED_ALL,
            NewRequestData,
            unpaused=PauseState.UNPAUSED,
        )
        scheduler.flash_session_transactions = transactions
        return engine, transactions, calls

    @staticmethod
    def request(name, prompt=32, max_tokens=1, resumable=True):
        return Request(
            name,
            [1] * prompt,
            SamplingParams(max_tokens=max_tokens, temperature=0, seed=173),
            None,
            resumable=resumable,
        )

    @staticmethod
    def step(scheduler):
        output = scheduler.schedule()
        dispatched = deepcopy(output)
        ids = list(output.num_scheduled_tokens)
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=ids,
                req_id_to_index={name: i for i, name in enumerate(ids)},
                sampled_token_ids=[[7] for _ in ids],
            ),
        )
        return dispatched

    def test_streaming_cold_reuse_resume_and_cancel_with_native_bookkeeping(self):
        for asynchronous, pp in ((False, 1), (True, 1), (True, 2)):
            with self.subTest(asynchronous=asynchronous, pp=pp):
                engine, tx, calls = self.setup_engine(asynchronous, pp)
                scheduler = engine.scheduler
                request = self.request("a")
                scheduler.add_request(request)
                self.step(scheduler)
                self.assertEqual(
                    request.status, RequestStatus.WAITING_FOR_STREAMING_REQ
                )
                self.assertEqual(request.num_in_flight_tokens, 0)
                scheduler.set_pause_state(PauseState.PAUSED_ALL)
                self.assertTrue(tx.describe("a", 0)["owns_hot_blocks"])
                key = tx.suspend("a", 0)["key"]
                self.assertFalse(tx.describe("a", 0)["owns_hot_blocks"])
                self.assertEqual(scheduler.num_waiting_for_streaming_input, 0)
                self.assertEqual(
                    scheduler.kv_cache_manager.block_pool.get_num_free_blocks(), 79
                )
                scheduler.set_pause_state(PauseState.UNPAUSED)
                self.assertEqual(scheduler.get_num_unfinished_requests(), 0)
                other = self.request("b", max_tokens=30, resumable=False)
                scheduler.add_request(other)
                self.step(scheduler)
                occupied = {
                    b
                    for group in scheduler.kv_cache_manager.get_block_ids("b")
                    for b in group
                    if b
                }
                scheduler.add_request(self.request("a", prompt=8, max_tokens=4))
                self.assertEqual(len(request.streaming_queue), 1)
                self.assertEqual(request.num_prompt_tokens, 32)
                tx.automatic_tick()
                self.assertIsNone(tx.policy_error, tx.stats())
                self.assertEqual(request.status, RequestStatus.WAITING)
                self.assertEqual(request.num_prompt_tokens, 40)
                self.assertEqual(request.num_computed_tokens, 32)
                restored = {
                    b
                    for group in scheduler.kv_cache_manager.get_block_ids("a")
                    for b in group
                    if b
                }
                self.assertFalse(occupied & restored)
                scheduled = self.step(scheduler)
                new_data = next(
                    x for x in scheduled.scheduled_new_reqs if x.req_id == "a"
                )
                self.assertEqual(new_data.num_computed_tokens, 32)
                self.assertEqual(len(new_data.prefill_token_ids), 40)
                self.assertEqual(
                    tx.lookup("a", 0, key["generation"])["cache"], "hot_hit"
                )
                scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
                self.assertEqual(scheduler.num_waiting_for_streaming_input, 0)
                self.assertEqual(
                    scheduler.kv_cache_manager.block_pool.get_num_free_blocks(), 79
                )
                self.assertFalse(tx.records)
                self.assertEqual(
                    [op for op, _ in calls],
                    ["capture", "commit_cold", "restore", "retire"],
                )

    def test_native_expiry_reclaims_cold_request_without_double_free(self):
        engine, tx, _ = self.setup_engine(True, 2)
        scheduler = engine.scheduler
        now = [0.0]
        tx.clock = lambda: now[0]
        scheduler.add_request(self.request("ttl"))
        self.step(scheduler)
        scheduler.set_pause_state(PauseState.PAUSED_ALL)
        key = tx.suspend("ttl", 0)["key"]
        now[0] = 3600
        scheduler.set_pause_state(PauseState.UNPAUSED)
        tx.automatic_tick()
        self.assertIsNone(tx.policy_error, tx.stats())
        self.assertNotIn("ttl", scheduler.requests)
        self.assertEqual(scheduler.num_waiting_for_streaming_input, 0)
        self.assertEqual(
            scheduler.kv_cache_manager.block_pool.get_num_free_blocks(), 79
        )
        self.assertEqual(tx.lookup("ttl", 0, key["generation"])["reason"], "expired")

    def test_closed_streams_release_admission_then_idle_lru_exceeds_hot_slots(self):
        engine, tx, _ = self.setup_engine(True, 2, mode="none")
        scheduler = engine.scheduler
        for index in range(4):
            name = f"reference-{index}"
            scheduler.add_request(self.request(name))
            self.step(scheduler)
            scheduler.add_request(self.request(name, prompt=8))
            self.step(scheduler)
            scheduler.add_request(self.request(name, resumable=False))
            self.step(scheduler)
            stats = tx.stats()["scheduler"]
            self.assertEqual(stats["retained_requests"], 0)
            self.assertEqual(stats["idle_hot"], 0)
            self.assertEqual(stats["free_gpu_blocks"], stats["usable_gpu_blocks"])

        for index in range(3):
            scheduler.add_request(self.request(f"idle-{index}"))
            tx.automatic_tick()
            self.assertIsNone(tx.policy_error, tx.stats())
            self.step(scheduler)
        self.assertEqual(tx.stats()["scheduler"]["retained_requests"], 3)
        self.assertEqual(tx.stats()["scheduler"]["idle_hot"], 2)
        self.assertEqual(tx.lookup("idle-0", 0)["cache"], "cold_hit")
        scheduler.add_request(self.request("idle-0", prompt=8))
        tx.automatic_tick()
        self.assertIsNone(tx.policy_error, tx.stats())
        self.step(scheduler)
        self.assertEqual(tx.lookup("idle-0", 0)["cache"], "hot_hit")
        for index in range(3):
            scheduler.add_request(self.request(f"idle-{index}", resumable=False))
        self.step(scheduler)
        stats = tx.stats()
        self.assertEqual(stats["cold_sessions"], 0)
        self.assertEqual(stats["scheduler"]["retained_requests"], 0)
        self.assertEqual(stats["scheduler"]["free_gpu_blocks"], 79)

    def test_resumable_decode_cannot_overlap_an_unsettled_turn_boundary(self):
        engine, _, _ = self.setup_engine(True, 2, mode="none")
        scheduler = engine.scheduler
        request = self.request("boundary", max_tokens=5)
        scheduler.add_request(request)
        self.step(scheduler)
        pending = scheduler.schedule()
        if not pending.total_num_scheduled_tokens:
            scheduler.update_from_output(
                pending, ModelRunnerOutput(req_ids=[], req_id_to_index={})
            )
            pending = scheduler.schedule()
        self.assertEqual(pending.num_scheduled_tokens, {"boundary": 3})
        # The real V2 PP2 engine permits three queued batches, not two. A
        # same-request step must not follow before the first result settles.
        for _ in range(3):
            output = scheduler.schedule()
            self.assertNotIn("boundary", output.num_scheduled_tokens)
            scheduler.update_from_output(
                output, ModelRunnerOutput(req_ids=[], req_id_to_index={})
            )
        scheduler.update_from_output(
            pending,
            ModelRunnerOutput(
                req_ids=["boundary"],
                req_id_to_index={"boundary": 0},
                sampled_token_ids=[[7, 8, 9]],
            ),
        )
        self.assertEqual(request.num_output_tokens, 4)
        self.assertEqual(request.num_in_flight_tokens, 0)

    def test_resumable_mtp_tail_cannot_compute_beyond_returned_turn(self):
        engine, _, _ = self.setup_engine(True, 2, mode="none")
        scheduler = engine.scheduler
        request = self.request("tail", max_tokens=2)
        scheduler.add_request(request)
        self.step(scheduler)
        for _ in range(3):
            output = scheduler.schedule()
            if output.total_num_scheduled_tokens:
                break
            scheduler.update_from_output(
                output, ModelRunnerOutput(req_ids=[], req_id_to_index={})
            )
        self.assertEqual(output.num_scheduled_tokens, {"tail": 1})
        self.assertFalse(output.scheduled_spec_decode_tokens)
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=["tail"], req_id_to_index={"tail": 0}, sampled_token_ids=[[7]]
            ),
        )
        self.assertEqual(request.status, RequestStatus.WAITING_FOR_STREAMING_REQ)
        self.assertEqual(request.num_computed_tokens, request.num_tokens - 1)
        self.assertEqual(request.num_output_placeholders, 0)
        self.assertEqual(request.num_in_flight_tokens, 0)


if __name__ == "__main__":
    assert not torch.cuda.is_initialized(), "CPU-only scheduler gate"
    unittest.main()
