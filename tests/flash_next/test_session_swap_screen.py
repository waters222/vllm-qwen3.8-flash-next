"""CPU gates for the model-screen comparator, not a substitute for GPU tests."""

import unittest
from types import SimpleNamespace as NS

from launch_session_swap_screen import (
    EXPERIMENTAL_KERNEL_FLAGS,
    KERNEL_FLAGS,
    kernel_environment,
    prefill_staging_environment,
    tp4_environment,
)
from prefix_cache_screen import (
    admitted_batch,
    compare_common_history_logprobs,
    compare_prefix_shapes,
    compare_request_execution,
    divergent_prefix_prompts,
    finite_logprobs,
    prefix_engine_config,
    prefix_verdict,
    validate_balanced_observation,
)
from qualify_session_swap import block_sets, compare_tokens, screen_verdict
from session_prefill_trace import (
    PrefillTraceWorkerExtension,
    canonical_expert_layout,
    canonical_qsa_selection,
    checked_host_registration,
    compare_prefix_pages,
    describe_request_batch,
    hash_page_chunks,
    index_prefix_page_requests,
    manager_block_ids,
    prefill_convolution_window,
    snapshot_prefix_pages,
    stable_qsa_topk,
)
from session_stream_screen import StreamingSession, cleanup_complete


class SessionScreenTests(unittest.TestCase):
    def test_lookup_observer_preserves_native_results_and_records_loads(self):
        import json
        import tempfile
        from pathlib import Path
        from prefix_load_barrier import CacheLookupObserverMixin

        request = NS(request_id="sustained-renewed-0")
        calls = []
        result = (944, True)

        def lookup(req, local, *args, **kwargs):
            calls.append((req, local, args, kwargs))
            return result

        marker = object()
        connector = NS(get_num_new_matched_tokens=lookup,
                       update_state_after_alloc=lambda *args: marker,
                       _req_status={request.request_id: NS(num_locally_computed_tokens=944)},
                       _jobs={7: NS(req_id=request.request_id, is_store=False,
                                   pending_count=4, keys={"a", "b"})})

        class Native:
            def __init__(self, **kwargs):
                self.connector = NS(connector_scheduler=connector)

        class Observed(CacheLookupObserverMixin, Native):
            pass

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lookups.jsonl"
            observer = Observed(vllm_config=NS(additional_config={
                "flash_next_diagnostic_lookup_observer": str(path)}))
            try:
                self.assertIs(connector.get_num_new_matched_tokens(
                    request, 944, max_num_new_tokens=1888), result)
                self.assertIs(connector.update_state_after_alloc(request, None, 944), marker)
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(len(calls), 1)
                self.assertEqual(rows[0]["external_tokens"], 944)
                self.assertEqual(rows[1]["loads"], [dict(job_id=7, pending_workers=4, chunks=2)])
            finally:
                observer._lookup_observer_stream.close()

    def test_sustained_mode_rejects_competing_fixtures_and_nonprivate_seeds(self):
        from session_sustained_decode import validate, workload

        args = dict(sustained_decode=True, native_prefix=True, tp4=True,
                    serving_decode=True, balanced_prefix_prefill=True,
                    untraced_balanced_prefix=True, private_cold_buffers=True,
                    continuation_context=32768)
        validate(NS(**args))
        for key in ("active_cancel", "continuations", "mlp_numerical_control",
                    "paired_cold_loads", "no_mtp"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(NS(**args, **{key: True}))
        with self.assertRaises(ValueError):
            workload([[1] * 32768 for _ in range(4)])

    def test_sustained_decode_requires_full_outputs_and_real_cold_loads(self):
        import asyncio
        import hashlib
        import json

        from session_sustained_decode import run, workload

        seeds = [[1] * 944 + [i + 2] * 32000 for i in range(4)]
        for budget in (1024, 4096):
            prompts, outputs = workload(seeds, budget)
            self.assertEqual(max(outputs), budget)
            self.assertTrue(all(len(p) + n + 4 <= 32768
                                for p, n in zip(prompts, outputs)))
            prompts, outputs = workload(seeds, budget, uniform_outputs=True)
            self.assertEqual(outputs, [budget] * 4)
            self.assertTrue(all(len(p) + n + 4 <= 32768
                                for p, n in zip(prompts, outputs)))

        async def exercise(fault=None, idle=False, ttl=3600, uniform=False):
            pressure_calls = []

            async def generate(name, **kwargs):
                pressure_calls.append(name)

            async def batch(name, prompts, output_tokens, **kwargs):
                fresh = name in ("sustained-seed", "sustained-expired") or kwargs["skip_cache"]
                rows = []
                for i, (p, n) in enumerate(zip(prompts, output_tokens)):
                    if fault == "truncated" and name == "sustained-cold-1" and i == 3:
                        n -= 1
                    rows.append(dict(
                        tokens=[2] * n, logprobs=[{"2": -0.1}] * n,
                        prompt_tokens=len(p), prompt_sha256=hashlib.sha256(
                            json.dumps(p).encode()).hexdigest(),
                        cached_tokens=0 if fresh else 944,
                    ))
                return dict(name=name, requests=rows, max_running=4,
                            load_bytes=int("cold" in name and fault != "no-load"))

            from unittest.mock import patch

            clock = [100.0]
            progress = []

            async def sleep(seconds):
                self.assertLessEqual(seconds, 30)
                clock[0] += seconds

            with patch("session_sustained_decode.time", NS(monotonic=lambda: clock[0])), \
                    patch("session_sustained_decode.asyncio.sleep", sleep):
                result = await run(generate, batch, seeds, lambda _: [1] * 4096,
                                   pressure_rounds=2, real_idle_expiry=idle,
                                   idle_progress=progress.append, idle_ttl=ttl,
                                   uniform_outputs=uniform)
            if idle:
                self.assertEqual(progress[-1]["elapsed_seconds"], ttl + 5)
                self.assertEqual(len(result["phases"]), 9)
            self.assertEqual(len(pressure_calls), 4)
            return result

        self.assertTrue(asyncio.run(exercise())["lifecycle_passed"])
        self.assertTrue(asyncio.run(exercise(idle=True))["lifecycle_passed"])
        self.assertTrue(asyncio.run(exercise(idle=True, ttl=300))["lifecycle_passed"])
        uniform = asyncio.run(exercise(uniform=True))
        self.assertTrue(uniform["lifecycle_passed"])
        self.assertEqual(uniform["output_tokens"], [1024] * 4)
        for fault in ("truncated", "no-load"):
            with self.subTest(fault=fault):
                self.assertFalse(asyncio.run(exercise(fault))["lifecycle_passed"])

    def test_untraced_config_installs_memory_rpc_before_worker_creation(self):
        import importlib

        config = prefix_engine_config({}, tp4=True, continuation_context=32768,
                                      balanced_prefill=True, continuation_concurrency=4,
                                      prefill_budget=2048, serving_decode=True,
                                      gpu_cache_mib=896)
        module, name = config["worker_extension_cls"].rsplit(".", 1)
        extension = getattr(importlib.import_module(module), name)
        self.assertTrue(callable(extension.snapshot_cuda_memory))
        with self.assertRaises(ValueError):
            prefix_engine_config({"worker_extension_cls": "existing.Extension"})

    def test_gpu_cache_budget_preserves_host_pool_offload_and_c4_geometry(self):
        from prefix_cache_screen import validate_gpu_cache_budget

        kwargs = dict(tp4=True, continuation_context=32768,
                      continuation_concurrency=4, balanced_prefill=True,
                      prefill_budget=2048, serving_decode=True)
        reference = prefix_engine_config({}, **kwargs)
        for mib in (512, 768, 896, 1024):
            actual = prefix_engine_config({}, gpu_cache_mib=mib, **kwargs)
            self.assertEqual(actual.pop("kv_cache_memory_bytes"), mib * 1024**2)
            self.assertEqual(actual, {k: v for k, v in reference.items()
                                      if k != "kv_cache_memory_bytes"})
        args = dict(native_prefix=True, tp4=True, continuation_context=32768,
                    prefix_gpu_cache_mib=512)
        validate_gpu_cache_budget(NS(**args))
        validate_gpu_cache_budget(NS(**{**args, "continuation_context": 8192,
            "prefill_staging_ab": True, "prefill_staging_concurrency": 4}))
        for key, value in (("native_prefix", False), ("tp4", False),
                           ("continuation_context", 8192), ("prefix_gpu_cache_mib", 0)):
            with self.assertRaises(ValueError):
                validate_gpu_cache_budget(NS(**{**args, key: value}))
        for mib in (0, 256, 2048):
            with self.assertRaises(ValueError):
                prefix_engine_config({}, gpu_cache_mib=mib, **kwargs)

    def test_memory_snapshot_preserves_peaks_and_reports_missing_counters(self):
        from unittest.mock import Mock, patch

        stats = {"allocated_bytes.all.peak": 1234, "num_alloc_retries": 7,
                 "num_ooms": 0}
        cuda = Mock()
        cuda.current_device.return_value = 2
        cuda.mem_get_info.return_value = (100, 2000)
        cuda.memory_stats.return_value = stats
        worker = PrefillTraceWorkerExtension()
        worker.rank = 2
        with patch.dict("sys.modules", {"torch": NS(cuda=cuda)}):
            result = worker.snapshot_cuda_memory()
        self.assertEqual(result["rank"], 2)
        self.assertEqual(result["free_bytes"], 100)
        self.assertEqual(result["allocator"]["allocated_bytes.all.peak"], 1234)
        self.assertEqual(result["allocator"]["num_alloc_retries"], 7)
        self.assertIsNone(result["allocator"]["reserved_bytes.all.peak"])
        self.assertEqual([call[0] for call in cuda.mock_calls],
                         ["current_device", "mem_get_info", "memory_stats"])

    def test_server_local_transport_does_not_require_self_ssh(self):
        from unittest.mock import patch

        import launch_session_swap_screen as launcher

        for local in (False, True):
            with (
                patch.object(launcher, "LOCAL_HOST", local),
                patch.object(
                    launcher.subprocess, "check_output", return_value="ok\n"
                ) as call,
            ):
                self.assertEqual(launcher.remote("printf", "has space"), "ok")
                command = call.call_args.args[0]
                if local:
                    self.assertEqual(command, ["printf", "has space"])
                else:
                    self.assertEqual(
                        command,
                        [
                            "ssh",
                            "-o",
                            "BatchMode=yes",
                            launcher.HOST,
                            "printf 'has space'",
                        ],
                    )

    def test_failure_fixture_rejects_overlapping_modes(self):
        from session_transfer_audit import validate_native_load_failure

        valid = dict(
            native_prefix=True,
            tp4=True,
            serving_decode=True,
            private_cold_buffers=True,
            balanced_prefix_prefill=True,
            untraced_balanced_prefix=True,
            prefix_concurrency=2,
            prefix_prefill_budget=2048,
        )
        validate_native_load_failure(NS())
        for mode in ("inject", "recovery"):
            args = dict(valid, native_load_failure=mode)
            validate_native_load_failure(NS(**args))
            for key, value in (
                ("tp4", False),
                ("private_cold_buffers", False),
                ("prefix_concurrency", 4),
                ("prefix_prefill_budget", 4096),
                ("performance_repeats", 3),
                ("audit_native_copies", True),
                ("paired_cold_loads", True),
                ("divergent_prefixes", True),
                ("no_mtp", True),
            ):
                with self.subTest(mode=mode, key=key), self.assertRaises(ValueError):
                    validate_native_load_failure(NS(**(args | {key: value})))

    def test_failure_screen_requires_dead_engine_no_output_and_fresh_recovery(self):
        """Mock the frontend contract; actual worker death needs the GPU fixture."""
        import asyncio
        import json
        import tempfile
        from pathlib import Path

        from session_transfer_audit import native_load_failure_screen

        class DeadError(RuntimeError):
            pass

        async def trial(mode="inject", defect=None):
            evidence, calls = {}, []
            state = dict(dead=False, paused=False)

            async def health():
                if state["dead"] and defect != "healthy":
                    raise DeadError()

            async def pause(**kwargs):
                self.assertEqual(kwargs, dict(mode="wait", clear_cache=False))
                state["paused"] = True

            async def resume():
                self.assertTrue(state["paused"])
                state["paused"] = False

            async def rpc(method):
                self.assertTrue(state["paused"])
                self.assertEqual(method, "arm_native_load_failure")
                return [
                    dict(rank=i, armed=i == 3)
                    for i in range(3 if defect == "ack" else 4)
                ]

            engine = NS(
                check_health=health,
                pause_generation=pause,
                resume_generation=resume,
                collective_rpc=rpc,
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "failure.json"

                async def generate(name, **kwargs):
                    self.assertFalse(state["paused"])
                    calls.append(name)
                    if name == "failure-cold":
                        self.assertEqual(len(calls), 35)
                        record = dict(
                            rank=3,
                            job_id=42,
                            request_id="failure-cold-uuid",
                            backend_load_completed=True,
                            transfer_bytes=4096,
                            completion_already_acknowledged=False,
                        )
                        if defect == "unrelated":
                            record["request_id"] = "other"
                        if defect != "missing":
                            path.write_text(json.dumps(record))
                        if defect == "output":
                            await kwargs["observe"](
                                name, NS(outputs=[NS(token_ids=[123])])
                            )
                        state["dead"] = True
                        raise DeadError()
                    if name == "failure-retry":
                        if defect != "retry":
                            raise DeadError()
                        return {}
                    return dict(
                        cached_tokens=(
                            5664 if name == "failure-hot" and defect != "hot" else 0
                        ),
                        prompt_sha256="same-prompt",
                        transfer_delta={"vllm:kv_offload_load_bytes": 0},
                    )

                await native_load_failure_screen(
                    engine, generate, mode, path, DeadError, evidence, lambda: None
                )
            self.assertTrue(evidence["passed"])
            self.assertEqual(evidence["failed_request_emitted_tokens"], 0)
            self.assertEqual(evidence["retry_emitted_tokens"], 0)
            if mode == "recovery":
                self.assertEqual(calls, ["failure-prime", "failure-hot"])
                self.assertNotIn("armed_workers", evidence)
            else:
                self.assertTrue(evidence["dead_health_rejected"])
                self.assertTrue(evidence["dead_retry_rejected"])
                self.assertEqual(len(calls), 36)

        asyncio.run(trial())
        asyncio.run(trial("recovery"))
        for defect, error in (
            ("ack", ValueError),
            ("unrelated", ValueError),
            ("missing", FileNotFoundError),
            ("hot", ValueError),
            ("healthy", AssertionError),
            ("retry", AssertionError),
            ("output", AssertionError),
        ):
            with self.subTest(defect=defect), self.assertRaises(error):
                asyncio.run(trial(defect=defect))

    def test_native_failure_injector_requires_gate_and_completed_named_load(self):
        import json
        import os
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from session_transfer_audit import install_native_load_failure

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failure.json"
            result = NS(job_id=42, success=True, transfer_size=4096)
            backend = NS(get_finished=lambda: [result])
            connector = NS(
                worker=backend,
                _load_jobs={},
                _connector_worker_meta=NS(completed_jobs={}),
            )
            with (
                patch.dict(os.environ, FLASH_NATIVE_LOAD_FAILURE="0"),
                self.assertRaises(ValueError),
            ):
                install_native_load_failure(connector, 3, path)
            with patch.dict(os.environ, FLASH_NATIVE_LOAD_FAILURE="1"):
                native = backend.get_finished
                for rank in (0, 1, 2):
                    self.assertEqual(
                        install_native_load_failure(connector, rank, path),
                        dict(rank=rank, armed=False),
                    )
                    self.assertIs(backend.get_finished, native)
                self.assertEqual(
                    install_native_load_failure(connector, 3, path),
                    dict(rank=3, armed=True),
                )
                self.assertEqual(
                    backend.get_finished(), [result]
                )  # Store is untouched.
                self.assertFalse(path.exists())
                connector._load_jobs[42] = "unrelated-request"
                with self.assertRaises(ValueError):
                    backend.get_finished()
                self.assertFalse(path.exists())
                connector._load_jobs[42] = "failure-cold-deadbeef"
                for success, size, ack in (
                    (False, 4096, False),
                    (True, 0, False),
                    (True, 4096, True),
                ):
                    result.success, result.transfer_size = success, size
                    connector._connector_worker_meta.completed_jobs = (
                        {42: 1} if ack else {}
                    )
                    with self.assertRaises(ValueError):
                        backend.get_finished()
                    self.assertFalse(path.exists())
                result.success, result.transfer_size = True, 4096
                connector._connector_worker_meta.completed_jobs = {}
                with self.assertRaisesRegex(RuntimeError, "Injected native"):
                    backend.get_finished()
                record = json.loads(path.read_text())
                self.assertEqual(
                    (record["rank"], record["job_id"], record["transfer_bytes"]),
                    (3, 42, 4096),
                )
                self.assertTrue(record["backend_load_completed"])
                self.assertFalse(record["completion_already_acknowledged"])
                self.assertEqual(connector._load_jobs, {42: "failure-cold-deadbeef"})
                self.assertEqual(connector._connector_worker_meta.completed_jobs, {})
                before = path.read_bytes()
                with self.assertRaisesRegex(RuntimeError, "remains fatal"):
                    backend.get_finished()
                self.assertEqual(path.read_bytes(), before)
                with self.assertRaises(ValueError):
                    install_native_load_failure(connector, 3, path)

    def test_staging_ab_drains_alternates_and_restores_even_on_bad_evidence(self):
        """CPU protocol test, not a CUDA timing or integrity qualification."""
        import asyncio

        from prefix_cache_benchmark import prefill_staging_ab_screen

        async def trial(*, bad_hot=False, bad_ack=False, c4=False, bad_count=False):
            prompt_length, cached_length = (29756, 28320) if c4 else (7100, 5664)
            state = dict(paused=False, enabled=None, requests=0)
            changes = []

            async def pause(**kwargs):
                self.assertEqual(kwargs, dict(mode="wait", clear_cache=False))
                state["paused"] = True

            async def resume():
                self.assertTrue(state["paused"])
                state["paused"] = False

            async def rpc(method, *, args):
                self.assertTrue(state["paused"])
                self.assertEqual(method, "set_qsa_prefill_staging")
                (enabled,) = args
                state["enabled"] = enabled
                changes.append(enabled)
                return [
                    dict(rank=rank, restored=True, modules=13)
                    if enabled is None
                    else dict(rank=rank, enabled=enabled, target=12, draft=1)
                    for rank in range(3 if bad_ack else 4)
                ]

            async def batch(name, prompts, **kwargs):
                self.assertFalse(state["paused"])
                self.assertEqual(kwargs["tokens"], 1)
                self.assertEqual(kwargs["cache_salt"], f"staging-ab-c{len(prompts)}")
                state["requests"] += len(prompts)
                return dict(
                    requests=[
                        dict(
                            prompt_tokens=prompt_length,
                            cached_tokens=0 if "-prime-" in name else cached_length,
                            requested_output_tokens=1,
                            tokens=[1],
                            finish_reason="length",
                            nonfinite_logprobs=[],
                            skip_cache=False,
                            prompt_sha256=str(i),
                        )
                        for i in range(len(prompts))
                    ],
                    max_running=len(prompts) - int(bad_count),
                    load_bytes=int(bad_hot and "-r0-" in name),
                )

            engine = NS(
                pause_generation=pause, resume_generation=resume, collective_rpc=rpc
            )
            try:
                result = await prefill_staging_ab_screen(
                    engine, batch,
                    [[i] * prompt_length for i in range(4 if c4 else 2)]
                )
            finally:
                self.assertIsNone(changes[-1])
                self.assertEqual(state["paused"], bad_ack)
                if bad_ack:
                    self.assertEqual(state["requests"], 0)
            self.assertTrue(result["restored"])
            self.assertEqual(state["requests"], 44 if c4 else 33)
            self.assertEqual(result["expected_requests"], state["requests"])
            self.assertEqual(len(result["phases"]), 11 if c4 else 22)
            self.assertEqual(
                changes,
                [False, False, True, False, True, True, False, False, True, True, False]
                * (1 if c4 else 2)
                + [None],
            )

        asyncio.run(trial())
        asyncio.run(trial(c4=True))
        for kwargs in (dict(bad_hot=True), dict(bad_ack=True)):
            with self.assertRaises(ValueError):
                asyncio.run(trial(**kwargs))
        for kwargs in (dict(bad_hot=True), dict(bad_count=True)):
            with self.assertRaises(ValueError):
                asyncio.run(trial(c4=True, **kwargs))

    def test_staging_worker_rejects_active_requests_and_restores_original_flags(self):
        import os
        from unittest.mock import patch

        class Qwen4ExpQSAAttention:
            _qsa_kv_offload = True
            _qsa_stage_all_prefill = False

        target = [Qwen4ExpQSAAttention() for _ in range(12)]
        draft = [Qwen4ExpQSAAttention()]
        draft[0]._qsa_stage_all_prefill = True
        worker = PrefillTraceWorkerExtension()
        worker.rank = 2
        worker.get_model = lambda: NS(modules=lambda: iter(target))
        worker.model_runner = NS(
            req_states=NS(req_id_to_index={}),
            get_draft_model=lambda: NS(modules=lambda: iter(draft)),
        )
        with (
            patch.dict(os.environ, FLASH_PREFILL_STAGING_AB="0"),
            self.assertRaises(ValueError),
        ):
            worker.set_qsa_prefill_staging(True)
        with (
            patch.dict(os.environ, FLASH_PREFILL_STAGING_AB="1"),
            patch("torch.cuda.synchronize") as synchronize,
        ):
            worker.model_runner.req_states.req_id_to_index["active"] = 0
            with self.assertRaises(ValueError):
                worker.set_qsa_prefill_staging(True)
            synchronize.assert_not_called()
            worker.model_runner.req_states.req_id_to_index.clear()
            for enabled in (True, False):
                result = worker.set_qsa_prefill_staging(enabled)
                self.assertEqual(
                    result, dict(rank=2, enabled=enabled, target=12, draft=1)
                )
                self.assertTrue(
                    all(m._qsa_stage_all_prefill is enabled for m in target + draft)
                )
            self.assertEqual(
                worker.set_qsa_prefill_staging(None),
                dict(rank=2, restored=True, modules=13),
            )
            self.assertFalse(any(m._qsa_stage_all_prefill for m in target))
            self.assertTrue(draft[0]._qsa_stage_all_prefill)
            target.pop()
            with self.assertRaises(ValueError):
                worker.set_qsa_prefill_staging(True)
            self.assertFalse(hasattr(worker, "_qsa_staging_ab"))

    def test_native_prefill_profile_is_bounded_and_stops_on_bad_hot_evidence(self):
        import asyncio
        import tempfile
        from pathlib import Path

        from prefix_cache_benchmark import (
            prefill_profile_screen,
            prefill_profiler_config,
        )

        async def trial(directory, *, bad_hot=False, missing_rank=False):
            calls = []

            async def batch(name, prompts, **kwargs):
                calls.append(name)
                self.assertEqual(kwargs["tokens"], 1)
                self.assertEqual(kwargs["cache_salt"], name.rsplit("-", 1)[0])
                return dict(
                    requests=[
                        dict(
                            cached_tokens=0 if name.endswith("prime") else 5664,
                            prompt_tokens=7100,
                            requested_output_tokens=1,
                            tokens=[1],
                            finish_reason="length",
                            nonfinite_logprobs=[],
                            skip_cache=False,
                            prompt_sha256=str(i),
                        )
                        for i in range(len(prompts))
                    ],
                    max_running=len(prompts),
                    load_bytes=int(bad_hot and name.endswith("capture")),
                )

            async def start_profile(**kwargs):
                calls.append("start")
                self.assertEqual(kwargs["profile_prefix"], "prefix-prefill")

            async def stop_profile():
                calls.append("stop")
                # Metadata-only fixture: real CUDA content is a separate GPU gate.
                for rank in range(3 if missing_rank else 4):
                    (
                        directory / f"prefix-prefill_rank{rank}.123.pt.trace.json.gz"
                    ).write_bytes(b"trace")

            engine = NS(start_profile=start_profile, stop_profile=stop_profile)
            try:
                result = await prefill_profile_screen(
                    engine, batch, [[1] * 7100, [2] * 7100], directory
                )
            finally:
                self.assertEqual(calls[-1], "stop")
            self.assertEqual(
                calls,
                [
                    "profile-c1-prime",
                    "profile-c1-warm",
                    "profile-c2-prime",
                    "profile-c2-warm",
                    "start",
                    "profile-c1-capture",
                    "profile-c2-capture",
                    "stop",
                ],
            )
            self.assertEqual(len(result["phases"]), 6)
            self.assertEqual([a["rank"] for a in result["artifacts"]], list(range(4)))
            self.assertTrue(result["native_artifacts_recorded"])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = prefill_profiler_config(root / "ok")
            self.assertEqual(config["max_iterations"], 32)
            self.assertTrue(config["ignore_frontend"])
            self.assertFalse(config["torch_profiler_record_shapes"])
            asyncio.run(trial(root / "ok"))
            with self.assertRaisesRegex(ValueError, "hot path"):
                asyncio.run(trial(root / "bad-hot", bad_hot=True))
            with self.assertRaisesRegex(ValueError, "four rank traces"):
                asyncio.run(trial(root / "missing", missing_rank=True))

    def test_native_iteration_capture_is_bounded_and_keeps_batch_geometry(self):
        from dataclasses import make_dataclass

        from prefix_cache_benchmark import NativeIterationCapture

        Detail = make_dataclass(
            "Detail",
            [
                ("iteration_index", int),
                ("num_ctx_requests", int),
                ("num_ctx_tokens", int),
                ("num_generation_requests", int),
                ("num_generation_tokens", int),
                ("elapsed_ms", float),
                ("num_encoder_inputs", int, 0),
                ("num_encoder_output_tokens", int, 0),
                ("is_dummy", bool, False),
            ],
        )
        capture = NativeIterationCapture()
        capture.record(Detail(0, 2, 1888, 0, 0, 12.0))
        self.assertEqual(capture.records, [])
        capture.start("hot")
        with self.assertRaises(ValueError):
            capture.start("cold")
        capture.record(None)
        capture.record(Detail(10, 2, 1888, 0, 0, 12.0))
        capture.record(Detail(11, 1, 492, 1, 5, 9.0))
        capture.record(Detail(12, 0, 0, 2, 10, 1.0))
        result = capture.finish()
        self.assertTrue(result["complete"])
        self.assertEqual(
            [r["num_ctx_tokens"] for r in result["records"]], [1888, 492, 0]
        )
        self.assertIsNone(capture.phase)
        for detail in (
            Detail(0, 2, -1, 0, 0, 0.0),
            Detail(0, 2, 1888, 0, 0, float("nan")),
            Detail(0, 2, 1888, 0, 0, None),
        ):
            capture.start("invalid")
            capture.record(detail)
            self.assertFalse(capture.finish()["complete"])
        capture.start("duplicate")
        capture.record(Detail(1, 0, 0, 2, 10, 1.0))
        capture.record(Detail(1, 0, 0, 2, 10, 1.0))
        self.assertFalse(capture.finish()["complete"])
        capture.start("overflow")
        for index in range(1025):
            capture.record(Detail(index, 0, 0, 2, 10, 1.0))
        result = capture.finish()
        self.assertFalse(result["complete"])
        self.assertEqual(len(result["records"]), 1024)
        capture.start("empty")
        self.assertFalse(capture.finish()["complete"])

    def test_warmed_performance_uses_real_tiers_and_excludes_warmup(self):
        import asyncio
        import hashlib

        from prefix_cache_benchmark import performance_screen, phase_measurements

        calls, pressure = [], []
        missing_load = False
        with_iterations = missing_iterations = False

        async def generate(name, **kwargs):
            pressure.append((name, kwargs))

        async def batch(name, prompts, **kwargs):
            calls.append((name, kwargs))
            mode = name.rsplit("-", 1)[-1]
            warmup = "-r0-" in name
            rows = []
            for index, ids in enumerate(prompts):
                token = 1 if mode == "hot" else 2
                rows.append(
                    dict(
                        prompt_sha256=hashlib.sha256(bytes(ids)).hexdigest(),
                        repetition_penalty=1.0,
                        cached_tokens=0 if mode in ("prime", "fresh") else 5664,
                        skip_cache=kwargs.get("skip_cache", False),
                        tokens=[token] * 128,
                        logprobs=[{str(token): -0.1}] * 128,
                        nonfinite_logprobs=[],
                        started_monotonic=100.0 + index,
                        stream_timing=dict(
                            first_output_seconds=1.0,
                            last_output_seconds=30.0 if warmup else 3.0,
                            first_output_tokens=1,
                        ),
                        speculative_decoding=dict(
                            num_spec_tokens=4,
                            num_spec_steps=43,
                            num_draft_tokens=172,
                            num_accepted_draft_tokens=86,
                        ),
                    )
                )
            phase = dict(
                requests=rows,
                started_monotonic=100.0,
                ended_monotonic=140.0,
                max_running=len(prompts),
                load_bytes=100 if mode == "cold" and not missing_load else 0,
            )
            if with_iterations:
                phase["native_iterations"] = dict(complete=not missing_iterations)
            return phase

        async def run():
            return await performance_screen(
                generate,
                batch,
                [[1] * 8, [2] * 8],
                lambda i: [i],
                128,
                repeats=3,
                pressure_rounds=2,
            )

        result = asyncio.run(run())
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["cycles"]), 8)
        self.assertEqual(sum(row["warmup"] for row in result["cycles"]), 2)
        self.assertEqual(len(pressure), 16)
        self.assertEqual(len({row[0] for row in pressure}), 16)
        for summary in result["summaries"]:
            self.assertEqual(summary["measured_cycles"], 3)
            self.assertEqual(summary["ttft_seconds"]["median"], 1.0)
            self.assertEqual(summary["draft_acceptance_rate"], 0.5)
            self.assertEqual(summary["mean_acceptance_length"], 3.0)
            expected = 63.5 if summary["concurrency"] == 1 else 254 / 3
            self.assertAlmostEqual(
                summary["median_aggregate_decode_tokens_per_second"], expected
            )
        names = [name for name, _ in calls]
        self.assertLess(
            names.index("perf-c1-r1-fresh"), names.index("perf-c1-r1-prime")
        )
        self.assertGreater(
            names.index("perf-c1-r2-fresh"), names.index("perf-c1-r2-cold")
        )
        phase = result["cycles"][-1]["phases"]["cold"]
        phase["requests"][0]["speculative_decoding"]["num_draft_tokens"] = 0
        with self.assertRaises(ValueError):
            phase_measurements(phase)
        missing_load = True
        self.assertFalse(asyncio.run(run())["passed"])
        missing_load = False
        with_iterations = True
        self.assertTrue(asyncio.run(run())["passed"])
        missing_iterations = True
        self.assertFalse(asyncio.run(run())["passed"])

    def test_performance_mode_rejects_tracing_and_other_lifecycle_fixtures(self):
        from prefix_cache_benchmark import validate_performance

        args = dict(
            performance_repeats=3,
            native_prefix=True,
            tp4=True,
            serving_decode=True,
            untraced_balanced_prefix=True,
            balanced_prefix_prefill=True,
            divergent_prefixes=True,
            prefix_concurrency=2,
        )
        validate_performance(NS(**args))
        validate_performance(NS(**args, performance_iteration_details=True))
        with self.assertRaises(ValueError):
            validate_performance(NS(performance_iteration_details=True))
        validate_performance(NS())
        profile_args = {
            **args,
            "performance_repeats": 0,
            "profile_prefix_prefill": True,
            "private_cold_buffers": True,
        }
        validate_performance(NS(**profile_args))
        ab_args = {
            **profile_args,
            "profile_prefix_prefill": False,
            "prefill_staging_ab": True,
        }
        validate_performance(NS(**ab_args))
        for key in (
            "profile_prefix_prefill",
            "performance_iteration_details",
            "paired_cold_loads",
            "stable_qsa_selection",
            "cuda_launch_blocking",
            "trace_prefix_shapes",
        ):
            with self.assertRaises(ValueError):
                validate_performance(NS(**{**ab_args, key: True}))
        for key, value in (
            ("performance_repeats", 3),
            ("private_cold_buffers", False),
            ("performance_iteration_details", True),
            ("trace_prefix_shapes", True),
        ):
            with self.assertRaises(ValueError):
                validate_performance(NS(**{**profile_args, key: value}))
        for key, value in (
            ("performance_repeats", 1),
            ("tp4", False),
            ("serving_decode", False),
            ("prefix_concurrency", 1),
            ("trace_prefix_shapes", True),
            ("audit_native_copies", True),
            ("active_cancel", True),
            ("idle_expiry", True),
            ("continuations", True),
            ("no_mtp", True),
        ):
            with self.assertRaises(ValueError):
                validate_performance(NS(**{**args, key: value}))

    def test_graph_decode_retains_cache_contract_and_bounds_c2_capture_sizes(self):
        source = {
            "compilation_config": {"cudagraph_mode": "NONE"},
            "kv_cache_memory_bytes": 750000000,
        }
        eager = prefix_engine_config(source, tp4=True)
        graph = prefix_engine_config(source, tp4=True, serving_decode=True)
        self.assertTrue(eager["enforce_eager"])
        self.assertFalse(graph["enforce_eager"])
        self.assertTrue(graph["cudagraph_metrics"])
        self.assertNotIn("cudagraph_metrics", eager)
        self.assertEqual(
            graph["compilation_config"]["cudagraph_mode"], "FULL_DECODE_ONLY"
        )
        self.assertEqual(
            graph["compilation_config"]["cudagraph_capture_sizes"],
            [1, 2, 3, 4, 5, 6, 8, 10],
        )
        for key in (
            "kv_transfer_config",
            "kv_cache_memory_bytes",
            "max_num_seqs",
            "max_model_len",
            "speculative_config",
            "enable_prefix_caching",
            "mamba_cache_mode",
        ):
            self.assertEqual(eager[key], graph[key])
        no_mtp = prefix_engine_config(
            source, tp4=True, no_mtp=True, serving_decode=True
        )
        self.assertIsNone(no_mtp["speculative_config"])
        self.assertEqual(
            no_mtp["compilation_config"]["cudagraph_capture_sizes"], [1, 2]
        )
        host = "flash_next_direct_host_kv"
        self.assertEqual(
            graph["additional_config"][host],
            {**eager["additional_config"][host], "allow_cudagraph": True},
        )
        self.assertEqual(source["compilation_config"], {"cudagraph_mode": "NONE"})
        self.assertNotIn("additional_config", source)
        with self.assertRaises(ValueError):
            prefix_engine_config(source, serving_decode=True)

    def test_graph_decode_rejects_automatic_or_invalid_gpu_cache_budgets(self):
        for budget in (None, 0, -1, True, 750000000.0):
            with (
                self.subTest(budget=budget),
                self.assertRaisesRegex(ValueError, "explicit GPU KV byte budget"),
            ):
                prefix_engine_config(
                    {"kv_cache_memory_bytes": budget}, tp4=True, serving_decode=True
                )

    def test_graph_decode_rejects_ordering_controls_or_incomplete_observation(self):
        from prefix_cache_screen import validate_serving_decode

        args = dict(
            serving_decode=True,
            native_prefix=True,
            tp4=True,
            untraced_balanced_prefix=True,
        )
        validate_serving_decode(NS(**args))
        validate_serving_decode(NS())
        for key, value in (
            ("native_prefix", False),
            ("tp4", False),
            ("untraced_balanced_prefix", False),
            ("audit_active_cancel", True),
            ("stable_expert_layout", True),
            ("stable_qsa_selection", True),
            ("native_kernels", True),
            ("optimized_canonical", True),
        ):
            with self.assertRaises(ValueError):
                validate_serving_decode(NS(**{**args, key: value}))

    def test_graph_decode_allows_complete_prefill_page_audit_without_eager_decode(self):
        from prefix_cache_screen import validate_serving_decode

        args = dict(
            serving_decode=True,
            native_prefix=True,
            tp4=True,
            balanced_prefix_prefill=True,
            prefix_concurrency=2,
            trace_prefix_shapes=True,
            trace_prefix_pages=True,
        )
        validate_balanced_observation(NS(**args))
        validate_serving_decode(NS(**args))
        for key in ("trace_prefix_shapes", "trace_prefix_pages"):
            with self.subTest(missing=key), self.assertRaises(ValueError):
                validate_serving_decode(NS(**{**args, key: False}))
        with self.assertRaises(ValueError):
            validate_balanced_observation(NS(**args, untraced_balanced_prefix=True))

    def test_c4_abort_ownership_requires_and_checks_every_survivor(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from prefix_load_barrier import PairedLoadMixin

        names = tuple(f"cancel-survivor-{i}" for i in range(3))
        for mode in ("valid", "missing", "changed_hash", "released"):
            with self.subTest(mode=mode), TemporaryDirectory() as directory:
                shared = NS(ref_cnt=4, is_null=False, block_hash="shared")
                private = {
                    name: NS(ref_cnt=1, is_null=False, block_hash=name)
                    for name in names
                }
                calls = []

                class Native:
                    def finish_requests(self, ids, status):
                        calls.append("abort")
                        del self.requests[ids[0]]
                        self.running = list(self.requests.values())
                        shared.ref_cnt -= 1
                        if mode == "changed_hash":
                            private[names[-1]].block_hash = "corrupt"
                        if mode == "released":
                            private[names[-1]].ref_cnt = 0
                        return "native-result"

                class Scheduler(PairedLoadMixin, Native):
                    pass

                scheduler = object.__new__(Scheduler)
                path = Path(directory) / "abort.json"
                scheduler._diagnostic_active_abort_path = path
                scheduler._diagnostic_active_survivors = names
                scheduler._diagnostic_active_expiry = True
                scheduler.requests = {
                    name: NS(request_id=name, status=NS(name="RUNNING"))
                    for name in ("cancel-victim", *names)
                }
                if mode == "missing":
                    del scheduler.requests[names[-1]]
                scheduler.running = list(scheduler.requests.values())
                scheduler.kv_cache_manager = NS(
                    get_blocks=lambda name: NS(blocks=[[shared, private[name]]])
                )
                with patch("prefix_load_barrier.expire_active_cache") as expiry:
                    expiry.side_effect = lambda *args: calls.append("expire") or {}
                    if mode == "valid":
                        self.assertEqual(
                            scheduler.finish_requests("cancel-victim", NS()),
                            "native-result",
                        )
                    else:
                        with self.assertRaises(RuntimeError):
                            scheduler.finish_requests("cancel-victim", NS())
                    if mode == "missing":
                        expiry.assert_not_called()
                        self.assertEqual(calls, [])
                        self.assertFalse(path.exists())
                        continue
                    expiry.assert_called_once_with(scheduler, ["cancel-victim", *names])
                evidence = json.loads(path.read_text())
                self.assertEqual(calls, ["expire", "abort"])
                self.assertEqual(set(evidence["survivors"]), set(names))
                self.assertEqual(evidence["passed"], mode == "valid")
                self.assertEqual(
                    evidence["survivors"][names[-1]]["passed"], mode == "valid"
                )

    def test_active_abort_observer_does_not_intercept_engine_shutdown(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from prefix_load_barrier import PairedLoadMixin

        class Native:
            def finish_requests(self, ids, status):
                removed = [self.requests.pop(name) for name in ids]
                self.running = [row for row in self.running if row not in removed]
                return removed

        class Scheduler(PairedLoadMixin, Native):
            pass

        with TemporaryDirectory() as directory:
            scheduler = object.__new__(Scheduler)
            path = Path(directory) / "abort.json"
            scheduler._diagnostic_active_abort_path = path
            scheduler.requests = {
                name: NS(request_id=name, status=NS(name="RUNNING"))
                for name in ("cancel-victim", "cancel-survivor")
            }
            scheduler.running = list(scheduler.requests.values())
            removed = scheduler.finish_requests(None, NS(name="FINISHED_ABORTED"))
            self.assertEqual(len(removed), 2)
            self.assertEqual(scheduler.running, [])
            self.assertEqual(scheduler.requests, {})
            self.assertFalse(path.exists())

    def test_stream_timing_excludes_initial_speculative_burst_and_single_output(self):
        from prefix_cache_screen import stream_timing

        source = {"per_request_spec_decode_metrics": "none"}
        self.assertEqual(
            prefix_engine_config(source)["per_request_spec_decode_metrics"], "summary"
        )
        self.assertEqual(source["per_request_spec_decode_metrics"], "none")
        row = stream_timing(2.0, 3.0, 5, 128)
        self.assertEqual(row["post_first_output_tokens"], 123)
        self.assertEqual(row["post_first_output_tokens_per_second"], 123.0)
        self.assertIsNone(
            stream_timing(2.0, 2.0, 128, 128)["post_first_output_tokens_per_second"]
        )
        for args in (
            (1, float("inf"), 1, 128),
            (float("nan"), 2, 1, 128),
            (None, 2, 1, 128),
            (3, 2, 1, 128),
            (1, 2, 0, 128),
            (1, 2, 129, 128),
        ):
            with self.assertRaises(ValueError):
                stream_timing(*args)

    def test_active_cache_audit_rejects_rank_page_and_category_gaps(self):
        from copy import deepcopy

        from prefix_cache_screen import compare_active_cache_audit

        groups = []
        for group_id in range(41):
            count = {37: 13, 38: 13, 39: 12, 40: 1}.get(group_id, 1)
            suffix = ".linear_attn" if group_id < 36 else ".compressed_key_cache"
            group = dict(group_id=group_id, host_resident=group_id >= 39, layers={})
            for i in range(count):
                name = ("mtp" if group_id == 40 else f"target.{i}") + suffix
                group["layers"][name] = [
                    [{"sha256": "same", "bytes": 4, "physical_id": 1}]
                    for _ in range(2 if group_id < 36 else 1)
                ]
            groups.append(group)
        snapshot = dict(
            requests=[dict(request_id="cancel-survivor", groups=groups)],
            bytes_hashed=500,
        )
        audit = {
            key: [deepcopy(snapshot) for _ in range(4)] for key in ("before", "after")
        }
        self.assertTrue(compare_active_cache_audit(audit)["passed"])
        names = tuple(f"cancel-survivor-{i}" for i in range(3))
        multiple = dict(survivors={name: deepcopy(audit) for name in names})
        for name, item in multiple["survivors"].items():
            for side in ("before", "after"):
                for rank, row in enumerate(item[side]):
                    row["rank"] = rank
                    row["requests"][0]["request_id"] = name + "-1234abcd"
        self.assertTrue(compare_active_cache_audit(multiple, names)["passed"])
        for missing in ("survivor", "rank", "identity", "bytes"):
            bad = deepcopy(multiple)
            item = bad["survivors"][names[-1]]
            if missing == "survivor":
                del bad["survivors"][names[-1]]
            elif missing == "rank":
                item["before"][3]["rank"] = item["after"][3]["rank"] = 2
            elif missing == "identity":
                for side in ("before", "after"):
                    item[side][3]["requests"][0]["request_id"] = names[0]
            else:
                item["after"][3]["requests"][0]["groups"][40]["layers"][
                    "mtp.compressed_key_cache"
                ][0][0]["sha256"] = "changed"
            self.assertFalse(compare_active_cache_audit(bad, names)["passed"], missing)
        for field in ("before", "after"):
            bad = deepcopy(audit)
            bad[field].pop()
            self.assertFalse(compare_active_cache_audit(bad)["passed"])
        bad = deepcopy(audit)
        bad["after"][2]["requests"][0]["groups"][40]["layers"][
            "mtp.compressed_key_cache"
        ][0][0]["sha256"] = "changed"
        self.assertFalse(compare_active_cache_audit(bad)["passed"])
        for missing in ("group", "part", "page"):
            bad = deepcopy(audit)
            for field in ("before", "after"):
                affected = bad[field][2]["requests"][0]["groups"]
                if missing == "group":
                    affected.pop()
                elif missing == "part":
                    affected[0]["layers"]["target.0.linear_attn"].pop()
                else:
                    affected[0]["layers"]["target.0.linear_attn"][0].clear()
            self.assertFalse(compare_active_cache_audit(bad)["passed"], missing)

    def test_active_abort_audit_freezes_hashes_and_resumes_on_rpc_failure(self):
        import asyncio

        from prefix_cache_screen import ActiveCancellation

        async def run(fail):
            calls = []

            class Engine:
                async def pause_generation(self, **kwargs):
                    self.pause_args = kwargs
                    calls.append("pause")

                async def is_paused(self):
                    return True

                async def collective_rpc(self, method, args):
                    calls.append("snapshot")
                    self_method = "snapshot_active_cache_audit"
                    if method != self_method or args != ("survivor",):
                        raise AssertionError("Unexpected RPC")
                    if fail:
                        raise RuntimeError("snapshot failed")
                    return [{"bytes_hashed": 10}] * 4

                async def abort(self, name):
                    if name != "victim":
                        raise AssertionError("Unexpected victim")
                    calls.append("abort")

                async def resume_generation(self):
                    calls.append("resume")

            engine = Engine()
            control = ActiveCancellation(engine, "victim", "survivor", audit=True)
            output = NS(outputs=[NS(token_ids=[1] * 8)], finished=False)
            await control.observe("victim", output)
            if fail:
                with self.assertRaisesRegex(RuntimeError, "snapshot failed"):
                    await control.observe("survivor", output)
                self.assertFalse(control.acknowledged)
                self.assertEqual(calls, ["pause", "snapshot", "resume"])
            else:
                await control.observe("survivor", output)
                self.assertEqual(
                    calls, ["pause", "snapshot", "abort", "snapshot", "resume"]
                )
                self.assertEqual(
                    control.cache_audit["before"], control.cache_audit["after"]
                )
            self.assertEqual(engine.pause_args, {"mode": "keep", "clear_cache": False})

        asyncio.run(run(False))
        asyncio.run(run(True))

    def test_abort_ownership_allows_shared_reference_release_not_page_loss(self):
        from prefix_load_barrier import (
            compare_owned_request_blocks,
            owned_request_blocks,
        )

        a = NS(block_hash=b"a", ref_cnt=2, is_null=False)
        b = NS(block_hash=None, ref_cnt=1, is_null=False)
        null = NS(is_null=True)
        scheduler = NS(
            kv_cache_manager=NS(get_blocks=lambda _: NS(blocks=[[null, a], [b]]))
        )
        before = owned_request_blocks(scheduler, "survivor")
        a.ref_cnt -= 1
        after = owned_request_blocks(scheduler, "survivor")
        self.assertTrue(compare_owned_request_blocks(before, after)["passed"])
        self.assertFalse(compare_owned_request_blocks(before, {})["passed"])
        for bad in ((NS(), b"a", 1), (a, b"changed", 1), (a, b"a", 0)):
            self.assertFalse(
                compare_owned_request_blocks(before, {**after, (0, 1): bad})["passed"]
            )

    def test_active_abort_waits_for_two_live_streams_and_uses_native_api_once(self):
        import asyncio

        from prefix_cache_screen import ActiveCancellation

        async def run():
            aborted = []

            async def abort(name):
                aborted.append(name)

            controller = ActiveCancellation(NS(abort=abort), "victim", "survivor")

            def output(n, finished=False):
                return NS(outputs=[NS(token_ids=[1] * n)], finished=finished)

            await controller.observe("victim", output(8))
            await controller.observe("survivor", output(7))
            self.assertEqual(aborted, [])
            await controller.observe("survivor", output(8))
            self.assertEqual(aborted, ["victim"])
            self.assertTrue(controller.acknowledged)
            await controller.observe("victim", output(9, True))
            self.assertFalse(controller.at_abort["victim"]["finished"])
            self.assertEqual(aborted, ["victim"])

            controller = ActiveCancellation(NS(abort=abort), "victim", "survivor")
            await controller.observe("victim", output(512, True))
            await controller.observe("survivor", output(8))
            self.assertIsNone(controller.at_abort)
            self.assertFalse(controller.acknowledged)
            with self.assertRaises(ValueError):
                await controller.observe("unrelated", output(8))

        asyncio.run(run())

    def test_c4_cancel_verdict_rejects_missing_or_incomplete_participants(self):
        from copy import deepcopy
        from unittest.mock import patch

        from prefix_cache_screen import cancellation_verdict

        names = ["cancel-victim", *[f"cancel-survivor-{i}" for i in range(3)]]
        def row(name):
            return dict(name=name, tokens=[1] * 128, requested_output_tokens=128,
                        finish_reason="length", cached_tokens=28320,
                        output_completeness=dict(passed=True))
        rows = [row(name) for name in names]
        rows[0].update(
            tokens=[1] * 8, requested_output_tokens=512, finish_reason="abort"
        )
        control = dict(
            concurrency=4, context_tokens=32768, serving_decode=False,
            requests=rows, abort_acknowledged=True,
            at_abort={name: dict(tokens=8, finished=False) for name in names},
            max_running=4, load_bytes=100, frontend_drained=True,
            reuse=[row(f"cancel-reuse-{i}") for i in range(4)],
            audit_required=True,
            ownership_audit=dict(passed=True, victim_removed_from_running=True,
                survivors={name: dict(survivor=name, passed=True, survivor_running=True,
                    blocks=41, checks=dict(complete_owned_table=True,
                        same_pages_and_hashes=True, live_references=True))
                    for name in names[1:]}),
        )
        for i, (original, replay) in enumerate(zip(rows, control["reuse"])):
            for item in (original, replay):
                item.update(prompt_tokens=29756, prompt_sha256=f"{i:064x}",
                            cache_salt="shared")
        with patch("prefix_cache_screen.compare_active_cache_audit",
                   return_value=dict(passed=True)):
            self.assertTrue(cancellation_verdict(control)["passed"])
            for mode in ("missing", "partial", "abort", "reuse", "concurrency",
                         "ownership", "ownership_identity", "short_context",
                         "wrong_replay", "no_graph"):
                bad = deepcopy(control)
                if mode == "missing":
                    bad["requests"].pop()
                elif mode == "partial":
                    bad["requests"][-1]["tokens"] = [1] * 127
                elif mode == "abort":
                    del bad["at_abort"][names[-1]]
                elif mode == "reuse":
                    bad["reuse"][-1]["name"] = "cancel-reuse-0"
                elif mode == "concurrency":
                    bad["max_running"] = 2
                elif mode == "ownership":
                    del bad["ownership_audit"]["survivors"][names[-1]]
                elif mode == "ownership_identity":
                    peer = bad["ownership_audit"]["survivors"][names[-1]]
                    peer["survivor"] = names[1]
                elif mode == "short_context":
                    bad["requests"][-1]["prompt_tokens"] = 7100
                elif mode == "wrong_replay":
                    bad["reuse"][-1]["prompt_sha256"] = "wrong"
                else:
                    bad["serving_decode"] = True
                self.assertFalse(cancellation_verdict(bad)["passed"], mode)

    def test_active_cancel_rejects_missing_load_abort_cleanup_or_survivor(self):
        from copy import deepcopy

        from prefix_cache_screen import cancellation_verdict

        victim = dict(
            tokens=[1] * 8,
            finish_reason="abort",
            requested_output_tokens=512,
            output_completeness={"passed": True},
            cached_tokens=5664,
        )
        survivor = dict(
            tokens=[2] * 128, output_completeness={"passed": True}, cached_tokens=5664
        )
        control = dict(
            requests=[victim, survivor],
            abort_acknowledged=True,
            at_abort={
                name: {"tokens": 8, "finished": False}
                for name in ("victim", "survivor")
            },
            max_running=2,
            load_bytes=100,
            frontend_drained=True,
            reuse=[deepcopy(survivor), deepcopy(survivor)],
            numerical_consistency={"survivor_vs_reuse": {"equal": False}},
        )
        self.assertTrue(cancellation_verdict(control)["passed"])
        for key, value in (
            ("abort_acknowledged", False),
            ("at_abort", None),
            ("max_running", 1),
            ("load_bytes", 0),
            ("frontend_drained", False),
            ("reuse", []),
        ):
            self.assertFalse(cancellation_verdict({**control, key: value})["passed"])
        for path, value in (
            (("requests", 0, "finish_reason"), "length"),
            (("requests", 0, "tokens"), []),
            (("requests", 0, "tokens"), [1] * 512),
            (("requests", 1, "output_completeness", "passed"), False),
            (("requests", 1, "cached_tokens"), 0),
            (("reuse", 1, "cached_tokens"), 0),
            (("at_abort", "survivor", "finished"), True),
        ):
            bad = deepcopy(control)
            cursor = bad
            for part in path[:-1]:
                cursor = cursor[part]
            cursor[path[-1]] = value
            self.assertFalse(cancellation_verdict(bad)["passed"], path)

        from unittest.mock import patch

        expiry = dict(
            active_requests=2,
            protected_host_blocks=3,
            expired_host_blocks=4,
            expired_cpu_chunks=5,
            configured_ttl_seconds=3600,
            advanced_seconds=3601,
            clocks_restored=True,
            ownership_preserved=True,
        )
        audited = dict(
            control,
            audit_required=True,
            active_expiry_required=True,
            ownership_audit=dict(passed=True, active_expiry=expiry),
        )
        with patch(
            "prefix_cache_screen.compare_active_cache_audit",
            return_value={"passed": True},
        ):
            self.assertTrue(cancellation_verdict(audited)["passed"])
            for key in expiry:
                bad = deepcopy(audited)
                del bad["ownership_audit"]["active_expiry"][key]
                self.assertFalse(cancellation_verdict(bad)["passed"], key)

    def test_active_cancel_configuration_keeps_other_lifecycle_fixtures_separate(self):
        from prefix_cache_screen import validate_cancellation_observation

        args = dict(
            active_cancel=True,
            native_prefix=True,
            untraced_balanced_prefix=True,
            divergent_prefixes=True,
            prefix_concurrency=2,
            tokens=128,
        )
        validate_cancellation_observation(NS(**args))
        validate_cancellation_observation(NS())
        with self.assertRaisesRegex(ValueError, "paused active-cancel"):
            validate_cancellation_observation(NS(**args, expire_active_cache=True))
        validate_cancellation_observation(
            NS(
                **args,
                expire_active_cache=True,
                audit_active_cancel=True,
                paired_cold_loads=True,
            )
        )
        for key, value in (
            ("native_prefix", False),
            ("untraced_balanced_prefix", False),
            ("divergent_prefixes", False),
            ("prefix_concurrency", 1),
            ("tokens", 16),
            ("idle_expiry", True),
            ("pending_expiry", True),
            ("continuations", True),
        ):
            with self.assertRaises(ValueError):
                validate_cancellation_observation(NS(**{**args, key: value}))

    def test_c4_cancel_launch_requires_long_context_and_separate_audit_mode(self):
        from prefix_cache_screen import (
            validate_cancellation_observation, validate_continuation_observation,
            validate_serving_decode,
        )

        args = dict(active_cancel=True, cancellation_concurrency=4,
                    native_prefix=True, tp4=True, serving_decode=True,
                    untraced_balanced_prefix=True, balanced_prefix_prefill=True,
                    divergent_prefixes=True, prefix_concurrency=2, tokens=128,
                    prefix_prefill_budget=2048, continuation_context=32768)
        for audit in (False, True):
            variant = dict(args, audit_active_cancel=audit, paired_cold_loads=audit,
                           serving_decode=not audit)
            validate_cancellation_observation(NS(**variant))
            validate_continuation_observation(NS(**variant))
            validate_serving_decode(NS(**variant))
        for key, value in (("active_cancel", False), ("tp4", False),
                           ("continuation_context", 8192), ("no_mtp", True),
                           ("serving_decode", False), ("prefix_prefill_budget", 1024)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_cancellation_observation(NS(**{**args, key: value}))

    def test_complete_output_rejects_truncation_and_missing_probability_evidence(self):
        from copy import deepcopy

        from prefix_cache_screen import complete_request_output

        row = dict(
            tokens=[1, 2],
            logprobs=[{"1": -0.1, "3": -1.0}, {"2": -0.2}],
            nonfinite_logprobs=[],
        )
        self.assertTrue(complete_request_output(row, 2)["passed"])
        self.assertFalse(complete_request_output(row, 3)["passed"])
        self.assertFalse(complete_request_output({}, 0)["passed"])
        for field, value in (
            ("tokens", [1]),
            ("logprobs", [{"1": -0.1}]),
            ("logprobs", [{"3": -0.1}, {"2": -0.2}]),
            ("logprobs", [{"1": -0.1}, {"2": None}]),
            ("logprobs", [{"1": -0.1, "3": float("nan")}, {"2": -0.2}]),
            ("logprobs", [{"1": -0.1}, {"2": float("-inf")}]),
            ("nonfinite_logprobs", [{"position": 0, "token": 3, "value": "nan"}]),
        ):
            with self.subTest(field=field, value=value):
                bad = deepcopy(row)
                bad[field] = value
                self.assertFalse(complete_request_output(bad, 2)["passed"])

    def test_long_continuation_keeps_alignment_context_and_cache_budgets_bounded(self):
        from prefix_cache_screen import append_continuation, continuation_seed_tokens

        self.assertEqual(continuation_seed_tokens(8192), 5212)
        seed_length = continuation_seed_tokens(32768)
        self.assertEqual(seed_length, 29756)
        prompt = [1] * seed_length
        for expected in (30700, 31644, 32588):
            prompt = append_continuation(prompt, [2] * 128, [3], 128, 32768)
            self.assertEqual(len(prompt), expected)
            self.assertEqual(len(prompt) % 944, 492)
            self.assertLessEqual(len(prompt) + 128 + 4, 32768)
        with self.assertRaises(ValueError):
            append_continuation(prompt, [2] * 128, [3], 128, 32768)
        with self.assertRaises(ValueError):
            continuation_seed_tokens(240000)
        source = {"kv_cache_memory_bytes": 750000000}
        short = prefix_engine_config(source)
        long = prefix_engine_config(source, continuation_context=32768)
        self.assertEqual(source, {"kv_cache_memory_bytes": 750000000})
        self.assertEqual(short["max_model_len"], 8192)
        self.assertEqual(short["kv_cache_memory_bytes"], 750000000)
        self.assertEqual(long["max_model_len"], 32768)
        self.assertEqual(long["kv_cache_memory_bytes"], 1024**3)
        self.assertEqual(short["kv_transfer_config"], long["kv_transfer_config"])
        short_host = short["additional_config"]["flash_next_direct_host_kv"]
        long_host = long["additional_config"]["flash_next_direct_host_kv"]
        self.assertEqual(short_host["num_blocks"], 512)
        self.assertEqual(long_host["num_blocks"], 1024)
        self.assertEqual(short_host["max_bytes"], long_host["max_bytes"])

    def test_continuation_page_plan_pairs_producer_and_consumer_boundaries(self):
        from prefix_cache_screen import (
            continuation_page_boundaries,
            validate_continuation_observation,
        )
        from session_prefill_trace import prefix_page_readback_budget

        self.assertEqual(prefix_page_readback_budget(28320, 2), 4 * 1024**3)
        self.assertEqual(prefix_page_readback_budget(28320, 4), 8 * 1024**3)
        self.assertEqual(prefix_page_readback_budget(5664, 4), 2 * 1024**3)
        with self.assertRaises(ValueError):
            prefix_page_readback_budget(28320, 5)

        plan = continuation_page_boundaries(32768)
        self.assertEqual(len(plan), 8)
        for mode in ("hot", "cold"):
            self.assertEqual(plan[f"turn2-{mode}-0"], [28320, 29264])
            self.assertEqual(plan[f"turn2-{mode}-1"], [28320, 29264, 30208])
            self.assertEqual(plan[f"turn2-{mode}-3"], [30208])
            for turn, boundary in enumerate((28320, 29264, 30208), 1):
                self.assertIn(boundary, plan[f"turn2-{mode}-{turn - 1}"])
                self.assertEqual(plan[f"turn2-{mode}-{turn}"][0], boundary)
        self.assertEqual(2 * sum(map(len, plan.values())), 32)
        with self.assertRaises(ValueError):
            continuation_page_boundaries(8192)
        args = dict(
            native_prefix=True,
            continuations=True,
            tp4=True,
            prefix_concurrency=2,
            continuation_concurrency=2,
            continuation_context=32768,
            balanced_prefix_prefill=True,
            trace_prefix_pages=True,
            trace_prefix_shapes=True,
        )
        validate_continuation_observation(NS(**args))
        validate_balanced_observation(NS(**args))
        validate_continuation_observation(NS(**{**args, "continuation_concurrency": 4}))
        c4 = continuation_page_boundaries(32768, 4)
        self.assertEqual(c4, {name.replace("turn2-", "turn4-"): bounds
                              for name, bounds in plan.items()})
        self.assertEqual(4 * sum(map(len, c4.values())), 64)
        with self.assertRaises(ValueError):
            continuation_page_boundaries(32768, 6)
        for key, value in (
            ("continuation_context", 8192),
            ("continuation_concurrency", 1),
            ("tp4", False),
            ("balanced_prefix_prefill", False),
            ("trace_prefix_shapes", False),
        ):
            with self.assertRaises(ValueError):
                validate_continuation_observation(NS(**{**args, key: value}))

    def test_continuation_appends_real_answer_without_exceeding_context(self):
        from prefix_cache_screen import append_continuation

        seed, answer = [1] * 5212, list(range(128))
        prompt = seed
        for length in (6156, 7100, 8044):
            previous = list(prompt)
            prompt = append_continuation(prompt, answer, [2, 3], 128)
            self.assertEqual(len(prompt), length)
            self.assertEqual(prompt[: len(previous)], previous)
            self.assertEqual(prompt[len(previous) : len(previous) + 128], answer)
            self.assertLessEqual(len(prompt) + len(answer) + 4, 8192)
        self.assertEqual(len(seed), 5212)
        for args in (
            (prompt, answer, [2], 128),
            (seed, [], [2], 128),
            (seed, answer, [], 128),
            (seed, answer, [2], 127),
        ):
            with self.assertRaises(ValueError):
                append_continuation(*args)

    def test_continuation_requests_require_isolation_real_load_and_exact_output(self):
        from copy import deepcopy

        from prefix_cache_screen import compare_continuation_requests

        hot = dict(
            prompt_sha256="same",
            prompt_tokens=6156,
            repetition_penalty=1.05,
            cache_salt="hot",
            cached_tokens=3776,
            skip_cache=False,
            transfer_delta={},
            tokens=[1, 2],
            logprobs=[{"1": -0.1}],
        )
        cold = {
            **hot,
            "cache_salt": "cold",
            "transfer_delta": {"vllm:kv_offload_load_bytes": 100},
        }
        first = {**hot, "cache_salt": "ref-a", "cached_tokens": 0, "skip_cache": True}
        second = {**first, "cache_salt": "ref-b"}
        rows = [hot, cold, first, second]
        self.assertTrue(compare_continuation_requests(*rows, 5340)["passed"])
        for index, field, value in (
            (0, "cached_tokens", 0),
            (1, "transfer_delta", {}),
            (1, "cached_tokens", 4720),
            (2, "skip_cache", False),
            (3, "cache_salt", "hot"),
            (2, "prompt_sha256", "other"),
            (1, "prompt_tokens", 7100),
            (1, "repetition_penalty", 1.0),
            (0, "repetition_penalty", None),
        ):
            bad = deepcopy(rows)
            bad[index][field] = value
            self.assertFalse(compare_continuation_requests(*bad, 5340)["passed"], field)
        for index in range(4):
            for field, value in (("tokens", [3]), ("logprobs", [{"1": -1}])):
                bad = deepcopy(rows)
                bad[index][field] = value
                self.assertFalse(compare_continuation_requests(*bad, 5340)["passed"])
        self.assertFalse(compare_continuation_requests(*rows, 7000)["passed"])

    def test_continuation_driver_runs_reuse_before_references_and_replays_answers(self):
        import asyncio
        import hashlib

        from prefix_cache_screen import continuation_screen

        calls, rows = [], {}
        ignore_sampling = False

        async def generate(
            name,
            *,
            prompt_ids,
            cache_salt,
            skip_cache=False,
            tokens=128,
            repetition_penalty=1.0,
        ):
            calls.append(
                dict(
                    name=name, prompt=list(prompt_ids), salt=cache_salt, skip=skip_cache
                )
            )
            cached = (
                (len(prompt_ids) // 944 - 2) * 944
                if name.startswith(("turn-hot-", "turn-cold-"))
                and not name.endswith("-0")
                else 0
            )
            load = 100 if cached and name.startswith("turn-cold-") else 0
            row = dict(
                name=name,
                prompt_sha256=hashlib.sha256(str(prompt_ids).encode()).hexdigest(),
                prompt_tokens=len(prompt_ids),
                cache_salt=cache_salt,
                repetition_penalty=1.0 if ignore_sampling else repetition_penalty,
                skip_cache=skip_cache,
                cached_tokens=cached,
                transfer_delta={"vllm:kv_offload_load_bytes": load},
                tokens=list(range(tokens)),
                logprobs=[{"1": -0.1}],
            )
            rows[name] = row
            return row

        result = asyncio.run(
            continuation_screen(
                generate, [1] * 5212, [2, 3], lambda i: [i] * 7100, 128, 2
            )
        )
        self.assertTrue(result["passed"])
        self.assertTrue(result["changed_sampling"])
        self.assertEqual(
            [rows[f"turn-hot-{turn}"]["repetition_penalty"] for turn in range(4)],
            [1.0, 1.05, 1.1, 1.0],
        )
        self.assertEqual(
            [r["prompt_tokens"] for r in result["turns"]], [6156, 7100, 8044]
        )
        names = [r["name"] for r in calls]
        self.assertLess(names.index("turn-cold-3"), names.index("turn-reference-a-1"))
        self.assertEqual(sum("pressure" in n for n in names), 6)
        by_name = {r["name"]: r for r in calls}
        for turn in range(1, 4):
            old = by_name[f"turn-hot-{turn - 1}"]["prompt"]
            current = by_name[f"turn-hot-{turn}"]["prompt"]
            self.assertEqual(
                current[len(old) : len(old) + 128],
                rows[f"turn-hot-{turn - 1}"]["tokens"],
            )
            for mode in ("cold", "reference-a", "reference-b"):
                self.assertEqual(by_name[f"turn-{mode}-{turn}"]["prompt"], current)
                self.assertEqual(
                    rows[f"turn-{mode}-{turn}"]["repetition_penalty"],
                    rows[f"turn-hot-{turn}"]["repetition_penalty"],
                )
        self.assertTrue(all(r["skip"] for r in calls if "reference" in r["name"]))
        # Identical outputs cannot waive failure to exercise changed settings.
        ignore_sampling = True
        ignored = asyncio.run(
            continuation_screen(
                generate, [1] * 5212, [2, 3], lambda i: [i] * 7100, 128, 2
            )
        )
        self.assertFalse(ignored["changed_sampling"])
        self.assertFalse(ignored["passed"])
        ignore_sampling = False
        extended = asyncio.run(
            continuation_screen(
                generate, [1] * 29756, [2, 3], lambda i: [i] * 7100, 128, 2, 32768
            )
        )
        self.assertTrue(extended["passed"])
        self.assertEqual(extended["context_tokens"], 32768)
        self.assertEqual(
            [row["prompt_tokens"] for row in extended["turns"]],
            [30700, 31644, 32588],
        )

    def test_continuation_configuration_keeps_bounded_isolated_control(self):
        from prefix_cache_screen import validate_continuation_observation

        args = dict(
            continuations=True,
            native_prefix=True,
            untraced_balanced_prefix=True,
            tokens=128,
        )
        validate_continuation_observation(NS(**args))
        validate_continuation_observation(NS(**args, continuation_context=32768))
        for changes in (
            {"native_prefix": False},
            {"untraced_balanced_prefix": False},
            {"idle_expiry": True},
            {"pending_expiry": True},
            {"tokens": 0},
            {"tokens": 129},
            {"continuation_context": 240000},
            {"continuations": False, "continuation_context": 32768},
        ):
            with self.assertRaises(ValueError):
                validate_continuation_observation(NS(**{**args, **changes}))

    def test_concurrent_turns_separate_lifecycle_from_numerics_and_replay_hot_inputs(
        self,
    ):
        import asyncio
        import hashlib
        import json

        from prefix_cache_screen import (
            concurrent_continuation_screen,
            continuation_seed_tokens,
        )

        calls, pressure_calls = {}, []
        fault = None

        async def generate(name, **kwargs):
            pressure_calls.append((name, kwargs))

        async def batch(name, prompts, *, cache_salt, skip_cache, repetition_penalty):
            calls[name] = [list(prompt) for prompt in prompts]
            turn = int(name.rsplit("-", 1)[1])
            is_cold = "-cold-" in name
            cached = (
                (len(prompts[0]) // 944 - 2) * 944 if turn and not skip_cache else 0
            )
            load = 100 if cached and is_cold else 0
            rows = []
            for branch, prompt in enumerate(prompts):
                token = branch + 10 + (1 if fault == "numerics" and is_cold else 0)
                rows.append(
                    dict(
                        tokens=[token] * 128,
                        logprobs=[{str(token): -0.1}] * 128,
                        prompt_sha256=hashlib.sha256(
                            json.dumps(prompt).encode()
                        ).hexdigest(),
                        prompt_tokens=len(prompt),
                        cached_tokens=cached,
                        cache_salt=cache_salt,
                        skip_cache=skip_cache,
                        repetition_penalty=repetition_penalty,
                        transfer_delta={"vllm:kv_offload_load_bytes": load},
                    )
                )
            if fault == "nonfinite":
                rows[0]["logprobs"] = [{"10": float("nan")}] * 128
            if fault == "truncated" and turn == 3:
                rows[0]["tokens"] = [10] * 127
            if fault == "fourth_branch" and turn == 3:
                rows[3]["tokens"] = [13] * 127
            if fault == "missing_load":
                load = 0
            return dict(
                requests=rows,
                max_running=1 if fault == "serial" else len(prompts),
                load_bytes=load,
            )

        def run(context=8192, concurrency=2):
            seed_length = continuation_seed_tokens(context)
            return asyncio.run(
                concurrent_continuation_screen(
                    generate,
                    batch,
                    [
                        [1] * (seed_length - 1) + [branch + 2]
                        for branch in range(concurrency)
                    ],
                    [4],
                    lambda i: [i] * 7100,
                    128,
                    2,
                    context,
                )
            )

        result = run()
        self.assertTrue(result["lifecycle_passed"])
        self.assertTrue(result["numerical_consistency_passed"])
        self.assertEqual(len(pressure_calls), 6)  # Shared pressure, not per branch.
        names = list(calls)
        self.assertLess(names.index("turn2-cold-3"), names.index("turn2-reference-a-1"))
        for turn in range(1, 4):
            for mode in ("cold", "reference-a", "reference-b"):
                self.assertEqual(
                    calls[f"turn2-{mode}-{turn}"], calls[f"turn2-hot-{turn}"]
                )
            for branch in range(2):
                previous = calls[f"turn2-hot-{turn - 1}"][branch]
                current = calls[f"turn2-hot-{turn}"][branch]
                self.assertEqual(current[: len(previous)], previous)
                self.assertEqual(
                    current[len(previous) : len(previous) + 128], [10 + branch] * 128
                )
        fault = "numerics"
        result = run()
        self.assertTrue(result["lifecycle_passed"])
        self.assertFalse(result["numerical_consistency_passed"])
        self.assertFalse(result["passed"])
        for fault in ("serial", "missing_load", "nonfinite", "truncated"):
            result = run()
            self.assertFalse(result["lifecycle_passed"], fault)
        fault = None
        result = run(32768)
        self.assertTrue(result["passed"])
        for branch in range(2):
            self.assertEqual(
                [turn["branches"][branch]["prompt_tokens"] for turn in result["turns"]],
                [30700, 31644, 32588],
            )
        self.assertLessEqual(
            result["turns"][-1]["branches"][0]["prompt_tokens"] + 132, 32768
        )
        result = run(32768, 4)
        self.assertTrue(result["passed"])
        self.assertEqual(result["concurrency"], 4)
        for turn in range(1, 4):
            self.assertEqual(len(result["turns"][turn - 1]["branches"]), 4)
            for mode in ("cold", "reference-a", "reference-b"):
                self.assertEqual(
                    calls[f"turn4-{mode}-{turn}"], calls[f"turn4-hot-{turn}"]
                )
        fault = "fourth_branch"
        result = run(32768, 4)
        self.assertFalse(result["lifecycle_passed"])
        self.assertFalse(result["passed"])
        fault = "serial"
        self.assertFalse(run(32768, 4)["lifecycle_passed"])

    def test_concurrent_continuations_reject_unbounded_or_unpaired_modes(self):
        from prefix_cache_screen import validate_continuation_observation

        args = dict(
            continuations=True,
            continuation_concurrency=2,
            native_prefix=True,
            tp4=True,
            balanced_prefix_prefill=True,
            prefix_concurrency=2,
            untraced_balanced_prefix=True,
            tokens=128,
        )
        validate_continuation_observation(NS(**args))
        c4 = {**args, "continuation_concurrency": 4, "continuation_context": 32768}
        validate_continuation_observation(NS(**c4))
        validate_continuation_observation(NS(**{
            **c4, "trace_prefix_pages": True, "trace_prefix_shapes": True,
        }))
        for changes in (
            {"continuation_context": 8192},
            {"continuation_concurrency": 6},
        ):
            with self.assertRaises(ValueError):
                validate_continuation_observation(NS(**{**c4, **changes}))
        for changes in (
            {"continuations": False},
            {"tp4": False},
            {"balanced_prefix_prefill": False},
            {"prefix_concurrency": 1},
            {"continuation_concurrency": 3},
        ):
            with self.assertRaises(ValueError):
                validate_continuation_observation(NS(**{**args, **changes}))

    def test_c4_engine_slots_and_graphs_cover_four_mtp_requests(self):
        source = {"kv_cache_memory_bytes": 1024**3}
        for no_mtp in (False, True):
            config = prefix_engine_config(
                source,
                tp4=True,
                no_mtp=no_mtp,
                prefill_budget=2048,
                balanced_prefill=True,
                continuation_context=32768,
                serving_decode=True,
                continuation_concurrency=4,
            )
            self.assertEqual(config["max_num_seqs"], 4)
            self.assertEqual(config["max_model_len"], 32768)
            self.assertEqual(config["long_prefill_token_threshold"], 472)
            self.assertLessEqual(4 * config["long_prefill_token_threshold"],
                                 config["max_num_scheduled_tokens"])
            self.assertIn(
                4 if no_mtp else 20,
                config["compilation_config"]["cudagraph_capture_sizes"],
            )
        self.assertNotIn("max_num_seqs", source)
        with self.assertRaises(ValueError):
            prefix_engine_config(source, continuation_concurrency=4)

    def test_native_copy_audit_tracks_store_versions_and_rejects_corrupt_transfers(
        self,
    ):
        import hashlib

        from session_transfer_audit import NativeCopyAudit, locate_fragment

        class ByteRows:
            ndim, dtype = 2, "torch.int8"

            def __init__(self, pointer):
                self.pointer = pointer
                self.shape = (3, 8)
                self.data = bytearray(36)

            def stride(self, dimension):
                return (12, 1)[dimension]

            def element_size(self):
                return 1

            def data_ptr(self):
                return self.pointer

            def __setitem__(self, key, value):
                self[key][:] = value

            def __getitem__(self, key):
                row, section = key
                return memoryview(self.data)[row * 12 : (row + 1) * 12][section]

        gpu, cpu = ByteRows(1000), ByteRows(2000)
        gpu[0, :8] = b"abcdefgh"
        syncs, copied = [], []
        corrupt = False

        def copy(src, dst, sizes, **kwargs):
            copied.append(kwargs)
            for a, b, size in zip(
                src.tolist(), dst.tolist(), sizes.tolist(), strict=True
            ):
                source = locate_fragment([gpu, cpu], a, size)[1]
                target = locate_fragment([gpu, cpu], b, size)[1]
                target[:] = source
                if corrupt:
                    target[0] ^= 1

        def descriptors(*values):
            return NS(tolist=lambda: list(values))

        store = NS(
            src_tensors=[gpu],
            dst_tensors=[cpu],
            gpu_to_cpu=True,
            _transfers=[],
            _canonical_copy_plans=None,
            _swap_blocks_batch=copy,
        )
        load = NS(
            src_tensors=[cpu],
            dst_tensors=[gpu],
            gpu_to_cpu=False,
            _transfers=[],
            _canonical_copy_plans=None,
            _swap_blocks_batch=copy,
        )
        audit = NativeCopyAudit(
            digest=lambda view: hashlib.sha256(view).hexdigest(),
            synchronize=lambda: syncs.append(True),
        )
        worker = NS(_store_handler=store, _load_handler=load, _mmap_region=None)
        audit.install(worker)
        with self.assertRaisesRegex(ValueError, "store version"):
            load._swap_blocks_batch(
                descriptors(2000), descriptors(1012), descriptors(8)
            )
        self.assertEqual(copied, [])  # A fabricated CPU source must not be consumed.
        store._swap_blocks_batch(
            descriptors(1000),
            descriptors(2000),
            descriptors(8),
            is_src_access_order_any=False,
        )
        load._swap_blocks_batch(
            descriptors(2000),
            descriptors(1012),
            descriptors(8),
            is_src_access_order_any=True,
        )
        self.assertEqual(bytes(gpu[1, :8]), b"abcdefgh")
        self.assertEqual(
            copied,
            [{"is_src_access_order_any": False}, {"is_src_access_order_any": True}],
        )
        gpu[0, :8] = b"newvalue"
        store._swap_blocks_batch(descriptors(1000), descriptors(2000), descriptors(8))
        load._swap_blocks_batch(descriptors(2000), descriptors(1024), descriptors(8))
        self.assertEqual(bytes(gpu[2, :8]), b"newvalue")
        cpu[0, :8] = b"tampered"
        with self.assertRaisesRegex(ValueError, "store version"):
            load._swap_blocks_batch(
                descriptors(2000), descriptors(1012), descriptors(8)
            )
        cpu[0, :8] = b"newvalue"
        corrupt = True
        with self.assertRaisesRegex(ValueError, "different bytes"):
            load._swap_blocks_batch(
                descriptors(2000), descriptors(1012), descriptors(8)
            )
        result = audit.remove()
        self.assertIs(store._swap_blocks_batch, copy)
        self.assertIs(load._swap_blocks_batch, copy)
        self.assertEqual(result["retained_versions"], 1)
        self.assertEqual(len(result["records"]), 5)
        self.assertTrue(all(row["exact"] for row in result["records"][:-1]))
        self.assertFalse(result["records"][-1]["exact"])
        self.assertGreaterEqual(len(syncs), 10)
        for address, size in ((999, 8), (1008, 1), (1004, 8), (1032, 8), (1000, 0)):
            with self.assertRaises(ValueError):
                locate_fragment([gpu], address, size)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            locate_fragment([gpu, gpu], 1000, 8)
        # Long model fixtures exceed 4096 copy calls without exceeding pool bounds.
        from session_transfer_audit import MAX_COPY_RECORDS

        corrupt = False
        long_audit = NativeCopyAudit(
            digest=lambda view: hashlib.sha256(view).hexdigest(),
            synchronize=lambda: None,
        )
        for _ in range(4097):
            long_audit.copy(
                store, copy, descriptors(1000), descriptors(2000), descriptors(8)
            )
        self.assertEqual(len(long_audit.records), 4097)
        self.assertTrue(all(row["exact"] for row in long_audit.records))
        self.assertEqual(len(long_audit.versions), 1)
        long_audit.records = [long_audit.records[0]] * MAX_COPY_RECORDS
        copied_before = len(copied)
        with self.assertRaisesRegex(ValueError, "record limit: 32768/32768"):
            long_audit.copy(
                store, copy, descriptors(1000), descriptors(2000), descriptors(8)
            )
        self.assertEqual(len(copied), copied_before)

    def test_native_copy_audit_requires_four_real_bidirectional_rank_observations(self):
        from copy import deepcopy

        from prefix_cache_screen import validate_copy_audit
        from session_transfer_audit import copy_audit_verdict

        args = dict(
            audit_native_copies=True,
            native_prefix=True,
            tp4=True,
            private_cold_buffers=True,
            balanced_prefix_prefill=True,
            prefix_concurrency=2,
        )
        validate_copy_audit(NS(**args))
        validate_balanced_observation(NS(**args))
        for changes in (
            {"private_cold_buffers": False},
            {"tp4": False},
            {"active_cancel": True},
            {"trace_prefix_pages": True},
            {"untraced_balanced_prefix": True},
        ):
            with self.assertRaises(ValueError):
                validate_copy_audit(NS(**{**args, **changes}))
        ranks = [
            dict(
                rank=i,
                records=[
                    dict(
                        direction=direction,
                        exact=True,
                        source_unchanged=True,
                        bytes=8,
                        descriptors=1,
                        stored_version_verified=direction == "load",
                    )
                    for direction in ("store", "load")
                ],
            )
            for i in range(4)
        ]
        self.assertTrue(copy_audit_verdict(ranks)["passed"])
        self.assertFalse(copy_audit_verdict([])["passed"])
        self.assertFalse(copy_audit_verdict(ranks[:3])["passed"])
        for field in ("exact", "source_unchanged", "stored_version_verified", "bytes"):
            bad = deepcopy(ranks)
            bad[2]["records"][1][field] = False
            self.assertFalse(copy_audit_verdict(bad)["passed"], field)
        bad = deepcopy(ranks)
        bad[0]["records"] = []
        self.assertFalse(copy_audit_verdict(bad)["passed"])
        bad = deepcopy(ranks)
        bad[3]["rank"] = 2
        self.assertFalse(copy_audit_verdict(bad)["passed"])

    def test_native_copy_audit_cuda_roundtrip_with_relocated_pages(self):
        import os

        if os.environ.get("FLASH_NATIVE_COPY_AUDIT_CUDA") != "1":
            self.skipTest("Run only in the authorized isolated CUDA diagnostic")
        import torch
        from session_transfer_audit import NativeCopyAudit

        from vllm.v1.kv_offload.base import (
            CanonicalKVCacheRef,
            CanonicalKVCaches,
            CanonicalKVCacheTensor,
            GPULoadStoreSpec,
        )
        from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
        from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker

        self.assertTrue(torch.cuda.is_available())
        for page_bytes in (256, 128 * 1024):
            with self.subTest(page_bytes=page_bytes):
                cache = (
                    (torch.arange(8 * page_bytes, device="cuda") % 113)
                    .to(torch.int8)
                    .reshape(8, page_bytes)
                )
                expected = cache.cpu().clone()
                expected[5] = expected[1]
                expected[7] = expected[3]
                worker = CPUOffloadingWorker(
                    kv_caches=CanonicalKVCaches(
                        tensors=[
                            CanonicalKVCacheTensor(
                                tensor=cache, page_size_bytes=page_bytes
                            )
                        ],
                        group_data_refs=[
                            [
                                CanonicalKVCacheRef(
                                    tensor_idx=0, page_size_bytes=page_bytes
                                )
                            ]
                        ],
                    ),
                    blocks_per_chunk=2,
                    num_cpu_chunks=4,
                )
                audit = NativeCopyAudit()
                try:
                    audit.install(worker)
                    self.assertTrue(
                        worker.submit_store(
                            1,
                            GPULoadStoreSpec(
                                [1, 3], group_sizes=(2,), block_indices=(0,)
                            ),
                            CPULoadStoreSpec([2]),
                        )
                    )
                    worker.wait({1})
                    self.assertEqual({row.job_id for row in worker.get_finished()}, {1})
                    self.assertTrue(
                        worker.submit_load(
                            2,
                            CPULoadStoreSpec([2]),
                            GPULoadStoreSpec(
                                [5, 7], group_sizes=(2,), block_indices=(0,)
                            ),
                        )
                    )
                    worker.wait({2})
                    self.assertEqual({row.job_id for row in worker.get_finished()}, {2})
                    self.assertTrue(torch.equal(cache.cpu(), expected))
                    self.assertEqual(
                        [row["direction"] for row in audit.records], ["store", "load"]
                    )
                    self.assertTrue(all(row["exact"] for row in audit.records))
                    self.assertEqual(
                        [row["bytes"] for row in audit.records], [2 * page_bytes] * 2
                    )
                finally:
                    audit.remove()
                    worker.shutdown()

    def test_pending_expiry_protects_real_owners_and_rejects_unpinned_sources(self):
        from prefix_load_barrier import expire_pending_cache

        host = NS(monotonic=lambda: 10)
        cpu_module = NS(monotonic=lambda: 20)
        clocks = (host.monotonic, cpu_module.monotonic)
        block = NS(ref_cnt=1, is_null=False, block_hash="shared")
        destination = NS(ref_cnt=1, is_null=False, block_hash=None)
        chunk = NS(ref_cnt=1)
        policy = {"key": chunk}

        class Pool:
            idle_ttl_seconds = 3600
            blocks = [block]

            def expire_idle(inner):
                self.assertEqual(host.monotonic(), 3611)
                return 3

        class CPU:
            idle_ttl_seconds = 3600
            _expired_chunks = 0
            _policy = policy

            def _expire_idle(inner):
                self.assertEqual(cpu_module.monotonic(), 3621)
                inner._expired_chunks += 4

        host.IdleExpiringHostBlockPool = Pool
        cpu = CPU()
        job = NS(req_id="batch-cold-0", keys={"key"}, is_store=False, pending_count=4)
        scheduler = NS(
            requests={job.req_id: NS(status=NS(name="WAITING_FOR_REMOTE_KVS"))},
            connector=NS(connector_scheduler=NS(_jobs={1: job}, manager=cpu)),
            kv_cache_manager=NS(
                coordinator=NS(single_type_managers=[NS(block_pool=Pool())]),
                get_blocks=lambda request_id: NS(blocks=[[block], [destination]]),
            ),
        )
        report = expire_pending_cache(scheduler, {1: object()}, host, cpu_module)
        self.assertTrue(report["ownership_preserved"])
        self.assertEqual(report["protected_cpu_chunks"], 1)
        self.assertEqual(report["protected_host_blocks"], 1)
        self.assertEqual(report["owned_destination_blocks"], 2)
        self.assertEqual(report["expired_host_blocks"], 3)
        self.assertEqual(report["expired_cpu_chunks"], 4)
        for target, field, bad in (
            (job, "is_store", True),
            (job, "pending_count", 0),
            (chunk, "ref_cnt", 0),
            (destination, "ref_cnt", 0),
            (block, "ref_cnt", 0),
        ):
            original = getattr(target, field)
            setattr(target, field, bad)
            with self.assertRaises(RuntimeError):
                expire_pending_cache(scheduler, {1: object()}, host, cpu_module)
            setattr(target, field, original)

        def corrupt():
            policy["key"] = NS(ref_cnt=1)

        cpu._expire_idle = corrupt
        with self.assertRaisesRegex(RuntimeError, "protected load ownership"):
            expire_pending_cache(scheduler, {1: object()}, host, cpu_module)
        self.assertIs(host.monotonic, clocks[0])
        self.assertIs(cpu_module.monotonic, clocks[1])

    def test_active_expiry_preserves_owners_and_restores_clocks_on_corruption(self):
        from prefix_load_barrier import expire_active_cache

        host, cpu_module = NS(monotonic=lambda: 10), NS(monotonic=lambda: 20)
        clocks = host.monotonic, cpu_module.monotonic
        block = NS(ref_cnt=2, is_null=False, block_hash="shared")
        chunk = NS(ref_cnt=-1, chunk_id=3)
        policy = {"writing": chunk}

        class Pool:
            idle_ttl_seconds = 3600
            blocks = [block]

            def expire_idle(inner):
                self.assertEqual(host.monotonic(), 3611)
                return 3

        class CPU:
            idle_ttl_seconds = 3600
            _expired_chunks = 0
            _policy = policy

            def _expire_idle(inner):
                self.assertEqual(cpu_module.monotonic(), 3621)
                inner._expired_chunks += 4

        host.IdleExpiringHostBlockPool = Pool
        cpu = CPU()
        scheduler = NS(
            requests={name: NS(status=NS(name="RUNNING")) for name in ("a", "b")},
            connector=NS(
                connector_scheduler=NS(
                    manager=cpu, _jobs={1: NS(pending_count=4, keys={"writing"})}
                )
            ),
            kv_cache_manager=NS(
                coordinator=NS(single_type_managers=[NS(block_pool=Pool())]),
                get_blocks=lambda name: NS(blocks=[[block]]),
            ),
        )
        evidence = expire_active_cache(scheduler, ["a", "b"], host, cpu_module)
        self.assertTrue(evidence["ownership_preserved"])
        self.assertEqual(evidence["active_requests"], 2)
        self.assertEqual(evidence["protected_host_blocks"], 1)
        self.assertEqual(evidence["protected_cpu_chunks"], 1)
        self.assertEqual(evidence["expired_host_blocks"], 3)
        self.assertEqual(evidence["expired_cpu_chunks"], 4)
        for names in (["a"], ["a", "a"], ["a", "missing"]):
            with self.assertRaises(RuntimeError):
                expire_active_cache(scheduler, names, host, cpu_module)
        scheduler.requests["a"].status.name = "WAITING_FOR_REMOTE_KVS"
        with self.assertRaises(RuntimeError):
            expire_active_cache(scheduler, ["a", "b"], host, cpu_module)
        scheduler.requests["a"].status.name = "RUNNING"
        for target, field, changed in (
            (block, "ref_cnt", 1),
            (block, "block_hash", "wrong-prefix"),
            (chunk, "ref_cnt", 0),
            (chunk, "chunk_id", 4),
        ):
            original = getattr(target, field)
            cpu._expire_idle = (
                lambda target=target, field=field, changed=changed: setattr(
                    target, field, changed
                )
            )
            with self.assertRaisesRegex(RuntimeError, "protected ownership"):
                expire_active_cache(scheduler, ["a", "b"], host, cpu_module)
            setattr(target, field, original)
            self.assertIs(host.monotonic, clocks[0])
            self.assertIs(cpu_module.monotonic, clocks[1])

    def test_active_expiry_precedes_native_abort_without_changing_its_result(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from prefix_load_barrier import PairedLoadMixin

        calls = []
        block = NS(ref_cnt=2, is_null=False, block_hash="shared")

        class Native:
            def finish_requests(self, ids, status):
                calls.append("abort")
                del self.requests[ids[0]]
                self.running = list(self.requests.values())
                block.ref_cnt -= 1
                return "native-result"

        class Scheduler(PairedLoadMixin, Native):
            pass

        with TemporaryDirectory() as directory:
            scheduler = object.__new__(Scheduler)
            scheduler._diagnostic_active_abort_path = Path(directory) / "abort.json"
            scheduler._diagnostic_active_expiry = True
            scheduler.requests = {
                name: NS(request_id=name, status=NS(name="RUNNING"))
                for name in ("cancel-victim", "cancel-survivor")
            }
            scheduler.running = list(scheduler.requests.values())
            scheduler.kv_cache_manager = NS(
                get_blocks=lambda name: NS(blocks=[[block]])
            )
            with patch("prefix_load_barrier.expire_active_cache") as expiry:
                expiry.side_effect = lambda *args: calls.append("expire") or {}
                self.assertEqual(
                    scheduler.finish_requests("cancel-victim", NS()), "native-result"
                )
                expiry.assert_called_once_with(
                    scheduler, ["cancel-victim", "cancel-survivor"]
                )
            self.assertEqual(calls, ["expire", "abort"])

    def test_active_expiry_with_native_host_pool_and_cpu_offload_manager(self):
        """Exercise real pool expiry and transfer pins without loading a model."""
        self._check_native_active_expiry(2)

    def test_c4_active_expiry_with_native_host_pool_and_cpu_offload_manager(self):
        """Protect all four owners while expiring unrelated native cache entries."""
        self._check_native_active_expiry(4)

    def _check_native_active_expiry(self, concurrency):
        import os
        from unittest.mock import patch

        if os.environ.get("FLASH_NATIVE_EXPIRY_CPU") != "1":
            self.skipTest("Requires the native vLLM CPU dependency environment")
        from prefix_load_barrier import expire_active_cache

        from vllm.distributed.kv_events import MEDIUM_CPU
        from vllm.v1.core.block_pool import BlockPool
        from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
        from vllm.v1.hisparse.block_pool import IdleExpiringHostBlockPool
        from vllm.v1.kv_offload.base import LookupResult, ReqContext, make_offload_key
        from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

        with (
            patch("vllm.v1.hisparse.block_pool.monotonic", return_value=0),
            patch("vllm.v1.kv_offload.cpu.manager.monotonic", return_value=0),
        ):
            device = BlockPool(8, True, 4, enable_kv_cache_events=True)
            host = IdleExpiringHostBlockPool(
                8, True, 4, medium=MEDIUM_CPU, event_owner=device
            )
            shared, *tails, idle = host.get_new_blocks(concurrency + 2)
            for index, block in enumerate((shared, *tails, idle)):
                host._insert_block_hash(
                    make_block_hash_with_group_id(BlockHash(str(index).encode()), 0),
                    block,
                    num_tokens=4,
                )
            for _ in range(concurrency - 1):
                host.touch([shared])
            host.free_blocks([idle])
            gpu = device.get_new_blocks(concurrency)
            ctx = ReqContext(req_id="expiry-native")
            cpu = CPUOffloadingManager(
                num_chunks=3, store_threshold=0, idle_ttl_seconds=3600
            )
            idle_key, read_key, write_key = [
                make_offload_key(str(i).encode(), 0) for i in range(3)
            ]
            self.assertIsNotNone(cpu.prepare_store([idle_key, read_key], ctx))
            cpu.complete_store([idle_key, read_key], ctx)
            cpu.prepare_load([read_key], ctx)
            self.assertIsNotNone(cpu.prepare_store([write_key], ctx))
            groups = {
                str(i): [[shared, tail], [gpu[i]]]
                for i, tail in enumerate(tails)
            }
            scheduler = NS(
                requests={name: NS(status=NS(name="RUNNING")) for name in groups},
                connector=NS(
                    connector_scheduler=NS(
                        manager=cpu,
                        _jobs={
                            1: NS(keys={read_key}, pending_count=4),
                            2: NS(keys={write_key}, pending_count=4),
                        },
                    )
                ),
                kv_cache_manager=NS(
                    coordinator=NS(single_type_managers=[NS(block_pool=host)]),
                    get_blocks=lambda name: NS(blocks=groups[name]),
                ),
            )
            evidence = expire_active_cache(scheduler, list(groups))
            self.assertTrue(evidence["ownership_preserved"])
            self.assertTrue(evidence["clocks_restored"])
            self.assertEqual(evidence["expired_host_blocks"], 1)
            self.assertEqual(evidence["expired_cpu_chunks"], 1)
            self.assertEqual(evidence["protected_cpu_chunks"], 2)
            self.assertEqual(evidence["active_requests"], concurrency)
            self.assertEqual(set(evidence["ownership"]), set(groups))
            self.assertEqual(evidence["protected_host_blocks"], concurrency + 1)
            self.assertEqual(shared.ref_cnt, concurrency)
            self.assertIsNone(host.get_cached_block(
                BlockHash(str(concurrency + 1).encode()), [0]
            ))
            self.assertEqual(cpu.lookup(idle_key, ctx), LookupResult.MISS)
            self.assertEqual(cpu.lookup(read_key, ctx), LookupResult.HIT)
            self.assertEqual(cpu.lookup(write_key, ctx), LookupResult.HIT_PENDING)
            host.free_blocks([shared, tails[0]])
            device.free_blocks([gpu[0]])
            self.assertEqual(shared.ref_cnt, concurrency - 1)
            self.assertTrue(all(tail.ref_cnt == 1 for tail in tails[1:]))
            cpu.complete_load([read_key], ctx)
            cpu.complete_store([write_key], ctx)
            for i, tail in enumerate(tails[1:], 1):
                host.free_blocks([shared, tail])
                device.free_blocks([gpu[i]])
            self.assertEqual(host.get_num_free_blocks(), 7)
            self.assertEqual(device.get_num_free_blocks(), 7)

    def test_pending_expiry_hook_preserves_native_schedule_and_runs_once(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from prefix_load_barrier import PairedLoadMixin

        metadata = NS(load_jobs={1: object()})
        output = NS(kv_connector_metadata=metadata)

        class Native:
            def schedule(self, throttle_prefills=False):
                self.called = throttle_prefills
                return output

        class Diagnostic(PairedLoadMixin, Native):
            pass

        scheduler = object.__new__(Diagnostic)
        first = NS(request_id="batch-cold-0", status=NS(name="WAITING_FOR_REMOTE_KVS"))
        second = NS(request_id="batch-cold-1", status=NS(name="WAITING"))
        scheduler.requests = {row.request_id: row for row in (first, second)}
        job = NS(req_id=first.request_id, is_store=False, pending_count=4)
        scheduler.connector = NS(connector_scheduler=NS(_jobs={1: job}))
        scheduler._diagnostic_pending_expired = False
        with tempfile.TemporaryDirectory() as root:
            scheduler._diagnostic_pending_expiry_path = Path(root) / "pending.json"

            def expire(instance, jobs):
                self.assertIs(instance, scheduler)
                self.assertTrue(instance.called)
                self.assertEqual(jobs, {1: job})
                return {"ownership_preserved": True}

            with patch(
                "prefix_load_barrier.expire_pending_cache", side_effect=expire
            ) as mock:
                self.assertIs(scheduler.schedule(throttle_prefills=True), output)
                mock.assert_not_called()
                second.status.name = "WAITING_FOR_REMOTE_KVS"
                self.assertIs(scheduler.schedule(throttle_prefills=True), output)
                self.assertIs(scheduler.schedule(), output)
                self.assertEqual(mock.call_count, 1)
                self.assertTrue(scheduler._diagnostic_pending_expiry_path.exists())

    def test_pending_expiry_gate_requires_real_protection_eviction_and_exact_c2(self):
        from copy import deepcopy

        from prefix_cache_screen import compare_pending_expiry

        evidence = dict(
            acquired_cold_requests=2,
            pending_load_jobs=1,
            protected_host_blocks=2,
            protected_cpu_chunks=3,
            owned_destination_blocks=4,
            expired_host_blocks=5,
            expired_cpu_chunks=6,
            ownership_preserved=True,
            clocks_restored=True,
            advanced_seconds=3601,
            configured_ttl_seconds=3600,
        )
        concurrent = dict(
            cold=dict(load_bytes=100, max_running=2),
            hot_cold_comparisons=[dict(tokens=dict(equal=True), logprobs_exact=True)]
            * 2,
        )
        self.assertTrue(compare_pending_expiry(evidence, concurrent)["passed"])
        for key in evidence:
            self.assertFalse(
                compare_pending_expiry({**evidence, key: 0}, concurrent)["passed"], key
            )
        for field in ("load_bytes", "max_running"):
            bad = deepcopy(concurrent)
            bad["cold"][field] = 0
            self.assertFalse(compare_pending_expiry(evidence, bad)["passed"])
        for field in ("tokens", "logprobs_exact"):
            bad = deepcopy(concurrent)
            bad["hot_cold_comparisons"][0][field] = (
                {"equal": False} if field == "tokens" else False
            )
            self.assertFalse(compare_pending_expiry(evidence, bad)["passed"])

    def test_idle_expiry_requires_quiescence_and_restores_private_clocks(self):
        from prefix_load_barrier import expire_idle_cache

        host = NS(monotonic=lambda: 10)
        cpu_module = NS(monotonic=lambda: 20)
        clocks = (host.monotonic, cpu_module.monotonic)

        class Pool:
            idle_ttl_seconds = 3600
            blocks = [NS(ref_cnt=0, is_null=False), NS(ref_cnt=1, is_null=True)]

            def expire_idle(inner):
                self.assertEqual(host.monotonic(), 3611)
                return 3

        class CPU:
            idle_ttl_seconds = 3600
            _expired_chunks = 5

            def _expire_idle(inner):
                self.assertEqual(cpu_module.monotonic(), 3621)
                inner._expired_chunks += 4

        host.IdleExpiringHostBlockPool = Pool
        pool, cpu = Pool(), CPU()
        connector = NS(_jobs={}, _req_status={}, manager=cpu)
        scheduler = NS(
            requests={},
            connector=NS(connector_scheduler=connector),
            kv_cache_manager=NS(
                coordinator=NS(single_type_managers=[NS(block_pool=pool)])
            ),
        )
        result = expire_idle_cache(scheduler, host, cpu_module)
        self.assertEqual(
            (result["expired_host_blocks"], result["expired_cpu_chunks"]), (3, 4)
        )
        self.assertTrue(result["clocks_restored"])
        for owner, attribute in (
            (scheduler, "requests"),
            (connector, "_jobs"),
            (connector, "_req_status"),
        ):
            setattr(owner, attribute, {"active": True})
            with self.assertRaisesRegex(RuntimeError, "drained"):
                expire_idle_cache(scheduler, host, cpu_module)
            setattr(owner, attribute, {})
        pool.blocks[0].ref_cnt = 1
        with self.assertRaisesRegex(RuntimeError, "unowned"):
            expire_idle_cache(scheduler, host, cpu_module)
        pool.blocks[0].ref_cnt = 0
        cpu.idle_ttl_seconds = 60
        with self.assertRaisesRegex(RuntimeError, "3600"):
            expire_idle_cache(scheduler, host, cpu_module)
        cpu.idle_ttl_seconds = 3600

        def fail():
            raise RuntimeError("injected expiry error")

        cpu._expire_idle = fail
        with self.assertRaisesRegex(RuntimeError, "injected"):
            expire_idle_cache(scheduler, host, cpu_module)
        self.assertIs(host.monotonic, clocks[0])
        self.assertIs(cpu_module.monotonic, clocks[1])

    def test_idle_expiry_trigger_precedes_native_admission_and_runs_once(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from prefix_load_barrier import PairedLoadMixin

        calls = []

        class Native:
            def add_request(self, request):
                calls.append(request.request_id)

        class Diagnostic(PairedLoadMixin, Native):
            pass

        scheduler = object.__new__(Diagnostic)
        scheduler._diagnostic_expired = False
        with tempfile.TemporaryDirectory() as root:
            scheduler._diagnostic_expiry_path = Path(root) / "expiry.json"

            def expire(instance):
                self.assertEqual(calls, ["ordinary"])
                self.assertIs(instance, scheduler)
                return {"expired_host_blocks": 1}

            with patch(
                "prefix_load_barrier.expire_idle_cache", side_effect=expire
            ) as mock:
                scheduler.add_request(NS(request_id="ordinary"))
                scheduler.add_request(NS(request_id="expiry-trigger-0123abcd"))
                self.assertEqual(mock.call_count, 1)
                self.assertTrue(scheduler._diagnostic_expiry_path.exists())
                with self.assertRaisesRegex(RuntimeError, "only once"):
                    scheduler.add_request(NS(request_id="expiry-trigger"))
        self.assertEqual(calls, ["ordinary", "expiry-trigger-0123abcd"])

    def test_idle_expiry_gate_requires_real_miss_and_exact_recomputation(self):
        from copy import deepcopy

        from prefix_cache_screen import compare_idle_expiry

        evidence = dict(
            expired_host_blocks=2,
            expired_cpu_chunks=3,
            configured_ttl_seconds=3600,
            advanced_seconds=3601,
            quiescent=True,
            clocks_restored=True,
        )
        expired = dict(
            prompt_sha256="same",
            cached_tokens=0,
            skip_cache=False,
            transfer_delta={},
            tokens=[1, 2],
            logprobs=[{"1": -0.1}],
        )
        reference = {**expired, "skip_cache": True}
        hot = {**expired, "cached_tokens": 944}
        args = [evidence, expired, reference, deepcopy(hot), hot]
        self.assertTrue(compare_idle_expiry(*args)["passed"])
        for index, field, value in (
            (0, "expired_host_blocks", 0),
            (0, "expired_cpu_chunks", 0),
            (0, "clocks_restored", False),
            (0, "quiescent", False),
            (1, "skip_cache", True),
            (1, "cached_tokens", 1),
            (1, "transfer_delta", {"vllm:kv_offload_load_bytes": 1}),
            (2, "tokens", [3]),
            (2, "logprobs", [{"1": -1}]),
            (2, "skip_cache", False),
            (3, "cached_tokens", 0),
            (3, "tokens", [3]),
            (4, "prompt_sha256", "different"),
        ):
            bad = deepcopy(args)
            bad[index][field] = value
            self.assertFalse(compare_idle_expiry(*bad)["passed"], (index, field))

    def test_untraced_balanced_followup_is_explicit_and_cannot_hide_tracing(self):
        args = dict(balanced_prefix_prefill=True, prefix_concurrency=2)
        with self.assertRaises(ValueError):
            validate_balanced_observation(NS(**args))
        validate_balanced_observation(NS(**args, trace_prefix_shapes=True))
        untraced = {**args, "untraced_balanced_prefix": True}
        validate_balanced_observation(NS(**untraced))
        for flag in (
            "trace_prefix_shapes",
            "trace_prefix_pages",
            "trace_prefix_prefill",
            "trace_prompt_chunks",
            "trace_prefill",
            "trace_host_registration",
        ):
            with self.assertRaises(ValueError):
                validate_balanced_observation(NS(**untraced, **{flag: True}))
        for changes in ({"prefix_concurrency": 1}, {"balanced_prefix_prefill": False}):
            with self.assertRaises(ValueError):
                validate_balanced_observation(NS(**{**untraced, **changes}))

    def test_load_mixin_delegates_native_promotion_and_records_both_requests(self):
        import json
        import tempfile
        from pathlib import Path

        from prefix_load_barrier import PairedLoadMixin

        class Native:
            def __init__(self, **kwargs):
                self.requests = {}
                self.finished_recving_kv_req_ids = set()
                self.failed_recving_kv_req_ids = set()
                self.calls = []

            def _try_promote_blocked_waiting_request(self, request):
                self.calls.append(request.request_id)
                self.finished_recving_kv_req_ids.remove(request.request_id)
                request.status = NS(name="WAITING")
                return True

        class Diagnostic(PairedLoadMixin, Native):
            pass

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "barrier.json"
            config = NS(
                additional_config={"flash_next_diagnostic_load_barrier": str(path)},
                scheduler_config=NS(async_scheduling=True),
            )
            scheduler = Diagnostic(vllm_config=config)
            a, b = [
                NS(
                    request_id=f"batch-cold-{i}",
                    status=NS(name="WAITING_FOR_REMOTE_KVS"),
                )
                for i in range(2)
            ]
            scheduler.requests = {r.request_id: r for r in (a, b)}
            scheduler.finished_recving_kv_req_ids.add(a.request_id)
            self.assertFalse(scheduler._try_promote_blocked_waiting_request(a))
            self.assertEqual(scheduler.calls, [])
            scheduler.finished_recving_kv_req_ids.add(b.request_id)
            self.assertTrue(scheduler._try_promote_blocked_waiting_request(a))
            self.assertTrue(scheduler._try_promote_blocked_waiting_request(b))
            result = json.loads(path.read_text())
            self.assertTrue(result["released"])
            self.assertEqual(result["promoted"], [a.request_id, b.request_id])
            self.assertEqual(result["deferred_calls"], 1)
            self.assertEqual(scheduler.finished_recving_kv_req_ids, set())

            # Opt-in C4 turns retain the original pair report and delegate acks.
            config.additional_config.update(
                flash_next_diagnostic_load_barrier=str(Path(root) / "c4.json"),
                flash_next_diagnostic_c4_load_gate=True,
            )
            scheduler = Diagnostic(vllm_config=config)
            for turn in (1, 2, 3):
                peers = [NS(request_id=f"turn4-cold-{turn}-{i}",
                            status=NS(name="WAITING_FOR_REMOTE_KVS"))
                         for i in range(4)]
                scheduler.requests = {p.request_id: p for p in peers}
                scheduler.finished_recving_kv_req_ids = {
                    p.request_id for p in peers[:3]
                }
                self.assertFalse(scheduler._try_promote_blocked_waiting_request(peers[0]))
                scheduler.finished_recving_kv_req_ids.add(peers[3].request_id)
                for peer in peers:
                    self.assertTrue(scheduler._try_promote_blocked_waiting_request(peer))
                self.assertEqual(scheduler.finished_recving_kv_req_ids, set())
            result = json.loads((Path(root) / "c4.json").read_text())
            self.assertEqual(result["promoted"], [])
            self.assertEqual(set(result["continuation_turns"]), {"1", "2", "3"})
            for row in result["continuation_turns"].values():
                self.assertTrue(row["released"])
                self.assertEqual(row["promoted"], row["ready_request_ids"])
                self.assertEqual(len(row["promoted"]), 4)

    def test_load_gate_waits_for_both_native_acks_without_editing_ownership(self):
        from prefix_load_barrier import PairLoadGate

        gate = PairLoadGate()
        a, b = [
            NS(
                request_id=f"batch-cold-{i}-abcdef01",
                status=NS(name="WAITING_FOR_REMOTE_KVS"),
            )
            for i in range(2)
        ]
        requests = {r.request_id: r for r in (a, b)}
        finished = {a.request_id}
        self.assertTrue(gate.hold(a, requests, finished, set(), 0))
        self.assertEqual(finished, {a.request_id})
        self.assertEqual(a.status.name, "WAITING_FOR_REMOTE_KVS")
        finished.add(b.request_id)
        self.assertFalse(gate.hold(a, requests, finished, set(), 1))
        self.assertTrue(gate.released)
        finished.remove(a.request_id)  # Native promotion consumes its acknowledgement.
        self.assertFalse(gate.hold(b, requests, finished, set(), 2))
        self.assertEqual(finished, {b.request_id})
        self.assertEqual(gate.ready, [a.request_id, b.request_id])

    def test_load_gate_is_scoped_and_fails_closed_on_missing_or_failed_peer(self):
        from prefix_load_barrier import PairLoadGate

        gate = PairLoadGate(timeout=3)
        unrelated = NS(request_id="cold", status=NS(name="WAITING_FOR_REMOTE_KVS"))
        self.assertFalse(gate.hold(unrelated, {}, set(), set(), 0))
        self.assertIsNone(gate.started)
        a = NS(request_id="batch-cold-0", status=NS(name="WAITING_FOR_REMOTE_KVS"))
        self.assertTrue(gate.hold(a, {a.request_id: a}, {a.request_id}, set(), 0))
        with self.assertRaises(TimeoutError):
            gate.hold(a, {a.request_id: a}, {a.request_id}, set(), 3)
        with self.assertRaisesRegex(RuntimeError, "failed native load"):
            PairLoadGate().hold(a, {a.request_id: a}, set(), {a.request_id}, 0)
        duplicate = NS(request_id="batch-cold-0-abcdef01", status=a.status)
        with self.assertRaisesRegex(AssertionError, "Ambiguous"):
            PairLoadGate().hold(
                a, {a.request_id: a, duplicate.request_id: duplicate}, set(), set(), 0
            )

    def test_c4_continuation_gate_waits_for_all_acks_and_isolates_turns(self):
        from prefix_load_barrier import ContinuationLoadGate

        for turn in (1, 2, 3):
            gate = ContinuationLoadGate(turn, timeout=3)
            peers = [NS(request_id=f"turn4-cold-{turn}-{i}-abcdef01",
                        status=NS(name="WAITING_FOR_REMOTE_KVS")) for i in range(4)]
            requests = {p.request_id: p for p in peers}
            done = {p.request_id for p in peers[:3]}
            self.assertTrue(gate.hold(peers[0], requests, done, set(), 0))
            self.assertEqual(len(done), 3)
            with self.assertRaises(TimeoutError):
                gate.hold(peers[0], requests, done, set(), 3)
            with self.assertRaises(RuntimeError):
                gate.hold(peers[0], requests, done, {peers[3].request_id}, 1)
            done.add(peers[3].request_id)
            self.assertFalse(gate.hold(peers[0], requests, done, set(), 1))
            self.assertEqual(gate.ready, [p.request_id for p in peers])
            done.remove(peers[0].request_id)
            self.assertFalse(gate.hold(peers[1], requests, done, set(), 2))
            self.assertIsNone(gate.branch(f"turn4-hot-{turn}-0"))
            self.assertIsNone(gate.branch(f"turn4-cold-{turn % 3 + 1}-0"))
            self.assertIsNone(gate.branch(f"turn4-cold-{turn}-4"))

    def test_batch_admission_uses_native_pause_and_restores_after_failure(self):
        import asyncio

        async def run(fail):
            class Engine:
                def __init__(self):
                    self.resumed = asyncio.Event()
                    self.names = []
                    self.pause_args = None

                async def pause_generation(self, **kwargs):
                    self.pause_args = kwargs

                async def resume_generation(self):
                    self.resumed.set()

                async def add_request(self, name):
                    if fail == "timeout":
                        await asyncio.Event().wait()
                    if fail is True and name == "b":
                        raise ValueError("native admission failed")
                    self.names.append(name)

            engine = Engine()

            async def generate(name):
                await engine.add_request(name)
                await engine.resumed.wait()
                self.assertEqual(engine.names, ["a", "b"])
                return name

            calls = [lambda: generate("a"), lambda: generate("b")]
            if fail == "timeout":
                with self.assertRaisesRegex(TimeoutError, "native admission"):
                    await admitted_batch(engine, calls, timeout=0.01)
            elif fail:
                with self.assertRaisesRegex(ValueError, "native admission failed"):
                    await admitted_batch(engine, calls)
            else:
                self.assertEqual(await admitted_batch(engine, calls), ["a", "b"])
            self.assertNotIn("add_request", vars(engine))
            self.assertTrue(engine.resumed.is_set())
            self.assertEqual(engine.pause_args, {"mode": "keep", "clear_cache": False})

        asyncio.run(run(False))
        asyncio.run(run(True))
        asyncio.run(run("timeout"))

    def test_balanced_prefill_uses_native_limits_without_mutating_source(self):
        source = {"long_prefill_token_threshold": 0}
        config = prefix_engine_config(
            source, tp4=True, prefill_budget=2048, balanced_prefill=True
        )
        self.assertEqual(config["long_prefill_token_threshold"], 944)
        self.assertNotIn("max_num_partial_prefills", config)
        self.assertNotIn("max_long_partial_prefills", config)
        self.assertEqual(config["max_num_scheduled_tokens"], 2048)
        self.assertTrue(config["enable_chunked_prefill"])
        self.assertEqual(source, {"long_prefill_token_threshold": 0})
        self.assertEqual(
            prefix_engine_config(source)["long_prefill_token_threshold"], 0
        )
        for tp4, budget in ((False, 2048), (True, 1024), (True, None)):
            with self.assertRaisesRegex(ValueError, "requires TP4/budget2048"):
                prefix_engine_config(
                    source, tp4=tp4, prefill_budget=budget, balanced_prefill=True
                )

    def test_execution_comparison_distinguishes_own_inputs_from_batch_geometry(self):
        from copy import deepcopy

        query = dict(
            request_id="reference",
            row_start=0,
            row_end=5,
            position_ranges=[[7100, 7105]],
            input_ids_sha256="same",
            is_prefilling=False,
        )
        left = [
            dict(total_rows=5, request_batch=dict(requests=[query], padding_rows=0))
        ]
        right = deepcopy(left)
        right[0]["request_batch"]["requests"][0]["request_id"] = "cold"
        same = compare_request_execution(left, "reference", right, "cold", 7100)
        self.assertTrue(same["observed"])
        self.assertIsNone(same["first_packed_query_difference"])
        right[0]["total_rows"] = 949
        right[0]["request_batch"]["padding_rows"] = 944
        mixed = compare_request_execution(left, "reference", right, "cold", 7100)
        self.assertIsNone(mixed["first_own_query_difference"])
        self.assertEqual(mixed["first_packed_query_difference"], 0)
        right[0]["request_batch"]["requests"][0]["input_ids_sha256"] = "different"
        changed = compare_request_execution(left, "reference", right, "cold", 7100)
        self.assertEqual(changed["first_own_query_difference"], 0)
        missing = compare_request_execution(left, "absent", right, "cold", 7100)
        self.assertFalse(missing["observed"])
        self.assertEqual(missing["first_own_query_difference"], 0)

    def test_prefill_convolution_hash_excludes_scratch_for_both_storage_layouts(self):
        import torch

        page = torch.arange(36, dtype=torch.float32).reshape(4, 9)
        for dim_first in (True, False):
            view = page.clone() if dim_first else page.clone().T
            canonical = view if dim_first else view.T
            window = prefill_convolution_window(view, 3, dim_first)
            self.assertTrue(torch.equal(window, page[:, :3]))
            before = hash_page_chunks(window)
            canonical[:, 3:] += 100
            self.assertEqual(hash_page_chunks(window), before)
            canonical[:, 1] += 1
            self.assertNotEqual(hash_page_chunks(window), before)
            ple = prefill_convolution_window(view, 3, dim_first, capacity=7)
            self.assertTrue(torch.equal(ple, canonical[:, 2:5]))
        with self.assertRaises(ValueError):
            prefill_convolution_window(page, 10, True)

    def test_request_batch_trace_preserves_native_ids_and_actual_mixed_rows(self):
        import torch

        batch = NS(
            req_ids=["decode-01234567", "prefill-89abcdef"],
            query_start_loc=torch.tensor([0, 5, 9]),
            input_ids=torch.arange(10, dtype=torch.int32),
            is_prefilling_np=[False, True],
        )
        positions = torch.tensor([7100, 7101, 7102, 7103, 7104, 944, 945, 946, 947, 0])
        result = describe_request_batch(batch, positions)
        self.assertEqual(result["padding_rows"], 1)
        self.assertEqual(result["requests"][0]["position_ranges"], [[7100, 7105]])
        self.assertEqual(result["requests"][1]["position_ranges"], [[944, 948]])
        self.assertEqual(result["requests"][1]["request_id"], "prefill-89abcdef")
        before = result["requests"][0]["input_ids_sha256"]
        batch.input_ids[0] += 1
        self.assertNotEqual(
            describe_request_batch(batch, positions)["requests"][0]["input_ids_sha256"],
            before,
        )
        batch.query_start_loc[-1] = 11
        with self.assertRaises(ValueError):
            describe_request_batch(batch, positions)

    def test_page_coverage_accepts_native_ids_but_rejects_missing_or_wrong_owners(self):
        def snapshot(*ids):
            return dict(
                bytes_hashed=16, requests=[dict(request_id=name) for name in ids]
            )

        expected = {"batch-hot-0", "batch-hot-1"}
        rows = index_prefix_page_requests(
            [snapshot("batch-hot-0-a39b5d00", "batch-hot-1")], expected
        )
        self.assertEqual(rows["batch-hot-0"]["request_id"], "batch-hot-0-a39b5d00")
        for ids in (
            ("batch-hot-0-a39b5d00",),
            ("batch-cold-0-a39b5d00", "batch-hot-1"),
            ("batch-hot-0-a39b5d000", "batch-hot-1"),
            ("batch-hot-0-notuuid!", "batch-hot-1"),
            ("batch-hot-0-a39b5d00", "batch-hot-0-a39b5d01", "batch-hot-1"),
        ):
            with self.assertRaises(ValueError):
                index_prefix_page_requests([snapshot(*ids)], expected)
        empty = snapshot("batch-hot-0", "batch-hot-1")
        empty["bytes_hashed"] = 0
        with self.assertRaises(ValueError):
            index_prefix_page_requests([empty], expected)

        pair = [snapshot("batch-hot-0", "batch-hot-1") for _ in range(2)]
        for value, boundary in zip(pair, (28320, 29264)):
            for request in value["requests"]:
                request["boundary"] = boundary
        for boundary in (28320, 29264):
            matched = index_prefix_page_requests(pair, expected, boundary=boundary)
            self.assertEqual({r["boundary"] for r in matched.values()}, {boundary})
        with self.assertRaises(ValueError):
            index_prefix_page_requests(pair, expected, boundary=30208)
        with self.assertRaises(ValueError):
            index_prefix_page_requests(pair + pair, expected, boundary=28320)

    def test_chunked_page_hash_matches_full_bytes_for_strided_and_bf16_pages(self):
        import hashlib

        import torch

        for dtype in (torch.float32, torch.bfloat16):
            backing = torch.arange(90, dtype=dtype).reshape(9, 10)
            view = backing[:, ::2]
            before = backing.clone()
            expected = hashlib.sha256(
                view.contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            self.assertEqual(hash_page_chunks(view, max_chunk_bytes=40), expected)
            self.assertTrue(torch.equal(backing, before))
            with self.assertRaisesRegex(ValueError, "chunk bound"):
                hash_page_chunks(view, max_chunk_bytes=1)

    def test_page_comparison_matches_content_not_physical_ids_and_keeps_missing_pages(
        self,
    ):
        from copy import deepcopy

        entry = dict(
            logical_index=0,
            physical_id=1,
            sha256="original",
            bytes=8,
            shape=[2],
            dtype="float32",
            before_boundary=True,
        )
        left = dict(
            request_id="reference",
            boundary=4,
            groups=[
                dict(
                    group_id=0,
                    host_resident=True,
                    spec_type="DirectHostAttentionSpec",
                    block_size=4,
                    layers={"target": [[entry]]},
                )
            ],
        )
        right = deepcopy(left)
        right["request_id"] = "cold"
        right["groups"][0]["layers"]["target"][0][0]["physical_id"] = 7
        result = compare_prefix_pages(left, right)
        row = next(iter(result["summaries"].values()))
        self.assertEqual(row["same_content"], 1)
        self.assertEqual(row["changed_physical_ids"], 1)
        right["groups"][0]["host_resident"] = False
        self.assertFalse(
            compare_prefix_pages(left, right)["mismatches"][0]["layout_equal"]
        )
        right["groups"][0]["host_resident"] = True
        right["groups"][0]["layers"]["target"][0][0]["sha256"] = "changed"
        self.assertEqual(len(compare_prefix_pages(left, right)["mismatches"]), 1)
        entry["prefill_convolution"] = dict(sha256="same-history", history_tokens=3)
        other = right["groups"][0]["layers"]["target"][0][0]
        other["prefill_convolution"] = deepcopy(entry["prefill_convolution"])
        row = next(iter(compare_prefix_pages(left, right)["summaries"].values()))
        self.assertEqual(row["different_content"], 1)
        self.assertEqual(row["prefill_convolution"]["equal"], 1)
        other["prefill_convolution"]["sha256"] = "changed-history"
        row = next(iter(compare_prefix_pages(left, right)["summaries"].values()))
        self.assertEqual(row["prefill_convolution"]["different"], 1)
        del other["prefill_convolution"]
        row = next(iter(compare_prefix_pages(left, right)["summaries"].values()))
        self.assertEqual(row["prefill_convolution"]["missing"], 1)
        right["groups"][0]["layers"]["target"][0].clear()
        row = next(iter(compare_prefix_pages(left, right)["summaries"].values()))
        self.assertEqual(row["only_reference"], 1)
        self.assertEqual(row["shared_entries"], 0)
        right["boundary"] = 8
        with self.assertRaises(ValueError):
            compare_prefix_pages(left, right)

    def test_prefix_page_tables_preserve_manager_ids_and_reject_invalid_splits(self):
        self.assertEqual(manager_block_ids([0, 1, 6, 7], 2), [0, 3])
        for ids, ratio in (([1], 0), ([0, 1, 2], 2), ([3, 4], 2)):
            with self.assertRaises(ValueError):
                manager_block_ids(ids, ratio)

    def test_prefix_page_snapshot_hashes_separate_pools_without_changing_pages(self):
        from unittest.mock import patch

        import numpy as np
        import torch

        gpu = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        host = torch.arange(16, dtype=torch.float32).reshape(4, 4) + 100
        originals = [gpu.clone(), host.clone()]
        runner = NS(
            kv_cache_config=NS(
                num_blocks=3,
                direct_host_num_blocks=4,
                kv_cache_groups=[
                    NS(host_resident=False, layer_names=["gdn"], kv_cache_spec=NS()),
                    NS(host_resident=True, layer_names=["qsa"], kv_cache_spec=NS()),
                ],
            ),
            compilation_config=NS(
                static_forward_context={
                    "gdn": NS(kv_cache=gpu),
                    "qsa": NS(_qsa_host_kv=host, kv_cache=None),
                }
            ),
            req_states=NS(req_id_to_index={"test": 0}),
            block_tables=NS(
                num_blocks=NS(np=np.array([[4], [3]])),
                block_tables=[
                    NS(gpu=torch.tensor([[2, 3, 4, 5]])),
                    NS(gpu=torch.tensor([[1, 2, 3]])),
                ],
                blocks_per_kv_block=[2, 1],
                block_sizes=[4, 4],
            ),
        )
        batch = NS(
            req_ids=["test"],
            is_prefilling_np=[True],
            num_computed_tokens_np=[8],
            query_start_loc_np=[0, 2],
            positions=torch.tensor([8, 9]),
            idx_mapping_np=[0],
        )
        first = snapshot_prefix_pages(runner, batch, 8)
        self.assertTrue(torch.equal(gpu, originals[0]))
        self.assertTrue(torch.equal(host, originals[1]))
        groups = first["requests"][0]["groups"]
        self.assertEqual(groups[0]["block_ids"], [1, 2])
        self.assertFalse(groups[1]["layers"]["qsa"][0][2]["before_boundary"])
        host[1, 0] += 1
        second = snapshot_prefix_pages(runner, batch, 8)
        self.assertEqual(groups[0], second["requests"][0]["groups"][0])
        self.assertNotEqual(
            groups[1]["layers"], second["requests"][0]["groups"][1]["layers"]
        )
        with patch("session_prefill_trace.hash_page_chunks") as hasher:
            with self.assertRaisesRegex(ValueError, "needs 112 bytes; limit is 1"):
                snapshot_prefix_pages(runner, batch, 8, max_bytes=1)
            hasher.assert_not_called()
        batch.is_prefilling_np = [False]
        batch.num_computed_tokens_np = [12]
        batch.positions = torch.tensor([12, 13])
        active = snapshot_prefix_pages(runner, batch, 8, active_request_ids={"test"})
        self.assertEqual(active["requests"], second["requests"])
        with self.assertRaisesRegex(ValueError, "current batch"):
            snapshot_prefix_pages(runner, batch, 8, active_request_ids={"missing"})
        batch.is_prefilling_np = [True]
        batch.num_computed_tokens_np = [8]
        batch.positions[0] = 7
        with self.assertRaises(ValueError):
            snapshot_prefix_pages(runner, batch, 8)

    def test_long_prefix_reader_keeps_complete_pages_and_private_tail_distinct(self):
        from unittest.mock import patch

        import numpy as np
        import torch

        host = torch.arange(35 * 4, dtype=torch.float32).reshape(35, 4)
        original = host.clone()
        runner = NS(
            kv_cache_config=NS(
                num_blocks=2,
                direct_host_num_blocks=35,
                kv_cache_groups=[
                    NS(host_resident=True, layer_names=["qsa"], kv_cache_spec=NS())
                ],
            ),
            compilation_config=NS(
                static_forward_context={"qsa": NS(_qsa_host_kv=host)}
            ),
            req_states=NS(req_id_to_index={"long": 0}),
            block_tables=NS(
                num_blocks=NS(np=np.array([[33]])),
                block_tables=[NS(gpu=torch.arange(1, 34).reshape(1, -1))],
                blocks_per_kv_block=[1],
                block_sizes=[944],
            ),
        )
        batch = NS(
            req_ids=["long"],
            is_prefilling_np=[True],
            num_computed_tokens_np=[30208],
            query_start_loc_np=[0, 1],
            positions=torch.tensor([30208]),
            idx_mapping_np=[0],
        )
        result = snapshot_prefix_pages(runner, batch, 30208)
        pages = result["requests"][0]["groups"][0]["layers"]["qsa"][0]
        self.assertEqual([p["before_boundary"] for p in pages], [True] * 32 + [False])
        self.assertEqual(len({p["sha256"] for p in pages}), 33)
        self.assertEqual(result["bytes_hashed"], 33 * 16)
        self.assertTrue(torch.equal(host, original))
        for boundary in (True, 0, 32768, 30208.0):
            with self.assertRaises(ValueError):
                snapshot_prefix_pages(runner, batch, boundary)
        # Metadata-only tensors cross the old work limit without allocating GBs.
        owner = runner.compilation_config.static_forward_context["qsa"]
        owner._qsa_host_kv = torch.empty((35, 24 * 1024**2), device="meta")
        with patch(
            "session_prefill_trace.hash_page_chunks", return_value="0" * 64
        ) as hash_page:
            with self.assertRaisesRegex(ValueError, "limit is 2147483648"):
                snapshot_prefix_pages(runner, batch, 30208)
            hash_page.assert_not_called()
            result = snapshot_prefix_pages(runner, batch, 30208, max_bytes=4 * 1024**3)
            self.assertEqual(result["bytes_hashed"], 33 * 96 * 1024**2)
            self.assertEqual(hash_page.call_count, 33)
            self.assertEqual(result["max_chunk_bytes"], 8 * 1024**2)
            hash_page.reset_mock()
            owner._qsa_host_kv = torch.empty((35, 36 * 1024**2), device="meta")
            with self.assertRaisesRegex(ValueError, "limit is 4294967296"):
                snapshot_prefix_pages(runner, batch, 30208, max_bytes=4 * 1024**3)
            hash_page.assert_not_called()

    def test_prefix_page_hook_delegates_inputs_and_restores_native_method(self):
        from unittest.mock import patch

        import torch

        class Qwen4ExpDecoderLayer(torch.nn.Module):
            def forward(self, *, positions):
                return positions

        class Runner:
            req_states = block_tables = None

            def prepare_inputs(self, value):
                return value

        runner, layer = Runner(), Qwen4ExpDecoderLayer()
        worker = PrefillTraceWorkerExtension()
        worker.model_runner = runner
        worker.get_model = lambda: layer
        worker.install_prefix_shape_trace(7100, True)
        batch = NS(
            req_ids=["hot"],
            query_start_loc=torch.tensor([0, 2]),
            input_ids=torch.tensor([123, 124]),
            is_prefilling_np=[True],
        )
        self.assertIs(runner.prepare_inputs(batch), batch)
        worker.begin_prefix_shape_trace("hot")
        snapshot = dict(requests=[dict(request_id="hot")], bytes_hashed=16)
        with patch(
            "session_prefill_trace.snapshot_prefix_pages", return_value=snapshot
        ) as spy:
            layer(positions=torch.tensor([5664, 5665]))
            spy.assert_called_once_with(runner, batch, 5664, max_bytes=2 * 1024**3)
            batch.is_prefilling_np = [False]
            layer(positions=torch.tensor([7100, 7101]))
        result = worker.collect_prefix_shape_trace()
        self.assertEqual(result["pages"], {"hot": [snapshot]})
        self.assertEqual(len(result["phases"]["hot"]), 2)
        decode = result["phases"]["hot"][1]
        self.assertEqual(decode["prefill_rows"], 0)
        self.assertEqual(
            decode["request_batch"]["requests"][0]["position_ranges"], [[7100, 7102]]
        )
        self.assertNotIn("prepare_inputs", runner.__dict__)
        self.assertIs(runner.prepare_inputs.__func__, Runner.prepare_inputs)

        # Reuse the same hook for explicit long-context continuation boundaries.
        worker.install_prefix_shape_trace(32768, True)
        runner.prepare_inputs(batch)
        batch.is_prefilling_np = [True]
        for boundary in (
            True,
            0,
            -1,
            32768,
            32769,
            30208.0,
            [],
            [28320] * 2,
            [29264, 28320],
            [27376, 28320, 29264, 30208],
            [28320, True],
        ):
            with self.assertRaises(ValueError):
                worker.begin_prefix_shape_trace("invalid", boundary)
        with patch(
            "session_prefill_trace.snapshot_prefix_pages", return_value=snapshot
        ) as spy:
            for boundary in (28320, 29264, 30208):
                worker.begin_prefix_shape_trace(f"turn-{boundary}", boundary)
                layer(positions=torch.tensor([boundary, boundary + 1]))
                spy.assert_called_with(runner, batch, boundary, max_bytes=4 * 1024**3)
            worker.begin_prefix_shape_trace("triple", (28320, 29264, 30208))
            for boundary in (28320, 29264, 30208):
                layer(positions=torch.tensor([boundary, boundary + 1]))
                spy.assert_called_with(runner, batch, boundary, max_bytes=4 * 1024**3)
            worker.begin_prefix_shape_trace("default")
            layer(positions=torch.tensor([5664, 5665]))
            spy.assert_called_with(runner, batch, 5664, max_bytes=2 * 1024**3)
            self.assertEqual(spy.call_count, 7)
        for index in range(27):
            worker.begin_prefix_shape_trace(f"unused-{index}")
        with self.assertRaises(ValueError):
            worker.begin_prefix_shape_trace("too-many")
        worker.collect_prefix_shape_trace()
        self.assertNotIn("prepare_inputs", runner.__dict__)

    def test_probability_diagnostic_excludes_changed_history_after_token_flip(self):
        left = dict(tokens=[1, 2, 3], logprobs=[{"1": -0.1}, {"2": -0.5}, {"3": -1.0}])
        right = dict(
            tokens=[1, 4, 3], logprobs=[{"1": -0.1}, {"2": -0.75}, {"3": -90.0}]
        )
        result = compare_common_history_logprobs(left, right)
        self.assertEqual(result["first_token_difference"], 1)
        self.assertEqual(result["first_logprob_difference"], 1)
        self.assertEqual(result["compared_positions"], 2)
        self.assertEqual(result["max_common_token_logprob_delta"], 0.25)
        self.assertTrue(result["complete_common_history"])
        incomplete = compare_common_history_logprobs(left, {**right, "logprobs": []})
        self.assertFalse(incomplete["complete_common_history"])
        self.assertIsNone(incomplete["max_common_token_logprob_delta"])
        with self.assertRaises(ValueError):
            compare_common_history_logprobs(left, {**right, "logprobs": [{"1": None}]})

    def test_probability_diagnostic_reports_absent_topk_overlap(self):
        left = dict(tokens=[1], logprobs=[{"1": -0.1}])
        right = dict(tokens=[2], logprobs=[{"2": -0.1}])
        result = compare_common_history_logprobs(left, right)
        self.assertEqual(result["min_topk_intersection"], 0)
        self.assertIsNone(result["max_common_token_logprob_delta"])
        self.assertEqual(result["first_logprob_difference"], 0)

    def test_suffix_shape_comparison_keeps_chunk_boundaries_and_packed_rows(self):
        def step(ranges, total=None):
            rows = sum(end - start for start, end in ranges)
            return dict(
                ranges=ranges,
                prefill_rows=rows,
                total_rows=rows if total is None else total,
            )

        suffix = [step([[4, 6], [4, 6]]), step([[6, 8], [6, 8]])]
        reference = [step([[0, 4], [0, 4]])] + suffix
        self.assertTrue(
            compare_prefix_shapes(reference, suffix, 4)["same_packed_steps"]
        )
        # Equal total tokens do not imply equal kernel shapes.
        combined = [step([[4, 8], [4, 8]])]
        self.assertFalse(
            compare_prefix_shapes(reference, combined, 4)["same_packed_steps"]
        )
        mixed = [step([[4, 6], [4, 6]], 5), suffix[1]]
        self.assertFalse(
            compare_prefix_shapes(reference, mixed, 4)["same_packed_steps"]
        )
        crossing = [step([[0, 6], [0, 6]]), suffix[1]]
        self.assertFalse(
            compare_prefix_shapes(crossing, suffix, 4)["same_packed_steps"]
        )

    def test_missing_shape_evidence_never_counts_as_matching(self):
        result = compare_prefix_shapes([], [], 0)
        self.assertFalse(result["observed"])
        self.assertFalse(result["same_packed_steps"])
        with self.assertRaises(ValueError):
            compare_prefix_shapes([], [], -1)

    def test_shape_trace_records_packed_prefills_and_removes_its_hook(self):
        import torch

        class Qwen4ExpDecoderLayer(torch.nn.Module):
            def forward(self, *, positions):
                return positions

        layer = Qwen4ExpDecoderLayer()
        worker = PrefillTraceWorkerExtension()
        worker.get_model = lambda: layer
        worker.install_prefix_shape_trace(8)
        with self.assertRaises(ValueError):
            worker.install_prefix_shape_trace(8)
        worker.begin_prefix_shape_trace("reference")
        layer(positions=torch.tensor([0, 1, 2, 3]))
        layer(positions=torch.tensor([8, 9]))
        worker.begin_prefix_shape_trace(None)
        layer(positions=torch.tensor([0, 1]))
        worker.begin_prefix_shape_trace("batch-hot")
        layer(positions=torch.tensor([[4, 5, 4, 5, 8]]).repeat(3, 1))
        with self.assertRaises(ValueError):
            worker.begin_prefix_shape_trace("reference")
        result = worker.collect_prefix_shape_trace()
        self.assertEqual(
            result["phases"],
            {
                "reference": [dict(ranges=[[0, 4]], prefill_rows=4, total_rows=4)],
                "batch-hot": [
                    dict(ranges=[[4, 6], [4, 6]], prefill_rows=4, total_rows=5)
                ],
            },
        )
        self.assertFalse(layer._forward_pre_hooks)
        self.assertFalse(hasattr(worker, "_prefix_shape_trace"))

    def test_prefill_control_changes_only_diagnostic_budgets_not_cache_or_mtp(self):
        source = dict(
            max_num_batched_tokens=2048,
            max_num_scheduled_tokens=2048,
            speculative_config={"method": "mtp", "num_speculative_tokens": 4},
        )
        baseline = prefix_engine_config(source, tp4=True)
        smaller = prefix_engine_config(source, tp4=True, prefill_budget=1024)
        self.assertEqual(
            {key for key in baseline if baseline[key] != smaller[key]},
            {"max_num_batched_tokens", "max_num_scheduled_tokens"},
        )
        self.assertEqual(source["max_num_batched_tokens"], 2048)
        for budget in (0, 944, 4096):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                prefix_engine_config(source, prefill_budget=budget)

    def test_divergent_prompts_keep_common_prefix_length_and_private_tails(self):
        base = list(range(12))
        prompts, shared = divergent_prefix_prompts(base, [[20], [30, 31]], 8)
        self.assertEqual(shared, 10)
        self.assertEqual(prompts, [list(range(11)) + [20], list(range(10)) + [30, 31]])
        self.assertEqual(base, list(range(12)))
        prompts[0][0] = 99
        self.assertEqual(prompts[1][0], 0)
        for tails in ([[20], [20]], [[], [20]], [[20] * 5, [30]], [[20]]):
            with self.subTest(tails=tails), self.assertRaises(ValueError):
                divergent_prefix_prompts(base, tails, 8)

    def test_qsa_tie_control_pads_when_context_is_shorter_than_selection_budget(self):
        import torch

        logits = torch.tensor([[1.0, 3.0], [2.0, 1.0]])
        result = stable_qsa_topk(logits, torch.tensor([2, 0]), 4)
        self.assertEqual(result.tolist(), [[1, 0, -1, -1], [-1, -1, -1, -1]])

    def test_qsa_tie_control_masks_invisible_scores_and_restores_native_hook(self):
        """Native tie membership must not affect this diagnostic's selections."""
        from unittest.mock import patch

        import torch

        logits = torch.tensor(
            [[1.0, 3.0, 3.0, 2.0, 999.0], [2.0, 1.0, 999.0, 999.0, 999.0]]
        )
        original = logits.clone()
        visible = torch.tensor([4, 2])
        selected = torch.empty((2, 3), dtype=torch.int32)
        calls = []

        def native(*args):
            args[4].fill_(len(calls))
            calls.append(1)

        module = NS(_topk=native)
        worker = PrefillTraceWorkerExtension()
        with patch("importlib.import_module", return_value=module):
            self.assertTrue(worker.install_stable_qsa_selection(True)["stable_ties"])
            for _ in range(2):
                module._topk(logits, visible, 12, 4, selected, None)
                self.assertEqual(selected.tolist(), [[1, 2, 3], [0, 1, -1]])
                self.assertTrue(torch.equal(logits, original))
            self.assertEqual(worker.remove_stable_qsa_selection()["calls"], 2)
            self.assertIs(module._topk, native)

    def test_qsa_canonical_order_preserves_set_padding_and_restores_native_hook(self):
        from unittest.mock import patch

        import torch

        original = torch.tensor([[5, 1, 3, -1], [-1, 7, -1, 2]], dtype=torch.int32)
        selected = original.clone()

        def native(*args):
            args[4].copy_(original)

        module = NS(_topk=native)
        worker = PrefillTraceWorkerExtension()
        with patch("importlib.import_module", return_value=module):
            worker.install_stable_qsa_selection()
            with self.assertRaisesRegex(ValueError, "already installed"):
                worker.install_stable_qsa_selection()
            module._topk(None, None, 8, 2, selected, None)
            self.assertEqual(selected.tolist(), [[1, 3, 5, -1], [2, 7, -1, -1]])
            self.assertEqual(worker.remove_stable_qsa_selection()["calls"], 1)
            self.assertIs(module._topk, native)
        self.assertTrue(torch.equal(canonical_qsa_selection(original), selected))
        self.assertEqual(original.tolist(), [[5, 1, 3, -1], [-1, 7, -1, 2]])

    def test_prompt_chunk_trace_compares_later_chunks_and_excludes_decode(self):
        """A matching first chunk must not hide drift in the next prefill chunk."""
        import torch

        class Qwen4ExpDecoderLayer(torch.nn.Module):
            def forward(self, *, hidden_states, positions):
                return hidden_states + self.bias, None, None

        model = torch.nn.Module()
        model.layer = Qwen4ExpDecoderLayer()
        worker = PrefillTraceWorkerExtension()
        worker.get_model = lambda: model
        worker.install_prompt_chunk_trace(8, 2)
        for run in range(2):
            worker.begin_prompt_chunk_trace(run)
            for start in (0, 4, 8):
                model.layer.bias = int(run == 1 and start == 4)
                model.layer(
                    hidden_states=torch.ones(4, 2),
                    positions=torch.arange(start, start + 4),
                )
        trace = worker.collect_session_prefill_trace()
        rows = trace["layers"]["layer"]
        self.assertEqual(
            [(r["run"], r["position_start"]) for r in rows],
            [(0, 0), (0, 4), (1, 0), (1, 4)],
        )
        self.assertTrue(rows[2]["outputs"][0]["exact"])
        self.assertTrue(rows[3]["inputs"][0]["exact"])
        self.assertFalse(rows[3]["outputs"][0]["exact"])
        self.assertEqual(trace["baseline_bytes"], 0)
        self.assertFalse(model.layer._forward_hooks)
        self.assertFalse(model.layer._forward_pre_hooks)

    def test_prompt_chunk_trace_does_not_match_different_schedules(self):
        import torch

        class Qwen4ExpDecoderLayer(torch.nn.Module):
            def forward(self, *, hidden_states, positions):
                return hidden_states, None, None

        model = torch.nn.Module()
        model.layer = Qwen4ExpDecoderLayer()
        worker = PrefillTraceWorkerExtension()
        worker.get_model = lambda: model
        worker.install_prompt_chunk_trace(8, 2)
        for run, start in enumerate((0, 4)):
            worker.begin_prompt_chunk_trace(run)
            model.layer(
                hidden_states=torch.ones(4, 2), positions=torch.arange(start, start + 4)
            )
        with self.assertRaisesRegex(ValueError, "advance exactly once"):
            worker.begin_prompt_chunk_trace(1)
        with self.assertRaisesRegex(ValueError, "Repeated prompt positions"):
            model.layer(hidden_states=torch.ones(4, 2), positions=torch.arange(4, 8))
        trace = worker.collect_session_prefill_trace()
        row = trace["layers"]["layer"][1]["outputs"][0]
        self.assertFalse(row["matched_baseline"])
        self.assertIsNone(row["exact"])

    def test_tp4_screen_removes_pp_placement_and_preserves_ram_kv(self):
        env = tp4_environment(
            [
                "VLLM_PP_LAYER_PARTITION=26,22",
                "VLLM_FLASH_PP_BALANCE=1",
                "VLLM_QSA_KV_OFFLOAD=1",
                "VLLM_FLASH_TP4_MARLIN_K32=0",
            ]
        )
        self.assertIn("VLLM_QSA_KV_OFFLOAD=1", env)
        self.assertIn("VLLM_FLASH_TP4_MARLIN_K32=1", env)
        self.assertFalse(
            any(row.startswith(("VLLM_PP_", "VLLM_FLASH_PP_")) for row in env)
        )
        source = {"tensor_parallel_size": 2, "pipeline_parallel_size": 2}
        selected = prefix_engine_config(source, tp4=True)
        self.assertEqual(selected["tensor_parallel_size"], 4)
        self.assertEqual(selected["pipeline_parallel_size"], 1)
        self.assertEqual(selected["speculative_config"]["num_speculative_tokens"], 4)
        self.assertIsNone(
            prefix_engine_config(source, no_mtp=True, tp4=True)["speculative_config"]
        )
        self.assertEqual(
            source, {"tensor_parallel_size": 2, "pipeline_parallel_size": 2}
        )

    def test_staging_control_does_not_inherit_experimental_choice(self):
        source = ["VLLM_QSA_KV_OFFLOAD=1", "VLLM_QSA_STAGE_ALL_PREFILL=1"]
        self.assertEqual(
            prefill_staging_environment(source, False),
            ["VLLM_QSA_KV_OFFLOAD=1", "VLLM_QSA_STAGE_ALL_PREFILL=0"],
        )
        self.assertEqual(prefill_staging_environment(source, True), source)
        self.assertEqual(source[-1], "VLLM_QSA_STAGE_ALL_PREFILL=1")

    def test_private_cold_buffers_select_native_path_without_changing_host_pool(self):
        source = {"speculative_config": {"num_speculative_tokens": 2}}
        default = prefix_engine_config(source)
        selected = prefix_engine_config(source, private_cold_buffers=True)
        extra = selected["kv_transfer_config"]["kv_connector_extra_config"]
        self.assertIs(extra.pop("use_shared_memory"), False)
        self.assertEqual(selected, default)
        self.assertEqual(source, {"speculative_config": {"num_speculative_tokens": 2}})

    def test_registration_trace_rejects_prior_errors_without_calling_registration(self):
        from unittest.mock import Mock

        register = Mock()
        records = []
        region = NS(rank=2, total_size_bytes=8192)
        with self.assertRaisesRegex(RuntimeError, "Pre-existing CUDA error 1"):
            checked_host_registration(register, region, lambda: 1, records.append)
        register.assert_not_called()
        self.assertEqual(records[0]["phase"], "before")
        error = RuntimeError("registration failed")
        register.side_effect = error
        peek = Mock(side_effect=[0, 2])
        with self.assertRaises(RuntimeError) as captured:
            checked_host_registration(register, region, peek, records.append)
        self.assertIs(captured.exception, error)
        self.assertEqual(records[-1], dict(rank=2, phase="failed", cuda_error=2))

    def test_stable_expert_layout_preserves_groups_padding_and_inputs(self):
        """Changing atomic insertion order must not change canonical routes."""
        import torch

        ids = torch.tensor([4, 0, 9, 9, 5, 3, 9, 9, 8, 1, 9, 9, 777], dtype=torch.int32)
        reordered = torch.tensor(
            [0, 4, 9, 9, 1, 8, 9, 9, 3, 5, 9, 9, -999], dtype=torch.int32
        )
        experts = torch.tensor([0, 2, 2, -9], dtype=torch.int32)
        total = torch.tensor([12], dtype=torch.int32)
        expected = torch.tensor(
            [0, 4, 9, 9, 1, 3, 5, 8, 9, 9, 9, 9, 9], dtype=torch.int32
        )
        before = ids.clone()
        for source in (ids, reordered):
            result = canonical_expert_layout((source, experts, total), 4, 9, 3)
            self.assertTrue(torch.equal(result[0], expected))
            self.assertIs(result[1], experts)
            self.assertIs(result[2], total)
        self.assertTrue(torch.equal(ids, before))

    def test_stable_expert_hook_covers_native_wrapper_and_restores_both(self):
        from unittest.mock import patch

        import torch

        def native(*args, **kwargs):
            return (
                torch.tensor([1, 0, 2, 2], dtype=torch.int32),
                torch.tensor([0, 0], dtype=torch.int32),
                torch.tensor([2], dtype=torch.int32),
            )

        wrapper = lambda *args, **kwargs: native(*args, **kwargs)
        marlin = NS(
            moe_align_block_size=native,
            flash_moe_layout_sm86=NS(_ENABLED=False, moe_align_block_size=wrapper),
        )
        worker = PrefillTraceWorkerExtension()
        with patch("importlib.import_module", return_value=marlin):
            worker.install_stable_expert_layout()
            for callback in (
                marlin.moe_align_block_size,
                marlin.flash_moe_layout_sm86.moe_align_block_size,
            ):
                result = callback(torch.tensor([[0, 0]]), 2, 1)
                self.assertEqual(result[0].tolist(), [0, 1, 2, 2])
            with self.assertRaisesRegex(ValueError, "does not support EP"):
                marlin.moe_align_block_size(torch.tensor([[0]]), 2, 1, object())
            self.assertEqual(worker.remove_stable_expert_layout()["calls"], 2)
        self.assertIs(marlin.moe_align_block_size, native)
        self.assertIs(marlin.flash_moe_layout_sm86.moe_align_block_size, wrapper)

    def test_nonfinite_prefix_logprobs_keep_json_safe_failure_evidence(self):
        import json

        rows, invalid = finite_logprobs(
            [
                {
                    1: NS(logprob=-0.5),
                    2: NS(logprob=float("nan")),
                    3: NS(logprob=float("-inf")),
                }
            ]
        )
        self.assertEqual(rows, [{"1": -0.5, "2": None, "3": None}])
        self.assertEqual([row["value"] for row in invalid], ["nan", "-inf"])
        json.dumps(dict(rows=rows, invalid=invalid), allow_nan=False)

    def test_native_prefix_screen_uses_native_cache_and_keeps_source_unchanged(self):
        source = {
            "enable_prefix_caching": False,
            "speculative_config": {"method": "mtp", "num_speculative_tokens": 2},
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 2,
        }
        config = prefix_engine_config(source)
        self.assertFalse(source["enable_prefix_caching"])
        self.assertTrue(config["enable_prefix_caching"])
        self.assertTrue(config["enforce_eager"])
        self.assertEqual(config["mamba_cache_mode"], "align")
        self.assertEqual(
            config["kv_transfer_config"]["kv_connector"], "OffloadingConnector"
        )
        self.assertEqual(config["speculative_config"], source["speculative_config"])
        self.assertIsNone(prefix_engine_config(source, True)["speculative_config"])
        self.assertEqual(config["max_model_len"], 8192)

    def test_prefix_verdict_requires_real_cold_load_and_exact_controls(self):
        report = dict(
            baseline_repeatable=True,
            hot_exact=True,
            cold_exact=True,
            hot_cached_tokens=16,
            cold_cached_tokens=16,
            cold_load_bytes=1024,
            store_bytes=1024,
        )
        self.assertTrue(prefix_verdict(report))
        self.assertFalse(prefix_verdict({**report, "performance_repeats": 3}))
        self.assertTrue(
            prefix_verdict(
                {
                    **report,
                    "performance_repeats": 3,
                    "performance_control": {"passed": True},
                }
            )
        )
        self.assertFalse(prefix_verdict({**report, "paired_cold_loads": True}))
        self.assertFalse(prefix_verdict({**report, "idle_expiry": True}))
        self.assertFalse(prefix_verdict({**report, "pending_expiry": True}))
        self.assertFalse(prefix_verdict({**report, "continuations": True}))
        self.assertTrue(
            prefix_verdict(
                {
                    **report,
                    "continuations": True,
                    "continuation_control": {"passed": True},
                }
            )
        )
        self.assertTrue(
            prefix_verdict(
                {
                    **report,
                    "pending_expiry": True,
                    "pending_expiry_control": {"passed": True},
                }
            )
        )
        self.assertFalse(
            prefix_verdict(
                {**report, "idle_expiry": True, "expiry_control": {"passed": False}}
            )
        )
        self.assertTrue(
            prefix_verdict(
                {**report, "idle_expiry": True, "expiry_control": {"passed": True}}
            )
        )
        self.assertTrue(
            prefix_verdict(
                {**report, "paired_cold_loads": True, "load_barrier_exercised": True}
            )
        )
        self.assertFalse(prefix_verdict({**report, "concurrent": {"passed": False}}))
        self.assertTrue(prefix_verdict({**report, "concurrent": {"passed": True}}))
        for key in report:
            self.assertFalse(prefix_verdict({**report, key: 0}), key)
        self.assertFalse(prefix_verdict({}))

    def test_native_kernel_control_preserves_quantization_and_topology(self):
        preserved = [
            "VLLM_FLASH_MTP_INT8_EXPERTS=1",
            "VLLM_FLASH_PP_BALANCE=26,22",
            "VLLM_QSA_HOST_KV=1",
        ]
        flags = KERNEL_FLAGS + EXPERIMENTAL_KERNEL_FLAGS
        source = preserved + [f"{name}=1" for name in flags]
        self.assertEqual(kernel_environment(source, False), source)
        native = kernel_environment(source, True)
        self.assertEqual(native[: len(preserved)], preserved)
        self.assertEqual(
            set(native[len(preserved) :]), {f"{n}=0" for n in flags}
        )
        self.assertEqual(len(native), len(source))
        self.assertNotIn("VLLM_FLASH_GDN_TP4_PROJECTION", KERNEL_FLAGS)

    def test_mixed_canonical_control_disables_only_expert_layout(self):
        source = ["VLLM_FLASH_MTP_INT8_EXPERTS=1", "VLLM_QSA_HOST_KV=1"] + [
            f"{name}=1" for name in KERNEL_FLAGS
        ]
        before = list(source)
        result = kernel_environment(source, False, optimized_canonical=True)
        values = dict(row.split("=", 1) for row in result)
        self.assertEqual(source, before)
        self.assertEqual(len(result), len(source))
        self.assertEqual(values["VLLM_FLASH_MOE_LAYOUT_SM86"], "0")
        self.assertTrue(
            all(
                value == "1"
                for key, value in values.items()
                if key != "VLLM_FLASH_MOE_LAYOUT_SM86"
            )
        )
        with self.assertRaisesRegex(ValueError, "disable all"):
            kernel_environment(source, True, optimized_canonical=True)
        for mode in ("native", "stable"):
            self.assertEqual(kernel_environment(source, False,
                                               expert_order_control=mode), result)
            for native, canonical in ((True, False), (False, True)):
                with self.assertRaisesRegex(ValueError, "cannot combine"):
                    kernel_environment(source, native, canonical, mode)

    def test_prefill_trace_localizes_changed_outputs_and_removes_hooks(self):
        import torch

        class Qwen4ExpDecoderLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = 0

            def forward(self, *, hidden_states, positions):
                return hidden_states + self.bias, hidden_states * 2, None

        model = torch.nn.Module()
        model.layer = Qwen4ExpDecoderLayer()
        worker = PrefillTraceWorkerExtension()
        worker.get_model = lambda: model
        worker.install_session_prefill_trace(3, 2)
        model.layer(hidden_states=torch.zeros(1, 4), positions=torch.tensor([4]))
        model.layer(hidden_states=torch.zeros(3, 4), positions=torch.arange(3) + 4)
        for bias in (0, 1, 2):
            model.layer.bias = bias
            positions = torch.arange(3)
            if bias == 1:
                positions = torch.tensor([0, 1, 2, 0]).expand(3, -1)
            hidden = torch.ones(positions.shape[-1], 4)
            if bias == 1:
                hidden[-1].fill_(999)  # Padded rows must not affect the comparison.
            model.layer(hidden_states=hidden, positions=positions)
        trace = worker.collect_session_prefill_trace()
        records = trace["layers"]["layer"]
        self.assertEqual(len(records), 2)
        self.assertTrue(records[1]["inputs"][0]["exact"])
        self.assertEqual(records[1]["inputs"][0]["shape"], [3, 4])
        self.assertEqual(records[1]["inputs"][0]["source_shape"], [4, 4])
        self.assertEqual(records[1]["inputs"][0]["source_stride"], [4, 1])
        self.assertFalse(records[1]["outputs"][0]["exact"])
        self.assertEqual(records[1]["outputs"][0]["max_abs_difference"], 1)
        self.assertTrue(records[1]["outputs"][1]["exact"])
        self.assertFalse(hasattr(model, "_flash_session_prefill_trace"))
        self.assertFalse(model.layer._forward_hooks)
        self.assertFalse(model.layer._forward_pre_hooks)

    def test_prefix_trace_detects_drift_outside_retained_sample(self):
        import torch

        class Qwen4ExpDecoderLayer(torch.nn.Module):
            def forward(self, *, hidden_states, positions):
                result = hidden_states.clone()
                result[-1] += self.bias
                return result, None, None

        model = torch.nn.Module()
        model.layer = Qwen4ExpDecoderLayer()
        worker = PrefillTraceWorkerExtension()
        worker.get_model = lambda: model
        worker.install_prefix_prefill_trace(2)
        for bias in (0, 1):
            model.layer.bias = bias
            model.layer(hidden_states=torch.ones(2048, 4), positions=torch.arange(2048))
        trace = worker.collect_session_prefill_trace()
        record = trace["layers"]["layer"][1]
        self.assertTrue(record["inputs"][0]["exact"])
        result = record["outputs"][0]
        self.assertFalse(result["exact"])
        self.assertTrue(result["sample_exact"])
        self.assertEqual(result["max_abs_difference"], 0)
        self.assertEqual(result["shape"], [32, 4])
        self.assertEqual(result["hashed_shape"], [2048, 4])
        self.assertLess(trace["baseline_bytes"], 4096)
        self.assertFalse(model.layer._forward_hooks)

    def test_mlp_trace_distinguishes_input_drift_from_component_drift(self):
        import torch

        class Qwen4ExpDecoderLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))

            def forward(self, *, hidden_states, positions):
                return hidden_states, self.mlp(hidden_states), None

        model = torch.nn.Module()
        model.layers = torch.nn.ModuleDict({"1": Qwen4ExpDecoderLayer()})
        worker = PrefillTraceWorkerExtension()
        worker.get_model = lambda: model
        installed = worker.install_session_prefill_trace(3, 2)
        layer = model.layers["1"]
        for weight in (1, 2):
            with torch.no_grad():
                layer.mlp[0].weight.fill_(weight)
            layer(hidden_states=torch.ones(3, 4), positions=torch.arange(3))
            # Decode and unrelated calls must not become a prefill reference.
            layer(hidden_states=torch.ones(1, 4), positions=torch.tensor([4]))
        trace = worker.collect_session_prefill_trace()
        self.assertEqual(set(installed["components"]), set(trace["components"]))
        for rows in trace["components"].values():
            self.assertEqual(len(rows), 2)
            self.assertTrue(rows[1]["inputs"][0]["exact"])
            self.assertFalse(rows[1]["outputs"][0]["exact"])
            self.assertEqual(rows[1]["outputs"][0]["max_abs_difference"], 4)
        for module in layer.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)

    def test_method_trace_separates_routing_and_reduce_and_restores_overrides(self):
        import torch

        class Experts(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.router = NS(select_experts=lambda x: (x, x.long()))
                self.routed_experts = NS(forward_modular=lambda x: x * 2)

            def _maybe_reduce_final_output(self, x):
                return x + 1

            def forward(self, x):
                weights, _ = self.router.select_experts(x)
                routed = self.routed_experts.forward_modular(weights)
                return self._maybe_reduce_final_output(routed)

        class Qwen4ExpDecoderLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = torch.nn.Sequential()
                self.mlp.add_module("experts", Experts())

            def forward(self, *, hidden_states, positions):
                return hidden_states, self.mlp(hidden_states), None

        model = torch.nn.Module()
        model.layers = torch.nn.ModuleDict({"1": Qwen4ExpDecoderLayer()})
        layer = model.layers["1"]
        experts = layer.mlp.experts
        original = experts.router.select_experts
        worker = PrefillTraceWorkerExtension()
        worker.get_model = lambda: model
        worker.install_session_prefill_trace(3, 2)
        expected = torch.full((3, 4), 3.0)
        for _ in range(2):
            result = layer(hidden_states=torch.ones(3, 4), positions=torch.arange(3))
            self.assertTrue(torch.equal(result[1], expected))
        trace = worker.collect_session_prefill_trace()
        for suffix in ("router", "routed_experts", "final_reduce"):
            rows = trace["components"]["layers.1.mlp." + suffix]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(t["exact"] for t in rows[1]["outputs"]))
        self.assertIs(experts.router.select_experts, original)
        self.assertNotIn("_maybe_reduce_final_output", vars(experts))

    def test_cold_misses_cannot_hide_leaked_hot_state_or_blocks(self):
        scheduler = dict(
            retained_requests=0,
            running=0,
            idle_hot=0,
            waiting=0,
            skipped_waiting=0,
            pending_batches=0,
            status_counts={},
            free_gpu_blocks=460,
            usable_gpu_blocks=460,
        )
        stats = dict(
            cold_sessions=0,
            unresolved_transactions=0,
            policy_error=None,
            scheduler=scheduler,
        )
        lookups = [{"cache": "miss"}]
        self.assertTrue(cleanup_complete(stats, lookups))
        for name in (
            "retained_requests",
            "running",
            "idle_hot",
            "waiting",
            "skipped_waiting",
            "pending_batches",
            "status_counts",
            "free_gpu_blocks",
        ):
            with self.subTest(field=name):
                previous = scheduler[name]
                scheduler[name] = {"WAITING": 1} if name == "status_counts" else 1
                self.assertFalse(cleanup_complete(stats, lookups))
                scheduler[name] = previous

    def test_successful_diagnostic_restore_cannot_waive_a_failed_baseline(self):
        report = dict(
            baseline_repeatable=False, exact_token_match=True, full_reuse_proven=True
        )
        self.assertFalse(screen_verdict(report))
        report["baseline_repeatable"] = True
        self.assertTrue(screen_verdict(report))
        report["full_reuse_proven"] = False
        self.assertFalse(screen_verdict(report))

    def test_exact_token_equality_includes_length_and_first_difference(self):
        self.assertTrue(compare_tokens([1, 2, 3], [1, 2, 3])["equal"])
        self.assertEqual(
            compare_tokens([1, 2, 3], [1, 9, 3]),
            {
                "equal": False,
                "index": 1,
                "expected": 2,
                "actual": 9,
            },
        )
        self.assertFalse(compare_tokens([1, 2], [1])["equal"])
        self.assertFalse(compare_tokens([1], [1, 2])["equal"])

    def test_reuse_evidence_excludes_null_pages_and_preserves_group_identity(self):
        groups = block_sets({"null_block_id": 0, "block_ids": [[0, 1, 2], [0, 3]]})
        self.assertEqual(groups, [{1, 2}, {3}])
        other = [{3}, {1, 2}]
        self.assertFalse(all(old <= reused for old, reused in zip(groups, other)))


class StreamingClientTests(unittest.IsolatedAsyncioTestCase):
    def engine(self, fail=False):
        engine = NS(output_processor=NS(external_req_ids={}))

        async def generate(prompt, sampling_params, request_id):
            engine.output_processor.external_req_ids[request_id] = ["internal"]
            cumulative = []
            try:
                async for chunk in prompt:
                    if fail:
                        raise ValueError("injected generation failure")
                    cumulative.extend(chunk.prompt["prompt_token_ids"])
                    yield NS(
                        outputs=[
                            NS(
                                token_ids=cumulative.copy(),
                                finish_reason="length",
                                logprobs=[{t: NS(logprob=-0.25)} for t in cumulative],
                            )
                        ]
                    )
            finally:
                engine.output_processor.external_req_ids.pop(request_id, None)

        engine.generate = generate
        return engine

    async def test_input_stream_stays_open_across_idle_turns_then_closes(self):
        engine = self.engine()
        session = StreamingSession(engine, None, "a", NS)
        await session.send([1, 2])
        self.assertEqual(await session.wait_turn(1), [1, 2])
        self.assertFalse(session.closed)
        self.assertEqual(session.internal_id(), "internal")
        await session.send([3, 4])
        self.assertEqual(await session.wait_turn(2), [1, 2, 3, 4])
        self.assertEqual(
            session.turn_logprobs,
            [
                [{"1": -0.25}, {"2": -0.25}],
                [{"1": -0.25}, {"2": -0.25}, {"3": -0.25}, {"4": -0.25}],
            ],
        )
        await session.finish()
        self.assertTrue(session.closed)
        self.assertEqual(engine.output_processor.external_req_ids, {})

    async def test_generation_error_unblocks_waiting_for_a_turn(self):
        session = StreamingSession(self.engine(fail=True), None, "a", NS)
        await session.send([1])
        with self.assertRaisesRegex(ValueError, "injected generation failure"):
            await session.wait_turn(1, timeout=1)
        self.assertTrue(session.closed)


if __name__ == "__main__":
    unittest.main()
