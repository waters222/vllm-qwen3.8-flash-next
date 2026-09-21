#!/usr/bin/env python3
"""CPU helper tests only; not a whole scheduler/worker-ring regression."""

import copy
import unittest
from types import SimpleNamespace as NS

from flash_pp_balance import PhaseBalance


def scheduler(step=3081):
    kind = type(
        "AsyncScheduler", (), {"__module__": "vllm.v1.core.sched.async_scheduler"}
    )
    s = kind()
    s.use_v2_model_runner = True
    s.scheduler_config = NS(async_scheduling=True)
    s.parallel_config = NS(pipeline_parallel_size=2, data_parallel_size=1)
    s.vllm_config = NS(
        model_config=NS(hf_config=NS(model_type="qwen4_exp")), speculative_config=None
    )
    s.num_spec_tokens = 0
    s.num_sampled_tokens_per_step = 1
    s.lora_config = None
    s.cache_config = NS(enable_prefix_caching=False)
    s.connector = None
    s.ec_connector = None
    s.is_encoder_decoder = False
    s.is_mm_encoder_only = False
    s.max_num_encoder_input_tokens = 0
    s.current_step = step
    s.num_waiting_for_streaming_input = 0
    s.requests = {}
    s.running = []
    return s


def fresh(key="new", length=31):
    return NS(
        request_id=key,
        status=NS(name="WAITING"),
        num_preemptions=0,
        num_stale_output_tokens=0,
        resumable=False,
        use_structured_output=False,
        has_encoder_inputs=False,
        mm_features=[],
        lora_request=None,
        prompt_token_ids=[1] * length,
        spec_token_ids=[],
        num_computed_tokens=0,
        num_output_tokens=0,
        num_output_placeholders=0,
        num_prompt_tokens=length,
        num_tokens=length,
        is_prefill_chunk=False,
        next_decode_eligible_step=0,
    )


def established(key="old", eligible=3081):
    r = fresh(key)
    r.status.name = "RUNNING"
    r.num_computed_tokens = 40
    r.num_tokens = 40
    r.num_output_tokens = 9
    r.next_decode_eligible_step = eligible
    return r


class BalanceTests(unittest.TestCase):
    def mtp_scheduler(self, step=1):
        s = scheduler(step)
        s.num_spec_tokens = 1
        s.vllm_config.speculative_config = NS(method="mtp", num_speculative_tokens=1)
        s.parallel_config.tensor_parallel_size = 2
        s.parallel_config.enable_expert_parallel = False
        s.max_num_running_reqs = 4
        return s

    def mtp_helper(self, s):
        return PhaseBalance.from_env(
            s,
            {
                "VLLM_FLASH_PP_BALANCE": "1",
                "VLLM_FLASH_PP_MTP_PAIRS": "1",
            },
        )

    def test_mtp_pairs_require_explicit_depth_and_topology_gate(self):
        s = self.mtp_scheduler()
        with self.assertRaises(ValueError):
            PhaseBalance.from_env(s, {"VLLM_FLASH_PP_BALANCE": "1"})
        self.assertTrue(self.mtp_helper(s).mtp_pairs)
        for field, value in [("num_spec_tokens", 2), ("max_num_running_reqs", 5)]:
            changed = self.mtp_scheduler()
            setattr(changed, field, value)
            with self.assertRaises(ValueError):
                self.mtp_helper(changed)
        for field, value in [
            ("tensor_parallel_size", 4),
            ("enable_expert_parallel", True),
        ]:
            changed = self.mtp_scheduler()
            setattr(changed.parallel_config, field, value)
            with self.assertRaises(ValueError):
                self.mtp_helper(changed)

    def test_mtp_cold_admission_forms_two_pairs_without_delaying_c1(self):
        s = self.mtp_scheduler()
        h = self.mtp_helper(s)
        a, b, c, d = [fresh(key) for key in "abcd"]
        selected = {}
        for r in (a, b):
            self.assertFalse(self.call(h, s, r, selected))
            r.status.name = "RUNNING"
            s.running.append(r)
            selected[r.request_id] = r.num_prompt_tokens
        self.assertTrue(self.call(h, s, c, selected))
        for r in (a, b):
            r.num_computed_tokens = 31
            r.num_output_placeholders = 2
            r.spec_token_ids = [-1]
            r.next_decode_eligible_step = 3
        s.current_step = 2
        selected = {}
        for r in (c, d):
            self.assertFalse(self.call(h, s, r, selected))
            h.clear(r)
            r.status.name = "RUNNING"
            s.running.append(r)
            selected[r.request_id] = r.num_prompt_tokens
        self.assertEqual([r.next_decode_eligible_step for r in (a, b)], [3, 3])
        self.assertEqual(h._deferred, {})

    def test_mtp_staggered_singleton_join_defers_once_and_keeps_decode_cadence(self):
        s = self.mtp_scheduler(step=4)
        h = self.mtp_helper(s)
        old = established(eligible=5)
        old.spec_token_ids = [-1]
        old.num_output_placeholders = 2
        s.running = [old]
        incoming = fresh()
        self.assertTrue(self.call(h, s, incoming))
        self.assertEqual(old.next_decode_eligible_step, 5)
        self.assertFalse(self.call(h, s, incoming))
        s.current_step = 5
        self.assertFalse(self.call(h, s, incoming, {"old": 2}))
        h.clear(incoming)
        self.assertEqual(h._deferred, {})

    def test_mtp_pair_pilot_bypasses_long_prefill_and_ambiguous_running_state(self):
        for mutation in (
            {"is_prefill_chunk": True},
            {"spec_token_ids": [-1, -1]},
            {"num_output_placeholders": 3},
            {"num_preemptions": 1},
        ):
            s = self.mtp_scheduler(step=4)
            old = established(eligible=5)
            old.spec_token_ids = [-1]
            vars(old).update(mutation)
            s.running = [old]
            self.assertFalse(self.call(self.mtp_helper(s), s, fresh()))
        s = self.mtp_scheduler(step=4)
        old = established(eligible=5)
        old.spec_token_ids = [-1]
        s.running = [old]
        self.assertFalse(self.call(self.mtp_helper(s), s, fresh(length=513)))

    def call(self, h, s, r, selected=None, **kw):
        s.requests[r.request_id] = r
        for old in s.running:
            s.requests[old.request_id] = old
        args = dict(
            num_new_tokens=r.num_prompt_tokens,
            num_computed_tokens=0,
            num_scheduled_tokens=selected or {},
            load_kv_async=False,
            num_external_computed_tokens=0,
        )
        args.update(kw)
        before = copy.deepcopy(vars(s))
        request_before = copy.deepcopy(vars(r))
        result = h.should_defer(s, r, **args)
        self.assertEqual(vars(s), before)
        self.assertEqual(vars(r), request_before)
        return result

    def test_off_trap_and_invalid_flag(self):
        class Trap:
            def __getattribute__(self, name):
                raise AssertionError(name)

        self.assertIsNone(PhaseBalance.from_env(Trap(), {}))
        for flag in ("", "true", "2", 1):
            with self.assertRaises(ValueError):
                PhaseBalance.from_env(Trap(), {"VLLM_FLASH_PP_BALANCE": flag})

    def test_guard_rejections(self):
        valid = scheduler()
        self.assertIsInstance(
            PhaseBalance.from_env(valid, {"VLLM_FLASH_PP_BALANCE": "1"}), PhaseBalance
        )
        paths = [
            "use_v2_model_runner",
            "scheduler_config.async_scheduling",
            "parallel_config.pipeline_parallel_size",
            "parallel_config.data_parallel_size",
            "num_spec_tokens",
            "lora_config",
            "cache_config.enable_prefix_caching",
            "connector",
            "ec_connector",
            "is_encoder_decoder",
            "is_mm_encoder_only",
            "max_num_encoder_input_tokens",
            "vllm_config.speculative_config",
            "num_sampled_tokens_per_step",
        ]
        for path in paths:
            s = scheduler()
            obj = s
            parts = path.split(".")
            for part in parts[:-1]:
                obj = getattr(obj, part)
            old = getattr(obj, parts[-1])
            setattr(
                obj,
                parts[-1],
                not old if type(old) is bool else (7 if old is not None else object()),
            )
            with self.subTest(path=path), self.assertRaises(ValueError):
                PhaseBalance.from_env(s, {"VLLM_FLASH_PP_BALANCE": "1"})
        for attr, value in [("__name__", "CustomScheduler"), ("__module__", "other")]:
            s = scheduler()
            setattr(type(s), attr, value)
            with self.assertRaises(ValueError):
                PhaseBalance.from_env(s, {"VLLM_FLASH_PP_BALANCE": "1"})
        s = scheduler()
        s.vllm_config.model_config.hf_config.model_type = "other"
        with self.assertRaises(ValueError):
            PhaseBalance.from_env(s, {"VLLM_FLASH_PP_BALANCE": "1"})

    def test_observed_3081_and_next_step(self):
        s = scheduler()
        h = PhaseBalance()
        r = fresh()
        old = established()
        s.running = [old]
        self.assertTrue(self.call(h, s, r, {"old": 1}))
        self.assertEqual(
            old.next_decode_eligible_step, 3081
        )  # Helper never updates it.
        old.next_decode_eligible_step = 3083
        s.current_step = 3082
        self.assertFalse(self.call(h, s, r))
        h.clear(r)
        self.assertEqual(h._deferred, {})

    def test_cold_c1_c2_c3_c4_balancing(self):
        s = scheduler(step=1)
        h = PhaseBalance()
        selected = {}
        a, b, c, d = [fresh(k) for k in "abcd"]
        self.assertFalse(self.call(h, s, a, selected))
        a.status.name = "RUNNING"
        s.running.append(a)
        selected["a"] = 31
        self.assertTrue(self.call(h, s, b, selected))
        # Apply expected async updates to first admission in the test fixture only.
        a.num_computed_tokens = 31
        a.next_decode_eligible_step = 3
        s.current_step = 2
        selected = {}
        self.assertFalse(self.call(h, s, b, selected))
        h.clear(b)
        b.status.name = "RUNNING"
        s.running.append(b)
        selected["b"] = 31
        self.assertFalse(self.call(h, s, c, selected))  # Equal 1/1 occupancy admits C3.
        c.status.name = "RUNNING"
        s.running.append(c)
        selected["c"] = 31
        self.assertTrue(
            self.call(h, s, d, selected)
        )  # Heavier 2/1 current phase defers C4.
        for r in (b, c):
            r.num_computed_tokens = 31
            r.next_decode_eligible_step = 4
        s.current_step = 3
        self.assertFalse(self.call(h, s, d, {"a": 1}))

    def test_one_deferral_cleanup_cancel_and_identity_reuse(self):
        h = PhaseBalance()
        s = scheduler()
        r = fresh()
        s.running = [established()]
        self.assertTrue(self.call(h, s, r, {"old": 1}))
        self.assertFalse(self.call(h, s, r, {"old": 1}))
        replacement = fresh(r.request_id)
        self.assertTrue(self.call(h, s, replacement, {"old": 1}))
        h.clear(r)
        self.assertIn(replacement.request_id, h._deferred)
        del s.requests[replacement.request_id]
        other = fresh("another")
        self.call(h, s, other, {"old": 1})
        self.assertNotIn(replacement.request_id, h._deferred)
        self.assertTrue(
            all(type(v[0]) is int and type(v[1]) is int for v in h._deferred.values())
        )

    def test_request_exclusions_and_clipped_prompt(self):
        mutations = {
            "num_preemptions": 1,
            "num_stale_output_tokens": 1,
            "resumable": True,
            "use_structured_output": True,
            "has_encoder_inputs": True,
            "mm_features": [1],
            "lora_request": object(),
            "prompt_token_ids": None,
            "spec_token_ids": [1],
            "prompt_embeds": object(),
            "inputs_embeds": object(),
            "num_computed_tokens": 1,
            "num_output_tokens": 1,
            "num_output_placeholders": 1,
            "next_decode_eligible_step": 3081,
            "session_id": "prior-session",
        }
        for name, value in mutations.items():
            h = PhaseBalance()
            s = scheduler()
            s.running = [established()]
            r = fresh()
            setattr(r, name, value)
            # opaque objects are not equal after deepcopy; use stable scalar stand-ins.
            if type(value) is object:
                setattr(r, name, "present")
            with self.subTest(name=name):
                self.assertFalse(self.call(h, s, r, {"old": 1}))
        for kw in (
            {"num_new_tokens": 30},
            {"num_computed_tokens": 1},
            {"load_kv_async": True},
            {"num_external_computed_tokens": 1},
        ):
            s = scheduler()
            s.running = [established()]
            self.assertFalse(self.call(PhaseBalance(), s, fresh(), {"old": 1}, **kw))
        s = scheduler()
        s.running = [established()]
        self.assertFalse(self.call(PhaseBalance(), s, fresh(length=513), {"old": 1}))

    def test_one_token_fresh_prompt_and_paused_streaming(self):
        s = scheduler()
        s.running = [established()]
        self.assertTrue(self.call(PhaseBalance(), s, fresh(length=1), {"old": 1}))
        s.num_waiting_for_streaming_input = 1
        self.assertFalse(self.call(PhaseBalance(), s, fresh(), {"old": 1}))

    def test_completing_running_prefill_conservatively_bypassed(self):
        s = scheduler()
        r = fresh("chunked")
        r.status.name = "RUNNING"
        r.num_computed_tokens = 30
        r.is_prefill_chunk = True
        s.running = [r]
        self.assertTrue(PhaseBalance._completes(r, 30, 1))
        self.assertFalse(self.call(PhaseBalance(), s, fresh(), {"chunked": 1}))

    def test_ambiguous_running_and_exact_completion_predicate(self):
        for change, selected in [
            (dict(is_prefill_chunk=True), {"old": 1}),
            (dict(num_computed_tokens=2), {"old": 1}),
            (dict(next_decode_eligible_step=3081), {}),
            (dict(next_decode_eligible_step=3084), {}),
            (dict(num_output_placeholders=2), {"old": 1}),
            (dict(num_tokens=42), {"old": 1}),
            ({}, {"old": 2}),
        ]:
            s = scheduler()
            old = established()
            vars(old).update(change)
            s.running = [old]
            self.assertFalse(self.call(PhaseBalance(), s, fresh(), selected))
        s = scheduler()
        s.running = [established(eligible=3082)]
        self.assertFalse(
            self.call(PhaseBalance(), s, fresh())
        )  # Other phase occupied: admit.


if __name__ == "__main__":
    unittest.main(verbosity=2)
