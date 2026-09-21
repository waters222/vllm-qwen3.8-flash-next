"""The MTP observer must record speculative batch size without changing scheduling."""

import copy
import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

PATH = Path(__file__).resolve().parents[2] / "vllm/v1/core/sched/flash_pp_phase.py"
SPEC = importlib.util.spec_from_file_location("phase_probe", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MTPPhaseTests(unittest.TestCase):
    def scheduler(self):
        cls = type(
            "AsyncScheduler", (), {"__module__": "vllm.v1.core.sched.async_scheduler"}
        )
        s = cls()
        s.use_v2_model_runner = True
        s.scheduler_config = NS(async_scheduling=True)
        s.parallel_config = NS(pipeline_parallel_size=2, data_parallel_size=1)
        s.num_spec_tokens = 2
        s.vllm_config = NS(
            speculative_config=NS(method="mtp", num_speculative_tokens=2),
            model_config=NS(hf_config=NS(model_type="qwen4_exp")),
        )
        s.current_step = 7
        s.running = [
            NS(
                request_id="private-id",
                is_prefill_chunk=False,
                num_computed_tokens=50,
                num_prompt_tokens=31,
                next_decode_eligible_step=7,
                spec_token_ids=[999, 888],
            )
        ]
        s.waiting = s.skipped_waiting = []
        return s

    def test_mtp_requires_explicit_opt_in(self):
        s = self.scheduler()
        with self.assertRaises(ValueError):
            MODULE.PhaseCapture.from_env(s, None, {"VLLM_FLASH_PP_PHASE_STEPS": "1"})
        s.vllm_config.speculative_config.method = "eagle"
        with self.assertRaises(ValueError):
            MODULE.PhaseCapture.from_env(
                s,
                None,
                {
                    "VLLM_FLASH_PP_PHASE_STEPS": "1",
                    "VLLM_FLASH_PP_PHASE_ALLOW_MTP": "1",
                },
            )

    def test_observe_three_token_verification_without_mutation_or_payload(self):
        s = self.scheduler()
        emitted = []
        observer = MODULE.PhaseCapture.from_env(
            s,
            emitted.append,
            {"VLLM_FLASH_PP_PHASE_STEPS": "1", "VLLM_FLASH_PP_PHASE_ALLOW_MTP": "1"},
        )
        out = NS(
            num_scheduled_tokens={"private-id": 3},
            total_num_scheduled_tokens=3,
            scheduled_new_reqs=[],
        )
        before = copy.deepcopy((vars(s), vars(out)))
        observer.before(s, out)
        observer.after(s, out)
        self.assertEqual(before, (vars(s), vars(out)))
        record = json.loads(emitted[0])["records"][0]
        self.assertEqual(record["draft_depth"], 2)
        self.assertEqual(record["requests"][0]["scheduled_tokens"], 3)
        for private in ("private-id", "999", "888", "spec_token_ids"):
            self.assertNotIn(private, emitted[0])


if __name__ == "__main__":
    unittest.main()
