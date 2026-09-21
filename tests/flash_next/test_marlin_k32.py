"""CPU checks for the default-off, exact-topology K32 integration gate."""

import ast
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace as NS


class K32GateTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[2] / (
            "vllm/model_executor/layers/fused_moe/experts/flash_marlin_k32_sm86.py"
        )
        tree = ast.parse(path.read_text())
        body = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "validate_config"
        ]
        namespace = {}
        exec(
            compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"),
            namespace,
        )
        self.gate = namespace["validate_config"]
        self.config = NS(
            model_config=NS(
                hf_text_config=NS(
                    model_type="qwen4_exp_text", hidden_size=2560, num_hidden_layers=48
                )
            ),
            parallel_config=NS(
                tensor_parallel_size=4,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                enable_expert_parallel=False,
            ),
            speculative_config=None,
            lora_config=None,
            scheduler_config=NS(max_num_batched_tokens=2048),
        )

    def test_default_off_without_inspecting_model(self):
        self.assertFalse(self.gate(None, {}))

    def test_exact_control_is_accepted_without_mutation(self):
        original = copy.deepcopy(self.config)
        self.assertTrue(self.gate(self.config, {"VLLM_FLASH_TP4_MARLIN_K32": "1"}))
        self.assertEqual(self.config, original)

    def test_other_topologies_speculation_or_precision_context_fail_closed(self):
        variants = [
            ("parallel_config", "tensor_parallel_size", 2),
            ("parallel_config", "pipeline_parallel_size", 2),
            ("parallel_config", "data_parallel_size", 2),
            ("parallel_config", "enable_expert_parallel", True),
            ("scheduler_config", "max_num_batched_tokens", 4096),
            (None, "speculative_config", NS(method="mtp")),
            (None, "lora_config", NS()),
        ]
        for section, field, value in variants:
            config = copy.deepcopy(self.config)
            setattr(getattr(config, section) if section else config, field, value)
            with self.assertRaises(ValueError):
                self.gate(config, {"VLLM_FLASH_TP4_MARLIN_K32": "1"})
        with self.assertRaises(ValueError):
            self.gate(None, {"VLLM_FLASH_TP4_MARLIN_K32": "yes"})
        with self.assertRaises(ValueError):
            self.gate(
                self.config,
                {
                    "VLLM_FLASH_TP4_MARLIN_K32": "1",
                    "VLLM_FLASH_TP4_CHUNKED_REPACK": "1",
                },
            )


if __name__ == "__main__":
    unittest.main()
