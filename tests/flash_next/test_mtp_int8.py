"""CPU gate tests: online INT8 must select only the approved draft experts."""

import ast
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace as NS


class DraftInt8Tests(unittest.TestCase):
    def setUp(self):
        path = (
            Path(__file__).resolve().parents[2] / "vllm/models/qwen4_exp/nvidia/mtp.py"
        )
        tree = ast.parse(path.read_text())
        body = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_draft_int8_targets"
        ]
        namespace = {}
        exec(
            compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"),
            namespace,
        )
        self.targets = namespace["_draft_int8_targets"]
        self.config = NS(
            model_config=NS(
                hf_text_config=NS(
                    model_type="qwen4_exp_text",
                    hidden_size=2560,
                    num_hidden_layers=48,
                    mtp_num_hidden_layers=1,
                )
            ),
            parallel_config=NS(
                tensor_parallel_size=2,
                pipeline_parallel_size=2,
                enable_expert_parallel=False,
            ),
            speculative_config=NS(method="mtp", num_speculative_tokens=3),
        )

    def test_default_off_does_not_inspect_model(self):
        self.assertIsNone(self.targets(None, None, {}))

    def test_only_one_draft_expert_module_selected(self):
        original = copy.deepcopy(self.config)
        self.assertEqual(
            self.targets(self.config, 48, {"VLLM_FLASH_MTP_INT8_EXPERTS": "1"}),
            {"mtp.layers.48.mlp.experts": "int8_per_channel_weight_only"},
        )
        self.assertEqual(self.config, original)

    def test_reject_wrong_topology_depth_or_flag(self):
        for depth in (0, 5):
            self.config.speculative_config.num_speculative_tokens = depth
            with self.assertRaises(ValueError):
                self.targets(self.config, 48, {"VLLM_FLASH_MTP_INT8_EXPERTS": "1"})
        self.config.speculative_config.num_speculative_tokens = 3
        self.config.parallel_config.enable_expert_parallel = True
        with self.assertRaises(ValueError):
            self.targets(self.config, 48, {"VLLM_FLASH_MTP_INT8_EXPERTS": "1"})
        with self.assertRaises(ValueError):
            self.targets(None, None, {"VLLM_FLASH_MTP_INT8_EXPERTS": "yes"})


if __name__ == "__main__":
    unittest.main()
