"""CPU contract for the optional host-QSA page cap, without importing CUDA."""

import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch


class PageCapTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[2] / "vllm/platforms/interface.py"
        tree = ast.parse(path.read_text())
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_qsa_offload_block_size"
        ]
        namespace = {"_qsa_offload_bytes_per_token_per_layer": lambda config: 66}
        exec(
            compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
            namespace,
        )
        self.block_size = namespace["_qsa_offload_block_size"]
        self.config = NS(
            model_config=NS(
                hf_text_config=NS(
                    layer_types=["full_attention"] * 12, mtp_num_hidden_layers=1
                )
            ),
            parallel_config=NS(pipeline_parallel_size=2),
            speculative_config=NS(),
        )

    def test_cap_reduces_host_bin_without_losing_four_short_request_state_slots(self):
        with patch.dict(
            os.environ,
            {"VLLM_QSA_KV_OFFLOAD": "1", "VLLM_QSA_OFFLOAD_BLOCK_SIZE_CAP": "0"},
        ):
            self.assertEqual(self.block_size(self.config, 1634304), 3536)
            os.environ["VLLM_QSA_OFFLOAD_BLOCK_SIZE_CAP"] = "3072"
            self.assertEqual(self.block_size(self.config, 1634304), 3072)
        blocks = 1050000000 // 1634304
        self.assertEqual(blocks, 642)
        self.assertGreater(blocks - 1, 4 * 150)
        for size, expected in [(3536, 104), (3072, 52)]:
            layer_bytes = blocks * size * 512 * 2
            rounded = 1 << (layer_bytes - 1).bit_length()
            self.assertEqual(rounded * 13 * 2 // 2**30, expected)

    def test_bad_or_expanding_cap_rejected(self):
        for cap in ("-16", "1", "3073", "4096", "abc"):
            with (
                patch.dict(
                    os.environ,
                    {
                        "VLLM_QSA_KV_OFFLOAD": "1",
                        "VLLM_QSA_OFFLOAD_BLOCK_SIZE_CAP": cap,
                    },
                ),
                self.assertRaises(ValueError),
            ):
                self.block_size(self.config, 1634304)


if __name__ == "__main__":
    unittest.main()
