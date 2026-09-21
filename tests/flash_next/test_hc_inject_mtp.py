"""CPU dispatch contract; the real-layer harness supplies GPU numeric evidence.

The wrapper must change only opted-in, ordinary BF16 M5--16 injection calls.
AST isolation catches guard/fallback regressions without importing CUDA vLLM.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[2] / "vllm/models/qwen4_exp/nvidia"


class Tensor:
    def __init__(self, rows=8, width=10240):
        self.shape = (rows, width)
        self.ndim = 2
        self.dtype = "bf16"
        self.device = "cuda:0"
        self.is_cuda = True
        self.contiguous = True

    def is_contiguous(self):
        return self.contiguous


class Unquantized:
    pass


class Linear:
    weight = Tensor(4)
    quant_method = Unquantized()
    bias = None
    lora_config = None
    return_bias = False

    def __call__(self, x):
        return "native"


class DispatchTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse((ROOT / "flash_hc_inject_wrapper.py").read_text())
        functions = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name in ("eligible", "hc_inject", "_injection")
        ]
        self.ns = dict(
            _ENABLED=True,
            _MTP_ENABLED=False,
            envs=NS(VLLM_BATCH_INVARIANT=False),
            torch=NS(
                Tensor=Tensor,
                bfloat16="bf16",
                cuda=NS(get_device_capability=lambda _: (8, 6)),
                ops=NS(vllm=NS(flash_hc_inject_sm86=lambda *a: "operator")),
            ),
            ReplicatedLinear=Linear,
            UnquantizedLinearMethod=Unquantized,
            kernel=NS(
                candidate=lambda *a: "old", candidate_speculative=lambda *a: "new"
            ),
        )
        exec(
            compile(ast.Module(body=functions, type_ignores=[]), "<wrapper>", "exec"),
            self.ns,
        )

    def test_extra_rows_require_opt_in_and_leave_other_sizes_unchanged(self):
        for enabled in (False, True):
            self.ns["_MTP_ENABLED"] = enabled
            for m in range(1, 33):
                accepted = m in (1, 2, 4) or (enabled and 5 <= m <= 16)
                self.assertEqual(
                    self.ns["hc_inject"](Linear(), Tensor(m)),
                    "operator" if accepted else "native",
                )
            self.assertEqual(self.ns["_injection"](Tensor(4), Tensor(4)), "old")
            self.assertEqual(
                self.ns["_injection"](Tensor(8), Tensor(4)), "new" if enabled else "old"
            )

    def test_safety_fallbacks_still_bypass_operator(self):
        self.ns["_MTP_ENABLED"] = True
        for field, value in (
            ("dtype", "fp32"),
            ("device", "cuda:1"),
            ("contiguous", False),
            ("is_cuda", False),
        ):
            x = Tensor()
            setattr(x, field, value)
            self.assertEqual(self.ns["hc_inject"](Linear(), x), "native")
        for field, value in (
            ("bias", object()),
            ("lora_config", object()),
            ("return_bias", True),
            ("quant_method", object()),
        ):
            layer = Linear()
            setattr(layer, field, value)
            self.assertEqual(self.ns["hc_inject"](layer, Tensor()), "native")
        self.ns["envs"].VLLM_BATCH_INVARIANT = True
        self.assertEqual(self.ns["hc_inject"](Linear(), Tensor()), "native")
        with self.assertRaises(RuntimeError):
            self.ns["_injection"](Tensor(), Tensor(4))
        self.ns["envs"].VLLM_BATCH_INVARIANT = False
        self.ns["_ENABLED"] = False
        self.assertEqual(self.ns["hc_inject"](Linear(), Tensor()), "native")

    def test_kernel_rejects_unqualified_rows_before_cuda_launch(self):
        tree = ast.parse((ROOT / "flash_hc_inject_multirow.py").read_text())
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "candidate_speculative"
        )
        ns = dict(torch=self.ns["torch"])
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "<kernel>", "exec"), ns)
        for m in (0, 1, 2, 3, 4, 17, 20, 32):
            with self.assertRaises(ValueError):
                ns["candidate_speculative"](Tensor(m), Tensor(4))


if __name__ == "__main__":
    unittest.main()
