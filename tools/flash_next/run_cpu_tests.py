"""Run CPU contract tests; native tests also need installed engine dependencies."""

import sys
import unittest
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    for relative in (
        "tools/flash_next",
        "vllm/models/qwen4_exp/nvidia",
        "vllm/v1/core/sched",
    ):
        sys.path.insert(0, str(root / relative))
    suite = unittest.defaultTestLoader.discover(str(root / "tests/flash_next"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
