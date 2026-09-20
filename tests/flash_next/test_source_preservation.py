"""CPU preservation contract: engine bytes, extension inputs, and syntax.

The manifest is the migration's output contract. These checks catch missing
files, accidentally reformatted pinned code, and an incomplete CUDA archive.
They do not establish GPU numerical correctness or serving performance.
"""

import ast
import hashlib
import json
import unittest
from pathlib import Path

import build_flash_marlin_schedule as build

ROOT = Path(__file__).resolve().parents[2]


class PreservationTests(unittest.TestCase):
    def test_installed_engine_sources_are_preserved(self):
        manifest = json.loads(
            (ROOT / "docs/flash_next/source_manifest.json").read_text()
        )
        self.assertEqual(len(manifest["engine_sources"]), 40)
        for name, expected in manifest["engine_sources"].items():
            with self.subTest(source=name):
                source = (ROOT / name).read_bytes()
                self.assertEqual(hashlib.sha256(source).hexdigest(), expected)
                ast.parse(source)

    def test_cuda_sources_reproduce_qualified_generator_outputs(self):
        base = build.load_base()
        for name, pin in base.PINS.items():
            source = (ROOT / "csrc" / name).read_bytes()
            self.assertEqual(hashlib.sha256(source).hexdigest(), pin)
        for original, generated, generate in (
            ("ops.cu", "flash_marlin_dp4.cu", build.generate_dispatch),
            ("marlin_template.h", "marlin_dp_template.h", build.generate_kernel),
        ):
            source = (ROOT / "csrc" / base.PREFIX / original).read_text()
            saved = (ROOT / "csrc/flash_next/marlin_dp4" / generated).read_text()
            self.assertEqual(generate(source), saved)


if __name__ == "__main__":
    unittest.main()
