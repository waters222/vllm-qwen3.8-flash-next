"""Only allowlisted public engine/build inputs may enter the Docker context."""

import tempfile
import unittest
from pathlib import Path

from export_build_context import export

ROOT = Path(__file__).resolve().parents[2]


class BuildContextTests(unittest.TestCase):
    def test_context_excludes_git_credentials_weights_and_binaries(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.assertEqual(export(ROOT, output), 73)
            files = [p.relative_to(output) for p in output.rglob("*") if p.is_file()]
            self.assertEqual(len(files), 86)
            self.assertTrue(
                all(
                    p.parts[0] in {"engine", "csrc", "builder", "Dockerfile"}
                    for p in files
                )
            )
            self.assertFalse(
                any(p.suffix in {".so", ".safetensors", ".key"} for p in files)
            )
            with self.assertRaises(ValueError):
                export(ROOT, output)


if __name__ == "__main__":
    unittest.main()
