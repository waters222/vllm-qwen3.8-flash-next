import unittest
from pathlib import Path

import build_flash_marlin_schedule as build

ROOT = Path(__file__).parents[2] / "csrc"


class WholeTileBuildTests(unittest.TestCase):
    def test_pinned_kernel_only_changes_work_ownership(self):
        base = build.load_base()
        source = (ROOT / (base.PREFIX + "marlin_template.h")).read_text()
        changed = build.generate_kernel(source)
        self.assertIn("part2_mn_tiles = 0;", changed)
        self.assertIn("init_part1_slice();", changed)
        self.assertNotIn("    init_part2_slice();", changed)
        for start, end in (
            ("  auto read_moe_block_data", "  auto init_slice"),
            ("  init_slice();", None),
        ):
            if end:
                self.assertEqual(
                    source[source.index(start) : source.index(end)],
                    changed[changed.index(start) : changed.index(end)],
                )
            else:
                self.assertEqual(
                    source[source.index(start) :], changed[changed.index(start) :]
                )
        with self.assertRaises(ValueError):
            build.generate_kernel(source + "\n")

    def test_dispatch_is_private_stage4_down_only(self):
        base = build.load_base()
        source = (ROOT / (base.PREFIX + "ops.cu")).read_text()
        changed = build.generate_dispatch(source)
        self.assertIn("#define MARLIN_NAMESPACE_NAME flash_marlin_dp4", changed)
        self.assertIn("Whole-tile probe is expert-down only", changed)
        self.assertIn("int stages = 4;", changed)
        self.assertNotIn("STABLE_TORCH_LIBRARY_IMPL(_moe_C", changed)
        self.assertIn(
            "kernel<<<blocks, num_threads, max_shared_mem, stream>>>", changed
        )
        self.assertIn("!use_atomic_add && use_fp32_reduce", changed)
        with self.assertRaises(ValueError):
            build.generate_dispatch(source + "\n")

    def test_work_coverage_native_and_whole_tile(self):
        for tiles in (1, 50, 100, 164, 200, 246, 247, 328, 340, 400, 600, 800):
            for k in (5, 20, 40):
                for blocks in (82, 164, 246, 328):
                    for whole in (False, True):
                        result = build.schedule(tiles, k, blocks, whole_tile=whole)
                        self.assertEqual(result["total_k_tile_work"], tiles * k)
                        if whole:
                            self.assertEqual(result["split_output_tiles"], 0)
                            self.assertEqual(result["extra_partial_results"], 0)
                            self.assertEqual(
                                result["active_blocks"], min(tiles, blocks)
                            )
                            self.assertEqual(
                                result["max_k_tiles_per_block"],
                                ((tiles + blocks - 1) // blocks) * k,
                            )

    def test_small_down_tradeoff_and_bad_input(self):
        native = build.schedule(200, 5, 246)
        whole = build.schedule(200, 5, 246, whole_tile=True)
        # ceil(200*5/246)==5: native M1 already assigns full K tiles.
        self.assertEqual(native["extra_partial_results"], 0)
        self.assertEqual(whole["active_blocks"], 200)
        self.assertEqual(whole["max_k_tiles_per_block"], 5)
        self.assertGreater(build.schedule(400, 5, 246)["extra_partial_results"], 0)
        for args in ((0, 5, 246), (200, -1, 246), (200, 5, True)):
            with self.assertRaises(ValueError):
                build.schedule(*args)


if __name__ == "__main__":
    unittest.main()
