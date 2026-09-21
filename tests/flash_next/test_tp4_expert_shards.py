"""CPU-only coverage and boundary checks for unequal TP4 expert partitions."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "vllm/model_executor/layers/fused_moe/flash_tp4_expert_shards.py"
spec = importlib.util.spec_from_file_location("tp4_shards", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PartitionTests(unittest.TestCase):
    def test_all_layers_cover_checkpoint_once_with_whole_tiles_and_groups(self):
        for layer in range(48):
            for extent in (640, 80, 20):
                covered = []
                for rank in range(4):
                    start, width = module.checkpoint_partition(layer, rank, extent)
                    covered.extend(range(start, start + width))
                    _, channels = module.expert_partition(layer, rank)
                    self.assertEqual(channels % 64, 0)
                self.assertEqual(covered, list(range(extent)))

    def test_memory_balances_across_ranks_over_48_layers(self):
        totals = [sum(module.expert_partition(layer, rank)[1] for layer in range(48))
                  for rank in range(4)]
        self.assertEqual(totals, [48 * 160] * 4)

    def test_invalid_model_axes_and_ranks_fail_closed(self):
        for layer, rank, extent in ((48, 0, 640), (-1, 0, 640), (0, 4, 640),
                                    (0, -1, 640), (0, 0, 128), (True, 0, 640)):
            with self.assertRaises(ValueError):
                module.checkpoint_partition(layer, rank, extent)


if __name__ == "__main__":
    unittest.main()
