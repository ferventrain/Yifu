import unittest

import numpy as np

from pipeline_modules.visualization.region_mip_export import (
    _mip_output_shapes,
    compute_region_mips,
    resolve_region_subtree_ids,
)


class RegionMipExportTests(unittest.TestCase):
    def test_mip_output_shapes_match_plane_conventions(self):
        shapes = _mip_output_shapes((10, 20, 30))
        self.assertEqual(shapes["horizontal"], (20, 30))
        self.assertEqual(shapes["coronal"], (10, 30))
        self.assertEqual(shapes["sagittal"], (10, 20))

    def test_compute_region_mips_masks_and_projects(self):
        signal = np.zeros((4, 5, 6), dtype=np.uint16)
        labels = np.zeros((4, 5, 6), dtype=np.uint16)
        labels[1:3, 2:4, 1:3] = 7
        signal[1, 2, 2] = 100
        signal[2, 3, 2] = 250
        signal[0, 0, 0] = 999

        mips = compute_region_mips(signal, labels, [7], block_shape=(2, 2, 2))
        self.assertEqual(int(mips["horizontal"].max()), 250)
        self.assertEqual(int(mips["coronal"].max()), 250)
        self.assertEqual(int(mips["sagittal"].max()), 250)
        self.assertEqual(int(mips["horizontal"][3, 2]), 250)
        self.assertEqual(int(mips["horizontal"][0, 0]), 0)

    def test_resolve_region_subtree_ids_accepts_acronym(self):
        from pathlib import Path

        cfg = Path(__file__).resolve().parents[1] / "pipeline_modules" / "registration" / "Region_Csv_Rev1_updated.CSV"
        subtree_ids, slug, name = resolve_region_subtree_ids("HIP", cfg_path=cfg)
        self.assertGreater(len(subtree_ids), 1)
        self.assertEqual(slug, "HIP")
        self.assertIn("Hippocampal", name)


if __name__ == "__main__":
    unittest.main()
