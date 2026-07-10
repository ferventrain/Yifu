import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import zarr

from pipeline_modules.visualization.region_scope_signal_stats import compute_region_scope_signal_stats


class RegionScopeSignalStatsTests(unittest.TestCase):
    def test_compute_region_scope_signal_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask_path = root / "mask.zarr"
            label_path = root / "label.zarr"
            signal_path = root / "signal.zarr"

            mask = zarr.open(str(mask_path), mode="w", shape=(2, 4, 4), chunks=(1, 4, 4), dtype=np.uint8)
            label = zarr.open(str(label_path), mode="w", shape=(2, 4, 4), chunks=(1, 4, 4), dtype=np.uint16)
            signal = zarr.open(str(signal_path), mode="w", shape=(2, 4, 4), chunks=(1, 4, 4), dtype=np.uint16)

            mask[:] = 0
            label[:] = 0
            signal[:] = 0
            mask[0, 1:3, 1:3] = 1
            label[0, 1:3, 1:3] = 7
            signal[0, 1:3, 1:3] = 100

            with mock.patch(
                "pipeline_modules.visualization.region_scope_signal_stats.resolve_region_subtree_ids",
                return_value=({7}, "HIP", "Hippocampus", {7: {"id": 7, "name": "Hippocampus", "acronym": "HIP"}}),
            ):
                payload = compute_region_scope_signal_stats(
                    sample_dir=root,
                    region_query="HIP",
                    signal_ch="ch3",
                    mask_zarr_path=mask_path,
                    label_zarr_path=label_path,
                    signal_zarr_path=signal_path,
                    block_shape=(1, 4, 4),
                    pass1_workers=1,
                    include_signal_count=True,
                    min_voxels=1,
                )

            summary = payload["summary"]
            self.assertEqual(summary["total_voxels"], 4)
            self.assertEqual(summary["signal_voxels"], 4)
            self.assertEqual(summary["sum_intensity"], 400.0)
            self.assertGreaterEqual(summary["signal_count"], 1)
            self.assertTrue(Path(payload["output_json"]).exists())


if __name__ == "__main__":
    unittest.main()
