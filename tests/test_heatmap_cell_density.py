import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pipeline_modules.visualization.heatmap import (
    _render_local_slice_array,
    compute_shared_density_vmax,
    default_sample_stack_volume,
    discover_sample_dirs,
    linspace_bregma_coords,
    mean_voxels_per_cell,
    sample_has_batch_inputs,
    voxel_density_to_cell_density,
)


class HeatmapCellDensityTests(unittest.TestCase):
    def test_mean_voxels_per_cell_uses_native_voxel_volume(self):
        voxels_per_cell = mean_voxels_per_cell((1.8, 1.8, 2.0), 500.0)
        self.assertAlmostEqual(voxels_per_cell, 500.0 / (1.8 * 1.8 * 2.0))

    def test_voxel_density_to_cell_density_scales_by_voxels_per_cell(self):
        voxels_per_cell = mean_voxels_per_cell((1.8, 1.8, 2.0), 500.0)
        voxel_density = np.array([voxels_per_cell * 100.0, voxels_per_cell * 200.0], dtype=np.float32)
        cell_density = voxel_density_to_cell_density(
            voxel_density,
            resolution_xyz_um=(1.8, 1.8, 2.0),
            mean_cell_volume_um3=500.0,
        )
        self.assertAlmostEqual(float(cell_density[0]), 100.0, places=4)
        self.assertAlmostEqual(float(cell_density[1]), 200.0, places=4)

    def test_linspace_bregma_coords_returns_requested_count(self):
        coords = linspace_bregma_coords(1.1, -5.2, 12)
        self.assertEqual(len(coords), 12)
        self.assertAlmostEqual(coords[0], 1.1)
        self.assertAlmostEqual(coords[-1], -5.2)

    def test_compute_shared_density_vmax_uses_max_percentile_across_volumes(self):
        atlas_mask = np.zeros((4, 4, 4), dtype=np.uint8)
        atlas_mask[1:3, 1:3, 1:3] = 1
        vol_a = np.zeros((4, 4, 4), dtype=np.float32)
        vol_b = np.zeros((4, 4, 4), dtype=np.float32)
        vol_a[1, 1, 1] = 100.0
        vol_b[1, 1, 1] = 250.0
        shared = compute_shared_density_vmax([vol_a, vol_b], atlas_mask, percentile=99.5)
        self.assertAlmostEqual(shared, 250.0)

    def test_compute_shared_density_vmax_honors_explicit_vmax(self):
        atlas_mask = np.ones((2, 2, 2), dtype=np.uint8)
        vol = np.full((2, 2, 2), 999.0, dtype=np.float32)
        shared = compute_shared_density_vmax([vol], atlas_mask, explicit_vmax=42.0)
        self.assertAlmostEqual(shared, 42.0)

    def test_discover_sample_dirs_finds_mask_subdirectories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_a = root / "sample_a"
            sample_b = root / "sample_b"
            other = root / "notes"
            sample_a.mkdir()
            sample_b.mkdir()
            other.mkdir()
            (sample_a / "ch1_mask.zarr").mkdir()
            (sample_b / "ch2_mask.zarr").mkdir()
            (other / "readme.txt").write_text("x", encoding="utf-8")
            discovered = discover_sample_dirs(root)
            self.assertEqual([path.name for path in discovered], ["sample_a", "sample_b"])

    def test_discover_sample_dirs_raises_when_none_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                discover_sample_dirs(Path(tmp))

    def test_sample_has_batch_inputs_accepts_cached_heatmap_volume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "nao_1"
            (sample / "visualization").mkdir(parents=True)
            default_sample_stack_volume(sample).write_bytes(b"")
            self.assertTrue(sample_has_batch_inputs(sample))
            discovered = discover_sample_dirs(root)
            self.assertEqual([path.name for path in discovered], ["nao_1"])

    def test_render_local_slice_array_can_hide_region_contours(self):
        label = np.zeros((8, 8), dtype=np.uint16)
        label[2:6, 2:6] = 1
        signal = np.zeros((8, 8), dtype=np.float32)
        signal[2:6, 2:6] = 1.0
        with patch("pipeline_modules.visualization.heatmap._label_contour_lines") as mock_regions:
            _render_local_slice_array(
                signal,
                label,
                cmap_name="coolwarm",
                vmin=0.0,
                vmax=1.0,
                dpi=50,
                line_width=0.2,
                brain_outline_width=0.4,
                colorbar_label="test",
                show_region_contours=False,
            )
            mock_regions.assert_not_called()


if __name__ == "__main__":
    unittest.main()
