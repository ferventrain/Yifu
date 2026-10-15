from __future__ import annotations

import numpy as np
import pytest
import tifffile

from pipeline_modules.visualization.volume_heatmap_3d import (
    MODE_CHOICES,
    prepare_heatmap_grid,
    resolve_count_volume,
)


def test_prepare_heatmap_grid_axes_and_mask(tmp_path):
    # counts tif axes: (SI, AP, LR); annotation axes: (AP, SI, LR)
    counts = np.zeros((20, 30, 40), dtype=np.float32)
    counts[5:15, 5:25, 5:35] = 10.0
    counts[0, 0, 0] = 5.0  # far-field leak, must be masked
    tif = tmp_path / "count.tif"
    tifffile.imwrite(tif, counts)

    annotation = np.zeros((30, 20, 40), dtype=bool)
    annotation[6:24, 6:14, 6:34] = True

    grid = prepare_heatmap_grid(tif, annotation=annotation, sigma_um=25.0)
    assert grid.shape == (30, 20, 40)  # transposed to (AP, SI, LR)
    assert grid.dtype == np.float32
    # bright block retained, corner leak zeroed
    assert grid[10, 10, 20] > 0
    assert grid[0, 0, 0] == 0.0
    assert float(grid.min()) == 0.0
    # expected: block voxels inside the annotation (leak excluded), smoothed
    block_in_grid = np.zeros((30, 20, 40), dtype=np.float32)
    block_in_grid[5:25, 5:15, 5:35] = 10.0  # counts block after (SI,AP,LR)->(AP,SI,LR)
    expected = (block_in_grid * annotation).sum()
    assert grid.sum() == pytest.approx(expected, rel=0.35)


def test_prepare_heatmap_grid_annotation_shape_guard(tmp_path):
    counts = np.zeros((4, 4, 4), dtype=np.float32)
    tif = tmp_path / "c.tif"
    tifffile.imwrite(tif, counts)
    with pytest.raises(ValueError, match="annotation shape"):
        prepare_heatmap_grid(tif, annotation=np.zeros((5, 5, 5), dtype=bool), sigma_um=25.0)


def test_mode_choices_and_resolve(tmp_path):
    assert MODE_CHOICES == ("mip", "composite")
    vis = tmp_path / "visualization"
    vis.mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_count_volume(tmp_path)
    (vis / "s_ch1_count_volume.tif").write_bytes(b"x")
    assert resolve_count_volume(tmp_path).name == "s_ch1_count_volume.tif"
