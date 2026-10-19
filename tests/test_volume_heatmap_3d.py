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


def test_camera_position_geometry():
    from pipeline_modules.visualization.volume_heatmap_3d import camera_position

    pos, focal, viewup = camera_position((528, 320, 456), 0.0, 0.0)
    # az=0/el=0: camera sits anterior of center; screen-up is -SI (+SI is inferior)
    assert pos[0] > focal[0]
    assert pos[1] == pytest.approx(focal[1])
    assert pos[2] == pytest.approx(focal[2])
    assert viewup == (0.0, -1.0, 0.0)
    pos_top, _ft, _upt = camera_position((528, 320, 456), 0.0, 20.0)
    assert pos_top[1] < _ft[1]  # positive elevation = camera dorsal (above, -SI)
    pos90, _f, up90 = camera_position((528, 320, 456), 90.0, 0.0)
    assert pos90[2] > 0 and pos90[0] == pytest.approx(_f[0])  # from the left
    _ptop, _ftop, uptop = camera_position((528, 320, 456), 0.0, 90.0)
    assert uptop == (0.0, 0.0, 1.0)  # degenerate viewup switches axis


def test_region_mask_from_acronyms(tmp_path):
    import pandas as pd

    from pipeline_modules.visualization.volume_heatmap_3d import REGION_CSV, region_mask_from_acronyms

    # synthetic CSV: TH(549) with child VL(123); unrelated region HY(200)
    frame = pd.DataFrame(
        [
            {"id": 549, "name": "Thalamus,TH", "structure_id_path": str([0, 1, 549])},
            {"id": 123, "name": "Ventrolateral thalamic nucleus,VL", "structure_id_path": str([0, 1, 549, 123])},
            {"id": 200, "name": "Hypothalamus,HY", "structure_id_path": str([0, 1, 200])},
        ]
    )
    csv = tmp_path / "tree.csv"
    frame.to_csv(csv, index=False)
    annotation = np.zeros((4, 4, 4), dtype=np.int32)
    annotation[0, 0, 0] = 549
    annotation[1, 1, 1] = 123
    annotation[2, 2, 2] = 200
    mask = region_mask_from_acronyms(["TH"], annotation, csv)
    assert mask[0, 0, 0] and mask[1, 1, 1] and not mask[2, 2, 2]  # descendants included
    with pytest.raises(ValueError, match="No region acronyms matched"):
        region_mask_from_acronyms(["XX"], annotation, csv)
    assert REGION_CSV.exists()
