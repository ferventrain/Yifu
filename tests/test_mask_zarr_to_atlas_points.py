from __future__ import annotations

import numpy as np
import pandas as pd
import zarr

from pipeline_modules.visualization.mask_zarr_to_atlas_points import (
    mask_zarr_to_points_table,
    resolve_mask_zarr,
    write_outputs,
)


def _write_mask_zarr(path, data, chunks=(2, 2, 2)):
    root = zarr.open(str(path), mode="w")
    arr = root.zeros("0", shape=data.shape, chunks=chunks, dtype=data.dtype)
    arr[:] = data
    return path


def test_bins_foreground_voxels_to_physical_centroids(tmp_path):
    mask = np.zeros((2, 3, 4), dtype=np.uint8)
    mask[0, 0, 0] = 1
    mask[0, 0, 1] = 1
    mask[1, 2, 3] = 1
    path = _write_mask_zarr(tmp_path / "ch1_mask.zarr", mask)

    table, summary = mask_zarr_to_points_table(
        path,
        resolution_xyz=(10.0, 10.0, 10.0),
        target_resolution_xyz=(20.0, 20.0, 20.0),
        block_shape=(1, 2, 2),
        foreground_mode="nonzero",
    )

    assert summary["foreground_voxels"] == 3
    assert summary["occupied_target_bins"] == 2
    assert len(table) == 2

    first = table.iloc[0]
    assert first["grid_x"] == 0
    assert first["grid_y"] == 0
    assert first["grid_z"] == 0
    assert first["voxel_count"] == 2
    assert first["x"] == 10.0
    assert first["y"] == 5.0
    assert first["z"] == 5.0

    second = table.iloc[1]
    assert second["grid_x"] == 1
    assert second["grid_y"] == 1
    assert second["grid_z"] == 0
    assert second["voxel_count"] == 1
    assert second["x"] == 35.0
    assert second["y"] == 25.0
    assert second["z"] == 15.0


def test_filters_and_caps_points_by_voxel_count(tmp_path):
    mask = np.zeros((1, 4, 6), dtype=np.uint8)
    mask[0, 0, 0:3] = 1
    mask[0, 2, 2:4] = 1
    mask[0, 3, 5] = 1
    path = _write_mask_zarr(tmp_path / "ch2_mask.zarr", mask)

    table, summary = mask_zarr_to_points_table(
        path,
        resolution_xyz=(10.0, 10.0, 10.0),
        target_resolution_xyz=(20.0, 20.0, 20.0),
        min_voxels_per_point=2,
        max_points=1,
    )

    assert summary["foreground_voxels"] == 6
    assert summary["exported_points"] == 1
    assert len(table) == 1
    assert table.iloc[0]["voxel_count"] == 2


def test_resolve_mask_from_sample_layout(tmp_path):
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    expected = sample_dir / "ch3_mask.zarr"
    expected.mkdir()

    resolved = resolve_mask_zarr(sample_dir=sample_dir, signal_ch="ch3", mask_zarr=None)

    assert resolved == expected


def test_write_outputs_writes_csv_and_summary(tmp_path):
    table = pd.DataFrame({"x": [1.0], "y": [2.0], "z": [3.0], "voxel_count": [4]})
    outputs = write_outputs(table, {"exported_points": 1}, tmp_path / "atlas_points.csv")

    assert outputs["csv"].exists()
    assert outputs["summary"].exists()
    loaded = pd.read_csv(outputs["csv"])
    assert loaded.loc[0, "x"] == 1.0
