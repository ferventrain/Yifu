"""Tests for the min/max object-size filter and physical volume columns
added to ``region_signal_analysis_zarr_graph.analyze_zarr_graph``."""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline_modules.registration.region_signal_analysis_zarr_graph import analyze_zarr_graph
from pipeline_modules.utils.zarr_io import create_output_zarr

# Volume geometry: shape (6, 16, 16), chunks (6, 8, 8) -> 4 blocks split at
# y=8 / x=8 so one object crosses a block boundary (exercises stitching).
VOL_SHAPE = (6, 16, 16)
CHUNKS = (6, 8, 8)

# Objects (26-connected components inside the mask):
#   small:  4 voxels  (below min)
#   medium: 9 voxels  (inside the [5, 12] window; crosses the y=8 block seam)
#   large: 48 voxels  (above max)
SMALL = (1, slice(1, 3), slice(1, 3))       # 2*2 = 4
MEDIUM = (1, slice(6, 9), slice(4, 7))      # 3*3 = 9, y spans 6..8 across the seam
LARGE = (slice(0, 3), slice(10, 14), slice(10, 14))  # 3*4*4 = 48


def _write_region_csv(path):
    rows = [
        {"id": 1, "name": "root", "acronym": "root", "structure_id_path": "[1]"},
        {"id": 10, "name": "RegionA", "acronym": "RA", "structure_id_path": "[1, 10]"},
        {"id": 20, "name": "RegionB", "acronym": "RB", "structure_id_path": "[1, 20]"},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture()
def analysis_inputs(tmp_path):
    mask = tmp_path / "mask.zarr"
    _, mask_arr = create_output_zarr(mask, VOL_SHAPE, CHUNKS, "uint8")
    mask_arr[SMALL] = 1
    mask_arr[MEDIUM] = 1
    mask_arr[LARGE] = 1

    label = tmp_path / "label.zarr"
    _, label_arr = create_output_zarr(label, VOL_SHAPE, CHUNKS, "int32")
    label_arr[:3, :, :] = 10
    label_arr[3:, :, :] = 20

    signal = tmp_path / "signal.zarr"
    _, signal_arr = create_output_zarr(signal, VOL_SHAPE, CHUNKS, "uint16")
    signal_arr[SMALL] = 100
    signal_arr[MEDIUM] = 100
    signal_arr[LARGE] = 100

    region_csv = _write_region_csv(tmp_path / "regions.csv")
    return mask, label, signal, region_csv


def _run(inputs, tmp_path, **overrides):
    mask, label, signal, region_csv = inputs
    kwargs = dict(
        mask_zarr_path=str(mask),
        label_zarr_path=str(label),
        signal_zarr_path=str(signal),
        cfg_path=str(region_csv),
        output_path=str(tmp_path / "out.xlsx"),
        dataset_name="0",
        block_size=None,
        foreground_mode="equal",
        foreground_label=1,
        min_voxels=5,
        flush_every=0,
        resolution_xyz=(1.0, 1.0, 2.0),
        tmp_dir="",
        keep_tmp=False,
        pass1_workers=1,
        max_voxels=12,
        report_physical_volume=True,
    )
    kwargs.update(overrides)
    analyze_zarr_graph(**kwargs)
    sheets = pd.read_excel(kwargs["output_path"], sheet_name=None)
    return sheets


def _region_row(sheets, sheet, name):
    frame = sheets[sheet]
    row = frame[frame["Name"] == name].iloc[0]
    return row


def test_max_voxels_drops_oversize_objects(analysis_inputs, tmp_path):
    sheets = _run(analysis_inputs, tmp_path)

    row = _region_row(sheets, "Level_1", "RegionA")
    assert int(row["Signal Count"]) == 1  # only the 9-voxel medium object
    assert int(row["Signal Voxels"]) == 9
    assert int(row["Sum Intensity"]) == 900

    # RegionB has no mask foreground at all.
    row_b = _region_row(sheets, "Level_1", "RegionB")
    assert int(row_b["Signal Count"]) == 0

    root = _region_row(sheets, "Level_0", "root")
    assert int(root["Signal Count"]) == 1
    assert int(root["Signal Voxels"]) == 9


def test_min_voxels_still_drops_undersize_objects(analysis_inputs, tmp_path):
    # min=5 drops the 4-voxel small object; max disabled keeps medium + large.
    sheets = _run(analysis_inputs, tmp_path, max_voxels=0)

    row = _region_row(sheets, "Level_1", "RegionA")
    assert int(row["Signal Count"]) == 2  # medium (9) + large (48)
    assert int(row["Signal Voxels"]) == 57


def test_physical_volume_columns(analysis_inputs, tmp_path):
    sheets = _run(analysis_inputs, tmp_path)

    row = _region_row(sheets, "Level_1", "RegionA")
    # resolution (1, 1, 2) -> voxel volume 2 um3; region = z<3 half of 6*16*16.
    assert float(row["Signal Volume (um3)"]) == pytest.approx(18.0)
    assert float(row["Signal Volume (mm3)"]) == pytest.approx(18.0e-9)
    assert float(row["Region Volume (mm3)"]) == pytest.approx(768 * 2.0 / 1e9)
    assert float(row["Mean Object Volume (um3)"]) == pytest.approx(18.0)
    expected_density = 1 / (768 * 2.0 / 1e9)
    assert float(row["Count Density (count/mm3)"]) == pytest.approx(expected_density)


def test_physical_volume_columns_disabled_by_default(analysis_inputs, tmp_path):
    sheets = _run(analysis_inputs, tmp_path, report_physical_volume=False)
    assert "Signal Volume (um3)" not in sheets["Level_1"].columns
