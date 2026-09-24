from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_modules.registration.region_signal_analysis_zarr_graph import (
    build_region_row,
    flush_rows_to_excel,
)

WHOLE_COLUMNS = ["Total Voxels", "Signal Voxels", "Voxel Density", "Signal Count", "Sum Intensity"]
HEMI_COLUMNS = [
    "Left Total Voxels", "Left Signal Voxels", "Left Voxel Density", "Left Signal Count", "Left Sum Intensity",
    "Right Total Voxels", "Right Signal Voxels", "Right Voxel Density", "Right Signal Count", "Right Sum Intensity",
]


def _stats(total=100, signal=10):
    return {
        "total_voxels": total,
        "signal_voxels": signal,
        "signal_count": 3,
        "sum_intensity": 1000.0,
        "hemispheres": {
            1: {"total_voxels": total - signal // 2, "signal_voxels": signal // 2, "signal_count": 1, "sum_intensity": 400.0},
            2: {"total_voxels": signal // 2, "signal_voxels": signal // 5, "signal_count": 2, "sum_intensity": 600.0},
        },
    }


def test_build_region_row_hemisphere_only_suppresses_whole_brain():
    row = build_region_row({"name": "root", "st_level": 0}, _stats(), hemisphere_only=True)
    assert row["Name"] == "root"
    for column in WHOLE_COLUMNS:
        assert column not in row
    for column in HEMI_COLUMNS:
        assert column in row


def test_build_region_row_default_keeps_whole_brain():
    row = build_region_row({"name": "root", "st_level": 0}, _stats())
    for column in WHOLE_COLUMNS + HEMI_COLUMNS:
        assert column in row


def test_flush_rows_to_excel_hemisphere_only_columns(tmp_path):
    rows = [
        build_region_row({"name": "root", "st_level": 0}, _stats(), hemisphere_only=True),
        build_region_row({"name": "child", "st_level": 1}, _stats(total=50, signal=5), hemisphere_only=True),
    ]
    out = tmp_path / "stats.xlsx"
    flush_rows_to_excel(rows, out, hemisphere_only=True)
    frame = pd.read_excel(out, sheet_name="Level_0")
    assert list(frame.columns) == ["Name"] + HEMI_COLUMNS
    assert len(frame) == 1
    assert len(pd.read_excel(out, sheet_name="Level_1")) == 1
