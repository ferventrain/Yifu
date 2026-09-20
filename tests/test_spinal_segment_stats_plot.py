"""Tests for ``pipeline_modules.visualization.spinal_segment_stats_plot``."""

from __future__ import annotations

import pandas as pd

from pipeline_modules.visualization.spinal_segment_stats_plot import (
    group_of,
    load_segment_frame,
    plot_spinal_segment_stats,
)


def _write_stats_csv(path):
    rows = [
        {"Level": 0, "Name": "Spinal cord all segments", "Signal Count": 10,
         "Signal Volume (mm3)": 0.002, "Count Density (count/mm3)": 5000.0, "Voxel Density": 0.001},
        {"Level": 1, "Name": "T2", "Signal Count": 4,
         "Signal Volume (mm3)": 0.0009, "Count Density (count/mm3)": 400.0, "Voxel Density": 0.0004},
        {"Level": 1, "Name": "C1", "Signal Count": 6,
         "Signal Volume (mm3)": 0.0011, "Count Density (count/mm3)": 600.0, "Voxel Density": 0.0006},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_segment_frame_sorts_anatomically(tmp_path):
    frame = load_segment_frame(_write_stats_csv(tmp_path / "stats.csv"))
    assert frame["Name"].tolist() == ["C1", "T2"]
    assert frame["Level"].tolist() == [1, 1]


def test_group_of_maps_prefixes():
    assert group_of("C3") == "C"
    assert group_of("T12") == "T"
    assert group_of("Co2") == "Co"
    assert group_of("??") == "Co"


def test_plot_spinal_segment_stats_writes_png(tmp_path):
    stats_csv = _write_stats_csv(tmp_path / "stats.csv")
    output_png = tmp_path / "fig.png"

    result = plot_spinal_segment_stats(stats_csv, output_png)

    assert result == output_png
    assert output_png.exists()
    assert output_png.stat().st_size > 0
