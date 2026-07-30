from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline_modules.tubule_reconstruction.region_vessel_diameter_bins import (
    parse_bin_edges,
    summarize_region_vessel_diameter_bins,
)


def test_parse_bin_edges_requires_increasing_values():
    assert parse_bin_edges("0,2,4") == (0.0, 2.0, 4.0)
    with pytest.raises(ValueError):
        parse_bin_edges("0,2,2")


def test_summarizes_only_branches_in_requested_region(tmp_path, tiny_annotation_zarr, tiny_region_csv):
    vertices = pd.DataFrame(
        {
            "skeleton_id": [1, 1, 1, 1],
            "node_id": [0, 1, 2, 3],
            "z_um": [1.0, 3.0, 5.0, 7.0],
            "y_um": [1.0, 1.0, 1.0, 1.0],
            "x_um": [1.0, 1.0, 1.0, 1.0],
        }
    )
    branches = pd.DataFrame(
        {
            "skeleton_id": [1, 1],
            "branch_id": [0, 1],
            "start_node": [0, 2],
            "end_node": [1, 3],
            "mean_radius_um": [1.0, 3.0],
            "branch_length_um": [10.0, 20.0],
        }
    )
    vertex_csv = tmp_path / "vertices.csv"
    branch_csv = tmp_path / "branches.csv"
    vertices.to_csv(vertex_csv, index=False)
    branches.to_csv(branch_csv, index=False)

    table = summarize_region_vessel_diameter_bins(
        vertex_csv,
        branch_csv,
        tiny_annotation_zarr,
        tiny_region_csv,
        "RA",
        annotation_resolution_xyz=(1.0, 1.0, 1.0),
        bin_edges_um=(0.0, 3.0, 5.0),
    )

    assert table["branch_count"].tolist() == [1, 0, 0]
    assert table["total_valid_branch_count"].tolist() == [1, 1, 1]
    assert table.loc[0, "total_branch_length_um"] == pytest.approx(10.0)
    assert table.loc[0, "branch_percent"] == pytest.approx(100.0)
