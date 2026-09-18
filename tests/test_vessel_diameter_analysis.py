from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline_modules.tubule_reconstruction.vessel_diameter_analysis import (
    build_parser,
    load_branch_diameters,
    parse_bin_edges,
    plot_vessel_diameter_histogram,
    summarize_diameters,
    summarize_region_vessel_diameter_bins,
)


class TestLoadBranchDiameters:
    def test_computes_diameter_and_filters_invalid_values(self, tmp_path):
        csv_path = tmp_path / "vessel_branch_metrics.csv"
        pd.DataFrame(
            {
                "mean_radius_um": [1.0, 2.5, np.nan, np.inf, -1.0, 0.0, 3.0],
            }
        ).to_csv(csv_path, index=False)

        diameters = load_branch_diameters(csv_path)

        assert np.allclose(diameters, np.array([2.0, 5.0, 6.0]))

    def test_requires_mean_radius_column(self, tmp_path):
        csv_path = tmp_path / "vessel_branch_metrics.csv"
        pd.DataFrame({"radius_um": [1.0, 2.0]}).to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="mean_radius_um"):
            load_branch_diameters(csv_path)

    def test_rejects_empty_valid_data(self, tmp_path):
        csv_path = tmp_path / "vessel_branch_metrics.csv"
        pd.DataFrame({"mean_radius_um": [np.nan, np.inf, -2.0, 0.0]}).to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="No valid vessel diameters"):
            load_branch_diameters(csv_path)


class TestPlotVesselDiameterHistogram:
    def test_writes_output_image(self, tmp_path):
        output_path = tmp_path / "histogram.png"

        saved = plot_vessel_diameter_histogram(
            np.array([2.0, 3.5, 4.0, 5.5, 8.0], dtype=np.float64),
            output_path=output_path,
            bins=4,
        )

        assert saved == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_summary_values(self):
        summary = summarize_diameters(np.array([2.0, 4.0, 6.0], dtype=np.float64))

        assert summary["n"] == 3
        assert summary["mean"] == pytest.approx(4.0)
        assert summary["median"] == pytest.approx(4.0)
        assert summary["min"] == pytest.approx(2.0)
        assert summary["max"] == pytest.approx(6.0)


def test_parse_bin_edges_requires_increasing_values():
    assert parse_bin_edges("0,2,4") == (0.0, 2.0, 4.0)
    with pytest.raises(ValueError):
        parse_bin_edges("0,2,2")


def test_parser_routes_modes_to_handlers():
    global_args = build_parser().parse_args(["global", "--branch_csv", "b.csv", "--output", "h.png"])
    assert global_args.mode == "global"
    assert global_args.branch_csv == "b.csv"

    region_args = build_parser().parse_args(
        [
            "region",
            "--vertex_csv", "v.csv",
            "--branch_csv", "b.csv",
            "--annotation_zarr", "labels.zarr",
            "--annotation_resolution_xyz", "1.8,1.8,2.0",
            "--cfg", "regions.csv",
            "--regions", "RA",
            "--output", "bins.csv",
        ]
    )
    assert region_args.mode == "region"
    assert region_args.annotation_dataset_name == "0"
    assert region_args.bin_edges_um == "0,2,4,6,8,10,12,15,20,30"


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
