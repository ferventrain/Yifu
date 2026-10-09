"""Tests for vessel_analysis_report branch-angle and mask metric additions.

All tests use synthetic numpy/pandas/zarr data only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline_modules.tubule_reconstruction.vessel_analysis_report import (
    build_branch_angle_rows,
    compute_branch_angles,
    compute_mask_surface_and_fractal,
    open_mask_array,
    summarize_branch_angles,
)
from pipeline_modules.utils.zarr_io import create_output_zarr


def _express_style_branch_table():
    rows = [
        # Junction at the origin (node 0, degree 3) with chords along +x, -x, +y:
        # pairwise angles 90, 90, 180.
        dict(skeleton_id=0, branch_id=0, start_node=0, end_node=1, start_degree=3, end_degree=1,
             source_z_um=0.0, source_y_um=0.0, source_x_um=0.0,
             target_z_um=0.0, target_y_um=0.0, target_x_um=10.0,
             length_um=10.0, mean_radius_um=2.0, is_loop=False),
        dict(skeleton_id=0, branch_id=1, start_node=0, end_node=2, start_degree=3, end_degree=1,
             source_z_um=0.0, source_y_um=0.0, source_x_um=0.0,
             target_z_um=0.0, target_y_um=0.0, target_x_um=-10.0,
             length_um=10.0, mean_radius_um=2.0, is_loop=False),
        dict(skeleton_id=0, branch_id=2, start_node=0, end_node=3, start_degree=3, end_degree=1,
             source_z_um=0.0, source_y_um=0.0, source_x_um=0.0,
             target_z_um=0.0, target_y_um=10.0, target_x_um=0.0,
             length_um=10.0, mean_radius_um=2.0, is_loop=False),
        # Loop branch at the junction node: non-zero chord, must be excluded.
        dict(skeleton_id=0, branch_id=3, start_node=0, end_node=0, start_degree=3, end_degree=3,
             source_z_um=0.0, source_y_um=0.0, source_x_um=0.0,
             target_z_um=5.0, target_y_um=0.0, target_x_um=0.0,
             length_um=5.0, mean_radius_um=2.0, is_loop=True),
        # Isolated terminal-terminal branch: no junction, must be ignored.
        dict(skeleton_id=0, branch_id=4, start_node=4, end_node=5, start_degree=1, end_degree=1,
             source_z_um=50.0, source_y_um=50.0, source_x_um=50.0,
             target_z_um=50.0, target_y_um=60.0, target_x_um=50.0,
             length_um=10.0, mean_radius_um=2.0, is_loop=False),
    ]
    return pd.DataFrame(rows)


def test_branch_angles_express_table():
    result = compute_branch_angles(_express_style_branch_table())
    angles = np.sort(result["angles_deg"])
    assert angles.size == 3
    assert np.allclose(angles, [90.0, 90.0, 180.0], atol=1e-6)
    summary = summarize_branch_angles(result)
    assert summary["num_junctions_measured"] == 1
    assert summary["branch_angle_pair_count"] == 3
    assert np.isclose(summary["branch_angle_mean_deg"], 120.0)
    assert np.isclose(summary["branch_angle_median_deg"], 90.0)
    assert np.isclose(summary["branch_angle_junction_mean_deg"], 120.0)


def test_branch_angles_kimimaro_join(tmp_path):
    table = _express_style_branch_table()
    table = table.drop(
        columns=[name for name in table.columns if name.startswith(("source_", "target_"))]
    )
    vertices = pd.DataFrame(
        [
            dict(skeleton_id=0, node_id=0, z_um=0.0, y_um=0.0, x_um=0.0),
            dict(skeleton_id=0, node_id=1, z_um=0.0, y_um=0.0, x_um=10.0),
            dict(skeleton_id=0, node_id=2, z_um=0.0, y_um=0.0, x_um=-10.0),
            dict(skeleton_id=0, node_id=3, z_um=0.0, y_um=10.0, x_um=0.0),
            dict(skeleton_id=0, node_id=4, z_um=50.0, y_um=50.0, x_um=50.0),
            dict(skeleton_id=0, node_id=5, z_um=50.0, y_um=60.0, x_um=50.0),
        ]
    )
    vertex_csv = tmp_path / "skeleton_vertices.csv"
    vertices.to_csv(vertex_csv, index=False)
    result = compute_branch_angles(table, vertex_csv)
    angles = np.sort(result["angles_deg"])
    assert angles.size == 3
    assert np.allclose(angles, [90.0, 90.0, 180.0], atol=1e-6)


def test_branch_angles_unavailable_tables():
    # EDT-style table without node identity / degrees returns None.
    table = pd.DataFrame(
        {
            "branch_length_um": [10.0, 12.0],
            "mean_radius_um": [2.0, 3.0],
            "tortuosity": [1.0, 1.2],
            "is_loop": [False, False],
            "is_branch_to_branch": [True, True],
            "is_terminal_branch": [False, False],
        }
    )
    assert compute_branch_angles(table) is None
    assert summarize_branch_angles(None) is None


def test_build_branch_angle_rows():
    rows = build_branch_angle_rows(np.array([5.0, 90.0, 175.0]))
    assert int(rows["pair_count"].sum()) == 3
    assert int(rows.loc[rows["angle_bin_deg"] == "0-15", "pair_count"].iloc[0]) == 1
    assert int(rows.loc[rows["angle_bin_deg"] == "90-105", "pair_count"].iloc[0]) == 1
    assert int(rows.loc[rows["angle_bin_deg"] == ">=165", "pair_count"].iloc[0]) == 1


def test_mask_surface_area_anisotropic_cube(tmp_path):
    # 8x8x8-voxel cube at res x=1, y=2, z=4 um: surface = 2*(16*32) + 2*(8*32) + 2*(8*16) = 1792.
    zarr_path = tmp_path / "mask.zarr"
    _, arr = create_output_zarr(zarr_path, (16, 16, 16), (8, 8, 8), "uint8")
    arr[4:12, 4:12, 4:12] = 1
    mask = open_mask_array(str(zarr_path))
    result = compute_mask_surface_and_fractal(mask, (1.0, 2.0, 4.0), box_sizes_um=(4.0, 8.0, 16.0))
    assert np.isclose(result["surface_area_um2"], 1792.0)


def test_fractal_dimension_solid_cube_exact(tmp_path):
    # Solid 64^3-voxel cube at 1 um isotropic: N(S) = (64/S)^3 for S in 4..32, slope exactly 3.
    # Chunks of 24 are deliberately misaligned with the box grid to exercise cross-chunk dedup.
    zarr_path = tmp_path / "solid.zarr"
    _, arr = create_output_zarr(zarr_path, (64, 64, 64), (24, 24, 24), "uint8")
    arr[:] = 1
    mask = open_mask_array(str(zarr_path))
    result = compute_mask_surface_and_fractal(mask, (1.0, 1.0, 1.0), box_sizes_um=(4.0, 8.0, 16.0, 32.0))
    assert list(result["fd_points"]["occupied_boxes"]) == [4096, 512, 64, 8]
    assert np.isclose(result["fractal_dimension"], 3.0, atol=1e-6)
    # Voxel-boundary surface of the full cube: 2*(64*64) per axis = 6 * 4096.
    assert np.isclose(result["surface_area_um2"], 6 * 64 * 64)


def test_mask_metrics_empty_mask(tmp_path):
    zarr_path = tmp_path / "empty.zarr"
    create_output_zarr(zarr_path, (16, 16, 16), (8, 8, 8), "uint8")
    mask = open_mask_array(str(zarr_path))
    result = compute_mask_surface_and_fractal(mask, (1.0, 1.0, 1.0), box_sizes_um=(4.0, 8.0, 16.0))
    assert result["surface_area_um2"] == 0.0
    assert not np.isfinite(result["fractal_dimension"])
