from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline_modules.tubule_reconstruction.vessel_analysis_report import (
    is_express_branch_metrics,
    load_express_branch_table,
    stream_vertex_stats_from_edges,
    stream_vertices,
)


def _write_express_fixtures(tmp_path, with_degrees):
    branch_rows = [
        {
            "skeleton_id": 0,
            "branch_id": 0,
            "start_node": 0 if with_degrees else None,
            "end_node": 1 if with_degrees else None,
            "start_degree": 3 if with_degrees else None,
            "end_degree": 3 if with_degrees else None,
            "source_z_um": 0.0,
            "source_y_um": 0.0,
            "source_x_um": 0.0,
            "target_z_um": 0.0,
            "target_y_um": 0.0,
            "target_x_um": 30.0,
            "length_um": 60.0,
            "num_points": 10,
            "mean_radius_um": 2.0,
            "is_stitch": False,
        },
        {
            "skeleton_id": 0,
            "branch_id": 1,
            "start_node": 1 if with_degrees else None,
            "end_node": 2 if with_degrees else None,
            "start_degree": 3 if with_degrees else None,
            "end_degree": 1 if with_degrees else None,
            "source_z_um": 10.0,
            "source_y_um": 0.0,
            "source_x_um": 0.0,
            "target_z_um": 10.0,
            "target_y_um": 40.0,
            "target_x_um": 0.0,
            "length_um": 80.0,
            "num_points": 12,
            "mean_radius_um": 4.0,
            "is_stitch": False,
        },
        {
            # loop: both ends at the same node, degenerate tortuosity
            "skeleton_id": 0,
            "branch_id": 2,
            "start_node": 3 if with_degrees else None,
            "end_node": 3 if with_degrees else None,
            "start_degree": 3 if with_degrees else None,
            "end_degree": 3 if with_degrees else None,
            "source_z_um": 5.0,
            "source_y_um": 10.0,
            "source_x_um": 10.0,
            "target_z_um": 5.0,
            "target_y_um": 10.0,
            "target_x_um": 10.0,
            "length_um": 50.0,
            "num_points": 20,
            "mean_radius_um": 3.0,
            "is_stitch": False,
        },
    ]
    branch_rows = [{k: v for k, v in row.items() if v is not None} for row in branch_rows]
    branch_csv = tmp_path / "vessel_branch_metrics.csv"
    pd.DataFrame(branch_rows).to_csv(branch_csv, index=False)

    edge_csv = tmp_path / "skeleton_edges.csv"
    pd.DataFrame(
        {
            "skeleton_id": [0] * 5,
            "edge_id": [0, 1, 2, 3, 4],
            "source_node": [0, 0, 1, 2, 3],
            "target_node": [1, 3, 2, 1, 0],
        }
    ).to_csv(edge_csv, index=False)

    vertex_csv = tmp_path / "skeleton_vertices.csv"
    pd.DataFrame(
        {
            "skeleton_id": [0] * 4,
            "node_id": [0, 1, 2, 3],
            "z_um": [0.0, 10.0, 10.0, 5.0],
            "y_um": [0.0, 0.0, 40.0, 10.0],
            "x_um": [0.0, 0.0, 0.0, 10.0],
        }
    ).to_csv(vertex_csv, index=False)
    return branch_csv, edge_csv, vertex_csv


@pytest.mark.parametrize("with_degrees", [False, True])
def test_load_express_branch_table(tmp_path, with_degrees):
    branch_csv, _, _ = _write_express_fixtures(tmp_path, with_degrees)
    assert is_express_branch_metrics(branch_csv)

    table = load_express_branch_table(branch_csv)

    assert np.allclose(table["branch_length_um"], [60.0, 80.0, 50.0])
    tortuosity = table["tortuosity"].to_numpy(dtype=np.float64)
    assert tortuosity[0] == pytest.approx(2.0)
    assert tortuosity[1] == pytest.approx(2.0)
    assert np.isnan(tortuosity[2])
    assert table["is_loop"].tolist() == [False, False, True]
    if with_degrees:
        assert table["is_branch_to_branch"].tolist() == [True, False, True]
        assert table["is_terminal_branch"].tolist() == [False, True, False]
    else:
        assert table["is_branch_to_branch"].tolist() == [True, True, True]
        assert table["is_terminal_branch"].tolist() == [False, False, False]


def test_stream_vertices_returns_none_without_degree_column(tmp_path):
    _, _, vertex_csv = _write_express_fixtures(tmp_path, with_degrees=False)
    assert stream_vertices(vertex_csv) is None


def test_stream_vertex_stats_from_edges(tmp_path):
    _, edge_csv, _ = _write_express_fixtures(tmp_path, with_degrees=False)

    stats = stream_vertex_stats_from_edges(edge_csv)

    # degrees: node0=3, node1=3, node2=2, node3=2
    assert stats["num_vertices"] == 4
    assert stats["num_endpoints"] == 0
    assert stats["num_branch_points"] == 2
    assert stats["branch_point_degree_histogram"] == {3: 2}
    assert stats["degree_histogram"] == {2: 2, 3: 2}
