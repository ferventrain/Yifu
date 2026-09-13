from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_modules.tubule_reconstruction.vessel_analysis_report import (
    is_edt_polyline_edges,
    spill_degree_stats,
    stream_edt_branches,
)


def test_stream_edt_branches_computes_tortuosity_and_skips_stitch(tmp_path: Path):
    csv_path = tmp_path / "skeleton_edges.csv"
    pd.DataFrame(
        [
            {
                "chunk_index": "0.0.0",
                "branch_id": 0,
                "source_z_um": 0.0,
                "source_y_um": 0.0,
                "source_x_um": 0.0,
                "target_z_um": 0.0,
                "target_y_um": 0.0,
                "target_x_um": 10.0,
                "length_um": 12.0,
                "num_points": 4,
                "mean_radius_um": 2.0,
                "is_stitch": False,
            },
            {
                "chunk_index": "0.0.0",
                "branch_id": -1,
                "source_z_um": 0.0,
                "source_y_um": 0.0,
                "source_x_um": 10.0,
                "target_z_um": 0.0,
                "target_y_um": 0.0,
                "target_x_um": 12.0,
                "length_um": 2.0,
                "num_points": 2,
                "mean_radius_um": np.nan,
                "is_stitch": True,
            },
        ]
    ).to_csv(csv_path, index=False)

    assert is_edt_polyline_edges(csv_path)
    result = stream_edt_branches(csv_path)
    stats = result["edge_stats"]
    table = result["branch_table"]
    assert stats["num_edges"] == 2
    assert stats["num_stitch_edges"] == 1
    assert stats["stitch_length_um"] == 2.0
    assert stats["total_vessel_length_um"] == 14.0
    assert len(table) == 1
    assert table.loc[0, "branch_length_um"] == 12.0
    assert np.isclose(table.loc[0, "tortuosity"], 1.2)
    assert bool(table.loc[0, "is_branch_to_branch"])


def test_spill_degree_stats(tmp_path: Path):
    import pickle

    spill = tmp_path / "_chunk_spill"
    spill.mkdir()
    payload = {
        "chunk_index": (0, 0, 0),
        "degrees": np.array([1, 2, 2, 3, 1], dtype=np.int32),
    }
    with (spill / "z00000_y00000_x00000.pkl").open("wb") as handle:
        pickle.dump(payload, handle)
    stats = spill_degree_stats(spill)
    assert stats["num_vertices"] == 5
    assert stats["num_endpoints"] == 2
    assert stats["num_branch_points"] == 1
    assert stats["branch_point_degree_histogram"][3] == 1
