from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline_modules.tubule_reconstruction.kimimaro_reconstruction import summarize_branch_table


def test_summarize_branch_table_uses_requested_fields_only():
    branch_table = pd.DataFrame(
        {
            "skeleton_id": [1, 1, 1],
            "start_node": [0, 1, 2],
            "end_node": [1, 2, 3],
            "start_degree": [3, 3, 1],
            "end_degree": [3, 3, 3],
            "is_branch_to_branch": [True, True, False],
            "branch_length_um": [10.0, 20.0, 5.0],
            "tortuosity": [1.0, 1.5, np.nan],
            "branch_depth": [1.0, 2.0, np.nan],
        }
    )

    summary = summarize_branch_table(branch_table)

    assert summary["num_branch_points"] == 4
    assert summary["branch_point_path_length_sum_um"] == 30.0
    assert summary["branch_point_path_length_mean_um"] == 15.0
    assert np.isclose(summary["branch_point_path_length_sd_um"], np.sqrt(50.0))
    assert summary["mean_tortuosity"] == 1.25
    assert summary["mean_branch_depth"] == 1.5

    removed_fields = {
        "num_vertices",
        "num_edges",
        "num_end_points",
        "total_branch_length_um",
        "mean_branch_length_um",
        "mean_skeleton_radius_um",
        "max_branch_depth",
        "num_terminal_branches",
    }
    assert removed_fields.isdisjoint(summary)
