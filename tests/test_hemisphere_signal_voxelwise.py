from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_modules.registration.region_signal_analysis_zarr_graph import (
    aggregate_final_region_stats,
)


def _synthetic_inputs():
    """One connected vessel-like object spanning BOTH hemispheres.

    Voxelwise truth: left signal 60, right signal 60 (balanced). The old
    object-collapse attribution assigned ALL signal voxels to the majority
    hemisphere, which is the bug this guards against.
    """
    n_components = 2  # component 0 (padding) + the vessel object
    parent = np.arange(n_components, dtype=np.int64)
    root_sizes = np.array([0, 120], dtype=np.int64)  # the single vessel object
    manifest_payload = {
        "blocks": [],
        "total_region_voxels": {"101": 200, "102": 200},
        "total_region_voxels_by_hemisphere": {"101:1": 100, "101:2": 100, "102:1": 100, "102:2": 100},
        # Voxelwise pass-1 counts: signal split evenly across hemispheres.
        "region_signal_voxels_by_hemisphere": {"101:1": 60, "101:2": 60, "102:1": 60, "102:2": 60},
        "region_sum_intensity_by_hemisphere": {"101:1": 60000.0, "101:2": 60000.0},
        # Object-collapse (wrong for vessels): everything on hemisphere 2.
        "region_signal_counts_by_hemisphere": {"102:2": 1},
    }
    # collapse result: the single object's voxels all attributed to region 102, hemi 2.
    collapsed_stats = {
        "region_signal_voxels": {"102": 120},
        "region_signal_counts": {"102": 1},
        "region_sum_intensity": {"102": 120000.0},
        "region_signal_voxels_by_hemisphere": {"102:2": 120},
        "region_signal_counts_by_hemisphere": {"102:2": 1},
        "region_sum_intensity_by_hemisphere": {"102:2": 120000.0},
    }
    return manifest_payload, parent, root_sizes, collapsed_stats


def test_hemisphere_signal_voxels_are_voxelwise_not_object_assigned():
    manifest_payload, parent, root_sizes, collapsed = _synthetic_inputs()
    stats = aggregate_final_region_stats(manifest_payload, parent, root_sizes, min_voxels=10)

    # Region 101 exists only in the voxelwise totals (it has no kept object):
    # its per-hemisphere signal must still be reported, split 60/60.
    assert stats["region_signal_voxels_by_hemisphere"][(101, 1)] == 60
    assert stats["region_signal_voxels_by_hemisphere"][(101, 2)] == 60
    # Region 102: voxelwise 60/60 must win over the collapse's 0/120.
    assert stats["region_signal_voxels_by_hemisphere"][(102, 1)] == 60
    assert stats["region_signal_voxels_by_hemisphere"][(102, 2)] == 60
    # Intensity stays voxelwise too.
    assert stats["region_sum_intensity_by_hemisphere"][(101, 1)] == 60000.0
    # Object counts remain collapse-based (no blocks here -> none produced).
    assert not stats.get("region_signal_counts_by_hemisphere")
    # Whole-brain columns come from the (block-artifact) collapse; with no
    # blocks in this synthetic manifest they are empty by construction.
    assert stats["region_signal_voxels"] == {}


def test_hemisphere_columns_absent_without_hemisphere_data():
    manifest_payload, parent, root_sizes, collapsed = _synthetic_inputs()
    for key in (
        "total_region_voxels_by_hemisphere",
        "region_signal_voxels_by_hemisphere",
        "region_sum_intensity_by_hemisphere",
    ):
        manifest_payload.pop(key)
    stats = aggregate_final_region_stats(manifest_payload, parent, root_sizes, min_voxels=10)
    assert "region_signal_voxels_by_hemisphere" not in stats
    assert "total_region_voxels_by_hemisphere" not in stats
