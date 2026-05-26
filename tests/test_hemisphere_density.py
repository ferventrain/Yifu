"""Unit tests for hemisphere-aware region density aggregation."""

from __future__ import annotations

import numpy as np

from pipeline_modules.registration.region_signal_analysis_zarr_graph import (
    LEFT_HEMISPHERE_ID,
    RIGHT_HEMISPHERE_ID,
    aggregate_final_region_stats,
    aggregate_region_totals,
    aggregate_signal_by_hemisphere,
    compute_block_artifacts,
    save_block_artifact,
)


def test_aggregate_region_totals_respects_hemisphere_zarr_labels():
    label_chunk = np.array([[[0, 10, 10, 20]]], dtype=np.int32)
    hemisphere_chunk = np.array([[[0, 1, 1, 2]]], dtype=np.uint8)
    total_region_voxels: dict[int, int] = {}
    total_region_voxels_by_hemisphere: dict[tuple[int, int], int] = {}

    aggregate_region_totals(
        total_region_voxels,
        total_region_voxels_by_hemisphere,
        label_chunk,
        hemisphere_chunk,
    )

    assert total_region_voxels == {10: 2, 20: 1}
    assert total_region_voxels_by_hemisphere == {
        (10, int(LEFT_HEMISPHERE_ID)): 2,
        (20, int(RIGHT_HEMISPHERE_ID)): 1,
    }


def test_aggregate_signal_by_hemisphere_respects_hemisphere_zarr_labels():
    mask_chunk = np.array([[[0, 1, 1, 1]]], dtype=np.uint8)
    label_chunk = np.array([[[0, 10, 10, 20]]], dtype=np.int32)
    signal_chunk = np.array([[[0, 100, 200, 400]]], dtype=np.float32)
    hemisphere_chunk = np.array([[[0, 1, 1, 2]]], dtype=np.uint8)

    region_signal_voxels_by_hemisphere: dict[tuple[int, int], int] = {}
    region_sum_intensity_by_hemisphere: dict[tuple[int, int], float] = {}

    aggregate_signal_by_hemisphere(
        region_signal_voxels_by_hemisphere,
        region_sum_intensity_by_hemisphere,
        mask_chunk,
        label_chunk,
        signal_chunk,
        foreground_mode="nonzero",
        foreground_label=1,
        hemisphere_chunk=hemisphere_chunk,
    )

    assert region_signal_voxels_by_hemisphere == {
        (10, int(LEFT_HEMISPHERE_ID)): 2,
        (20, int(RIGHT_HEMISPHERE_ID)): 1,
    }
    assert region_sum_intensity_by_hemisphere == {
        (10, int(LEFT_HEMISPHERE_ID)): 300.0,
        (20, int(RIGHT_HEMISPHERE_ID)): 400.0,
    }


def test_final_region_stats_counts_multiple_components_per_hemisphere(tmp_path):
    mask_chunk = np.array([[[1, 0, 1, 0, 1]]], dtype=np.uint8)
    label_chunk = np.array([[[10, 10, 10, 10, 10]]], dtype=np.int32)
    signal_chunk = np.array([[[5, 0, 7, 0, 11]]], dtype=np.float32)
    hemisphere_chunk = np.array([[[1, 1, 1, 2, 2]]], dtype=np.uint8)

    artifact = compute_block_artifacts(
        mask_chunk=mask_chunk,
        label_chunk=label_chunk,
        signal_chunk=signal_chunk,
        start=(0, 0, 0),
        foreground_mode="nonzero",
        foreground_label=1,
        next_component_id=1,
        boundary_mask_cache={},
        volume_shape=mask_chunk.shape,
        hemisphere_chunk=hemisphere_chunk,
    )
    artifact_path = tmp_path / "block.npz"
    save_block_artifact(artifact_path, artifact)

    parent = np.arange(artifact["next_component_id"], dtype=np.int64)
    root_sizes = np.zeros(artifact["next_component_id"], dtype=np.int64)
    root_sizes[artifact["component_ids"]] = artifact["component_sizes"]
    manifest_payload = {
        "blocks": [{"artifact_path": str(artifact_path)}],
        "total_region_voxels": {"10": 5},
        "total_region_voxels_by_hemisphere": {
            f"10:{int(LEFT_HEMISPHERE_ID)}": 3,
            f"10:{int(RIGHT_HEMISPHERE_ID)}": 2,
        },
    }

    stats = aggregate_final_region_stats(
        manifest_payload=manifest_payload,
        parent=parent,
        root_sizes=root_sizes,
        min_voxels=1,
    )

    assert stats["region_signal_counts"][10] == 3
    assert stats["region_signal_counts_by_hemisphere"][(10, int(LEFT_HEMISPHERE_ID))] == 2
    assert stats["region_signal_counts_by_hemisphere"][(10, int(RIGHT_HEMISPHERE_ID))] == 1
    assert stats["region_signal_voxels_by_hemisphere"][(10, int(LEFT_HEMISPHERE_ID))] == 2
    assert stats["region_sum_intensity_by_hemisphere"][(10, int(LEFT_HEMISPHERE_ID))] == 12.0
