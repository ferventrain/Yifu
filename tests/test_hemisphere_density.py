"""Unit tests for hemisphere-aware region density aggregation."""

from __future__ import annotations

import numpy as np

from pipeline_modules.registration.region_signal_analysis_zarr_graph import (
    LEFT_HEMISPHERE_ID,
    RIGHT_HEMISPHERE_ID,
    aggregate_final_region_stats,
    aggregate_region_totals,
    aggregate_signal_by_hemisphere,
    choose_majority_hemisphere,
    choose_majority_region,
    compute_block_artifacts,
    flatten_region_rows,
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


def test_hemisphere_rows_include_all_regions():
    region_tree = {
        "id": 1,
        "name": "root,root",
        "st_level": 0,
        "children": [
            {
                "id": 10,
                "name": "paired child,PAIR",
                "st_level": 1,
                "children": [],
            },
            {
                "id": 20,
                "name": "midline child,MID",
                "st_level": 1,
                "children": [],
            },
        ],
    }
    direct_stats = {
        "total_region_voxels": {1: 0, 10: 10, 20: 90},
        "region_signal_voxels": {10: 4, 20: 45},
        "region_signal_counts": {10: 2, 20: 9},
        "region_sum_intensity": {10: 40.0, 20: 900.0},
        "total_region_voxels_by_hemisphere": {
            (10, int(LEFT_HEMISPHERE_ID)): 6,
            (10, int(RIGHT_HEMISPHERE_ID)): 4,
            (20, int(LEFT_HEMISPHERE_ID)): 90,
        },
        "region_signal_voxels_by_hemisphere": {
            (10, int(LEFT_HEMISPHERE_ID)): 2,
            (10, int(RIGHT_HEMISPHERE_ID)): 2,
            (20, int(LEFT_HEMISPHERE_ID)): 45,
        },
        "region_signal_counts_by_hemisphere": {
            (10, int(LEFT_HEMISPHERE_ID)): 1,
            (10, int(RIGHT_HEMISPHERE_ID)): 1,
            (20, int(LEFT_HEMISPHERE_ID)): 9,
        },
        "region_sum_intensity_by_hemisphere": {
            (10, int(LEFT_HEMISPHERE_ID)): 15.0,
            (10, int(RIGHT_HEMISPHERE_ID)): 25.0,
            (20, int(LEFT_HEMISPHERE_ID)): 900.0,
        },
    }

    rows = flatten_region_rows(region_tree, direct_stats)
    rows_by_name = {row["Name"]: row for row in rows}

    root_row = rows_by_name["root,root"]
    assert root_row["Total Voxels"] == 100
    assert root_row["Left Total Voxels"] == 96
    assert root_row["Right Total Voxels"] == 4

    paired_row = rows_by_name["paired child,PAIR"]
    assert paired_row["Left Total Voxels"] == 6
    assert paired_row["Right Total Voxels"] == 4

    midline_row = rows_by_name["midline child,MID"]
    assert midline_row["Total Voxels"] == 90
    assert midline_row["Left Total Voxels"] == 90
    assert midline_row["Right Total Voxels"] == 0


def test_majority_hemisphere_prefers_left_on_ties():
    assert choose_majority_hemisphere(3, 3) == int(LEFT_HEMISPHERE_ID)
    assert choose_majority_hemisphere(4, 3) == int(LEFT_HEMISPHERE_ID)
    assert choose_majority_hemisphere(2, 5) == int(RIGHT_HEMISPHERE_ID)


def test_cross_midline_component_counts_once_in_majority_hemisphere(tmp_path):
    mask_chunk = np.array([[[1, 1, 1, 1, 1]]], dtype=np.uint8)
    label_chunk = np.array([[[10, 10, 10, 10, 10]]], dtype=np.int32)
    signal_chunk = np.array([[[5, 6, 7, 8, 9]]], dtype=np.float32)
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

    assert stats["region_signal_counts"][10] == 1
    assert stats["region_signal_counts_by_hemisphere"][(10, int(LEFT_HEMISPHERE_ID))] == 1
    assert (10, int(RIGHT_HEMISPHERE_ID)) not in stats["region_signal_counts_by_hemisphere"]
    assert stats["region_signal_voxels_by_hemisphere"][(10, int(LEFT_HEMISPHERE_ID))] == 5
    assert stats["region_sum_intensity_by_hemisphere"][(10, int(LEFT_HEMISPHERE_ID))] == 35.0
    assert sum(stats["region_signal_counts_by_hemisphere"].values()) == stats["region_signal_counts"][10]


def test_cross_region_component_assigned_to_majority_region(tmp_path):
    mask_chunk = np.array([[[1, 1, 1, 1, 1]]], dtype=np.uint8)
    label_chunk = np.array([[[10, 10, 10, 20, 20]]], dtype=np.int32)
    signal_chunk = np.array([[[1, 2, 3, 4, 5]]], dtype=np.float32)

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
    )
    artifact_path = tmp_path / "block.npz"
    save_block_artifact(artifact_path, artifact)

    parent = np.arange(artifact["next_component_id"], dtype=np.int64)
    root_sizes = np.zeros(artifact["next_component_id"], dtype=np.int64)
    root_sizes[artifact["component_ids"]] = artifact["component_sizes"]
    manifest_payload = {
        "blocks": [{"artifact_path": str(artifact_path)}],
        "total_region_voxels": {"10": 3, "20": 2},
    }

    stats = aggregate_final_region_stats(
        manifest_payload=manifest_payload,
        parent=parent,
        root_sizes=root_sizes,
        min_voxels=1,
    )

    assert stats["region_signal_counts"][10] == 1
    assert 20 not in stats["region_signal_counts"]
    assert stats["region_signal_voxels"][10] == 5
    assert stats["region_sum_intensity"][10] == 15.0
    assert sum(stats["region_signal_counts"].values()) == 1


def test_majority_region_prefers_smaller_id_on_ties():
    assert choose_majority_region({10: 3, 20: 3}) == 10
    assert choose_majority_region({10: 4, 20: 3}) == 10
    assert choose_majority_region({10: 2, 20: 5}) == 20
