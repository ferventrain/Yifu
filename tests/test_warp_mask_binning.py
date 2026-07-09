from __future__ import annotations

import numpy as np


def test_bin_mask_block_array_and_merge() -> None:
    from pipeline_modules.visualization.warp_mask_zarr_to_atlas_points import (
        _bin_mask_block_array,
        _merge_bin_partials,
        _volume_from_bin_counts,
    )

    block_a = np.zeros((2, 2, 2), dtype=np.uint8)
    block_a[0, 0, 0] = 1
    block_b = np.zeros((2, 2, 2), dtype=np.uint8)
    block_b[1, 1, 1] = 1

    partial_a = _bin_mask_block_array(
        block_a,
        z_start=0,
        y_start=0,
        x_start=0,
        resolution_xyz=(2.0, 2.0, 2.0),
        target_resolution_xyz=(4.0, 4.0, 4.0),
        grid_shape_zyx=(2, 2, 2),
        foreground_mode="nonzero",
        foreground_label=1,
    )
    partial_b = _bin_mask_block_array(
        block_b,
        z_start=2,
        y_start=2,
        x_start=2,
        resolution_xyz=(2.0, 2.0, 2.0),
        target_resolution_xyz=(4.0, 4.0, 4.0),
        grid_shape_zyx=(2, 2, 2),
        foreground_mode="nonzero",
        foreground_label=1,
    )

    foreground, clipped, merged = _merge_bin_partials([partial_a, partial_b])
    volume, occupied = _volume_from_bin_counts(
        merged,
        output_shape_zyx=(2, 2, 2),
        min_voxels_per_point=1,
        volume_mode="binary",
    )

    assert foreground == 2
    assert clipped == 0
    assert occupied == 2
    assert int(volume.sum()) == 2


def test_resolve_bin_workers_auto_and_cap() -> None:
    from pipeline_modules.visualization.warp_mask_zarr_to_atlas_points import resolve_bin_workers

    assert resolve_bin_workers(1) == 1
    assert resolve_bin_workers(4) == 4
    assert resolve_bin_workers(0) >= 1
