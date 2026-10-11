from __future__ import annotations

import numpy as np
import pytest

from pipeline_modules.segmentation.cfos_mask_postprocess import (
    postprocess_cfos_mask_3d,
    select_keep_labels_3d,
)
from pipeline_modules.segmentation.cfos_unet_inference import (
    _percentile_bounds_from_histogram,
)
from pipeline_modules.segmentation.cfos_unet_model import normalize_volume
from pipeline_modules.utils.zarr_io import create_output_zarr


def test_percentile_bounds_from_histogram():
    counts = np.zeros(256, dtype=np.int64)
    counts[10] = 10
    counts[20] = 10
    counts[30] = 10
    counts[40] = 70
    low, high = _percentile_bounds_from_histogram(counts, 1.0, 99.5)
    # cumulative: 10@10, 20@20, 30@30, 100@40
    assert low == 10.0  # 1% of 100 = 1 -> first bin with cum >= 1
    assert high == 40.0  # 99.5% of 100 = 99.5 -> bin with cum >= 99.5


def test_percentile_bounds_empty_counts():
    assert _percentile_bounds_from_histogram(np.zeros(4, dtype=np.int64), 1.0, 99.5) == (0.0, 1.0)


def test_normalize_volume_with_global_bounds_matches_global_percentiles():
    volume_a = np.array([[0.0, 5.0], [10.0, 15.0]], dtype=np.float32)
    volume_b = np.array([[100.0, 105.0], [110.0, 1000.0]], dtype=np.float32)
    # Per-chunk bounds saturate both chunks' max to 1.0, hiding the 10x
    # brightness difference; global bounds preserve it.
    per_chunk_a = normalize_volume(volume_a, 1.0, 99.5)
    per_chunk_b = normalize_volume(volume_b, 1.0, 99.5)
    assert per_chunk_a[1, 1] == pytest.approx(1.0, abs=1e-6)
    assert per_chunk_b[1, 1] == pytest.approx(1.0, abs=1e-6)
    global_a = normalize_volume(volume_a, low=0.0, high=500.0)
    global_b = normalize_volume(volume_b, low=0.0, high=500.0)
    assert global_a[1, 1] == pytest.approx(0.03, abs=1e-6)  # 15/500
    assert global_b[0, 0] == pytest.approx(0.2, abs=1e-6)  # 100/500
    assert global_b[1, 1] == pytest.approx(1.0, abs=1e-6)  # 1000 clips at the global high


def test_select_keep_labels_filters_single_slice_and_edge():
    stats = {
        "voxel_counts": np.array([0, 50, 50, 50]),
        "bounding_boxes": [
            None,
            (slice(0, 5), slice(0, 5), slice(0, 5)),
            (slice(0, 5), slice(0, 5), slice(0, 5)),
            (slice(0, 5), slice(0, 5), slice(0, 5)),
        ],
    }
    # factor 4: single-slice threshold 200/16 = 12.5 downsampled voxels.
    max_slice_counts = np.array([0, 4, 20, 4], dtype=np.int64)  # label 2 has a big slice
    edge_hits = np.array([0, 0, 0, 1], dtype=np.int64)  # label 3 fails the edge-shrink exclusion
    keep, removed_volume, removed_extent, removed_slice, removed_edge = select_keep_labels_3d(
        None,
        stats,
        max_voxels=0,
        min_voxels=0,
        max_extent_ratio=0.0,
        downsample_factor=4,
        max_single_slice_voxels=200,
        max_slice_counts_ds=max_slice_counts,
        edge_hits=edge_hits,
    )
    assert keep == {1}
    assert (removed_volume, removed_extent, removed_slice, removed_edge) == (0, 0, 1, 1)


def _make_zarr(path, volume, chunks=(4, 4, 4)):
    _, arr = create_output_zarr(path, volume.shape, chunks, str(volume.dtype))
    arr[:, :, :] = volume
    return arr


def test_postprocess_applies_edge_and_single_slice_rules(tmp_path=None):
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shape = (16, 64, 64)
        signal = np.zeros(shape, dtype=np.uint16)
        mask = np.zeros(shape, dtype=np.uint8)
        label = np.zeros(shape, dtype=np.uint32)

        # Brain block: z 4..15, y/x 8..55. On the factor-4 cache grid
        # (4, 16, 16) this leaves empty z-row 0 and empty y/x border rows,
        # so the shrunk surface mesh separates inside from outside blobs.
        label[4:16, 8:56, 8:56] = 500

        mask[8, 32, 32] = 1  # brain-center blob -> kept
        signal[8, 32, 32] = 900
        mask[2, 32, 32] = 1  # outside the brain block -> dropped by edge shrink
        signal[2, 32, 32] = 800
        mask[8, 20:28, 20:28] = 1  # flat slab: single slice of 64 voxels -> dropped by slice rule
        signal[8, 20:28, 20:28] = 700

        _make_zarr(tmp / "signal.zarr", signal)
        _make_zarr(tmp / "mask_raw.zarr", mask)
        _make_zarr(tmp / "label.zarr", label)

        result = postprocess_cfos_mask_3d(
            signal_zarr=tmp / "signal.zarr",
            mask_zarr=tmp / "mask_raw.zarr",
            output_mask_zarr=tmp / "mask_filtered.zarr",
            masked_signal_zarr=None,
            masked_tiff_dir=None,
            max_voxels=0,
            min_voxels=0,
            max_extent_ratio=0.0,
            downsample_factor=4,
            edge_shrink_factor=0.5,
            max_single_slice_voxels=32,
            label_zarr=tmp / "label.zarr",
        )
        assert result["labels_kept_3d"] == 1
        assert result["labels_removed_single_slice_3d"] == 1
        assert result["labels_removed_edge_3d"] == 1
        # fallback path must persist the brain-mask cache for the next run
        assert (tmp / "label_ds_brain_mask.zarr").exists()

        from pipeline_modules.segmentation.zarr_utils import open_zarr_dataset

        filtered = np.asarray(open_zarr_dataset(tmp / "mask_filtered.zarr"))
        assert int(filtered.sum()) == 1
        assert filtered[8, 32, 32] == 1
