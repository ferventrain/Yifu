from __future__ import annotations

import numpy as np

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
    assert per_chunk_a[1, 1] == per_chunk_b[1, 1] == 1.0
    global_a = normalize_volume(volume_a, low=0.0, high=500.0)
    global_b = normalize_volume(volume_b, low=0.0, high=500.0)
    assert global_a[1, 1] == 0.03  # 15/500
    assert global_b[0, 0] == 0.2  # 100/500
    assert global_b[1, 1] == 1.0  # 1000 clips at the global high


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
    shell_hits = np.array([0, 0, 0, 3], dtype=np.int64)  # label 3 touches the edge shell
    keep, removed_volume, removed_extent, removed_slice, removed_edge = select_keep_labels_3d(
        None,
        stats,
        max_voxels=0,
        min_voxels=0,
        max_extent_ratio=0.0,
        downsample_factor=4,
        max_single_slice_voxels=200,
        max_slice_counts_ds=max_slice_counts,
        shell_hits=shell_hits,
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
        shape = (8, 12, 12)
        signal = np.zeros(shape, dtype=np.uint16)
        mask = np.zeros(shape, dtype=np.uint8)
        label = np.zeros(shape, dtype=np.uint32)

        label[1:8, 1:11, 1:11] = 500  # brain block; erosion k=2 -> core z3..5, y/x 3..8

        mask[5, 3, 3] = 1  # good cell inside the core -> kept
        signal[5, 3, 3] = 900
        mask[1, 6, 6] = 1  # inside brain but in the 2px surface shell -> dropped
        signal[1, 6, 6] = 800
        mask[3, 5:8, 5:8] = 1  # flat slab: single slice of 9 voxels -> dropped by slice rule
        signal[3, 5:8, 5:8] = 700

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
            downsample_factor=1,
            exclude_edge_px=2,
            max_single_slice_voxels=8,
            label_zarr=tmp / "label.zarr",
        )
        assert result["labels_kept_3d"] == 1
        assert result["labels_removed_single_slice_3d"] == 1
        assert result["labels_removed_edge_3d"] == 1

        from pipeline_modules.segmentation.zarr_utils import open_zarr_dataset

        filtered = np.asarray(open_zarr_dataset(tmp / "mask_filtered.zarr"))
        assert int(filtered.sum()) == 1
        assert filtered[5, 3, 3] == 1
