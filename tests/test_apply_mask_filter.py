import numpy as np

from pipeline_modules.segmentation.apply_mask_to_signal import (
    filter_elongated_components_chunk,
    filter_mask_volume,
)


def test_filter_drops_large_and_elongated_components():
    mask = np.zeros((32, 32, 32), dtype=np.uint8)
    mask[1:4, 1:4, 1:4] = 1  # 27 voxels, compact -> keep
    mask[10:28, 10:12, 10:12] = 1  # elongated bbox 18/2=9 -> drop
    mask[2:16, 16:30, 16:30] = 1  # 14^3 voxels -> drop by volume

    filtered, n_labels, n_aspect, n_volume = filter_elongated_components_chunk(
        mask,
        max_aspect_ratio=3.0,
        max_voxels=200,
    )
    assert n_labels == 3
    assert n_aspect == 1
    assert n_volume == 1
    assert int(filtered.sum()) == 27


def test_filter_mask_volume_keeps_compact_cell_and_uint8_scale():
    mask = np.zeros((16, 16, 16), dtype=np.uint8)
    mask[6:9, 6:9, 6:9] = 255
    out, stats = filter_mask_volume(mask, max_aspect_ratio=3.0, max_voxels=200)
    assert stats["components_kept"] == 1
    assert stats["voxels_after"] == 27
    assert set(np.unique(out).tolist()) <= {0, 255}
