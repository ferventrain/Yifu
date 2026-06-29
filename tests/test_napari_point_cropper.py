from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipeline_modules.visualization.napari_point_cropper import (
    _all_points,
    _save_crop,
    default_crop_output_dir,
    image_layer_stem,
    crop_array_from_point,
    crop_bounds_from_point,
)


def test_crop_bounds_anchor_point_as_start_corner():
    start, stop = crop_bounds_from_point((2.2, 3.4, 4.5), (20, 30, 40), (5, 6, 7))
    assert start == (2, 3, 4)
    assert stop == (7, 9, 11)


def test_crop_array_from_point_returns_expected_block():
    data = np.arange(10 * 12 * 14, dtype=np.uint16).reshape(10, 12, 14)
    crop, metadata = crop_array_from_point(data, (1, 2, 3), (4, 5, 6))
    np.testing.assert_array_equal(crop, data[1:5, 2:7, 3:9])
    assert metadata["start_zyx"] == [1, 2, 3]
    assert metadata["stop_zyx"] == [5, 7, 9]


def test_crop_bounds_rejects_out_of_bounds_crop():
    with pytest.raises(ValueError, match="exceeds image shape"):
        crop_bounds_from_point((8, 8, 8), (10, 10, 10), (4, 4, 4))


def test_output_defaults_use_zarr_parent_and_image_stem():
    layer = SimpleNamespace(name="ignored", source=SimpleNamespace(path=r"H:\sample\ch3.zarr"))
    assert default_crop_output_dir(layer).as_posix().endswith("H:/sample")
    assert image_layer_stem(layer) == "sample"


def test_zarr_dataset_source_uses_sample_folder_name():
    layer = SimpleNamespace(name="ignored", source=SimpleNamespace(path=r"H:\sample\WT_2\ch3.zarr\0"))
    assert default_crop_output_dir(layer).as_posix().endswith("H:/sample/WT_2")
    assert image_layer_stem(layer) == "WT_2"


def test_save_crop_names_file_from_image_stem_and_coordinates(tmp_path):
    crop = np.zeros((2, 2, 2), dtype=np.uint8)
    path = _save_crop(crop, {"start_zyx": [1, 2, 3]}, tmp_path, "WT_2", "npy")
    assert path.name == "WT_2_z1_y2_x3.npy"
    assert path.with_suffix(".json").exists()


def test_all_points_returns_every_point():
    layer = SimpleNamespace(data=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    points = _all_points(layer)
    assert points.shape == (2, 3)
    np.testing.assert_array_equal(points, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64))
