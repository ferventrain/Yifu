from pathlib import Path

import numpy as np
import zarr

from pipeline_modules.registration.atlas_label_to_hemisphere import (
    convert_atlas_label_to_hemisphere,
)
from pipeline_modules.utils.zarr_io import create_output_zarr, open_zarr_array


def _root_attrs(path: Path) -> dict:
    return dict(zarr.open(str(path), mode="r").attrs)


def _write_label_zarr(path: Path, volume: np.ndarray) -> None:
    create_output_zarr(path, shape=volume.shape, chunks=(2, 4, 4), dtype=volume.dtype, data=volume)


def test_split_follows_label_midline_not_volume_midplane(tmp_path):
    """Brain pushed to high x: a volume-midplane split would miss it entirely."""
    volume = np.zeros((8, 16, 32), dtype=np.uint16)
    volume[2:6, 4:12, 20:30] = 5
    input_zarr = tmp_path / "label.zarr"
    _write_label_zarr(input_zarr, volume)

    output_zarr = tmp_path / "hemi.zarr"
    convert_atlas_label_to_hemisphere(input_zarr, output_zarr, dataset_name="0")

    hemi = np.asarray(open_zarr_array(output_zarr))
    # label x extent 20..29 -> split_x = 20 + (10 // 2) = 25
    mask = volume > 0
    assert (hemi[mask] == 1).sum() > 0
    assert (hemi[mask] == 2).sum() > 0
    xs = np.broadcast_to(np.arange(volume.shape[2]), volume.shape)
    assert np.all(xs[(hemi == 1) & mask] < 25)
    assert np.all(xs[(hemi == 2) & mask] >= 25)
    attrs = _root_attrs(output_zarr)
    assert attrs["split_x"] == 25
    assert attrs["label_x_extent"] == [20, 29]


def test_empty_label_falls_back_to_volume_midplane(tmp_path):
    input_zarr = tmp_path / "label.zarr"
    _write_label_zarr(input_zarr, np.zeros((8, 16, 32), dtype=np.uint16))

    output_zarr = tmp_path / "hemi.zarr"
    convert_atlas_label_to_hemisphere(input_zarr, output_zarr, dataset_name="0")

    attrs = _root_attrs(output_zarr)
    assert attrs["split_x"] == 16
    assert attrs["label_x_extent"] is None


def test_centered_label_matches_volume_midplane(tmp_path):
    volume = np.zeros((8, 16, 32), dtype=np.uint16)
    volume[2:6, 4:12, 4:28] = 3
    input_zarr = tmp_path / "label.zarr"
    _write_label_zarr(input_zarr, volume)

    output_zarr = tmp_path / "hemi.zarr"
    convert_atlas_label_to_hemisphere(input_zarr, output_zarr, dataset_name="0")

    hemi = np.asarray(open_zarr_array(output_zarr))
    # extent 4..27 -> split_x = 4 + 12 = 16 = volume midplane
    assert np.all(hemi[volume > 0] > 0)
    assert (hemi == 1).sum() == (volume[:, :, 4:16] > 0).sum()
    assert (hemi == 2).sum() == (volume[:, :, 16:28] > 0).sum()
