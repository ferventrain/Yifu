from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest

from pipeline_modules.preprocessing.zarr_to_registration_nii import (
    convert_zarr_to_registration_nii,
    parse_resolution_xyz,
)
from pipeline_modules.utils.zarr_io import create_output_zarr


def _write_input_zarr(path, volume: np.ndarray) -> None:
    create_output_zarr(path, shape=volume.shape, chunks=(2, 8, 8), dtype=volume.dtype, data=volume)


def _smooth_volume(shape, offset: float = 1000.0, scale: float = 500.0) -> np.ndarray:
    nz, ny, nx = shape
    z, y, x = np.mgrid[0:nz, 0:ny, 0:nx]
    volume = offset + scale * np.sin(2 * np.pi * z / max(nz, 1)) * np.cos(2 * np.pi * y / max(ny, 1))
    return np.clip(volume + 0.1 * x, 0, 65535).astype(np.uint16)


def test_parse_resolution_xyz_accepts_triplet():
    assert parse_resolution_xyz("14.384,14.384,16") == (14.384, 14.384, 16.0)


def test_parse_resolution_xyz_rejects_bad_input():
    with pytest.raises(ValueError):
        parse_resolution_xyz("1,2")
    with pytest.raises(ValueError):
        parse_resolution_xyz("1,0,3")


def test_convert_rescales_and_matches_nifti_convention(tmp_path):
    volume = _smooth_volume((16, 64, 48))
    input_zarr = tmp_path / "ch0_L2.zarr"
    _write_input_zarr(input_zarr, volume)

    output_nii = tmp_path / "ch0_downsample" / "volume.nii.gz"
    result = convert_zarr_to_registration_nii(
        input_zarr,
        output_nii,
        input_resolution_xyz=(12.0, 12.0, 16.0),
        target_resolution_xyz=(24.0, 24.0, 32.0),
    )

    assert result["input_shape_zyx"] == [16, 64, 48]
    assert result["output_shape_zyx"] == [8, 32, 24]

    image = nib.load(str(output_nii))
    # Convention matches the proven dbdb36 registration: (z, y, x) transposed
    # to (x, y, z) with a UNIT affine — physical spacing must NOT be baked
    # into the header (the atlas side is read at unit spacing).
    assert image.shape == (24, 32, 8)
    assert image.get_data_dtype() == np.uint16
    assert np.allclose(image.affine, np.eye(4))

    data = np.asanyarray(image.dataobj)
    assert data.min() >= 0 and data.max() <= 65535
    # Mean intensity is preserved by the rescale.
    assert abs(float(data.mean()) - float(volume.astype(np.float64).mean())) < 120.0

    original = json.loads((tmp_path / "ch0_downsample" / "original_shape.json").read_text(encoding="utf-8"))
    assert original["original_shape"] == [16, 64, 48]


def test_convert_rejects_missing_input_and_existing_output(tmp_path):
    volume = _smooth_volume((8, 16, 16))
    input_zarr = tmp_path / "in.zarr"
    _write_input_zarr(input_zarr, volume)
    output_nii = tmp_path / "volume.nii.gz"

    with pytest.raises(FileNotFoundError):
        convert_zarr_to_registration_nii(
            tmp_path / "missing.zarr",
            output_nii,
            input_resolution_xyz=(12.0, 12.0, 16.0),
            target_resolution_xyz=(24.0, 24.0, 32.0),
        )

    convert_zarr_to_registration_nii(
        input_zarr,
        output_nii,
        input_resolution_xyz=(12.0, 12.0, 16.0),
        target_resolution_xyz=(24.0, 24.0, 32.0),
    )
    with pytest.raises(FileExistsError):
        convert_zarr_to_registration_nii(
            input_zarr,
            output_nii,
            input_resolution_xyz=(12.0, 12.0, 16.0),
            target_resolution_xyz=(24.0, 24.0, 32.0),
        )
