from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest

from pipeline_modules.preprocessing.zarr_to_registration_nii import (
    convert_zarr_to_registration_nii,
    parse_resolution_xyz,
    read_flip_atlas,
)
from pipeline_modules.utils.zarr_io import create_output_zarr


def _write_input_zarr(path, volume: np.ndarray) -> None:
    create_output_zarr(path, shape=volume.shape, chunks=(2, 8, 8), dtype=volume.dtype, data=volume)


def _write_full_mask(path, shape) -> None:
    """DL brain mask covering the whole grid (all ones) so nothing is zeroed."""
    create_output_zarr(path, shape=shape, chunks=(2, 8, 8), dtype=np.uint8, data=np.ones(shape, np.uint8))


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
    mask_zarr = tmp_path / "ch0_brain_mask_dl.zarr"
    _write_full_mask(mask_zarr, (16, 64, 48))

    output_nii = tmp_path / "ch0_downsample" / "volume.nii.gz"
    result = convert_zarr_to_registration_nii(
        input_zarr,
        output_nii,
        input_resolution_xyz=(12.0, 12.0, 16.0),
        target_resolution_xyz=(24.0, 24.0, 32.0),
        brain_mask=mask_zarr,
        brain_mask_erode=0,
    )

    assert result["input_shape_zyx"] == [16, 64, 48]
    assert result["output_shape_zyx"] == [8, 32, 24]
    assert result["brain_mask"]["applied"] is True
    assert result["flip_atlas"] == [False, False, False]

    image = nib.load(str(output_nii))
    # Convention matches the proven dbdb36 registration: (z, y, x) transposed
    # to (x, y, z) with a UNIT affine — physical spacing must NOT be baked
    # into the header (the atlas side is read at unit spacing).
    assert image.shape == (24, 32, 8)
    assert image.get_data_dtype() == np.uint16
    assert np.allclose(image.affine, np.eye(4))

    data = np.asanyarray(image.dataobj)
    assert data.min() >= 0 and data.max() <= 65535
    # Mean intensity is preserved by the rescale (full mask, no zeroing).
    assert abs(float(data.mean()) - float(volume.astype(np.float64).mean())) < 120.0

    original = json.loads((tmp_path / "ch0_downsample" / "original_shape.json").read_text(encoding="utf-8"))
    assert original["original_shape"] == [16, 64, 48]
    assert original["brain_mask"]["applied"] is True
    # A full mask makes every (kept) voxel nonzero, so the mask NIfTI is all ones.
    assert (tmp_path / "ch0_downsample" / "brain_mask.nii.gz").exists()


def test_convert_brain_mask_zeroes_outside(tmp_path):
    """A central DL mask must zero the halo outside the brain and keep the
    masked tissue, writing a matching brain_mask.nii.gz."""
    nz, ny, nx = 12, 48, 40
    volume = np.full((nz, ny, nx), 2000, np.uint16)

    input_zarr = tmp_path / "in.zarr"
    _write_input_zarr(input_zarr, volume)
    # Mask: central block only.
    mask = np.zeros((nz, ny, nx), np.uint8)
    mask[3:9, 14:34, 10:30] = 1
    mask_zarr = tmp_path / "ch0_brain_mask_dl.zarr"
    create_output_zarr(mask_zarr, shape=mask.shape, chunks=(4, 16, 16), dtype=np.uint8, data=mask)

    out_dir = tmp_path / "ch0_downsample"
    output_nii = out_dir / "volume.nii.gz"

    result = convert_zarr_to_registration_nii(
        input_zarr,
        output_nii,
        input_resolution_xyz=(12.0, 12.0, 16.0),
        target_resolution_xyz=(24.0, 24.0, 32.0),
        brain_mask=mask_zarr,
        brain_mask_erode=0,
    )
    assert result["brain_mask"]["applied"] is True

    from scipy import ndimage

    out_shape = tuple(result["output_shape_zyx"])
    mapped = ndimage.zoom(mask, np.array(out_shape) / np.array(mask.shape), order=0, mode="nearest") > 0
    mapped_xyz = np.transpose(mapped, (2, 1, 0))

    data = np.asanyarray(nib.load(str(output_nii)).dataobj)
    assert (data[mapped_xyz] == 2000).all(), "voxels inside the mask survive"
    assert (data[~mapped_xyz] == 0).all(), "outside the mask is zeroed"

    mask_img = nib.load(str(out_dir / "brain_mask.nii.gz"))
    assert mask_img.shape == data.shape
    assert np.allclose(mask_img.affine, np.eye(4))


def test_convert_existing_output_overwrites(tmp_path):
    volume = _smooth_volume((16, 64, 48))
    input_zarr = tmp_path / "in.zarr"
    _write_input_zarr(input_zarr, volume)
    mask_zarr = tmp_path / "mask.zarr"
    _write_full_mask(mask_zarr, (16, 64, 48))
    output_nii = tmp_path / "volume.nii.gz"

    kwargs = dict(
        input_resolution_xyz=(12.0, 12.0, 16.0),
        target_resolution_xyz=(24.0, 24.0, 32.0),
        brain_mask=mask_zarr,
        brain_mask_erode=0,
    )
    first = convert_zarr_to_registration_nii(input_zarr, output_nii, **kwargs)
    assert output_nii.exists()
    # Existing volume.nii.gz is OVERWRITTEN, never refused.
    second = convert_zarr_to_registration_nii(input_zarr, output_nii, **kwargs)
    assert second["output_shape_zyx"] == first["output_shape_zyx"]


def test_convert_rejects_missing_input_and_mask(tmp_path):
    volume = _smooth_volume((8, 16, 16))
    input_zarr = tmp_path / "in.zarr"
    _write_input_zarr(input_zarr, volume)
    mask_zarr = tmp_path / "mask.zarr"
    _write_full_mask(mask_zarr, (8, 16, 16))
    output_nii = tmp_path / "volume.nii.gz"

    with pytest.raises(FileNotFoundError):
        convert_zarr_to_registration_nii(
            tmp_path / "missing.zarr",
            output_nii,
            input_resolution_xyz=(12.0, 12.0, 16.0),
            target_resolution_xyz=(24.0, 24.0, 32.0),
            brain_mask=mask_zarr,
        )

    with pytest.raises(FileNotFoundError):
        convert_zarr_to_registration_nii(
            input_zarr,
            output_nii,
            input_resolution_xyz=(12.0, 12.0, 16.0),
            target_resolution_xyz=(24.0, 24.0, 32.0),
            brain_mask=tmp_path / "missing_mask.zarr",
        )


def test_read_flip_atlas(tmp_path):
    # No config / missing key -> no flip.
    assert read_flip_atlas(None) == (False, False, False)
    assert read_flip_atlas(tmp_path / "absent.json") == (False, False, False)

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"registration": {}}), encoding="utf-8")
    assert read_flip_atlas(cfg) == (False, False, False)

    cfg.write_text(json.dumps({"registration": {"flip_atlas": [False, True, False]}}), encoding="utf-8")
    assert read_flip_atlas(cfg) == (False, True, False)

    cfg.write_text(json.dumps({"registration": {"flip_atlas": [True, False]}}), encoding="utf-8")
    with pytest.raises(ValueError):
        read_flip_atlas(cfg)


def test_convert_applies_config_flip_ap(tmp_path):
    """flip_atlas=[False,True,False] (AP) mirrors the volume along y (axis 1),
    matching the removed --flip_y behaviour."""
    volume = _smooth_volume((16, 64, 48))
    # Make it asymmetric along y so the flip is detectable.
    volume[:, :16, :] = 60000
    input_zarr = tmp_path / "in.zarr"
    _write_input_zarr(input_zarr, volume)
    mask_zarr = tmp_path / "mask.zarr"
    _write_full_mask(mask_zarr, (16, 64, 48))
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"registration": {"flip_atlas": [False, True, False]}}), encoding="utf-8")
    flip = read_flip_atlas(cfg)

    plain = np.asanyarray(
        nib.load(
            str(
                convert_zarr_to_registration_nii(
                    input_zarr,
                    tmp_path / "a" / "volume.nii.gz",
                    input_resolution_xyz=(12.0, 12.0, 16.0),
                    target_resolution_xyz=(24.0, 24.0, 32.0),
                    brain_mask=mask_zarr,
                    brain_mask_erode=0,
                )["output_nii"]
            )
        ).dataobj
    )
    flipped = np.asanyarray(
        nib.load(
            str(
                convert_zarr_to_registration_nii(
                    input_zarr,
                    tmp_path / "b" / "volume.nii.gz",
                    input_resolution_xyz=(12.0, 12.0, 16.0),
                    target_resolution_xyz=(24.0, 24.0, 32.0),
                    brain_mask=mask_zarr,
                    brain_mask_erode=0,
                    flip_atlas=flip,
                )["output_nii"]
            )
        ).dataobj
    )
    # flip_atlas AP -> numpy axis 1 (y), which is NIfTI axis 1 after xyz transpose.
    from pipeline_modules.preprocessing.zarr_to_registration_nii import flip_axes_from_flip_atlas

    assert flip_axes_from_flip_atlas((False, True, False)) == (1,)
    np.testing.assert_array_equal(flipped, np.flip(plain, axis=1))
