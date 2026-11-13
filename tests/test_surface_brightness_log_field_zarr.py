"""Tests for the Zarr log_field path in surface_brightness_homogenize."""

from __future__ import annotations

import json

import numpy as np
import pytest
import zarr
from scipy import ndimage as ndi

from pipeline_modules.preprocessing import surface_brightness_homogenize as sbh
from pipeline_modules.utils.errors import PipelineError


def _synthetic_region_volume(shape=(80, 160, 160), spacing=(8.0, 7.2, 7.2)):
    zz, yy, xx = np.meshgrid(
        (np.arange(shape[0]) + 0.5) * spacing[0],
        (np.arange(shape[1]) + 0.5) * spacing[1],
        (np.arange(shape[2]) + 0.5) * spacing[2],
        indexing="ij",
    )
    center = np.array(shape, dtype=np.float64) * 0.5 * np.array(spacing)
    mask = (
        ((zz - center[0]) / 100.0) ** 2
        + ((yy - center[1]) / 300.0) ** 2
        + ((xx - center[2]) / 300.0) ** 2
        <= 1.0
    )
    ramp = np.exp((yy - yy.mean()) / 400.0)  # smooth ~4.5x regional brightness ramp
    rng = np.random.default_rng(0)
    image = np.where(mask, 700.0 * ramp * (1.0 + 0.08 * rng.normal(size=shape)), 5.0)
    return np.maximum(image, 0.0).astype(np.uint16), mask


def _write_zarr(path, array, chunks=(16, 32, 32)):
    group = zarr.open_group(str(path), mode="w")
    dataset = group.create_dataset("0", shape=array.shape, chunks=chunks, dtype=array.dtype)
    dataset[:] = array
    return path


def test_log_field_zarr_improves_regional_uniformity_and_keeps_outside(tmp_path):
    import tifffile

    image, mask = _synthetic_region_volume()
    image_path = _write_zarr(tmp_path / "img.zarr", image)
    mask_path = tmp_path / "mask.tif"
    tifffile.imwrite(str(mask_path), mask.astype(np.uint8))
    out_path = tmp_path / "out.zarr"

    summary = sbh.normalize_log_field_zarr(
        image_path,
        mask_path,
        out_path,
        voxel_size_zyx=(8.0, 7.2, 7.2),
        field_sigma_um=80.0,
        strength=1.0,
        min_gain=0.2,
        max_gain=5.0,
        apply_chunk_z=16,
        qc_dir=tmp_path / "qc",
    )

    out = zarr.open_group(str(out_path), mode="r")["0"][:]
    assert out.shape == image.shape
    assert out.dtype == np.uint16
    assert summary["regional_before"]["tiles"] >= 3
    assert summary["regional_after"]["ratio_p90_p10"] < 0.6 * summary["regional_before"]["ratio_p90_p10"]
    assert 0.2 - 1e-6 <= summary["gain_inside_min"] <= summary["gain_inside_max"] <= 5.0 + 1e-6
    assert int(out[0, 0, 0]) == int(image[0, 0, 0])  # far outside the mask is untouched
    assert (out_path / "_run_manifest.json").exists()
    attrs = dict(zarr.open_group(str(out_path), mode="r").attrs)
    assert attrs["log_field_normalize"]["field_sigma_um"] == 80.0
    for name in ("slice_before.png", "slice_after.png", "gain_slice.png"):
        assert (tmp_path / "qc" / name).exists()


def test_log_field_zarr_fit_only_writes_json(tmp_path):
    import tifffile

    image, mask = _synthetic_region_volume()
    image_path = _write_zarr(tmp_path / "img.zarr", image)
    mask_path = tmp_path / "mask.tif"
    tifffile.imwrite(str(mask_path), mask.astype(np.uint8))
    summary_path = tmp_path / "fit.json"

    result = sbh.normalize_log_field_zarr(
        image_path,
        mask_path,
        summary_path,
        voxel_size_zyx=(8.0, 7.2, 7.2),
        fit_only=True,
    )

    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["fit_only"] is True
    assert result["i_ref"] > 0


def test_log_field_zarr_mask_grid_mismatch_raises(tmp_path):
    import tifffile

    image, mask = _synthetic_region_volume()
    image_path = _write_zarr(tmp_path / "img.zarr", image)
    mask_path = tmp_path / "mask.tif"
    tifffile.imwrite(str(mask_path), mask[:, :, :-1].astype(np.uint8))

    with pytest.raises(PipelineError):
        sbh.normalize_log_field_zarr(
            image_path,
            mask_path,
            tmp_path / "out.zarr",
            voxel_size_zyx=(8.0, 7.2, 7.2),
            fit_only=True,
        )


def test_log_field_zarr_rejects_bad_gain_limits(tmp_path):
    image, mask = _synthetic_region_volume()
    image_path = _write_zarr(tmp_path / "img.zarr", image)

    with pytest.raises(PipelineError):
        sbh.normalize_log_field_zarr(
            image_path,
            image_path,
            tmp_path / "out.zarr",
            voxel_size_zyx=(8.0, 7.2, 7.2),
            min_gain=1.5,
            max_gain=5.0,
            fit_only=True,
        )
