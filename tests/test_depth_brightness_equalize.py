"""Tests for the depth_profile brightness equalization (merged into
surface_brightness_homogenize.py; synthetic volumes, seconds)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import zarr
from scipy import ndimage as ndi

from pipeline_modules.preprocessing import surface_brightness_homogenize as sbh
from pipeline_modules.utils.errors import PipelineError


def _synthetic_volume(shape=(80, 160, 160), radii=(30.0, 70.0, 70.0), spacing=(2.0, 2.0, 2.0)):
    zz, yy, xx = np.meshgrid(
        (np.arange(shape[0]) + 0.5) * spacing[0],
        (np.arange(shape[1]) + 0.5) * spacing[1],
        (np.arange(shape[2]) + 0.5) * spacing[2],
        indexing="ij",
    )
    center = np.array(shape, dtype=np.float64) * 0.5 * np.array(spacing)
    radius = np.array(radii)
    ellipsoid = ((zz - center[0]) / radius[0]) ** 2 + ((yy - center[1]) / radius[1]) ** 2 + (
        (xx - center[2]) / radius[2]
    ) ** 2 <= 1.0
    d_in = ndi.distance_transform_edt(ellipsoid, sampling=spacing)
    d_out = ndi.distance_transform_edt(~ellipsoid, sampling=spacing)
    shell = 1500.0 * np.exp(-(((d_in - 12.0) / 8.0) ** 2))
    inside = 180.0 + 2900.0 * np.exp(-d_in / 25.0) + shell
    outside = 2.0 + 320.0 * np.exp(-d_out / 25.0)
    image = np.where(ellipsoid, inside, outside)
    rng = np.random.default_rng(0)
    image = image + rng.normal(0.0, 3.0, image.shape)
    return np.maximum(image, 0.0).astype(np.float32), ellipsoid


def test_fit_improves_uniformity_and_respects_limits():
    image, mask = _synthetic_volume()
    fitted, gain, d_in, _ = sbh._fit_volume(
        image, mask, (2.0, 2.0, 2.0), bin_um=4.0, max_gain=5.0, min_gain=0.1, smooth_bins=1.2, deep_fraction=0.2
    )

    assert fitted["gain_inside_min"] >= 0.1 - 1e-6
    assert fitted["gain_inside_max"] <= 5.0 + 1e-6
    assert fitted["gain_outside_min"] >= 0.1 - 1e-6
    assert fitted["gain_outside_max"] <= 1.0 + 1e-6
    assert np.all(gain > 0)

    corrected = image * gain
    before = _bin_medians(image, d_in, mask, 4.0)
    after = _bin_medians(corrected, d_in, mask, 4.0)
    assert after.max() / after.min() < 0.5 * (before.max() / before.min())
    assert fitted["target_inside"] * 0.75 < np.median(corrected[mask]) < fitted["target_inside"] * 1.25


def _bin_medians(values, distances, mask, bin_um):
    d = distances[mask]
    v = values[mask]
    bins = (d // bin_um).astype(int)
    medians = [np.median(v[bins == b]) for b in np.unique(bins) if np.count_nonzero(bins == b) >= 50]
    return np.asarray(medians)


def test_gain_limits_are_clamped():
    image, mask = _synthetic_volume()
    fitted, gain, _, _ = sbh._fit_volume(
        image, mask, (2.0, 2.0, 2.0), bin_um=4.0, max_gain=2.0, min_gain=0.5, smooth_bins=1.2, deep_fraction=0.2
    )
    assert fitted["gain_inside_min"] >= 0.5 - 1e-6
    assert fitted["gain_inside_max"] <= 2.0 + 1e-6


def test_target_scale_and_edge_gain():
    image, mask = _synthetic_volume()
    base, _, d_in, _ = sbh._fit_volume(
        image, mask, (2.0, 2.0, 2.0), bin_um=4.0, max_gain=5.0, min_gain=0.1,
        smooth_bins=1.2, deep_fraction=0.2,
    )
    bright, gain_bright, _, _ = sbh._fit_volume(
        image, mask, (2.0, 2.0, 2.0), bin_um=4.0, max_gain=5.0, min_gain=0.1,
        smooth_bins=1.2, deep_fraction=0.2, target_scale=1.25,
    )
    assert bright["target_inside"] == pytest.approx(base["target_inside"] * 1.25, rel=1e-6)

    _, gain_edge, _, _ = sbh._fit_volume(
        image, mask, (2.0, 2.0, 2.0), bin_um=4.0, max_gain=5.0, min_gain=0.1,
        smooth_bins=1.2, deep_fraction=0.2, target_scale=1.25, edge_gain=0.6, edge_um=10.0,
    )
    edge = mask & (d_in < 4.0)
    assert gain_edge[edge].mean() < gain_bright[edge].mean() * 0.8


def test_apply_chunk_preserves_dtype_and_clips():
    image = np.array([[[0, 100], [60000, 50000]]], dtype=np.uint16)
    gain = np.full(image.shape, 3.0, dtype=np.float32)
    out, clipped = sbh._apply_chunk(image, gain, np.dtype(np.uint16))
    assert out.dtype == np.uint16
    assert out[0, 0, 1] == 300
    assert out[0, 1, 0] == 65535
    assert clipped == 2


def test_gain_for_slab_identity():
    gain = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    slab = sbh.gain_for_slab(gain, (2.0, 2.0, 2.0), (2.0, 2.0, 2.0), gain.shape, 1, 2)
    np.testing.assert_array_equal(slab, gain[1:2])


def test_gain_for_block_matches_full_upsample():
    gain = np.arange(6 * 6 * 6, dtype=np.float32).reshape(6, 6, 6)
    full = sbh.gain_for_block(gain, (4.0, 4.0, 4.0), (2.0, 2.0, 2.0), (12, 12, 12), 0, 12, 0, 12, 0, 12)
    block = sbh.gain_for_block(gain, (4.0, 4.0, 4.0), (2.0, 2.0, 2.0), (12, 12, 12), 4, 8, 4, 8, 0, 12)
    np.testing.assert_allclose(block, full[4:8, 4:8, :], rtol=1e-5, atol=1e-5)


def test_gain_for_slab_upsample_constant():
    gain = np.full((6, 6, 6), 2.5, dtype=np.float32)
    slab = sbh.gain_for_slab(gain, (4.0, 4.0, 4.0), (2.0, 2.0, 2.0), (12, 12, 12), 0, 12)
    assert slab.shape == (12, 12, 12)
    np.testing.assert_allclose(slab, 2.5, rtol=1e-6)


def test_mask_shape_mismatch_raises(tmp_path):
    image = np.zeros((8, 16, 16), dtype=np.uint16)
    image[2:6, 4:12, 4:12] = 100
    mask = np.zeros((8, 16, 15), dtype=np.uint8)
    mask[2:6, 4:12, 4:11] = 1
    _write_zarr(tmp_path / "img.zarr", image)
    import tifffile

    tifffile.imwrite(str(tmp_path / "mask.tif"), mask)
    with pytest.raises(PipelineError):
        sbh.equalize_depth_brightness(
            tmp_path / "img.zarr",
            tmp_path / "mask.tif",
            tmp_path / "out.zarr",
            voxel_size_zyx=(2.0, 2.0, 2.0),
            fit_only=True,
        )


def test_end_to_end_zarr_roundtrip(tmp_path):
    image, mask = _synthetic_volume(shape=(32, 64, 64), radii=(12.0, 26.0, 26.0))
    image_u16 = np.clip(image, 0, 65535).astype(np.uint16)
    image_path = tmp_path / "img.zarr"
    _write_zarr(image_path, image_u16)
    import tifffile

    mask_path = tmp_path / "mask.tif"
    tifffile.imwrite(str(mask_path), mask.astype(np.uint8))

    out_path = tmp_path / "out.zarr"
    summary = sbh.equalize_depth_brightness(
        image_path,
        mask_path,
        out_path,
        voxel_size_zyx=(2.0, 2.0, 2.0),
        bin_um=4.0,
        apply_chunk_z=8,
        qc_dir=tmp_path / "qc",
    )

    out = zarr.open_group(str(out_path), mode="r")["0"][:]
    assert out.shape == image_u16.shape
    assert out.dtype == np.uint16
    assert summary["processed_slices"] == image_u16.shape[0]
    assert summary["clipped_voxels"] == 0
    assert not np.array_equal(out, image_u16)
    assert (out_path / "_run_manifest.json").exists()
    assert (tmp_path / "qc" / "slice_after.png").exists()
    attrs = dict(zarr.open_group(str(out_path), mode="r").attrs)
    assert "depth_brightness_equalize" in attrs


def test_fit_only_writes_json(tmp_path):
    image, mask = _synthetic_volume(shape=(32, 64, 64), radii=(12.0, 26.0, 26.0))
    image_path = tmp_path / "img.zarr"
    _write_zarr(image_path, image.astype(np.uint16))
    import tifffile

    mask_path = tmp_path / "mask.tif"
    tifffile.imwrite(str(mask_path), mask.astype(np.uint8))
    summary_path = tmp_path / "fit.json"
    result = sbh.equalize_depth_brightness(
        image_path,
        mask_path,
        summary_path,
        voxel_size_zyx=(2.0, 2.0, 2.0),
        bin_um=4.0,
        fit_only=True,
    )
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["fit_only"] is True
    assert result["target_inside"] > 0


def _write_zarr(path, array):
    group = zarr.open_group(str(path), mode="w")
    dataset = group.create_dataset("0", shape=array.shape, chunks=(8, 16, 16), dtype=array.dtype)
    dataset[:] = array
    return path
