"""Tests for the aind fusion3 brain-mask module and its registration wiring."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scipy import ndimage

from pipeline_modules.segmentation.brain_mask_aind import (
    DEFAULT_MODEL_PATH,
    fuse_slab,
    run_brain_mask,
)
from pipeline_modules.preprocessing.zarr_to_registration_nii import (
    convert_zarr_to_registration_nii,
)
from pipeline_modules.utils.zarr_io import create_output_zarr, open_zarr_array

pytest.importorskip("torch", reason="torch not installed")


def _slab_with_cracked_hole() -> np.ndarray:
    """64x64 probabilities: brain disc at 0.9 with an interior hole (0.55)
    connected to the outside by a thin low-probability crack (0.55)."""
    yy, xx = np.mgrid[0:64, 0:64]
    r = np.sqrt((yy - 32) ** 2 + (xx - 32) ** 2)
    brain = np.where(r < 24, 0.9, 0.05).astype(np.float32)
    brain[r < 7] = 0.55  # interior hole
    brain[(np.abs(xx - 32) < 1) & (yy < 32) & (yy > 8)] = 0.55  # crack to outside
    return brain


def test_fuse_slab_closes_crack_via_other_planes():
    pa = _slab_with_cracked_hole()  # hole + crack open in plane A
    pb = _slab_with_cracked_hole()
    pb[(pb > 0.1) & (pb < 0.8)] = 0.9  # planes B/C see solid brain there
    pc = pb.copy()

    mask = fuse_slab(pa * 255, pb * 255, pc * 255, threshold=0.7)

    # (0.55 + 0.9 + 0.9) / 3 = 0.783 > 0.7 -> both get filled by fusion
    assert mask[32, 32], "interior hole should be filled by plane fusion"
    assert mask[20, 32], "crack channel should be sealed by plane fusion"
    assert not mask[2, 2], "background stays out"


def test_fuse_slab_drops_small_components():
    yy, xx = np.mgrid[0:64, 0:64]
    big = np.sqrt((yy - 32) ** 2 + (xx - 32) ** 2) < 20
    tiny = np.sqrt((yy - 8) ** 2 + (xx - 52) ** 2) < 3
    plane = np.where(big | tiny, 0.9, 0.05).astype(np.float32)
    stack = np.repeat(plane[None], 4, axis=0)

    mask = fuse_slab(stack * 255, stack * 255, stack * 255, threshold=0.7)

    assert mask[:, big].all()
    # tiny blob (~28 px < 0.2 * largest) is dropped per slice
    assert not mask[:, tiny].any()


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="model weights not bundled")
def test_checkpoint_matches_swin_unetr_skeleton():
    from pipeline_modules.segmentation.brain_mask_aind import build_model

    # strict state-dict load: any divergence between the bundled checkpoint and
    # the vendored SwinUNETR skeleton fails right here
    build_model(DEFAULT_MODEL_PATH, device="cpu")


def _write_zarr(path, volume, chunks=(2, 8, 8)):
    create_output_zarr(path, shape=volume.shape, chunks=chunks, dtype=volume.dtype, data=volume)


def test_convert_uses_external_brain_mask(tmp_path):
    volume = np.full((12, 48, 40), 2000, np.uint16)
    _write_zarr(tmp_path / "ch0_L2.zarr", volume)

    # Coarse-grid mask: central block + a stray fragment on the same zyx grid.
    mask = np.zeros((12, 48, 40), np.uint8)
    mask[2:10, 12:36, 10:30] = 1
    mask[4:6, 4:8, 34:38] = 1
    _write_zarr(tmp_path / "ch0_brain_mask_dl.zarr", mask, chunks=(4, 16, 16))

    output_nii = tmp_path / "ch0_downsample" / "volume.nii.gz"
    result = convert_zarr_to_registration_nii(
        tmp_path / "ch0_L2.zarr",
        output_nii,
        input_resolution_xyz=(12.0, 12.0, 16.0),
        target_resolution_xyz=(24.0, 24.0, 32.0),
        halo_mask=True,
        brain_mask_zarr=tmp_path / "ch0_brain_mask_dl.zarr",
        brain_mask_erode=0,
    )

    assert result["brain_mask"]["applied"] is True
    assert not result["halo_mask"]["applied"]

    # Expected mask on the registration grid: nearest zoom + largest component,
    # mirroring what load_external_brain_mask does (erode=0 here).
    out_shape = tuple(result["output_shape_zyx"])
    mapped = ndimage.zoom(
        (mask > 0).astype(np.uint8),
        np.array(out_shape) / np.array(mask.shape),
        order=0,
        mode="nearest",
    ) > 0
    labels, count = ndimage.label(mapped)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    expected = labels == int(np.argmax(sizes))
    expected_xyz = np.transpose(expected, (2, 1, 0))

    import nibabel as nib

    reg = np.asarray(nib.load(str(output_nii)).dataobj)  # xyz, unit affine
    assert reg.shape == expected_xyz.shape
    assert (reg[expected_xyz] == 2000).all(), "voxels inside the DL mask survive"
    assert (reg[~expected_xyz] == 0).all(), "fragment and background are zeroed"

    mask_nii = np.asarray(nib.load(str(output_nii.parent / "brain_mask.nii.gz")).dataobj)
    assert set(np.unique(mask_nii)) <= {0, 1}
    assert (mask_nii == expected_xyz.astype(np.uint8)).all()

    shape_meta = json.loads((output_nii.parent / "original_shape.json").read_text())
    assert shape_meta["brain_mask"]["applied"] is True
    # the stray fragment is mapped too, so the pre-cleanup count is 2
    assert shape_meta["brain_mask"]["components"] == 2


def test_run_brain_mask_end_to_end_tiny(tmp_path, monkeypatch):
    """Tiny synthetic volume through the full 3-plane driver with a stub model."""
    volume = np.zeros((6, 32, 24), np.uint16)
    volume[:, 8:24, 6:18] = 900
    _write_zarr(tmp_path / "ch0_downsampled.zarr", volume, chunks=(2, 16, 12))

    checkpoint = tmp_path / "fake.safetensors"
    checkpoint.write_bytes(b"stub")  # existence is checked; weights are not loaded

    import pipeline_modules.segmentation.brain_mask_aind as mod

    def _fake_infer(model, slices_np, device="cuda"):
        lo, hi = float(slices_np.min()), float(slices_np.max())
        normed = (slices_np - lo) / (hi - lo) if hi > lo else slices_np * 0
        return (normed > 0.5).astype(np.float32)

    monkeypatch.setattr(mod, "build_model", lambda model_path, device="cuda": None)
    monkeypatch.setattr(mod, "infer_probabilities", _fake_infer)

    result = run_brain_mask(
        tmp_path, channel=0, device="cpu", model_path=checkpoint, min_area=20
    )

    assert result["mean_fraction"] > 0
    mask = np.asarray(open_zarr_array(tmp_path / "ch0_brain_mask_dl.zarr"))
    assert mask.shape == volume.shape
    # the bright block is masked (all three planes agree), background is not
    assert mask[:, 10:20, 8:16].all()
    assert not mask[:, 0:4, 0:4].any()
    # probability caches are cleaned up by default
    assert not (tmp_path / "ch0_brain_mask_probs").exists()
    assert (tmp_path / "qc" / "brain_mask_dl.png").exists()
