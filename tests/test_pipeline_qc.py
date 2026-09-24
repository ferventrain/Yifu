from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import tifffile
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_modules.utils.zarr_io import create_output_zarr
from pipeline_modules.visualization.pipeline_qc import run_qc

import openpyxl


def _blob(shape, start, size, fill=5):
    vol = np.zeros(shape, dtype=np.uint16)
    z0, y0, x0 = start
    vol[z0:z0 + size[0], y0:y0 + size[1], x0:x0 + size[2]] = fill
    return vol


def _build_sample(tmp_path: Path, *, bad_balance=False) -> Path:
    sample = tmp_path / "SAMPLE_1"
    (sample / "results").mkdir(parents=True)
    (sample / "ch0_downsample").mkdir()
    (sample / "ch0_warped_image").mkdir()

    full_shape = (20, 24, 16)  # z,y,x
    res_xyz = (1.0, 1.0, 1.0)
    target = (1.0, 1.0, 1.0)
    config = {
        "input": {"resolution_xyz": res_xyz, "channels": {"signal": "1", "registration": "0"}},
        "preprocessing": {"downsample": {"target_resolution_xyz": list(target)}},
    }
    (sample / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (sample / "original_shape.json").write_text(json.dumps({"original_shape": list(full_shape)}), encoding="utf-8")

    # signal + mask zarrs with a bright blob; label with an offset blob; hemisphere
    z, y, x = np.mgrid[0:20, 0:24, 0:16]
    signal = (3000 * ((z - 10) ** 2 + (y - 12) ** 2 + (x - 8) ** 2 < 18)).astype(np.uint16)
    mask = (signal > 0).astype(np.uint8)
    create_output_zarr(sample / "ch1.zarr", shape=signal.shape, chunks=(8, 12, 8), dtype=signal.dtype, data=signal)
    create_output_zarr(sample / "ch1_mask.zarr", shape=mask.shape, chunks=(8, 12, 8), dtype=mask.dtype, data=mask)

    label = _blob(full_shape, (2, 3, 2), (16, 18, 12))
    create_output_zarr(sample / "upsampled_atlas_label.zarr", shape=label.shape, chunks=(8, 12, 8), dtype=label.dtype, data=label)

    hemi = np.zeros(full_shape, dtype=np.uint8)
    hemi[:, :, :8][label[:, :, :8] > 0] = 1
    hemi[:, :, 8:][label[:, :, 8:] > 0] = 2
    root_group, _ = create_output_zarr(sample / "atlas_label_hemisphere.zarr", shape=hemi.shape, chunks=(8, 12, 8), dtype=hemi.dtype, data=hemi)
    root_group.attrs["split_x"] = 8
    root_group.attrs["label_x_extent"] = [2, 13]

    nii_zyx = _blob(full_shape, (2, 3, 2), (16, 18, 12), fill=800).astype(np.uint16) + 50
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(np.transpose(nii_zyx, (2, 1, 0)), affine), sample / "ch0_downsample" / "volume.nii.gz")

    # warped atlas derived from the same blob -> correlation ~1
    warped = np.transpose(nii_zyx, (2, 1, 0))  # x,y,z
    for i in range(warped.shape[2]):
        tifffile.imwrite(sample / "ch0_warped_image" / f"image_{i:04d}.tiff", warped[:, :, i])

    left_total = 100 if bad_balance else 4000
    right_total = 4000 - left_total if bad_balance else 4000
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Level_0"
    ws.append(["Name", "Left Total Voxels", "Right Total Voxels", "Left Signal Voxels", "Right Signal Voxels"])
    ws.append(["root,root", left_total, right_total, 100, 100])
    wb.save(sample / "results" / "SAMPLE_1_ch1_brain_distribution_stats.xlsx")
    return sample


def _run(sample):
    config = json.loads((sample / "config.json").read_text(encoding="utf-8"))
    return run_qc(sample, config=config)


def test_qc_passes_on_good_sample(tmp_path):
    sample = _build_sample(tmp_path)
    verdict = _run(sample)
    assert verdict["checks"]["hemisphere_balance"]["status"] == "PASS"
    assert verdict["checks"]["grid_shape"]["status"] == "PASS"
    assert verdict["checks"]["registration_correlation"]["status"] == "PASS"
    # synthetic xlsx totals sit below the production magnitude range by design
    assert verdict["checks"]["label_total_voxels"]["status"] == "FAIL"
    assert (sample / "qc" / "registration_views.png").exists()
    blocks = sorted((sample / "qc" / "seg_blocks").glob("block_*.zarr"))
    assert 1 <= len(blocks) <= 5
    block = zarr.open(str(blocks[0]), mode="r")
    assert "0" in block and "1" in block
    assert block["0"].dtype == np.uint16 and block["1"].dtype == np.uint8
    assert "block_offset_zyx" in dict(block.attrs)


def test_qc_fails_on_bad_hemisphere_balance(tmp_path):
    sample = _build_sample(tmp_path, bad_balance=True)
    verdict = _run(sample)
    assert verdict["checks"]["hemisphere_balance"]["status"] == "FAIL"
    assert verdict["overall"] == "FAIL"


def test_qc_flags_oversampled_grid(tmp_path):
    sample = _build_sample(tmp_path)
    # halve the grid: nii now 2x per axis vs expectation
    nii_zyx = np.zeros((10, 12, 8), dtype=np.uint16)
    nib.save(nib.Nifti1Image(np.transpose(nii_zyx, (2, 1, 0)), np.eye(4)), sample / "ch0_downsample" / "volume.nii.gz")
    verdict = _run(sample)
    assert verdict["checks"]["grid_shape"]["status"] == "FAIL"
