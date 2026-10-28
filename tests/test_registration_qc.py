"""registration_qc / check_orientation / 多平面 boundary 单测（合成样本）。"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_modules.visualization.pipeline_qc import (
    check_orientation,
    registration_qc,
)


def _make_synthetic(tmp_path: Path, flip: bool) -> Path:
    """合成样本：25um nii (x,y,z)=(8,32,24)，label 沿 z 后移且 y 偏一侧。
    flip=True 时 label 的 y 走向与信号相反。"""
    import nibabel as nib

    nz, ny, nx = 24, 32, 8  # nii 以 xyz 存
    vol = np.zeros((nx, ny, nz), dtype=np.float32)
    label = np.zeros((48, 64, 16), dtype=np.uint16)  # zyx @ 半分辨率示意
    for k in range(nz):
        y0 = 8 + k  # 信号质心随 z 增大而增大
        vol[:, y0 - 2:y0 + 2, k] = 100.0
    for zi in range(label.shape[0]):
        frac = zi / label.shape[0]
        y0 = int(16 + 24 * frac)  # label 质心同向增大
        if flip:
            y0 = int(40 - 24 * frac)
        label[zi, y0 * 2 - 4:y0 * 2 + 4, :] = 7

    d = tmp_path / ("flip" if flip else "ok")
    (d / "ch0_downsample").mkdir(parents=True)
    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(d / "ch0_downsample" / "volume.nii.gz"))
    g = zarr.open_group(str(d / "upsampled_atlas_label.zarr"), mode="w", zarr_format=2)
    arr = g.create_array("0", shape=label.shape, chunks=(4, 16, 16), dtype="uint16", compressors=None)
    arr[:] = label
    (d / "original_shape.json").write_text(json.dumps({"original_shape": [48, 64, 16]}), encoding="utf-8")
    (d / "config.json").write_text(json.dumps({
        "input": {"resolution_xyz": [1.0, 1.0, 1.0]},
        "preprocessing": {"downsample": {"target_resolution_xyz": [2.0, 2.0, 2.0]}},
    }), encoding="utf-8")
    return d


def test_check_orientation_detects_flip(tmp_path):
    ok_dir = _make_synthetic(tmp_path, flip=False)
    result = check_orientation(ok_dir / "ch0_downsample" / "volume.nii.gz",
                               ok_dir / "upsampled_atlas_label.zarr")
    assert result["y_centroid_corr"] > 0.3
    assert result["status"] in ("PASS", "INFO")

    flip_dir = _make_synthetic(tmp_path, flip=True)
    result2 = check_orientation(flip_dir / "ch0_downsample" / "volume.nii.gz",
                                flip_dir / "upsampled_atlas_label.zarr")
    assert result2["y_centroid_corr"] < 0
    assert result2["status"] == "FAIL"


def test_registration_qc_end_to_end(tmp_path):
    ok_dir = _make_synthetic(tmp_path, flip=False)
    verdict = registration_qc(ok_dir, render=True)
    assert (ok_dir / "qc" / "registration_verdict.json").exists()
    assert (ok_dir / "qc" / "registration_views.png").exists()
    assert verdict["checks"]["orientation_check"]["status"] in ("PASS", "INFO")
    assert "boundary_alignment" in verdict["checks"]
    assert "planes" in verdict["checks"]["boundary_alignment"]
