from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from pipeline_modules.segmentation.cfos_unet_qc import build_review_queue, compute_sample_qc_metrics


def _write_zarr(path: Path, array: np.ndarray) -> Path:
    root = zarr.open(str(path), mode="w")
    data = root.zeros("0", shape=array.shape, chunks=array.shape, dtype=array.dtype)
    data[:] = array
    return path


def test_compute_sample_qc_metrics_uses_probability_uncertainty(tmp_path: Path):
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[1, 1, 1] = 1
    mask[2, 2, 2] = 1
    prob = np.zeros((4, 4, 4), dtype=np.float32)
    prob[0, :, :] = 0.5
    prob[1, 1, 1] = 0.9

    mask_path = _write_zarr(tmp_path / "sample_mask.zarr", mask)
    prob_path = _write_zarr(tmp_path / "sample_prob.zarr", prob)

    metrics = compute_sample_qc_metrics(
        sample_id="sample",
        mask_zarr=mask_path,
        probability_zarr=prob_path,
        uncertainty_low=0.4,
        uncertainty_high=0.6,
        small_component_max_voxels=2,
    )

    assert metrics["sample_id"] == "sample"
    assert metrics["foreground_voxels"] == 2
    assert metrics["uncertain_voxels"] == 16
    assert metrics["uncertain_ratio"] == 16 / 64
    assert metrics["small_component_ratio"] >= 0


def test_build_review_queue_ranks_uncertain_sample_first(tmp_path: Path):
    mask_a = np.zeros((4, 4, 4), dtype=np.uint8)
    mask_a[1, 1, 1] = 1
    prob_a = np.full((4, 4, 4), 0.5, dtype=np.float32)

    mask_b = np.zeros((4, 4, 4), dtype=np.uint8)
    mask_b[1:3, 1:3, 1:3] = 1
    prob_b = np.full((4, 4, 4), 0.95, dtype=np.float32)

    mask_a_path = _write_zarr(tmp_path / "a_mask.zarr", mask_a)
    prob_a_path = _write_zarr(tmp_path / "a_prob.zarr", prob_a)
    mask_b_path = _write_zarr(tmp_path / "b_mask.zarr", mask_b)
    prob_b_path = _write_zarr(tmp_path / "b_prob.zarr", prob_b)

    ranked = build_review_queue(
        [
            {"sample_id": "a", "mask_zarr": str(mask_a_path), "probability_zarr": str(prob_a_path)},
            {"sample_id": "b", "mask_zarr": str(mask_b_path), "probability_zarr": str(prob_b_path)},
        ],
        small_component_max_voxels=2,
    )

    assert ranked[0]["sample_id"] == "a"
    assert ranked[0]["review_score"] > ranked[1]["review_score"]
