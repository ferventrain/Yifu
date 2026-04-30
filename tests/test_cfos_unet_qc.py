from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
import zarr

from pipeline_modules.segmentation.cfos_unet_qc import (
    _records_from_sample_dirs,
    build_review_queue,
    compute_block_qc_metrics,
    compute_sample_qc_metrics,
    export_block_previews,
)


def _write_zarr(path: Path, array: np.ndarray, *, chunks=None) -> Path:
    root = zarr.open(str(path), mode="w")
    if chunks is None:
        chunks = array.shape
    data = root.zeros("0", shape=array.shape, chunks=chunks, dtype=array.dtype)
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


def test_compute_block_qc_metrics_emits_one_record_per_block(tmp_path: Path):
    image = np.arange(64, dtype=np.uint16).reshape(4, 4, 4)
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[0:2, 0:2, 0:2] = 1
    prob = np.zeros((4, 4, 4), dtype=np.float32)
    prob[0:2, 0:2, 0:2] = 0.5

    image_path = _write_zarr(tmp_path / "sample.zarr", image, chunks=(2, 2, 2))
    mask_path = _write_zarr(tmp_path / "sample_mask.zarr", mask, chunks=(2, 2, 2))
    prob_path = _write_zarr(tmp_path / "sample_prob.zarr", prob, chunks=(2, 2, 2))

    records = compute_block_qc_metrics(
        sample_id="sample",
        image_zarr=image_path,
        mask_zarr=mask_path,
        probability_zarr=prob_path,
        chunk_size=(2, 2, 2),
    )

    assert len(records) == 8
    assert records[0]["sample_id"] == "sample"
    assert records[0]["block_id"] == "block_000001"
    assert records[0]["chunk_index"] == "0.0.0"
    assert records[0]["block_start_zyx"] == "0,0,0"
    assert records[0]["block_stop_zyx"] == "2,2,2"
    assert records[0]["block_shape_zyx"] == "2,2,2"


def test_build_review_queue_ranks_uncertain_block_first(tmp_path: Path):
    image = np.arange(64, dtype=np.uint16).reshape(4, 4, 4)
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[0:2, 0:2, 0:2] = 1
    mask[2:4, 2:4, 2:4] = 1
    prob = np.full((4, 4, 4), 0.95, dtype=np.float32)
    prob[0:2, 0:2, 0:2] = 0.5

    image_path = _write_zarr(tmp_path / "sample.zarr", image, chunks=(2, 2, 2))
    mask_path = _write_zarr(tmp_path / "sample_mask.zarr", mask, chunks=(2, 2, 2))
    prob_path = _write_zarr(tmp_path / "sample_prob.zarr", prob, chunks=(2, 2, 2))

    ranked = build_review_queue(
        [
            {
                "sample_id": "sample",
                "image_zarr": str(image_path),
                "mask_zarr": str(mask_path),
                "probability_zarr": str(prob_path),
            }
        ],
        chunk_size=(2, 2, 2),
        small_component_max_voxels=2,
    )

    assert ranked[0]["sample_id"] == "sample"
    assert ranked[0]["chunk_index"] == "0.0.0"
    assert ranked[0]["review_score"] > ranked[1]["review_score"]


def test_records_from_sample_dirs_infers_image_zarr(tmp_path: Path):
    sample_dir = tmp_path / "mouse01"
    sample_dir.mkdir()

    records = _records_from_sample_dirs(
        [sample_dir],
        signal_ch="2",
        mask_suffix="_mask.zarr",
        probability_suffix="_prob.zarr",
        threshold_suffix="",
    )

    assert records[0]["image_zarr"] == str(sample_dir / "ch2.zarr")
    assert records[0]["mask_zarr"] == str(sample_dir / "ch2_mask.zarr")




def test_build_review_queue_uses_small_component_ratio_in_score(tmp_path: Path):
    image = np.arange(128, dtype=np.uint16).reshape(4, 4, 8)
    mask = np.zeros((4, 4, 8), dtype=np.uint8)
    prob = np.full((4, 4, 8), 0.95, dtype=np.float32)

    mask[0:2, 0:2, 0:2] = 1
    mask[0, 0, 4] = 1
    mask[1, 1, 5] = 1

    image_path = _write_zarr(tmp_path / "sample.zarr", image, chunks=(2, 2, 2))
    mask_path = _write_zarr(tmp_path / "sample_mask.zarr", mask, chunks=(2, 2, 2))
    prob_path = _write_zarr(tmp_path / "sample_prob.zarr", prob, chunks=(2, 2, 2))

    ranked = build_review_queue(
        [
            {
                "sample_id": "sample",
                "image_zarr": str(image_path),
                "mask_zarr": str(mask_path),
                "probability_zarr": str(prob_path),
            }
        ],
        chunk_size=(2, 2, 2),
        small_component_max_voxels=2,
    )



def test_export_block_previews_writes_image_and_mask_tiffs(tmp_path: Path):
    image = np.arange(64, dtype=np.uint16).reshape(4, 4, 4)
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[0:2, 0:2, 0:2] = 1
    prob = np.full((4, 4, 4), 0.95, dtype=np.float32)
    prob[0:2, 0:2, 0:2] = 0.5

    image_path = _write_zarr(tmp_path / "sample.zarr", image, chunks=(2, 2, 2))
    mask_path = _write_zarr(tmp_path / "sample_mask.zarr", mask, chunks=(2, 2, 2))
    prob_path = _write_zarr(tmp_path / "sample_prob.zarr", prob, chunks=(2, 2, 2))

    ranked = build_review_queue(
        [
            {
                "sample_id": "sample",
                "image_zarr": str(image_path),
                "mask_zarr": str(mask_path),
                "probability_zarr": str(prob_path),
            }
        ],
        chunk_size=(2, 2, 2),
    )
    top_records = ranked[:1]
    preview_dir = tmp_path / "previews"
    export_block_previews(top_records, preview_dir)

    image_preview_path = Path(top_records[0]["preview_image_tiff"])
    mask_preview_path = Path(top_records[0]["preview_mask_tiff"])
    assert image_preview_path.exists()
    assert mask_preview_path.exists()

    image_preview = tifffile.imread(str(image_preview_path))
    mask_preview = tifffile.imread(str(mask_preview_path))
    assert image_preview.shape == (2, 2, 2)
    assert mask_preview.shape == (2, 2, 2)
    np.testing.assert_array_equal(image_preview, image[0:2, 0:2, 0:2])
    np.testing.assert_array_equal(mask_preview, mask[0:2, 0:2, 0:2])
