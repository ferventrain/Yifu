"""Label-gated chunk skipping for cfos_unet_inference (chunk-local normalization guard)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from pipeline_modules.segmentation.cfos_unet_inference import chunk_has_label
from pipeline_modules.segmentation.cfos_unet_model import build_cfos_unet_classes
from pipeline_modules.utils.errors import ErrorCode, PipelineError


def make_label_store(path: Path, fill: np.ndarray) -> zarr.Array:
    store = zarr.open(str(path), mode="w", shape=fill.shape, chunks=(16, 16, 16), dtype="uint32")
    store[:] = fill
    return store


def test_chunk_has_label_detects_labeled_core(tmp_path: Path) -> None:
    fill = np.zeros((64, 64, 64), dtype="uint32")
    fill[16:49:8, 16:49:8, 16:49:8] = 7  # labeled voxels on the stride-8 lattice
    label = make_label_store(tmp_path / "label.zarr", fill)
    whole = (slice(0, 64), slice(0, 64), slice(0, 64))
    assert chunk_has_label(label, whole, 8) is True


def test_chunk_has_label_false_for_background(tmp_path: Path) -> None:
    fill = np.zeros((64, 64, 64), dtype="uint32")
    label = make_label_store(tmp_path / "label.zarr", fill)
    assert chunk_has_label(label, (slice(0, 64), slice(0, 64), slice(0, 64)), 8) is False


def test_chunk_has_label_per_chunk_window(tmp_path: Path) -> None:
    fill = np.zeros((64, 128, 128), dtype="uint32")
    fill[:, 64:, :] = 3  # only the second y-half labeled
    label = make_label_store(tmp_path / "label.zarr", fill)
    assert chunk_has_label(label, (slice(0, 64), slice(0, 64), slice(0, 128)), 8) is False
    assert chunk_has_label(label, (slice(0, 64), slice(64, 128), slice(0, 128)), 8) is True


def _save_tiny_checkpoint(path: Path, patch_size: list[int]) -> None:
    torch, SimpleUNet3D = build_cfos_unet_classes()
    model = SimpleUNet3D(in_channels=1, num_classes=2, base_channels=2)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": {"patch_size": patch_size, "base_channels": 2, "num_classes": 2},
        },
        path,
    )


def test_label_gating_skips_unlabeled_chunks_end_to_end(tmp_path: Path) -> None:
    """Full tiny run: unlabeled chunks are skipped and produce no mask voxels."""
    from pipeline_modules.segmentation.cfos_unet_inference import run_cfos_unet_inference

    shape = (16, 64, 128)
    sig_store = zarr.open(str(tmp_path / "sig.zarr"), mode="w", shape=shape, chunks=(16, 64, 32), dtype="uint16")
    sig_store[:] = 200
    label_fill = np.zeros(shape, dtype="uint32")
    label_fill[:, :, :32] = 1  # only the first x-quarter is atlas-labeled tissue
    lbl_store = zarr.open(str(tmp_path / "lbl.zarr"), mode="w", shape=shape, chunks=(16, 64, 32), dtype="uint32")
    lbl_store[:] = label_fill

    ckpt = tmp_path / "best_model.pt"
    _save_tiny_checkpoint(ckpt, [16, 64, 64])

    out = tmp_path / "mask.zarr"
    result = run_cfos_unet_inference(
        input_zarr=str(tmp_path / "sig.zarr"),
        output_zarr=out,
        checkpoint_path=ckpt,
        patch_size=(16, 64, 64),
        chunk_size=(16, 64, 32),
        batch_size=1,
        normalize_scope="chunk",
        label_zarr=str(tmp_path / "lbl.zarr"),
        label_stride=4,
        skip_below_threshold=None,
    )
    # chunks are (16,64,32): four along x, the unlabeled three are gated out
    assert result["skipped_no_label"] == 3
    mask = zarr.open(str(out), mode="r")["0"][:]
    assert mask.shape == shape
    assert not np.asarray(mask[:, :, 32:]).any()  # nothing written in the unlabeled part


def test_shape_mismatch_rejected(tmp_path: Path) -> None:
    from pipeline_modules.segmentation.cfos_unet_inference import run_cfos_unet_inference

    zarr.open(str(tmp_path / "sig.zarr"), mode="w", shape=(4, 8, 8), dtype="uint16")
    zarr.open(str(tmp_path / "lbl.zarr"), mode="w", shape=(8, 8, 8), dtype="uint32")
    ckpt = tmp_path / "m.pt"
    _save_tiny_checkpoint(ckpt, [4, 8, 8])
    with pytest.raises(PipelineError) as excinfo:
        run_cfos_unet_inference(
            input_zarr=str(tmp_path / "sig.zarr"),
            output_zarr=tmp_path / "out.zarr",
            checkpoint_path=ckpt,
            label_zarr=str(tmp_path / "lbl.zarr"),
        )
    assert excinfo.value.code == ErrorCode.ARGUMENT_INVALID
