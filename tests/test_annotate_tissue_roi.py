"""Tests for the 组织选区 (tissue ROI) one-command annotation entry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline_modules.utils.errors import ErrorCode, PipelineError
from pipeline_modules.visualization.annotate_tissue_sam_napari import (
    TISSUE_ROI_FILENAME,
    build_parser,
    find_tissue_qc_volume,
    tissue_roi_output_path,
)


def test_tissue_roi_output_is_fixed_per_sample(tmp_path: Path):
    sample = tmp_path / "mouse01"
    out = tissue_roi_output_path(sample)
    assert out == sample / TISSUE_ROI_FILENAME
    assert out.name == "tissue_roi.zarr"  # never depends on channel or input stem


def test_find_tissue_qc_volume_requires_exactly_one(tmp_path: Path):
    sample = tmp_path / "mouse01"
    sample.mkdir()
    with pytest.raises(PipelineError) as exc:
        find_tissue_qc_volume(sample)
    assert exc.value.code == ErrorCode.INPUT_NOT_FOUND
    assert "surface_brightness_homogenize" in exc.value.context["next_step"]

    qc = sample / "ch0_surface_homogenized_qc.zarr"
    qc.mkdir()
    assert find_tissue_qc_volume(sample) == qc

    (sample / "ch1_surface_homogenized_qc.zarr").mkdir()
    with pytest.raises(PipelineError) as exc:
        find_tissue_qc_volume(sample)
    assert exc.value.code == ErrorCode.ARGUMENT_INVALID
    assert len(exc.value.context["candidates"]) == 2


def test_find_tissue_qc_volume_rejects_missing_sample(tmp_path: Path):
    with pytest.raises(PipelineError) as exc:
        find_tissue_qc_volume(tmp_path / "nope")
    assert exc.value.code == ErrorCode.INPUT_NOT_FOUND


def test_parser_accepts_sample_dir_mode():
    args = build_parser().parse_args(["--sample-dir", "S:/x"])
    assert args.sample_dir == "S:/x" and args.input is None
    args = build_parser().parse_args(["--input", "a_qc.zarr"])
    assert args.input == "a_qc.zarr" and args.sample_dir is None
