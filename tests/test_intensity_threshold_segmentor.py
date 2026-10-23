"""Tests for the merged threshold segmentation methods."""

from __future__ import annotations

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from pipeline_modules.segmentation.intensity_threshold_segmentor import (
    estimate_percentile_from_histogram,
    run_threshold_segmentation,
)
from pipeline_modules.utils.errors import PipelineError


@pytest.fixture
def signal_zarr(tmp_path):
    rng = np.random.default_rng(0)
    volume = rng.integers(0, 50, size=(4, 32, 32), dtype=np.uint16)
    volume[1:3, 8:16, 8:16] = rng.integers(3000, 5000, size=(2, 8, 8), dtype=np.uint16)
    path = tmp_path / "in.zarr"
    zarr.save(str(path), volume)
    return path, volume


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "global", "threshold": "otsu"},
        {"method": "hysteresis", "low": 500, "high": 3000},
        {"method": "percentile"},
        {"method": "windowed"},
        {"method": "windowed", "clean_3d": True},
    ],
)
def test_methods_write_mask(tmp_path, signal_zarr, kwargs):
    path, volume = signal_zarr
    out = tmp_path / f"out_{kwargs['method']}.zarr"
    result = run_threshold_segmentation(
        input_zarr=path,
        output_zarr=out,
        min_size=10,
        **kwargs,
    )
    mask = zarr.open(str(out), mode="r")["0"]
    assert mask.shape == volume.shape
    assert int(np.asarray(mask[:]).sum()) > 0
    assert result["method"] == kwargs["method"]


def test_hysteresis_requires_high(signal_zarr):
    path, _ = signal_zarr
    with pytest.raises(PipelineError):
        run_threshold_segmentation(
            input_zarr=path,
            output_zarr=path.parent / "bad.zarr",
            method="hysteresis",
            low=500,
        )


def test_percentile_estimate_stays_in_clamp():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 100, size=(64, 64), dtype=np.uint16)
    image[:8, :8] = 5000
    est = estimate_percentile_from_histogram(image, method="blend", clamp_lo=95.0, clamp_hi=99.5)
    assert 95.0 <= est["suggested_percentile"] <= 99.5
    assert est["n_positive"] == int((image > 0).sum())
