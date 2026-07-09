from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import time
from scipy import ndimage
import zarr

from pipeline_modules.qc.grading import grade_qc_results
from pipeline_modules.qc.image_qc import ImageQcConfig, compute_global_exposure_metrics, compute_image_qc, run_image_qc
from pipeline_modules.qc.progress import QcProgressTracker
from pipeline_modules.qc.metrics import (
    DEFAULT_DARK_PIXEL_THRESHOLD,
    compute_brightness_uniformity_metrics,
    compute_contrast_metrics,
    compute_diffuse_signal_metrics,
    compute_exposure_dynamic_range_metrics,
    compute_focus_metrics,
    compute_stripe_metrics,
)


def _make_uniform_image(value: float = 1000.0, shape: tuple[int, int] = (512, 512)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint16)


def _make_ramp_image(shape: tuple[int, int] = (512, 512)) -> np.ndarray:
    y = np.linspace(500, 1500, shape[0], dtype=np.float64)
    x = np.linspace(800, 1200, shape[1], dtype=np.float64)
    return (y[:, None] + x[None, :]).astype(np.uint16)


def _make_striped_image(shape: tuple[int, int] = (512, 512)) -> np.ndarray:
    image = np.full(shape, 1000, dtype=np.float64)
    image[::8, :] += 400
    return image.astype(np.uint16)


def _make_diffuse_noise_image(shape: tuple[int, int] = (512, 512)) -> np.ndarray:
    image = np.full(shape, 1800, dtype=np.float64)
    image += np.random.default_rng(0).normal(0, 20, size=shape)
    return np.clip(image, 0, 65535).astype(np.uint16)


def _make_structured_signal_image(shape: tuple[int, int] = (512, 512)) -> np.ndarray:
    image = np.full(shape, 300, dtype=np.float64)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    fibers = ((np.sin(xx / 6.0) + np.sin(yy / 10.0)) > 0.4).astype(np.float64)
    image += fibers * 5000
    return np.clip(image, 0, 65535).astype(np.uint16)


def _make_sharp_vs_blur_pair(shape: tuple[int, int] = (256, 256)) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    sharp = ((np.sin(xx / 8.0) + np.sin(yy / 8.0)) * 500 + 1500).astype(np.uint16)
    blur = np.asarray(ndimage.gaussian_filter(sharp.astype(np.float64), sigma=4.0), dtype=np.uint16)
    return sharp, blur


def _write_zarr(path: Path, array: np.ndarray, *, chunks=None) -> Path:
    if chunks is None:
        chunks = (1, min(array.shape[1], 128), min(array.shape[2], 128))
    root = zarr.open_group(str(path), mode="w")
    data = root.create_array("0", shape=array.shape, chunks=chunks, dtype=array.dtype)
    data[:] = array
    return path


def test_exposure_metrics_use_fixed_dark_threshold():
    image = np.zeros((64, 64), dtype=np.uint16)
    image[:32, :] = 10
    image[32:, :] = 65535

    metrics = compute_exposure_dynamic_range_metrics(image, dark_pixel_threshold=100)
    assert metrics["dark_pixel_ratio"] == pytest.approx(0.5, rel=1e-3)
    assert metrics["saturated_pixel_ratio"] == pytest.approx(0.5, rel=1e-3)
    assert metrics["dark_pixel_threshold"] == 100


def test_diffuse_noise_scores_higher_than_structured_signal():
    noise = compute_diffuse_signal_metrics(_make_diffuse_noise_image(), tile_size=64)
    structured = compute_diffuse_signal_metrics(_make_structured_signal_image(), tile_size=64)
    assert noise["diffuse_noise_score"] > structured["diffuse_noise_score"]
    assert structured["structured_signal_fraction"] > noise["structured_signal_fraction"]
    assert noise["noise_to_signal_area_ratio"] > structured["noise_to_signal_area_ratio"]


def test_uniformity_metrics_increase_for_ramp():
    uniform = compute_brightness_uniformity_metrics(_make_uniform_image())
    ramp = compute_brightness_uniformity_metrics(_make_ramp_image())
    assert ramp["tile_median_cv"] > uniform["tile_median_cv"]
    assert ramp["x_slab_slope"] != 0.0
    assert ramp["center_to_edge_ratio"] != 1.0


def test_stripe_metrics_detect_horizontal_stripes():
    uniform = compute_stripe_metrics(_make_uniform_image())
    striped = compute_stripe_metrics(_make_striped_image(), fft_enabled=True)
    assert striped["row_profile_cv"] > uniform["row_profile_cv"]
    assert striped["slab_stripe_score"] > uniform["slab_stripe_score"]
    assert striped["row_fft_peak_ratio"] > 0.0


def test_focus_metrics_rank_sharp_above_blur():
    sharp, blur = _make_sharp_vs_blur_pair()
    sharp_metrics = compute_focus_metrics(sharp)
    blur_metrics = compute_focus_metrics(blur)
    assert sharp_metrics["laplacian_variance"] > blur_metrics["laplacian_variance"]
    assert sharp_metrics["tenengrad_score"] > blur_metrics["tenengrad_score"]


def test_contrast_metrics_include_otsu_and_cnr():
    image = np.zeros((128, 128), dtype=np.uint16)
    image[:, :64] = 500
    image[:, 64:] = 4000
    metrics = compute_contrast_metrics(image)
    assert metrics["otsu_threshold"] > 500
    assert metrics["candidate_cnr"] > 0
    assert metrics["percentile_contrast_ratio"] > 0


def test_compute_image_qc_includes_grading():
    volume = np.stack([_make_uniform_image(1000), _make_striped_image()], axis=0)
    results = compute_image_qc(volume, config=ImageQcConfig(max_slices=2))
    assert "grading" in results
    assert results["grading"]["overall_verdict"] in {"pass", "warn", "fail"}
    assert "rules" in results["grading"]


def test_global_exposure_metrics_use_histogram():
    volume = np.stack([_make_uniform_image(1000), _make_uniform_image(2000)], axis=0)
    metrics = compute_global_exposure_metrics(
        volume,
        histogram_bins=256,
        saturation_margin=0.001,
        dark_pixel_threshold=DEFAULT_DARK_PIXEL_THRESHOLD,
    )
    assert metrics["mean"] == pytest.approx(1500.0, rel=0.05)
    assert metrics["robust_dynamic_range"] > 0


def test_run_image_qc_zarr_streaming(tmp_path: Path):
    volume = np.stack([_make_diffuse_noise_image((128, 128)), _make_structured_signal_image((128, 128))], axis=0)
    zarr_path = _write_zarr(tmp_path / "sample.zarr", volume, chunks=(1, 64, 64))
    output_json = tmp_path / "qc.json"
    results = run_image_qc(
        input_zarr=zarr_path,
        output_json=output_json,
        config=ImageQcConfig(max_slices=2, grading_enabled=True, show_progress=False),
    )
    assert output_json.exists()
    assert results["source"]["streaming"] is True
    assert len(results["slice_metrics"]) == 2
    assert results["grading"]["overall_verdict"] in {"pass", "warn", "fail"}
    assert "timing_breakdown" in results
    assert results["timing_breakdown"]["total_seconds"] >= 0
    assert any(step["name"] == "global_histogram" for step in results["timing_breakdown"]["steps"])


def test_progress_tracker_renders_distribution_bar():
    tracker = QcProgressTracker(enabled=True)
    with tracker.step("global_histogram", "Global histogram"):
        time.sleep(0.01)
    with tracker.step("slice_metrics", "Slice metrics"):
        time.sleep(0.02)
    chart = tracker.render_bars()
    assert "Global histogram" in chart
    assert "Slice metrics" in chart
    assert "█" in chart
    assert tracker.to_dict()["steps"][1]["seconds"] >= tracker.to_dict()["steps"][0]["seconds"]


def test_grade_qc_results_flags_high_dark_pixel_ratio():
    results = {
        "global_exposure_dynamic_range": {"dark_pixel_ratio": 0.30, "saturated_pixel_ratio": 0.0},
        "slice_aggregate": {
            "diffuse_signal.diffuse_noise_score.median": 0.05,
            "diffuse_signal.noise_to_signal_area_ratio.median": 0.1,
            "diffuse_signal.large_diffuse_component_fraction.median": 0.02,
            "brightness_uniformity.tile_median_cv.median": 0.05,
            "stripe_artifacts.slab_stripe_score.median": 2.0,
            "contrast.candidate_cnr.median": 5.0,
            "focus.laplacian_variance.median": 100.0,
        },
    }
    grading = grade_qc_results(results)
    dark_rule = next(item for item in grading["rules"] if item["metric"].endswith("dark_pixel_ratio"))
    assert dark_rule["verdict"] == "pass"
    stripe_rule = next(item for item in grading["rules"] if "slab_stripe_score" in item["metric"])
    assert stripe_rule["verdict"] == "fail"
    assert grading["overall_verdict"] == "fail"


def test_grade_qc_results_resolves_slice_aggregate_flat_keys():
    results = {
        "global_exposure_dynamic_range": {
            "dark_pixel_ratio": 0.98,
            "saturated_pixel_ratio": 0.0,
        },
        "slice_aggregate": {
            "brightness_uniformity.tile_median_cv.median": 0.05,
            "stripe_artifacts.slab_stripe_score.median": 0.03,
            "diffuse_signal.diffuse_noise_score.median": 0.05,
            "diffuse_signal.noise_to_signal_area_ratio.median": 0.1,
            "diffuse_signal.large_diffuse_component_fraction.median": 0.02,
            "contrast.candidate_cnr.median": 5.0,
            "focus.laplacian_variance.median": 100.0,
        },
    }
    grading = grade_qc_results(results)
    stripe_rule = next(
        item for item in grading["rules"] if "slab_stripe_score" in item["metric"]
    )
    assert stripe_rule["value"] == 0.03
    assert stripe_rule["verdict"] == "pass"
    assert stripe_rule.get("reason") != "metric_missing"
