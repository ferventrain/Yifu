from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def _mad(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


def _robust_std(values: np.ndarray) -> float:
    return 1.4826 * _mad(values)


def _clip_ratio(numerator: float, denominator: float, *, default: float = 0.0) -> float:
    if denominator <= 0:
        return default
    return float(numerator / denominator)


def _percentile_dict(values: np.ndarray, percentiles: tuple[int, ...]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return {f"p{p}": 0.0 for p in percentiles}
    pts = np.percentile(arr, list(percentiles))
    return {f"p{int(p)}": float(v) for p, v in zip(percentiles, pts, strict=True)}


def dtype_range(dtype: np.dtype) -> tuple[float, float]:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return float(info.min), float(info.max)
    if np.issubdtype(dtype, np.floating):
        return 0.0, 1.0
    return 0.0, 255.0


DEFAULT_DARK_PIXEL_THRESHOLD = 100.0
DEFAULT_LOW_CONTRAST_THRESHOLD = 0.08


def compute_exposure_dynamic_range_metrics(
    values: np.ndarray,
    *,
    dtype_min: float | None = None,
    dtype_max: float | None = None,
    saturation_margin: float = 0.001,
    dark_pixel_threshold: float = DEFAULT_DARK_PIXEL_THRESHOLD,
) -> dict[str, float]:
    arr = np.asarray(values)
    flat = arr.astype(np.float64, copy=False).ravel()
    if flat.size == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "saturated_pixel_ratio": 0.0,
            "dark_pixel_ratio": 0.0,
            "robust_dynamic_range": 0.0,
            "dynamic_range_utilization": 0.0,
            "dark_pixel_threshold": float(dark_pixel_threshold),
            **_percentile_dict(flat, (1, 5, 25, 50, 75, 95, 99)),
        }

    if dtype_min is None or dtype_max is None:
        dmin, dmax = dtype_range(arr.dtype)
    else:
        dmin, dmax = float(dtype_min), float(dtype_max)

    span = max(dmax - dmin, 1.0)
    sat_threshold = dmax - saturation_margin * span

    percentiles = _percentile_dict(flat, (1, 5, 25, 50, 75, 95, 99))
    p1 = percentiles["p1"]
    p99 = percentiles["p99"]
    robust_dynamic_range = max(p99 - p1, 0.0)

    return {
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        **percentiles,
        "saturated_pixel_ratio": float(np.mean(flat >= sat_threshold)),
        "dark_pixel_ratio": float(np.mean(flat < float(dark_pixel_threshold))),
        "robust_dynamic_range": robust_dynamic_range,
        "dynamic_range_utilization": _clip_ratio(robust_dynamic_range, span),
        "dark_pixel_threshold": float(dark_pixel_threshold),
    }


def _tile_grid(image_2d: np.ndarray, tile_size: int) -> tuple[np.ndarray, int, int]:
    image = np.asarray(image_2d, dtype=np.float64)
    tile = max(int(tile_size), 8)
    height, width = image.shape
    n_rows = max(height // tile, 1)
    n_cols = max(width // tile, 1)
    cropped = image[: n_rows * tile, : n_cols * tile]
    row_size = max(cropped.shape[0] // n_rows, 1)
    col_size = max(cropped.shape[1] // n_cols, 1)
    cropped = cropped[: n_rows * row_size, : n_cols * col_size]
    blocks = cropped.reshape(n_rows, row_size, n_cols, col_size)
    return blocks, n_rows, n_cols


def _tile_medians(image_2d: np.ndarray, tile_size: int) -> np.ndarray:
    blocks, n_rows, n_cols = _tile_grid(image_2d, tile_size)
    reshaped = blocks.reshape(n_rows, n_cols, -1)
    return np.median(reshaped, axis=2)


def _tile_percentile(image_2d: np.ndarray, tile_size: int, percentile: float) -> np.ndarray:
    blocks, n_rows, n_cols = _tile_grid(image_2d, tile_size)
    reshaped = blocks.reshape(n_rows, n_cols, -1)
    return np.percentile(reshaped, percentile, axis=2)


def _slab_means(image_2d: np.ndarray, axis: int, n_slabs: int) -> np.ndarray:
    image = np.asarray(image_2d, dtype=np.float64)
    length = image.shape[axis]
    n = max(int(n_slabs), 4)
    slab_size = max(length // n, 1)
    usable = slab_size * n
    if axis == 1:
        cropped = image[:, :usable]
        slabs = cropped.reshape(cropped.shape[0], n, slab_size)
        return slabs.mean(axis=(0, 2))
    cropped = image[:usable, :]
    slabs = cropped.reshape(n, slab_size, cropped.shape[1])
    return slabs.mean(axis=(1, 2))


def _profile_stats(profile: np.ndarray) -> dict[str, float]:
    arr = np.asarray(profile, dtype=np.float64).ravel()
    if arr.size == 0:
        return {"cv": 0.0, "range": 0.0, "max_min_ratio": 0.0, "outlier_fraction": 0.0, "slope": 0.0}
    mean = float(np.mean(arr))
    med = float(np.median(arr))
    mad = _mad(arr)
    robust_sigma = max(1.4826 * mad, 1e-6)
    outlier_fraction = float(np.mean(np.abs(arr - med) > 3.0 * robust_sigma))
    minimum = float(np.min(arr))
    maximum = float(np.max(arr))
    positions = np.arange(arr.size, dtype=np.float64)
    slope = float(np.polyfit(positions, arr, 1)[0]) if arr.size >= 2 else 0.0
    return {
        "cv": _clip_ratio(float(np.std(arr)), abs(mean)),
        "range": _clip_ratio(maximum - minimum, abs(mean)),
        "max_min_ratio": _clip_ratio(maximum, max(minimum, 1e-6), default=0.0),
        "outlier_fraction": outlier_fraction,
        "slope": slope,
    }


def compute_brightness_uniformity_metrics(
    image_2d: np.ndarray,
    *,
    tile_size: int = 256,
    n_slabs: int = 32,
    center_fraction: float = 0.5,
    edge_fraction: float = 0.1,
) -> dict[str, float]:
    image = np.asarray(image_2d, dtype=np.float64)
    if image.size == 0:
        return {
            "tile_median_cv": 0.0,
            "x_slab_cv": 0.0,
            "y_slab_cv": 0.0,
            "x_slab_slope": 0.0,
            "y_slab_slope": 0.0,
            "x_slab_max_min_ratio": 0.0,
            "y_slab_max_min_ratio": 0.0,
            "center_to_edge_ratio": 0.0,
        }

    tile_medians = _tile_medians(image, tile_size).ravel()
    tile_mean = float(np.mean(tile_medians)) if tile_medians.size else 0.0
    tile_median_cv = _clip_ratio(float(np.std(tile_medians)), abs(tile_mean))

    x_stats = _profile_stats(_slab_means(image, axis=0, n_slabs=n_slabs))
    y_stats = _profile_stats(_slab_means(image, axis=1, n_slabs=n_slabs))

    height, width = image.shape
    edge_y = max(int(height * edge_fraction), 1)
    edge_x = max(int(width * edge_fraction), 1)
    cy0 = int(height * (0.5 - center_fraction / 2))
    cy1 = int(height * (0.5 + center_fraction / 2))
    cx0 = int(width * (0.5 - center_fraction / 2))
    cx1 = int(width * (0.5 + center_fraction / 2))
    edge_mask = np.ones(image.shape, dtype=bool)
    edge_mask[edge_y : height - edge_y, edge_x : width - edge_x] = False
    center = image[cy0:cy1, cx0:cx1]
    edge_values = image[edge_mask]
    center_mean = float(np.mean(center)) if center.size else 0.0
    edge_mean = float(np.mean(edge_values)) if edge_values.size else 0.0

    return {
        "tile_median_cv": tile_median_cv,
        "x_slab_cv": x_stats["cv"],
        "y_slab_cv": y_stats["cv"],
        "x_slab_slope": x_stats["slope"],
        "y_slab_slope": y_stats["slope"],
        "x_slab_max_min_ratio": x_stats["max_min_ratio"],
        "y_slab_max_min_ratio": y_stats["max_min_ratio"],
        "center_to_edge_ratio": _clip_ratio(center_mean, max(edge_mean, 1e-6)),
    }


def otsu_threshold(values: np.ndarray, *, bins: int = 256) -> float:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if vmax <= vmin:
        return vmin
    hist, bin_edges = np.histogram(arr, bins=bins, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    prob = hist / max(hist.sum(), 1.0)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * bin_centers)
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / np.maximum(omega * (1.0 - omega), 1e-12)
    idx = int(np.argmax(sigma_b))
    return float(bin_centers[idx])


def compute_contrast_metrics(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return {
            "robust_contrast_score": 0.0,
            "percentile_contrast_ratio": 0.0,
            "otsu_threshold": 0.0,
            "otsu_foreground_ratio": 0.0,
            "otsu_background_ratio": 0.0,
            "candidate_cnr": 0.0,
            "median_foreground": 0.0,
            "median_background": 0.0,
            "mad_background": 0.0,
        }

    percentiles = _percentile_dict(arr, (25, 50, 75, 95))
    p25 = percentiles["p25"]
    p50 = percentiles["p50"]
    p75 = percentiles["p75"]
    p95 = percentiles["p95"]
    robust_sigma = max(_robust_std(arr), 1e-6)
    iqr = max(p75 - p25, 1e-6)

    threshold = otsu_threshold(arr)
    fg = arr >= threshold
    bg = ~fg
    median_fg = float(np.median(arr[fg])) if np.any(fg) else float(np.max(arr))
    median_bg = float(np.median(arr[bg])) if np.any(bg) else float(np.min(arr))
    mad_bg = _mad(arr[bg]) if np.any(bg) else 0.0
    candidate_cnr = _clip_ratio(median_fg - median_bg, max(1.4826 * mad_bg, 1e-6))

    return {
        "robust_contrast_score": _clip_ratio(p95 - p50, robust_sigma),
        "percentile_contrast_ratio": _clip_ratio(p95 - p50, iqr),
        "otsu_threshold": threshold,
        "otsu_foreground_ratio": float(np.mean(fg)),
        "otsu_background_ratio": float(np.mean(bg)),
        "candidate_cnr": candidate_cnr,
        "median_foreground": median_fg,
        "median_background": median_bg,
        "mad_background": mad_bg,
    }


def _connected_component_count(tile_mask: np.ndarray) -> int:
    labeled, count = ndimage.label(tile_mask)
    return int(count)


def _largest_component_fraction(mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    labeled, _ = ndimage.label(mask)
    sizes = np.bincount(labeled.ravel())[1:]
    if sizes.size == 0:
        return 0.0
    return float(np.max(sizes) / mask.size)


def compute_diffuse_signal_metrics(
    image_2d: np.ndarray,
    *,
    tile_size: int = 256,
    global_percentiles: dict[str, float] | None = None,
    low_contrast_threshold: float = DEFAULT_LOW_CONTRAST_THRESHOLD,
) -> dict[str, float]:
    image = np.asarray(image_2d, dtype=np.float64)
    flat = image.ravel()
    if flat.size == 0:
        return {
            "bbsi": 0.0,
            "tile_p90_bright_fraction": 0.0,
            "diffuse_brightness_ratio": 0.0,
            "ebi_z": 0.0,
            "sss": 0.0,
            "ndss": 0.0,
            "ndss_area": 0.0,
            "ndss_intensity": 0.0,
            "ndss_diffusion": 0.0,
            "diffuse_noise_fraction": 0.0,
            "structured_signal_fraction": 0.0,
            "noise_to_signal_area_ratio": 0.0,
            "large_diffuse_component_fraction": 0.0,
            "diffuse_noise_score": 0.0,
            "mean_diffuse_tile_contrast": 0.0,
            "mean_structured_tile_contrast": 0.0,
        }

    if global_percentiles is None:
        global_percentiles = _percentile_dict(flat, (10, 25, 50, 75, 90))

    p10 = global_percentiles.get("p10", float(np.percentile(flat, 10)))
    p25 = global_percentiles.get("p25", float(np.percentile(flat, 25)))
    p50 = global_percentiles.get("p50", float(np.percentile(flat, 50)))
    p75 = global_percentiles.get("p75", float(np.percentile(flat, 75)))
    p90 = global_percentiles.get("p90", float(np.percentile(flat, 90)))

    tile_p10 = _tile_percentile(image, tile_size, 10.0)
    tile_p25 = _tile_percentile(image, tile_size, 25.0)
    tile_p90 = _tile_percentile(image, tile_size, 90.0)
    tile_medians = _tile_medians(image, tile_size)
    tile_spread = tile_p90 - tile_p10
    tile_contrast_ratio = tile_spread / np.maximum(tile_medians, 1.0)
    spread_floor = max(0.15 * float(np.median(tile_medians)), 50.0)

    structured_tile_mask = (tile_contrast_ratio >= low_contrast_threshold) & (tile_spread >= spread_floor)
    diffuse_tile_mask = (tile_medians > p25) & (tile_contrast_ratio < low_contrast_threshold) & (~structured_tile_mask)

    diffuse_noise_fraction = float(np.mean(diffuse_tile_mask))
    structured_signal_fraction = float(np.mean(structured_tile_mask))
    noise_to_signal_area_ratio = _clip_ratio(
        diffuse_noise_fraction,
        max(structured_signal_fraction, 1e-6),
    )
    large_diffuse_component_fraction = _largest_component_fraction(diffuse_tile_mask)
    sss = float(_connected_component_count(diffuse_tile_mask))

    flat_diffuse = tile_contrast_ratio[diffuse_tile_mask]
    flat_structured = tile_contrast_ratio[structured_tile_mask]
    mean_diffuse_tile_contrast = float(np.mean(flat_diffuse)) if flat_diffuse.size else 0.0
    mean_structured_tile_contrast = float(np.mean(flat_structured)) if flat_structured.size else 0.0

    if np.any(diffuse_tile_mask):
        bbsi = _clip_ratio(float(np.mean(tile_p25[diffuse_tile_mask])), max(p50, 1e-6))
        intensity_lift = _clip_ratio(float(np.mean(tile_medians[diffuse_tile_mask]) - p50), max(p90 - p50, 1e-6))
        flatness = min(
            _clip_ratio(low_contrast_threshold - mean_diffuse_tile_contrast, max(low_contrast_threshold, 1e-6)),
            1.0,
        )
    else:
        bbsi = _clip_ratio(float(np.mean(tile_p25)), max(p50, 1e-6))
        intensity_lift = 0.0
        flatness = 0.0

    tile_p90_bright_fraction = float(np.mean(tile_p90 > p75))
    diffuse_brightness_ratio = _clip_ratio(p50 - p10, max(p90 - p10, 1e-6))

    height, width = image.shape
    edge_y = max(int(height * 0.05), 1)
    edge_x = max(int(width * 0.05), 1)
    edge_mask = np.ones(image.shape, dtype=bool)
    edge_mask[edge_y : height - edge_y, edge_x : width - edge_x] = False
    edge_mean = float(np.mean(image[edge_mask]))
    center = image[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
    center_mean = float(np.mean(center)) if center.size else float(np.mean(flat))
    ebi_z = _clip_ratio(edge_mean - center_mean, max(_robust_std(flat), 1e-6))

    ndss_area = min(diffuse_noise_fraction, 1.0)
    ndss_intensity = min(intensity_lift, 1.0)
    ndss_diffusion = min(flatness, 1.0)
    ndss = 0.4 * ndss_area + 0.3 * ndss_intensity + 0.3 * ndss_diffusion
    diffuse_noise_score = min(
        0.45 * diffuse_noise_fraction
        + 0.30 * large_diffuse_component_fraction
        + 0.25 * ndss_diffusion,
        1.0,
    )

    return {
        "bbsi": bbsi,
        "tile_p90_bright_fraction": tile_p90_bright_fraction,
        "diffuse_brightness_ratio": diffuse_brightness_ratio,
        "ebi_z": ebi_z,
        "sss": sss,
        "ndss": ndss,
        "ndss_area": ndss_area,
        "ndss_intensity": ndss_intensity,
        "ndss_diffusion": ndss_diffusion,
        "diffuse_noise_fraction": diffuse_noise_fraction,
        "structured_signal_fraction": structured_signal_fraction,
        "noise_to_signal_area_ratio": noise_to_signal_area_ratio,
        "large_diffuse_component_fraction": large_diffuse_component_fraction,
        "diffuse_noise_score": diffuse_noise_score,
        "mean_diffuse_tile_contrast": mean_diffuse_tile_contrast,
        "mean_structured_tile_contrast": mean_structured_tile_contrast,
    }


def compute_stripe_metrics(
    image_2d: np.ndarray,
    *,
    n_slabs: int = 32,
    fft_enabled: bool = False,
) -> dict[str, float]:
    image = np.asarray(image_2d, dtype=np.float64)
    if image.size == 0:
        base = {
            "row_profile_cv": 0.0,
            "col_profile_cv": 0.0,
            "row_profile_range": 0.0,
            "col_profile_range": 0.0,
            "row_profile_outlier_fraction": 0.0,
            "col_profile_outlier_fraction": 0.0,
            "x_slab_cv": 0.0,
            "y_slab_cv": 0.0,
            "slab_stripe_score": 0.0,
        }
        if fft_enabled:
            base["row_fft_peak_ratio"] = 0.0
            base["col_fft_peak_ratio"] = 0.0
        return base

    row_profile = image.mean(axis=1)
    col_profile = image.mean(axis=0)
    row_stats = _profile_stats(row_profile)
    col_stats = _profile_stats(col_profile)
    x_slab_cv = _profile_stats(_slab_means(image, axis=0, n_slabs=n_slabs))["cv"]
    y_slab_cv = _profile_stats(_slab_means(image, axis=1, n_slabs=n_slabs))["cv"]
    slab_stripe_score = float(max(row_stats["cv"], col_stats["cv"], x_slab_cv, y_slab_cv))

    metrics = {
        "row_profile_cv": row_stats["cv"],
        "col_profile_cv": col_stats["cv"],
        "row_profile_range": row_stats["range"],
        "col_profile_range": col_stats["range"],
        "row_profile_outlier_fraction": row_stats["outlier_fraction"],
        "col_profile_outlier_fraction": col_stats["outlier_fraction"],
        "x_slab_cv": x_slab_cv,
        "y_slab_cv": y_slab_cv,
        "slab_stripe_score": slab_stripe_score,
    }

    if fft_enabled:
        metrics["row_fft_peak_ratio"] = _fft_peak_ratio(row_profile)
        metrics["col_fft_peak_ratio"] = _fft_peak_ratio(col_profile)

    return metrics


def _fft_peak_ratio(profile: np.ndarray) -> float:
    arr = np.asarray(profile, dtype=np.float64).ravel()
    if arr.size < 8:
        return 0.0
    arr = arr - np.mean(arr)
    power = np.abs(np.fft.rfft(arr)) ** 2
    if power.size <= 1:
        return 0.0
    dc = power[0]
    rest = power[1:]
    total = float(np.sum(power))
    if total <= 0:
        return 0.0
    peak = float(np.max(rest))
    return _clip_ratio(peak, max(total - dc, 1e-6))


def compute_focus_metrics(image_2d: np.ndarray) -> dict[str, float]:
    image = np.asarray(image_2d, dtype=np.float64)
    if image.size == 0:
        return {"laplacian_variance": 0.0, "tenengrad_score": 0.0}

    laplacian = ndimage.laplace(image)
    sobel_x = ndimage.sobel(image, axis=1)
    sobel_y = ndimage.sobel(image, axis=0)
    tenengrad = sobel_x * sobel_x + sobel_y * sobel_y
    return {
        "laplacian_variance": float(np.var(laplacian)),
        "tenengrad_score": float(np.mean(tenengrad)),
    }


def compute_slice_metrics(
    image_2d: np.ndarray,
    *,
    tile_size: int = 256,
    n_slabs: int = 32,
    fft_enabled: bool = False,
    global_percentiles: dict[str, float] | None = None,
    dark_pixel_threshold: float = DEFAULT_DARK_PIXEL_THRESHOLD,
    low_contrast_threshold: float = DEFAULT_LOW_CONTRAST_THRESHOLD,
) -> dict[str, dict[str, float]]:
    exposure = compute_exposure_dynamic_range_metrics(
        image_2d,
        dark_pixel_threshold=dark_pixel_threshold,
    )
    percentiles_for_diffuse = global_percentiles or _percentile_dict(
        np.asarray(image_2d).ravel(),
        (10, 25, 50, 75, 90),
    )

    return {
        "exposure_dynamic_range": exposure,
        "brightness_uniformity": compute_brightness_uniformity_metrics(
            image_2d,
            tile_size=tile_size,
            n_slabs=n_slabs,
        ),
        "contrast": compute_contrast_metrics(image_2d),
        "diffuse_signal": compute_diffuse_signal_metrics(
            image_2d,
            tile_size=tile_size,
            global_percentiles=percentiles_for_diffuse,
            low_contrast_threshold=low_contrast_threshold,
        ),
        "stripe_artifacts": compute_stripe_metrics(
            image_2d,
            n_slabs=n_slabs,
            fft_enabled=fft_enabled,
        ),
        "focus": compute_focus_metrics(image_2d),
    }


def flatten_metric_groups(groups: dict[str, dict[str, float]], *, prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    for group_name, metrics in groups.items():
        for metric_name, value in metrics.items():
            key = f"{prefix}{group_name}.{metric_name}" if prefix else f"{group_name}.{metric_name}"
            flat[key] = _safe_float(value)
    return flat


def aggregate_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not metric_dicts:
        return {}
    keys = metric_dicts[0].keys()
    aggregated: dict[str, float] = {}
    for key in keys:
        values = [_safe_float(item.get(key)) for item in metric_dicts]
        aggregated[f"{key}.mean"] = float(np.mean(values))
        aggregated[f"{key}.median"] = float(np.median(values))
        aggregated[f"{key}.min"] = float(np.min(values))
        aggregated[f"{key}.max"] = float(np.max(values))
    return aggregated
