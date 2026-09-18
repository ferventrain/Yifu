"""Equalize regional tissue brightness for viewing, keeping local contrast.

Non-specific stain often makes the exterior brighter than the interior. That
is a slow illumination-like offset, not a reason to crush bright voxels: the
highlights still carry structure. The correction is a log-domain shading
normalize:

1. Build a cheap downsampled volume (every Nth slice, XY resized).
2. Estimate a tight tissue envelope from a bright seed + hole fill, or load a manual/SAM mask.
3. Smooth log-intensity into a regional brightness envelope.
4. Inside tissue, scale by ``(I_ref / field) ** strength`` so inner/outer
   overall brightness matches, while high-frequency detail stays.

CLI::

    python -m pipeline_modules.preprocessing.surface_brightness_homogenize --input_dir ch0 --output_zarr ch0_surface_homogenized.zarr --qc_only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from scipy import ndimage as ndi
from tqdm import tqdm

try:
    from pipeline_modules.preprocessing.tiff_to_zarr import (
        _create_array,
        _open_output_group,
        resolve_compressor,
    )
    from pipeline_modules.preprocessing.zarr_pyramid import add_resolution_pyramid, write_preview_zarr
    from pipeline_modules.utils.errors import ErrorCode, PipelineError
    from pipeline_modules.utils.run_manifest import write_run_manifest
    from pipeline_modules.utils.tiff_stack_io import iter_batch_ranges, resolve_stack_workers
except ImportError:  # pragma: no cover - direct script execution fallback
    from .tiff_to_zarr import _create_array, _open_output_group, resolve_compressor
    from .zarr_pyramid import add_resolution_pyramid, write_preview_zarr
    from ..utils.errors import ErrorCode, PipelineError
    from ..utils.run_manifest import write_run_manifest
    from ..utils.tiff_stack_io import iter_batch_ranges, resolve_stack_workers

logger = logging.getLogger(__name__)


def _configure_logging(json_logs: bool) -> None:
    if json_logs:
        class _JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                payload = {
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }
                return json.dumps(payload, ensure_ascii=False)

        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonFormatter())
        logging.root.handlers.clear()
        logging.root.addHandler(handler)
        logging.root.setLevel(logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _pipeline_error_to_stderr(exc: PipelineError) -> None:
    print(json.dumps(exc.to_dict(), ensure_ascii=False), file=sys.stderr)


def _list_tiff_files(path: Path) -> list[Path]:
    files = sorted(path.glob("*.tif*"))
    if not files:
        raise PipelineError(
            ErrorCode.INPUT_NOT_FOUND,
            "No TIFF files found",
            {"input_dir": str(path)},
        )
    return files


def _coerce_chunk_size(value: str | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, tuple):
        return tuple(int(v) for v in value)
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 3:
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            "chunk_size must be three comma-separated integers",
            {"chunk_size": value},
        )
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def _resolve_output_zarr(input_path: Path, output_zarr: str | Path | None) -> Path:
    if output_zarr is None:
        return input_path.parent / f"{input_path.name}_surface_homogenized.zarr"
    path = Path(output_zarr)
    if path.suffix.lower() != ".zarr":
        return path.with_name(path.name + ".zarr") if path.suffix else path.with_suffix(".zarr")
    return path


def _border_mask(shape: tuple[int, ...]) -> np.ndarray:
    border = np.zeros(shape, dtype=bool)
    border[0] = True
    border[-1] = True
    if len(shape) > 1:
        border[:, 0] = True
        border[:, -1] = True
    if len(shape) > 2:
        border[:, :, 0] = True
        border[:, :, -1] = True
    return border


def _interior_hole_count(tissue: np.ndarray) -> int:
    struct = ndi.generate_binary_structure(tissue.ndim, 1)
    labeled, count = ndi.label(~tissue, structure=struct)
    if count == 0:
        return 0
    touching = np.unique(labeled[_border_mask(tissue.shape)])
    return int(count - np.count_nonzero(touching > 0))


def _seal_single_closed_surface(tissue: np.ndarray, close_iter: int) -> np.ndarray:
    """Seal tunnels, then keep only background that touches the volume border.

    Internal cavities become tissue, so distance-from-surface is a single outer shell.
    """
    sealed = np.asarray(tissue, dtype=bool)
    if close_iter > 0 and sealed.any():
        sealed = ndi.binary_closing(sealed, iterations=int(close_iter))
    struct = ndi.generate_binary_structure(sealed.ndim, 1)
    labeled, count = ndi.label(~sealed, structure=struct)
    if count == 0:
        return sealed
    touching = [int(v) for v in np.unique(labeled[_border_mask(sealed.shape)]) if int(v) != 0]
    exterior = np.isin(labeled, touching) if touching else np.zeros_like(sealed)
    solid = ~exterior
    if solid.any():
        solid = _largest_component(solid)
    return solid


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndi.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(sizes.argmax())


def estimate_tissue_mask(
    volume: np.ndarray,
    *,
    close_iter: int = 0,
    seed_pct: float = 90.0,
    grow_pct: float = 84.0,
    max_grow_px: float = 32.0,
    erode_px: float = 16.0,
    sampling_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, dict[str, float]]:
    """Tight tissue envelope: bright seed, limited-distance grow, fill interior."""
    bg = float(np.percentile(volume, 50))
    p90 = float(np.percentile(volume, 90))
    floor = max(bg + 10.0, p90)
    pool = volume[volume > floor]
    if pool.size < 1024:
        pool = volume[volume > bg + 1.0]
    if pool.size == 0:
        raise PipelineError(ErrorCode.EMPTY_RESULT, "No voxels above background for tissue seeding")
    high_pct = max(float(seed_pct), float(grow_pct))
    low_pct = min(float(seed_pct), float(grow_pct))
    high = float(np.percentile(pool, high_pct))
    low = float(np.percentile(pool, low_pct))
    if low > high:
        low = high

    seed = volume > high
    seed = ndi.binary_opening(seed, iterations=1)
    dist_from_seed = ndi.distance_transform_edt(~seed, sampling=sampling_zyx)
    allowed = (volume > low) & (dist_from_seed <= max(float(max_grow_px), 1e-3))
    tissue = ndi.binary_propagation(seed, mask=allowed)
    tissue = ndi.binary_fill_holes(tissue)
    tissue = _largest_component(tissue)

    erode_iter = max(0, int(round(float(erode_px) / max(float(sampling_zyx[1]), 1e-3))))
    if erode_iter > 0 and tissue.any():
        eroded = ndi.binary_erosion(tissue, iterations=erode_iter)
        if eroded.any():
            tissue = _largest_component(eroded)

    holes_before = _interior_hole_count(tissue)
    tissue = _seal_single_closed_surface(tissue, close_iter=int(close_iter))
    holes_after = _interior_hole_count(tissue)

    if not tissue.any():
        raise PipelineError(
            ErrorCode.EMPTY_RESULT,
            "Tissue mask is empty after tightening; lower --seed_pct, --grow_pct or --erode_px",
            {
                "threshold": high,
                "grow_threshold": low,
                "seed_pct": seed_pct,
                "grow_pct": grow_pct,
                "max_grow_px": max_grow_px,
                "erode_px": erode_px,
            },
        )
    stats = {
        "threshold": high,
        "grow_threshold": low,
        "seed_floor": floor,
        "seed_pct": float(high_pct),
        "grow_pct": float(low_pct),
        "max_grow_px": float(max_grow_px),
        "erode_px": float(erode_px),
        "close_iter": float(close_iter),
        "holes_before": float(holes_before),
        "holes_after": float(holes_after),
        "tissue_fraction": float(tissue.mean()),
    }
    return tissue, stats


def load_tissue_mask(
    path: str | Path,
    target_shape: tuple[int, int, int],
    *,
    close_iter: int = 0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Load a manual / SAM tissue mask and resample it onto the QC grid."""
    mask_path = Path(path)
    if not mask_path.exists():
        raise PipelineError(
            ErrorCode.INPUT_NOT_FOUND,
            "Tissue mask not found",
            {"tissue_mask": str(mask_path)},
        )
    suffix = mask_path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        data = np.asarray(tifffile.imread(str(mask_path)))
    elif suffix == ".npy":
        data = np.asarray(np.load(str(mask_path)))
    else:
        try:
            from pipeline_modules.utils.zarr_io import open_zarr_array
        except ImportError:  # pragma: no cover
            from ..utils.zarr_io import open_zarr_array
        arr = open_zarr_array(mask_path)
        attrs = dict(getattr(arr, "attrs", {}) or {})
        names = list(attrs.get("channel_names") or [])
        data = np.asarray(arr)
        if data.ndim == 4:
            axis = int(attrs.get("channel_axis") or 0)
            if axis != 0:
                data = np.moveaxis(data, axis, 0)
            index = names.index("tissue_mask") if "tissue_mask" in names else 0
            data = data[index]
    if data.ndim != 3:
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            "Tissue mask must be a 3D volume",
            {"tissue_mask": str(mask_path), "shape": list(data.shape)},
        )
    tissue = np.asarray(data) > 0
    target = tuple(int(v) for v in target_shape)
    if tissue.shape != target:
        tz, ty, tx = target
        mz, my, mx = (int(v) for v in tissue.shape)
        if (my, mx) != (ty, tx):
            tissue = ndi.zoom(
                tissue.astype(np.uint8),
                (1.0, ty / max(my, 1), tx / max(mx, 1)),
                order=0,
            ) > 0
            mz, my, mx = (int(v) for v in tissue.shape)
        if mz != tz:
            fitted = np.zeros((tz, ty, tx), dtype=bool)
            n = min(mz, tz)
            fitted[:n] = tissue[:n]
            logger.info("Tissue mask Z fitted %d -> %d by start-crop/pad, not zoom", mz, tz)
            tissue = fitted
    holes_before = _interior_hole_count(tissue)
    if int(close_iter) > 0:
        tissue = _seal_single_closed_surface(tissue, close_iter=int(close_iter))
    holes_after = _interior_hole_count(tissue)
    if not tissue.any():
        raise PipelineError(
            ErrorCode.EMPTY_RESULT,
            "Loaded tissue mask is empty",
            {"tissue_mask": str(mask_path), "shape": list(target_shape)},
        )
    stats = {
        "threshold": 0.0,
        "grow_threshold": 0.0,
        "seed_floor": 0.0,
        "seed_pct": 0.0,
        "grow_pct": 0.0,
        "max_grow_px": 0.0,
        "erode_px": 0.0,
        "close_iter": float(close_iter),
        "holes_before": float(holes_before),
        "holes_after": float(holes_after),
        "tissue_fraction": float(tissue.mean()),
        "tissue_source": "manual",
        "tissue_mask_path": str(mask_path),
    }
    return tissue, stats


def estimate_shading_field(
    volume: np.ndarray,
    tissue: np.ndarray,
    *,
    sigma_ds: float,
    interior_pct: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Low-frequency brightness field and a tissue intensity reference."""
    if not tissue.any():
        raise PipelineError(ErrorCode.EMPTY_RESULT, "No tissue voxels for shading field")
    i_ref = float(np.percentile(volume[tissue], float(interior_pct)))
    filled = np.where(tissue, np.maximum(volume, 1.0), max(i_ref, 1.0)).astype(np.float32, copy=False)
    sigma = max(float(sigma_ds), 0.5)
    # Geometric-mean field: regional envelope, not local peaks.
    field = np.exp(ndi.gaussian_filter(np.log(filled), sigma=sigma))
    field = np.maximum(field, 1.0)
    return field.astype(np.float32, copy=False), {
        "i_ref": i_ref,
        "field_sigma_ds": float(sigma),
        "field_min": float(field.min()),
        "field_max": float(field.max()),
        "tissue_voxels": int(tissue.sum()),
    }


def _distance_transform_edt(mask: np.ndarray, sampling_zyx: tuple[float, float, float]) -> np.ndarray:
    """Memory-efficient EDT; scipy allocates a (3, Z, Y, X) index buffer on large 3D volumes."""
    aniso = tuple(float(v) for v in sampling_zyx)
    binary = np.asarray(mask, dtype=bool)
    try:
        import edt as edt_mod

        return edt_mod.edt(binary, anisotropy=aniso, parallel=1).astype(np.float32, copy=False)
    except ImportError:
        return ndi.distance_transform_edt(binary, sampling=aniso).astype(np.float32, copy=False)


def surface_membership(
    tissue: np.ndarray,
    *,
    d_max_px: float,
    d_out_px: float,
    sampling_zyx: tuple[float, float, float],
    smooth_sigma_ds: float = 0.4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binary surface shell: 1 throughout the inward/outward band, 0 elsewhere."""
    dist_in = _distance_transform_edt(tissue, sampling_zyx)
    dist_out = _distance_transform_edt(~tissue, sampling_zyx)
    in_shell = (tissue & (dist_in <= max(float(d_max_px), 1e-3))) | (
        (~tissue) & (dist_out <= max(float(d_out_px), 1e-3))
    )
    membership = in_shell.astype(np.float32)
    if smooth_sigma_ds > 0:
        membership = ndi.gaussian_filter(membership, sigma=float(smooth_sigma_ds))
        membership = np.clip(membership, 0.0, 1.0)
    return membership.astype(np.float32), dist_in, dist_out


def estimate_intensity_anchors(
    volume: np.ndarray,
    tissue: np.ndarray,
    dist_in: np.ndarray,
    dist_out: np.ndarray,
    *,
    d_max_px: float,
    d_out_px: float,
    ref_band_px: float,
    interior_pct: float,
    bright_hi_pct: float,
) -> dict[str, float]:
    """Intensity reference from the band just inside the surface, not the core."""
    inner = tissue & (dist_in > float(d_max_px)) & (dist_in <= float(d_max_px) + float(ref_band_px))
    if inner.sum() < 64:
        inner = tissue & (dist_in > float(d_max_px))
    if inner.sum() < 64:
        inner = tissue
    if not inner.any():
        raise PipelineError(ErrorCode.EMPTY_RESULT, "No interior voxels for intensity reference")

    surface = (tissue & (dist_in <= float(d_max_px))) | ((~tissue) & (dist_out <= float(d_out_px)))
    i_ref = float(np.percentile(volume[inner], float(interior_pct)))
    surface_vals = volume[surface] if surface.any() else volume[inner]
    bright = surface_vals[surface_vals > i_ref]
    if bright.size >= 16:
        i_hi = float(np.percentile(bright, float(bright_hi_pct)))
    else:
        i_hi = float(np.percentile(volume[tissue], min(99.0, float(bright_hi_pct) + 5.0)))
    if i_hi <= i_ref + 1.0:
        i_hi = i_ref + 1.0
    return {
        "i_ref": i_ref,
        "i_hi": i_hi,
        "inner_voxels": int(inner.sum()),
        "surface_voxels": int(surface.sum()),
    }


def _masked_log_field(values: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Slow log-brightness field using only ``mask`` voxels (normalized convolution)."""
    weight = mask.astype(np.float32)
    work = np.log(np.maximum(values.astype(np.float32), 1.0)) * weight
    num = ndi.gaussian_filter(work, sigma=float(sigma))
    den = ndi.gaussian_filter(weight, sigma=float(sigma))
    field = np.exp(num / np.maximum(den, 1e-6))
    return np.where(den > 1e-3, field, 1.0).astype(np.float32)


def shell_correction_weight(shape: tuple[int, ...]) -> np.ndarray:
    """Correction applies to every voxel: tissue and all exterior."""
    return np.ones(shape, dtype=np.float32)


def _depth_gain_at(frac: float, g0: float, g1: float, gamma: float) -> float:
    t = min(max(float(frac), 0.0), 1.0) ** float(gamma)
    return float(g0 + (g1 - g0) * t)


def surface_depth_gain(
    volume: np.ndarray,
    tissue: np.ndarray,
    dist_in: np.ndarray,
    dist_out: np.ndarray,
    *,
    d_out_px: float,
    max_gain: float = 6.0,
    surface_gain: float = 0.05,
    depth_gamma: float = 1.5,
    exterior_gain: float = 0.01,
    target_scale: float = 1.0,
    field_sigma_ds: float = 0.0,
    local_suppress_strength: float = 3.0,
    local_suppress_below_px: float = 24.0,
    local_suppress_pct: float = 50.0,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    """Depth gain inside tissue, then suppress locally bright sub-surface patches.

    Local suppression is multiplicative against a slow log-brightness field, so
    high-frequency contrast is kept and voxels are not clipped to a ceiling.
    Gain is never raised by this step (suppression-only).
    """
    del d_out_px, dist_out
    g0 = max(float(surface_gain), 1e-4)
    g1 = max(float(max_gain), g0 + 1e-4) * max(float(target_scale), 0.05)
    g_ext = max(float(exterior_gain), 1e-4)
    gamma = max(float(depth_gamma), 0.25)
    gain = np.full(volume.shape, g_ext, dtype=np.float32)
    suppress_map = np.ones(volume.shape, dtype=np.float32)
    stats: dict[str, float] = {
        "model_a": 0.0,
        "model_mu": 0.0,
        "model_b": 0.0,
        "i_ref": 1.0,
        "used_fit": 0.0,
        "gain_min": g_ext,
        "gain_max": g_ext,
        "profile_bins": 0.0,
        "target_scale": float(target_scale),
        "shallow_level": 1.0,
        "deep_level": 1.0,
        "max_gain": g1,
        "surface_gain": g0,
        "depth_gamma": gamma,
        "exterior_gain": g_ext,
        "d_deep_px": 0.0,
        "d_deep_p95_px": 0.0,
        "gain_at_25": _depth_gain_at(0.25, g0, g1, gamma),
        "gain_at_50": _depth_gain_at(0.50, g0, g1, gamma),
        "gain_at_75": _depth_gain_at(0.75, g0, g1, gamma),
        "local_suppress_min": 1.0,
        "local_suppress_max": 1.0,
        "local_suppress_target": 1.0,
    }
    if not tissue.any():
        return gain, stats, suppress_map

    depths = dist_in[tissue]
    d_deep = float(np.max(depths))
    d_p95 = float(np.percentile(depths, 95))
    d_deep = max(d_deep, 1.0)
    frac = np.clip(dist_in.astype(np.float64) / d_deep, 0.0, 1.0)
    tissue_gain = (g0 + (g1 - g0) * np.power(frac, gamma)).astype(np.float32)
    gain[tissue] = tissue_gain[tissue]

    values = np.maximum(volume[tissue].astype(np.float64), 1.0)
    shallow_sel = depths <= max(d_deep * 0.15, 1.0)
    deep_sel = depths >= d_deep * 0.85
    shallow_level = float(np.percentile(values[shallow_sel], 50)) if shallow_sel.any() else float(np.percentile(values, 90))
    deep_level = float(np.percentile(values[deep_sel], 50)) if deep_sel.any() else float(np.percentile(values, 40))

    strength = max(float(local_suppress_strength), 0.0)
    sigma = float(field_sigma_ds)
    d_start = max(float(local_suppress_below_px), 0.0)
    if strength > 0.0 and sigma > 0.5 and tissue.any():
        deep_zone = tissue & (dist_in >= d_start)
        if deep_zone.any():
            proxy = np.maximum(volume.astype(np.float32) * tissue_gain, 1.0)
            target = float(np.percentile(proxy[deep_zone], float(np.clip(local_suppress_pct, 1.0, 99.0))))
            target = max(target, 1.0)
            field = np.maximum(_masked_log_field(proxy, deep_zone, sigma), 1.0)
            ratio = np.minimum(target / field, 1.0)
            suppress = np.power(ratio, strength).astype(np.float32)
            fade = np.clip((dist_in.astype(np.float32) - np.float32(d_start)) / np.float32(max(d_start, 8.0)), 0.0, 1.0)
            fade = np.where(tissue, fade, 0.0).astype(np.float32)
            blended = tissue_gain * (1.0 + fade * (suppress - 1.0))
            gain[tissue] = blended[tissue]
            suppress_map = (1.0 + fade * (suppress - 1.0)).astype(np.float32)
            stats["local_suppress_min"] = float(suppress[deep_zone].min())
            stats["local_suppress_max"] = float(suppress[deep_zone].max())
            stats["local_suppress_target"] = target
            stats["field_min"] = float(field[deep_zone].min())
            stats["field_max"] = float(field[deep_zone].max())

    stats.update(
        {
            "model_a": float(max(shallow_level - deep_level, 0.0)),
            "model_b": deep_level,
            "shallow_level": shallow_level,
            "deep_level": deep_level,
            "i_ref": float(np.sqrt(max(shallow_level, 1.0) * max(deep_level, 1.0))),
            "d_deep_px": d_deep,
            "d_deep_p95_px": d_p95,
            "gain_min": float(gain.min()),
            "gain_max": float(gain.max()),
        }
    )
    return gain, stats, suppress_map


def apply_depth_gain(
    image: np.ndarray,
    gain: np.ndarray,
    weight: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """Multiply by the depth model gain inside a soft tissue weight."""
    dtype_in = image.dtype
    work = image.astype(np.float32, copy=False)
    blend = np.clip(float(alpha) * weight.astype(np.float32, copy=False), 0.0, 1.0)
    result = (1.0 - blend) * work + blend * (work * gain.astype(np.float32, copy=False))
    result = np.clip(result, 0.0, None)
    if np.issubdtype(dtype_in, np.integer):
        max_val = np.iinfo(dtype_in).max
        return np.clip(result, 0, max_val).astype(dtype_in)
    return result.astype(dtype_in, copy=False)


def clamp_bright_in_shell(
    image: np.ndarray,
    membership: np.ndarray,
    *,
    i_ref: float,
    alpha: float,
    bright_ramp: float = 64.0,
) -> np.ndarray:
    """Press shell voxels brighter than ``i_ref`` down to ``i_ref``."""
    dtype_in = image.dtype
    work = image.astype(np.float32, copy=False)
    ramp = max(float(bright_ramp), 1.0)
    bright = np.clip((work - float(i_ref)) / ramp, 0.0, 1.0)
    weight = np.clip(membership.astype(np.float32, copy=False) * bright, 0.0, 1.0)
    blend = np.clip(float(alpha) * weight, 0.0, 1.0)
    result = (1.0 - blend) * work + blend * float(i_ref)
    result = np.clip(result, 0.0, None)
    if np.issubdtype(dtype_in, np.integer):
        max_val = np.iinfo(dtype_in).max
        return np.clip(result, 0, max_val).astype(dtype_in)
    return result.astype(dtype_in, copy=False)


def tissue_weight(tissue: np.ndarray, *, smooth_sigma_ds: float = 1.0) -> np.ndarray:
    weight = tissue.astype(np.float32)
    if smooth_sigma_ds > 0:
        weight = ndi.gaussian_filter(weight, sigma=float(smooth_sigma_ds))
        weight = np.clip(weight, 0.0, 1.0)
    return weight


def build_downsampled_volume(
    tiff_files: list[Path],
    *,
    downsample: int,
) -> tuple[np.ndarray, list[int]]:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise PipelineError(
            ErrorCode.DEPENDENCY_MISSING,
            "opencv-python is required",
            {"dependency": "cv2", "error": str(exc)},
        ) from exc

    factor = max(1, int(downsample))
    z_indices = list(range(0, len(tiff_files), factor))
    planes: list[np.ndarray] = []
    for z_index in z_indices:
        image = np.asarray(tifffile.imread(str(tiff_files[z_index])), dtype=np.float32)
        if image.ndim > 2:
            image = np.squeeze(image)
        height, width = image.shape[:2]
        small = cv2.resize(
            image,
            (max(1, width // factor), max(1, height // factor)),
            interpolation=cv2.INTER_AREA,
        )
        planes.append(small.astype(np.float32, copy=False))
    return np.stack(planes, axis=0), z_indices


def resample_plane(
    volume_ds: np.ndarray,
    z_index: int,
    *,
    downsample: int,
    out_hw: tuple[int, int],
) -> np.ndarray:
    import cv2

    factor = max(1, int(downsample))
    z_ds = np.clip(z_index / factor, 0.0, volume_ds.shape[0] - 1.0)
    z0 = int(np.floor(z_ds))
    z1 = min(z0 + 1, volume_ds.shape[0] - 1)
    t = float(z_ds - z0)
    plane = (1.0 - t) * volume_ds[z0] + t * volume_ds[z1]
    height, width = out_hw
    return cv2.resize(plane, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def normalize_slice(
    image: np.ndarray,
    field: np.ndarray,
    weight: np.ndarray,
    *,
    i_ref: float,
    alpha: float,
    strength: float = 2.0,
) -> np.ndarray:
    """Equalize regional brightness by ``(I_ref / field) ** strength``.

    Local contrast rides on top of the slow field, so it is preserved. ``strength``
    above 1 pulls inner/outer overall brightness closer without flattening
    highlights to a constant.
    """
    dtype_in = image.dtype
    work = image.astype(np.float32, copy=False)
    safe_field = np.maximum(field.astype(np.float32, copy=False), 1.0)
    gain = np.power(float(i_ref) / safe_field, float(np.clip(strength, 0.1, 4.0)))
    corrected = work * gain
    blend = np.clip(float(alpha) * weight.astype(np.float32, copy=False), 0.0, 1.0)
    result = (1.0 - blend) * work + blend * corrected
    result = np.clip(result, 0.0, None)
    if np.issubdtype(dtype_in, np.integer):
        max_val = np.iinfo(dtype_in).max
        return np.clip(result, 0, max_val).astype(dtype_in)
    return result.astype(dtype_in, copy=False)


def _qc_zarr_path(output_zarr: Path) -> Path:
    return output_zarr.with_name(f"{output_zarr.stem}_qc.zarr")


def _write_qc_zarr(
    output_path: Path,
    *,
    volume_ds: np.ndarray,
    tissue: np.ndarray,
    extra_channels: list[tuple[str, np.ndarray]],
    downsample: int,
    compressor: Any,
    zarr_mod: Any,
) -> list[str]:
    """Write one napari-readable array ``0`` with QC channels stacked on axis 0."""
    planes = [
        np.clip(volume_ds, 0, np.iinfo(np.uint16).max).astype(np.uint16),
        (tissue.astype(np.uint8) * 255).astype(np.uint16),
    ]
    names = ["downsampled_input", "tissue_mask"]
    for name, array in extra_channels:
        names.append(name)
        if array.dtype.kind == "f" and float(array.max(initial=0)) <= 1.0 + 1e-3:
            planes.append((np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint16))
        else:
            planes.append(np.clip(array, 0, np.iinfo(np.uint16).max).astype(np.uint16))
    stacked = np.stack(planes, axis=0)
    chunks = (
        1,
        min(64, int(stacked.shape[1])),
        min(128, int(stacked.shape[2])),
        min(128, int(stacked.shape[3])),
    )
    root = _open_output_group(zarr_mod, output_path)
    try:
        from pipeline_modules.utils.zarr_io import ome_ngff_multiscales
    except ImportError:  # pragma: no cover
        from ..utils.zarr_io import ome_ngff_multiscales
    ds = float(downsample)
    root.attrs["multiscales"] = ome_ngff_multiscales(
        ["0"],
        ndim=stacked.ndim,
        base_scale=(1.0, ds, ds, ds),
        name="qc",
    )
    arr = _create_array(
        root,
        "0",
        shape=stacked.shape,
        chunks=chunks,
        dtype=stacked.dtype,
        compressor=compressor,
    )
    arr[:] = stacked
    arr.attrs["downsample"] = int(downsample)
    arr.attrs["channel_axis"] = 0
    arr.attrs["channel_names"] = names
    return names


def homogenize_surface_brightness(
    input_dir: str | Path,
    output_zarr: str | Path | None = None,
    *,
    mode: str = "shading",
    downsample: int = 8,
    d_max_px: float = 32.0,
    d_out_px: float = 48.0,
    ref_band_px: float = 40.0,
    field_sigma_px: float = 64.0,
    alpha: float = 1.0,
    interior_pct: float = 50.0,
    strength: float = 2.0,
    bright_hi_pct: float = 90.0,
    smooth_sigma: float = 3.0,
    close_iter: int = 0,
    seed_pct: float = 90.0,
    grow_pct: float = 84.0,
    max_grow_px: float = 32.0,
    erode_px: float = 16.0,
    chunk_size: tuple[int, int, int] = (32, 256, 256),
    compressor: str = "default",
    max_workers: int | None = None,
    save_qc: bool = True,
    qc_only: bool = False,
    pyramid: bool = False,
    preview: bool = True,
    preview_factor: int = 4,
    tissue_mask: str | Path | None = None,
    ref_mix: float = 0.5,
    target_scale: float = 1.0,
    max_gain: float = 6.0,
    surface_gain: float = 0.05,
    depth_gamma: float = 1.5,
    exterior_gain: float = 0.01,
    exterior_suppress_px: float = 128.0,
    local_suppress_strength: float = 3.0,
    local_suppress_below_px: float = 24.0,
    local_suppress_pct: float = 50.0,
    output_tiff_dir: str | Path | None = None,
    write_zarr: bool = True,
) -> dict[str, Any]:
    """Normalize tissue-wide brightness and write a Zarr volume and/or TIFF stack."""
    started_at = time.time()
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise PipelineError(
            ErrorCode.INPUT_NOT_FOUND,
            "Input TIFF directory not found",
            {"input_dir": str(input_path)},
        )
    output_path = _resolve_output_zarr(input_path, output_zarr)

    try:
        import zarr
    except ModuleNotFoundError as exc:
        raise PipelineError(
            ErrorCode.DEPENDENCY_MISSING,
            "zarr is required",
            {"dependency": "zarr", "error": str(exc)},
        ) from exc

    try:
        import cv2
        cv2.setNumThreads(1)
    except ModuleNotFoundError as exc:
        raise PipelineError(
            ErrorCode.DEPENDENCY_MISSING,
            "opencv-python is required",
            {"dependency": "cv2", "error": str(exc)},
        ) from exc

    tiff_files = _list_tiff_files(input_path)
    factor = max(1, int(downsample))
    chunks = _coerce_chunk_size(chunk_size)
    resolved_compressor = resolve_compressor(compressor)

    logger.info("Building downsampled volume from %d slices (factor=%d)", len(tiff_files), factor)
    volume_ds, z_indices = build_downsampled_volume(tiff_files, downsample=factor)
    mode_name = str(mode).strip().lower()
    if mode_name not in {"shell", "shading"}:
        raise PipelineError(ErrorCode.ARGUMENT_INVALID, "mode must be shell or shading", {"mode": mode})
    sampling = (float(factor), float(factor), float(factor))
    if tissue_mask:
        tissue, mask_stats = load_tissue_mask(
            tissue_mask,
            volume_ds.shape,
            close_iter=close_iter,
        )
        logger.info(
            "Using manual tissue mask %s frac=%.3f holes %d->%d",
            mask_stats.get("tissue_mask_path", tissue_mask),
            mask_stats["tissue_fraction"],
            int(mask_stats.get("holes_before", 0)),
            int(mask_stats.get("holes_after", 0)),
        )
    else:
        tissue, mask_stats = estimate_tissue_mask(
            volume_ds,
            close_iter=close_iter,
            seed_pct=seed_pct,
            grow_pct=grow_pct,
            max_grow_px=max_grow_px,
            erode_px=erode_px,
            sampling_zyx=sampling,
        )
    extra_qc: list[tuple[str, np.ndarray]] = []
    field_ds: np.ndarray | None = None
    weight_ds: np.ndarray | None = None
    membership_ds: np.ndarray | None = None
    gain_ds: np.ndarray | None = None
    if mode_name == "shell":
        membership_ds, dist_in, dist_out = surface_membership(
            tissue,
            d_max_px=float(d_max_px),
            d_out_px=float(d_out_px),
            sampling_zyx=sampling,
            smooth_sigma_ds=0.4,
        )
        gain_ds, anchors, suppress_ds = surface_depth_gain(
            volume_ds,
            tissue,
            dist_in,
            dist_out,
            d_out_px=float(d_out_px),
            target_scale=float(target_scale),
            max_gain=float(max_gain),
            surface_gain=float(surface_gain),
            depth_gamma=float(depth_gamma),
            exterior_gain=float(exterior_gain),
            field_sigma_ds=max(float(field_sigma_px), 1.0) / float(factor),
            local_suppress_strength=float(local_suppress_strength),
            local_suppress_below_px=float(local_suppress_below_px),
            local_suppress_pct=float(local_suppress_pct),
        )
        i_ref = float(anchors["i_ref"])
        weight_ds = shell_correction_weight(tissue.shape)
        corrected_ds = apply_depth_gain(
            volume_ds,
            gain_ds,
            weight_ds,
            alpha=float(alpha),
        )
        surface_rind = (tissue & (dist_in <= max(sampling[1], 1.0))).astype(np.float32)
        gain_qc_scale = max(float(anchors.get("max_gain", max_gain)), 1.0)
        extra_qc = [
            ("surface", surface_rind),
            ("depth_gain", np.clip(gain_ds / gain_qc_scale, 0.0, 1.0)),
            ("local_suppress", np.clip(suppress_ds, 0.0, 1.0)),
            ("homogenized", corrected_ds),
        ]
        logger.info(
            "Shell mode frac=%.3f d_deep_max=%.1f gain=[%.3f, %.2f] ext=%.3f gamma=%.2f curve@25/50/75=%.2f/%.2f/%.2f local_suppress=[%.3f, %.3f] target=%.1f",
            mask_stats["tissue_fraction"],
            anchors.get("d_deep_px", 0.0),
            anchors["gain_min"],
            anchors["gain_max"],
            exterior_gain,
            depth_gamma,
            anchors.get("gain_at_25", 0.0),
            anchors.get("gain_at_50", 0.0),
            anchors.get("gain_at_75", 0.0),
            anchors.get("local_suppress_min", 1.0),
            anchors.get("local_suppress_max", 1.0),
            anchors.get("local_suppress_target", 0.0),
        )
    else:
        sigma_ds = max(float(field_sigma_px), 1.0) / float(factor)
        field_ds, anchors = estimate_shading_field(
            volume_ds,
            tissue,
            sigma_ds=sigma_ds,
            interior_pct=float(interior_pct),
        )
        weight_ds = tissue_weight(tissue, smooth_sigma_ds=1.0)
        i_ref = float(anchors["i_ref"])
        corrected_ds = normalize_slice(
            volume_ds,
            field_ds,
            weight_ds,
            i_ref=i_ref,
            alpha=float(alpha),
            strength=float(strength),
        )
        extra_qc = [("normalized", corrected_ds)]
        logger.info(
            "Tissue high=%.2f low=%.2f frac=%.3f i_ref=%.1f field_sigma_px=%.1f strength=%.2f field=[%.1f, %.1f]",
            mask_stats["threshold"],
            mask_stats.get("grow_threshold", mask_stats["threshold"]),
            mask_stats["tissue_fraction"],
            i_ref,
            field_sigma_px,
            strength,
            anchors["field_min"],
            anchors["field_max"],
        )

    sample = tifffile.imread(str(tiff_files[0]))
    if sample.ndim > 2:
        sample = np.squeeze(sample)
    dtype = sample.dtype
    out_hw = (int(sample.shape[0]), int(sample.shape[1]))
    shape = (len(tiff_files),) + tuple(sample.shape)
    workers = resolve_stack_workers(0 if max_workers is None else int(max_workers))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    qc_path = _qc_zarr_path(output_path)
    qc_names = ["downsampled_input", "tissue_mask"] + [name for name, _ in extra_qc]
    if save_qc:
        qc_names = _write_qc_zarr(
            qc_path,
            volume_ds=volume_ds,
            tissue=tissue,
            extra_channels=extra_qc,
            downsample=factor,
            compressor=resolved_compressor,
            zarr_mod=zarr,
        )

    extra = {
        "success": True,
        "input_dir": str(input_path),
        "output_zarr": str(output_path),
        "qc_zarr": str(qc_path) if save_qc else None,
        "dataset_name": "0",
        "qc_arrays": qc_names if save_qc else [],
        "shape": list(shape),
        "dtype": str(dtype),
        "chunk_size": list(chunks),
        "total_files": len(tiff_files),
        "processed_files": 0,
        "mode": mode_name,
        "downsample": factor,
        "field_sigma_px": field_sigma_px,
        "strength": strength,
        "alpha": alpha,
        "interior_pct": interior_pct,
        "ref_mix": ref_mix,
        "target_scale": target_scale,
        "max_gain": max_gain,
        "surface_gain": surface_gain,
        "depth_gamma": depth_gamma,
        "exterior_gain": exterior_gain,
        "exterior_suppress_px": exterior_suppress_px,
        "sampled_z": z_indices,
        "qc_only": bool(qc_only),
        "pyramid": None,
        "preview": None,
        **mask_stats,
        **anchors,
    }
    extra["output_tiff_dir"] = None
    extra["write_zarr"] = bool(write_zarr)
    if qc_only:
        extra["duration_seconds"] = time.time() - started_at
        logger.info("QC-only run finished: %s", qc_path)
        return extra

    tiff_dir: Path | None = Path(output_tiff_dir) if output_tiff_dir else None
    if tiff_dir is not None:
        tiff_dir.mkdir(parents=True, exist_ok=True)
    if not write_zarr and tiff_dir is None:
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            "Full-res run needs --output_zarr (default) or --output_tiff_dir; use --skip_zarr only with TIFF output",
        )
    if not write_zarr:
        pyramid = False
        preview = False

    dataset = None
    if write_zarr:
        root = _open_output_group(zarr, output_path)
        dataset = _create_array(
            root,
            "0",
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressor=resolved_compressor,
        )
        root.attrs["multiscales"] = [{"version": "0.4", "datasets": [{"path": "0"}]}]

    def _homogenize_index(index: int) -> np.ndarray:
        image = np.asarray(tifffile.imread(str(tiff_files[index])))
        if image.ndim > 2:
            image = np.squeeze(image)
        if not np.any(image):
            return image.astype(dtype, copy=False)
        if mode_name == "shell":
            assert gain_ds is not None and weight_ds is not None
            gain_plane = resample_plane(gain_ds, index, downsample=factor, out_hw=out_hw)
            weight_plane = resample_plane(weight_ds, index, downsample=factor, out_hw=out_hw)
            return apply_depth_gain(
                image,
                gain_plane,
                weight_plane,
                alpha=float(alpha),
            )
        assert field_ds is not None and weight_ds is not None
        field_plane = resample_plane(field_ds, index, downsample=factor, out_hw=out_hw)
        weight_plane = resample_plane(weight_ds, index, downsample=factor, out_hw=out_hw)
        return normalize_slice(
            image,
            field_plane,
            weight_plane,
            i_ref=i_ref,
            alpha=float(alpha),
            strength=float(strength),
        )

    z_batch = max(1, int(chunks[0]))
    dest = str(tiff_dir) if tiff_dir is not None else str(output_path)
    logger.info(
        "Writing %s shape=%s chunks=%s with %d workers, z-batch=%d zarr=%s tiff=%s",
        dest,
        list(shape),
        list(chunks),
        workers,
        z_batch,
        write_zarr,
        str(tiff_dir) if tiff_dir is not None else None,
    )
    processed = 0
    progress = tqdm(total=len(tiff_files), desc="Tissue brightness normalize", unit="slice", file=sys.stderr)
    pool: ThreadPoolExecutor | None = None
    try:
        if workers > 1:
            pool = ThreadPoolExecutor(max_workers=workers)
        for start, end in iter_batch_ranges(len(tiff_files), z_batch):
            indices = list(range(start, end))
            if pool is None:
                slab = np.stack([_homogenize_index(index) for index in indices])
            else:
                slab = np.stack(list(pool.map(_homogenize_index, indices)))
            if dataset is not None:
                dataset[start:end] = slab
            if tiff_dir is not None:
                for offset, index in enumerate(indices):
                    tifffile.imwrite(str(tiff_dir / tiff_files[index].name), slab[offset])
            processed += end - start
            progress.update(end - start)
    finally:
        progress.close()
        if pool is not None:
            pool.shutdown(wait=True)

    pyramid_info: dict[str, Any] | None = None
    preview_info: dict[str, Any] | None = None
    if pyramid and write_zarr:
        pyramid_info = add_resolution_pyramid(output_path, write_manifest=False)
    if preview and write_zarr:
        preview_info = write_preview_zarr(output_path, factor=int(preview_factor))

    extra = {
        "success": True,
        "input_dir": str(input_path),
        "output_zarr": str(output_path) if write_zarr else None,
        "output_tiff_dir": str(tiff_dir) if tiff_dir is not None else None,
        "write_zarr": bool(write_zarr),
        "qc_zarr": str(qc_path) if save_qc else None,
        "dataset_name": "0",
        "qc_arrays": qc_names if save_qc else [],
        "shape": list(shape),
        "dtype": str(dtype),
        "chunk_size": list(chunks),
        "total_files": len(tiff_files),
        "processed_files": int(processed),
        "mode": mode_name,
        "downsample": factor,
        "field_sigma_px": field_sigma_px,
        "strength": strength,
        "alpha": alpha,
        "interior_pct": interior_pct,
        "ref_mix": ref_mix,
        "target_scale": target_scale,
        "max_gain": max_gain,
        "surface_gain": surface_gain,
        "depth_gamma": depth_gamma,
        "exterior_gain": exterior_gain,
        "exterior_suppress_px": exterior_suppress_px,
        "sampled_z": z_indices,
        "pyramid": pyramid_info,
        "preview": preview_info,
        **mask_stats,
        **anchors,
    }
    manifest_root = output_path if write_zarr else (tiff_dir if tiff_dir is not None else output_path)
    outputs = []
    if write_zarr:
        outputs.append(output_path)
    if tiff_dir is not None:
        outputs.append(tiff_dir)
    if save_qc:
        outputs.append(qc_path)
    if preview_info:
        outputs.append(Path(preview_info["output_zarr"]))
    manifest_path = write_run_manifest(
        manifest_root,
        module="preprocessing",
        entrypoint="homogenize_surface_brightness",
        inputs={
            "input_dir": str(input_path),
            "output_zarr": str(output_path) if write_zarr else None,
            "output_tiff_dir": str(tiff_dir) if tiff_dir is not None else None,
            "mode": mode_name,
            "downsample": factor,
            "field_sigma_px": field_sigma_px,
            "strength": strength,
            "alpha": alpha,
            "interior_pct": interior_pct,
            "close_iter": close_iter,
            "seed_pct": seed_pct,
            "grow_pct": grow_pct,
            "max_grow_px": max_grow_px,
            "erode_px": erode_px,
            "chunk_size": chunks,
            "compressor": compressor,
        },
        outputs=outputs,
        started_at=started_at,
        extra=extra,
    )
    extra["manifest_path"] = str(manifest_path)
    extra["duration_seconds"] = time.time() - started_at
    return extra


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Homogenize tissue brightness for viewing: shell+bright clamp, or regional shading",
    )
    parser.add_argument("--input_dir", required=True, help="Input signal TIFF folder")
    parser.add_argument(
        "--output_zarr",
        default=None,
        help="Output Zarr path (default: <input>_surface_homogenized.zarr beside the input)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Deprecated alias of --output_zarr; a .zarr suffix is added if missing",
    )
    parser.add_argument("--downsample", type=int, default=8, help="Integer downsample factor for the 3D field (default: 8)")
    parser.add_argument("--mode", choices=["shell", "shading"], default="shading", help="shell: press bright surface-band voxels to I_ref; shading: regional log-field normalize")
    parser.add_argument("--d_max_px", type=float, default=32.0, help="Inward surface band width in full-res pixels, shell mode (default: 32)")
    parser.add_argument("--d_out_px", type=float, default=48.0, help="Outward surface band width in full-res pixels, shell mode (default: 48)")
    parser.add_argument("--ref_band_px", type=float, default=40.0, help="Interior reference band just inside the surface, shell mode (default: 40)")
    parser.add_argument("--field_sigma_px", type=float, default=64.0, help="Gaussian sigma of the regional brightness field in full-res pixels (default: 64)")
    parser.add_argument("--strength", type=float, default=2.0, help="Equalization strength; 1 is I*I_ref/field, >1 darkens the outer rim and lifts the interior more (default: 2.0)")
    parser.add_argument("--bright_keep", type=float, default=0.2, help="Unused legacy flag")
    parser.add_argument("--alpha", type=float, default=1.0, help="Blend strength 0-1 of the normalized result inside tissue (default: 1.0)")
    parser.add_argument("--interior_pct", type=float, default=50.0, help="Percentile of tissue used as I_ref in shading mode (default: 50)")
    parser.add_argument("--ref_mix", type=float, default=0.5, help="Shell I_ref mix in log space: 0=observed surface, 1=observed interior (default: 0.5, geometric mean)")
    parser.add_argument("--target_scale", type=float, default=1.0, help="Multiply deep shell gain ceiling; <1 overall darker, >1 overall brighter (default: 1.0)")
    parser.add_argument("--max_gain", type=float, default=6.0, help="Shell mode max depth gain at the tissue core (default: 6)")
    parser.add_argument("--surface_gain", type=float, default=0.05, help="Shell mode gain at the tissue surface, <1 darkens (default: 0.05)")
    parser.add_argument("--depth_gamma", type=float, default=1.5, help="Shell depth exponent on (d/d_max); 1=linear with depth, >1 stays darker longer before the core (default: 1.5)")
    parser.add_argument("--exterior_gain", type=float, default=0.01, help="Gain applied to ALL voxels outside the tissue mask (default: 0.01)")
    parser.add_argument("--exterior_suppress_px", type=float, default=0.0, help="Deprecated/ignored; exterior_gain always applies to the entire exterior")
    parser.add_argument("--local_suppress_strength", type=float, default=3.0, help="Sub-surface local-brightness suppression exponent; 0 disables, 1 is I_target/field (default: 3.0)")
    parser.add_argument("--local_suppress_below_px", type=float, default=24.0, help="Start local suppression this many full-res pixels below the surface (default: 24)")
    parser.add_argument("--local_suppress_pct", type=float, default=50.0, help="Percentile of depth-corrected sub-surface tissue used as the local suppression target (default: 50)")
    parser.add_argument("--bright_hi_pct", type=float, default=90.0, help="Percentile of bright surface voxels used as I_hi in shell mode (default: 90)")
    parser.add_argument("--smooth_sigma", type=float, default=3.0, help="Unused legacy flag")
    parser.add_argument("--close_iter", type=int, default=0, help="After tightening, close this many voxels then fill all cavities not connected to the volume border (default: 0)")
    parser.add_argument("--seed_pct", type=float, default=90.0, help="High hysteresis percentile for the bright seed (default: 90)")
    parser.add_argument("--grow_pct", type=float, default=84.0, help="Low hysteresis percentile; seed grows only through voxels above this (default: 84)")
    parser.add_argument("--max_grow_px", type=float, default=32.0, help="Max distance from the bright seed that the mask may grow (default: 32)")
    parser.add_argument("--erode_px", type=float, default=16.0, help="Erode the filled mask by this many full-res pixels (default: 16)")
    parser.add_argument(
        "--tissue_mask",
        default=None,
        help="Manual/SAM tissue mask Zarr or TIFF at QC resolution; skips automatic hysteresis",
    )
    parser.add_argument("--chunk_size", default="32,256,256", help="Zarr chunk size z,y,x (default: 32,256,256)")
    parser.add_argument("--compressor", default="default", help="Zarr compressor: default/fast/none")
    parser.add_argument("--max_workers", type=int, default=0, help="Worker threads; 0 auto-caps")
    parser.add_argument("--no_qc", action="store_true", help="Do not write downsampled QC arrays into the Zarr group")
    parser.add_argument("--qc_only", action="store_true", help="Write the downsampled QC zarr and exit without the full-res volume")
    parser.add_argument("--no_pyramid", action="store_true", help="Skip in-place OME-NGFF pyramid (default: skipped; it breaks napari builtin reader)")
    parser.add_argument("--pyramid", action="store_true", help="Write resolution levels 1..N into the same Zarr (needs napari-ome-zarr to open)")
    parser.add_argument("--no_preview", action="store_true", help="Do not write a sibling *_preview.zarr for 3D viewing")
    parser.add_argument("--preview_factor", type=int, default=4, help="Downsample factor for *_preview.zarr (default: 4)")
    parser.add_argument(
        "--output_tiff_dir",
        default=None,
        help="Write processed slices as TIFFs here, keeping original filenames (for Imaris File Converter)",
    )
    parser.add_argument("--skip_zarr", action="store_true", help="Do not write the full-res Zarr; requires --output_tiff_dir")
    parser.add_argument("--json_logs", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    _configure_logging(args.json_logs)
    output_zarr = args.output_zarr or args.output_dir
    try:
        result = homogenize_surface_brightness(
            args.input_dir,
            output_zarr,
            mode=args.mode,
            downsample=args.downsample,
            d_max_px=args.d_max_px,
            d_out_px=args.d_out_px,
            ref_band_px=args.ref_band_px,
            field_sigma_px=args.field_sigma_px,
            alpha=args.alpha,
            interior_pct=args.interior_pct,
            strength=args.strength,
            bright_hi_pct=args.bright_hi_pct,
            smooth_sigma=args.smooth_sigma,
            close_iter=args.close_iter,
            seed_pct=args.seed_pct,
            grow_pct=args.grow_pct,
            max_grow_px=args.max_grow_px,
            erode_px=args.erode_px,
            chunk_size=_coerce_chunk_size(args.chunk_size),
            compressor=args.compressor,
            max_workers=args.max_workers,
            save_qc=not args.no_qc,
            qc_only=bool(args.qc_only),
            pyramid=bool(args.pyramid),
            preview=not args.no_preview,
            preview_factor=args.preview_factor,
            tissue_mask=args.tissue_mask,
            ref_mix=args.ref_mix,
            target_scale=args.target_scale,
            max_gain=args.max_gain,
            surface_gain=args.surface_gain,
            depth_gamma=args.depth_gamma,
            exterior_gain=args.exterior_gain,
            exterior_suppress_px=args.exterior_suppress_px,
            local_suppress_strength=args.local_suppress_strength,
            local_suppress_below_px=args.local_suppress_below_px,
            local_suppress_pct=args.local_suppress_pct,
            output_tiff_dir=args.output_tiff_dir,
            write_zarr=not args.skip_zarr,
        )
    except PipelineError as exc:
        _pipeline_error_to_stderr(exc)
        return exc.exit_code
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
