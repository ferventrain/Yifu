"""Equalize tissue brightness for viewing — one CLI, two correction methods.

Selected with ``--method``:

* ``log_field`` (default): log-domain regional brightness field over a signal
  TIFF folder (the historical shell/shading behaviour of this module).
  Non-specific stain often makes the exterior brighter than the interior. That
  is a slow illumination-like offset, not a reason to crush bright voxels: the
  highlights still carry structure. The correction is a log-domain shading
  normalize:

  1. Build a cheap downsampled volume (every Nth slice, XY resized).
  2. Estimate a tight tissue envelope from a bright seed + hole fill, or load a
     manual/SAM mask.
  3. Smooth log-intensity into a regional brightness envelope.
  4. Inside tissue, scale by ``(I_ref / field) ** strength`` so inner/outer
     overall brightness matches, while high-frequency detail stays.

* ``depth_profile``: 1D brightness-vs-distance-to-surface inverse gain over a
  Zarr volume (the former ``depth_brightness_equalize`` module). Measures the
  actual median intensity as a function of distance from the tissue surface,
  fits a smooth curve through that profile, and applies the inverse gain:
  inside the mask ``gain = T / I(d)`` (T = geometric mean of shell peak and
  deep plateau), outside the mask suppression-only, both clamped to
  [min_gain, max_gain]. The curve is fitted on the mask grid and sampled onto
  the target grid by physical spacing, so one downsample fit can be applied to
  the full-resolution volume.

CLI::

    python -m pipeline_modules.preprocessing.surface_brightness_homogenize --input_dir ch0 --output_zarr ch0_surface_homogenized.zarr --qc_only
    python -m pipeline_modules.preprocessing.surface_brightness_homogenize --method depth_profile --image ch0_downsampled.zarr --mask tissue_mask.tif --voxel_size_zyx 8,7.2,7.2
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
from scipy.interpolate import PchipInterpolator
from tqdm import tqdm

try:
    from pipeline_modules.harness.progress import report, report_done
    from pipeline_modules.preprocessing.tiff_to_zarr import (
        _create_array,
        _open_output_group,
        resolve_compressor,
    )
    from pipeline_modules.preprocessing.zarr_pyramid import add_resolution_pyramid, write_preview_zarr
    from pipeline_modules.utils.errors import ErrorCode, PipelineError
    from pipeline_modules.utils.run_manifest import write_run_manifest
    from pipeline_modules.utils.tiff_stack_io import iter_batch_ranges, resolve_stack_workers
    from pipeline_modules.utils.zarr_io import (
        array_compressor,
        create_array,
        open_output_group,
        open_zarr_array,
        read_zarr_attrs,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from ..harness.progress import report, report_done
    from .tiff_to_zarr import _create_array, _open_output_group, resolve_compressor
    from .zarr_pyramid import add_resolution_pyramid, write_preview_zarr
    from ..utils.errors import ErrorCode, PipelineError
    from ..utils.run_manifest import write_run_manifest
    from ..utils.tiff_stack_io import iter_batch_ranges, resolve_stack_workers
    from ..utils.zarr_io import (
        array_compressor,
        create_array,
        open_output_group,
        open_zarr_array,
        read_zarr_attrs,
    )

logger = logging.getLogger(__name__)

DEFAULT_MIN_GAIN = 0.1
DEFAULT_MAX_GAIN = 5.0
DEFAULT_BIN_UM = 10.0


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
        # Full NGFF axes + ct (bare multiscales breaks ome-zarr readers).
        try:
            from pipeline_modules.utils.zarr_io import ome_ngff_multiscales
        except ImportError:
            from ..utils.zarr_io import ome_ngff_multiscales
        root.attrs["multiscales"] = ome_ngff_multiscales(["0"], ndim=3)

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


# ---------------------------------------------------------------------------
# depth_profile method - 1D brightness-vs-distance-to-surface inverse gain
# (merged from the former pipeline_modules.preprocessing.depth_brightness_equalize)
# ---------------------------------------------------------------------------

def parse_triplet(value: str | list[float] | tuple[float, ...], *, name: str) -> tuple[float, float, float]:
    if isinstance(value, (list, tuple)):
        parts = [float(v) for v in value]
    else:
        text = str(value).replace("x", ",").replace(";", ",")
        parts = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(parts) != 3 or any(not np.isfinite(v) or v <= 0 for v in parts):
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            f"{name} must be three positive numbers (z,y,x)",
            {name: value},
        )
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def load_mask(path: str | Path) -> np.ndarray:
    """Load a 3D binary mask from TIFF, .npy or Zarr."""
    mask_path = Path(path)
    if not mask_path.exists():
        raise PipelineError(ErrorCode.INPUT_NOT_FOUND, "Mask not found", {"mask": str(mask_path)})
    suffix = mask_path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        import tifffile

        data = np.asarray(tifffile.imread(str(mask_path)))
    elif suffix == ".npy":
        data = np.asarray(np.load(str(mask_path)))
    else:
        data = np.asarray(open_zarr_array(mask_path))
    if data.ndim != 3:
        raise PipelineError(
            ErrorCode.INPUT_FORMAT_INVALID,
            "Mask must be a 3D volume",
            {"mask": str(mask_path), "shape": list(data.shape)},
        )
    mask = data > 0
    if not mask.any():
        raise PipelineError(ErrorCode.EMPTY_RESULT, "Mask is empty", {"mask": str(mask_path)})
    return mask


def distance_transforms(mask: np.ndarray, spacing_zyx: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Distance to the mask boundary inside (positive) and outside (positive), in µm."""
    binary = np.asarray(mask, dtype=bool)
    try:
        import edt as edt_mod

        d_in = edt_mod.edt(binary, anisotropy=spacing_zyx, parallel=1).astype(np.float32, copy=False)
        d_out = edt_mod.edt(~binary, anisotropy=spacing_zyx, parallel=1).astype(np.float32, copy=False)
    except ImportError:
        d_in = ndi.distance_transform_edt(binary, sampling=spacing_zyx).astype(np.float32, copy=False)
        d_out = ndi.distance_transform_edt(~binary, sampling=spacing_zyx).astype(np.float32, copy=False)
    return d_in, d_out


def _profile_1d(
    values: np.ndarray,
    distances: np.ndarray,
    *,
    bin_um: float,
    min_voxels: int,
    max_sample: int = 20_000_000,
) -> dict[str, np.ndarray]:
    """Median intensity per distance bin, evaluated on masked voxels only.

    Large masks are subsampled before sorting; the medians are envelope
    estimates, not per-voxel statistics, so a uniform stride is enough.
    """
    if values.size == 0:
        raise PipelineError(ErrorCode.EMPTY_RESULT, "No voxels to profile")
    step = max(1, int(np.ceil(values.size / float(max_sample))))
    values = values[::step]
    distances = distances[::step]
    bin_um = max(float(bin_um), 0.1)
    nbins = int(np.floor(float(distances.max()) / bin_um)) + 1
    index = np.clip((distances / bin_um).astype(np.int32), 0, nbins - 1)
    counts = np.bincount(index, minlength=nbins)
    order = np.argsort(index, kind="stable")
    sorted_index = index[order]
    sorted_values = values[order]
    centers = (np.arange(nbins, dtype=np.float64) + 0.5) * bin_um
    medians = np.full(nbins, np.nan, dtype=np.float64)
    p75 = np.full(nbins, np.nan, dtype=np.float64)
    start = 0
    for b in range(nbins):
        count = int(counts[b])
        if count >= int(min_voxels):
            chunk = sorted_values[start : start + count]
            medians[b] = float(np.median(chunk))
            p75[b] = float(np.percentile(chunk, 75))
        start += count
    keep = np.isfinite(medians)
    if np.count_nonzero(keep) < 3:
        raise PipelineError(
            ErrorCode.EMPTY_RESULT,
            "Too few populated distance bins to fit a curve",
            {"bin_um": bin_um, "populated_bins": int(np.count_nonzero(keep))},
        )
    return {
        "centers": centers[keep],
        "medians": medians[keep],
        "p75": p75[keep],
        "counts": counts[keep],
        "d_max": float(distances.max()),
    }


def _smooth(values: np.ndarray, sigma_bins: float) -> np.ndarray:
    sigma = max(float(sigma_bins), 0.0)
    if sigma <= 0 or values.size < 3:
        return values
    return ndi.gaussian_filter1d(values, sigma, mode="nearest")


def fit_gain_lut(
    inside: dict[str, np.ndarray],
    outside: dict[str, np.ndarray],
    *,
    max_gain: float = DEFAULT_MAX_GAIN,
    min_gain: float = DEFAULT_MIN_GAIN,
    smooth_bins: float = 1.2,
    deep_fraction: float = 0.1,
    target_scale: float = 1.0,
) -> dict[str, Any]:
    """Fit monotone-ish brightness envelopes and return 1 µm gain lookup tables.

    Inside the target is the geometric mean of the smoothed shell peak and the
    deep plateau, which equalizes both ends within the gain limits. Outside the
    target is the far-background level and the gain is capped at 1.
    """
    max_gain = float(max_gain)
    min_gain = float(min_gain)
    if not (0 < min_gain < 1 < max_gain):
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            "Require 0 < min_gain < 1 < max_gain",
            {"min_gain": min_gain, "max_gain": max_gain},
        )

    x_in = inside["centers"].astype(np.float64)
    y_in = _smooth(inside["medians"].astype(np.float64), smooth_bins)
    if x_in.size < 3:
        raise PipelineError(ErrorCode.EMPTY_RESULT, "Inside profile needs at least 3 bins")

    peak_index = int(np.argmax(y_in))
    tail = max(1, int(round(x_in.size * float(np.clip(deep_fraction, 0.02, 0.5)))))
    plateau = float(np.median(y_in[-tail:]))
    peak = float(y_in[peak_index])
    target_in = float(np.sqrt(max(peak, 1e-3) * max(plateau, 1e-3))) * max(float(target_scale), 0.05)

    # Constant extrapolation below the first bin center keeps the mask edge
    # from being amplified by a polynomial extrapolation.
    f_in = PchipInterpolator(x_in, np.maximum(y_in, 1e-3), extrapolate=False)
    d_in_max = float(np.ceil(max(float(inside.get("d_max", x_in[-1])), x_in[-1])))
    lut_d_in = np.arange(0.0, d_in_max + 1.0, 1.0)
    eval_in = f_in(np.clip(lut_d_in, x_in[0], x_in[-1]))
    eval_in = np.where(np.isfinite(eval_in), eval_in, y_in[0])
    gain_in = np.clip(target_in / np.maximum(eval_in, 1e-3), min_gain, max_gain)
    gain_in = np.clip(_smooth(gain_in.astype(np.float64), 1.0), min_gain, max_gain)

    x_out = outside["centers"].astype(np.float64)
    y_out = _smooth(outside["medians"].astype(np.float64), 1.5)
    tail_out = max(1, int(round(x_out.size * float(np.clip(deep_fraction, 0.02, 0.5)))))
    # Zero-filled padding makes the far median collapse to 0; floor the target
    # so far background is preserved instead of gaining a meaningless 10x dim.
    target_out = max(float(np.median(y_out[-tail_out:])), 1.0)
    f_out = PchipInterpolator(x_out, np.maximum(y_out, 1e-3), extrapolate=False)
    d_out_max = float(np.ceil(max(float(outside.get("d_max", x_out[-1])), x_out[-1])))
    lut_d_out = np.arange(0.0, d_out_max + 1.0, 1.0)
    eval_out = f_out(np.clip(lut_d_out, x_out[0], x_out[-1]))
    eval_out = np.where(np.isfinite(eval_out), eval_out, y_out[0])
    gain_out = np.clip(float(target_out) / np.maximum(eval_out, 1e-3), min_gain, 1.0)

    raw_ratio = peak / max(plateau, 1e-6)
    fitted = {
        "target_inside": target_in,
        "target_outside": target_out,
        "shell_peak": peak,
        "deep_plateau": plateau,
        "profile_ratio": raw_ratio,
        "inside_median_ratio_before": float(np.nanmax(inside["medians"]) / max(np.nanmin(inside["medians"]), 1e-6)),
        "gain_inside_min": float(gain_in.min()),
        "gain_inside_max": float(gain_in.max()),
        "gain_outside_min": float(gain_out.min()),
        "gain_outside_max": float(gain_out.max()),
        "gain_limits": [min_gain, max_gain],
        "lut_d_in": lut_d_in,
        "gain_in": gain_in,
        "lut_d_out": lut_d_out,
        "gain_out": gain_out,
        "inside_centers": x_in,
        "inside_medians": y_in,
        "outside_centers": x_out,
        "outside_medians": y_out,
    }
    return fitted


def build_gain_field(
    d_in: np.ndarray,
    d_out: np.ndarray,
    mask: np.ndarray,
    fitted: dict[str, Any],
    *,
    edge_gain: float = 1.0,
    edge_um: float = 0.0,
) -> np.ndarray:
    """Per-voxel gain on the mask grid.

    ``edge_gain`` < 1 pins the innermost voxel to that gain and blends back to
    the fitted curve over ``edge_um`` microns. The mask edge mixes partial-volume
    voxels, so its measured intensity is not a reliable curve anchor.
    """
    lut_in = fitted["gain_in"]
    lut_out = fitted["gain_out"]
    gi = lut_in[np.clip(d_in.astype(np.int32), 0, lut_in.size - 1)]
    edge_gain = float(edge_gain)
    edge_um = float(edge_um)
    if edge_gain < 1.0 and edge_um > 0:
        # Fixed low gain at the surface, blending back to the fitted curve gain
        # by edge_um. The curve alone cannot suppress the partial-volume edge
        # because its measured intensity there is unreliable.
        edge_gain = max(edge_gain, float(fitted["gain_limits"][0]))
        curve_at_edge = float(lut_in[min(int(edge_um), lut_in.size - 1)])
        blend = np.clip(d_in.astype(np.float32) / edge_um, 0.0, 1.0)
        ramp = edge_gain + (curve_at_edge - edge_gain) * blend
        gi = np.where(d_in < edge_um, np.minimum(gi, ramp), gi)
    go = lut_out[np.clip(d_out.astype(np.int32), 0, lut_out.size - 1)]
    return np.where(mask, gi, go).astype(np.float32, copy=False)


def gain_for_block(
    gain: np.ndarray,
    fit_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
    target_shape: tuple[int, int, int],
    z0: int,
    z1: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> np.ndarray:
    """Resample the gain field onto a block of the target grid by physical spacing.

    Blocking lets the caller stream chunk-aligned slabs without holding a
    full-resolution gain volume in memory.
    """
    ratio = tuple(float(target_spacing[a]) / float(fit_spacing[a]) for a in range(3))
    if np.allclose(ratio, 1.0) and tuple(gain.shape) == tuple(target_shape):
        return gain[z0:z1, y0:y1, x0:x1]
    matrix = np.diag(np.asarray(ratio, dtype=np.float64))
    offset = np.asarray(
        [
            (float(z0) + 0.5) * ratio[0] - 0.5,
            (float(y0) + 0.5) * ratio[1] - 0.5,
            (float(x0) + 0.5) * ratio[2] - 0.5,
        ],
        dtype=np.float64,
    )
    shape = (int(z1 - z0), int(y1 - y0), int(x1 - x0))
    return ndi.affine_transform(
        gain,
        matrix,
        offset=offset,
        output_shape=shape,
        order=1,
        mode="nearest",
        prefilter=False,
    )


def gain_for_slab(
    gain: np.ndarray,
    fit_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
    target_shape: tuple[int, int, int],
    z0: int,
    z1: int,
) -> np.ndarray:
    """Full-width slab convenience wrapper around :func:`gain_for_block`."""
    return gain_for_block(
        gain,
        fit_spacing,
        target_spacing,
        target_shape,
        z0,
        z1,
        0,
        int(target_shape[1]),
        0,
        int(target_shape[2]),
    )


def _apply_chunk(
    image: np.ndarray,
    gain: np.ndarray,
    dtype: np.dtype,
) -> tuple[np.ndarray, int]:
    work = image.astype(np.float32, copy=False)
    result = work * gain
    if np.issubdtype(dtype, np.integer):
        max_val = float(np.iinfo(dtype).max)
        clipped = int(np.count_nonzero(result > max_val))
        result = np.clip(result, 0.0, max_val)
        return np.rint(result).astype(dtype, copy=False), clipped
    return result.astype(dtype, copy=False), 0


def _write_qc(
    qc_dir: Path,
    image: np.ndarray,
    corrected: np.ndarray,
    gain: np.ndarray,
    mask: np.ndarray,
    spacing_zyx: tuple[float, float, float],
) -> list[str]:
    """Before/after slice and MIP previews with a shared display window."""
    try:
        from PIL import Image
    except ModuleNotFoundError:
        logger.warning("Pillow missing; skipping QC images")
        return []
    qc_dir.mkdir(parents=True, exist_ok=True)
    window_source = corrected[mask] if mask.any() else corrected
    lo, hi = np.percentile(window_source, [1.0, 99.8])
    hi = max(float(hi), float(lo) + 1.0)
    z = image.shape[0] // 2
    written: list[str] = []

    def _save(array: np.ndarray, name: str, vmin: float, vmax: float, cmap: str = "gray") -> None:
        scaled = np.clip((array.astype(np.float32) - vmin) / (vmax - vmin), 0.0, 1.0)
        if cmap == "gray":
            rgb = np.stack([scaled] * 3, axis=-1)
        else:
            import matplotlib

            rgb = matplotlib.colormaps[cmap](scaled)[..., :3]
        path = qc_dir / name
        Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(path)
        written.append(str(path))

    _save(image[z], "slice_before.png", lo, hi)
    _save(corrected[z], "slice_after.png", lo, hi)
    _save(gain[z], "gain_slice.png", 0.0, 5.0, cmap="viridis")
    _save(image.max(axis=2), "mip_along_x_before.png", lo, hi)
    _save(corrected.max(axis=2), "mip_along_x_after.png", lo, hi)
    return written


def _fit_volume(
    image: np.ndarray,
    mask: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    *,
    bin_um: float,
    max_gain: float,
    min_gain: float,
    smooth_bins: float,
    deep_fraction: float,
    target_scale: float = 1.0,
    edge_gain: float = 1.0,
    edge_um: float = 0.0,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    d_in, d_out = distance_transforms(mask, spacing_zyx)
    inside = _profile_1d(image[mask], d_in[mask], bin_um=bin_um, min_voxels=200)
    outside = _profile_1d(image[~mask], d_out[~mask], bin_um=bin_um * 2.0, min_voxels=200)
    fitted = fit_gain_lut(
        inside,
        outside,
        max_gain=max_gain,
        min_gain=min_gain,
        smooth_bins=smooth_bins,
        deep_fraction=deep_fraction,
        target_scale=target_scale,
    )
    gain = build_gain_field(d_in, d_out, mask, fitted, edge_gain=edge_gain, edge_um=edge_um)
    inside_gain = gain[mask]
    fitted["gain_inside_min"] = float(inside_gain.min())
    fitted["gain_inside_max"] = float(inside_gain.max())
    return fitted, gain, d_in, d_out


def equalize_depth_brightness(
    image: str | Path,
    mask: str | Path,
    output_zarr: str | Path | None = None,
    *,
    voxel_size_zyx: tuple[float, float, float] | str,
    fit_image: str | Path | None = None,
    fit_voxel_size_zyx: tuple[float, float, float] | str | None = None,
    max_gain: float = DEFAULT_MAX_GAIN,
    min_gain: float = DEFAULT_MIN_GAIN,
    bin_um: float = DEFAULT_BIN_UM,
    smooth_bins: float = 1.2,
    deep_fraction: float = 0.1,
    target_scale: float = 1.0,
    edge_gain: float = 1.0,
    edge_um: float = 20.0,
    apply_chunk_z: int = 0,
    compressor: str = "default",
    qc_dir: str | Path | None = None,
    fit_only: bool = False,
    overwrite: bool = False,
    sample_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Fit the depth curve and write a brightness-equalized copy of ``image``."""
    started_at = time.time()
    image_path = Path(image)
    mask_path = Path(mask)
    if not image_path.exists():
        raise PipelineError(ErrorCode.INPUT_NOT_FOUND, "Image Zarr not found", {"image": str(image_path)})
    spacing = parse_triplet(voxel_size_zyx, name="voxel_size_zyx")
    fit_path = Path(fit_image) if fit_image else image_path
    fit_spacing = (
        parse_triplet(fit_voxel_size_zyx, name="fit_voxel_size_zyx") if fit_voxel_size_zyx else spacing
    )

    mask_volume = load_mask(mask_path)
    fit_array = open_zarr_array(fit_path)
    fit_shape = tuple(int(v) for v in fit_array.shape)
    if fit_shape != mask_volume.shape:
        raise PipelineError(
            ErrorCode.INPUT_FORMAT_INVALID,
            "Mask and fit image grids differ; resample the mask onto the fit grid first",
            {"mask_shape": list(mask_volume.shape), "fit_shape": list(fit_shape), "fit_image": str(fit_path)},
        )

    logger.info(
        "Fitting depth curve on %s shape=%s spacing=%s mask_frac=%.3f",
        fit_path,
        list(fit_shape),
        list(fit_spacing),
        float(mask_volume.mean()),
    )
    report(done=0, total=4, phase="fit profile", sample_dir=sample_dir)
    fit_data = np.asarray(fit_array[:], dtype=np.float32)
    fitted, gain, d_in, d_out = _fit_volume(
        fit_data,
        mask_volume,
        fit_spacing,
        bin_um=float(bin_um),
        max_gain=float(max_gain),
        min_gain=float(min_gain),
        smooth_bins=float(smooth_bins),
        deep_fraction=float(deep_fraction),
        target_scale=float(target_scale),
        edge_gain=float(edge_gain),
        edge_um=float(edge_um),
    )
    corrected_fit = fit_data * gain
    before = fit_data[mask_volume]
    after = corrected_fit[mask_volume]
    after_profile = _profile_1d(after, d_in[mask_volume], bin_um=float(bin_um), min_voxels=200)
    summary: dict[str, Any] = {
        "image": str(image_path),
        "mask": str(mask_path),
        "fit_image": str(fit_path),
        "fit_shape": list(fit_shape),
        "fit_voxel_size_zyx": list(fit_spacing),
        "target_voxel_size_zyx": list(spacing),
        "mask_fraction": float(mask_volume.mean()),
        "target_inside": fitted["target_inside"],
        "target_outside": fitted["target_outside"],
        "shell_peak": fitted["shell_peak"],
        "deep_plateau": fitted["deep_plateau"],
        "profile_ratio_before": fitted["profile_ratio"],
        "inside_bin_ratio_before": fitted["inside_median_ratio_before"],
        "inside_bin_ratio_after": float(
            np.max(after_profile["medians"]) / max(float(np.min(after_profile["medians"])), 1e-6)
        ),
        "inside_median_before": float(np.median(before)),
        "inside_median_after": float(np.median(after)),
        "inside_p10_after": float(np.percentile(after, 10)),
        "inside_p90_after": float(np.percentile(after, 90)),
        "gain_inside_min": fitted["gain_inside_min"],
        "gain_inside_max": fitted["gain_inside_max"],
        "gain_outside_min": fitted["gain_outside_min"],
        "gain_outside_max": fitted["gain_outside_max"],
        "gain_limits": fitted["gain_limits"],
        "target_scale": float(target_scale),
        "edge_gain": float(edge_gain),
        "edge_um": float(edge_um),
        "distance_in_max_um": float(d_in.max()),
        "distance_out_max_um": float(d_out.max()),
    }
    logger.info(
        "Fitted target_in=%.0f target_out=%.1f shell=%.0f plateau=%.0f gain_in=[%.3f,%.3f]",
        summary["target_inside"],
        summary["target_outside"],
        summary["shell_peak"],
        summary["deep_plateau"],
        summary["gain_inside_min"],
        summary["gain_inside_max"],
    )
    report(done=1, total=4, phase="curve fitted", sample_dir=sample_dir)

    output_path = Path(output_zarr) if output_zarr else image_path.with_name(
        f"{image_path.stem}_brightness_equalized.zarr"
    )
    if output_path.suffix.lower() != ".zarr":
        output_path = output_path.with_suffix(".zarr")
    if output_path.resolve() == image_path.resolve():
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            "Output must differ from the input image",
            {"output": str(output_path)},
        )
    if output_path.exists() and not overwrite and not fit_only:
        raise PipelineError(
            ErrorCode.OUTPUT_INVALID,
            "Output Zarr already exists; pass --overwrite to replace it",
            {"output": str(output_path)},
        )

    if qc_dir:
        summary["qc_images"] = _write_qc(
            Path(qc_dir), fit_data, corrected_fit, gain, mask_volume, fit_spacing
        )
    del corrected_fit, fit_data

    if fit_only:
        if output_zarr:
            output_path = Path(output_zarr)
            if output_path.suffix.lower() != ".json":
                output_path = output_path.with_suffix(".json")
        else:
            output_path = image_path.with_name(f"{image_path.stem}_depth_gain_fit.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary["fit_only"] = True
        summary["duration_seconds"] = time.time() - started_at
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        report_done(sample_dir=sample_dir)
        return summary

    target = open_zarr_array(image_path)
    target_shape = tuple(int(v) for v in target.shape)
    if len(target_shape) != 3:
        raise PipelineError(
            ErrorCode.INPUT_FORMAT_INVALID,
            "Image must be a 3D volume",
            {"image": str(image_path), "shape": list(target_shape)},
        )
    dtype = np.dtype(target.dtype)
    chunks = tuple(int(v) for v in (target.chunks or (1, target_shape[1], target_shape[2])))
    # Read/write in chunk-aligned z blocks; a smaller non-aligned z range makes
    # zarr decompress every 256-slice chunk once per slab.
    chunk_z = int(apply_chunk_z) if int(apply_chunk_z) > 0 else chunks[0]
    chunk_z = max(1, min(chunk_z, target_shape[0]))
    chunk_y = max(1, int(chunks[1]))
    max_tile_voxels = 300_000_000
    tile_y = max(chunk_y, int(max_tile_voxels // max(chunk_z * target_shape[2], 1)))
    tile_y = min(target_shape[1], ((tile_y + chunk_y - 1) // chunk_y) * chunk_y)
    z_starts = list(range(0, target_shape[0], chunk_z))
    y_starts = list(range(0, target_shape[1], tile_y))
    total_tiles = len(z_starts) * len(y_starts)

    source_attrs = read_zarr_attrs(image_path)
    out_compressor = array_compressor(target)
    if out_compressor is None:
        out_compressor = resolve_compressor(compressor)
    root = open_output_group(output_path, overwrite=True)
    dataset = create_array(
        root,
        "0",
        shape=target_shape,
        chunks=chunks,
        dtype=dtype,
        compressor=out_compressor,
    )
    for key, value in source_attrs.items():
        try:
            root.attrs[key] = value
        except (TypeError, ValueError):
            logger.warning("Could not copy attr %s to output", key)
    root.attrs["depth_brightness_equalize"] = {
        "target_inside": summary["target_inside"],
        "target_outside": summary["target_outside"],
        "gain_inside": [summary["gain_inside_min"], summary["gain_inside_max"]],
        "gain_outside": [summary["gain_outside_min"], summary["gain_outside_max"]],
        "fit_image": str(fit_path),
        "fit_voxel_size_zyx": list(fit_spacing),
        "mask": str(mask_path),
    }

    clipped = 0
    done = 0
    for z0 in z_starts:
        z1 = min(z0 + chunk_z, target_shape[0])
        for y0 in y_starts:
            y1 = min(y0 + tile_y, target_shape[1])
            slab = np.asarray(target[z0:z1, y0:y1, :])
            gain_block = gain_for_block(
                gain,
                fit_spacing,
                spacing,
                target_shape,
                z0,
                z1,
                y0,
                y1,
                0,
                target_shape[2],
            )
            out, clipped_block = _apply_chunk(slab, gain_block, dtype)
            dataset[z0:z1, y0:y1, :] = out
            clipped += clipped_block
            done += 1
            report(done=done, total=total_tiles, phase="apply gain", sample_dir=sample_dir)
            print(f"PROGRESS {done}/{total_tiles} apply_gain", flush=True)
    processed = target_shape[0]

    summary.update(
        {
            "output_zarr": str(output_path),
            "target_shape": list(target_shape),
            "dtype": str(dtype),
            "chunk_z": chunk_z,
            "tile_y": tile_y,
            "total_tiles": total_tiles,
            "processed_slices": processed,
            "clipped_voxels": int(clipped),
            "duration_seconds": time.time() - started_at,
        }
    )
    write_run_manifest(
        output_path,
        module="preprocessing",
        entrypoint="equalize_depth_brightness",
        inputs={
            "image": str(image_path),
            "mask": str(mask_path),
            "fit_image": str(fit_path),
            "voxel_size_zyx": list(spacing),
            "fit_voxel_size_zyx": list(fit_spacing),
            "max_gain": float(max_gain),
            "min_gain": float(min_gain),
            "bin_um": float(bin_um),
            "target_scale": float(target_scale),
            "edge_gain": float(edge_gain),
            "edge_um": float(edge_um),
        },
        outputs=[output_path],
        started_at=started_at,
        extra=summary,
    )
    report_done(sample_dir=sample_dir)
    logger.info("Wrote %s (%d slices, %.1f s)", output_path, processed, summary["duration_seconds"])
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Brightness equalization, one CLI with two methods: --method log_field "
            "(regional log-domain brightness field over a TIFF folder, shell+bright clamp or shading) "
            "or --method depth_profile (1D brightness-vs-distance-to-surface inverse gain over a Zarr)"
        ),
    )
    parser.add_argument(
        "--method",
        choices=["log_field", "depth_profile"],
        default="log_field",
        help="log_field: current shading/shell behaviour (needs --input_dir); "
        "depth_profile: former depth_brightness_equalize (needs --image --mask --voxel_size_zyx)",
    )
    parser.add_argument("--input_dir", default=None, help="Input signal TIFF folder (log_field method)")
    parser.add_argument(
        "--output_zarr",
        default=None,
        help="Output Zarr path. log_field default: <input>_surface_homogenized.zarr beside the input; "
        "depth_profile default: <image>_brightness_equalized.zarr",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Deprecated alias of --output_zarr; a .zarr suffix is added if missing",
    )
    parser.add_argument("--downsample", type=int, default=8, help="Integer downsample factor for the 3D field (default: 8)")
    parser.add_argument("--mode", choices=["shell", "shading"], default="shading", help="log_field sub-mode: shell: press bright surface-band voxels to I_ref; shading: regional log-field normalize")
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
    parser.add_argument("--max_gain", type=float, default=None, help="Depth gain ceiling: log_field default 6.0, depth_profile default 5.0")
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
        help="Manual/SAM tissue mask Zarr or TIFF at QC resolution; skips automatic hysteresis (log_field)",
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
    # --- depth_profile method (former depth_brightness_equalize) ---
    parser.add_argument("--image", default=None, help="depth_profile: input image Zarr (target grid)")
    parser.add_argument("--mask", default=None, help="depth_profile: 3D tissue mask (TIFF, .npy or Zarr) on the fit grid")
    parser.add_argument(
        "--voxel_size_zyx",
        default=None,
        help="depth_profile: target image voxel size in µm as z,y,x (e.g. 2,1.8,1.8)",
    )
    parser.add_argument(
        "--fit_image",
        default=None,
        help="depth_profile: volume used to fit the curve; default is --image (must share the mask grid)",
    )
    parser.add_argument(
        "--fit_voxel_size_zyx",
        default=None,
        help="depth_profile: voxel size of --fit_image in µm as z,y,x; default is --voxel_size_zyx",
    )
    parser.add_argument("--min_gain", type=float, default=DEFAULT_MIN_GAIN, help="depth_profile: max darkening 1/min_gain (default 0.1 => 10x)")
    parser.add_argument("--bin_um", type=float, default=DEFAULT_BIN_UM, help="depth_profile: distance bin width in µm (default 10)")
    parser.add_argument("--smooth_bins", type=float, default=1.2, help="depth_profile: Gaussian smoothing of the profile in bins")
    parser.add_argument("--deep_fraction", type=float, default=0.1, help="depth_profile: fraction of deepest bins used as the plateau")
    parser.add_argument(
        "--edge_gain",
        type=float,
        default=1.0,
        help="depth_profile: fixed gain at the innermost mask voxel, blending back to the curve by --edge_um (1.0 = off)",
    )
    parser.add_argument("--edge_um", type=float, default=20.0, help="depth_profile: width of the dark edge ramp in microns (default 20)")
    parser.add_argument("--apply_chunk_z", type=int, default=0, help="depth_profile: z-slab size when applying (0 = auto)")
    parser.add_argument("--qc_dir", default=None, help="depth_profile: write before/after PNG previews here")
    parser.add_argument("--fit_only", action="store_true", help="depth_profile: fit the curve, write a JSON summary, do not apply")
    parser.add_argument("--overwrite", action="store_true", help="depth_profile: replace an existing output Zarr")
    parser.add_argument("--sample_dir", default=None, help="Sample dir for pm_progress.json reporting")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    _configure_logging(args.json_logs)
    max_gain = args.max_gain if args.max_gain is not None else (5.0 if args.method == "depth_profile" else 6.0)
    output_zarr = args.output_zarr or args.output_dir
    try:
        if args.method == "depth_profile":
            missing = [
                name
                for name, value in (
                    ("--image", args.image),
                    ("--mask", args.mask),
                    ("--voxel_size_zyx", args.voxel_size_zyx),
                )
                if not value
            ]
            if missing:
                raise PipelineError(
                    ErrorCode.ARGUMENT_INVALID,
                    "depth_profile needs " + " ".join(missing),
                    {"method": args.method},
                )
            result = equalize_depth_brightness(
                args.image,
                args.mask,
                output_zarr,
                voxel_size_zyx=args.voxel_size_zyx,
                fit_image=args.fit_image,
                fit_voxel_size_zyx=args.fit_voxel_size_zyx,
                max_gain=max_gain,
                min_gain=args.min_gain,
                bin_um=args.bin_um,
                smooth_bins=args.smooth_bins,
                deep_fraction=args.deep_fraction,
                target_scale=args.target_scale,
                edge_gain=args.edge_gain,
                edge_um=args.edge_um,
                apply_chunk_z=args.apply_chunk_z,
                compressor=args.compressor,
                qc_dir=args.qc_dir,
                fit_only=bool(args.fit_only),
                overwrite=bool(args.overwrite),
                sample_dir=args.sample_dir,
            )
        else:
            if not args.input_dir:
                raise PipelineError(
                    ErrorCode.ARGUMENT_INVALID,
                    "log_field needs --input_dir (signal TIFF folder)",
                    {"method": args.method},
                )
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
                max_gain=max_gain,
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
