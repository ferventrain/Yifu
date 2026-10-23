from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

try:
    from pipeline_modules.harness.progress import report_child_units, utc_now_iso
except ImportError:  # pragma: no cover
    def report_child_units(*_args, **_kwargs):
        return

    def utc_now_iso():
        return ""

try:
    from pipeline_modules.utils.errors import ErrorCode, PipelineError
    from pipeline_modules.utils.run_manifest import write_run_manifest
    from pipeline_modules.segmentation.zarr_utils import create_output_zarr, list_existing_chunk_indices, open_zarr_dataset
except ImportError:  # pragma: no cover
    from ..utils.errors import ErrorCode, PipelineError
    from ..utils.run_manifest import write_run_manifest
    from .zarr_utils import create_output_zarr, list_existing_chunk_indices, open_zarr_dataset

logger = logging.getLogger(__name__)

THRESHOLD_METHODS = ("global", "hysteresis", "percentile", "windowed")
PERCENTILE_METHODS = ("blend", "otsu", "triangle", "yen", "li", "knee")
WINDOWED_DEFAULTS = {
    "window": 151,
    "stride": 32,
    "floor": 2000.0,
    "pct_nudge": -1.3,
    "thr_scale": 0.85,
    "clamp_lo": 93.0,
    "clamp_hi": 98.5,
    "bright_local_p99": 5000.0,
    "bright_extra_nudge": -1.0,
}


def _tqdm_with_harness(total: int, desc: str, unit: str):
    started = utc_now_iso()
    pbar = tqdm(total=total, desc=desc, unit=unit)
    orig_update = pbar.update

    def update(n=1):
        result = orig_update(n)
        report_child_units(
            int(pbar.n),
            int(pbar.total or 0),
            phase=desc,
            phase_started_at=started,
            force=bool(pbar.total and pbar.n >= pbar.total),
        )
        return result

    pbar.update = update  # type: ignore[method-assign]
    report_child_units(0, int(total), phase=desc, phase_started_at=started, force=True)
    return pbar


def _require_threshold_stack():
    try:
        import numpy as np
        from scipy import ndimage
        from skimage.filters import threshold_otsu
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "numpy, scipy, and scikit-image are required for threshold segmentation"
        ) from exc
    return np, ndimage, threshold_otsu


def _configure_logging(json_logs: bool) -> None:
    if json_logs:
        class _JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                return json.dumps(
                    {
                        "level": record.levelname,
                        "logger": record.name,
                        "message": record.getMessage(),
                    },
                    ensure_ascii=False,
                )

        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonFormatter())
        logging.root.handlers.clear()
        logging.root.addHandler(handler)
        logging.root.setLevel(logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def segment_chunk(img, threshold, sigma, min_size, output_mode: str = "label"):
    np, ndimage, threshold_otsu = _require_threshold_stack()

    if sigma > 0:
        img = ndimage.gaussian_filter(img, sigma=sigma)

    if threshold == "otsu":
        try:
            thresh_val = threshold_otsu(img)
        except ValueError:
            thresh_val = 0
    else:
        thresh_val = float(threshold)

    binary_mask = img > thresh_val
    labeled, num_features = ndimage.label(binary_mask)

    if min_size > 0:
        sizes = ndimage.sum(binary_mask, labeled, range(num_features + 1))
        mask_size = sizes < min_size
        remove_pixel = mask_size[labeled]
        labeled[remove_pixel] = 0

    if output_mode == "binary":
        return (labeled > 0).astype(np.uint8)
    return labeled.astype(np.uint16)


# ---------------------------------------------------- histogram / hysteresis --
def knee_threshold(pos: np.ndarray, nbins: int = 512) -> float:
    """Knee on the bright-tail survival curve → intensity threshold."""
    lo, hi = np.percentile(pos, (1.0, 99.9))
    if hi <= lo:
        return float(np.median(pos))
    hist, edges = np.histogram(pos, bins=nbins, range=(float(lo), float(hi)), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    cdf = np.cumsum(hist)
    cdf = cdf / max(float(cdf[-1]), 1e-12)
    surv = 1.0 - cdf
    xs = (centers - centers[0]) / (centers[-1] - centers[0] + 1e-12)
    x0, y0 = float(xs[0]), float(surv[0])
    x1, y1 = float(xs[-1]), float(surv[-1])
    num = np.abs((y1 - y0) * xs - (x1 - x0) * surv + x1 * y0 - y1 * x0)
    den = float(np.hypot(y1 - y0, x1 - x0)) + 1e-12
    i = int(np.argmax(num / den))
    return float(centers[i])


def threshold_to_percentile(pos: np.ndarray, thr: float) -> float:
    """Percentile rank of ``thr`` among positive pixels (0-100)."""
    return float(100.0 * np.mean(pos <= thr))


def estimate_percentile_from_histogram(
    image: np.ndarray,
    *,
    method: str = "blend",
    clamp_lo: float = 95.0,
    clamp_hi: float = 99.5,
) -> dict:
    """Suggested percentile + diagnostics for one 2D/3D array.

    Methods: ``otsu`` / ``triangle`` / ``yen`` / ``li`` / ``knee`` use that
    single histogram rule; ``blend`` takes ``max(otsu%, knee%)`` then clamps.
    """
    from skimage.filters import threshold_li, threshold_otsu, threshold_triangle, threshold_yen

    img = np.asarray(image, dtype=np.float32)
    pos = img[img > 0]
    if pos.size < 64:
        pct = float(np.clip(99.0, clamp_lo, clamp_hi))
        return {
            "method": method,
            "suggested_percentile": pct,
            "threshold": float(np.percentile(img, pct)) if img.size else 0.0,
            "n_positive": int(pos.size),
            "candidates": {},
        }

    hi = float(np.percentile(pos, 99.9))
    clip = pos[pos <= hi] if hi > 0 else pos
    candidates_thr = {
        "otsu": float(threshold_otsu(clip)),
        "triangle": float(threshold_triangle(clip)),
        "yen": float(threshold_yen(clip)),
        "li": float(threshold_li(clip)),
        "knee": knee_threshold(pos),
    }
    candidates_pct = {k: threshold_to_percentile(pos, thr) for k, thr in candidates_thr.items()}

    method = method.lower().strip()
    if method == "blend":
        raw = max(candidates_pct["otsu"], candidates_pct["knee"])
    elif method in candidates_pct:
        raw = candidates_pct[method]
    else:
        raise ValueError(f"Unknown method={method!r}; expected blend|otsu|triangle|yen|li|knee")

    pct = float(np.clip(raw, clamp_lo, clamp_hi))
    return {
        "method": method,
        "raw_percentile": float(raw),
        "suggested_percentile": pct,
        "threshold": float(np.percentile(pos, pct)),
        "clamp": [float(clamp_lo), float(clamp_hi)],
        "n_positive": int(pos.size),
        "candidates_threshold": candidates_thr,
        "candidates_percentile": candidates_pct,
    }


def clean_mask_2d(mask: np.ndarray, min_size: int = 48, *, morph: bool = True) -> np.ndarray:
    from skimage.morphology import closing, disk, opening, remove_small_objects

    m = np.asarray(mask, dtype=bool)
    if int(min_size) > 1:
        m = remove_small_objects(m, max_size=max(int(min_size) - 1, 0))
    if morph:
        m = closing(m, disk(1))
        m = opening(m, disk(1))
        if int(min_size) > 1:
            m = remove_small_objects(m, max_size=max(int(min_size) - 1, 0))
    return m.astype(np.uint8)


def clean_mask_3d(volume: np.ndarray, min_size: int = 48) -> np.ndarray:
    """Remove small 3D connected components (26-neighborhood)."""
    from skimage.morphology import remove_small_objects

    m = np.asarray(volume, dtype=bool)
    if int(min_size) > 1:
        m = remove_small_objects(m, max_size=max(int(min_size) - 1, 0), connectivity=3)
    return m.astype(np.uint8)


def local_hist_threshold_map(
    img: np.ndarray,
    *,
    window: int = 151,
    stride: int = 32,
    floor: float = 2000.0,
    method: str = "blend",
    pct_nudge: float = -1.3,
    thr_scale: float = 0.85,
    clamp_lo: float = 93.0,
    clamp_hi: float = 98.5,
    bright_local_p99: float = 5000.0,
    bright_extra_nudge: float = -1.0,
) -> tuple[np.ndarray, dict]:
    import cv2

    h, w = img.shape
    half = int(window) // 2
    ys = list(range(half, max(h - half, half + 1), int(stride)))
    xs = list(range(half, max(w - half, half + 1), int(stride)))
    if not ys:
        ys = [h // 2]
    if not xs:
        xs = [w // 2]
    if h > window and ys[-1] != h - 1 - half:
        ys.append(h - 1 - half)
    if w > window and xs[-1] != w - 1 - half:
        xs.append(w - 1 - half)

    grid = np.zeros((len(ys), len(xs)), dtype=np.float32)
    pct_grid = np.zeros_like(grid)
    for iy, cy in enumerate(ys):
        y0, y1 = cy - half, cy + half + 1
        for ix, cx in enumerate(xs):
            x0, x1 = cx - half, cx + half + 1
            patch = img[y0:y1, x0:x1]
            pos = patch[patch > 0]
            if pos.size < 64:
                thr = float(floor)
                pct = 99.0
            else:
                est = estimate_percentile_from_histogram(
                    patch, method=method, clamp_lo=clamp_lo, clamp_hi=clamp_hi
                )
                pct = float(est["suggested_percentile"]) + float(pct_nudge)
                if float(np.percentile(pos, 99)) >= float(bright_local_p99):
                    pct += float(bright_extra_nudge)
                pct = float(np.clip(pct, clamp_lo, clamp_hi))
                thr = float(np.percentile(pos, pct)) * float(thr_scale)
                thr = max(thr, float(floor))
            grid[iy, ix] = thr
            pct_grid[iy, ix] = pct

    thr_map = cv2.resize(grid, (w, h), interpolation=cv2.INTER_LINEAR)
    thr_map = np.maximum(thr_map, float(floor)).astype(np.float32)
    stats = {
        "thr_p50": float(np.median(grid)),
        "thr_p95": float(np.percentile(grid, 95)),
        "pct_p50": float(np.median(pct_grid)),
        "pct_min": float(np.min(pct_grid)),
        "pct_max": float(np.max(pct_grid)),
        "n_windows": int(grid.size),
    }
    return thr_map, stats


def hysteresis_threshold_3d(volume: np.ndarray, high: float, low: float) -> np.ndarray:
    """Seeds above ``high`` grow through 26-connected voxels above ``low``."""
    from skimage.filters import apply_hysteresis_threshold

    return apply_hysteresis_threshold(volume, low, high)


def morphological_bridge(
    mask: np.ndarray,
    close_radius: int = 3,
    dilate_radius: int = 0,
) -> np.ndarray:
    from scipy import ndimage as ndi
    from skimage.morphology import ball

    if close_radius <= 0 and dilate_radius <= 0:
        return mask

    result = np.asarray(mask, dtype=bool)
    if dilate_radius > 0:
        result = ndi.binary_dilation(result, structure=ball(dilate_radius))
    if close_radius > 0:
        result = ndi.binary_closing(result, structure=ball(close_radius))
    if dilate_radius > 0:
        result = ndi.binary_erosion(result, structure=ball(dilate_radius))
    return result.astype(np.uint8)


def _threshold_slice(
    img: np.ndarray,
    *,
    method: str,
    sigma: float,
    min_size: int,
    percentile_method: str,
    clamp_lo: float,
    clamp_hi: float,
    windowed: dict[str, float],
    clean_3d: bool,
) -> np.ndarray:
    _, ndimage, _ = _require_threshold_stack()
    if sigma > 0:
        img = ndimage.gaussian_filter(img, sigma=sigma)
    img = np.asarray(img, dtype=np.float32)
    slice_min_size = 0 if clean_3d else int(min_size)

    if method == "percentile":
        pos = img[img > 0]
        if pos.size == 0:
            return np.zeros(img.shape, dtype=np.uint8)
        est = estimate_percentile_from_histogram(
            img, method=percentile_method, clamp_lo=clamp_lo, clamp_hi=clamp_hi
        )
        thr = float(np.percentile(pos, float(est["suggested_percentile"])))
        return clean_mask_2d(img > thr, min_size=slice_min_size)

    if method == "windowed":
        thr_map, _stats = local_hist_threshold_map(
            img,
            window=int(windowed["window"]),
            stride=int(windowed["stride"]),
            floor=float(windowed["floor"]),
            method=percentile_method,
            pct_nudge=float(windowed["pct_nudge"]),
            thr_scale=float(windowed["thr_scale"]),
            clamp_lo=clamp_lo,
            clamp_hi=clamp_hi,
            bright_local_p99=float(windowed["bright_local_p99"]),
            bright_extra_nudge=float(windowed["bright_extra_nudge"]),
        )
        mask = (img >= thr_map) & (img >= float(windowed["floor"]))
        return clean_mask_2d(mask, min_size=slice_min_size)

    raise ValueError(f"Unsupported per-slice method: {method}")


def _run_per_slice(
    *,
    input_path: Path,
    output_path: Path,
    data_in,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    method: str,
    percentile_method: str,
    clamp_lo: float,
    clamp_hi: float,
    windowed: dict[str, float],
    sigma: float,
    min_size: int,
    output_mode: str,
    dataset_name: str,
    resume: bool,
    clean_3d: bool,
) -> dict[str, Any]:
    _, ndimage, _ = _require_threshold_stack()
    output_dtype = "uint8" if output_mode == "binary" else "uint16"

    if resume and output_path.exists():
        import zarr

        root_out = zarr.open(str(output_path), mode="r+")
        data_out = root_out[dataset_name]
        incomplete = _find_incomplete_z_ranges(data_out, shape, chunks)
        z_chunk = chunks[0]
        z_indices = [
            z
            for z_idx in incomplete
            for z in range(z_idx * z_chunk, min((z_idx + 1) * z_chunk, shape[0]))
        ]
        logger.info("Resuming per-slice segmentation: %d slices in incomplete z-chunks", len(z_indices))
    else:
        _, data_out = create_output_zarr(
            output_path, shape, chunks, output_dtype, dataset_name=dataset_name
        )
        z_indices = list(range(shape[0]))

    mask_stack = np.zeros(shape, dtype=np.uint8) if clean_3d else None
    pbar = _tqdm_with_harness(len(z_indices), f"Threshold segmentation ({method})", "slice")
    for z in z_indices:
        mask = _threshold_slice(
            np.asarray(data_in[z]),
            method=method,
            sigma=sigma,
            min_size=min_size,
            percentile_method=percentile_method,
            clamp_lo=clamp_lo,
            clamp_hi=clamp_hi,
            windowed=windowed,
            clean_3d=clean_3d,
        )
        if clean_3d:
            mask_stack[z] = mask
            data_out[z] = mask
        elif output_mode == "label":
            labeled, _ = ndimage.label(mask)
            data_out[z] = labeled.astype(np.uint16)
        else:
            data_out[z] = mask
        pbar.update(1)
    pbar.close()

    if clean_3d:
        cleaned = clean_mask_3d(mask_stack, min_size=min_size)
        if output_mode == "label":
            cleaned, _ = ndimage.label(cleaned)
            cleaned = cleaned.astype(np.uint16)
        pbar = _tqdm_with_harness(shape[0], "Write 3D-cleaned mask", "slice")
        for z in range(shape[0]):
            data_out[z] = cleaned[z]
            pbar.update(1)
        pbar.close()

    return {"processed_slices": len(z_indices), "per_slice_method": method}


def _run_hysteresis(
    *,
    input_path: Path,
    output_path: Path,
    data_in,
    shape: tuple[int, ...],
    low: float,
    high: float,
    close_radius: int,
    slab_depth: int,
    dataset_name: str,
) -> dict[str, Any]:
    nbytes = int(np.prod(shape)) * np.dtype(data_in.dtype).itemsize
    use_full_volume = nbytes < 4 * 1024**3
    out_chunks = (min(64, shape[0]), min(256, shape[1]), min(256, shape[2]))
    _, data_out = create_output_zarr(
        output_path, shape, out_chunks, "uint8", dataset_name=dataset_name
    )
    n_fg = 0
    total_slabs = 1

    if use_full_volume:
        logger.info("Loading full volume (%d MB) for global hysteresis", nbytes // (1024 * 1024))
        vol = np.asarray(data_in[:])
        mask = hysteresis_threshold_3d(vol, high=high, low=low).astype(np.uint8)
        del vol
        if close_radius > 0:
            mask = morphological_bridge(mask, close_radius=close_radius)
        data_out[:] = mask
        n_fg = int(mask.sum())
    else:
        overlap = max(close_radius * 2, 16)
        depth = shape[0]
        total_slabs = max(1, (depth + slab_depth - 1) // slab_depth)
        logger.info(
            "Volume too large for single pass (%d MB); %d slabs at depth %d, overlap=%d",
            nbytes // (1024 * 1024), total_slabs, slab_depth, overlap,
        )
        pbar = _tqdm_with_harness(total_slabs, "Hysteresis threshold", "slab")
        for slab_idx in range(total_slabs):
            write_start = slab_idx * slab_depth
            write_end = min(write_start + slab_depth, depth)
            read_start = max(0, write_start - overlap)
            read_end = min(depth, write_end + overlap)
            slab = np.asarray(data_in[read_start:read_end])
            mask_slab = hysteresis_threshold_3d(slab, high=high, low=low).astype(np.uint8)
            if close_radius > 0:
                mask_slab = morphological_bridge(mask_slab, close_radius=close_radius)
            inner_start = write_start - read_start
            inner_end = inner_start + (write_end - write_start)
            data_out[write_start:write_end] = mask_slab[inner_start:inner_end]
            n_fg += int(mask_slab[inner_start:inner_end].sum())
            pbar.update(1)
        pbar.close()

    return {
        "processed_chunks": total_slabs if not use_full_volume else 1,
        "full_volume_mode": bool(use_full_volume),
        "foreground_voxels": n_fg,
        "low": float(low),
        "high": float(high),
        "close_radius": int(close_radius),
        "slab_depth": int(slab_depth),
    }



def _find_incomplete_z_ranges(data_out, shape, chunks):
    try:
        from pipeline_modules.utils.zarr_io import list_existing_chunk_indices
    except ImportError:
        from ..utils.zarr_io import list_existing_chunk_indices

    existing = set(list_existing_chunk_indices(data_out))

    z_chunk_size = chunks[0]
    n_z_chunks = (shape[0] + z_chunk_size - 1) // z_chunk_size
    n_y_chunks = (shape[1] + chunks[1] - 1) // chunks[1]
    n_x_chunks = (shape[2] + chunks[2] - 1) // chunks[2]
    expected_per_z = n_y_chunks * n_x_chunks

    incomplete = []
    for z_idx in range(n_z_chunks):
        z_found = sum(1 for (cz, cy, cx) in existing if cz == z_idx)
        if z_found < expected_per_z:
            incomplete.append(z_idx)

    return incomplete


def _write_result_manifest(
    result: dict[str, Any],
    *,
    started_at: float,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = write_run_manifest(
        result["output_zarr"],
        module="segmentation.intensity_threshold",
        entrypoint="run_threshold_segmentation",
        inputs=inputs,
        outputs=[result["output_zarr"]],
        started_at=started_at,
        extra=result,
    )
    result["manifest_path"] = str(manifest_path)
    return result


def run_threshold_segmentation(
    *,
    input_zarr: str | Path,
    output_zarr: str | Path,
    threshold: str = "otsu",
    sigma: float = 0.0,
    min_size: int = 10,
    output_mode: str = "binary",
    dataset_name: str = "0",
    process_existing_only: bool = False,
    resume: bool = False,
    tile_size: int = 1024,
    method: str = "global",
    low: float | None = None,
    high: float | None = None,
    close_radius: int = 0,
    slab_depth: int = 64,
    percentile_method: str = "blend",
    clamp_lo: float | None = None,
    clamp_hi: float | None = None,
    window: int | None = None,
    stride: int | None = None,
    floor: float | None = None,
    pct_nudge: float | None = None,
    thr_scale: float | None = None,
    bright_local_p99: float | None = None,
    bright_extra_nudge: float | None = None,
    clean_3d: bool = False,
) -> dict[str, Any]:
    started_at = time.time()
    input_path = Path(input_zarr)
    if not input_path.exists():
        raise PipelineError(
            ErrorCode.INPUT_NOT_FOUND,
            "Input Zarr not found",
            {"input_zarr": str(input_path)},
        )

    method = str(method or "global").strip().lower()
    if method not in THRESHOLD_METHODS:
        raise PipelineError(
            ErrorCode.CONFIG_INVALID,
            "Unknown threshold method",
            {"method": method, "expected": list(THRESHOLD_METHODS)},
        )
    percentile_method = str(percentile_method or "blend").strip().lower()
    if method in ("percentile", "windowed") and percentile_method not in PERCENTILE_METHODS:
        raise PipelineError(
            ErrorCode.CONFIG_INVALID,
            "Unknown percentile method",
            {"percentile_method": percentile_method, "expected": list(PERCENTILE_METHODS)},
        )

    data_in = open_zarr_dataset(input_path, dataset_name=dataset_name)
    shape = tuple(int(v) for v in data_in.shape)
    chunks = tuple(int(v) for v in data_in.chunks)
    output_dtype = "uint8" if output_mode == "binary" else "uint16"
    output_path = Path(output_zarr)

    if method == "hysteresis":
        if low is None or high is None:
            raise PipelineError(
                ErrorCode.CONFIG_INVALID,
                "Hysteresis thresholding requires low and high",
                {"low": low, "high": high},
            )
        if float(low) > float(high):
            raise PipelineError(
                ErrorCode.CONFIG_INVALID,
                "Hysteresis low must not exceed high",
                {"low": float(low), "high": float(high)},
            )
        if sigma > 0:
            raise PipelineError(
                ErrorCode.CONFIG_INVALID,
                "sigma is not supported for hysteresis (global connectivity)",
                {"sigma": float(sigma)},
            )
        if resume or process_existing_only:
            raise PipelineError(
                ErrorCode.CONFIG_INVALID,
                "Hysteresis runs one global pass; resume/test modes are not supported",
                {"resume": bool(resume), "process_existing_only": bool(process_existing_only)},
            )
        if output_mode != "binary":
            raise PipelineError(
                ErrorCode.CONFIG_INVALID,
                "Hysteresis only produces a binary mask",
                {"output_mode": output_mode},
            )
        result = {
            "success": True,
            "input_zarr": str(input_path),
            "output_zarr": str(output_path),
            "dataset_name": dataset_name,
            "method": method,
            "shape": list(shape),
            "chunks": list(chunks),
            "min_size": int(min_size),
            "output_mode": output_mode,
            **_run_hysteresis(
                input_path=input_path,
                output_path=output_path,
                data_in=data_in,
                shape=shape,
                low=float(low),
                high=float(high),
                close_radius=int(close_radius),
                slab_depth=int(slab_depth),
                dataset_name=dataset_name,
            ),
        }
        return _write_result_manifest(
            result,
            started_at=started_at,
            inputs={
                "input_zarr": str(input_path),
                "output_zarr": str(output_path),
                "dataset_name": dataset_name,
                "method": method,
                "low": low,
                "high": high,
                "close_radius": close_radius,
                "slab_depth": slab_depth,
            },
        )

    if method in ("percentile", "windowed"):
        if clean_3d and (resume or process_existing_only):
            raise PipelineError(
                ErrorCode.CONFIG_INVALID,
                "clean_3d needs a full pass; resume/test modes are not supported",
                {"resume": bool(resume), "process_existing_only": bool(process_existing_only)},
            )
        if method == "percentile":
            clamp_lo = 95.0 if clamp_lo is None else float(clamp_lo)
            clamp_hi = 99.5 if clamp_hi is None else float(clamp_hi)
        else:
            clamp_lo = WINDOWED_DEFAULTS["clamp_lo"] if clamp_lo is None else float(clamp_lo)
            clamp_hi = WINDOWED_DEFAULTS["clamp_hi"] if clamp_hi is None else float(clamp_hi)
        if not (0.0 <= clamp_lo < clamp_hi <= 100.0):
            raise PipelineError(
                ErrorCode.CONFIG_INVALID,
                "Percentile clamp must satisfy 0 <= clamp_lo < clamp_hi <= 100",
                {"clamp_lo": clamp_lo, "clamp_hi": clamp_hi},
            )
        windowed_params = {
            key: float(WINDOWED_DEFAULTS[key] if value is None else value)
            for key, value in {
                "window": window,
                "stride": stride,
                "floor": floor,
                "pct_nudge": pct_nudge,
                "thr_scale": thr_scale,
                "bright_local_p99": bright_local_p99,
                "bright_extra_nudge": bright_extra_nudge,
            }.items()
        }
        result = {
            "success": True,
            "input_zarr": str(input_path),
            "output_zarr": str(output_path),
            "dataset_name": dataset_name,
            "method": method,
            "percentile_method": percentile_method,
            "clamp_lo": clamp_lo,
            "clamp_hi": clamp_hi,
            "shape": list(shape),
            "chunks": list(chunks),
            "sigma": float(sigma),
            "min_size": int(min_size),
            "output_mode": output_mode,
            "clean_3d": bool(clean_3d),
            "windowed_params": windowed_params if method == "windowed" else None,
            **_run_per_slice(
                input_path=input_path,
                output_path=output_path,
                data_in=data_in,
                shape=shape,
                chunks=chunks,
                method=method,
                percentile_method=percentile_method,
                clamp_lo=clamp_lo,
                clamp_hi=clamp_hi,
                windowed=windowed_params,
                sigma=float(sigma),
                min_size=int(min_size),
                output_mode=output_mode,
                dataset_name=dataset_name,
                resume=bool(resume),
                clean_3d=bool(clean_3d),
            ),
        }
        return _write_result_manifest(
            result,
            started_at=started_at,
            inputs={
                "input_zarr": str(input_path),
                "output_zarr": str(output_path),
                "dataset_name": dataset_name,
                "method": method,
                "percentile_method": percentile_method,
                "clamp_lo": clamp_lo,
                "clamp_hi": clamp_hi,
                "sigma": sigma,
                "min_size": min_size,
                "output_mode": output_mode,
                "clean_3d": clean_3d,
                **{f"windowed_{key}": value for key, value in windowed_params.items()},
            },
        )

    processed_chunks = 0

    if resume and output_path.exists():
        import zarr
        logger.info("Resuming: opening existing output Zarr at %s", output_path)
        root_out = zarr.open(str(output_path), mode='r+')
        data_out = root_out[dataset_name]
        incomplete_z_indices = _find_incomplete_z_ranges(data_out, shape, chunks)
        logger.info("Found %d incomplete z-ranges out of %d total z-chunks (incomplete: %s)",
                     len(incomplete_z_indices),
                     (shape[0] + chunks[0] - 1) // chunks[0],
                     incomplete_z_indices)

        z_chunk_size = chunks[0]
        n_y = (shape[1] + tile_size - 1) // tile_size
        n_x = (shape[2] + tile_size - 1) // tile_size
        pbar = _tqdm_with_harness(len(incomplete_z_indices) * n_y * n_x, "Resume threshold segmentation", "tile")
        for z_idx in incomplete_z_indices:
            z = z_idx * z_chunk_size
            z_end = min(z + z_chunk_size, shape[0])
            for y in range(0, shape[1], tile_size):
                y_end = min(y + tile_size, shape[1])
                for x in range(0, shape[2], tile_size):
                    x_end = min(x + tile_size, shape[2])
                    data_out[z:z_end, y:y_end, x:x_end] = segment_chunk(
                        data_in[z:z_end, y:y_end, x:x_end],
                        threshold,
                        sigma,
                        min_size,
                        output_mode=output_mode,
                    )
                    pbar.update(1)
            processed_chunks += 1
        pbar.close()
    else:
        _, data_out = create_output_zarr(output_zarr, shape, chunks, output_dtype, dataset_name=dataset_name)

        if process_existing_only:
            existing_indices = list_existing_chunk_indices(data_in)
            if not existing_indices:
                raise PipelineError(
                    ErrorCode.EMPTY_RESULT,
                    "No physical chunks found in input store",
                    {"input_zarr": str(input_path)},
                )
            pbar = _tqdm_with_harness(len(existing_indices), "Threshold segmentation (test mode)", "chunk")
            for idx in existing_indices:
                slices = []
                for axis, chunk_idx in enumerate(idx):
                    start = chunk_idx * chunks[axis]
                    stop = min(start + chunks[axis], shape[axis])
                    slices.append(slice(start, stop))
                slices = tuple(slices)
                data_out[slices] = segment_chunk(
                    data_in[slices],
                    threshold,
                    sigma,
                    min_size,
                    output_mode=output_mode,
                )
                processed_chunks += 1
                pbar.update(1)
            pbar.close()
        else:
            z_chunk_size = chunks[0]
            n_z = (shape[0] + z_chunk_size - 1) // z_chunk_size
            n_y = (shape[1] + tile_size - 1) // tile_size
            n_x = (shape[2] + tile_size - 1) // tile_size
            pbar = _tqdm_with_harness(n_z * n_y * n_x, "Threshold segmentation", "tile")
            for z in range(0, shape[0], z_chunk_size):
                z_end = min(z + z_chunk_size, shape[0])
                for y in range(0, shape[1], tile_size):
                    y_end = min(y + tile_size, shape[1])
                    for x in range(0, shape[2], tile_size):
                        x_end = min(x + tile_size, shape[2])
                        data_out[z:z_end, y:y_end, x:x_end] = segment_chunk(
                            data_in[z:z_end, y:y_end, x:x_end],
                            threshold,
                            sigma,
                            min_size,
                            output_mode=output_mode,
                        )
                        pbar.update(1)
                processed_chunks += 1
            pbar.close()

    result = {
        "success": True,
        "input_zarr": str(input_path),
        "output_zarr": str(Path(output_zarr)),
        "dataset_name": dataset_name,
        "method": method,
        "shape": list(shape),
        "chunks": list(chunks),
        "processed_chunks": processed_chunks,
        "threshold": str(threshold),
        "sigma": float(sigma),
        "min_size": int(min_size),
        "output_mode": output_mode,
    }
    return _write_result_manifest(
        result,
        started_at=started_at,
        inputs={
            "input_zarr": str(input_path),
            "output_zarr": str(output_zarr),
            "dataset_name": dataset_name,
            "method": method,
            "threshold": threshold,
            "sigma": sigma,
            "min_size": min_size,
            "output_mode": output_mode,
            "process_existing_only": process_existing_only,
            "resume": resume,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run threshold segmentation on a Zarr volume")
    parser.add_argument("--input_zarr", required=True, help="Path to input .zarr directory")
    parser.add_argument("--output_zarr", required=True, help="Path for output .zarr directory")
    parser.add_argument(
        "--method",
        choices=THRESHOLD_METHODS,
        default="global",
        help="global: one threshold (or otsu) per tile; hysteresis: dual thresholds with "
             "global connectivity; percentile: per-slice histogram percentile; "
             "windowed: per-slice local histogram percentile",
    )
    parser.add_argument("--threshold", default="otsu", help='Global threshold value or "otsu"')
    parser.add_argument("--sigma", type=float, default=0.0, help="Gaussian smoothing sigma")
    parser.add_argument("--min_size", type=int, default=10, help="Minimum object size")
    parser.add_argument("--output_mode", choices=["label", "binary"], default="binary")
    parser.add_argument("--dataset_name", default="0", help="Dataset name inside the Zarr group")
    parser.add_argument("--test", action="store_true", help="Only process chunks that physically exist in the input store")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output Zarr, skipping completed z-chunks")
    parser.add_argument("--tile_size", type=int, default=1024, help="XY tile size for memory-limited processing (default: 1024)")
    parser.add_argument("--json_logs", action="store_true", help="Emit NDJSON log records to stderr")

    parser.add_argument("--low", type=float, default=None, help="Hysteresis low (extension) threshold")
    parser.add_argument("--high", type=float, default=None, help="Hysteresis high (seed) threshold")
    parser.add_argument("--close_radius", type=int, default=0, help="Hysteresis morphological closing radius")
    parser.add_argument("--slab_depth", type=int, default=64, help="Hysteresis slab depth for volumes > 4 GB")

    parser.add_argument(
        "--percentile_method",
        choices=PERCENTILE_METHODS,
        default="blend",
        help="Histogram rule for percentile/windowed methods (default: blend)",
    )
    parser.add_argument("--clamp_lo", type=float, default=None, help="Percentile clamp lower bound")
    parser.add_argument("--clamp_hi", type=float, default=None, help="Percentile clamp upper bound")
    parser.add_argument("--window", type=int, default=None, help="Windowed method: local window size")
    parser.add_argument("--stride", type=int, default=None, help="Windowed method: window stride")
    parser.add_argument("--floor", type=float, default=None, help="Windowed method: absolute intensity floor")
    parser.add_argument("--pct_nudge", type=float, default=None, help="Windowed method: percentile nudge")
    parser.add_argument("--thr_scale", type=float, default=None, help="Windowed method: threshold scale factor")
    parser.add_argument("--bright_local_p99", type=float, default=None, help="Windowed method: bright-window p99 cutoff")
    parser.add_argument("--bright_extra_nudge", type=float, default=None, help="Windowed method: extra nudge for bright windows")
    parser.add_argument(
        "--clean_3d",
        action="store_true",
        help="Percentile/windowed methods: defer min_size to 3D connected components",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _configure_logging(args.json_logs)
    try:
        result = run_threshold_segmentation(
            input_zarr=args.input_zarr,
            output_zarr=args.output_zarr,
            threshold=args.threshold,
            sigma=args.sigma,
            min_size=args.min_size,
            output_mode=args.output_mode,
            dataset_name=args.dataset_name,
            process_existing_only=args.test,
            resume=args.resume,
            tile_size=args.tile_size,
            method=args.method,
            low=args.low,
            high=args.high,
            close_radius=args.close_radius,
            slab_depth=args.slab_depth,
            percentile_method=args.percentile_method,
            clamp_lo=args.clamp_lo,
            clamp_hi=args.clamp_hi,
            window=args.window,
            stride=args.stride,
            floor=args.floor,
            pct_nudge=args.pct_nudge,
            thr_scale=args.thr_scale,
            bright_local_p99=args.bright_local_p99,
            bright_extra_nudge=args.bright_extra_nudge,
            clean_3d=args.clean_3d,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except PipelineError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False), file=sys.stderr)
        return exc.exit_code
    except Exception as exc:  # pragma: no cover
        logger.exception("Unhandled threshold segmentation error: %s", exc)
        wrapped = PipelineError(ErrorCode.INTERNAL_ERROR, "Unhandled threshold segmentation error", {"error": str(exc)})
        print(json.dumps(wrapped.to_dict(), ensure_ascii=False), file=sys.stderr)
        return wrapped.exit_code


if __name__ == "__main__":
    sys.exit(main())
