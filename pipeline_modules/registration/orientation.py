"""Detect atlas/sample axis flips from brain-mask shape.

The sample and atlas masks are compared after bounding-box crop and isotropic
resampling. Mouse brain masks are asymmetric along AP (olfactory bulb vs
cerebellum) and DV (cortex vs hypothalamus), which makes those flips
recoverable from Dice alone. Left/right is anatomically near-symmetric, so the
ML component is reported as ambiguous and is never auto-applied.

Array axes are (X, Y, Z) = (ML, AP, DV), matching the registration NIfTI and
the atlas label loaded in ANTs order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import ndimage

DEFAULT_GRID = 96
DEFAULT_MIN_CONFIDENCE = 0.02
ML_TIE_TOL = 1e-3


def axes_from_flip(flip: Sequence[bool]) -> tuple[int, ...]:
    return tuple(i for i, value in enumerate(flip) if bool(value))


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, count = ndimage.label(mask)
    if count <= 1:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == int(sizes.argmax())


def _crop_bbox(mask: np.ndarray) -> np.ndarray:
    coords = np.nonzero(mask)
    if coords[0].size == 0:
        raise ValueError("mask is empty")
    slices = tuple(slice(int(axis.min()), int(axis.max()) + 1) for axis in coords)
    return mask[slices]


def _fit_to_grid(mask: np.ndarray, grid: int) -> np.ndarray:
    if mask.shape == (grid, grid, grid):
        return mask
    out = np.zeros((grid, grid, grid), dtype=bool)
    src: list[slice] = []
    dst: list[slice] = []
    for size in mask.shape:
        if size >= grid:
            start = (size - grid) // 2
            src.append(slice(start, start + grid))
            dst.append(slice(0, grid))
        else:
            start = (grid - size) // 2
            src.append(slice(0, size))
            dst.append(slice(start, start + size))
    out[tuple(dst)] = mask[tuple(src)]
    return out


def _resize_mask(mask: np.ndarray, grid: int) -> np.ndarray:
    if mask.shape == (grid, grid, grid):
        return mask
    zoom = tuple(grid / size for size in mask.shape)
    resized = ndimage.zoom(
        mask.astype(np.float32), zoom, order=1, mode="nearest", prefilter=False
    ) > 0.5
    return _fit_to_grid(resized, grid)


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return 0.0
    return 2.0 * float(np.logical_and(a, b).sum()) / denom


def _prepare_mask(mask: np.ndarray, grid: int, name: str) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 3:
        raise ValueError(f"{name} mask must be 3-D, got shape {arr.shape}")
    if not arr.any():
        raise ValueError(f"{name} mask is empty")
    return _resize_mask(_crop_bbox(_largest_component(arr)), grid)


def detect_axis_flips(
    sample_mask: np.ndarray,
    atlas_mask: np.ndarray,
    *,
    grid: int = DEFAULT_GRID,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> dict[str, Any]:
    """Return the atlas flip that best matches the sample orientation.

    ``detected_flip`` is in ``[flip_x, flip_y, flip_z]`` config order
    (x=ML, y=AP, z=DV). Its ML component is always ``False``: the ML decision
    is ambiguous for a near-symmetric brain and is left to manual config.

    ``confidence`` is the Dice gap between the best and second-best AP/DV
    orientation (each scored with its better ML orientation), so the
    unknowable ML component cannot inflate it.
    """
    sample_small = _prepare_mask(sample_mask, grid, "sample")
    atlas_small = _prepare_mask(atlas_mask, grid, "atlas")

    scores: dict[tuple[bool, bool, bool], float] = {}
    for flip_x in (False, True):
        for flip_y in (False, True):
            for flip_z in (False, True):
                flip = (flip_x, flip_y, flip_z)
                candidate = sample_small
                axes = axes_from_flip(flip)
                if axes:
                    candidate = np.flip(candidate, axis=axes)
                scores[flip] = _dice(candidate, atlas_small)

    best_by_apdv: dict[tuple[bool, bool], tuple[float, tuple[bool, bool, bool]]] = {}
    for flip, score in scores.items():
        key = (flip[1], flip[2])
        current = best_by_apdv.get(key)
        if current is None or score > current[0]:
            best_by_apdv[key] = (score, flip)

    ranked = sorted(best_by_apdv.items(), key=lambda item: item[1][0], reverse=True)
    _, (best_score, best_flip) = ranked[0]
    runner_up_score = ranked[1][1][0] if len(ranked) > 1 else 0.0
    confidence = best_score - runner_up_score

    ml_false_score = scores[(False, best_flip[1], best_flip[2])]
    ml_true_score = scores[(True, best_flip[1], best_flip[2])]
    ml_gap = abs(ml_false_score - ml_true_score)

    return {
        "axis_order": ["ML", "AP", "DV"],
        "detected_flip": [False, bool(best_flip[1]), bool(best_flip[2])],
        "best_score": round(best_score, 6),
        "runner_up_score": round(runner_up_score, 6),
        "confidence": round(confidence, 6),
        "threshold": float(min_confidence),
        "reliable": bool(confidence >= min_confidence),
        "ml_ambiguous": bool(ml_gap <= ML_TIE_TOL),
        "ml_score_gap": round(ml_gap, 6),
        "scores": {
            ",".join("1" if value else "0" for value in flip): round(score, 6)
            for flip, score in scores.items()
        },
        "grid": int(grid),
        "sample_shape_xyz": [int(v) for v in np.asarray(sample_mask).shape],
        "atlas_shape_xyz": [int(v) for v in np.asarray(atlas_mask).shape],
    }


def write_orientation_report(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
