#!/usr/bin/env python3
"""Preview tissue mask + centerline on a downsampled SpinalJ-oriented volume.

Intended for quick visual QC before straightening / atlas registration.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import tifffile

logger = logging.getLogger(__name__)


def _to_u8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = np.percentile(arr, (1, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    return (np.clip((arr - lo) / (hi - lo), 0, 1) * 255.0).astype(np.uint8)


def keep_largest_cc_2d(mask2d: np.ndarray) -> np.ndarray:
    """Keep only the largest 2D connected component on a binary slice."""
    from scipy import ndimage

    if not mask2d.any():
        return mask2d
    labeled, nlab = ndimage.label(mask2d)
    if nlab <= 1:
        return mask2d.astype(bool, copy=False)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    return labeled == int(np.argmax(counts))


def filter_small_cc_2d(
    mask_zyx: np.ndarray,
    *,
    min_area_frac_of_largest: float = 0.15,
    min_area_vox: int = 50,
    keep_largest_only: bool = True,
    open_radius_2d: int = 5,
    close_radius_2d: int = 3,
) -> tuple[np.ndarray, dict]:
    """Per-Z slice: open to cut thin bridges, then drop small 2D CCs.

    Default keeps only the largest component per slice. Opening severs rootlets /
    noise peninsulas that stay attached by thin bridges after 3D cleanup.
    """
    from scipy import ndimage

    if open_radius_2d > 0:
        struct_open = np.ones((open_radius_2d, open_radius_2d), dtype=bool)
    else:
        struct_open = None
    if close_radius_2d > 0:
        struct_close = np.ones((close_radius_2d, close_radius_2d), dtype=bool)
    else:
        struct_close = None

    out = np.zeros_like(mask_zyx, dtype=bool)
    n_removed = 0
    n_empty = 0
    for z in range(mask_zyx.shape[0]):
        sl = mask_zyx[z]
        if not sl.any():
            n_empty += 1
            continue
        # Cut thin bridges / rootlets before CC analysis.
        if struct_open is not None:
            sl = ndimage.binary_opening(sl, structure=struct_open)
        if not sl.any():
            # Opening wiped the slice; fall back to original largest CC.
            sl = keep_largest_cc_2d(mask_zyx[z])
            if not sl.any():
                n_empty += 1
                continue
            labeled, nlab = ndimage.label(sl)
            counts = np.bincount(labeled.ravel())
            counts[0] = 0
            cleaned = labeled == int(np.argmax(counts))
        else:
            labeled, nlab = ndimage.label(sl)
            counts = np.bincount(labeled.ravel())
            counts[0] = 0
            largest = int(np.argmax(counts))
            largest_area = int(counts[largest])
            if keep_largest_only:
                keep = {largest}
                n_removed += int(max(nlab - 1, 0))
            else:
                thr_area = max(min_area_vox, int(round(min_area_frac_of_largest * largest_area)))
                keep = {i for i in range(1, nlab + 1) if counts[i] >= thr_area}
                n_removed += int(nlab - len(keep))
            cleaned = np.isin(labeled, list(keep))
            if struct_close is not None:
                cleaned = ndimage.binary_closing(cleaned, structure=struct_close)
            cleaned = ndimage.binary_fill_holes(cleaned)
        out[z] = cleaned

    meta = {
        "keep_largest_only": keep_largest_only,
        "min_area_frac_of_largest": min_area_frac_of_largest,
        "min_area_vox": min_area_vox,
        "open_radius_2d": open_radius_2d,
        "close_radius_2d": close_radius_2d,
        "components_removed_2d": n_removed,
        "empty_slices": n_empty,
        "fg_frac_after": float(out.mean()),
    }
    return out, meta


def make_tissue_mask_zyx(
    vol_zyx: np.ndarray,
    *,
    thr_frac_of_p99: float = 0.10,
    smooth_sigma: float = 1.5,
    close_radius: int = 4,
    open_radius: int = 2,
    filter_2d_cc: bool = True,
    keep_largest_2d_only: bool = True,
    min_area_frac_of_largest: float = 0.15,
    min_area_vox: int = 50,
    open_radius_2d: int = 5,
    close_radius_2d: int = 3,
) -> tuple[np.ndarray, dict]:
    from scipy import ndimage

    vol = np.asarray(vol_zyx, dtype=np.float32)
    smooth = ndimage.gaussian_filter(vol, sigma=smooth_sigma)
    p99 = float(np.percentile(smooth, 99.0))
    thr = max(p99 * thr_frac_of_p99, 1e-6)
    mask = smooth > thr
    if close_radius > 0:
        mask = ndimage.binary_closing(mask, iterations=1, structure=np.ones((close_radius,) * 3))
    if open_radius > 0:
        mask = ndimage.binary_opening(mask, iterations=1, structure=np.ones((open_radius,) * 3))

    labeled, nlab = ndimage.label(mask)
    if nlab == 0:
        raise RuntimeError(f"Empty mask at thr={thr:.4g} (p99={p99:.4g})")
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    mask = labeled == int(np.argmax(counts))
    for z in range(mask.shape[0]):
        if mask[z].any():
            mask[z] = ndimage.binary_fill_holes(mask[z])

    cc2d_meta = None
    if filter_2d_cc:
        mask, cc2d_meta = filter_small_cc_2d(
            mask,
            min_area_frac_of_largest=min_area_frac_of_largest,
            min_area_vox=min_area_vox,
            keep_largest_only=keep_largest_2d_only,
            open_radius_2d=open_radius_2d,
            close_radius_2d=close_radius_2d,
        )

    meta = {
        "thr": thr,
        "p99": p99,
        "thr_frac_of_p99": thr_frac_of_p99,
        "fg_frac": float(mask.mean()),
        "smooth_sigma": smooth_sigma,
        "close_radius": close_radius,
        "open_radius": open_radius,
        "filter_2d_cc": filter_2d_cc,
        "cc2d": cc2d_meta,
    }
    return mask, meta


def extract_centerline_zyx(
    mask_zyx: np.ndarray,
    vol_zyx: np.ndarray | None = None,
    *,
    mode: str = "intensity",
) -> np.ndarray:
    """Per-Z centerline; returns Nx3 (z, y, x).

    mode='distance': EDT peak on largest 2D CC (geometric).
    mode='intensity': intensity-weighted centroid inside largest 2D CC
    (better when a bright central canal/tract pulls geometric peaks to side lobes).
    """
    from scipy import ndimage

    if mode not in {"distance", "intensity"}:
        raise ValueError(f"Unknown centerline mode: {mode}")
    if mode == "intensity" and vol_zyx is None:
        raise ValueError("intensity mode requires vol_zyx")

    pts = []
    prev = None
    for z in range(mask_zyx.shape[0]):
        sl = keep_largest_cc_2d(mask_zyx[z])
        if not sl.any():
            continue
        if mode == "distance":
            dist = ndimage.distance_transform_edt(sl)
            dmax = float(dist.max())
            cand = dist >= max(dmax * 0.95, dmax - 1.0)
            ys, xs = np.where(cand)
            if prev is None:
                y = float(ys.mean())
                x = float(xs.mean())
            else:
                d2 = (ys.astype(np.float64) - prev[0]) ** 2 + (xs.astype(np.float64) - prev[1]) ** 2
                j = int(np.argmin(d2))
                y = float(ys[j])
                x = float(xs[j])
        else:
            inten = np.asarray(vol_zyx[z], dtype=np.float64)
            inten = np.where(sl, inten, 0.0)
            # Soften speckles so a few bright outliers don't dominate.
            inten = ndimage.gaussian_filter(inten, sigma=1.0)
            inten = np.where(sl, np.clip(inten, 0, None), 0.0)
            wsum = float(inten.sum())
            if wsum <= 0:
                ys, xs = np.where(sl)
                y, x = float(ys.mean()), float(xs.mean())
            else:
                yy, xx = np.indices(sl.shape)
                y = float((inten * yy).sum() / wsum)
                x = float((inten * xx).sum() / wsum)
            if prev is not None:
                # Mild continuity blend if jump is huge.
                jump = float(np.hypot(y - prev[0], x - prev[1]))
                if jump > 30:
                    y = 0.7 * prev[0] + 0.3 * y
                    x = 0.7 * prev[1] + 0.3 * x
        pts.append((z, y, x))
        prev = (y, x)
    if not pts:
        raise RuntimeError("No centerline points found")
    return np.asarray(pts, dtype=np.float64)


def estimate_cord_axis_pca(
    mask_zyx: np.ndarray,
    spacing_zyx_um: tuple[float, float, float],
    *,
    max_points: int = 400_000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """PCA of mask voxels in physical um → (mean_zyx_um, axis_unit_zyx, meta).

    ``axis`` points toward increasing Z when possible. Meta includes tilt of the
    long axis relative to the Z axis and the Y/X plane angles.
    """
    zz, yy, xx = np.where(mask_zyx)
    if zz.size < 100:
        raise RuntimeError("Mask too empty for PCA axis")
    if zz.size > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(zz.size, size=max_points, replace=False)
        zz, yy, xx = zz[idx], yy[idx], xx[idx]
    sp = np.asarray(spacing_zyx_um, dtype=np.float64)
    pts = np.column_stack([zz * sp[0], yy * sp[1], xx * sp[2]]).astype(np.float64)
    mean = pts.mean(axis=0)
    centered = pts - mean
    # Economy SVD; first right-singular vector = principal axis.
    _, s, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0].copy()
    if axis[0] < 0:
        axis = -axis
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    # Angle between principal axis and +Z (lab frame).
    tilt_from_z_deg = float(np.degrees(np.arccos(np.clip(axis[0], -1.0, 1.0))))
    # Spherical angles of axis: pitch from XY plane, yaw in XY... here Z is long-ish.
    # tilt_y: rotation about X that moves Z toward Y; tilt_x: about Y toward X.
    tilt_about_x_deg = float(np.degrees(np.arctan2(axis[1], axis[0])))  # Z->Y
    tilt_about_y_deg = float(np.degrees(np.arctan2(axis[2], axis[0])))  # Z->X
    var = (s**2) / max(len(pts) - 1, 1)
    meta = {
        "n_points": int(zz.size),
        "mean_zyx_um": mean.tolist(),
        "axis_zyx": axis.tolist(),
        "singular_values": s.tolist(),
        "variance_frac": (var / (var.sum() + 1e-12)).tolist(),
        "tilt_from_z_deg": tilt_from_z_deg,
        "tilt_about_x_deg": tilt_about_x_deg,
        "tilt_about_y_deg": tilt_about_y_deg,
    }
    return mean, axis, meta


def extract_centerline_pca3d(
    mask_zyx: np.ndarray,
    vol_zyx: np.ndarray | None,
    spacing_zyx_um: tuple[float, float, float],
    *,
    step_um: float = 20.0,
    mode: str = "intensity",
    min_vox_per_bin: int = 30,
) -> tuple[np.ndarray, dict]:
    """Centerline by binning mask voxels along the 3D PCA long axis.

    Unlike per-Z centroids, this follows the cord when it is tilted relative to Z,
    so straighten planes are closer to true orthogonal cross-sections.
    Returns Nx3 (z,y,x) voxels and PCA meta.
    """
    if mode not in {"distance", "intensity"}:
        raise ValueError(f"Unknown centerline mode for pca3d: {mode}")
    if mode == "intensity" and vol_zyx is None:
        raise ValueError("intensity mode requires vol_zyx")

    mean_um, axis, pca_meta = estimate_cord_axis_pca(mask_zyx, spacing_zyx_um)
    sp = np.asarray(spacing_zyx_um, dtype=np.float64)
    zz, yy, xx = np.where(mask_zyx)
    pts_um = np.column_stack([zz * sp[0], yy * sp[1], xx * sp[2]]).astype(np.float64)
    t = (pts_um - mean_um) @ axis
    t0, t1 = float(t.min()), float(t.max())
    step = float(max(step_um, 1.0))
    edges = np.arange(t0, t1 + step, step)
    if len(edges) < 2:
        raise RuntimeError("PCA projection span too small for centerline bins")

    inten = None
    if mode == "intensity":
        # Avoid full-volume Gaussian (too slow on LSFM grids); use raw mask voxels.
        inten = np.asarray(vol_zyx, dtype=np.float64)[zz, yy, xx]
        inten = np.clip(inten, 0, None)

    pts_out = []
    for i in range(len(edges) - 1):
        m = (t >= edges[i]) & (t < edges[i + 1])
        n = int(m.sum())
        if n < min_vox_per_bin:
            continue
        if mode == "intensity":
            w = inten[m]
            wsum = float(w.sum())
            if wsum <= 0:
                c_um = pts_um[m].mean(axis=0)
            else:
                c_um = (pts_um[m] * w[:, None]).sum(axis=0) / wsum
        else:
            c_um = pts_um[m].mean(axis=0)
        c_vox = c_um / sp
        pts_out.append((float(c_vox[0]), float(c_vox[1]), float(c_vox[2])))

    if len(pts_out) < 3:
        raise RuntimeError(f"Too few PCA centerline bins ({len(pts_out)})")
    meta = dict(pca_meta)
    meta.update(
        {
            "method": "pca3d",
            "step_um": step,
            "t_span_um": [t0, t1],
            "n_bins_used": len(pts_out),
            "weight": mode,
        }
    )
    logger.info(
        "PCA3D axis tilt_from_z=%.1f° about_x=%.1f° about_y=%.1f° bins=%d",
        meta["tilt_from_z_deg"],
        meta["tilt_about_x_deg"],
        meta["tilt_about_y_deg"],
        len(pts_out),
    )
    return np.asarray(pts_out, dtype=np.float64), meta


def _polyline_tangents_um(pts_um: np.ndarray) -> np.ndarray:
    """Unit tangents along a polyline in physical space (Nx3)."""
    n = len(pts_um)
    out = np.zeros_like(pts_um)
    for i in range(n):
        if i == 0:
            t = pts_um[min(1, n - 1)] - pts_um[0]
        elif i == n - 1:
            t = pts_um[-1] - pts_um[-2]
        else:
            t = pts_um[i + 1] - pts_um[i - 1]
        nrm = float(np.linalg.norm(t))
        out[i] = t / nrm if nrm > 1e-12 else np.array([1.0, 0.0, 0.0])
    # Light smoothing of direction via neighbor blend in cartesian coords then renorm.
    if n >= 3:
        sm = out.copy()
        sm[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        sm /= np.linalg.norm(sm, axis=1, keepdims=True) + 1e-12
        out = sm
    return out


def _local_frame_from_tangent(t_hat: np.ndarray, prev_n: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal (normal, binormal) with parallel transport."""
    t = t_hat / (np.linalg.norm(t_hat) + 1e-12)
    if prev_n is None:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(np.dot(t, ref)) > 0.9:
            ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        n = ref - np.dot(ref, t) * t
        n = n / (np.linalg.norm(n) + 1e-12)
    else:
        n = prev_n - np.dot(prev_n, t) * t
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-8:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            if abs(np.dot(t, ref)) > 0.9:
                ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            n = ref - np.dot(ref, t) * t
            n = n / (np.linalg.norm(n) + 1e-12)
        else:
            n = n / n_norm
    b = np.cross(t, n)
    b = b / (np.linalg.norm(b) + 1e-12)
    return n, b


def _resample_centerline_uniform_zyx(
    pts_zyx: np.ndarray,
    spacing_zyx_um: tuple[float, float, float],
    step_um: float,
) -> np.ndarray:
    sp = np.asarray(spacing_zyx_um, dtype=np.float64)
    pts_um = pts_zyx * sp
    d = np.linalg.norm(np.diff(pts_um, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = float(s[-1])
    if total <= step_um:
        return pts_zyx.copy()
    s_new = np.arange(0.0, total + 0.5 * step_um, step_um)
    return np.column_stack([np.interp(s_new, s, pts_zyx[:, i]) for i in range(3)])


def extract_centerline_local_ortho(
    mask_zyx: np.ndarray,
    vol_zyx: np.ndarray | None,
    spacing_zyx_um: tuple[float, float, float],
    *,
    init_pts_zyx: np.ndarray,
    step_um: float = 20.0,
    radius_um: float = 800.0,
    plane_pitch_um: float = 10.0,
    n_iters: int = 2,
    section_mode: str = "distance",
    max_shift_um: float | None = None,
    max_drift_from_init_um: float | None = None,
    prior_blend: float = 0.25,
    smooth_window: int = 21,
) -> tuple[np.ndarray, dict]:
    """Refine centerline on planes orthogonal to the local travel direction.

    1) Resample ``init_pts_zyx`` to uniform arc length.
    2) Estimate local unit tangents.
    3) On each orthogonal disk, find mask center (EDT peak or intensity centroid).
    4) Clamp per-iter jumps and total drift from init; blend toward init; smooth; repeat.
    """
    from scipy import ndimage
    from scipy.ndimage import map_coordinates

    if section_mode not in {"distance", "intensity"}:
        raise ValueError(f"Unknown section_mode: {section_mode}")
    if section_mode == "intensity" and vol_zyx is None:
        raise ValueError("intensity section_mode requires vol_zyx")

    sp = np.asarray(spacing_zyx_um, dtype=np.float64)
    mask_f = mask_zyx.astype(np.float32)
    vol_f = None if vol_zyx is None else np.asarray(vol_zyx, dtype=np.float32)

    r_vox = int(max(4, round(radius_um / max(plane_pitch_um, 1e-6))))
    size = 2 * r_vox
    yy, xx = np.mgrid[-r_vox:r_vox, -r_vox:r_vox]
    off_y_um = yy.astype(np.float64) * plane_pitch_um
    off_x_um = xx.astype(np.float64) * plane_pitch_um
    disk = (yy.astype(np.float64) ** 2 + xx.astype(np.float64) ** 2) <= (r_vox - 0.5) ** 2
    # Default: do not allow a single iter to jump more than ~35% of search radius.
    if max_shift_um is None:
        max_shift_um = float(0.35 * radius_um)
    max_shift_um = float(max(1.0, max_shift_um))
    if max_drift_from_init_um is None:
        max_drift_from_init_um = float(max(max_shift_um * 1.5, 0.5 * radius_um))
    max_drift_from_init_um = float(max(1.0, max_drift_from_init_um))
    prior_blend = float(np.clip(prior_blend, 0.0, 0.95))

    cl0 = _resample_centerline_uniform_zyx(init_pts_zyx, spacing_zyx_um, step_um)
    cl = cl0.copy()
    shift_stats = []
    n_clamped_total = 0
    n_drift_clamped_total = 0

    for it in range(max(1, int(n_iters))):
        pts_um = cl * sp
        tangents = _polyline_tangents_um(pts_um)
        # Light tangent smoothing reduces frame flip / zigzag on bent cords.
        if len(tangents) >= 5:
            from scipy.ndimage import uniform_filter1d

            tangents = uniform_filter1d(tangents, size=5, axis=0, mode="nearest")
            tangents = tangents / (np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12)

        new_cl = np.zeros_like(cl)
        prev_n = None
        shifts = []
        n_clamped = 0
        n_drift = 0
        for i in range(len(cl)):
            t_hat = tangents[i]
            n_hat, b_hat = _local_frame_from_tangent(t_hat, prev_n)
            prev_n = n_hat
            c_um = pts_um[i]
            oz = (c_um[0] + n_hat[0] * off_y_um + b_hat[0] * off_x_um) / sp[0]
            oy = (c_um[1] + n_hat[1] * off_y_um + b_hat[1] * off_x_um) / sp[1]
            ox = (c_um[2] + n_hat[2] * off_y_um + b_hat[2] * off_x_um) / sp[2]
            coords = np.stack([oz.ravel(), oy.ravel(), ox.ravel()], axis=0)
            sec_m = map_coordinates(mask_f, coords, order=0, mode="constant", cval=0.0).reshape(size, size)
            sec_m = (sec_m > 0.5) & disk
            if not np.any(sec_m):
                new_cl[i] = cl[i]
                shifts.append(0.0)
                continue

            if section_mode == "distance":
                dist = ndimage.distance_transform_edt(sec_m)
                thr = float(dist.max()) * 0.9
                ys, xs = np.where(dist >= thr)
                py = float(ys.mean())
                px = float(xs.mean())
            else:
                sec_v = map_coordinates(vol_f, coords, order=1, mode="constant", cval=0.0).reshape(size, size)
                w = np.where(sec_m, np.clip(sec_v, 0, None), 0.0)
                wsum = float(w.sum())
                if wsum <= 0:
                    ys, xs = np.where(sec_m)
                    py, px = float(ys.mean()), float(xs.mean())
                else:
                    py = float((w * yy).sum() / wsum)
                    px = float((w * xx).sum() / wsum)

            dy_um = float(py - r_vox) * plane_pitch_um
            dx_um = float(px - r_vox) * plane_pitch_um
            delta_um = n_hat * dy_um + b_hat * dx_um
            shift = float(np.linalg.norm(delta_um))
            if shift > max_shift_um:
                delta_um = delta_um * (max_shift_um / shift)
                shift = max_shift_um
                n_clamped += 1
            c_new_um = c_um + delta_um
            # Soft prior toward init + hard cap on total drift from init.
            c0_um = cl0[i] * sp
            if prior_blend > 0:
                c_new_um = (1.0 - prior_blend) * c_new_um + prior_blend * c0_um
            drift = c_new_um - c0_um
            drift_n = float(np.linalg.norm(drift))
            if drift_n > max_drift_from_init_um:
                c_new_um = c0_um + drift * (max_drift_from_init_um / drift_n)
                n_drift += 1
            new_cl[i] = c_new_um / sp
            shifts.append(float(np.linalg.norm((new_cl[i] - cl[i]) * sp)))

            if (i + 1) % 400 == 0 or i + 1 == len(cl):
                logger.info(
                    "  local_ortho iter %d/%d plane %d/%d shift=%.1f um",
                    it + 1,
                    n_iters,
                    i + 1,
                    len(cl),
                    shifts[-1],
                )

        # Stabilize between iters: reject spikes then light moving-average.
        new_cl = reject_centerline_jumps(new_cl, max_jump_vox=max(8.0, max_shift_um / float(np.mean(sp[1:]) + 1e-6)))
        if smooth_window and smooth_window > 1:
            new_cl = smooth_centerline(new_cl, window=max(5, int(smooth_window) | 1))

        shift_stats.append(
            {
                "iter": it + 1,
                "shift_um_mean": float(np.mean(shifts)),
                "shift_um_p95": float(np.percentile(shifts, 95)),
                "shift_um_max": float(np.max(shifts)),
                "n_clamped": int(n_clamped),
                "n_drift_clamped": int(n_drift),
                "max_shift_um": float(max_shift_um),
                "max_drift_from_init_um": float(max_drift_from_init_um),
            }
        )
        n_clamped_total += n_clamped
        n_drift_clamped_total += n_drift
        cl = new_cl
        logger.info(
            "local_ortho iter %d done: mean shift=%.1f um p95=%.1f um clamped=%d drift_cap=%d",
            it + 1,
            shift_stats[-1]["shift_um_mean"],
            shift_stats[-1]["shift_um_p95"],
            n_clamped,
            n_drift,
        )

    meta = {
        "method": "local_ortho",
        "step_um": float(step_um),
        "radius_um": float(radius_um),
        "plane_pitch_um": float(plane_pitch_um),
        "n_iters": int(n_iters),
        "section_mode": section_mode,
        "max_shift_um": float(max_shift_um),
        "max_drift_from_init_um": float(max_drift_from_init_um),
        "prior_blend": float(prior_blend),
        "smooth_window": int(smooth_window),
        "n_points": int(len(cl)),
        "n_clamped_total": int(n_clamped_total),
        "n_drift_clamped_total": int(n_drift_clamped_total),
        "iters": shift_stats,
    }
    return cl.astype(np.float64), meta


def refine_centerline_where_oblique(
    mask_zyx: np.ndarray,
    vol_zyx: np.ndarray | None,
    spacing_zyx_um: tuple[float, float, float],
    pts_zyx: np.ndarray,
    *,
    tilt_deg_thresh: float = 25.0,
    radius_um: float = 900.0,
    plane_pitch_um: float = 10.0,
    section_mode: str = "distance",
    dilate_n: int = 5,
    n_iters: int = 1,
) -> tuple[np.ndarray, dict]:
    """Keep per-Z centerline; only re-estimate points where local tangent is oblique to Z.

    Obliqueness: angle between local tangent and +Z exceeds ``tilt_deg_thresh``.
    Those segments (plus ``dilate_n`` neighbors) are refined on planes orthogonal
    to the local tangent (EDT peak or intensity centroid).
    """
    from scipy import ndimage
    from scipy.ndimage import map_coordinates

    if section_mode not in {"distance", "intensity"}:
        raise ValueError(f"Unknown section_mode: {section_mode}")
    if section_mode == "intensity" and vol_zyx is None:
        raise ValueError("intensity section_mode requires vol_zyx")

    sp = np.asarray(spacing_zyx_um, dtype=np.float64)
    cl = np.asarray(pts_zyx, dtype=np.float64).copy()
    if len(cl) < 3:
        return cl, {"method": "oblique_refine", "skipped": True, "reason": "too_few_points"}

    mask_f = mask_zyx.astype(np.float32)
    vol_f = None if vol_zyx is None else np.asarray(vol_zyx, dtype=np.float32)
    r_vox = int(max(4, round(radius_um / max(plane_pitch_um, 1e-6))))
    size = 2 * r_vox
    yy, xx = np.mgrid[-r_vox:r_vox, -r_vox:r_vox]
    off_y_um = yy.astype(np.float64) * plane_pitch_um
    off_x_um = xx.astype(np.float64) * plane_pitch_um
    disk = (yy.astype(np.float64) ** 2 + xx.astype(np.float64) ** 2) <= (r_vox - 0.5) ** 2

    def _tilt_deg(pts: np.ndarray) -> np.ndarray:
        tangents = _polyline_tangents_um(pts * sp)
        # Angle between tangent and +Z axis in ZYX physical space.
        return np.degrees(np.arccos(np.clip(np.abs(tangents[:, 0]), 0.0, 1.0)))

    tilt0 = _tilt_deg(cl)
    refine = tilt0 >= float(tilt_deg_thresh)
    if dilate_n > 0 and np.any(refine):
        refine = ndimage.binary_dilation(refine, iterations=int(dilate_n))

    n_seed = int(np.sum(tilt0 >= float(tilt_deg_thresh)))
    n_refine = int(np.sum(refine))
    if n_refine == 0:
        meta = {
            "method": "oblique_refine",
            "tilt_deg_thresh": float(tilt_deg_thresh),
            "n_seed_oblique": 0,
            "n_refined": 0,
            "tilt_deg_p50": float(np.median(tilt0)),
            "tilt_deg_p95": float(np.percentile(tilt0, 95)),
            "tilt_deg_max": float(np.max(tilt0)),
            "skipped": True,
            "reason": "no_oblique_points",
        }
        logger.info(
            "oblique_refine: no points above %.1f° (max tilt=%.1f°)",
            tilt_deg_thresh,
            float(np.max(tilt0)),
        )
        return cl, meta

    logger.info(
        "oblique_refine: seed=%d dilated=%d / %d (thresh=%.1f°, tilt p95=%.1f max=%.1f)",
        n_seed,
        n_refine,
        len(cl),
        tilt_deg_thresh,
        float(np.percentile(tilt0, 95)),
        float(np.max(tilt0)),
    )

    shift_stats = []
    for it in range(max(1, int(n_iters))):
        pts_um = cl * sp
        tangents = _polyline_tangents_um(pts_um)
        prev_n = None
        # Precompute frames for all points so parallel transport stays continuous.
        normals = np.zeros_like(pts_um)
        binormals = np.zeros_like(pts_um)
        for i in range(len(cl)):
            n_hat, b_hat = _local_frame_from_tangent(tangents[i], prev_n)
            normals[i], binormals[i] = n_hat, b_hat
            prev_n = n_hat

        shifts = []
        new_cl = cl.copy()
        idxs = np.where(refine)[0]
        for k, i in enumerate(idxs):
            t_hat = tangents[i]
            n_hat, b_hat = normals[i], binormals[i]
            c_um = pts_um[i]
            oz = (c_um[0] + n_hat[0] * off_y_um + b_hat[0] * off_x_um) / sp[0]
            oy = (c_um[1] + n_hat[1] * off_y_um + b_hat[1] * off_x_um) / sp[1]
            ox = (c_um[2] + n_hat[2] * off_y_um + b_hat[2] * off_x_um) / sp[2]
            coords = np.stack([oz.ravel(), oy.ravel(), ox.ravel()], axis=0)
            sec_m = map_coordinates(mask_f, coords, order=0, mode="constant", cval=0.0).reshape(size, size)
            sec_m = (sec_m > 0.5) & disk
            if not np.any(sec_m):
                shifts.append(0.0)
                continue

            if section_mode == "distance":
                dist = ndimage.distance_transform_edt(sec_m)
                thr = float(dist.max()) * 0.9
                ys, xs = np.where(dist >= thr)
                py, px = float(ys.mean()), float(xs.mean())
            else:
                sec_v = map_coordinates(vol_f, coords, order=1, mode="constant", cval=0.0).reshape(size, size)
                w = np.where(sec_m, np.clip(sec_v, 0, None), 0.0)
                wsum = float(w.sum())
                if wsum <= 0:
                    ys, xs = np.where(sec_m)
                    py, px = float(ys.mean()), float(xs.mean())
                else:
                    py = float((w * yy).sum() / wsum)
                    px = float((w * xx).sum() / wsum)

            dy_um = float(py - r_vox) * plane_pitch_um
            dx_um = float(px - r_vox) * plane_pitch_um
            c_new_um = c_um + n_hat * dy_um + b_hat * dx_um
            # Blend toward new center; keep some continuity with original per-Z point.
            blend = 0.85
            c_blend_um = (1.0 - blend) * c_um + blend * c_new_um
            new_cl[i] = c_blend_um / sp
            shifts.append(float(np.linalg.norm(c_blend_um - c_um)))
            if (k + 1) % 200 == 0 or k + 1 == len(idxs):
                logger.info(
                    "  oblique_refine iter %d/%d %d/%d shift=%.1f um tilt0=%.1f",
                    it + 1,
                    n_iters,
                    k + 1,
                    len(idxs),
                    shifts[-1],
                    float(tilt0[i]),
                )

        cl = new_cl
        shift_stats.append(
            {
                "iter": it + 1,
                "n_updated": int(len(idxs)),
                "shift_um_mean": float(np.mean(shifts)) if shifts else 0.0,
                "shift_um_p95": float(np.percentile(shifts, 95)) if shifts else 0.0,
                "shift_um_max": float(np.max(shifts)) if shifts else 0.0,
            }
        )
        logger.info(
            "oblique_refine iter %d done: mean shift=%.1f um p95=%.1f um",
            it + 1,
            shift_stats[-1]["shift_um_mean"],
            shift_stats[-1]["shift_um_p95"],
        )

    tilt1 = _tilt_deg(cl)
    meta = {
        "method": "oblique_refine",
        "tilt_deg_thresh": float(tilt_deg_thresh),
        "radius_um": float(radius_um),
        "plane_pitch_um": float(plane_pitch_um),
        "section_mode": section_mode,
        "dilate_n": int(dilate_n),
        "n_iters": int(n_iters),
        "n_seed_oblique": n_seed,
        "n_refined": n_refine,
        "n_points": int(len(cl)),
        "tilt_deg_before": {
            "p50": float(np.median(tilt0)),
            "p95": float(np.percentile(tilt0, 95)),
            "max": float(np.max(tilt0)),
        },
        "tilt_deg_after": {
            "p50": float(np.median(tilt1)),
            "p95": float(np.percentile(tilt1, 95)),
            "max": float(np.max(tilt1)),
        },
        "iters": shift_stats,
        "skipped": False,
    }
    return cl, meta


def slice_mask_incomplete(mask2d: np.ndarray, border_margin: int = 2) -> bool:
    """True if mask is empty or touches (within margin) the 2D FOV border."""
    if not mask2d.any():
        return True
    m = int(max(border_margin, 0))
    h, w = mask2d.shape
    if m == 0:
        return bool(
            mask2d[0, :].any()
            or mask2d[-1, :].any()
            or mask2d[:, 0].any()
            or mask2d[:, -1].any()
        )
    top = mask2d[: m + 1, :].any()
    bot = mask2d[-(m + 1) :, :].any()
    left = mask2d[:, : m + 1].any()
    right = mask2d[:, -(m + 1) :].any()
    return bool(top or bot or left or right)


def mark_unreliable_z(
    mask_zyx: np.ndarray,
    *,
    border_margin: int = 2,
    end_frac: float = 0.01,
    border_check_frac: float = 0.05,
) -> np.ndarray:
    """Boolean length Z: True = incomplete end slice, do not trust centerline.

    Strategy (matches ~1% head/tail incomplete FOV):
    - Always mark first/last ``end_frac`` of the occupied RC span.
    - Within first/last ``border_check_frac``, also mark slices whose mask
      touches the transverse FOV border (partial cross-section).
    - Empty slices are always unreliable.
    """
    z_n = mask_zyx.shape[0]
    unreliable = np.zeros(z_n, dtype=bool)
    occupied = [z for z in range(z_n) if mask_zyx[z].any()]
    for z in range(z_n):
        if not mask_zyx[z].any():
            unreliable[z] = True

    if not occupied:
        return unreliable

    z0, z1 = occupied[0], occupied[-1]
    span = max(z1 - z0 + 1, 1)
    n_end = max(1, int(round(span * max(end_frac, 0.0))))
    n_border = max(n_end, int(round(span * max(border_check_frac, end_frac))))

    for z in range(z0, min(z0 + n_end, z1 + 1)):
        unreliable[z] = True
    for z in range(max(z0, z1 - n_end + 1), z1 + 1):
        unreliable[z] = True

    head = range(z0, min(z0 + n_border, z1 + 1))
    tail = range(max(z0, z1 - n_border + 1), z1 + 1)
    for z in list(head) + list(tail):
        if slice_mask_incomplete(mask_zyx[z], border_margin=border_margin):
            unreliable[z] = True

    return unreliable


def interpolate_unreliable_centerline(
    pts: np.ndarray,
    unreliable_z: np.ndarray,
    *,
    drop_end_unreliable: bool = True,
) -> tuple[np.ndarray, dict]:
    """Fix centerline on unreliable Z.

    - Interior unreliable points: linear interpolate from reliable neighbors.
    - Leading/trailing unreliable ends: drop (do not hold-constant extrapolate,
      which creates fake straight segments when the cord curves).
    """
    if len(pts) < 2:
        return pts.copy(), {"n_unreliable_used": 0, "n_reliable": len(pts)}

    z_idx = np.clip(pts[:, 0].astype(int), 0, len(unreliable_z) - 1)
    bad = unreliable_z[z_idx]
    good = ~bad
    n_good = int(good.sum())
    if n_good < 2:
        logger.warning("Too few reliable centerline points (%d); skip incomplete-slice interp", n_good)
        return pts.copy(), {"n_unreliable_used": 0, "n_reliable": n_good, "skipped": True}

    out = pts.copy()
    zg = pts[good, 0]
    for col in (1, 2):
        out[:, col] = np.interp(pts[:, 0], zg, pts[good, col])

    n_interp = int(bad.sum())
    if drop_end_unreliable:
        # Keep only from first to last reliable sample (crop unreliable tips).
        first_g = int(np.argmax(good))
        last_g = int(len(good) - 1 - np.argmax(good[::-1]))
        out = out[first_g : last_g + 1]
        n_dropped_ends = int(first_g + (len(good) - 1 - last_g))
    else:
        n_dropped_ends = 0

    meta = {
        "n_points_in": int(len(pts)),
        "n_points_out": int(len(out)),
        "n_reliable": n_good,
        "n_unreliable_interpolated": n_interp,
        "n_end_points_dropped": n_dropped_ends,
        "unreliable_frac_in": float(bad.mean()),
        "skipped": False,
        "drop_end_unreliable": drop_end_unreliable,
    }
    return out, meta


def extend_centerline_polynomial(
    reliable_pts: np.ndarray,
    *,
    z_min: float,
    z_max: float,
    degree: int = 2,
    local_end_frac: float = 0.15,
    max_extend_vox: float | None = 60.0,
    yx_clip: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Extend reliable centerline to [z_min, z_max] with polynomial ends.

    Interior keeps reliable samples. Each end is extrapolated with a low-order
    polynomial fit on the nearest ``local_end_frac`` of reliable points.
    ``max_extend_vox`` caps how far beyond the reliable span we go (avoids
    wild curves into incomplete hook/end slices).
    """
    if len(reliable_pts) < 3:
        raise RuntimeError("Need >=3 reliable centerline points for polynomial fit")

    z_r = reliable_pts[:, 0]
    y_r = reliable_pts[:, 1]
    x_r = reliable_pts[:, 2]
    z_lo, z_hi = float(z_r.min()), float(z_r.max())
    if max_extend_vox is not None and max_extend_vox >= 0:
        z_min = max(float(z_min), z_lo - float(max_extend_vox))
        z_max = min(float(z_max), z_hi + float(max_extend_vox))
    z0 = int(np.floor(min(z_min, z_lo)))
    z1 = int(np.ceil(max(z_max, z_hi)))
    z_all = np.arange(z0, z1 + 1, dtype=np.float64)

    n_local = max(degree + 2, int(round(len(reliable_pts) * local_end_frac)))
    n_local = min(n_local, len(reliable_pts))
    deg = int(max(1, min(degree, n_local - 1)))

    head = reliable_pts[:n_local]
    tail = reliable_pts[-n_local:]
    coef_y_h = np.polyfit(head[:, 0], head[:, 1], deg)
    coef_x_h = np.polyfit(head[:, 0], head[:, 2], deg)
    coef_y_t = np.polyfit(tail[:, 0], tail[:, 1], deg)
    coef_x_t = np.polyfit(tail[:, 0], tail[:, 2], deg)

    y_out = np.zeros_like(z_all)
    x_out = np.zeros_like(z_all)
    extrapolated = np.zeros(len(z_all), dtype=bool)

    rel_map = {int(np.rint(z)): (y, x) for z, y, x in reliable_pts}
    for i, z in enumerate(z_all):
        zi = int(np.rint(z))
        if zi in rel_map:
            y_out[i], x_out[i] = rel_map[zi]
            continue
        if z < z_lo:
            y_out[i] = float(np.polyval(coef_y_h, z))
            x_out[i] = float(np.polyval(coef_x_h, z))
            extrapolated[i] = True
        elif z > z_hi:
            y_out[i] = float(np.polyval(coef_y_t, z))
            x_out[i] = float(np.polyval(coef_x_t, z))
            extrapolated[i] = True
        else:
            y_out[i] = float(np.interp(z, z_r, y_r))
            x_out[i] = float(np.interp(z, z_r, x_r))
            extrapolated[i] = True

    if yx_clip is not None:
        (y_min, y_max), (x_min, x_max) = yx_clip
        y_out = np.clip(y_out, y_min, y_max)
        x_out = np.clip(x_out, x_min, x_max)

    blend = 12
    for boundary in (int(np.rint(z_lo)), int(np.rint(z_hi))):
        for i, z in enumerate(z_all):
            if abs(z - boundary) <= blend and extrapolated[i]:
                if boundary in rel_map:
                    yr, xr = rel_map[boundary]
                    w = 1.0 - abs(z - boundary) / float(blend)
                    w = float(np.clip(w, 0.0, 1.0))
                    y_out[i] = w * yr + (1.0 - w) * y_out[i]
                    x_out[i] = w * xr + (1.0 - w) * x_out[i]

    out = np.column_stack([z_all, y_out, x_out])
    meta = {
        "degree": deg,
        "local_end_frac": local_end_frac,
        "n_local_fit": n_local,
        "max_extend_vox": None if max_extend_vox is None else float(max_extend_vox),
        "z_min": z0,
        "z_max": z1,
        "n_reliable_fit": int(len(reliable_pts)),
        "n_out": int(len(out)),
        "n_extrapolated": int(extrapolated.sum()),
        "reliable_z_span": [z_lo, z_hi],
        "yx_clip": yx_clip,
    }
    return out, meta, extrapolated


def reject_centerline_jumps(pts: np.ndarray, max_jump_vox: float = 20.0) -> np.ndarray:
    """Median-filter y/x, then clamp outliers relative to filtered curve."""
    if len(pts) < 5:
        return pts.copy()
    out = pts.copy()
    w = 11
    for col in (1, 2):
        from scipy.ndimage import median_filter

        med = median_filter(pts[:, col], size=w, mode="nearest")
        jump = np.abs(pts[:, col] - med)
        bad = jump > max_jump_vox
        fixed = pts[:, col].copy()
        fixed[bad] = med[bad]
        out[:, col] = fixed
    return out


def smooth_centerline(pts: np.ndarray, window: int = 21) -> np.ndarray:
    """Moving-average smooth on y/x; keep z unchanged."""
    if pts.shape[0] < 3:
        return pts.copy()
    w = max(3, window | 1)
    kernel = np.ones(w, dtype=np.float64) / w
    out = pts.copy()
    for col in (1, 2):
        pad = w // 2
        x = np.pad(pts[:, col], (pad, pad), mode="edge")
        out[:, col] = np.convolve(x, kernel, mode="valid")
    return out


def smooth_centerline_spline_yx(
    pts: np.ndarray,
    spacing_zyx_um: tuple[float, float, float],
    *,
    smooth_um: float = 50.0,
) -> np.ndarray:
    """Spline-smooth Y/X along arc length; keep Z samples.

    ``smooth_um`` is an approximate allowed RMS lateral deviation (microns).
    Large-scale bends are preserved; only high-frequency jitter is removed.
    """
    from scipy.interpolate import UnivariateSpline

    if pts.shape[0] < 8 or smooth_um <= 0:
        return pts.copy()
    sp = np.asarray(spacing_zyx_um, dtype=np.float64)
    d = np.linalg.norm(np.diff(pts * sp, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    keep = np.concatenate([[True], np.diff(s) > 1e-6])
    s_k = s[keep]
    pts_k = pts[keep]
    if len(s_k) < 8:
        return pts.copy()
    n = len(s_k)
    # Convert micron RMS → voxel RMS on Y/X, then s ≈ n * sigma^2 (UnivariateSpline).
    pitch_yx = float(0.5 * (sp[1] + sp[2]))
    sigma_vox = max(float(smooth_um) / max(pitch_yx, 1e-6), 0.25)
    s_factor = float(n * sigma_vox ** 2)
    out = pts_k.copy()
    for col in (1, 2):
        spl = UnivariateSpline(s_k, pts_k[:, col], k=3, s=s_factor)
        out[:, col] = spl(s_k)
    if len(out) == len(pts):
        return out
    y = np.interp(s, s_k, out[:, 1])
    x = np.interp(s, s_k, out[:, 2])
    return np.column_stack([pts[:, 0], y, x])


def _draw_cross(rgb: np.ndarray, y: int, x: int, color, r: int = 3) -> None:
    h, w = rgb.shape[:2]
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    rgb[y, x0:x1] = color
    rgb[y0:y1, x] = color


def write_qc(
    out_dir: Path,
    vol_zyx: np.ndarray,
    mask_zyx: np.ndarray,
    centerline: np.ndarray,
    spacing_zyx_um: tuple[float, float, float],
    unreliable_z: np.ndarray | None = None,
    centerline_before: np.ndarray | None = None,
) -> None:
    qc = out_dir / "qc"
    qc.mkdir(parents=True, exist_ok=True)

    # Mid transverse: intensity + mask outline + centerline point
    zmid = int(vol_zyx.shape[0] // 2)
    base = cv2.cvtColor(_to_u8(vol_zyx[zmid]), cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask_zyx[zmid].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(base, contours, -1, (0, 255, 0), 1)
    if len(centerline):
        i = int(np.argmin(np.abs(centerline[:, 0] - zmid)))
        _draw_cross(base, int(round(centerline[i, 1])), int(round(centerline[i, 2])), (0, 0, 255), r=4)
    cv2.imwrite(str(qc / "transverse_zmid_mask_centerline.png"), base)

    for frac, name in [(0.25, "z25"), (0.50, "z50"), (0.75, "z75")]:
        z = int(vol_zyx.shape[0] * frac)
        z = min(max(z, 0), vol_zyx.shape[0] - 1)
        img = cv2.cvtColor(_to_u8(vol_zyx[z]), cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(mask_zyx[z].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
        if len(centerline):
            i = int(np.argmin(np.abs(centerline[:, 0] - z)))
            _draw_cross(img, int(round(centerline[i, 1])), int(round(centerline[i, 2])), (0, 0, 255), r=4)
        cv2.imwrite(str(qc / f"transverse_{name}_mask_centerline.png"), img)

    def _draw_mip_centerline(mip: np.ndarray, axis_for_col: int, path: Path) -> None:
        """axis_for_col: 1=Y for sagittal (max X), 2=X for coronal (max Y)."""
        rgb = cv2.cvtColor(_to_u8(mip), cv2.COLOR_GRAY2BGR)
        if unreliable_z is not None:
            for z in range(min(len(unreliable_z), rgb.shape[0])):
                if unreliable_z[z]:
                    rgb[z, :3] = (255, 128, 0)  # orange strip = unreliable
        # Final centerline as thick cyan polyline (easy to see vs tissue bright core).
        pts_draw = []
        for z, y, x in centerline:
            zz = int(round(z))
            cc = int(round(y if axis_for_col == 1 else x))
            if 0 <= zz < rgb.shape[0] and 0 <= cc < rgb.shape[1]:
                pts_draw.append((cc, zz))
        if len(pts_draw) >= 2:
            cv2.polylines(rgb, [np.asarray(pts_draw, dtype=np.int32)], False, (255, 255, 0), 2, cv2.LINE_AA)
        # Optional: raw/before in thin magenta for comparison.
        if centerline_before is not None:
            pts_b = []
            for z, y, x in centerline_before:
                zz = int(round(z))
                cc = int(round(y if axis_for_col == 1 else x))
                if 0 <= zz < rgb.shape[0] and 0 <= cc < rgb.shape[1]:
                    pts_b.append((cc, zz))
            if len(pts_b) >= 2:
                cv2.polylines(rgb, [np.asarray(pts_b, dtype=np.int32)], False, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(path), rgb)

    _draw_mip_centerline(vol_zyx.max(axis=2), 1, qc / "sagittal_mip_centerline.png")
    _draw_mip_centerline(vol_zyx.max(axis=1), 2, qc / "coronal_mip_centerline.png")

    # Mask-only mid slices
    tifffile.imwrite(qc / "mask_zmid.tif", (mask_zyx[zmid].astype(np.uint8) * 255))
    tifffile.imwrite(qc / "intensity_zmid.tif", _to_u8(vol_zyx[zmid]))

    zs, ys, xs = np.where(mask_zyx)
    sz, sy, sx = spacing_zyx_um
    bbox_mm = {
        "z_mm": (float(zs.max() - zs.min() + 1) * sz) / 1000.0,
        "y_mm": (float(ys.max() - ys.min() + 1) * sy) / 1000.0,
        "x_mm": (float(xs.max() - xs.min() + 1) * sx) / 1000.0,
    }
    n_unrel = int(unreliable_z.sum()) if unreliable_z is not None else 0
    (qc / "README.txt").write_text(
        "\n".join(
            [
                "Green contour = tissue mask outline",
                "Cyan thick = final centerline (continuous)",
                "Magenta thin = raw centerline before end fix (comparison)",
                "Orange strip on MIP left edge = unreliable Z (border-touch or end_frac)",
                "Note: bright white ridge inside tissue is image signal, not the centerline overlay",
                "transverse_* = cross-section (atlas-like)",
                f"mask bbox approx mm (Z,Y,X / RC,DV,ML): {bbox_mm}",
                f"unreliable_z_count: {n_unrel}",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--volume_nii", required=True, help="Downsampled sample volume.nii.gz (XYZ)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--thr_frac", type=float, default=0.10, help="Threshold as fraction of p99")
    p.add_argument("--smooth_window", type=int, default=31, help="Centerline moving-average window (odd)")
    p.add_argument("--no_filter_2d_cc", action="store_true", help="Disable per-slice small-CC removal")
    p.add_argument("--open_radius_2d", type=int, default=8, help="2D opening radius to cut thin bridges before CC keep-largest")
    p.add_argument("--close_radius_2d", type=int, default=4, help="2D closing radius after keep-largest")
    p.add_argument(
        "--centerline_mode",
        choices=("intensity", "distance", "pca3d", "local_ortho"),
        default="intensity",
        help="intensity/distance: per-Z; pca3d: global PCA bins; local_ortho: orthogonal to local tangent",
    )
    p.add_argument(
        "--pca_step_um",
        type=float,
        default=20.0,
        help="For pca3d/local_ortho: arc/bin step along cord in microns (default 20)",
    )
    p.add_argument(
        "--ortho_radius_um",
        type=float,
        default=800.0,
        help="For local_ortho: orthogonal disk radius in microns",
    )
    p.add_argument(
        "--ortho_iters",
        type=int,
        default=2,
        help="For local_ortho: refinement iterations",
    )
    p.add_argument(
        "--ortho_max_shift_um",
        type=float,
        default=80.0,
        help="For local_ortho: clamp per-plane lateral shift per iter (um; default 80)",
    )
    p.add_argument(
        "--ortho_max_drift_um",
        type=float,
        default=150.0,
        help="For local_ortho: max total lateral drift from coarse init (um; default 150)",
    )
    p.add_argument(
        "--ortho_prior_blend",
        type=float,
        default=0.35,
        help="For local_ortho: blend refined point toward coarse init (0-1; default 0.35)",
    )
    p.add_argument(
        "--ortho_section_mode",
        choices=("distance", "intensity"),
        default="distance",
        help="For local_ortho / oblique refine: EDT peak or intensity centroid",
    )
    p.add_argument(
        "--final_spline_smooth_um",
        type=float,
        default=35.0,
        help="Final Y/X spline smooth along arc (um RMS-ish; 0 disables). Reduces zigzag before straighten.",
    )
    p.add_argument(
        "--refine_oblique_deg",
        type=float,
        default=0.0,
        help="If >0, after per-Z/pca3d centerline only refine points whose local tangent "
        "vs Z exceeds this angle (degrees). Example: 25",
    )
    p.add_argument(
        "--refine_oblique_dilate",
        type=int,
        default=5,
        help="Also refine this many neighbors around oblique seeds",
    )
    p.add_argument(
        "--max_centerline_jump",
        type=float,
        default=15.0,
        help="Clamp centerline deviations from local median larger than this (voxels)",
    )
    p.add_argument(
        "--border_margin",
        type=int,
        default=2,
        help="Mask within this many voxels of FOV border => incomplete slice",
    )
    p.add_argument(
        "--end_frac",
        type=float,
        default=0.05,
        help="Also treat first/last this fraction of occupied RC as unreliable (default 5%)",
    )
    p.add_argument(
        "--no_incomplete_interp",
        action="store_true",
        help="Disable incomplete-slice centerline interpolation",
    )
    p.add_argument(
        "--no_poly_extend",
        action="store_true",
        help="Disable polynomial end extension (keep cropped reliable centerline only)",
    )
    p.add_argument(
        "--poly_degree",
        type=int,
        default=2,
        help="Polynomial degree for end extrapolation (default 2; lower = safer)",
    )
    p.add_argument(
        "--poly_local_end_frac",
        type=float,
        default=0.12,
        help="Fraction of reliable points used to fit each end (default 0.12)",
    )
    p.add_argument(
        "--poly_max_extend_vox",
        type=float,
        default=40.0,
        help="Max Z voxels to extrapolate beyond reliable span (-1 = unlimited)",
    )
    args = p.parse_args()

    import nibabel as nib

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nii = nib.load(str(args.volume_nii))
    vol_xyz = np.asanyarray(nii.dataobj).astype(np.float32)
    spacing_xyz = tuple(float(abs(nii.affine[i, i])) for i in range(3))
    vol_zyx = np.transpose(vol_xyz, (2, 1, 0))
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])

    logger.info("Volume ZYX=%s spacing_zyx_um=%s", vol_zyx.shape, spacing_zyx)
    mask_zyx, mask_meta = make_tissue_mask_zyx(
        vol_zyx,
        thr_frac_of_p99=args.thr_frac,
        filter_2d_cc=not args.no_filter_2d_cc,
        keep_largest_2d_only=True,
        open_radius_2d=args.open_radius_2d,
        close_radius_2d=args.close_radius_2d,
    )
    logger.info("Mask meta: %s", mask_meta)

    unreliable_z = mark_unreliable_z(
        mask_zyx,
        border_margin=args.border_margin,
        end_frac=0.0 if args.no_incomplete_interp else args.end_frac,
    )

    pca_meta = None
    ortho_meta = None
    if args.centerline_mode == "pca3d":
        raw_cl, pca_meta = extract_centerline_pca3d(
            mask_zyx,
            vol_zyx,
            spacing_zyx,
            step_um=args.pca_step_um,
            mode="intensity",
        )
        interp_cl, interp_meta = raw_cl, {"skipped": True, "reason": "pca3d"}
        before_cl = None
    elif args.centerline_mode == "local_ortho":
        # Coarse per-Z init, then refine on planes orthogonal to local tangent.
        coarse = extract_centerline_zyx(mask_zyx, vol_zyx, mode="intensity")
        if args.no_incomplete_interp:
            coarse_use = coarse
            interp_meta = {"skipped": True, "reason": "local_ortho_init"}
        else:
            coarse_use, interp_meta = interpolate_unreliable_centerline(coarse, unreliable_z)
        coarse_use = reject_centerline_jumps(coarse_use, max_jump_vox=args.max_centerline_jump)
        coarse_use = smooth_centerline(coarse_use, window=max(5, args.smooth_window // 2 * 2 + 1))
        before_cl = coarse_use.copy()
        raw_cl, ortho_meta = extract_centerline_local_ortho(
            mask_zyx,
            vol_zyx,
            spacing_zyx,
            init_pts_zyx=coarse_use,
            step_um=args.pca_step_um,
            radius_um=args.ortho_radius_um,
            n_iters=args.ortho_iters,
            section_mode=args.ortho_section_mode,
            max_shift_um=float(args.ortho_max_shift_um),
            max_drift_from_init_um=float(args.ortho_max_drift_um),
            prior_blend=float(args.ortho_prior_blend),
            smooth_window=max(5, int(args.smooth_window) | 1),
        )
        interp_cl = raw_cl
    else:
        raw_cl = extract_centerline_zyx(mask_zyx, vol_zyx, mode=args.centerline_mode)
        if args.no_incomplete_interp:
            interp_cl, interp_meta = raw_cl, {"skipped": True}
            before_cl = None
        else:
            before_cl = raw_cl.copy()
            interp_cl, interp_meta = interpolate_unreliable_centerline(raw_cl, unreliable_z)
    cleaned_cl = reject_centerline_jumps(interp_cl, max_jump_vox=args.max_centerline_jump)
    sm_cl = smooth_centerline(cleaned_cl, window=args.smooth_window)

    oblique_meta = None
    if args.refine_oblique_deg and args.refine_oblique_deg > 0 and args.centerline_mode != "local_ortho":
        before_cl = sm_cl.copy()  # QC: magenta = before oblique refine
        sm_cl, oblique_meta = refine_centerline_where_oblique(
            mask_zyx,
            vol_zyx,
            spacing_zyx,
            sm_cl,
            tilt_deg_thresh=args.refine_oblique_deg,
            radius_um=args.ortho_radius_um,
            section_mode=args.ortho_section_mode,
            dilate_n=args.refine_oblique_dilate,
            n_iters=max(1, args.ortho_iters),
        )
        sm_cl = reject_centerline_jumps(sm_cl, max_jump_vox=args.max_centerline_jump)
        sm_cl = smooth_centerline(sm_cl, window=max(5, (args.smooth_window // 2) * 2 + 1))

    do_poly = not args.no_poly_extend
    poly_meta = None
    extrapolated = None
    if do_poly:
        # Extend back to the original raw centerline Z span (pre end-drop).
        z_min = float(raw_cl[:, 0].min())
        z_max = float(raw_cl[:, 0].max())
        max_ext = None if args.poly_max_extend_vox < 0 else float(args.poly_max_extend_vox)
        sm_cl, poly_meta, extrapolated = extend_centerline_polynomial(
            sm_cl,
            z_min=z_min,
            z_max=z_max,
            degree=args.poly_degree,
            local_end_frac=float(args.poly_local_end_frac),
            max_extend_vox=max_ext,
            yx_clip=((0.0, vol_zyx.shape[1] - 1.0), (0.0, vol_zyx.shape[2] - 1.0)),
        )
        logger.info("Polynomial extend: %s", poly_meta)

    # QC magenta: before final spline (shows jump-clamped path vs over-smoothed).
    if args.final_spline_smooth_um and args.final_spline_smooth_um > 0:
        before_cl = sm_cl.copy()
        sm_cl = smooth_centerline_spline_yx(
            sm_cl,
            spacing_zyx,
            smooth_um=float(args.final_spline_smooth_um),
        )
        logger.info("Final spline smooth_um=%.1f", args.final_spline_smooth_um)

    logger.info(
        "Centerline mode=%s points=%d unreliable_z=%d/%d interp=%s pca_tilt=%s ortho=%s oblique=%s",
        args.centerline_mode,
        len(sm_cl),
        int(unreliable_z.sum()),
        len(unreliable_z),
        interp_meta,
        (
            {
                "from_z": pca_meta["tilt_from_z_deg"],
                "about_x": pca_meta["tilt_about_x_deg"],
                "about_y": pca_meta["tilt_about_y_deg"],
            }
            if pca_meta
            else None
        ),
        (
            {
                "n_iters": ortho_meta["n_iters"],
                "last_mean_shift_um": ortho_meta["iters"][-1]["shift_um_mean"],
            }
            if ortho_meta
            else None
        ),
        (
            {
                "n_refined": oblique_meta.get("n_refined"),
                "tilt_max": (oblique_meta.get("tilt_deg_before") or {}).get("max"),
            }
            if oblique_meta
            else None
        ),
    )

    # Save NIfTI mask (XYZ)
    mask_xyz = np.transpose(mask_zyx.astype(np.uint8), (2, 1, 0))
    mask_nii = out_dir / "tissue_mask.nii.gz"
    nib.save(nib.Nifti1Image(mask_xyz, nii.affine), str(mask_nii))

    # Masked volume for optional inspection
    masked_xyz = vol_xyz * mask_xyz.astype(np.float32)
    nib.save(nib.Nifti1Image(masked_xyz, nii.affine), str(out_dir / "volume_masked.nii.gz"))

    # Centerline CSV in voxel + mm (atlas-like ZYX)
    csv_path = out_dir / "centerline_zyx.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("z_vox,y_vox,x_vox,z_mm,y_mm,x_mm,unreliable,extrapolated\n")
        for i, (z, y, x) in enumerate(sm_cl):
            zi = int(np.clip(round(z), 0, len(unreliable_z) - 1))
            ext = int(extrapolated[i]) if extrapolated is not None else 0
            f.write(
                f"{z:.3f},{y:.3f},{x:.3f},"
                f"{z * spacing_zyx[0] / 1000.0:.5f},"
                f"{y * spacing_zyx[1] / 1000.0:.5f},"
                f"{x * spacing_zyx[2] / 1000.0:.5f},"
                f"{int(unreliable_z[zi])},{ext}\n"
            )

    write_qc(
        out_dir,
        vol_zyx,
        mask_zyx,
        sm_cl,
        spacing_zyx,
        unreliable_z=unreliable_z,
        centerline_before=before_cl,
    )

    summary = {
        "volume_nii": str(args.volume_nii),
        "shape_zyx": list(vol_zyx.shape),
        "spacing_xyz_um": list(spacing_xyz),
        "mask": mask_meta,
        "centerline_n": int(len(sm_cl)),
        "centerline_mode": args.centerline_mode,
        "pca3d": pca_meta,
        "local_ortho": ortho_meta,
        "oblique_refine": oblique_meta,
        "incomplete_slice_interp": interp_meta,
        "poly_extend": poly_meta,
        "final_spline_smooth_um": float(args.final_spline_smooth_um),
        "unreliable_z_count": int(unreliable_z.sum()),
        "unreliable_z_frac": float(unreliable_z.mean()),
        "border_margin": args.border_margin,
        "end_frac": args.end_frac,
        "outputs": {
            "tissue_mask": str(mask_nii),
            "volume_masked": str(out_dir / "volume_masked.nii.gz"),
            "centerline_csv": str(csv_path),
            "qc_dir": str(out_dir / "qc"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Done. QC in %s", out_dir / "qc")


if __name__ == "__main__":
    main()
