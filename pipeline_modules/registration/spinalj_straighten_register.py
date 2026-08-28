#!/usr/bin/env python3
"""Straighten LSFM spinal cord along a centerline, then register SpinalJ atlas.

Uses centerline CSV from spinalj_mask_centerline_preview (z,y,x voxels in
atlas-like ZYX). Does not use the SpinalJ Fiji plugin.

2D untwist is OFF by default; pass ``--untwist`` only after manual confirmation
(``--untwist_method atlas|symmetry``).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import cv2
import numpy as np
import tifffile

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _to_u8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = np.percentile(arr, (1, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    return (np.clip((arr - lo) / (hi - lo), 0, 1) * 255.0).astype(np.uint8)


def _write_overlay_png(path: Path, fixed: np.ndarray, warped: np.ndarray) -> None:
    f = _to_u8(fixed)
    w = _to_u8(warped)
    if f.shape != w.shape:
        h = max(f.shape[0], w.shape[0])
        ww = max(f.shape[1], w.shape[1])
        f2 = np.zeros((h, ww), dtype=np.uint8)
        w2 = np.zeros((h, ww), dtype=np.uint8)
        f2[: f.shape[0], : f.shape[1]] = f
        w2[: w.shape[0], : w.shape[1]] = w
        f, w = f2, w2
    cv2.imwrite(str(path), np.stack([f, w, f], axis=-1))  # BGR: magenta/green


def load_centerline_csv(path: Path) -> np.ndarray:
    """Return Nx3 (z,y,x) in voxels."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
    if not rows:
        raise RuntimeError(f"No centerline points in {path}")
    return np.asarray(rows, dtype=np.float64)


def _arc_length_um(
    pts_zyx: np.ndarray,
    spacing_zyx_um: tuple[float, float, float],
) -> np.ndarray:
    sp = np.asarray(spacing_zyx_um, dtype=np.float64)
    d = np.linalg.norm(np.diff(pts_zyx * sp, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def _resample_centerline_uniform(
    pts_zyx: np.ndarray,
    spacing_zyx_um: tuple[float, float, float],
    step_um: float,
) -> np.ndarray:
    """Resample centerline to approximately uniform arc-length spacing."""
    s = _arc_length_um(pts_zyx, spacing_zyx_um)
    total = float(s[-1])
    if total <= step_um:
        return pts_zyx.copy()
    s_new = np.arange(0.0, total + 0.5 * step_um, step_um)
    out = np.column_stack([np.interp(s_new, s, pts_zyx[:, i]) for i in range(3)])
    return out


def _smooth_centerline_spline(
    pts_zyx: np.ndarray,
    spacing_zyx_um: tuple[float, float, float],
    *,
    smooth_um: float,
) -> np.ndarray:
    """Smoothing spline along arc length (reduces lateral jitter that makes slices uneven)."""
    from scipy.interpolate import UnivariateSpline

    if pts_zyx.shape[0] < 8 or smooth_um <= 0:
        return pts_zyx.copy()
    s = _arc_length_um(pts_zyx, spacing_zyx_um)
    # Drop duplicate arc samples (zero-length steps).
    keep = np.concatenate([[True], np.diff(s) > 1e-6])
    s_k = s[keep]
    pts_k = pts_zyx[keep]
    if len(s_k) < 8:
        return pts_zyx.copy()
    # UnivariateSpline: larger ``s`` → smoother. Scale with length and noise target.
    # smooth_um ≈ allowed RMS deviation in microns on Y/X (roughly).
    n = len(s_k)
    s_factor = float(max(smooth_um, 1.0) ** 2 * n)
    out = np.zeros_like(pts_k)
    for i in range(3):
        # Keep Z slightly tighter than Y/X so arc order stays stable.
        sf = s_factor if i > 0 else s_factor * 0.25
        spl = UnivariateSpline(s_k, pts_k[:, i], k=3, s=sf)
        out[:, i] = spl(s_k)
    # Re-embed onto original arc parametrization length via uniform resample later.
    return out


def _moving_average_1d(x: np.ndarray, window: int) -> np.ndarray:
    w = max(1, int(window) | 1)
    if w <= 1 or len(x) < 3:
        return np.asarray(x, dtype=np.float64)
    kernel = np.ones(w, dtype=np.float64) / w
    pad = w // 2
    return np.convolve(np.pad(np.asarray(x, dtype=np.float64), (pad, pad), mode="edge"), kernel, mode="valid")


def _smooth_unit_vectors(vecs: np.ndarray, window: int) -> np.ndarray:
    """Smooth Nx3 unit vectors by averaging then renormalizing."""
    if window <= 1 or len(vecs) < 3:
        return vecs.copy()
    sm = np.column_stack([_moving_average_1d(vecs[:, i], window) for i in range(3)])
    n = np.linalg.norm(sm, axis=1, keepdims=True) + 1e-12
    return sm / n


def _local_frame(tangent: np.ndarray, prev_normal: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Build orthonormal (normal, binormal) for unit tangent, with parallel transport."""
    t = tangent / (np.linalg.norm(tangent) + 1e-12)
    if prev_normal is None:
        # Prefer a normal roughly in YX plane.
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(np.dot(t, ref)) > 0.9:
            ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        n = ref - np.dot(ref, t) * t
        n = n / (np.linalg.norm(n) + 1e-12)
    else:
        # Parallel transport: project previous normal onto plane ⊥ t.
        n = prev_normal - np.dot(prev_normal, t) * t
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


def _build_smoothed_frames(
    pts_um: np.ndarray,
    *,
    tangent_smooth_window: int,
    frame_smooth_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tangents + parallel-transport frames with optional smoothing along the path."""
    n = len(pts_um)
    tang = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        if i == 0:
            tang[i] = pts_um[min(1, n - 1)] - pts_um[0]
        elif i == n - 1:
            tang[i] = pts_um[-1] - pts_um[-2]
        else:
            tang[i] = pts_um[i + 1] - pts_um[i - 1]
    tang = tang / (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12)
    tang = _smooth_unit_vectors(tang, tangent_smooth_window)

    normals = np.zeros((n, 3), dtype=np.float64)
    binormals = np.zeros((n, 3), dtype=np.float64)
    prev_n = None
    for i in range(n):
        n_hat, b_hat = _local_frame(tang[i], prev_n)
        normals[i] = n_hat
        binormals[i] = b_hat
        prev_n = n_hat

    if frame_smooth_window > 1:
        # Smooth normals in ambient space, re-orthonormalize against tangent.
        normals = _smooth_unit_vectors(normals, frame_smooth_window)
        for i in range(n):
            t = tang[i]
            n_hat = normals[i] - np.dot(normals[i], t) * t
            nn = np.linalg.norm(n_hat)
            if nn < 1e-8:
                n_hat, b_hat = _local_frame(t, None)
            else:
                n_hat = n_hat / nn
                b_hat = np.cross(t, n_hat)
                b_hat = b_hat / (np.linalg.norm(b_hat) + 1e-12)
            normals[i] = n_hat
            binormals[i] = b_hat
    return tang, normals, binormals


def straighten_along_centerline(
    vol_zyx: np.ndarray,
    centerline_zyx: np.ndarray,
    *,
    spacing_zyx_um: tuple[float, float, float],
    out_radius_yx_vox: int = 160,
    step_um: float | None = None,
    order: int = 3,
    smooth_centerline_um: float = 40.0,
    tangent_smooth_window: int = 31,
    frame_smooth_window: int = 31,
) -> tuple[np.ndarray, dict]:
    """Resample perpendicular planes along centerline into a straight Z stack.

    Smoother defaults (vs older linear path):
    - spline-smooth centerline before uniform resampling
    - moving-average tangents / parallel-transport frames
    - cubic ``map_coordinates`` (order=3)

    Output shape: (N, 2R, 2R) in ZYX, with centerline mapped to (R, R).
    """
    from scipy.ndimage import map_coordinates

    step = float(step_um if step_um is not None else spacing_zyx_um[0])
    cl_in = np.asarray(centerline_zyx, dtype=np.float64)
    if smooth_centerline_um and smooth_centerline_um > 0:
        cl_in = _smooth_centerline_spline(cl_in, spacing_zyx_um, smooth_um=float(smooth_centerline_um))
    cl = _resample_centerline_uniform(cl_in, spacing_zyx_um, step)
    n_planes = len(cl)
    r = int(out_radius_yx_vox)
    size = 2 * r
    sy, sx = spacing_zyx_um[1], spacing_zyx_um[2]
    pitch = 0.5 * (sy + sx)
    sp = np.asarray(spacing_zyx_um, dtype=np.float64)
    pts_um = cl * sp

    yy, xx = np.mgrid[-r:r, -r:r]
    off_y_um = yy.astype(np.float64) * pitch
    off_x_um = xx.astype(np.float64) * pitch

    _tang, normals, binormals = _build_smoothed_frames(
        pts_um,
        tangent_smooth_window=int(tangent_smooth_window),
        frame_smooth_window=int(frame_smooth_window),
    )

    out = np.zeros((n_planes, size, size), dtype=np.float32)
    order_i = int(order)
    for i in range(n_planes):
        n_hat = normals[i]
        b_hat = binormals[i]
        c_um = pts_um[i]
        oz = (c_um[0] + n_hat[0] * off_y_um + b_hat[0] * off_x_um) / sp[0]
        oy = (c_um[1] + n_hat[1] * off_y_um + b_hat[1] * off_x_um) / sp[1]
        ox = (c_um[2] + n_hat[2] * off_y_um + b_hat[2] * off_x_um) / sp[2]
        coords = np.stack([oz.ravel(), oy.ravel(), ox.ravel()], axis=0)
        samp = map_coordinates(vol_zyx, coords, order=order_i, mode="constant", cval=0.0)
        out[i] = samp.reshape(size, size).astype(np.float32, copy=False)
        if (i + 1) % 200 == 0 or i + 1 == n_planes:
            logger.info("  straighten plane %d/%d", i + 1, n_planes)

    meta = {
        "n_planes": n_planes,
        "out_shape_zyx": list(out.shape),
        "out_radius_yx_vox": r,
        "plane_pitch_um": pitch,
        "step_um": step,
        "spacing_zyx_um_out": [step, pitch, pitch],
        "centerline_in_n": int(len(centerline_zyx)),
        "centerline_resampled_n": n_planes,
        "interp_order": order_i,
        "smooth_centerline_um": float(smooth_centerline_um),
        "tangent_smooth_window": int(tangent_smooth_window),
        "frame_smooth_window": int(frame_smooth_window),
    }
    return out, meta


def _mirror_asymmetry(img: np.ndarray) -> float:
    """Mean |left - flip(right)|; lower = more left-right symmetric."""
    h, w = img.shape
    mid = w // 2
    left = img[:, :mid]
    right = np.fliplr(img[:, w - mid :])
    m = min(left.shape[1], right.shape[1])
    if m <= 0:
        return 1e9
    d = np.abs(left[:, -m:].astype(np.float32) - right[:, :m].astype(np.float32))
    return float(d.mean())


def _normalize_u8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    pos = img[img > 0]
    if pos.size == 0:
        return np.zeros(img.shape, dtype=np.uint8)
    lo, hi = np.percentile(pos, (5, 99))
    if hi <= lo:
        hi = lo + 1.0
    return (np.clip((img - lo) / (hi - lo), 0, 1) * 255.0).astype(np.uint8)


def _mutual_information_u8(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    """Higher is better. Expects uint8 images of equal shape."""
    a = a.ravel()
    b = b.ravel()
    # Keep voxels where either channel has signal to avoid empty-background domination.
    m = (a > 0) | (b > 0)
    if not np.any(m):
        return -1e9
    aa = a[m]
    bb = b[m]
    hist_2d, _, _ = np.histogram2d(aa, bb, bins=bins, range=[[0, 255], [0, 255]])
    pxy = hist_2d / hist_2d.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    px_py = px[:, None] * py[None, :]
    nz = pxy > 0
    return float(np.sum(pxy[nz] * np.log(pxy[nz] / (px_py[nz] + 1e-12))))


def _center_match_yx(src: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Center-crop or zero-pad 2D array to target (H, W)."""
    th, tw = target_hw
    sh, sw = src.shape
    out = np.zeros((th, tw), dtype=src.dtype)
    src_y0 = max(0, (sh - th) // 2)
    src_x0 = max(0, (sw - tw) // 2)
    dst_y0 = max(0, (th - sh) // 2)
    dst_x0 = max(0, (tw - sw) // 2)
    h = min(th, sh)
    w = min(tw, sw)
    out[dst_y0 : dst_y0 + h, dst_x0 : dst_x0 + w] = src[src_y0 : src_y0 + h, src_x0 : src_x0 + w]
    return out


def _unwrap_angles_deg(angles: np.ndarray, period: float = 180.0) -> np.ndarray:
    out = np.asarray(angles, dtype=np.float64).copy()
    half = period / 2.0
    for i in range(1, len(out)):
        while out[i] - out[i - 1] > half:
            out[i] -= period
        while out[i] - out[i - 1] < -half:
            out[i] += period
    return out


def _interp_smooth_angles(
    measured_z: np.ndarray,
    measured_a: np.ndarray,
    n: int,
    *,
    smooth_window: int,
    zero_center: bool,
) -> np.ndarray:
    ma = _unwrap_angles_deg(measured_a)
    z_all = np.arange(n, dtype=np.float64)
    ang = np.interp(z_all, measured_z.astype(np.float64), ma)
    w = max(3, int(smooth_window) | 1)
    kernel = np.ones(w, dtype=np.float64) / w
    pad = w // 2
    ang = np.convolve(np.pad(ang, (pad, pad), mode="edge"), kernel, mode="valid")
    if zero_center:
        ang = ang - np.median(ang)
    return ang


def _apply_slice_rotations_zyx(vol_zyx: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    out = np.zeros_like(vol_zyx)
    cy = (vol_zyx.shape[1] - 1) / 2.0
    cx = (vol_zyx.shape[2] - 1) / 2.0
    n = vol_zyx.shape[0]
    for z in range(n):
        a = float(angles_deg[z])
        if abs(a) < 1e-3:
            out[z] = vol_zyx[z]
            continue
        M = cv2.getRotationMatrix2D((cx, cy), a, 1.0)
        out[z] = cv2.warpAffine(
            vol_zyx[z],
            M,
            (vol_zyx.shape[2], vol_zyx.shape[1]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if (z + 1) % 400 == 0 or z + 1 == n:
            logger.info("  untwist apply %d/%d", z + 1, n)
    return out


def _angle_meta(ang: np.ndarray, *, method: str, angle_lim: float, smooth_window: int, subsample_z: int) -> dict:
    return {
        "method": method,
        "angle_lim_deg": float(angle_lim),
        "smooth_window": int(smooth_window),
        "subsample_z": int(subsample_z),
        "angle_deg_p01": float(np.percentile(ang, 1)),
        "angle_deg_p50": float(np.percentile(ang, 50)),
        "angle_deg_p99": float(np.percentile(ang, 99)),
        "angle_deg_abs_mean": float(np.mean(np.abs(ang))),
    }


def estimate_slice_rotation_deg(
    img: np.ndarray,
    *,
    angle_lim: float = 90.0,
    coarse_step: float = 5.0,
    fine_step: float = 1.0,
    prefer_angle: float | None = None,
    prefer_weight: float = 0.0,
) -> tuple[float, float, float]:
    """Find in-plane rotation (degrees, CCW) minimizing |L-R| asymmetry.

    Returns (angle_deg, asymmetry_score, fg_frac).
    If ``prefer_angle`` is set, score is penalized away from it to favor continuity
    when multiple minima exist (e.g. near ±90/±180).
    """
    img = np.asarray(img, dtype=np.float32)
    if not np.any(img > 0):
        return 0.0, 1e9, 0.0
    u8 = _normalize_u8(img)
    pos = u8[u8 > 0]
    thr = float(np.percentile(pos, 70)) if pos.size else 0.0
    bin0 = (u8 > thr).astype(np.uint8) * 255
    fg_frac = float(np.mean(bin0 > 0))
    if fg_frac < 1e-4:
        return 0.0, 1e9, fg_frac
    cy, cx = (bin0.shape[0] - 1) / 2.0, (bin0.shape[1] - 1) / 2.0

    def asym(a: float) -> float:
        M = cv2.getRotationMatrix2D((cx, cy), float(a), 1.0)
        rot = cv2.warpAffine(bin0, M, (bin0.shape[1], bin0.shape[0]), flags=cv2.INTER_NEAREST)
        return _mirror_asymmetry(rot)

    def total(a: float) -> float:
        s = asym(a)
        if prefer_angle is not None and prefer_weight > 0:
            # Soft continuity pull (degrees).
            s = s + prefer_weight * abs(a - prefer_angle)
        return s

    best_a = 0.0 if prefer_angle is None else float(prefer_angle)
    best_s = total(best_a)
    best_raw = asym(best_a)
    a = -angle_lim
    while a <= angle_lim + 1e-9:
        s = total(a)
        if s < best_s:
            best_s, best_a, best_raw = s, float(a), asym(a)
        a += coarse_step

    a0 = best_a
    a = a0 - coarse_step
    while a <= a0 + coarse_step + 1e-9:
        s = total(a)
        if s < best_s:
            best_s, best_a, best_raw = s, float(a), asym(a)
        a += fine_step
    return best_a, float(best_raw), fg_frac


def _reject_angle_outliers(
    z: np.ndarray,
    ang: np.ndarray,
    score: np.ndarray,
    fg: np.ndarray,
    *,
    max_jump_deg: float,
    score_p95: float | None,
    min_fg_frac: float,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mark bad samples; return (clean_angles with NaN at bad, bad_mask)."""
    a = np.asarray(ang, dtype=np.float64).copy()
    bad = np.zeros(len(a), dtype=bool)
    if min_fg_frac > 0:
        bad |= fg < min_fg_frac
    if score_p95 is not None and np.isfinite(score_p95):
        # High residual |L-R| => damaged / non-symmetric section.
        bad |= score > score_p95
    # Jump vs local median.
    w = max(3, int(window) | 1)
    half = w // 2
    for i in range(len(a)):
        lo, hi = max(0, i - half), min(len(a), i + half + 1)
        med = float(np.nanmedian(a[lo:hi]))
        if abs(a[i] - med) > max_jump_deg:
            bad[i] = True
    a[bad] = np.nan
    return a, bad


def untwist_volume_zyx(
    vol_zyx: np.ndarray,
    *,
    angle_lim: float = 170.0,
    smooth_window: int = 51,
    subsample_z: int = 2,
    max_jump_deg: float = 25.0,
    min_fg_frac: float = 0.01,
    prefer_weight: float = 0.25,
    zero_center: bool = True,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Correct per-slice in-plane rotation via bilateral |L-R| symmetry.

    Wide search (default ±170°) covers near-180° embedding rotations. Walk from
    mid-Z with continuity preference, then reject high-|L-R| / jump outliers from
    damaged slices before smooth interpolation.
    """
    n = vol_zyx.shape[0]
    zs = list(range(0, n, max(1, subsample_z)))
    if zs[-1] != n - 1:
        zs.append(n - 1)
    mid_idx = len(zs) // 2
    z_mid = zs[mid_idx]

    seed, seed_s, seed_fg = estimate_slice_rotation_deg(
        vol_zyx[z_mid],
        angle_lim=angle_lim,
        coarse_step=5.0,
        fine_step=1.0,
        prefer_angle=None,
    )
    logger.info(
        "  untwist(symmetry) mid seed z=%d a=%.1f score=%.2f fg=%.3f lim=±%.0f",
        z_mid,
        seed,
        seed_s,
        seed_fg,
        angle_lim,
    )

    measured_z: list[int] = []
    measured_a: list[float] = []
    measured_s: list[float] = []
    measured_fg: list[float] = []

    def _append_walk(indices: list[int], start_pref: float, label: str) -> None:
        pref = start_pref
        for i, zi in enumerate(indices):
            use_pref = None if (label == "fwd" and i == 0) else pref
            wpref = 0.0 if use_pref is None else prefer_weight
            a, s, fg = estimate_slice_rotation_deg(
                vol_zyx[zi],
                angle_lim=angle_lim,
                coarse_step=5.0,
                fine_step=1.0,
                prefer_angle=use_pref,
                prefer_weight=wpref,
            )
            if use_pref is not None and abs(a - pref) > max_jump_deg:
                a2, s2, fg2 = estimate_slice_rotation_deg(
                    vol_zyx[zi],
                    angle_lim=angle_lim,
                    coarse_step=2.0,
                    fine_step=0.5,
                    prefer_angle=pref,
                    prefer_weight=prefer_weight * 5.0,
                )
                if abs(a2 - pref) <= max_jump_deg * 1.25:
                    a, s, fg = a2, s2, fg2
                else:
                    a = pref  # hold through damaged / ambiguous slice
            measured_z.append(int(zi))
            measured_a.append(float(a))
            measured_s.append(float(s))
            measured_fg.append(float(fg))
            pref = float(a)
            if (i + 1) % 50 == 0 or i + 1 == len(indices):
                logger.info(
                    "  untwist(symmetry) %s %d/%d (z=%d a=%.1f)",
                    label,
                    i + 1,
                    len(indices),
                    zi,
                    a,
                )

    _append_walk(zs[mid_idx:], seed, "fwd")
    _append_walk(list(reversed(zs[:mid_idx])), seed, "bak")

    order = np.argsort(np.asarray(measured_z))
    mz = np.asarray(measured_z, dtype=np.float64)[order]
    ma = np.asarray(measured_a, dtype=np.float64)[order]
    ms = np.asarray(measured_s, dtype=np.float64)[order]
    mfg = np.asarray(measured_fg, dtype=np.float64)[order]

    score_p90 = float(np.percentile(ms, 90))
    ma_clean, bad = _reject_angle_outliers(
        mz,
        ma,
        ms,
        mfg,
        max_jump_deg=max_jump_deg,
        score_p95=score_p90,
        min_fg_frac=min_fg_frac,
        window=max(7, smooth_window // 2),
    )
    n_bad = int(bad.sum())
    if n_bad:
        logger.info("  untwist(symmetry) rejected %d / %d outlier angles", n_bad, len(ma))

    good = np.isfinite(ma_clean)
    if int(good.sum()) < 3:
        logger.warning("Too few valid symmetry angles; using raw estimates")
        ma_fill = ma.copy()
    else:
        ma_fill = ma_clean.copy()
        ma_fill[~good] = np.interp(mz[~good], mz[good], ma_clean[good])
    ma_fill = _unwrap_angles_deg(ma_fill, period=180.0)

    w = max(3, int(smooth_window) | 1)
    ang = _interp_smooth_angles(mz, ma_fill, n, smooth_window=w, zero_center=False)
    # Suppress residual spikes after interp.
    kernel = np.ones(w, dtype=np.float64) / w
    pad = w // 2
    med = np.convolve(np.pad(ang, (pad, pad), mode="edge"), kernel, mode="valid")
    ang = np.clip(ang, med - max_jump_deg, med + max_jump_deg)
    ang = np.convolve(np.pad(ang, (pad, pad), mode="edge"), kernel, mode="valid")
    if zero_center:
        ang = ang - np.median(ang)

    out = _apply_slice_rotations_zyx(vol_zyx, ang)
    meta = _angle_meta(ang, method="symmetry", angle_lim=angle_lim, smooth_window=w, subsample_z=subsample_z)
    meta.update(
        {
            "max_jump_deg": float(max_jump_deg),
            "min_fg_frac": float(min_fg_frac),
            "prefer_weight": float(prefer_weight),
            "n_measured": int(len(mz)),
            "n_rejected": n_bad,
            "score_p90": score_p90,
            "seed_deg": float(seed),
            "zero_center": bool(zero_center),
        }
    )
    return out, meta, ang


def estimate_slice_rotation_vs_atlas_deg(
    sample: np.ndarray,
    atlas: np.ndarray,
    *,
    angle_lim: float = 90.0,
    coarse_step: float = 5.0,
    fine_step: float = 1.0,
) -> float:
    """Find in-plane rotation of sample (CCW deg) maximizing MI vs atlas slice."""
    sample = np.asarray(sample, dtype=np.float32)
    atlas = _center_match_yx(np.asarray(atlas, dtype=np.float32), sample.shape)
    if not np.any(sample > 0) or not np.any(atlas > 0):
        return 0.0
    samp_u8 = _normalize_u8(sample)
    atl_u8 = _normalize_u8(atlas)
    cy, cx = (samp_u8.shape[0] - 1) / 2.0, (samp_u8.shape[1] - 1) / 2.0

    def score(a: float) -> float:
        M = cv2.getRotationMatrix2D((cx, cy), float(a), 1.0)
        rot = cv2.warpAffine(samp_u8, M, (samp_u8.shape[1], samp_u8.shape[0]), flags=cv2.INTER_LINEAR)
        return _mutual_information_u8(rot, atl_u8)

    best_a = 0.0
    best_s = score(0.0)
    a = -angle_lim
    while a <= angle_lim + 1e-9:
        s = score(a)
        if s > best_s:
            best_s, best_a = s, float(a)
        a += coarse_step

    a0 = best_a
    a = a0 - coarse_step
    while a <= a0 + coarse_step + 1e-9:
        s = score(a)
        if s > best_s:
            best_s, best_a = s, float(a)
        a += fine_step
    return best_a


def untwist_volume_zyx_atlas(
    vol_zyx: np.ndarray,
    atlas_zyx: np.ndarray,
    *,
    angle_lim: float = 90.0,
    local_lim: float = 25.0,
    smooth_window: int = 31,
    subsample_z: int = 4,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Correct per-slice in-plane rotation by maximizing MI vs atlas Template.

    1) Estimate a global in-plane angle from mid-Z slices (full ``angle_lim`` search).
    2) For each slice, search only within ``global_seed ± local_lim`` (no sequential
       drift). This models a slow twist around a stable DV orientation vs atlas.
    Absolute angles vs atlas are kept (not median-centered).
    """
    n = vol_zyx.shape[0]
    na = atlas_zyx.shape[0]
    zs = list(range(0, n, max(1, subsample_z)))
    if zs[-1] != n - 1:
        zs.append(n - 1)

    def atlas_z(z: int) -> int:
        az = int(round(z / max(1, n - 1) * (na - 1)))
        return int(np.clip(az, 0, na - 1))

    # Global orientation from central third of the cord.
    z0, z1 = int(0.33 * (n - 1)), int(0.67 * (n - 1))
    mid_zs = [z for z in zs if z0 <= z <= z1]
    if not mid_zs:
        mid_zs = [n // 2]
    global_angles = []
    for z in mid_zs[:: max(1, len(mid_zs) // 8)]:
        a = estimate_slice_rotation_vs_atlas_deg(
            vol_zyx[z],
            atlas_zyx[atlas_z(z)],
            angle_lim=angle_lim,
        )
        global_angles.append(a)
        logger.info("  untwist(atlas) global seed z=%d a=%.1f", z, a)
    ga = np.asarray(global_angles, dtype=np.float64)
    # Circular mean on 180° period (cord approx 2-fold for weak multimodal MI).
    seed = float(np.angle(np.mean(np.exp(1j * np.deg2rad(ga * 2.0)))) / 2.0 * 180.0 / np.pi)
    logger.info("  untwist(atlas) global seed=%.1f (from %d mid slices)", seed, len(global_angles))

    measured = []
    for i, z in enumerate(zs):
        az = atlas_z(z)
        sample = vol_zyx[z]
        atlas_m = _center_match_yx(np.asarray(atlas_zyx[az], dtype=np.float32), sample.shape)
        if not np.any(sample > 0) or not np.any(atlas_m > 0):
            a = seed
        else:
            samp_u8 = _normalize_u8(sample)
            atl_u8 = _normalize_u8(atlas_m)
            cy, cx = (samp_u8.shape[0] - 1) / 2.0, (samp_u8.shape[1] - 1) / 2.0

            def score(a_abs: float) -> float:
                M = cv2.getRotationMatrix2D((cx, cy), float(a_abs), 1.0)
                rot = cv2.warpAffine(
                    samp_u8, M, (samp_u8.shape[1], samp_u8.shape[0]), flags=cv2.INTER_LINEAR
                )
                return _mutual_information_u8(rot, atl_u8)

            best_a = float(seed)
            best_s = score(best_a)
            a = seed - local_lim
            while a <= seed + local_lim + 1e-9:
                s = score(a)
                if s > best_s:
                    best_s, best_a = s, float(a)
                a += 2.0
            a0 = best_a
            a = a0 - 2.0
            while a <= a0 + 2.0 + 1e-9:
                s = score(a)
                if s > best_s:
                    best_s, best_a = s, float(a)
                a += 0.5
            a = best_a
        measured.append((z, a))
        if (i + 1) % 25 == 0 or i + 1 == len(zs):
            logger.info("  untwist(atlas) %d/%d (z=%d az=%d a=%.1f)", i + 1, len(zs), z, az, a)

    mz = np.asarray([t[0] for t in measured], dtype=np.float64)
    ma = np.asarray([t[1] for t in measured], dtype=np.float64)
    w = max(3, smooth_window | 1)
    ang = _interp_smooth_angles(mz, ma, n, smooth_window=w, zero_center=False)
    out = _apply_slice_rotations_zyx(vol_zyx, ang)
    meta = _angle_meta(ang, method="atlas", angle_lim=angle_lim, smooth_window=w, subsample_z=subsample_z)
    meta["local_lim_deg"] = float(local_lim)
    meta["global_seed_deg"] = float(seed)
    meta["atlas_shape_zyx"] = list(map(int, atlas_zyx.shape))
    return out, meta, ang


def parse_float_triplet(text: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _write_angle_plot(path: Path, angles: np.ndarray) -> None:
    ang_img = np.zeros((200, len(angles)), dtype=np.uint8)
    a_norm = np.clip((angles - angles.min()) / (np.ptp(angles) + 1e-6) * 199, 0, 199).astype(int)
    for i, v in enumerate(a_norm):
        ang_img[199 - v, i] = 255
    cv2.imwrite(str(path), ang_img)


def _prepare_atlas_paths(
    args: argparse.Namespace,
    out_dir: Path,
    convert_atlas_to_nifti,
) -> dict:
    atlas_nifti_dir = out_dir / 'atlas_nifti'
    if args.atlas_template_nii and args.atlas_annotation_nii:
        atlas_nifti_dir.mkdir(exist_ok=True)
        template_src = Path(args.atlas_template_nii)
        annotation_src = Path(args.atlas_annotation_nii)
        template_dst = atlas_nifti_dir / 'Template.nii.gz'
        annotation_dst = atlas_nifti_dir / 'Annotation.nii.gz'
        if template_src.resolve() != template_dst.resolve():
            shutil.copy2(template_src, template_dst)
        if annotation_src.resolve() != annotation_dst.resolve():
            shutil.copy2(annotation_src, annotation_dst)
        return {'template': template_dst, 'annotation': annotation_dst}
    if args.atlas_dir:
        return convert_atlas_to_nifti(Path(args.atlas_dir), atlas_nifti_dir)
    raise SystemExit('Provide --atlas_dir or both --atlas_template_nii and --atlas_annotation_nii')


def _resample_atlas_to_spacing(
    src: Path,
    dst: Path,
    *,
    spacing_xyz: tuple[float, float, float],
    is_label: bool,
) -> Path:
    import ants
    import nibabel as nib

    sx, sy, sz = spacing_xyz
    nii_a = nib.load(str(src))
    spacing = tuple(float(abs(nii_a.affine[i, i])) for i in range(3))
    data = np.asanyarray(nii_a.dataobj).astype(np.float32)
    img = ants.from_numpy(data, spacing=spacing)
    img.set_direction(np.eye(3))
    ref = ants.from_numpy(
        np.zeros(
            (
                max(1, int(round(img.shape[0] * spacing[0] / sx))),
                max(1, int(round(img.shape[1] * spacing[1] / sy))),
                max(1, int(round(img.shape[2] * spacing[2] / sz))),
            ),
            dtype=np.float32,
        ),
        spacing=(sx, sy, sz),
    )
    ref.set_direction(np.eye(3))
    out = ants.resample_image_to_target(
        img, ref, interp_type='nearestNeighbor' if is_label else 'linear'
    )
    ants.image_write(out, str(dst))
    return dst


def main() -> None:
    _configure_logging()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--volume_nii', required=True, help='Sample volume NIfTI (XYZ)')
    p.add_argument('--centerline_csv', default='', help='centerline_zyx.csv from preview (required unless --skip_straighten)')
    p.add_argument('--atlas_dir', default='', help='SpinalJ atlas folder with Template.tif (optional if nifti given)')
    p.add_argument('--atlas_template_nii', default='', help='Optional preconverted Template.nii.gz')
    p.add_argument('--atlas_annotation_nii', default='', help='Optional preconverted Annotation.nii.gz')
    p.add_argument('--out_dir', required=True)
    p.add_argument(
        '--spacing_zyx_um',
        type=parse_float_triplet,
        default=None,
        help='Override Z,Y,X spacing in um (default: from NIfTI affine as Z,Y,X)',
    )
    p.add_argument('--out_radius_yx_vox', type=int, default=160)
    p.add_argument(
        '--straighten_smooth_um',
        type=float,
        default=40.0,
        help='Spline-smooth centerline before straighten (um RMS-ish; 0 disables). Reduces slice jitter.',
    )
    p.add_argument(
        '--straighten_tangent_window',
        type=int,
        default=31,
        help='Moving-average window for path tangents (odd; 1 disables)',
    )
    p.add_argument(
        '--straighten_frame_window',
        type=int,
        default=31,
        help='Moving-average window for slice frames / twist (odd; 1 disables)',
    )
    p.add_argument(
        '--straighten_interp_order',
        type=int,
        default=3,
        help='map_coordinates order for plane resampling (3=cubic, smoother than 1)',
    )
    p.add_argument(
        '--transform',
        default='SyNRA',
        help='ANTs type_of_transform: SyNRA (Rigid+Affine+SyN, default), SyN, Affine, ...',
    )
    p.add_argument(
        '--allow_reflection',
        action='store_true',
        help='Allow affine reflection if sample appears mirrored vs atlas',
    )
    p.add_argument(
        '--direction_y',
        type=float,
        default=1.0,
        help='Fixed-image ANTs direction YY (default +1 = no Y flip; use -1 to flip Y). Atlas stays +1.',
    )
    p.add_argument('--skip_straighten', action='store_true')
    p.add_argument('--skip_register', action='store_true')
    p.add_argument(
        '--untwist',
        action='store_true',
        help='Enable 2D rotation untwist after straighten (OFF by default; only use after manual confirm)',
    )
    p.add_argument(
        '--no_untwist',
        action='store_true',
        help='Deprecated: untwist is already off unless --untwist is set',
    )
    p.add_argument(
        '--untwist_method',
        choices=('atlas', 'symmetry'),
        default='atlas',
        help='Used only with --untwist: atlas=MI vs Template; symmetry=bilateral mirror',
    )
    p.add_argument(
        '--untwist_angle_lim',
        type=float,
        default=None,
        help='In-plane search limit in degrees (default: 90 atlas / 170 symmetry)',
    )
    p.add_argument(
        '--untwist_subsample_z',
        type=int,
        default=None,
        help='Estimate angle every N slices (default: 4 atlas / 2 symmetry)',
    )
    p.add_argument('--untwist_smooth_window', type=int, default=31)
    p.add_argument(
        '--untwist_max_jump_deg',
        type=float,
        default=25.0,
        help='Symmetry untwist: reject/hold angles jumping more than this vs neighbors',
    )
    p.add_argument(
        '--untwist_min_fg_frac',
        type=float,
        default=0.01,
        help='Symmetry untwist: reject slices with foreground fraction below this',
    )
    args = p.parse_args()

    import nibabel as nib

    from pipeline_modules.registration.spinalj_atlas_register import (
        convert_atlas_to_nifti,
        run_ants_registration,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    straight_dir = out_dir / 'straightened'
    straight_dir.mkdir(exist_ok=True)
    straight_nii = straight_dir / 'volume_straight.nii.gz'
    # Untwist only when explicitly requested (--untwist). Manual confirm first.
    do_untwist = bool(args.untwist) and not bool(args.no_untwist)
    untwist_method = args.untwist_method
    angle_lim = args.untwist_angle_lim
    if angle_lim is None:
        angle_lim = 90.0 if untwist_method == 'atlas' else 170.0
    subsample_z = args.untwist_subsample_z
    if subsample_z is None:
        subsample_z = 4 if untwist_method == 'atlas' else 2
    smooth_window = int(args.untwist_smooth_window)

    nii = nib.load(str(args.volume_nii))
    vol_xyz = np.asanyarray(nii.dataobj).astype(np.float32)
    aff = nii.affine
    spacing_xyz = tuple(float(abs(aff[i, i])) for i in range(3))
    if args.spacing_zyx_um is None:
        spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    else:
        spacing_zyx = args.spacing_zyx_um
    vol_zyx = np.transpose(vol_xyz, (2, 1, 0))
    logger.info('Input ZYX=%s spacing_zyx_um=%s', vol_zyx.shape, spacing_zyx)

    if args.skip_straighten and straight_nii.exists() and (straight_dir / 'straighten_meta.json').exists():
        logger.info('Reusing straightened volume from %s', straight_nii)
        straight_meta = json.loads((straight_dir / 'straighten_meta.json').read_text(encoding='utf-8'))
        st = np.asanyarray(nib.load(str(straight_nii)).dataobj).astype(np.float32)
        straight_zyx = np.transpose(st, (2, 1, 0))
        already_untwisted = (straight_dir / 'untwist_meta.json').exists()
    else:
        if not args.centerline_csv:
            raise SystemExit('--centerline_csv required unless --skip_straighten with existing volume')
        cl = load_centerline_csv(Path(args.centerline_csv))
        logger.info('Centerline points=%d', len(cl))
        straight_zyx, straight_meta = straighten_along_centerline(
            vol_zyx,
            cl,
            spacing_zyx_um=spacing_zyx,
            out_radius_yx_vox=args.out_radius_yx_vox,
            order=int(args.straighten_interp_order),
            smooth_centerline_um=float(args.straighten_smooth_um),
            tangent_smooth_window=int(args.straighten_tangent_window),
            frame_smooth_window=int(args.straighten_frame_window),
        )
        logger.info(
            'Straighten smooth: centerline_um=%.1f tangent_win=%d frame_win=%d order=%d',
            args.straighten_smooth_um,
            args.straighten_tangent_window,
            args.straighten_frame_window,
            args.straighten_interp_order,
        )
        already_untwisted = False
        (straight_dir / 'straighten_meta.json').write_text(json.dumps(straight_meta, indent=2), encoding='utf-8')

    sz, sy, sx = straight_meta['spacing_zyx_um_out']
    spacing_xyz_out = (float(sx), float(sy), float(sz))
    affine = np.diag([sx, sy, sz, 1.0])

    need_atlas = (do_untwist and untwist_method == 'atlas' and not already_untwisted) or (not args.skip_register)
    atlas_paths = None
    template_work = None
    annotation_work = None
    work = out_dir / 'reg_work'
    if need_atlas:
        atlas_paths = _prepare_atlas_paths(args, out_dir, convert_atlas_to_nifti)
        work.mkdir(exist_ok=True)
        template_work = _resample_atlas_to_spacing(
            atlas_paths['template'], work / 'Template_work.nii.gz', spacing_xyz=spacing_xyz_out, is_label=False
        )
        annotation_work = _resample_atlas_to_spacing(
            atlas_paths['annotation'], work / 'Annotation_work.nii.gz', spacing_xyz=spacing_xyz_out, is_label=True
        )

    untwist_meta = None
    if do_untwist and already_untwisted:
        logger.info('Reusing existing untwist: %s', straight_dir / 'untwist_meta.json')
        untwist_meta = json.loads((straight_dir / 'untwist_meta.json').read_text(encoding='utf-8'))
    elif do_untwist:
        if untwist_method == 'atlas':
            assert template_work is not None
            atl_xyz = np.asanyarray(nib.load(str(template_work)).dataobj).astype(np.float32)
            atlas_zyx = np.transpose(atl_xyz, (2, 1, 0))
            logger.info(
                'Untwisting via atlas MI (lim=+/-%.0f deg, subsample=%d); sample ZYX=%s atlas ZYX=%s',
                angle_lim,
                subsample_z,
                straight_zyx.shape,
                atlas_zyx.shape,
            )
            straight_zyx, untwist_meta, angles = untwist_volume_zyx_atlas(
                straight_zyx,
                atlas_zyx,
                angle_lim=angle_lim,
                smooth_window=smooth_window,
                subsample_z=subsample_z,
            )
        else:
            logger.info(
                'Untwisting via bilateral symmetry (lim=+/-%.0f deg, max_jump=%.0f)...',
                angle_lim,
                args.untwist_max_jump_deg,
            )
            straight_zyx, untwist_meta, angles = untwist_volume_zyx(
                straight_zyx,
                angle_lim=angle_lim,
                smooth_window=smooth_window,
                subsample_z=subsample_z,
                max_jump_deg=args.untwist_max_jump_deg,
                min_fg_frac=args.untwist_min_fg_frac,
            )
        np.save(straight_dir / 'untwist_angles_deg.npy', angles)
        (straight_dir / 'untwist_meta.json').write_text(json.dumps(untwist_meta, indent=2), encoding='utf-8')

    nib.save(nib.Nifti1Image(np.transpose(straight_zyx, (2, 1, 0)), affine), str(straight_nii))
    qc = straight_dir / 'qc'
    qc.mkdir(exist_ok=True)
    tag = 'untwist' if do_untwist else 'straight'
    cv2.imwrite(str(qc / f'{tag}_sagittal_mip.png'), _to_u8(straight_zyx.max(axis=2)))
    cv2.imwrite(str(qc / f'{tag}_coronal_mip.png'), _to_u8(straight_zyx.max(axis=1)))
    cv2.imwrite(str(qc / f'{tag}_zmid.png'), _to_u8(straight_zyx[straight_zyx.shape[0] // 2]))
    if do_untwist and (straight_dir / 'untwist_angles_deg.npy').exists():
        _write_angle_plot(qc / 'untwist_angles.png', np.load(straight_dir / 'untwist_angles_deg.npy'))
        if template_work is not None and untwist_method == 'atlas':
            atl_xyz = np.asanyarray(nib.load(str(template_work)).dataobj).astype(np.float32)
            atlas_zyx = np.transpose(atl_xyz, (2, 1, 0))
            z = straight_zyx.shape[0] // 2
            az = int(round(z / max(1, straight_zyx.shape[0] - 1) * (atlas_zyx.shape[0] - 1)))
            atl_sl = _center_match_yx(atlas_zyx[az], straight_zyx[z].shape)
            _write_overlay_png(qc / 'untwist_vs_atlas_zmid.png', straight_zyx[z], atl_sl)
    logger.info('Straightened%s ZYX=%s', f'+untwist({untwist_method})' if do_untwist else '', straight_zyx.shape)

    if args.skip_register:
        (out_dir / 'run_summary.json').write_text(
            json.dumps({'straighten': straight_meta, 'untwist': untwist_meta, 'registration': None}, indent=2),
            encoding='utf-8',
        )
        return

    assert template_work is not None and annotation_work is not None
    fixed_nii = nib.load(str(straight_nii))
    fixed_data = np.asanyarray(fixed_nii.dataobj).astype(np.float32)
    thr = float(np.percentile(fixed_data, 99) * 0.05)
    mask = (fixed_data > thr).astype(np.uint8)
    mask_path = straight_dir / 'straight_mask.nii.gz'
    nib.save(nib.Nifti1Image(mask, fixed_nii.affine), str(mask_path))

    summary = run_ants_registration(
        straight_nii,
        template_work,
        annotation_work,
        out_dir / 'ants_out',
        transform=args.transform,
        fixed_mask_nii=mask_path,
        allow_reflection=bool(args.allow_reflection),
        direction_y=float(getattr(args, 'direction_y', 1.0)),
    )
    (out_dir / 'run_summary.json').write_text(
        json.dumps(
            {'straighten': straight_meta, 'untwist': untwist_meta, 'registration': summary},
            indent=2,
        ),
        encoding='utf-8',
    )
    logger.info('Done. Summary: %s', out_dir / 'run_summary.json')


if __name__ == '__main__':
    main()
