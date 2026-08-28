#!/usr/bin/env python3
"""Manual in-plane untwist UI for straightened spinal cord volumes.

Linearly samples ~1% of transverse slices (skipping Z ends), lets you set
rotation angles, polynomial-interpolates along Z, then optionally runs 2D XY
centroid alignment. From the same panel you can run formal atlas→image ANTs
registration and preview QC overlays.

Example:
  python -m pipeline_modules.registration.spinalj_manual_untwist_ui --volume_nii ".../volume_straight.nii.gz" --out_dir ".../manual_untwist"
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def run_formal_register_job(
    *,
    fixed_nii: Path,
    atlas_dir: Path,
    out_dir: Path,
    transform: str = "SyNRA",
    allow_reflection: bool = False,
    direction_y: float = 1.0,
    ants_out_name: str = "ants_out",
    landmarks_json: Path | None = None,
    landmark_transform_type: str = "similarity",
) -> dict:
    """CLI/worker entry: atlas prep + ANTs registration (no GUI)."""
    import nibabel as nib

    from pipeline_modules.registration.spinalj_atlas_register import (
        convert_atlas_to_nifti,
        run_ants_registration,
    )
    from pipeline_modules.registration.spinalj_straighten_register import (
        _resample_atlas_to_spacing,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ants_out = out_dir / ants_out_name
    fixed_nii = Path(fixed_nii)
    atlas_dir = Path(atlas_dir)
    if landmarks_json is None:
        guess = out_dir / "landmarks" / "landmarks.json"
        landmarks_json = guess if guess.exists() else None
    else:
        landmarks_json = Path(landmarks_json)

    t0 = time.time()
    logger.info(
        "Register job start: fixed=%s transform=%s direction_y=%s landmarks=%s out=%s",
        fixed_nii,
        transform,
        direction_y,
        landmarks_json,
        ants_out,
    )
    atlas_nifti = convert_atlas_to_nifti(atlas_dir, out_dir / "atlas_nifti")
    aff = nib.load(str(fixed_nii)).affine
    spacing_xyz = tuple(float(abs(aff[i, i])) for i in range(3))
    work = out_dir / "reg_work"
    work.mkdir(parents=True, exist_ok=True)
    logger.info("Resampling atlas to spacing_xyz=%s ...", spacing_xyz)
    template_work = _resample_atlas_to_spacing(
        atlas_nifti["template"],
        work / "Template_work.nii.gz",
        spacing_xyz=spacing_xyz,
        is_label=False,
    )
    annotation_work = _resample_atlas_to_spacing(
        atlas_nifti["annotation"],
        work / "Annotation_work.nii.gz",
        spacing_xyz=spacing_xyz,
        is_label=True,
    )

    fixed_img = nib.load(str(fixed_nii))
    fixed_data = np.asanyarray(fixed_img.dataobj).astype(np.float32)
    thr = float(np.percentile(fixed_data, 99) * 0.05)
    mask = (fixed_data > thr).astype(np.uint8)
    mask_path = out_dir / "straight_mask.nii.gz"
    nib.save(nib.Nifti1Image(mask, fixed_img.affine), str(mask_path))
    logger.info(
        "Fixed shape XYZ=%s fg_frac=%.3f — starting ANTs %s (UI stays responsive in parent)",
        fixed_data.shape,
        float(mask.mean()),
        transform,
    )

    summary = run_ants_registration(
        fixed_nii,
        template_work,
        annotation_work,
        ants_out,
        transform=transform,
        fixed_mask_nii=mask_path,
        allow_reflection=allow_reflection,
        direction_y=direction_y,
        landmarks_json=landmarks_json,
        landmark_transform_type=landmark_transform_type,
    )
    payload = {
        "fixed": str(fixed_nii),
        "transform": transform,
        "allow_reflection": bool(allow_reflection),
        "direction_y": float(direction_y),
        "landmarks_json": str(landmarks_json) if landmarks_json else None,
        "landmark_transform_type": landmark_transform_type if landmarks_json else None,
        "ants_out": str(ants_out),
        "elapsed_sec": float(time.time() - t0),
        "registration": summary,
    }
    summary_name = (
        "registration_run_summary.json"
        if ants_out_name == "ants_out"
        else f"registration_run_summary_{ants_out_name}.json"
    )
    (out_dir / summary_name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Register job done in %.1f min → %s", (time.time() - t0) / 60.0, ants_out)
    return payload


def _to_u8(arr: np.ndarray) -> np.ndarray:
    """Legacy alias."""
    return _auto_contrast_u8(arr)


def _auto_contrast_u8(arr: np.ndarray, *, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    """Per-image auto-contrast → uint8 (ignore zeros as background)."""
    arr = np.asarray(arr, dtype=np.float32)
    fg = arr[arr > 0]
    if fg.size < 16:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(fg, (p_lo, p_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(fg.min()), float(fg.max())
        if hi <= lo:
            return np.zeros(arr.shape, dtype=np.uint8)
    out = np.zeros(arr.shape, dtype=np.float32)
    m = arr > 0
    out[m] = np.clip((arr[m] - lo) / (hi - lo), 0, 1) * 255.0
    u8 = out.astype(np.uint8)
    # Mild CLAHE so sparse C1 signal is easier to judge for LR / DV.
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        u8 = clahe.apply(u8)
        # Keep background black.
        u8 = np.where(m, u8, 0).astype(np.uint8)
    except Exception:
        pass
    return u8


def _rotate_slice(img: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(float(angle_deg)) < 1e-6:
        return np.asarray(img)
    h, w = img.shape[:2]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), float(angle_deg), 1.0)
    return cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _sample_z_indices(n_z: int, frac: float, *, end_margin_frac: float = 0.05) -> np.ndarray:
    """Linearly spaced Z keyframes, skipping end margins (never starts at z=0).

    Samples about ``frac * n_z`` indices inside
    ``[end_margin_frac, 1-end_margin_frac]`` of the Z span.
    """
    n = max(3, int(round(max(frac, 1e-6) * n_z)))
    n = min(n, max(3, n_z - 2))
    m = float(np.clip(end_margin_frac, 0.0, 0.45))
    z0 = int(np.floor(m * (n_z - 1)))
    z1 = int(np.ceil((1.0 - m) * (n_z - 1)))
    z0 = max(1, z0)  # never use z=0
    z1 = min(n_z - 2, z1)  # avoid last slice too
    if z1 <= z0:
        z0, z1 = max(1, n_z // 4), min(n_z - 2, (3 * n_z) // 4)
    if n == 1:
        return np.asarray([(z0 + z1) // 2], dtype=int)
    return np.unique(np.round(np.linspace(z0, z1, n)).astype(int))


def _poly_interp_angles(z_key: np.ndarray, a_key: np.ndarray, n_z: int, degree: int) -> np.ndarray:
    z_key = np.asarray(z_key, dtype=np.float64)
    a_key = np.asarray(a_key, dtype=np.float64)
    # Unwrap large jumps before fitting (180° period for LR-ish orientations).
    a = a_key.copy()
    for i in range(1, len(a)):
        while a[i] - a[i - 1] > 90:
            a[i] -= 180
        while a[i] - a[i - 1] < -90:
            a[i] += 180
    deg = int(max(1, min(degree, len(a) - 1)))
    coef = np.polyfit(z_key, a, deg)
    z_all = np.arange(n_z, dtype=np.float64)
    return np.polyval(coef, z_all)


def _apply_angles_zyx(vol_zyx: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    out = np.zeros_like(vol_zyx)
    for z in range(vol_zyx.shape[0]):
        out[z] = _rotate_slice(vol_zyx[z], float(angles_deg[z]))
        if (z + 1) % 400 == 0 or z + 1 == vol_zyx.shape[0]:
            logger.info("  apply %d/%d", z + 1, vol_zyx.shape[0])
    return out


def _slice_centroid_yx(img: np.ndarray, *, thr_frac: float = 0.05) -> tuple[float, float] | None:
    """Intensity-weighted (y, x) centroid of foreground; None if empty."""
    img = np.asarray(img, dtype=np.float32)
    pos = img[img > 0]
    if pos.size < 32:
        return None
    thr = float(np.percentile(pos, 99) * thr_frac)
    w = np.where(img > thr, img, 0.0)
    wsum = float(w.sum())
    if wsum <= 0:
        return None
    yy, xx = np.indices(img.shape, dtype=np.float64)
    y = float((w * yy).sum() / wsum)
    x = float((w * xx).sum() / wsum)
    return y, x


def _smooth_1d(x: np.ndarray, window: int) -> np.ndarray:
    w = max(3, int(window) | 1)
    kernel = np.ones(w, dtype=np.float64) / w
    pad = w // 2
    return np.convolve(np.pad(x, (pad, pad), mode="edge"), kernel, mode="valid")


def _apply_xy_shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return np.asarray(img)
    h, w = img.shape[:2]
    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def align_stack_xy(
    vol_zyx: np.ndarray,
    *,
    max_shift_vox: float = 24.0,
    smooth_window: int = 51,
) -> tuple[np.ndarray, dict]:
    """Stable 2D XY alignment after rotation via smoothed centroid shifts.

    Phase-correlation neighbor chaining is avoided (it zig-zags on sparse C1).
    Each slice is translated so its intensity centroid matches the mid-Z band
    centroid; dx/dy are spike-rejected then heavily smoothed along Z.
    """
    n = vol_zyx.shape[0]
    cy = np.full(n, np.nan, dtype=np.float64)
    cx = np.full(n, np.nan, dtype=np.float64)
    for z in range(n):
        c = _slice_centroid_yx(vol_zyx[z])
        if c is not None:
            cy[z], cx[z] = c
        if (z + 1) % 400 == 0:
            logger.info("  centroid %d/%d", z + 1, n)

    z_all = np.arange(n, dtype=np.float64)
    good = np.isfinite(cy) & np.isfinite(cx)
    if int(good.sum()) < 8:
        raise RuntimeError("Too few valid slice centroids for XY align")
    cy[~good] = np.interp(z_all[~good], z_all[good], cy[good])
    cx[~good] = np.interp(z_all[~good], z_all[good], cx[good])

    # Anchor: robust mid-band centroid (not a single noisy mid slice).
    z0, z1 = int(0.4 * (n - 1)), int(0.6 * (n - 1))
    ty = float(np.median(cy[z0 : z1 + 1]))
    tx = float(np.median(cx[z0 : z1 + 1]))

    dy = ty - cy
    dx = tx - cx
    dy = np.clip(dy, -max_shift_vox, max_shift_vox)
    dx = np.clip(dx, -max_shift_vox, max_shift_vox)

    w = max(5, int(smooth_window) | 1)
    half = w // 2
    for arr in (dx, dy):
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            med = float(np.median(arr[lo:hi]))
            if abs(arr[i] - med) > max_shift_vox * 0.5:
                arr[i] = med
    dx = _smooth_1d(dx, w)
    dy = _smooth_1d(dy, w)
    dx = np.clip(dx, -max_shift_vox, max_shift_vox)
    dy = np.clip(dy, -max_shift_vox, max_shift_vox)

    out = np.zeros_like(vol_zyx)
    for z in range(n):
        out[z] = _apply_xy_shift(vol_zyx[z], float(dx[z]), float(dy[z]))
        if (z + 1) % 400 == 0 or z + 1 == n:
            logger.info("  apply XY %d/%d  shift=(%.2f, %.2f)", z + 1, n, dx[z], dy[z])

    mag = np.hypot(dx, dy)
    meta = {
        "method": "xy_centroid_to_midband_smooth",
        "max_shift_vox": float(max_shift_vox),
        "smooth_window": int(w),
        "anchor_yx": [ty, tx],
        "anchor_z_band": [z0, z1],
        "shift_vox_mean": float(mag.mean()),
        "shift_vox_p95": float(np.percentile(mag, 95)),
        "shift_vox_max": float(mag.max()),
        "n_valid_centroids": int(good.sum()),
    }
    logger.info(
        "XY align (centroid+smooth): mean|shift|=%.2f vox  p95=%.2f  max=%.2f",
        meta["shift_vox_mean"],
        meta["shift_vox_p95"],
        meta["shift_vox_max"],
    )
    return out, meta


class ManualUntwistUI:
    def __init__(
        self,
        vol_zyx: np.ndarray,
        z_keys: np.ndarray,
        *,
        out_dir: Path,
        affine: np.ndarray,
        poly_degree: int = 3,
        angle_lim: float = 180.0,
        session_path: Path | None = None,
        atlas_dir: Path | None = None,
        transform: str = "SyNRA",
        allow_reflection: bool = False,
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider

        self.vol_zyx = vol_zyx
        self.z_keys = np.asarray(z_keys, dtype=int)
        self.angles = np.zeros(len(self.z_keys), dtype=np.float64)
        self.idx = 0
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.affine = affine
        self.poly_degree = int(poly_degree)
        self.angle_lim = float(angle_lim)
        self.session_path = session_path or (self.out_dir / "manual_keypoints.json")
        self.atlas_dir = Path(atlas_dir) if atlas_dir else None
        self.transform = transform
        self.allow_reflection = bool(allow_reflection)
        self.untwist_nii = self.out_dir / "volume_straight_manual_untwist.nii.gz"
        self.aligned_nii = self.out_dir / "volume_straight_manual_untwist_xyalign.nii.gz"
        self.ants_out_dir = self.out_dir / "ants_out"
        self.max_xy_shift = 24.0
        self.xy_smooth_window = 51
        self._reg_busy = False
        self._reg_proc: subprocess.Popen | None = None
        self._reg_log_path = self.out_dir / "register_job.log"
        self._reg_timer = None
        self._reg_t0 = 0.0

        if self.session_path.exists():
            self._load_session()

        self.fig, (self.ax_img, self.ax_ang) = plt.subplots(
            1, 2, figsize=(13, 7.2), gridspec_kw={"width_ratios": [1.2, 1.0]}
        )
        plt.subplots_adjust(bottom=0.30, wspace=0.25)
        self.ax_img.set_title("rotated slice")
        self.ax_img.axis("off")
        self.im = self.ax_img.imshow(np.zeros((10, 10), dtype=np.uint8), cmap="gray", vmin=0, vmax=255)
        (self.line_keys,) = self.ax_ang.plot([], [], "o-", color="C0", label="keyframes")
        (self.line_poly,) = self.ax_ang.plot([], [], "-", color="C1", alpha=0.8, label="poly")
        (self.mark,) = self.ax_ang.plot([], [], "ro", markersize=8)
        self.ax_ang.set_xlabel("Z index")
        self.ax_ang.set_ylabel("angle (deg)")
        self.ax_ang.grid(True, alpha=0.3)
        self.ax_ang.legend(loc="best", fontsize=8)
        self.status = self.fig.text(0.02, 0.005, "", fontsize=9)

        ax_slider = plt.axes([0.12, 0.18, 0.50, 0.035])
        self.slider = Slider(
            ax_slider, "angle", -self.angle_lim, self.angle_lim, valinit=float(self.angles[0]), valstep=0.5
        )
        self.slider.on_changed(self._on_slider)

        # Row 1: untwist / XY prep
        self.btn_prev = Button(plt.axes([0.12, 0.105, 0.07, 0.04]), "Prev")
        self.btn_next = Button(plt.axes([0.20, 0.105, 0.07, 0.04]), "Next")
        self.btn_save = Button(plt.axes([0.29, 0.105, 0.09, 0.04]), "Save keys")
        self.btn_apply = Button(plt.axes([0.39, 0.105, 0.11, 0.04]), "Apply poly")
        self.btn_xy = Button(plt.axes([0.51, 0.105, 0.12, 0.04]), "Align XY")
        self.btn_zero = Button(plt.axes([0.65, 0.105, 0.07, 0.04]), "Zero")
        # Row 2: formal atlas registration + QC preview
        self.btn_ants = Button(plt.axes([0.12, 0.045, 0.18, 0.04]), f"Reg {self.transform}")
        self.btn_preview = Button(plt.axes([0.32, 0.045, 0.16, 0.04]), "Preview QC")
        self.btn_prev.on_clicked(lambda _e: self._step(-1))
        self.btn_next.on_clicked(lambda _e: self._step(+1))
        self.btn_save.on_clicked(lambda _e: self._save_session())
        self.btn_apply.on_clicked(lambda _e: self._apply_and_write())
        self.btn_xy.on_clicked(lambda _e: self._align_xy())
        self.btn_zero.on_clicked(lambda _e: self._set_angle(0.0))
        self.btn_ants.on_clicked(lambda _e: self._run_formal_register())
        self.btn_preview.on_clicked(lambda _e: self._preview_registration_qc())

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._refresh()
        self.fig.suptitle(
            "Untwist → Apply poly → Align XY → Register (atlas→image ANTs).  "
            "Keys: ←/→, [ ] ±1°, { } ±5°, s save, a apply, x Align XY, g Register, v Preview QC",
            fontsize=10,
        )

    def _load_session(self) -> None:
        data = json.loads(self.session_path.read_text(encoding="utf-8"))
        z = np.asarray(data["z_vox"], dtype=int)
        a = np.asarray(data["angle_deg"], dtype=float)
        if len(z) != len(self.z_keys) or not np.array_equal(z, self.z_keys):
            # Map by nearest Z if sampling changed.
            for i, zi in enumerate(self.z_keys):
                j = int(np.argmin(np.abs(z - zi)))
                self.angles[i] = float(a[j])
            logger.info("Loaded session with Z remap from %s", self.session_path)
        else:
            self.angles = a.astype(np.float64)
            logger.info("Loaded session %s", self.session_path)

    def _save_session(self) -> None:
        payload = {
            "z_vox": self.z_keys.tolist(),
            "angle_deg": self.angles.tolist(),
            "poly_degree": self.poly_degree,
            "n_volume_z": int(self.vol_zyx.shape[0]),
            "frac_hint": len(self.z_keys) / max(1, self.vol_zyx.shape[0]),
        }
        self.session_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        # Angle curve preview
        ang_full = _poly_interp_angles(self.z_keys, self.angles, self.vol_zyx.shape[0], self.poly_degree)
        np.save(self.out_dir / "manual_angles_interp_preview.npy", ang_full)
        self.status.set_text(f"Saved keypoints → {self.session_path}")
        self.fig.canvas.draw_idle()
        logger.info("Saved %s", self.session_path)

    def _set_angle(self, a: float) -> None:
        a = float(np.clip(a, -self.angle_lim, self.angle_lim))
        self.angles[self.idx] = a
        if abs(self.slider.val - a) > 1e-6:
            self.slider.set_val(a)
        else:
            self._refresh()

    def _on_slider(self, val: float) -> None:
        self.angles[self.idx] = float(val)
        self._refresh(update_slider=False)

    def _step(self, delta: int) -> None:
        self.idx = int(np.clip(self.idx + delta, 0, len(self.z_keys) - 1))
        self.slider.set_val(float(self.angles[self.idx]))
        self._refresh(update_slider=False)

    def _on_key(self, event) -> None:
        if event.key in ("right", "n"):
            self._step(+1)
        elif event.key in ("left", "p"):
            self._step(-1)
        elif event.key == "]":
            self._set_angle(self.angles[self.idx] + 1.0)
        elif event.key == "[":
            self._set_angle(self.angles[self.idx] - 1.0)
        elif event.key == "}":
            self._set_angle(self.angles[self.idx] + 5.0)
        elif event.key == "{":
            self._set_angle(self.angles[self.idx] - 5.0)
        elif event.key == "s":
            self._save_session()
        elif event.key == "a":
            self._apply_and_write()
        elif event.key == "x":
            self._align_xy()
        elif event.key == "g":
            self._run_formal_register()
        elif event.key == "v":
            self._preview_registration_qc()
        elif event.key == "0":
            self._set_angle(0.0)

    def _refresh(self, update_slider: bool = True) -> None:
        z = int(self.z_keys[self.idx])
        a = float(self.angles[self.idx])
        if update_slider and abs(self.slider.val - a) > 1e-6:
            self.slider.set_val(a)
            return
        rot = _rotate_slice(self.vol_zyx[z], a)
        # Auto-contrast every displayed slice (independent of other Z).
        u8 = _auto_contrast_u8(rot)
        rgb = np.stack([u8, u8, u8], axis=-1)
        h, w = u8.shape
        mid = w // 2
        rgb[:, mid, :] = (0, 255, 255)
        # Magenta/green LR check strips
        left = u8[:, :mid]
        right = np.fliplr(u8[:, w - mid :])
        m = min(left.shape[1], right.shape[1])
        if m > 0:
            diff = np.abs(left[:, -m:].astype(np.float32) - right[:, :m].astype(np.float32))
            d = (np.clip(diff / (diff.max() + 1e-6), 0, 1) * 255).astype(np.uint8)
            # Put |L-R| as red overlay on left half edge.
            rgb[:, mid - m : mid, 0] = np.maximum(rgb[:, mid - m : mid, 0], d)
        self.im.set_data(rgb)
        self.im.set_extent((-0.5, w - 0.5, h - 0.5, -0.5))
        self.ax_img.set_title(f"key {self.idx + 1}/{len(self.z_keys)}  z={z}  angle={a:.1f}°")

        ang_full = _poly_interp_angles(self.z_keys, self.angles, self.vol_zyx.shape[0], self.poly_degree)
        self.line_keys.set_data(self.z_keys, self.angles)
        self.line_poly.set_data(np.arange(len(ang_full)), ang_full)
        self.mark.set_data([z], [a])
        self.ax_ang.relim()
        self.ax_ang.autoscale_view()
        self.status.set_text(
            f"poly degree={self.poly_degree} | red=|L-R| hint near midline | cyan=midline | "
            f"session={self.session_path.name}"
        )
        self.fig.canvas.draw_idle()

    def _apply_and_write(self) -> None:
        import nibabel as nib

        self._save_session()
        n_z = self.vol_zyx.shape[0]
        ang = _poly_interp_angles(self.z_keys, self.angles, n_z, self.poly_degree)
        logger.info("Applying poly degree=%d to %d slices...", self.poly_degree, n_z)
        out = _apply_angles_zyx(self.vol_zyx, ang)
        out_nii = self.out_dir / "volume_straight_manual_untwist.nii.gz"
        nib.save(nib.Nifti1Image(np.transpose(out, (2, 1, 0)), self.affine), str(out_nii))
        np.save(self.out_dir / "untwist_angles_deg.npy", ang)
        meta = {
            "method": "manual_poly",
            "poly_degree": self.poly_degree,
            "n_keyframes": int(len(self.z_keys)),
            "z_keys": self.z_keys.tolist(),
            "angles_key_deg": self.angles.tolist(),
            "angle_deg_p01": float(np.percentile(ang, 1)),
            "angle_deg_p50": float(np.percentile(ang, 50)),
            "angle_deg_p99": float(np.percentile(ang, 99)),
            "angle_deg_abs_mean": float(np.mean(np.abs(ang))),
            "volume_out": str(out_nii),
        }
        (self.out_dir / "untwist_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        # QC
        qc = self.out_dir / "qc"
        qc.mkdir(exist_ok=True)
        cv2.imwrite(str(qc / "untwist_zmid.png"), _to_u8(out[out.shape[0] // 2]))
        cv2.imwrite(str(qc / "untwist_sagittal_mip.png"), _to_u8(out.max(axis=2)))
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(self.z_keys, self.angles, "o", label="keys")
        ax.plot(ang, "-", label=f"poly deg={self.poly_degree}")
        ax.set_xlabel("Z")
        ax.set_ylabel("angle (deg)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(qc / "untwist_angles.png", dpi=120)
        plt.close(fig)
        self.status.set_text(f"Wrote {out_nii.name}. Next: Align XY (fix neighbor mismatch).")
        self.fig.canvas.draw_idle()
        logger.info("Done. Outputs in %s", self.out_dir)

    def _align_xy(self) -> None:
        """After rotation: 2D XY translation registration between neighboring slices."""
        import nibabel as nib

        if not self.untwist_nii.exists():
            self.status.set_text("Applying poly first...")
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self._apply_and_write()

        self.status.set_text("Aligning XY (centroid + Z-smooth, no phase zig-zag)...")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        try:
            vol_xyz = np.asanyarray(nib.load(str(self.untwist_nii)).dataobj).astype(np.float32)
            vol_zyx = np.transpose(vol_xyz, (2, 1, 0))
            aligned, meta = align_stack_xy(
                vol_zyx,
                max_shift_vox=self.max_xy_shift,
                smooth_window=self.xy_smooth_window,
            )
            nib.save(
                nib.Nifti1Image(np.transpose(aligned, (2, 1, 0)), self.affine),
                str(self.aligned_nii),
            )
            (self.out_dir / "xy_align_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            qc = self.out_dir / "qc"
            qc.mkdir(exist_ok=True)
            cv2.imwrite(str(qc / "xyalign_zmid.png"), _auto_contrast_u8(aligned[aligned.shape[0] // 2]))
            cv2.imwrite(str(qc / "xyalign_sagittal_mip.png"), _auto_contrast_u8(aligned.max(axis=2)))
            self.status.set_text(
                f"XY align done → {self.aligned_nii.name}  "
                f"(mean|shift|={meta['shift_vox_mean']:.1f}vox)"
            )
            logger.info("Wrote %s", self.aligned_nii)
        except Exception as e:
            logger.exception("XY align failed")
            self.status.set_text(f"XY align failed: {e}")
        self.fig.canvas.draw_idle()

    def _resolve_fixed_nii(self) -> Path:
        """Prefer XY-aligned volume, else untwisted; raise if neither exists."""
        if self.aligned_nii.exists():
            return self.aligned_nii
        if self.untwist_nii.exists():
            return self.untwist_nii
        raise FileNotFoundError("Run Apply poly (and optionally Align XY) first")

    def _run_formal_register(self) -> None:
        """Launch atlas→image ANTs in a subprocess so the UI stays responsive."""
        if self._reg_busy and self._reg_proc is not None and self._reg_proc.poll() is None:
            elapsed = (time.time() - self._reg_t0) / 60.0
            self.status.set_text(
                f"Register still running ({self.transform}, {elapsed:.1f} min) — UI OK; wait or Preview later"
            )
            self.fig.canvas.draw_idle()
            return

        try:
            fixed_nii = self._resolve_fixed_nii()
        except FileNotFoundError as e:
            self.status.set_text(str(e))
            self.fig.canvas.draw_idle()
            return
        if self.atlas_dir is None or not self.atlas_dir.exists():
            self.status.set_text(f"Atlas dir missing: {self.atlas_dir}")
            self.fig.canvas.draw_idle()
            return

        # Avoid clobbering an in-GUI SyN that was started before backgrounding existed.
        self._reg_log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(self._reg_log_path, "w", encoding="utf-8")
        cmd = [
            sys.executable,
            "-m",
            "pipeline_modules.registration.spinalj_manual_untwist_ui",
            "--register_only",
            "--fixed_nii",
            str(fixed_nii),
            "--out_dir",
            str(self.out_dir),
            "--atlas_dir",
            str(self.atlas_dir),
            "--transform",
            str(self.transform),
        ]
        if self.allow_reflection:
            cmd.append("--allow_reflection")
        logger.info("Spawning register subprocess: %s", " ".join(cmd))
        self._reg_t0 = time.time()
        self._reg_proc = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).resolve().parents[2]),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._reg_busy = True
        self.status.set_text(
            f"Register started in background (pid={self._reg_proc.pid}, {self.transform}=Rigid+Affine+SyN). "
            f"Large volume: often 20–60+ min. UI stays usable. Log: {self._reg_log_path.name}"
        )
        self.fig.canvas.draw_idle()
        if self._reg_timer is None:
            self._reg_timer = self.fig.canvas.new_timer(interval=5000)
            self._reg_timer.add_callback(self._poll_register_job)
            self._reg_timer.start()

    def _poll_register_job(self) -> None:
        if self._reg_proc is None:
            return
        rc = self._reg_proc.poll()
        elapsed = (time.time() - self._reg_t0) / 60.0
        if rc is None:
            # Tail last log line for a lightweight heartbeat.
            tail = ""
            try:
                lines = self._reg_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                if lines:
                    tail = lines[-1][-80:]
            except Exception:
                pass
            self.status.set_text(
                f"Register running… {elapsed:.1f} min (pid={self._reg_proc.pid})  {tail}"
            )
            self.fig.canvas.draw_idle()
            return

        self._reg_busy = False
        if self._reg_timer is not None:
            self._reg_timer.stop()
            self._reg_timer = None
        if rc == 0 and (self.ants_out_dir / "qc_slices" / "overlay_zmid.png").exists():
            self.status.set_text(
                f"Register done in {elapsed:.1f} min → {self.ants_out_dir.name}/  (press Preview QC)"
            )
        else:
            self.status.set_text(
                f"Register exited rc={rc} after {elapsed:.1f} min — see {self._reg_log_path.name}"
            )
        self._reg_proc = None
        self.fig.canvas.draw_idle()

    def _preview_registration_qc(self) -> None:
        """Show magenta/green overlay QC from the last formal registration."""
        import matplotlib.pyplot as plt

        qc = self.ants_out_dir / "qc_slices"
        overlays = [
            ("mid-Z overlay (magenta=sample, green=atlas)", qc / "overlay_zmid.png"),
            ("mid-Y overlay", qc / "overlay_ymid.png"),
        ]
        missing = [str(p) for _, p in overlays if not p.exists()]
        if missing:
            self.status.set_text("No QC yet — run Register first")
            self.fig.canvas.draw_idle()
            return

        # Optional quick NCC on mid-Z for a one-line quality cue.
        ncc_txt = ""
        try:
            import nibabel as nib

            fixed_nii = self._resolve_fixed_nii()
            warped = self.ants_out_dir / "warped_template.nii.gz"
            if warped.exists():
                f = np.asanyarray(nib.load(str(fixed_nii)).dataobj).astype(np.float32)
                w = np.asanyarray(nib.load(str(warped)).dataobj).astype(np.float32)
                # NIfTI XYZ → take mid-Z as axis 2
                fz = f[:, :, f.shape[2] // 2]
                wz = w[:, :, w.shape[2] // 2]
                fy = f[:, f.shape[1] // 2, :]
                wy = w[:, w.shape[1] // 2, :]

                def _ncc(a: np.ndarray, b: np.ndarray) -> float:
                    a = a.ravel().astype(np.float64)
                    b = b.ravel().astype(np.float64)
                    a = a - a.mean()
                    b = b - b.mean()
                    d = np.linalg.norm(a) * np.linalg.norm(b)
                    return float((a @ b) / d) if d > 0 else 0.0

                ncc_txt = f"  NCC midZ={_ncc(fz, wz):.3f}  midY={_ncc(fy, wy):.3f}"
        except Exception:
            ncc_txt = ""

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle(f"Registration QC ({self.transform}){ncc_txt}", fontsize=11)
        for ax, (title, path) in zip(axes, overlays):
            rgb = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.tight_layout()
        self.status.set_text(f"Preview QC ← {qc}{ncc_txt}")
        self.fig.canvas.draw_idle()
        fig.show()
        fig.canvas.draw_idle()

    def show(self) -> None:
        import matplotlib.pyplot as plt

        plt.show()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--volume_nii", default="", help="Straightened volume NIfTI (XYZ); required for UI mode")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--sample_frac", type=float, default=0.01, help="Fraction of Z slices as keyframes (default 0.01)")
    p.add_argument(
        "--end_margin_frac",
        type=float,
        default=0.05,
        help="Skip this fraction at each Z end when sampling keyframes (default 0.05; never uses z=0)",
    )
    p.add_argument("--poly_degree", type=int, default=3)
    p.add_argument("--angle_lim", type=float, default=180.0)
    p.add_argument("--session_json", default="", help="Optional existing manual_keypoints.json to resume")
    p.add_argument(
        "--atlas_dir",
        default=r"S:\Yifu_data\reference\SC_P56_Atlas_10x10x20_v5_2020",
        help="SpinalJ atlas dir for Register button",
    )
    p.add_argument(
        "--transform",
        default="SyNRA",
        help="ANTs type_of_transform for Register: SyNRA (Rigid+Affine+SyN, default), SyN, Affine, ...",
    )
    p.add_argument(
        "--allow_reflection",
        action="store_true",
        help="Allow affine reflection if sample appears left-right mirrored vs atlas",
    )
    p.add_argument(
        "--direction_y",
        type=float,
        default=1.0,
        help="Fixed-image ANTs direction YY (default +1 = no Y flip; use -1 to flip Y). Atlas stays +1.",
    )
    p.add_argument(
        "--ants_out_name",
        default="ants_out",
        help="Subfolder under out_dir for ANTs outputs (default ants_out)",
    )
    p.add_argument(
        "--reset_session",
        action="store_true",
        help="Ignore existing manual_keypoints.json (use when Z sampling changed)",
    )
    p.add_argument(
        "--register_only",
        action="store_true",
        help="Headless ANTs job (used by Register button subprocess); needs --fixed_nii",
    )
    p.add_argument("--fixed_nii", default="", help="Fixed volume for --register_only")
    p.add_argument(
        "--landmarks_json",
        default="",
        help="Landmark UI JSON (sample/atlas pairs). Default: <out_dir>/landmarks/landmarks.json if present",
    )
    p.add_argument(
        "--landmark_transform_type",
        default="similarity",
        choices=("rigid", "similarity", "affine"),
        help="Linear transform fitted to landmark pairs before ANTs (default similarity)",
    )
    args = p.parse_args()

    if args.register_only:
        if not args.fixed_nii:
            raise SystemExit("--register_only requires --fixed_nii")
        run_formal_register_job(
            fixed_nii=Path(args.fixed_nii),
            atlas_dir=Path(args.atlas_dir),
            out_dir=Path(args.out_dir),
            transform=args.transform,
            allow_reflection=bool(args.allow_reflection),
            direction_y=float(args.direction_y),
            ants_out_name=str(args.ants_out_name),
            landmarks_json=Path(args.landmarks_json) if args.landmarks_json else None,
            landmark_transform_type=str(args.landmark_transform_type),
        )
        return

    if not args.volume_nii:
        raise SystemExit("--volume_nii is required for UI mode")

    import matplotlib

    # Interactive backend for desktop.
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
    import nibabel as nib

    nii = nib.load(str(args.volume_nii))
    vol_xyz = np.asanyarray(nii.dataobj).astype(np.float32)
    vol_zyx = np.transpose(vol_xyz, (2, 1, 0))
    z_keys = _sample_z_indices(vol_zyx.shape[0], args.sample_frac, end_margin_frac=args.end_margin_frac)
    logger.info(
        "Volume ZYX=%s  keyframes=%d (%.2f%%)  z=%s...%s  (skipped end margins)",
        vol_zyx.shape,
        len(z_keys),
        100.0 * len(z_keys) / vol_zyx.shape[0],
        z_keys[:3],
        z_keys[-3:],
    )
    out_dir = Path(args.out_dir)
    session = Path(args.session_json) if args.session_json else (out_dir / "manual_keypoints.json")
    if args.reset_session and session.exists():
        session.unlink()
        logger.info("Removed old session %s", session)
    ui = ManualUntwistUI(
        vol_zyx,
        z_keys,
        out_dir=out_dir,
        affine=nii.affine,
        poly_degree=args.poly_degree,
        angle_lim=args.angle_lim,
        session_path=session if session.exists() else None,
        atlas_dir=Path(args.atlas_dir) if args.atlas_dir else None,
        transform=args.transform,
        allow_reflection=bool(args.allow_reflection),
    )
    ui.show()


if __name__ == "__main__":
    main()
