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


def reject_centerline_jumps(pts: np.ndarray, max_jump_vox: float = 20.0) -> np.ndarray:
    """Median-filter y/x, then clamp outliers relative to filtered curve."""
    if len(pts) < 5:
        return pts.copy()
    out = pts.copy()
    # Odd window ~ local neighborhood along RC.
    w = 11
    for col in (1, 2):
        # Running median via scipy.
        from scipy.ndimage import median_filter

        med = median_filter(pts[:, col], size=w, mode="nearest")
        jump = np.abs(pts[:, col] - med)
        # Replace large deviations with median.
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
) -> None:
    qc = out_dir / "qc"
    qc.mkdir(parents=True, exist_ok=True)

    # Mid transverse: intensity + mask outline + centerline point
    zmid = int(vol_zyx.shape[0] // 2)
    base = cv2.cvtColor(_to_u8(vol_zyx[zmid]), cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask_zyx[zmid].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(base, contours, -1, (0, 255, 0), 1)
    # nearest centerline sample to zmid
    if len(centerline):
        i = int(np.argmin(np.abs(centerline[:, 0] - zmid)))
        _draw_cross(base, int(round(centerline[i, 1])), int(round(centerline[i, 2])), (0, 0, 255), r=4)
    cv2.imwrite(str(qc / "transverse_zmid_mask_centerline.png"), base)

    # A few more Z samples
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

    # Sagittal MIP (max over X) with centerline overlay
    mip_yx = vol_zyx.max(axis=2)  # Z,Y
    sag = cv2.cvtColor(_to_u8(mip_yx), cv2.COLOR_GRAY2BGR)
    # Draw centerline in (row=z, col=y)
    for z, y, _x in centerline:
        zz, yy = int(round(z)), int(round(y))
        if 0 <= zz < sag.shape[0] and 0 <= yy < sag.shape[1]:
            sag[zz, yy] = (0, 0, 255)
            if yy + 1 < sag.shape[1]:
                sag[zz, yy + 1] = (0, 0, 255)
    cv2.imwrite(str(qc / "sagittal_mip_centerline.png"), sag)

    # Coronal MIP (max over Y) with centerline
    mip_zx = vol_zyx.max(axis=1)  # Z,X
    cor = cv2.cvtColor(_to_u8(mip_zx), cv2.COLOR_GRAY2BGR)
    for z, _y, x in centerline:
        zz, xx = int(round(z)), int(round(x))
        if 0 <= zz < cor.shape[0] and 0 <= xx < cor.shape[1]:
            cor[zz, xx] = (0, 0, 255)
            if xx + 1 < cor.shape[1]:
                cor[zz, xx + 1] = (0, 0, 255)
    cv2.imwrite(str(qc / "coronal_mip_centerline.png"), cor)

    # Mask-only mid slices
    tifffile.imwrite(qc / "mask_zmid.tif", (mask_zyx[zmid].astype(np.uint8) * 255))
    tifffile.imwrite(qc / "intensity_zmid.tif", _to_u8(vol_zyx[zmid]))

    # Physical extent of mask bbox
    zs, ys, xs = np.where(mask_zyx)
    sz, sy, sx = spacing_zyx_um
    bbox_mm = {
        "z_mm": (float(zs.max() - zs.min() + 1) * sz) / 1000.0,
        "y_mm": (float(ys.max() - ys.min() + 1) * sy) / 1000.0,
        "x_mm": (float(xs.max() - xs.min() + 1) * sx) / 1000.0,
    }
    (qc / "README.txt").write_text(
        "\n".join(
            [
                "Green contour = tissue mask outline",
                "Red cross / red trail = centerline",
                "transverse_* = cross-section (atlas-like)",
                "sagittal_mip_centerline.png = max over X, centerline in red",
                "coronal_mip_centerline.png = max over Y, centerline in red",
                f"mask bbox approx mm (Z,Y,X / RC,DV,ML): {bbox_mm}",
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
        choices=("intensity", "distance"),
        default="intensity",
        help="intensity=weighted centroid in mask (default); distance=EDT peak",
    )
    p.add_argument(
        "--max_centerline_jump",
        type=float,
        default=15.0,
        help="Clamp centerline deviations from local median larger than this (voxels)",
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

    raw_cl = extract_centerline_zyx(mask_zyx, vol_zyx, mode=args.centerline_mode)
    cleaned_cl = reject_centerline_jumps(raw_cl, max_jump_vox=args.max_centerline_jump)
    sm_cl = smooth_centerline(cleaned_cl, window=args.smooth_window)
    logger.info("Centerline mode=%s points=%d (raw=%d)", args.centerline_mode, len(sm_cl), len(raw_cl))

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
        f.write("z_vox,y_vox,x_vox,z_mm,y_mm,x_mm\n")
        for z, y, x in sm_cl:
            f.write(
                f"{z:.3f},{y:.3f},{x:.3f},"
                f"{z * spacing_zyx[0] / 1000.0:.5f},"
                f"{y * spacing_zyx[1] / 1000.0:.5f},"
                f"{x * spacing_zyx[2] / 1000.0:.5f}\n"
            )

    write_qc(out_dir, vol_zyx, mask_zyx, sm_cl, spacing_zyx)

    summary = {
        "volume_nii": str(args.volume_nii),
        "shape_zyx": list(vol_zyx.shape),
        "spacing_xyz_um": list(spacing_xyz),
        "mask": mask_meta,
        "centerline_n": int(len(sm_cl)),
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
