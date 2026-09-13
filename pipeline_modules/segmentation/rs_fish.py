"""RS-FISH-style 3D spot detection in Python (no Fiji/Java).

Follows the Radial Symmetry-FISH pipeline (Bahry et al., Nature Methods 2022):
  1. percentile-normalize intensity
  2. difference-of-Gaussians (DoG) predetection
  3. local maxima
  4. radial-symmetry localization (Parthasarathy 2012) in a local patch
  5. gradient inlier-ratio filter (lightweight stand-in for RANSAC)
  6. intensity threshold on the original image

Uses SciPy / scikit-image only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from tqdm import tqdm

try:
    import tifffile
except ImportError as exc:  # pragma: no cover
    raise ModuleNotFoundError("tifffile is required") from exc


def normalize_percentiles(volume: np.ndarray, low: float = 1.0, high: float = 99.5) -> np.ndarray:
    x = volume.astype(np.float32, copy=False)
    p1, p2 = np.percentile(x, (low, high))
    if p2 <= p1:
        p2 = p1 + 1.0
    return np.clip((x - p1) / (p2 - p1), 0.0, 1.0)


def difference_of_gaussians(volume: np.ndarray, sigma: float, anisotropy: float = 1.0) -> np.ndarray:
    sigma_z = max(float(sigma) * float(anisotropy), 0.5)
    sigma_xy = max(float(sigma), 0.5)
    sig1 = (sigma_z, sigma_xy, sigma_xy)
    sig2 = tuple(s * 1.6 for s in sig1)
    g1 = ndi.gaussian_filter(volume, sigma=sig1)
    g2 = ndi.gaussian_filter(volume, sigma=sig2)
    return g1 - g2


def _unit_gradients_3d(patch: np.ndarray):
    img = np.asarray(patch, dtype=np.float64)
    gz, gy, gx = np.gradient(img)
    mag = np.sqrt(gx * gx + gy * gy + gz * gz)
    w = mag * mag
    mag_safe = np.maximum(mag, 1e-12)
    uz = gz / mag_safe
    uy = gy / mag_safe
    ux = gx / mag_safe
    zz, yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1], 0 : img.shape[2]].astype(np.float64)
    return xx.ravel(), yy.ravel(), zz.ravel(), ux.ravel(), uy.ravel(), uz.ravel(), w.ravel()


def radial_center_3d(patch: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float, float]:
    """3D radial-symmetry center: least-squares intersection of intensity-gradient lines.

    Solves sum_i w_i (I - u_i u_i^T) c = sum_i w_i (I - u_i u_i^T) p_i
    where u_i is the unit gradient at voxel p_i. Returns (z, y, x) in patch coords.
    """
    xx, yy, zz, ux, uy, uz, w = _unit_gradients_3d(patch)
    if weights is not None:
        w = w * np.asarray(weights, dtype=np.float64).ravel()
    if float(w.sum()) < 1e-12:
        c = (np.array(patch.shape) - 1) / 2.0
        return float(c[0]), float(c[1]), float(c[2])

    # P = I - uu^T  (projector orthogonal to the gradient line)
    pxx = 1.0 - ux * ux
    pyy = 1.0 - uy * uy
    pzz = 1.0 - uz * uz
    pxy = -ux * uy
    pxz = -ux * uz
    pyz = -uy * uz

    ww = w
    a00 = np.sum(ww * pxx)
    a01 = np.sum(ww * pxy)
    a02 = np.sum(ww * pxz)
    a11 = np.sum(ww * pyy)
    a12 = np.sum(ww * pyz)
    a22 = np.sum(ww * pzz)
    bx = np.sum(ww * (pxx * xx + pxy * yy + pxz * zz))
    by = np.sum(ww * (pxy * xx + pyy * yy + pyz * zz))
    bz = np.sum(ww * (pxz * xx + pyz * yy + pzz * zz))

    A = np.array([[a00, a01, a02], [a01, a11, a12], [a02, a12, a22]], dtype=np.float64)
    b = np.array([bx, by, bz], dtype=np.float64)
    try:
        cx, cy, cz = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        c = (np.array(patch.shape) - 1) / 2.0
        return float(c[0]), float(c[1]), float(c[2])
    return float(cz), float(cy), float(cx)


def _closest_point_two_lines(p1, u1, p2, u2) -> np.ndarray | None:
    w0 = p1 - p2
    a = float(np.dot(u1, u1))
    b = float(np.dot(u1, u2))
    c = float(np.dot(u2, u2))
    d = float(np.dot(u1, w0))
    e = float(np.dot(u2, w0))
    denom = a * c - b * b
    if abs(denom) < 1e-12:
        return None
    t = (b * e - c * d) / denom
    s = (a * e - b * d) / denom
    return 0.5 * ((p1 + t * u1) + (p2 + s * u2))


def _line_distances(xx, yy, zz, ux, uy, uz, center) -> np.ndarray:
    cz, cy, cx = center
    dx = xx - cx
    dy = yy - cy
    dz = zz - cz
    cross_x = dy * uz - dz * uy
    cross_y = dz * ux - dx * uz
    cross_z = dx * uy - dy * ux
    return np.sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z)


def radial_center_3d_ransac(
    patch: np.ndarray,
    *,
    max_error: float = 1.5,
    min_inlier_ratio: float = 0.10,
    n_iter: int = 32,
    rng: np.random.Generator | None = None,
) -> tuple[tuple[float, float, float], float]:
    """RS-FISH 3D localization: RANSAC on gradient lines, then LS refit on inliers."""
    xx, yy, zz, ux, uy, uz, w = _unit_gradients_3d(patch)
    order = np.argsort(w)[::-1]
    keep = max(int(0.3 * len(order)), 16)
    idx = order[:keep]
    xx, yy, zz, ux, uy, uz, w = xx[idx], yy[idx], zz[idx], ux[idx], uy[idx], uz[idx], w[idx]
    n = len(w)
    if n < 8:
        c = (np.array(patch.shape) - 1) / 2.0
        return (float(c[0]), float(c[1]), float(c[2])), 0.0

    rng = rng or np.random.default_rng(0)
    pts = np.stack([xx, yy, zz], axis=1)
    dirs = np.stack([ux, uy, uz], axis=1)
    best_inliers = None
    best_count = -1
    for _ in range(int(n_iter)):
        i, j = rng.choice(n, size=2, replace=False)
        cand = _closest_point_two_lines(pts[i], dirs[i], pts[j], dirs[j])
        if cand is None:
            continue
        center = (float(cand[2]), float(cand[1]), float(cand[0]))  # z,y,x
        dist = _line_distances(xx, yy, zz, ux, uy, uz, center)
        inliers = dist <= max_error
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < 6:
        c = radial_center_3d(patch)
        dist = _line_distances(xx, yy, zz, ux, uy, uz, c)
        return c, float(np.mean(dist <= max_error))

    weights = np.zeros(patch.size, dtype=np.float64)
    # Map selected inliers back via a dense weight volume for LS on the patch.
    w_patch = np.zeros(patch.shape, dtype=np.float64)
    zz_i = np.round(zz[best_inliers]).astype(int)
    yy_i = np.round(yy[best_inliers]).astype(int)
    xx_i = np.round(xx[best_inliers]).astype(int)
    w_patch[zz_i, yy_i, xx_i] = w[best_inliers]
    center = radial_center_3d(patch, weights=w_patch)
    dist = _line_distances(xx, yy, zz, ux, uy, uz, center)
    inlier_ratio = float(np.mean(dist <= max_error))
    return center, inlier_ratio


def radial_center_2d(patch: np.ndarray) -> tuple[float, float]:
    """Parthasarathy 2012 closed-form radial symmetry center in 2D.

    Returns (row, col) in patch coordinates.
    """
    img = np.asarray(patch, dtype=np.float64)
    if img.ndim != 2 or min(img.shape) < 5:
        cy, cx = (np.array(img.shape) - 1) / 2.0
        return float(cy), float(cx)

    gy, gx = np.gradient(img)
    gy = gy[1:-1, 1:-1]
    gx = gx[1:-1, 1:-1]
    rows, cols = np.mgrid[1 : img.shape[0] - 1, 1 : img.shape[1] - 1].astype(np.float64)

    mag2 = gx * gx + gy * gy
    w = mag2.copy()
    mag2 = np.maximum(mag2, 1e-12)
    vx = gx / np.sqrt(mag2)
    vy = gy / np.sqrt(mag2)

    # Gradient line constraint: (r - r_c) × v = 0  =>  vy * x - vx * y = vy * xc - vx * yc
    rhs = vy * cols - vx * rows
    a = vy
    b = -vx
    wsum = np.sum(w)
    if wsum < 1e-12:
        return float((img.shape[0] - 1) / 2.0), float((img.shape[1] - 1) / 2.0)

    ata00 = np.sum(w * a * a)
    ata01 = np.sum(w * a * b)
    ata11 = np.sum(w * b * b)
    atb0 = np.sum(w * a * rhs)
    atb1 = np.sum(w * b * rhs)
    det = ata00 * ata11 - ata01 * ata01
    if abs(det) < 1e-12:
        return float((img.shape[0] - 1) / 2.0), float((img.shape[1] - 1) / 2.0)
    xc = (ata11 * atb0 - ata01 * atb1) / det
    yc = (-ata01 * atb0 + ata00 * atb1) / det
    return float(yc), float(xc)


def _inlier_ratio_3d(patch: np.ndarray, center: tuple[float, float, float], max_error: float) -> float:
    """Fraction of strong gradients whose lines pass near ``center``."""
    gz, gy, gx = np.gradient(np.asarray(patch, dtype=np.float64))
    mag = np.sqrt(gx * gx + gy * gy + gz * gz)
    strong = mag > np.percentile(mag, 70)
    if int(strong.sum()) < 8:
        return 0.0
    zz, yy, xx = np.nonzero(strong)
    vx = gx[strong] / (mag[strong] + 1e-12)
    vy = gy[strong] / (mag[strong] + 1e-12)
    vz = gz[strong] / (mag[strong] + 1e-12)
    cz, cy, cx = center
    dx = xx - cx
    dy = yy - cy
    dz = zz - cz
    # distance from point to 3D line through voxel along gradient
    cross_x = dy * vz - dz * vy
    cross_y = dz * vx - dx * vz
    cross_z = dx * vy - dy * vx
    dist = np.sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z)
    return float(np.mean(dist <= max_error))


def detect_spots_rsfish(
    volume: np.ndarray,
    *,
    sigma: float = 1.5,
    dog_threshold: float = 0.007,
    min_distance: int = 3,
    support: int = 3,
    min_inlier_ratio: float = 0.10,
    max_error: float = 1.5,
    intensity_percentile: float = 0.0,
    anisotropy: float = 1.0,
) -> np.ndarray:
    """Return (N, 3) array of z,y,x subpixel coordinates from 3D RS-FISH."""
    norm = normalize_percentiles(volume)
    dog = difference_of_gaussians(norm, sigma=sigma, anisotropy=anisotropy)
    peaks = peak_local_max(
        dog,
        min_distance=int(min_distance),
        threshold_abs=float(dog_threshold),
        exclude_border=support,
    )
    if peaks.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    depth, height, width = norm.shape
    r = int(support)
    spots: list[tuple[float, float, float, float]] = []
    for z0, y0, x0 in peaks:
        z0, y0, x0 = int(z0), int(y0), int(x0)
        z1, z2 = z0 - r, z0 + r + 1
        y1, y2 = y0 - r, y0 + r + 1
        x1, x2 = x0 - r, x0 + r + 1
        if z1 < 0 or y1 < 0 or x1 < 0 or z2 > depth or y2 > height or x2 > width:
            continue
        patch = norm[z1:z2, y1:y2, x1:x2]
        (pz, py, px), inliers = radial_center_3d_ransac(
            patch,
            max_error=max_error,
            min_inlier_ratio=min_inlier_ratio,
        )
        pz = float(np.clip(pz, 0, 2 * r))
        py = float(np.clip(py, 0, 2 * r))
        px = float(np.clip(px, 0, 2 * r))
        if inliers < min_inlier_ratio:
            continue
        intensity = float(norm[z0, y0, x0])
        spots.append((z1 + pz, y1 + py, x1 + px, intensity))

    if not spots:
        return np.zeros((0, 3), dtype=np.float32)
    arr = np.asarray(spots, dtype=np.float32)
    if intensity_percentile > 0:
        thr = float(np.percentile(arr[:, 3], intensity_percentile))
        arr = arr[arr[:, 3] >= thr]
    return arr[:, :3]


def points_to_mask(shape: tuple[int, int, int], points: np.ndarray, radius: int = 2) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if points.size == 0:
        return mask
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    r2 = float(radius * radius)
    for z, y, x in points:
        mask |= ((zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2) <= r2
    return mask.astype(np.uint8)


def points_to_mask_fast(shape: tuple[int, int, int], points: np.ndarray, radius: int = 2) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if points.size == 0:
        return mask
    r = int(radius)
    depth, height, width = shape
    offsets = [
        (dz, dy, dx)
        for dz in range(-r, r + 1)
        for dy in range(-r, r + 1)
        for dx in range(-r, r + 1)
        if dz * dz + dy * dy + dx * dx <= r * r
    ]
    for z, y, x in np.round(points).astype(int):
        for dz, dy, dx in offsets:
            zz, yy, xx = z + dz, y + dy, x + dx
            if 0 <= zz < depth and 0 <= yy < height and 0 <= xx < width:
                mask[zz, yy, xx] = 1
    return mask


def gt_centroids(mask: np.ndarray) -> np.ndarray:
    labeled, n = ndi.label(mask > 0)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    centers = ndi.center_of_mass(mask > 0, labeled, index=list(range(1, n + 1)))
    return np.asarray(centers, dtype=np.float32)


def match_points(pred: np.ndarray, gt: np.ndarray, max_dist: float) -> tuple[int, int, int]:
    from scipy.spatial import cKDTree

    if len(gt) == 0 and len(pred) == 0:
        return 0, 0, 0
    if len(gt) == 0:
        return 0, int(len(pred)), 0
    if len(pred) == 0:
        return 0, 0, int(len(gt))
    tree = cKDTree(gt)
    dist, idx = tree.query(pred, distance_upper_bound=max_dist)
    used = set()
    tp = 0
    for d, j in zip(dist, idx):
        if np.isfinite(d) and j < len(gt) and j not in used:
            used.add(int(j))
            tp += 1
    fp = int(len(pred) - tp)
    fn = int(len(gt) - tp)
    return tp, fp, fn


def _find_image(image_dir: Path, case_id: str) -> Path:
    for ext in (".tiff", ".tif"):
        path = image_dir / f"{case_id}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(case_id)


def _find_mask(mask_dir: Path, case_id: str) -> Path:
    for name in (f"{case_id}_mask.tiff", f"{case_id}_mask.tif"):
        path = mask_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(case_id)


def dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    p = pred > 0
    g = gt > 0
    inter = np.logical_and(p, g).sum()
    return float(2.0 * inter / (p.sum() + g.sum() + eps))


def evaluate_cases(
    case_ids: list[str],
    image_dir: Path,
    mask_dir: Path,
    out_dir: Path,
    params: dict,
    match_dist: float,
    paint_radius: int,
    n_overlays: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "metrics.csv"
    rows = []
    tp_all = fp_all = fn_all = 0
    dices = []

    from PIL import Image, ImageDraw

    for i, case_id in enumerate(tqdm(case_ids, desc="RS-FISH")):
        image = tifffile.imread(str(_find_image(image_dir, case_id)))
        mask = tifffile.imread(str(_find_mask(mask_dir, case_id))) > 0
        spots = detect_spots_rsfish(image, **params)
        gt = gt_centroids(mask)
        tp, fp, fn = match_points(spots, gt, max_dist=match_dist)
        tp_all += tp
        fp_all += fp
        fn_all += fn
        pred_mask = points_to_mask_fast(image.shape, spots, radius=paint_radius)
        d = dice(pred_mask, mask)
        dices.append(d)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        rows.append(
            {
                "case_id": case_id,
                "n_pred": int(len(spots)),
                "n_gt": int(len(gt)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": f"{prec:.4f}",
                "recall": f"{rec:.4f}",
                "f1": f"{f1:.4f}",
                "dice": f"{d:.4f}",
            }
        )
        if i < n_overlays:
            z = int(image.shape[0] // 2)
            gt_z = np.where(mask.sum(axis=(1, 2)) > 0)[0]
            if len(gt_z):
                z = int(gt_z[len(gt_z) // 2])
            sl = image[z].astype(np.float32)
            sl = sl / max(sl.max(), 1.0)
            rgb = np.stack([sl, sl, sl], axis=-1)
            rgb[mask[z], 1] = np.maximum(rgb[mask[z], 1], 0.7)
            rgb[pred_mask[z] > 0, 0] = np.maximum(rgb[pred_mask[z] > 0, 0], 0.7)
            im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
            draw = ImageDraw.Draw(im)
            draw.text((6, 6), f"{case_id} z={z} F1={f1:.3f} Dice={d:.3f}", fill=(255, 220, 80))
            im.save(overlay_dir / f"{case_id}.png")

    prec = tp_all / (tp_all + fp_all + 1e-8)
    rec = tp_all / (tp_all + fn_all + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    summary = {
        "n_cases": len(case_ids),
        "params": params,
        "match_dist": match_dist,
        "tp": tp_all,
        "fp": fp_all,
        "fn": fn_all,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mean_dice": float(np.mean(dices) if dices else 0.0),
    }
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["case_id"])
        writer.writeheader()
        writer.writerows(rows)
        if rows:
            writer.writerow(
                {
                    "case_id": "MICRO_AVG",
                    "n_pred": "",
                    "n_gt": "",
                    "tp": tp_all,
                    "fp": fp_all,
                    "fn": fn_all,
                    "precision": f"{prec:.4f}",
                    "recall": f"{rec:.4f}",
                    "f1": f"{f1:.4f}",
                    "dice": f"{summary['mean_dice']:.4f}",
                }
            )
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RS-FISH-style Python spot detection on cFos patches.")
    p.add_argument("--image_dir", type=Path, default=Path(r"S:\BaiduNetdiskDownload\cfos_dataset\image"))
    p.add_argument("--mask_dir", type=Path, default=Path(r"S:\Yifu_data\datasets\cfos_cc_filtered_masks"))
    p.add_argument("--split_json", type=Path, default=Path(r"S:\Yifu_data\outputs\cfos_cc_filtered_v1\data_split.json"))
    p.add_argument("--split", choices=("val", "train", "all"), default="val")
    p.add_argument("--out_dir", type=Path, default=Path(r"S:\Yifu_data\outputs\rs_fish_cfos_val"))
    p.add_argument("--sigma", type=float, default=1.5)
    p.add_argument("--dog_threshold", type=float, default=0.007)
    p.add_argument("--min_distance", type=int, default=3)
    p.add_argument("--support", type=int, default=3)
    p.add_argument("--min_inlier_ratio", type=float, default=0.10)
    p.add_argument("--max_error", type=float, default=1.5)
    p.add_argument("--intensity_percentile", type=float, default=0.0)
    p.add_argument("--anisotropy", type=float, default=1.0)
    p.add_argument("--match_dist", type=float, default=4.0)
    p.add_argument("--paint_radius", type=int, default=2)
    p.add_argument("--n_overlays", type=int, default=12)
    p.add_argument("--max_cases", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    if args.split == "val":
        cases = split["val_cases"]
    elif args.split == "train":
        cases = split["train_cases"]
    else:
        cases = split["train_cases"] + split["val_cases"]
    if args.max_cases:
        cases = cases[: args.max_cases]
    params = {
        "sigma": args.sigma,
        "dog_threshold": args.dog_threshold,
        "min_distance": args.min_distance,
        "support": args.support,
        "min_inlier_ratio": args.min_inlier_ratio,
        "max_error": args.max_error,
        "intensity_percentile": args.intensity_percentile,
        "anisotropy": args.anisotropy,
    }
    summary = evaluate_cases(
        cases,
        args.image_dir,
        args.mask_dir,
        args.out_dir,
        params,
        match_dist=args.match_dist,
        paint_radius=args.paint_radius,
        n_overlays=args.n_overlays,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
