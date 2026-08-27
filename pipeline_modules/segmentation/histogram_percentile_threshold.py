#!/usr/bin/env python3
"""Per-slice histogram-driven percentile thresholding for vessel-like signals.

For each image, estimate an intensity threshold from the histogram (Otsu / knee /
Yen / Triangle / Li), convert that threshold into a percentile of positive
pixels, optionally clamp to a safe range, then threshold.

Typical LSFM lymphatic/vessel stacks have sparse bright structures on a dark
background, so a fixed p95 / p97.5 / p99 is rarely optimal for every slice.
This module picks a percentile per slice from the histogram shape.

Example::

    python -m pipeline_modules.segmentation.histogram_percentile_threshold \\
      --input_dir ".../crop1" \\
      --output_dir ".../crop1_seg_trials/auto_percentile" \\
      --files_glob "*_Z0{025,109,114}*.tif" \\
      --method blend --write_overlay
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


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
    """Percentile rank of ``thr`` among positive pixels (0–100)."""
    return float(100.0 * np.mean(pos <= thr))


def estimate_percentile_from_histogram(
    image: np.ndarray,
    *,
    method: str = "blend",
    clamp_lo: float = 95.0,
    clamp_hi: float = 99.5,
) -> dict:
    """Return suggested percentile + diagnostics for one 2D/3D array.

    Methods
    -------
    otsu / triangle / yen / li / knee
        Use that single histogram rule.
    blend
        ``max(otsu%, knee%)`` then clamp — usually best for sparse bright vessels.
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
    thr = float(np.percentile(pos, pct))
    return {
        "method": method,
        "raw_percentile": float(raw),
        "suggested_percentile": pct,
        "threshold": thr,
        "clamp": [float(clamp_lo), float(clamp_hi)],
        "n_positive": int(pos.size),
        "p50": float(np.percentile(pos, 50)),
        "p95": float(np.percentile(pos, 95)),
        "p97_5": float(np.percentile(pos, 97.5)),
        "p99": float(np.percentile(pos, 99)),
        "candidates_threshold": candidates_thr,
        "candidates_percentile": candidates_pct,
    }


def clean_mask(mask: np.ndarray, min_size: int = 48) -> np.ndarray:
    from skimage.morphology import closing, disk, opening, remove_small_objects

    m = np.asarray(mask, dtype=bool)
    # skimage>=0.26: max_size removes objects <= value
    m = remove_small_objects(m, max_size=max(int(min_size) - 1, 0))
    m = closing(m, disk(1))
    m = opening(m, disk(1))
    m = remove_small_objects(m, max_size=max(int(min_size) - 1, 0))
    return m.astype(np.uint8)


def segment_percentile(image: np.ndarray, percentile: float, *, min_size: int = 48) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    pos = img[img > 0]
    if pos.size == 0:
        return np.zeros(img.shape, dtype=np.uint8)
    thr = float(np.percentile(pos, float(percentile)))
    return clean_mask(img > thr, min_size=min_size)


def to_u8_display(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32)
    pos = a[a > 0]
    if pos.size < 32:
        return np.zeros(a.shape, dtype=np.uint8)
    lo, hi = np.percentile(pos, (1.0, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    return (np.clip((a - lo) / (hi - lo), 0, 1) * 255.0).astype(np.uint8)


def overlay_mask(img_u8: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = np.stack([img_u8, img_u8, img_u8], axis=-1)
    m = np.asarray(mask, dtype=bool)
    rgb[m, 0] = np.clip(0.35 * rgb[m, 0] + 0.65 * 255, 0, 255).astype(np.uint8)
    rgb[m, 1] = (rgb[m, 1] * 0.35).astype(np.uint8)
    rgb[m, 2] = (rgb[m, 2] * 0.35).astype(np.uint8)
    return rgb


def list_input_files(input_dir: Path, files: list[str] | None, files_regex: str | None) -> list[Path]:
    all_tiffs = sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    )
    if files:
        want = set(files)
        picked = [p for p in all_tiffs if p.name in want]
        missing = sorted(want - {p.name for p in picked})
        if missing:
            raise FileNotFoundError(f"Missing files under {input_dir}: {missing[:5]}")
        return picked
    if files_regex:
        rx = re.compile(files_regex)
        picked = [p for p in all_tiffs if rx.search(p.name)]
        if not picked:
            raise FileNotFoundError(f"No files matched regex {files_regex!r} in {input_dir}")
        return picked
    return all_tiffs


def process_tiff_folder(
    input_dir: Path | str,
    output_dir: Path | str,
    *,
    method: str = "blend",
    clamp_lo: float = 95.0,
    clamp_hi: float = 99.5,
    min_size: int = 48,
    write_overlay: bool = True,
    also_fixed: Iterable[float] = (95.0, 97.5, 99.0),
    files: list[str] | None = None,
    files_regex: str | None = None,
) -> dict:
    """Segment each TIFF with auto percentile; optionally compare fixed percentiles."""
    import cv2
    import tifffile

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    montage_dir = output_dir / "_montages"
    if write_overlay:
        montage_dir.mkdir(parents=True, exist_ok=True)

    paths = list_input_files(input_dir, files, files_regex)
    records: list[dict] = []

    for path in paths:
        img = tifffile.imread(str(path))
        if img.ndim != 2:
            raise ValueError(f"Expected 2D TIFF, got shape={img.shape} for {path.name}")
        est = estimate_percentile_from_histogram(
            img, method=method, clamp_lo=clamp_lo, clamp_hi=clamp_hi
        )
        auto_pct = float(est["suggested_percentile"])
        auto_mask = segment_percentile(img, auto_pct, min_size=min_size)

        slice_dir = output_dir / path.stem
        slice_dir.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(slice_dir / "auto_mask.tif"), (auto_mask * 255).astype(np.uint8))
        (slice_dir / "auto_meta.json").write_text(json.dumps(est, indent=2), encoding="utf-8")

        u8 = to_u8_display(img)
        tiles = []
        if write_overlay:
            cv2.imwrite(str(slice_dir / "00_input_display.png"), u8)
            ov = overlay_mask(u8, auto_mask)
            pct_tag = f"{auto_pct:.2f}".replace(".", "p")
            cv2.imwrite(
                str(slice_dir / f"auto_p{pct_tag}_overlay.png"),
                cv2.cvtColor(ov, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                str(slice_dir / "auto_overlay.png"),
                cv2.cvtColor(ov, cv2.COLOR_RGB2BGR),
            )
            tile = cv2.resize(cv2.cvtColor(ov, cv2.COLOR_RGB2BGR), (320, 320))
            label = f"auto p{auto_pct:.2f}"
            cv2.putText(tile, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            tiles.append(tile)

        fixed_stats = {}
        for fp in also_fixed:
            fp = float(fp)
            m = segment_percentile(img, fp, min_size=min_size)
            tifffile.imwrite(str(slice_dir / f"fixed_p{fp:g}_mask.tif"), (m * 255).astype(np.uint8))
            fixed_stats[f"p{fp:g}"] = float(m.mean())
            if write_overlay:
                ov = overlay_mask(u8, m)
                cv2.imwrite(
                    str(slice_dir / f"fixed_p{fp:g}_overlay.png"),
                    cv2.cvtColor(ov, cv2.COLOR_RGB2BGR),
                )
                tile = cv2.resize(cv2.cvtColor(ov, cv2.COLOR_RGB2BGR), (320, 320))
                cv2.putText(
                    tile, f"fixed p{fp:g}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1
                )
                tiles.append(tile)

        if write_overlay and tiles:
            # prepend input
            inp = cv2.resize(cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR), (320, 320))
            cv2.putText(inp, "input", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            all_tiles = [inp] + tiles
            montage = np.hstack(all_tiles)
            cv2.imwrite(str(slice_dir / "compare_montage.png"), montage)
            cv2.imwrite(str(montage_dir / f"{path.stem}_compare.png"), montage)

        rec = {
            "file": path.name,
            "suggested_percentile": auto_pct,
            "threshold": est["threshold"],
            "fg_frac_auto": float(auto_mask.mean()),
            "fg_frac_fixed": fixed_stats,
            "estimate": est,
        }
        records.append(rec)
        logger.info(
            "%s → auto p=%.2f thr=%.1f fg=%.4f (fixed %s)",
            path.name,
            auto_pct,
            est["threshold"],
            float(auto_mask.mean()),
            fixed_stats,
        )

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "method": method,
        "clamp": [clamp_lo, clamp_hi],
        "min_size": min_size,
        "n_files": len(records),
        "records": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Done. %d files → %s", len(records), output_dir)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_dir", required=True, help="Folder of 2D TIFF slices")
    p.add_argument("--output_dir", required=True, help="Output folder for masks/overlays")
    p.add_argument(
        "--method",
        default="blend",
        choices=("blend", "otsu", "triangle", "yen", "li", "knee"),
        help="Histogram rule used to choose percentile (default blend=max(otsu,knee))",
    )
    p.add_argument("--clamp_lo", type=float, default=95.0)
    p.add_argument("--clamp_hi", type=float, default=99.5)
    p.add_argument("--min_size", type=int, default=48, help="Remove components smaller than this")
    p.add_argument("--write_overlay", action="store_true", help="Write PNG overlays + montages")
    p.add_argument(
        "--also_fixed",
        default="95,97.5,99",
        help="Comma-separated fixed percentiles to compare (empty to skip)",
    )
    p.add_argument(
        "--files",
        default="",
        help="Optional comma-separated exact filenames to process",
    )
    p.add_argument(
        "--files_regex",
        default="",
        help="Optional regex to select filenames (ignored if --files is set)",
    )
    return p


def main() -> int:
    _configure_logging()
    args = build_parser().parse_args()
    fixed = []
    if str(args.also_fixed).strip():
        fixed = [float(x) for x in str(args.also_fixed).split(",") if x.strip()]
    files = [x.strip() for x in str(args.files).split(",") if x.strip()] or None
    process_tiff_folder(
        args.input_dir,
        args.output_dir,
        method=args.method,
        clamp_lo=float(args.clamp_lo),
        clamp_hi=float(args.clamp_hi),
        min_size=int(args.min_size),
        write_overlay=bool(args.write_overlay),
        also_fixed=fixed,
        files=files,
        files_regex=str(args.files_regex).strip() or None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
