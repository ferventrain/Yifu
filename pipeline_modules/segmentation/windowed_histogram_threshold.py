#!/usr/bin/env python3
"""Windowed histogram-percentile thresholding for vessel/lymphatic-like TIFF stacks.

For each local window, estimate a percentile from the local intensity histogram
(blend of Otsu/knee by default), convert to a threshold, optionally loosen it,
then enforce an absolute intensity floor.

The ``loose2`` preset matches the tuned trial:
  window=151, floor=2000, pct_nudge=-1.3, thr_scale=0.85,
  clamp=[93, 98.5], bright windows get an extra -1.0 percentile nudge.

Example::

    python -m pipeline_modules.segmentation.windowed_histogram_threshold --preset loose2 --input_dir ".../crop1" --output_dir ".../crop1_mask_loose2" --write_overlay

Use ``--clean_3d`` to apply ``min_size`` on 3D connected components after
per-slice thresholding (keeps thin tubes that look small in a single slice).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from pipeline_modules.segmentation.histogram_percentile_threshold import (
    estimate_percentile_from_histogram,
    list_input_files,
    overlay_mask,
    to_u8_display,
)

logger = logging.getLogger(__name__)

PRESETS = {
    "loose2": {
        "window": 151,
        "stride": 32,
        "floor": 2000.0,
        "method": "blend",
        "pct_nudge": -1.3,
        "thr_scale": 0.85,
        "clamp_lo": 93.0,
        "clamp_hi": 98.5,
        "bright_local_p99": 5000.0,
        "bright_extra_nudge": -1.0,
        "min_size": 48,
    },
}


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def clean_mask(mask: np.ndarray, min_size: int = 48, *, morph: bool = True) -> np.ndarray:
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


def process_tiff_folder(
    input_dir: Path | str,
    output_dir: Path | str,
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
    min_size: int = 48,
    clean_3d: bool = False,
    write_overlay: bool = False,
    write_threshold_map: bool = True,
    files: list[str] | None = None,
    files_regex: str | None = None,
    preset: str | None = None,
) -> dict:
    import cv2
    import tifffile

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(exist_ok=True)
    overlay_dir = output_dir / "overlays"
    thr_dir = output_dir / "threshold_maps"
    montage_dir = output_dir / "_montages"
    if write_overlay:
        overlay_dir.mkdir(exist_ok=True)
        montage_dir.mkdir(exist_ok=True)
    if write_threshold_map:
        thr_dir.mkdir(exist_ok=True)

    paths = list_input_files(input_dir, files, files_regex)
    records: list[dict] = []
    params = {
        "preset": preset,
        "window": window,
        "stride": stride,
        "floor": floor,
        "method": method,
        "pct_nudge": pct_nudge,
        "thr_scale": thr_scale,
        "clamp": [clamp_lo, clamp_hi],
        "bright_local_p99": bright_local_p99,
        "bright_extra_nudge": bright_extra_nudge,
        "min_size": min_size,
        "clean_3d": bool(clean_3d),
    }
    # When cleaning in 3D, keep per-slice morphology light and defer size filter.
    slice_min_size = 0 if clean_3d else int(min_size)
    mask_paths: list[Path] = []

    for i, path in enumerate(paths, 1):
        img = tifffile.imread(str(path))
        if img.ndim != 2:
            raise ValueError(f"Expected 2D TIFF, got shape={img.shape} for {path.name}")
        img_f = np.asarray(img, dtype=np.float32)
        thr_map, st = local_hist_threshold_map(
            img_f,
            window=window,
            stride=stride,
            floor=floor,
            method=method,
            pct_nudge=pct_nudge,
            thr_scale=thr_scale,
            clamp_lo=clamp_lo,
            clamp_hi=clamp_hi,
            bright_local_p99=bright_local_p99,
            bright_extra_nudge=bright_extra_nudge,
        )
        mask = clean_mask(
            (img_f >= thr_map) & (img_f >= float(floor)),
            min_size=slice_min_size,
            morph=True,
        )
        out_mask = mask_dir / f"{path.stem}_mask.tif"
        tifffile.imwrite(str(out_mask), (mask * 255).astype(np.uint8))
        mask_paths.append(out_mask)

        if write_threshold_map:
            import cv2 as _cv2

            _cv2.imwrite(str(thr_dir / f"{path.stem}_thr.png"), to_u8_display(thr_map))

        if write_overlay and not clean_3d:
            u8 = to_u8_display(img_f)
            ov = overlay_mask(u8, mask)
            cv2.imwrite(str(overlay_dir / f"{path.stem}_overlay.png"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
            tile_in = cv2.resize(cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR), (320, 320))
            tile_ov = cv2.resize(cv2.cvtColor(ov, cv2.COLOR_RGB2BGR), (320, 320))
            cv2.putText(tile_in, "input", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            cv2.putText(tile_ov, "mask", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            cv2.imwrite(str(montage_dir / f"{path.stem}_montage.png"), np.hstack([tile_in, tile_ov]))

        rec = {
            "file": path.name,
            "fg_frac": float(mask.mean()),
            "n_fg": int(mask.sum()),
            **st,
        }
        records.append(rec)
        logger.info(
            "[%d/%d] %s fg=%.4f pct~%.2f thr_med=%.1f",
            i,
            len(paths),
            path.name,
            rec["fg_frac"],
            st["pct_p50"],
            st["thr_p50"],
        )

    if clean_3d and mask_paths:
        logger.info("3D connected-component cleanup (min_size=%d) on %d slices...", int(min_size), len(mask_paths))
        vol = np.stack([tifffile.imread(str(p)) > 0 for p in mask_paths], axis=0)
        vol = clean_mask_3d(vol, min_size=int(min_size))
        for zi, (path, out_mask) in enumerate(zip(paths, mask_paths)):
            mask_z = vol[zi]
            tifffile.imwrite(str(out_mask), (mask_z.astype(np.uint8) * 255))
            records[zi]["fg_frac"] = float(mask_z.mean())
            records[zi]["n_fg"] = int(mask_z.sum())
            if write_overlay:
                img_f = np.asarray(tifffile.imread(str(path)), dtype=np.float32)
                u8 = to_u8_display(img_f)
                ov = overlay_mask(u8, mask_z)
                cv2.imwrite(str(overlay_dir / f"{path.stem}_overlay.png"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
                tile_in = cv2.resize(cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR), (320, 320))
                tile_ov = cv2.resize(cv2.cvtColor(ov, cv2.COLOR_RGB2BGR), (320, 320))
                cv2.putText(tile_in, "input", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                cv2.putText(tile_ov, "mask", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                cv2.imwrite(str(montage_dir / f"{path.stem}_montage.png"), np.hstack([tile_in, tile_ov]))
        logger.info("3D cleanup done. fg voxels=%d", int(vol.sum()))

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "params": params,
        "n_files": len(records),
        "records": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Done. %d files → %s", len(records), output_dir)
    return summary


def process_zarr_volume(
    input_zarr: Path | str,
    output_dir: Path | str,
    *,
    dataset_name: str = "0",
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
    min_size: int = 48,
    clean_3d: bool = False,
    write_overlay: bool = False,
    write_threshold_map: bool = True,
    preset: str | None = None,
) -> dict:
    """Run loose2-style per-slice thresholding on a 3D Zarr volume (ZYX)."""
    import cv2
    import tifffile
    from pipeline_modules.segmentation.zarr_utils import open_zarr_dataset

    input_zarr = Path(input_zarr)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(exist_ok=True)
    overlay_dir = output_dir / "overlays"
    thr_dir = output_dir / "threshold_maps"
    montage_dir = output_dir / "_montages"
    if write_overlay:
        overlay_dir.mkdir(exist_ok=True)
        montage_dir.mkdir(exist_ok=True)
    if write_threshold_map:
        thr_dir.mkdir(exist_ok=True)

    arr = open_zarr_dataset(input_zarr, dataset_name=dataset_name)
    if len(arr.shape) != 3:
        raise ValueError(f"Expected 3D Zarr (ZYX), got shape={arr.shape}")
    nz = int(arr.shape[0])
    records: list[dict] = []
    params = {
        "preset": preset,
        "input_zarr": str(input_zarr),
        "dataset_name": dataset_name,
        "window": window,
        "stride": stride,
        "floor": floor,
        "method": method,
        "pct_nudge": pct_nudge,
        "thr_scale": thr_scale,
        "clamp": [clamp_lo, clamp_hi],
        "bright_local_p99": bright_local_p99,
        "bright_extra_nudge": bright_extra_nudge,
        "min_size": min_size,
        "clean_3d": bool(clean_3d),
        "shape": list(arr.shape),
    }
    slice_min_size = 0 if clean_3d else int(min_size)
    mask_stack = np.zeros(arr.shape, dtype=np.uint8)

    for zi in range(nz):
        img_f = np.asarray(arr[zi], dtype=np.float32)
        thr_map, st = local_hist_threshold_map(
            img_f,
            window=window,
            stride=stride,
            floor=floor,
            method=method,
            pct_nudge=pct_nudge,
            thr_scale=thr_scale,
            clamp_lo=clamp_lo,
            clamp_hi=clamp_hi,
            bright_local_p99=bright_local_p99,
            bright_extra_nudge=bright_extra_nudge,
        )
        mask = clean_mask(
            (img_f >= thr_map) & (img_f >= float(floor)),
            min_size=slice_min_size,
            morph=True,
        )
        mask_stack[zi] = mask
        stem = f"z{zi:04d}"
        tifffile.imwrite(str(mask_dir / f"{stem}_mask.tif"), (mask * 255).astype(np.uint8))
        if write_threshold_map:
            cv2.imwrite(str(thr_dir / f"{stem}_thr.png"), to_u8_display(thr_map))
        if write_overlay and not clean_3d:
            u8 = to_u8_display(img_f)
            ov = overlay_mask(u8, mask)
            cv2.imwrite(str(overlay_dir / f"{stem}_overlay.png"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
            tile_in = cv2.resize(cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR), (320, 320))
            tile_ov = cv2.resize(cv2.cvtColor(ov, cv2.COLOR_RGB2BGR), (320, 320))
            cv2.putText(tile_in, "input", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            cv2.putText(tile_ov, "mask", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            cv2.imwrite(str(montage_dir / f"{stem}_montage.png"), np.hstack([tile_in, tile_ov]))
        rec = {"file": stem, "fg_frac": float(mask.mean()), "n_fg": int(mask.sum()), **st}
        records.append(rec)
        if (zi + 1) % 50 == 0 or zi == 0 or zi + 1 == nz:
            logger.info("[%d/%d] %s fg=%.4f thr_med=%.1f", zi + 1, nz, stem, rec["fg_frac"], st["thr_p50"])

    if clean_3d:
        logger.info("3D connected-component cleanup (min_size=%d)...", int(min_size))
        mask_stack = clean_mask_3d(mask_stack, min_size=int(min_size))
        for zi in range(nz):
            mask_z = mask_stack[zi]
            stem = f"z{zi:04d}"
            tifffile.imwrite(str(mask_dir / f"{stem}_mask.tif"), (mask_z.astype(np.uint8) * 255))
            records[zi]["fg_frac"] = float(mask_z.mean())
            records[zi]["n_fg"] = int(mask_z.sum())
            if write_overlay:
                img_f = np.asarray(arr[zi], dtype=np.float32)
                u8 = to_u8_display(img_f)
                ov = overlay_mask(u8, mask_z)
                cv2.imwrite(str(overlay_dir / f"{stem}_overlay.png"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
                tile_in = cv2.resize(cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR), (320, 320))
                tile_ov = cv2.resize(cv2.cvtColor(ov, cv2.COLOR_RGB2BGR), (320, 320))
                cv2.putText(tile_in, "input", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                cv2.putText(tile_ov, "mask", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                cv2.imwrite(str(montage_dir / f"{stem}_montage.png"), np.hstack([tile_in, tile_ov]))
        logger.info("3D cleanup done. fg voxels=%d", int(mask_stack.sum()))

    summary = {
        "input_zarr": str(input_zarr),
        "output_dir": str(output_dir),
        "params": params,
        "n_files": len(records),
        "records": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Done. %d slices → %s", len(records), output_dir)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_dir", default="", help="Input TIFF folder (real TIFF slices)")
    p.add_argument("--input_zarr", default="", help="Input 3D intensity Zarr (ZYX), preferred when TIFF folder is corrupt/HDF")
    p.add_argument("--dataset_name", default="0", help="Dataset name inside the Zarr group")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--preset", default="", choices=("", *PRESETS.keys()), help="Named parameter set, e.g. loose2")
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--floor", type=float, default=None)
    p.add_argument("--method", default=None, choices=("blend", "otsu", "triangle", "yen", "li", "knee"))
    p.add_argument("--pct_nudge", type=float, default=None)
    p.add_argument("--thr_scale", type=float, default=None)
    p.add_argument("--clamp_lo", type=float, default=None)
    p.add_argument("--clamp_hi", type=float, default=None)
    p.add_argument("--bright_local_p99", type=float, default=None)
    p.add_argument("--bright_extra_nudge", type=float, default=None)
    p.add_argument("--min_size", type=int, default=None)
    p.add_argument(
        "--clean_3d",
        action="store_true",
        help="Defer min_size filter to 3D connected components across the whole stack",
    )
    p.add_argument("--write_overlay", action="store_true")
    p.add_argument("--no_threshold_map", action="store_true")
    p.add_argument("--files", default="")
    p.add_argument("--files_regex", default="")
    return p


def main() -> int:
    _configure_logging()
    args = build_parser().parse_args()
    preset_name = args.preset or "loose2"
    if preset_name not in PRESETS:
        raise SystemExit(f"Unknown preset: {preset_name}")
    params = dict(PRESETS[preset_name])
    for key in (
        "window",
        "stride",
        "floor",
        "method",
        "pct_nudge",
        "thr_scale",
        "clamp_lo",
        "clamp_hi",
        "bright_local_p99",
        "bright_extra_nudge",
        "min_size",
    ):
        val = getattr(args, key)
        if val is not None:
            params[key] = val

    input_zarr = str(args.input_zarr).strip()
    input_dir = str(args.input_dir).strip()
    if input_zarr:
        process_zarr_volume(
            input_zarr,
            args.output_dir,
            dataset_name=str(args.dataset_name),
            write_overlay=bool(args.write_overlay),
            write_threshold_map=not bool(args.no_threshold_map),
            preset=preset_name,
            clean_3d=bool(args.clean_3d),
            **params,
        )
        return 0
    if not input_dir:
        raise SystemExit("Provide --input_zarr or --input_dir")
    files = [x.strip() for x in str(args.files).split(",") if x.strip()] or None
    process_tiff_folder(
        input_dir,
        args.output_dir,
        write_overlay=bool(args.write_overlay),
        write_threshold_map=not bool(args.no_threshold_map),
        files=files,
        files_regex=str(args.files_regex).strip() or None,
        preset=preset_name,
        clean_3d=bool(args.clean_3d),
        **params,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
