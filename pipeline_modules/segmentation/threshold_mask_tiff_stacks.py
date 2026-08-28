#!/usr/bin/env python3
"""Threshold one TIFF stack to a mask, then apply that mask to another TIFF stack.

No Zarr required. Processes slice-by-slice (paired by sorted order).

Example:
  python -m pipeline_modules.segmentation.threshold_mask_tiff_stacks --mask_from_dir ".../ch1" --threshold 15000 --signal_dir ".../ch0" --mask_out_dir ".../ch1_mask" --masked_signal_out_dir ".../ch0_masked"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import tifffile
from tqdm import tqdm


def _list_tiffs(folder: Path) -> list[Path]:
    files = sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff"))
    files = sorted(set(files), key=lambda p: p.name.lower())
    if not files:
        raise FileNotFoundError(f"No TIFF files in {folder}")
    return files


def _z_key(path: Path) -> tuple:
    m = re.search(r"[Zz](\d+)", path.name)
    if m:
        return (0, int(m.group(1)), path.name.lower())
    return (1, path.name.lower())


def _pair_stacks(mask_files: list[Path], signal_files: list[Path]) -> list[tuple[Path, Path]]:
    if len(mask_files) != len(signal_files):
        raise ValueError(
            f"Slice count mismatch: mask={len(mask_files)} signal={len(signal_files)}"
        )
    mask_sorted = sorted(mask_files, key=_z_key)
    signal_sorted = sorted(signal_files, key=_z_key)
    return list(zip(mask_sorted, signal_sorted))


def run(
    *,
    mask_from_dir: Path,
    signal_dir: Path,
    threshold: float,
    mask_out_dir: Path,
    masked_signal_out_dir: Path,
    mask_dtype: str = "uint8",
    resume: bool = False,
) -> dict:
    mask_from_dir = Path(mask_from_dir)
    signal_dir = Path(signal_dir)
    mask_out_dir = Path(mask_out_dir)
    masked_signal_out_dir = Path(masked_signal_out_dir)
    mask_out_dir.mkdir(parents=True, exist_ok=True)
    masked_signal_out_dir.mkdir(parents=True, exist_ok=True)

    pairs = _pair_stacks(_list_tiffs(mask_from_dir), _list_tiffs(signal_dir))
    thr = float(threshold)
    n_fg = 0
    n_pix = 0
    n_skip = 0
    n_done = 0

    for mask_src, sig_src in tqdm(pairs, desc="threshold+mask", unit="slice"):
        mask_dst = mask_out_dir / mask_src.name
        sig_dst = masked_signal_out_dir / sig_src.name
        if resume and mask_dst.exists() and sig_dst.exists():
            n_skip += 1
            continue

        mask_img = tifffile.imread(str(mask_src))
        sig_img = tifffile.imread(str(sig_src))
        if mask_img.shape != sig_img.shape:
            raise ValueError(
                f"Shape mismatch at {mask_src.name} vs {sig_src.name}: "
                f"{mask_img.shape} vs {sig_img.shape}"
            )
        binary = mask_img > thr
        n_fg += int(binary.sum())
        n_pix += int(binary.size)

        if mask_dtype == "uint16":
            mask_out = binary.astype(np.uint16) * np.uint16(65535)
        else:
            mask_out = binary.astype(np.uint8) * np.uint8(255)

        masked = np.where(binary, sig_img, 0).astype(sig_img.dtype, copy=False)

        tifffile.imwrite(str(mask_dst), mask_out, compression=None)
        tifffile.imwrite(str(sig_dst), masked, compression=None)
        n_done += 1

    return {
        "n_slices": len(pairs),
        "n_written": n_done,
        "n_skipped": n_skip,
        "threshold": thr,
        "fg_frac": float(n_fg / max(n_pix, 1)) if n_pix else None,
        "mask_out_dir": str(mask_out_dir),
        "masked_signal_out_dir": str(masked_signal_out_dir),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mask_from_dir", required=True, help="TIFF stack used to build threshold mask (e.g. ch1)")
    p.add_argument("--signal_dir", required=True, help="TIFF stack to mask (e.g. ch0)")
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--mask_out_dir", required=True, help="Output folder for binary mask TIFFs")
    p.add_argument("--masked_signal_out_dir", required=True, help="Output folder for masked signal TIFFs")
    p.add_argument("--mask_dtype", choices=("uint8", "uint16"), default="uint8")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip slices whose mask and masked-signal outputs already exist",
    )
    args = p.parse_args()
    result = run(
        mask_from_dir=Path(args.mask_from_dir),
        signal_dir=Path(args.signal_dir),
        threshold=args.threshold,
        mask_out_dir=Path(args.mask_out_dir),
        masked_signal_out_dir=Path(args.masked_signal_out_dir),
        mask_dtype=args.mask_dtype,
        resume=bool(args.resume),
    )
    print(result)


if __name__ == "__main__":
    main()
