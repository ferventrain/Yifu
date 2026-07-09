"""
Batch virtual H&E false coloring for paired channel TIFF stacks.

Expects slice files named like:
    681220_493440_000000_ch0.tiff  (cytoplasm)
    681220_493440_000000_ch1.tiff  (nuclei)

Uses the falsecolor package (Liu Lab) from the yifu conda environment.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import falsecolor
import numpy as np
from scipy import ndimage
from skimage.io import imread, imsave

CHANNEL_SUFFIX_RE = re.compile(r"^(?P<prefix>.+)_ch(?P<channel>\d+)\.tiff?$", re.IGNORECASE)


def discover_channel_pairs(
    input_dir: Path,
    nuclei_channel: int,
    cyto_channel: int,
) -> list[tuple[str, Path, Path]]:
    """Return sorted (slice_prefix, nuclei_path, cyto_path) tuples."""
    by_prefix: dict[str, dict[int, Path]] = {}

    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        match = CHANNEL_SUFFIX_RE.match(path.name)
        if match is None:
            continue
        prefix = match.group("prefix")
        channel = int(match.group("channel"))
        by_prefix.setdefault(prefix, {})[channel] = path

    pairs: list[tuple[str, Path, Path]] = []
    for prefix in sorted(by_prefix):
        channels = by_prefix[prefix]
        if nuclei_channel not in channels or cyto_channel not in channels:
            continue
        pairs.append((prefix, channels[nuclei_channel], channels[cyto_channel]))

    return pairs


def select_slice_pairs(
    pairs: list[tuple[str, Path, Path]],
    *,
    slice_start: int,
    slice_count: int,
    middle: int,
) -> tuple[list[tuple[str, Path, Path]], int, int]:
    """Return (selected_pairs, start_index, end_index) from the full sorted list."""
    total = len(pairs)
    if middle > 0:
        count = min(middle, total)
        start = max(0, (total - count) // 2)
        end = start + count
    else:
        start = max(0, min(slice_start, total))
        count = slice_count if slice_count > 0 else total - start
        end = min(total, start + count)

    return pairs[start:end], start, end


def _progress_path(output_dir: Path) -> Path:
    return output_dir / "_falsecolor_progress.json"


def load_progress(output_dir: Path) -> dict:
    path = _progress_path(output_dir)
    if not path.exists():
        return {"completed": []}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_progress(output_dir: Path, completed_prefixes: list[str]) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "completed": completed_prefixes,
    }
    path = _progress_path(output_dir)
    tmp_path = path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def _gpu_available() -> bool:
    try:
        from numba import cuda

        return cuda.is_available()
    except Exception:
        return False


def render_falsecolor(
    nuclei: np.ndarray,
    cyto: np.ndarray,
    *,
    backend: str,
    color_key: str,
    nuc_threshold: int,
    cyto_threshold: int,
    nuc_normfactor: int,
    cyto_normfactor: int,
) -> np.ndarray:
    if backend == "gpu":
        color_settings = falsecolor.getColorSettings(key=color_key)
        if color_key.upper() == "IHC":
            cyto_settings = color_settings["anti"]
        else:
            cyto_settings = color_settings["cyto"]
        return falsecolor.rapidFalseColor(
            nuclei,
            cyto,
            color_settings["nuclei"],
            cyto_settings,
            nuc_normfactor=nuc_normfactor,
            cyto_normfactor=cyto_normfactor,
            nuc_bg_threshold=nuc_threshold,
            cyto_bg_threshold=cyto_threshold,
        )

    return falsecolor.falseColor(
        nuclei,
        cyto,
        nuc_threshold=nuc_threshold,
        cyto_threshold=cyto_threshold,
        nuc_normfactor=nuc_normfactor,
        cyto_normfactor=cyto_normfactor,
        color_key=color_key,
    )


def apply_background_mask(
    rgb: np.ndarray,
    nuclei: np.ndarray,
    cyto: np.ndarray,
    *,
    mode: str,
    background_threshold: int,
    white_threshold: int,
    pale_signal_threshold: int,
    hsv_mask_val: float,
    hsv_min_size: int,
) -> np.ndarray:
    """Set background pixels to black after false coloring."""
    if mode == "none":
        return rgb

    output = rgb.copy()
    background = np.zeros(rgb.shape[:2], dtype=bool)
    signal = np.maximum(nuclei, cyto)

    if mode in {"channels", "channels+rgb"}:
        background |= signal <= background_threshold

    if mode in {"rgb", "channels+rgb"}:
        pale = np.all(rgb >= white_threshold, axis=-1)
        if pale_signal_threshold > 0:
            background |= pale & (signal <= pale_signal_threshold)
        else:
            background |= pale

    if mode == "hsv":
        empty_mask = falsecolor.maskEmpty(
            rgb,
            mask_val=hsv_mask_val,
            return3D=True,
            min_size=hsv_min_size,
        )
        background = empty_mask[:, :, 0] == 0

    output[background] = 0
    return output


def fill_internal_holes(
    rgb: np.ndarray,
    rgb_unmasked: np.ndarray,
    nuclei: np.ndarray,
    cyto: np.ndarray,
    *,
    min_signal: int,
    closing_size: int,
) -> np.ndarray:
    """Restore falsecolor inside small masked holes within tissue."""
    signal = np.maximum(nuclei, cyto)
    tissue = signal > min_signal
    visible = np.any(rgb > 0, axis=-1)
    region = visible & tissue
    filled = ndimage.binary_fill_holes(region)
    if closing_size > 1:
        structure = np.ones((closing_size, closing_size), dtype=bool)
        filled = ndimage.binary_closing(filled, structure=structure)
    holes = filled & ~visible & tissue
    output = rgb.copy()
    output[holes] = rgb_unmasked[holes]
    return output


def _process_one_slice(task: dict) -> dict:
    prefix = task["prefix"]
    nuclei_path = Path(task["nuclei_path"])
    cyto_path = Path(task["cyto_path"])
    output_path = Path(task["output_path"])

    started = time.time()
    nuclei = imread(str(nuclei_path))
    cyto = imread(str(cyto_path))
    rgb = render_falsecolor(
        nuclei,
        cyto,
        backend=task["backend"],
        color_key=task["color_key"],
        nuc_threshold=task["nuc_threshold"],
        cyto_threshold=task["cyto_threshold"],
        nuc_normfactor=task["nuc_normfactor"],
        cyto_normfactor=task["cyto_normfactor"],
    )
    rgb_unmasked = rgb.copy()
    rgb = apply_background_mask(
        rgb,
        nuclei,
        cyto,
        mode=task["mask_background"],
        background_threshold=task["background_threshold"],
        white_threshold=task["white_threshold"],
        pale_signal_threshold=task["pale_signal_threshold"],
        hsv_mask_val=task["hsv_mask_val"],
        hsv_min_size=task["hsv_min_size"],
    )
    if task["fill_holes"]:
        rgb = fill_internal_holes(
            rgb,
            rgb_unmasked,
            nuclei,
            cyto,
            min_signal=task["fill_min_signal"],
            closing_size=task["fill_closing_size"],
        )
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    imsave(str(tmp_path), rgb, check_contrast=False)
    tmp_path.replace(output_path)
    return {
        "prefix": prefix,
        "output_path": str(output_path),
        "elapsed_s": time.time() - started,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply falsecolor virtual staining to paired ch0/ch1 TIFF slices.",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path(
            r"Z:\YF2026061901\20260701_09_44_49_YF2026061901_CHYY_fei_Destripe_DONE\All_Channels"
        ),
        help="Directory containing paired *_ch0.tiff / *_ch1.tiff files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <input_dir>_FalseColor.",
    )
    parser.add_argument(
        "--nuclei_channel",
        type=int,
        default=1,
        help="Channel index used as nuclei channel (default: 1, ch1).",
    )
    parser.add_argument(
        "--cyto_channel",
        type=int,
        default=0,
        help="Channel index used as cytoplasm channel (default: 0, ch0).",
    )
    parser.add_argument(
        "--backend",
        choices=("gpu", "cpu"),
        default="gpu",
        help="Use rapidFalseColor on GPU or falseColor on CPU.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="CPU worker processes. Ignored when --backend gpu.",
    )
    parser.add_argument(
        "--color_key",
        choices=("HE", "IHC"),
        default="HE",
        help="falsecolor palette preset.",
    )
    parser.add_argument(
        "--nuc_threshold",
        type=int,
        default=50,
        help="Background threshold for nuclei channel.",
    )
    parser.add_argument(
        "--cyto_threshold",
        type=int,
        default=50,
        help="Background threshold for cyto channel.",
    )
    parser.add_argument(
        "--nuc_normfactor",
        type=int,
        default=5000,
        help="Normalization factor for nuclei channel (CPU falseColor default).",
    )
    parser.add_argument(
        "--cyto_normfactor",
        type=int,
        default=2000,
        help="Normalization factor for cyto channel (CPU falseColor default).",
    )
    parser.add_argument(
        "--suffix",
        default="_HE.tif",
        help="Output filename suffix appended to slice prefix.",
    )
    parser.add_argument(
        "--test",
        type=int,
        default=0,
        help="Process only the first N slice pairs (0 = disabled).",
    )
    parser.add_argument(
        "--middle",
        type=int,
        default=0,
        help="Process N slice pairs from the center of the stack.",
    )
    parser.add_argument(
        "--slice_start",
        type=int,
        default=0,
        help="0-based start index into the sorted slice list.",
    )
    parser.add_argument(
        "--slice_count",
        type=int,
        default=0,
        help="Number of slices to process from --slice_start (0 = to end).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip slices whose output already exists (default: enabled).",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mask_background",
        choices=("none", "channels", "rgb", "channels+rgb", "hsv"),
        default="channels+rgb",
        help=(
            "Set background to black after false coloring. "
            "'channels' masks pixels where max(raw ch0, ch1) is below "
            "--background_threshold. "
            "'channels+rgb' also masks near-white falsecolor output when "
            "max(raw ch0, ch1) is below --pale_signal_threshold "
            "(recommended). "
            "'rgb' masks near-white output pixels only. "
            "'hsv' uses falsecolor.maskEmpty."
        ),
    )
    parser.add_argument(
        "--background_threshold",
        type=int,
        default=300,
        help="Raw intensity threshold for --mask_background channels.",
    )
    parser.add_argument(
        "--white_threshold",
        type=int,
        default=245,
        help="RGB threshold for --mask_background rgb / channels+rgb.",
    )
    parser.add_argument(
        "--pale_signal_threshold",
        type=int,
        default=600,
        help=(
            "Only apply RGB pale masking when max(raw ch0, ch1) is at or "
            "below this value. Set 0 to mask all pale pixels."
        ),
    )
    parser.add_argument(
        "--hsv_mask_val",
        type=float,
        default=0.05,
        help="HSV saturation cutoff for --mask_background hsv.",
    )
    parser.add_argument(
        "--hsv_min_size",
        type=int,
        default=150,
        help="Minimum object size passed to falsecolor.maskEmpty.",
    )
    parser.add_argument(
        "--fill_holes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fill masked holes inside tissue using pre-mask falsecolor.",
    )
    parser.add_argument(
        "--fill_min_signal",
        type=int,
        default=300,
        help="Minimum raw signal for hole filling (default: same as background_threshold).",
    )
    parser.add_argument(
        "--fill_closing_size",
        type=int,
        default=5,
        help="Morphological closing kernel size for hole filling.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = args.input_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else Path(str(input_dir) + "_FalseColor")
    )

    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    if args.backend == "gpu" and not _gpu_available():
        print("CUDA is not available; falling back to CPU falseColor.", file=sys.stderr)
        args.backend = "cpu"

    if args.backend == "gpu" and args.workers != 1:
        print("GPU backend uses a single worker; forcing --workers 1.", file=sys.stderr)
        args.workers = 1

    pairs = discover_channel_pairs(
        input_dir,
        nuclei_channel=args.nuclei_channel,
        cyto_channel=args.cyto_channel,
    )
    if not pairs:
        print(
            f"No paired ch{args.nuclei_channel}/ch{args.cyto_channel} TIFF files found in {input_dir}",
            file=sys.stderr,
        )
        return 1

    total_pairs = len(pairs)
    if args.test > 0:
        pairs, range_start, range_end = select_slice_pairs(
            pairs, slice_start=0, slice_count=args.test, middle=0
        )
    else:
        pairs, range_start, range_end = select_slice_pairs(
            pairs,
            slice_start=args.slice_start,
            slice_count=args.slice_count,
            middle=args.middle,
        )

    resume = args.resume or args.skip_existing
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[dict] = []
    skipped = 0
    completed_prefixes: list[str] = []
    for prefix, nuclei_path, cyto_path in pairs:
        output_name = f"{prefix}{args.suffix}"
        output_path = output_dir / output_name
        if resume and output_path.exists() and output_path.stat().st_size > 0:
            skipped += 1
            completed_prefixes.append(prefix)
            continue
        tasks.append(
            {
                "prefix": prefix,
                "nuclei_path": str(nuclei_path),
                "cyto_path": str(cyto_path),
                "output_path": str(output_path),
                "backend": args.backend,
                "color_key": args.color_key,
                "nuc_threshold": args.nuc_threshold,
                "cyto_threshold": args.cyto_threshold,
                "nuc_normfactor": args.nuc_normfactor,
                "cyto_normfactor": args.cyto_normfactor,
                "mask_background": args.mask_background,
                "background_threshold": args.background_threshold,
                "white_threshold": args.white_threshold,
                "pale_signal_threshold": args.pale_signal_threshold,
                "hsv_mask_val": args.hsv_mask_val,
                "hsv_min_size": args.hsv_min_size,
                "fill_holes": args.fill_holes,
                "fill_min_signal": args.fill_min_signal,
                "fill_closing_size": args.fill_closing_size,
            }
        )

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(
        f"Stack: {total_pairs} pairs total, selected [{range_start}:{range_end}] "
        f"({len(pairs)} slices)"
    )
    print(
        f"Run: {len(tasks)} to process, {skipped} skipped, "
        f"backend={args.backend}, workers={args.workers}, "
        f"mask_background={args.mask_background}, resume={resume}, "
        f"fill_holes={args.fill_holes}"
    )

    if not tasks:
        print("Nothing to do.")
        return 0

    started = time.time()
    completed = 0

    if args.workers <= 1:
        for task in tasks:
            result = _process_one_slice(task)
            completed += 1
            completed_prefixes.append(result["prefix"])
            save_progress(output_dir, completed_prefixes)
            print(
                f"[{completed}/{len(tasks)}] {result['prefix']} "
                f"-> {result['output_path']} ({result['elapsed_s']:.1f}s)"
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_process_one_slice, task): task["prefix"]
                for task in tasks
            }
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                completed_prefixes.append(result["prefix"])
                save_progress(output_dir, completed_prefixes)
                print(
                    f"[{completed}/{len(tasks)}] {result['prefix']} "
                    f"-> {result['output_path']} ({result['elapsed_s']:.1f}s)"
                )

    save_progress(output_dir, completed_prefixes)

    elapsed = time.time() - started
    print(f"Done. Processed {completed} slices in {elapsed / 60:.1f} min.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
