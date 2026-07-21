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

# HE-style Beer-Lambert RGB absorption presets (R, G, B).
# "purple" matches reference display colors approx.
#   cytoplasm rgba(245, 179, 249), nuclei rgba(129, 4, 170)
# via OD = -ln(channel/255), cyto scaled to max=1, nuclei at moderate stain.
HE_COLOR_PRESETS: dict[str, dict[str, list[float]]] = {
    "default": {
        "nuclei": [0.17, 0.27, 0.105],
        "cyto": [0.05, 1.0, 0.54],
    },
    "blue": {
        "nuclei": [0.24, 0.32, 0.08],
        "cyto": [0.05, 1.0, 0.54],
    },
    "purple": {
        "nuclei": [0.085, 0.519, 0.051],
        "cyto": [0.113, 1.0, 0.067],
    },
}


def resolve_color_settings(color_key: str, nuclei_hue: str) -> dict[str, list[float]]:
    settings = falsecolor.getColorSettings(key=color_key)
    if color_key.upper() == "IHC":
        return settings
    preset = HE_COLOR_PRESETS.get(nuclei_hue, HE_COLOR_PRESETS["purple"])
    settings = dict(settings)
    settings["nuclei"] = list(preset["nuclei"])
    settings["cyto"] = list(preset["cyto"])
    return settings


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


def build_flat_field_2d(
    image: np.ndarray,
    *,
    tile_size: int = 256,
    bg_threshold: int = 50,
    scale: float = 1.0,
) -> np.ndarray:
    """Per-tile foreground median map, upsampled to full plane resolution."""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"Flat field expects a 2D image, got shape {img.shape}")
    if tile_size < 8:
        raise ValueError("--flatfield_tile_size must be >= 8")
    if scale <= 0:
        raise ValueError("flat-field scale must be > 0")

    midrange, background = falsecolor.getBackgroundLevels(img, threshold=bg_threshold)
    height, width = img.shape
    n_rows = int(np.ceil(height / tile_size))
    n_cols = int(np.ceil(width / tile_size))
    down = np.zeros((n_rows, n_cols), dtype=np.float64)

    for row in range(n_rows):
        r0 = row * tile_size
        r1 = min((row + 1) * tile_size, height)
        for col in range(n_cols):
            c0 = col * tile_size
            c1 = min((col + 1) * tile_size, width)
            roi = img[r0:r1, c0:c1]
            foreground = roi[roi > background]
            down[row, col] = (
                float(np.median(foreground)) if foreground.size else float(midrange)
            )

    field = ndimage.zoom(
        down,
        (height / max(n_rows, 1), width / max(n_cols, 1)),
        order=1,
        mode="nearest",
    )
    field = np.asarray(field[:height, :width], dtype=np.float64) * float(scale)
    return np.maximum(field, 1.0)


def sharpen_image(image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Edge sharpening used by Liu-lab color_script (pre-coloring)."""
    img = np.asarray(image, dtype=np.float64)
    if alpha <= 0:
        return img
    if _gpu_available():
        return np.asarray(falsecolor.sharpenImage(img, alpha=float(alpha)), dtype=np.float64)

    hkernel = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]])
    vkernel = np.array([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]])
    vertical = ndimage.convolve(img, vkernel, mode="nearest")
    horizontal = ndimage.convolve(img, hkernel, mode="nearest")
    return img + float(alpha) * np.sqrt(vertical**2 + horizontal**2)


def apply_nuclei_clahe(
    image: np.ndarray,
    *,
    tile_grid_size: tuple[int, int] = (8, 8),
    clip_limit: float = 1.5,
) -> np.ndarray:
    """Optional CLAHE on nuclei, matching FC_CLAHE.py defaults."""
    return np.asarray(
        falsecolor.applyCLAHE(
            np.asarray(image, dtype=np.uint16),
            tileGridSize=tile_grid_size,
            clipLimit=float(clip_limit),
        ),
        dtype=np.float64,
    )


def subtract_asymmetric_background(
    nuclei: np.ndarray,
    cyto: np.ndarray,
    *,
    nuc_threshold: int,
    cyto_threshold: int,
    nuc_factor: float = 0.5,
    cyto_factor: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Colleague-style bg subtract: nuclei -= 0.5*bkg, cyto -= 3*bkg."""
    nuclei_f = np.asarray(nuclei, dtype=np.float64)
    cyto_f = np.asarray(cyto, dtype=np.float64)
    if nuc_factor > 0:
        _, bkg_nuc = falsecolor.getBackgroundLevels(nuclei_f, threshold=nuc_threshold)
        nuclei_f = np.clip(nuclei_f - float(nuc_factor) * bkg_nuc, 0, 65535)
    if cyto_factor > 0:
        _, bkg_cyto = falsecolor.getBackgroundLevels(cyto_f, threshold=cyto_threshold)
        cyto_f = np.clip(cyto_f - float(cyto_factor) * bkg_cyto, 0, 65535)
    return nuclei_f, cyto_f


def preprocess_colleague_tricks(
    nuclei: np.ndarray,
    cyto: np.ndarray,
    *,
    nuc_threshold: int,
    cyto_threshold: int,
    flatfield: bool,
    bg_nuc_factor: float,
    bg_cyto_factor: float,
    sharpen_alpha: float,
    clahe: bool,
    clahe_clip_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply colleague color_script / FC_CLAHE preprocessing.

    Order matches their scripts: optional CLAHE (nuclei) -> asymmetric bg
    subtract (flatfield path) -> sharpen both channels.
    """
    nuclei_f = np.asarray(nuclei, dtype=np.float64)
    cyto_f = np.asarray(cyto, dtype=np.float64)

    if clahe:
        nuclei_f = apply_nuclei_clahe(nuclei_f, clip_limit=clahe_clip_limit)

    if flatfield and (bg_nuc_factor > 0 or bg_cyto_factor > 0):
        nuclei_f, cyto_f = subtract_asymmetric_background(
            nuclei_f,
            cyto_f,
            nuc_threshold=nuc_threshold,
            cyto_threshold=cyto_threshold,
            nuc_factor=bg_nuc_factor,
            cyto_factor=bg_cyto_factor,
        )

    if sharpen_alpha > 0:
        nuclei_f = sharpen_image(nuclei_f, alpha=sharpen_alpha)
        cyto_f = sharpen_image(cyto_f, alpha=sharpen_alpha)
        nuclei_f = np.clip(nuclei_f, 0, 65535)
        cyto_f = np.clip(cyto_f, 0, 65535)

    return nuclei_f, cyto_f


def render_falsecolor(
    nuclei: np.ndarray,
    cyto: np.ndarray,
    *,
    backend: str,
    color_key: str,
    nuclei_hue: str,
    nuc_threshold: int,
    cyto_threshold: int,
    nuc_normfactor: int,
    cyto_normfactor: int,
    flatfield: bool = False,
    flatfield_tile_size: int = 256,
    flatfield_scale: float = 1.0,
    flatfield_nuc_scale: float = 1.5,
    flatfield_cyto_scale: float = 3.72,
) -> np.ndarray:
    color_settings = resolve_color_settings(color_key, nuclei_hue)
    if color_key.upper() == "IHC":
        cyto_settings = color_settings["anti"]
    else:
        cyto_settings = color_settings["cyto"]

    if flatfield:
        if backend != "gpu":
            raise ValueError("Flat field mode requires --backend gpu")
        nuc_field = build_flat_field_2d(
            nuclei,
            tile_size=flatfield_tile_size,
            bg_threshold=nuc_threshold,
            scale=float(flatfield_scale) * float(flatfield_nuc_scale),
        )
        cyto_field = build_flat_field_2d(
            cyto,
            tile_size=flatfield_tile_size,
            bg_threshold=cyto_threshold,
            scale=float(flatfield_scale) * float(flatfield_cyto_scale),
        )
        return falsecolor.rapidFalseColor(
            nuclei,
            cyto,
            color_settings["nuclei"],
            cyto_settings,
            nuc_normfactor=nuc_field,
            cyto_normfactor=cyto_field,
            run_FlatField_nuc=True,
            run_FlatField_cyto=True,
            nuc_bg_threshold=nuc_threshold,
            cyto_bg_threshold=cyto_threshold,
        )

    if backend == "gpu":
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
        color_settings=color_settings,
    )


def sharpen_rgb(rgb: np.ndarray, amount: float, sigma: float = 1.0) -> np.ndarray:
    """Unsharp-mask RGB while leaving pure-black background untouched."""
    if amount <= 0:
        return rgb
    image = rgb.astype(np.float32)
    blurred = ndimage.gaussian_filter(image, sigma=(sigma, sigma, 0))
    sharp = image + float(amount) * (image - blurred)
    mask = np.any(rgb > 0, axis=-1)
    output = rgb.copy()
    output[mask] = np.clip(sharp[mask], 0, 255).astype(np.uint8)
    return output


def adjust_rgb_contrast(rgb: np.ndarray, contrast: float, pivot: float = 220.0) -> np.ndarray:
    """Increase/decrease stain contrast around near-white H&E background."""
    if abs(contrast - 1.0) < 1e-6:
        return rgb
    image = rgb.astype(np.float32)
    adjusted = pivot + float(contrast) * (image - pivot)
    mask = np.any(rgb > 0, axis=-1)
    output = rgb.copy()
    output[mask] = np.clip(adjusted[mask], 0, 255).astype(np.uint8)
    return output


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
    nuclei_raw = imread(str(nuclei_path))
    cyto_raw = imread(str(cyto_path))
    nuclei, cyto = preprocess_colleague_tricks(
        nuclei_raw,
        cyto_raw,
        nuc_threshold=task["nuc_threshold"],
        cyto_threshold=task["cyto_threshold"],
        flatfield=task["flatfield"],
        bg_nuc_factor=task["bg_nuc_factor"],
        bg_cyto_factor=task["bg_cyto_factor"],
        sharpen_alpha=task["sharpen_alpha"],
        clahe=task["clahe"],
        clahe_clip_limit=task["clahe_clip_limit"],
    )
    rgb = render_falsecolor(
        nuclei,
        cyto,
        backend=task["backend"],
        color_key=task["color_key"],
        nuclei_hue=task["nuclei_hue"],
        nuc_threshold=task["nuc_threshold"],
        cyto_threshold=task["cyto_threshold"],
        nuc_normfactor=task["nuc_normfactor"],
        cyto_normfactor=task["cyto_normfactor"],
        flatfield=task["flatfield"],
        flatfield_tile_size=task["flatfield_tile_size"],
        flatfield_scale=task["flatfield_scale"],
        flatfield_nuc_scale=task["flatfield_nuc_scale"],
        flatfield_cyto_scale=task["flatfield_cyto_scale"],
    )
    rgb_unmasked = rgb.copy()
    rgb = apply_background_mask(
        rgb,
        nuclei_raw,
        cyto_raw,
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
            nuclei_raw,
            cyto_raw,
            min_signal=task["fill_min_signal"],
            closing_size=task["fill_closing_size"],
        )
    rgb = adjust_rgb_contrast(rgb, contrast=task["contrast"])
    rgb = sharpen_rgb(rgb, amount=task["sharpen"], sigma=task["sharpen_sigma"])
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
        "--nuclei_hue",
        choices=tuple(HE_COLOR_PRESETS),
        default="purple",
        help=(
            "HE palette bias (default: purple). "
            "purple ≈ cyto rgba(245,179,249) + nuclei rgba(129,4,170); "
            "blue = cooler hematoxylin; default = package HE."
        ),
    )
    parser.add_argument(
        "--nuc_threshold",
        type=int,
        default=100,
        help="Background threshold for nuclei channel.",
    )
    parser.add_argument(
        "--cyto_threshold",
        type=int,
        default=100,
        help="Background threshold for cyto channel.",
    )
    parser.add_argument(
        "--nuc_normfactor",
        type=int,
        default=8200,
        help="Normalization factor for nuclei channel (ignored when --flatfield).",
    )
    parser.add_argument(
        "--cyto_normfactor",
        type=int,
        default=2100,
        help="Normalization factor for cyto channel (ignored when --flatfield).",
    )
    parser.add_argument(
        "--flatfield",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable local tile flat-field equalization on GPU "
            "(per-plane tile medians; disables global nuc/cyto_normfactor)."
        ),
    )
    parser.add_argument(
        "--flatfield_tile_size",
        type=int,
        default=256,
        help="Tile size for --flatfield local intensity map (default: 256).",
    )
    parser.add_argument(
        "--flatfield_scale",
        type=float,
        default=1.0,
        help=(
            "Extra global multiplier on flat-field maps (default: 1). "
            "Final field scale = flatfield_scale * channel scale."
        ),
    )
    parser.add_argument(
        "--flatfield_nuc_scale",
        type=float,
        default=1.5,
        help="Nuclei flat-field beta from colleague color_script (default: 1.5).",
    )
    parser.add_argument(
        "--flatfield_cyto_scale",
        type=float,
        default=3.72,
        help="Cyto flat-field beta from colleague color_script (default: 3.72).",
    )
    parser.add_argument(
        "--sharpen_alpha",
        type=float,
        default=0.5,
        help=(
            "Pre-coloring edge sharpen strength (colleague sharpenImage; "
            "default: 0.5). Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--bg_nuc_factor",
        type=float,
        default=0.5,
        help=(
            "With --flatfield, subtract factor*bkg from nuclei before coloring "
            "(colleague default: 0.5)."
        ),
    )
    parser.add_argument(
        "--bg_cyto_factor",
        type=float,
        default=3.0,
        help=(
            "With --flatfield, subtract factor*bkg from cyto before coloring "
            "(colleague default: 3.0)."
        ),
    )
    parser.add_argument(
        "--clahe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optional nuclei CLAHE before coloring (FC_CLAHE.py style).",
    )
    parser.add_argument(
        "--clahe_clip_limit",
        type=float,
        default=1.5,
        help="CLAHE clipLimit when --clahe is set (default: 1.5).",
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=1.0,
        help=(
            "Post false-color contrast around near-white background "
            "(1=unchanged; try 1.2-1.6 for punchier stain)."
        ),
    )
    parser.add_argument(
        "--sharpen",
        type=float,
        default=0.0,
        help="Unsharp-mask amount after coloring (0=off; try 0.4-1.0).",
    )
    parser.add_argument(
        "--sharpen_sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma for --sharpen (default: 1.0).",
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
        default=250,
        help="Raw intensity threshold for --mask_background channels.",
    )
    parser.add_argument(
        "--white_threshold",
        type=int,
        default=250,
        help="RGB threshold for --mask_background rgb / channels+rgb.",
    )
    parser.add_argument(
        "--pale_signal_threshold",
        type=int,
        default=2000,
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
        default=False,
        help="Fill masked holes inside tissue using pre-mask falsecolor.",
    )
    parser.add_argument(
        "--fill_min_signal",
        type=int,
        default=200,
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
                "nuclei_hue": args.nuclei_hue,
                "nuc_threshold": args.nuc_threshold,
                "cyto_threshold": args.cyto_threshold,
                "nuc_normfactor": args.nuc_normfactor,
                "cyto_normfactor": args.cyto_normfactor,
                "flatfield": args.flatfield,
                "flatfield_tile_size": args.flatfield_tile_size,
                "flatfield_scale": args.flatfield_scale,
                "flatfield_nuc_scale": args.flatfield_nuc_scale,
                "flatfield_cyto_scale": args.flatfield_cyto_scale,
                "sharpen_alpha": args.sharpen_alpha,
                "bg_nuc_factor": args.bg_nuc_factor,
                "bg_cyto_factor": args.bg_cyto_factor,
                "clahe": args.clahe,
                "clahe_clip_limit": args.clahe_clip_limit,
                "contrast": args.contrast,
                "sharpen": args.sharpen,
                "sharpen_sigma": args.sharpen_sigma,
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
        f"nuclei_hue={args.nuclei_hue}, flatfield={args.flatfield}, "
        f"flatfield_tile_size={args.flatfield_tile_size}, "
        f"flatfield_scale={args.flatfield_scale}, "
        f"flatfield_nuc_scale={args.flatfield_nuc_scale}, "
        f"flatfield_cyto_scale={args.flatfield_cyto_scale}, "
        f"sharpen_alpha={args.sharpen_alpha}, clahe={args.clahe}, "
        f"fill_holes={args.fill_holes}"
    )

    if args.flatfield and args.backend != "gpu":
        print("Flat field requires --backend gpu", file=sys.stderr)
        return 1

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
