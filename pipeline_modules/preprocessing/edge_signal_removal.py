"""Remove non-specific bright sheet-like signals at brain edges in 2D TIFF slices.

Uses the atlas label TIFF stack (warped to sample space) to define the brain
boundary per slice, then suppresses bright plate-like structures in a 2D edge
band. The output is a folder of cleaned TIFF slices.

CLI::

    micromamba run -n yifu python -m pipeline_modules.preprocessing.edge_signal_removal \
        --input_dir ch1_preprocessed --label_dir upsampled_atlas_label \
        --output_dir ch1_preprocessed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from tqdm import tqdm as _tqdm

logger = logging.getLogger(__name__)

try:
    from pipeline_modules.preprocessing.preprocessor import Preprocessor, apply_processing_steps
except ImportError:  # pragma: no cover - fallback for package-relative execution
    from .preprocessor import Preprocessor, apply_processing_steps


def _list_tiff_files(path: Path) -> list[Path]:
    files = sorted(path.glob("*.tif*"))
    if not files:
        raise FileNotFoundError(f"No TIFF files found in {path}")
    return files


def _edge_mask_2d(label_img: np.ndarray, edge_width_px: int) -> np.ndarray:
    """Create a 2D edge band around the brain boundary."""
    from scipy import ndimage as ndi

    if label_img.ndim > 2:
        label_img = np.squeeze(label_img)
    if label_img.ndim != 2:
        raise ValueError(f"Expected a 2D label slice, got shape {label_img.shape}")

    brain = label_img.astype(bool)
    if not brain.any() or edge_width_px <= 0:
        return np.zeros_like(brain, dtype=np.float32)

    inward_px = edge_width_px // 2
    outward_px = edge_width_px - inward_px
    struct = ndi.generate_binary_structure(2, 1)

    eroded = (
        ndi.binary_erosion(brain, structure=struct, iterations=inward_px)
        if inward_px > 0
        else brain.copy()
    )
    dilated = (
        ndi.binary_dilation(brain, structure=struct, iterations=outward_px)
        if outward_px > 0
        else brain.copy()
    )

    return ((brain & ~eroded) | (dilated & ~brain)).astype(np.float32)


def _compute_suppression_mask_2d(
    image: np.ndarray,
    edge_mask: np.ndarray,
    brightness_pct: float,
    smooth_sigma: float,
    min_area_px: int,
) -> np.ndarray:
    """Build a 2D float mask [0,1]: 1.0 = fully suppressed, 0.0 = untouched."""
    from scipy import ndimage as ndi

    bright_threshold = np.percentile(image, brightness_pct)
    bright_edge = (image >= bright_threshold) & (edge_mask > 0)

    labels, num_labels = ndi.label(bright_edge)
    if num_labels == 0:
        return np.zeros_like(edge_mask, dtype=np.float32)

    areas = np.bincount(labels.ravel())
    keep_labels = np.flatnonzero(areas > min_area_px)
    keep_labels = keep_labels[keep_labels != 0]
    if keep_labels.size == 0:
        return np.zeros_like(edge_mask, dtype=np.float32)

    suppress = np.isin(labels, keep_labels).astype(np.float32)
    if smooth_sigma > 0:
        suppress = ndi.gaussian_filter(suppress, sigma=smooth_sigma)
    return np.clip(suppress, 0, 1)


def _clean_slice(
    image: np.ndarray,
    label_img: np.ndarray,
    *,
    edge_width_px: int,
    suppression_weight: float,
    brightness_pct: float,
    smooth_sigma: float,
    min_area_px: int,
) -> np.ndarray:
    if image.ndim > 2:
        image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D signal slice, got shape {image.shape}")

    dtype_in = image.dtype
    edge_mask = _edge_mask_2d(label_img, edge_width_px=edge_width_px)
    suppress_mask = _compute_suppression_mask_2d(
        image,
        edge_mask,
        brightness_pct=brightness_pct,
        smooth_sigma=smooth_sigma,
        min_area_px=min_area_px,
    )
    suppress_mask = np.clip(suppress_mask * suppression_weight, 0, 1)

    cleaned = np.clip(image.astype(np.float32) * (1.0 - suppress_mask), 0, None)
    if np.issubdtype(dtype_in, np.integer):
        max_val = np.iinfo(dtype_in).max
        return np.clip(cleaned, 0, max_val).astype(dtype_in)
    return cleaned.astype(dtype_in)


def _process_slice_task(args: tuple[Path, Path, Path, dict[str, Any], list[tuple[str, dict[str, Any]]]]) -> dict[str, Any]:
    signal_path, label_path, output_path, cfg, preprocessing_steps = args
    try:
        image = tifffile.imread(str(signal_path))
        if preprocessing_steps:
            image = apply_processing_steps(image, preprocessing_steps)
        label_img = tifffile.imread(str(label_path))
        cleaned = _clean_slice(image, label_img, **cfg)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(output_path), cleaned, compression=None)
        return {"success": True, "input": str(signal_path), "label": str(label_path), "output": str(output_path)}
    except Exception as exc:  # pragma: no cover - exercised via parent process
        return {
            "success": False,
            "input": str(signal_path),
            "label": str(label_path),
            "output": str(output_path),
            "error": str(exc),
        }


def remove_edge_signal(
    input_dir: str | Path,
    label_dir: str | Path,
    output_dir: str | Path,
    *,
    edge_width_px: int = 20,
    suppression_weight: float = 0.8,
    brightness_pct: float = 90.0,
    smooth_sigma: float = 5.0,
    min_area_px: int = 50,
    max_workers: int | None = None,
    resume: bool = True,
    preprocessing_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Remove non-specific bright edge signal from a TIFF slice folder."""
    started_at = time.time()
    input_path = Path(input_dir)
    label_path = Path(label_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        raise NotADirectoryError(f"Input TIFF directory not found: {input_path}")
    if not label_path.is_dir():
        raise NotADirectoryError(f"Label TIFF directory not found: {label_path}")

    signal_files = _list_tiff_files(input_path)
    label_files = _list_tiff_files(label_path)
    if len(signal_files) != len(label_files):
        raise ValueError(
            f"Signal TIFF count ({len(signal_files)}) != label TIFF count ({len(label_files)})"
        )

    output_path.mkdir(parents=True, exist_ok=True)
    cfg = {
        "edge_width_px": edge_width_px,
        "suppression_weight": suppression_weight,
        "brightness_pct": brightness_pct,
        "smooth_sigma": smooth_sigma,
        "min_area_px": min_area_px,
    }
    preprocessing_steps = []
    if preprocessing_config:
        preprocessing_steps = [
            (name, dict(step_cfg))
            for name, step_cfg in Preprocessor(preprocessing_config).steps
        ]

    tasks = []
    skipped_existing = 0
    for signal_file, label_file in zip(signal_files, label_files):
        output_file = output_path / signal_file.name
        if resume and output_file.exists():
            skipped_existing += 1
            continue
        tasks.append((signal_file, label_file, output_file, cfg, preprocessing_steps))

    workers = max_workers or 1
    processed = 0
    failed = 0
    errors: list[dict[str, Any]] = []

    if tasks:
        logger.info("Processing %d TIFF slices with %d workers", len(tasks), workers)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_slice_task, task): task[0].name for task in tasks}
            for future in _tqdm(
                as_completed(futures),
                total=len(futures),
                desc="2D edge signal removal",
                unit="slice",
                file=sys.stderr,
            ):
                result = future.result()
                if result["success"]:
                    processed += 1
                else:
                    failed += 1
                    errors.append(result)
                    logger.error("Failed processing %s: %s", futures[future], result.get("error", "unknown error"))

    duration = time.time() - started_at
    return {
        "success": failed == 0,
        "input_dir": str(input_path),
        "label_dir": str(label_path),
        "output_dir": str(output_path),
        "total_files": len(signal_files),
        "processed_files": processed,
        "skipped_existing": skipped_existing,
        "failed_files": failed,
        "edge_width_px": edge_width_px,
        "suppression_weight": suppression_weight,
        "brightness_pct": brightness_pct,
        "smooth_sigma": smooth_sigma,
        "min_area_px": min_area_px,
        "preprocessing_steps": [name for name, _ in preprocessing_steps],
        "duration_seconds": duration,
        "errors": errors[:10],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove non-specific bright sheet-like signals at brain edges from 2D TIFF slices",
    )
    parser.add_argument("--input_dir", required=True, help="Input signal TIFF folder")
    parser.add_argument("--label_dir", required=True, help="Warped atlas label TIFF folder")
    parser.add_argument("--output_dir", required=True, help="Output cleaned TIFF folder")
    parser.add_argument("--edge_width_px", type=int, default=20,
                        help="Pixel width of the brain edge band (default: 20)")
    parser.add_argument("--suppression_weight", type=float, default=0.8,
                        help="Blend weight 0~1 (default: 0.8)")
    parser.add_argument("--brightness_pct", type=float, default=90.0,
                        help="Only suppress pixels above this brightness percentile per slice (default: 90)")
    parser.add_argument("--smooth_sigma", type=float, default=5.0,
                        help="Gaussian blur on suppression mask per slice (default: 5)")
    parser.add_argument("--min_area_px", type=int, default=50,
                        help="Suppress only bright edge objects larger than this area in pixels (default: 50)")
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--no_resume", action="store_true", help="Reprocess existing output TIFF files")
    parser.add_argument("--preprocess_config", default=None,
                        help="Optional full JSON config; enabled 2D preprocessing steps run before edge removal")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    preprocessing_config = None
    if args.preprocess_config:
        with open(args.preprocess_config, "r", encoding="utf-8") as fh:
            preprocessing_config = json.load(fh).get("preprocessing", {})
    result = remove_edge_signal(
        args.input_dir,
        args.label_dir,
        args.output_dir,
        edge_width_px=args.edge_width_px,
        suppression_weight=args.suppression_weight,
        brightness_pct=args.brightness_pct,
        smooth_sigma=args.smooth_sigma,
        min_area_px=args.min_area_px,
        max_workers=args.max_workers,
        resume=not args.no_resume,
        preprocessing_config=preprocessing_config,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
