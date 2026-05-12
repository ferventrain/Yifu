"""Remove non-specific bright sheet-like signals at brain edges.

Uses the atlas label (warped to sample space) to define the brain
boundary, then suppresses bright plate-like structures in an edge
band using Hessian analysis.

CLI::

    micromamba run -n yifu python -m pipeline_modules.preprocessing.edge_signal_removal \\
        --input_zarr ch1.zarr --label_zarr upsampled_atlas_label.zarr \\
        --output_zarr ch1_clean.zarr
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm as _tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_array(z) -> Any:
    """Resolve a Zarr group to the single array it wraps."""
    import zarr

    if isinstance(z, zarr.Array):
        return z
    if "0" in z:
        return z["0"]
    arrays = list(z.arrays())
    if len(arrays) == 1:
        return arrays[0][1]
    raise KeyError(f"Zarr group has {len(arrays)} arrays; cannot determine which to use")


def _brain_edge_mask(label_vol: np.ndarray, edge_width_px: int) -> np.ndarray:
    """Create a float weight mask: 1.0 inside the edge band, 0.0 elsewhere.

    The edge band is the outer *edge_width_px* voxels of the brain
    (non-zero label region).
    """
    from scipy import ndimage as ndi

    brain = label_vol.astype(bool)
    if not brain.any():
        return np.zeros_like(label_vol, dtype=np.float32)

    # Erode inward to get interior; difference = outer shell
    struct = ndi.generate_binary_structure(3, 1)
    eroded = ndi.binary_erosion(brain, structure=struct, iterations=edge_width_px)
    edge_band = brain & ~eroded
    return edge_band.astype(np.float32)


def _plate_response(volume: np.ndarray) -> np.ndarray:
    """3D Hessian plate response (suppress sheet-like structures).

    Frangi with alpha=0.5, beta=0.25 emphasises plate-like response
    over tubular.  We then invert: high response = likely edge noise.
    """
    from skimage.filters import frangi

    return frangi(
        volume,
        sigmas=[4.0, 8.0, 16.0],
        black_ridges=False,
        alpha_sq=0.25,   # plate > tube sensitivity
        beta_sq=0.0625,
        mode="constant",
        cval=0,
    )


def _compute_suppression_mask(
    volume: np.ndarray,
    edge_mask: np.ndarray,
    brightness_pct: float,
    smooth_sigma: float,
) -> np.ndarray:
    """Build a 3D float mask [0,1]: 1.0 = fully suppressed, 0.0 = untouched."""
    from scipy import ndimage as ndi

    br = _plate_response(volume)
    br_norm = br / (br.max() + 1e-12)

    # Only suppress if above brightness percentile
    bright_threshold = np.percentile(volume, brightness_pct)
    bright = (volume >= bright_threshold).astype(np.float32)

    suppress = edge_mask * br_norm * bright
    if smooth_sigma > 0:
        suppress = ndi.gaussian_filter(suppress, sigma=smooth_sigma)
    return np.clip(suppress, 0, 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def remove_edge_signal(
    input_zarr: str | Path,
    label_zarr: str | Path,
    output_zarr: str | Path,
    *,
    edge_width_px: int = 20,
    suppression_weight: float = 0.8,
    brightness_pct: float = 90.0,
    smooth_sigma: float = 5.0,
    slab_depth: int = 32,
    export_tiff: str | None = None,
) -> dict[str, Any]:
    """Remove non-specific bright signals at the brain edge.

    Parameters
    ----------
    input_zarr:
        Path to the signal Zarr volume (3D).
    label_zarr:
        Path to the atlas label Zarr (same space as *input_zarr*).
    output_zarr:
        Output path (created / overwritten).
    edge_width_px:
        How many voxels from the brain edge to consider.
    suppression_weight:
        Blend strength: 0 = no effect, 1 = fully zero out edge signal.
    brightness_pct:
        Only suppress voxels above this brightness percentile of the volume.
    smooth_sigma:
        Gaussian blur on the suppression mask to avoid hard boundaries.
    slab_depth:
        Z-slices per processing batch (controls memory).
    export_tiff:
        If given, export the output to TIFF slices in this directory.

    Returns
    -------
    dict
        Processing summary.
    """
    import zarr
    from scipy import ndimage as ndi

    started_at = time.time()
    input_path = Path(input_zarr)
    label_path = Path(label_zarr)
    output_path = Path(output_zarr)

    z_in = zarr.open(str(input_path), mode="r")
    arr_in = _get_array(z_in)
    shape = arr_in.shape
    dtype_in = arr_in.dtype

    if len(shape) != 3:
        raise ValueError(f"Expected a 3D signal Zarr, got shape {shape}")

    z_lab = zarr.open(str(label_path), mode="r")
    arr_lab = _get_array(z_lab)

    if arr_lab.shape != shape:
        raise ValueError(
            f"Label Zarr shape {arr_lab.shape} != signal shape {shape}. "
            "Ensure atlas label is already warped to sample space."
        )

    logger.info("Label shape: %s", arr_lab.shape)

    # Full-volume edge mask (lightweight; label is typically small after downsample + warp)
    edge_mask = _brain_edge_mask(arr_lab, edge_width_px=edge_width_px)
    logger.info("Edge band voxels: %d / %d (%.1f%%)", edge_mask.sum(), np.prod(shape),
                100 * edge_mask.sum() / np.prod(shape))

    # Build suppression mask on a coarse downsample to save time
    downsample_factor = max(1, int(np.round(arr_in.shape[0] / 256)))
    if downsample_factor > 1:
        coarse = ndi.zoom(arr_in, (1 / downsample_factor,) * 3, order=1)
        coarse_edge = ndi.zoom(edge_mask, (1 / downsample_factor,) * 3, order=0)
        coarse_suppress = _compute_suppression_mask(
            coarse, coarse_edge, brightness_pct, smooth_sigma / downsample_factor,
        )
        suppress_mask = ndi.zoom(coarse_suppress, downsample_factor, order=1)
    else:
        suppress_mask = _compute_suppression_mask(
            arr_in, edge_mask, brightness_pct, smooth_sigma,
        )

    suppress_mask = np.clip(suppress_mask * suppression_weight, 0, 1)
    logger.info("Suppression mask: min=%.3f max=%.3f mean=%.3f",
                suppress_mask.min(), suppress_mask.max(), suppress_mask.mean())

    # Create output Zarr inheriting chunking
    in_chunks = list(arr_in.chunks)
    in_chunks[0] = min(in_chunks[0], slab_depth)

    zarr.open(
        str(output_path), mode="w",
        shape=shape, chunks=tuple(in_chunks), dtype=dtype_in,
    )
    z_out = zarr.open(str(output_path), mode="a")
    out_arr = _get_array(z_out)

    # Process in slabs
    depth = shape[0]
    total_slabs = max(1, (depth + slab_depth - 1) // slab_depth)
    logger.info("Processing %d slabs of depth %d", total_slabs, slab_depth)

    for slab_idx in _tqdm(
        range(total_slabs),
        desc="Edge signal removal",
        unit="slab",
        leave=False,
        total=total_slabs,
        file=sys.stderr,
    ):
        z_start = slab_idx * slab_depth
        z_end = min(z_start + slab_depth, depth)
        slab = arr_in[z_start:z_end]
        mask_slice = suppress_mask[z_start:z_end]
        cleaned = np.clip(slab.astype(np.float32) * (1.0 - mask_slice), 0, None)
        if np.issubdtype(dtype_in, np.integer):
            max_val = np.iinfo(dtype_in).max
            cleaned = np.clip(cleaned, 0, max_val).astype(dtype_in)
        out_arr[z_start:z_end] = cleaned

    duration = time.time() - started_at
    result = {
        "success": True,
        "input": str(input_path),
        "label": str(label_path),
        "output": str(output_path),
        "shape": list(shape),
        "edge_width_px": edge_width_px,
        "suppression_weight": suppression_weight,
        "brightness_pct": brightness_pct,
        "smooth_sigma": smooth_sigma,
        "duration_seconds": duration,
    }

    if export_tiff:
        import tifffile
        tiff_dir = Path(export_tiff)
        tiff_dir.mkdir(parents=True, exist_ok=True)
        out_vol = out_arr[:]
        for idx in range(out_vol.shape[0]):
            tiff_path = tiff_dir / f"slice_{idx:04d}.tif"
            tifffile.imwrite(str(tiff_path), out_vol[idx])
        result["tiff_export"] = str(tiff_dir)
        logger.info("TIFF exported to %s", tiff_dir)

    logger.info("Edge signal removal done: %.1f s", duration)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove non-specific bright sheet-like signals at brain edges",
    )
    parser.add_argument("--input_zarr", required=True)
    label_choices = [
        "s:/Arivis_Analysis/pizang/upsampled_atlas_label.zarr",
    ]
    parser.add_argument("--label_zarr", required=True)
    parser.add_argument("--output_zarr", required=True)
    parser.add_argument("--edge_width_px", type=int, default=20,
                        help="Voxel width of the brain edge band (default: 20)")
    parser.add_argument("--suppression_weight", type=float, default=0.8,
                        help="Blend weight 0~1 (default: 0.8)")
    parser.add_argument("--brightness_pct", type=float, default=90.0,
                        help="Only suppress voxels above this brightness percentile (default: 90)")
    parser.add_argument("--smooth_sigma", type=float, default=5.0,
                        help="Gaussian blur on suppression mask (default: 5)")
    parser.add_argument("--slab_depth", type=int, default=32)
    parser.add_argument("--export_tiff", default=None,
                        help="Export result as TIFF slices to this directory")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    result = remove_edge_signal(
        args.input_zarr,
        args.label_zarr,
        args.output_zarr,
        edge_width_px=args.edge_width_px,
        suppression_weight=args.suppression_weight,
        brightness_pct=args.brightness_pct,
        smooth_sigma=args.smooth_sigma,
        slab_depth=args.slab_depth,
        export_tiff=args.export_tiff,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
