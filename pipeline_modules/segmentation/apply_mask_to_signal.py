"""Apply a segmentation mask to a signal volume, filtering objects per Zarr chunk.

For each Zarr chunk (3D block), connected components are dropped if:
  - 3D bbox aspect ratio (max_side / min_side) > ``--max_aspect_ratio``, or
  - volume > ``--max_voxels``
before the mask is applied to the signal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cc3d
import numpy as np
from tqdm import tqdm

try:
    from pipeline_modules.segmentation.zarr_utils import (
        create_output_zarr,
        export_zarr_to_tiff,
        open_zarr_dataset,
    )
except ImportError:  # pragma: no cover
    from .zarr_utils import (
        create_output_zarr,
        export_zarr_to_tiff,
        open_zarr_dataset,
    )


def _chunk_slices(shape: tuple[int, int, int], chunks: tuple[int, int, int]) -> list[tuple[slice, slice, slice]]:
    depth, height, width = (int(shape[0]), int(shape[1]), int(shape[2]))
    cz, cy, cx = (int(chunks[0]), int(chunks[1]), int(chunks[2]))
    out: list[tuple[slice, slice, slice]] = []
    for z0 in range(0, depth, cz):
        for y0 in range(0, height, cy):
            for x0 in range(0, width, cx):
                out.append(
                    (
                        slice(z0, min(z0 + cz, depth)),
                        slice(y0, min(y0 + cy, height)),
                        slice(x0, min(x0 + cx, width)),
                    )
                )
    return out


def _extent_ratio(bbox: tuple[slice, slice, slice]) -> float:
    extents = [max(int(sl.stop) - int(sl.start), 1) for sl in bbox]
    return float(max(extents)) / float(min(extents))


def filter_elongated_components_chunk(
    mask_chunk: np.ndarray,
    *,
    max_aspect_ratio: float,
    max_voxels: int,
) -> tuple[np.ndarray, int, int, int]:
    """Keep 3D CCs inside one chunk that pass aspect and volume checks."""
    binary = np.asarray(mask_chunk, dtype=np.uint8) > 0
    if not binary.any():
        return binary.astype(np.uint8), 0, 0, 0

    labels = cc3d.connected_components(binary.astype(np.uint8), connectivity=26)
    stats = cc3d.statistics(labels)
    voxel_counts = stats["voxel_counts"]
    bounding_boxes = stats["bounding_boxes"]

    keep = np.zeros(len(voxel_counts), dtype=bool)
    removed_aspect = 0
    removed_volume = 0
    for label_idx in range(1, len(voxel_counts)):
        count = int(voxel_counts[label_idx])
        if count > max_voxels:
            removed_volume += 1
            continue
        ratio = _extent_ratio(bounding_boxes[label_idx])
        if ratio > max_aspect_ratio:
            removed_aspect += 1
            continue
        keep[label_idx] = True

    filtered = keep[labels].astype(np.uint8)
    return filtered, int(len(voxel_counts) - 1), int(removed_aspect), int(removed_volume)


def _list_tiff_names(tiff_dir: Path) -> list[str]:
    files = sorted(tiff_dir.glob("*.tif")) + sorted(tiff_dir.glob("*.tiff"))
    # Prefer one extension set; if both exist, take unique stems by sorted full names
    files = sorted({p.name: p for p in files}.values(), key=lambda p: p.name)
    if not files:
        raise FileNotFoundError(f"No TIFF files found in {tiff_dir}")
    return [p.name for p in files]


def apply_mask_to_signal(
    *,
    signal_zarr: Path,
    mask_zarr: Path,
    output_mask_zarr: Path | None,
    masked_signal_zarr: Path | None,
    masked_tiff_dir: Path | None,
    max_aspect_ratio: float = 3.0,
    max_voxels: int = 200,
    dataset_name: str = "0",
    export_prefix: str = "masked_",
    name_from_tiff_dir: Path | None = None,
) -> dict[str, str | int | float]:
    if masked_signal_zarr is None and masked_tiff_dir is None and output_mask_zarr is None:
        raise ValueError("Provide at least one of --output_mask_zarr / --masked_signal_zarr / --masked_tiff_dir")

    signal = open_zarr_dataset(signal_zarr, dataset_name=dataset_name)
    mask_in = open_zarr_dataset(mask_zarr, dataset_name=dataset_name)
    if signal.shape != mask_in.shape:
        raise ValueError(f"Shape mismatch: signal={signal.shape}, mask={mask_in.shape}")

    chunks = tuple(int(c) for c in mask_in.chunks)
    shape = tuple(int(s) for s in mask_in.shape)

    mask_out = None
    if output_mask_zarr is not None:
        _, mask_out = create_output_zarr(
            output_mask_zarr,
            shape=shape,
            chunks=chunks,
            dtype="uint8",
            dataset_name=dataset_name,
        )

    masked_out = None
    if masked_signal_zarr is not None or masked_tiff_dir is not None:
        if masked_signal_zarr is None:
            raise ValueError("--masked_tiff_dir requires --masked_signal_zarr")
        _, masked_out = create_output_zarr(
            masked_signal_zarr,
            shape=signal.shape,
            chunks=tuple(int(c) for c in signal.chunks),
            dtype=signal.dtype,
            dataset_name=dataset_name,
        )

    chunk_list = _chunk_slices(shape, chunks)
    total_components = 0
    removed_aspect = 0
    removed_volume = 0
    for zyx in tqdm(chunk_list, desc="Apply mask", unit="chunk"):
        mask_chunk = np.asarray(mask_in[zyx]) > 0
        filtered, n_labels, n_removed_aspect, n_removed_volume = filter_elongated_components_chunk(
            mask_chunk,
            max_aspect_ratio=max_aspect_ratio,
            max_voxels=max_voxels,
        )
        total_components += n_labels
        removed_aspect += n_removed_aspect
        removed_volume += n_removed_volume

        if mask_out is not None:
            mask_out[zyx] = filtered
        if masked_out is not None:
            signal_chunk = np.asarray(signal[zyx])
            masked_out[zyx] = np.where(filtered > 0, signal_chunk, 0)

    exports: dict[str, str | int | float] = {
        "max_aspect_ratio": float(max_aspect_ratio),
        "max_voxels": int(max_voxels),
        "chunk_shape": ",".join(str(c) for c in chunks),
        "chunks_total": int(len(chunk_list)),
        "components_total_3d_chunk": int(total_components),
        "components_removed_aspect_3d_chunk": int(removed_aspect),
        "components_removed_volume_3d_chunk": int(removed_volume),
        "components_kept_3d_chunk": int(total_components - removed_aspect - removed_volume),
    }
    if output_mask_zarr is not None:
        exports["output_mask_zarr"] = str(output_mask_zarr)
    if masked_signal_zarr is not None:
        exports["masked_signal_zarr"] = str(masked_signal_zarr)
    if masked_tiff_dir is not None:
        masked_tiff_dir.mkdir(parents=True, exist_ok=True)
        slice_names = None
        if name_from_tiff_dir is not None:
            slice_names = _list_tiff_names(name_from_tiff_dir)
        export_zarr_to_tiff(
            masked_signal_zarr,
            masked_tiff_dir,
            dataset_name=dataset_name,
            prefix=export_prefix,
            slice_names=slice_names,
        )
        exports["masked_tiff_dir"] = str(masked_tiff_dir)
        if name_from_tiff_dir is not None:
            exports["name_from_tiff_dir"] = str(name_from_tiff_dir)
    return exports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply mask to signal; within each Zarr chunk, drop 3D connected "
            "components whose bbox aspect ratio exceeds the threshold."
        ),
    )
    parser.add_argument("--signal_zarr", type=Path, required=True)
    parser.add_argument("--mask_zarr", type=Path, required=True)
    parser.add_argument("--output_mask_zarr", type=Path, default=None)
    parser.add_argument("--masked_signal_zarr", type=Path, default=None)
    parser.add_argument("--masked_tiff_dir", type=Path, default=None)
    parser.add_argument(
        "--name_from_tiff_dir",
        type=Path,
        default=None,
        help="If set, export TIFF names match this folder's sorted stack names.",
    )
    parser.add_argument(
        "--max_aspect_ratio",
        type=float,
        default=3.0,
        help="Drop chunk-local 3D components whose bbox max/min side exceeds this ratio.",
    )
    parser.add_argument(
        "--max_voxels",
        type=int,
        default=200,
        help="Drop chunk-local 3D components larger than this many voxels.",
    )
    parser.add_argument("--dataset_name", default="0")
    parser.add_argument("--export_prefix", default="masked_")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = apply_mask_to_signal(
        signal_zarr=args.signal_zarr,
        mask_zarr=args.mask_zarr,
        output_mask_zarr=args.output_mask_zarr,
        masked_signal_zarr=args.masked_signal_zarr,
        masked_tiff_dir=args.masked_tiff_dir,
        max_aspect_ratio=args.max_aspect_ratio,
        max_voxels=args.max_voxels,
        dataset_name=args.dataset_name,
        export_prefix=args.export_prefix,
        name_from_tiff_dir=args.name_from_tiff_dir,
    )
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
