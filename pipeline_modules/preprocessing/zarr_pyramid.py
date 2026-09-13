"""Add an OME-NGFF resolution pyramid to an existing 3D Zarr group.

Napari's builtin zarr reader always loads dataset ``0`` at full resolution, which
freezes 3D view on large volumes. A pyramid (datasets ``0``, ``1``, ``2``, ...)
plus ``napari-ome-zarr`` lets the viewer fetch a coarse level first.

CLI::

    python -m pipeline_modules.preprocessing.zarr_pyramid --input volume.zarr
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
from tqdm import tqdm

try:
    from pipeline_modules.preprocessing.tiff_to_zarr import _create_array, _open_output_group, resolve_compressor
    from pipeline_modules.utils.errors import ErrorCode, PipelineError
    from pipeline_modules.utils.run_manifest import write_run_manifest
except ImportError:  # pragma: no cover
    from .tiff_to_zarr import _create_array, _open_output_group, resolve_compressor
    from ..utils.errors import ErrorCode, PipelineError
    from ..utils.run_manifest import write_run_manifest

logger = logging.getLogger(__name__)


def _list_array_keys(group: Any) -> list[str]:
    try:
        from pipeline_modules.utils.zarr_io import list_array_keys
    except ImportError:  # pragma: no cover
        from ..utils.zarr_io import list_array_keys
    return list_array_keys(group)


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def ngff_multiscales(
    n_levels: int,
    *,
    base_scale_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> list[dict[str, Any]]:
    datasets = []
    for level in range(n_levels):
        factor = float(2**level)
        datasets.append(
            {
                "path": str(level),
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": [
                            base_scale_zyx[0] * factor,
                            base_scale_zyx[1] * factor,
                            base_scale_zyx[2] * factor,
                        ],
                    }
                ],
            }
        )
    return [
        {
            "version": "0.4",
            "name": "volume",
            "axes": [
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": datasets,
        }
    ]


def _even_crop(volume: np.ndarray) -> np.ndarray:
    z, y, x = volume.shape
    return volume[: z // 2 * 2, : y // 2 * 2, : x // 2 * 2]


def downsample_mean2(volume: np.ndarray) -> np.ndarray:
    """2x isotropic mean downsample of a 3D array."""
    work = _even_crop(np.asarray(volume))
    if min(work.shape) < 2:
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            "Volume is too small to downsample by 2",
            {"shape": list(work.shape)},
        )
    z, y, x = work.shape
    stacked = work.reshape(z // 2, 2, y // 2, 2, x // 2, 2)
    return stacked.mean(axis=(1, 3, 5), dtype=np.float32)


def downsample_nearest2(volume: np.ndarray) -> np.ndarray:
    """2x isotropic nearest downsample (keep even samples). Use for integer labels."""
    work = _even_crop(np.asarray(volume))
    if min(work.shape) < 2:
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            "Volume is too small to downsample by 2",
            {"shape": list(work.shape)},
        )
    return work[::2, ::2, ::2]


def _resolve_z_block(src: Any, requested: int, *, max_bytes: int = 3_500_000_000) -> int:
    """Keep each pyramid slab under ``max_bytes`` so native 56k-wide slices do not OOM."""
    ny, nx = int(src.shape[1]), int(src.shape[2])
    itemsize = int(np.dtype(src.dtype).itemsize)
    slice_bytes = max(1, ny * nx * itemsize)
    fit = max(2, (int(max_bytes) // slice_bytes) // 2 * 2)
    z_block = max(2, int(requested) // 2 * 2)
    return min(z_block, fit, int(src.shape[0]) // 2 * 2 or 2)


def _copy_compressor(src: Any) -> Any:
    try:
        from pipeline_modules.utils.zarr_io import array_compressor
    except ImportError:  # pragma: no cover
        from ..utils.zarr_io import array_compressor
    return array_compressor(src) or resolve_compressor("default")


def _fill_level(
    src: Any,
    dst: Any,
    *,
    z_block: int = 64,
    label: str = "",
    method: str = "mean",
) -> None:
    depth = int(src.shape[0])
    z_block = _resolve_z_block(src, z_block)
    logger.info("Pyramid %s using z_block=%d method=%s", label or "level", z_block, method)
    progress = tqdm(total=depth, desc=f"Pyramid {label or 'level'}", unit="slice", file=sys.stderr)
    down_fn = downsample_nearest2 if method == "nearest" else downsample_mean2
    try:
        for z0 in range(0, depth, z_block):
            z1 = min(z0 + z_block, depth)
            if (z1 - z0) < 2:
                break
            if (z1 - z0) % 2:
                z1 -= 1
            slab = np.asarray(src[z0:z1])
            down = down_fn(slab).astype(src.dtype, copy=False)
            dst[z0 // 2 : z0 // 2 + down.shape[0]] = down
            progress.update(z1 - z0)
    finally:
        progress.close()


def add_resolution_pyramid(
    zarr_path: str | Path,
    *,
    dataset_name: str = "0",
    max_levels: int = 5,
    min_size: int = 64,
    z_block: int = 64,
    overwrite_levels: bool = True,
    write_manifest: bool = True,
    method: str = "mean",
    scale_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, Any]:
    """Append datasets ``1..N`` under an existing Zarr group that already has ``0``.

    ``method='mean'`` for intensity; ``method='nearest'`` for integer labels.
    Writes OME-NGFF 0.4 ``multiscales`` so ``napari-ome-zarr`` can open the store.
    """
    started_at = time.time()
    method = str(method or "mean").strip().lower()
    if method not in ("mean", "nearest"):
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            "pyramid method must be mean or nearest",
            {"method": method},
        )
    path = Path(zarr_path)
    if not path.exists():
        raise PipelineError(ErrorCode.INPUT_NOT_FOUND, "Zarr path not found", {"zarr_path": str(path)})

    try:
        import zarr
    except ModuleNotFoundError as exc:
        raise PipelineError(
            ErrorCode.DEPENDENCY_MISSING,
            "zarr is required",
            {"dependency": "zarr", "error": str(exc)},
        ) from exc

    root = zarr.open_group(str(path), mode="r+")
    if dataset_name not in root:
        raise PipelineError(
            ErrorCode.INPUT_FORMAT_INVALID,
            "Full-resolution dataset not found in Zarr group",
            {"zarr_path": str(path), "dataset": dataset_name, "arrays": _list_array_keys(root)},
        )
    current = root[dataset_name]
    if current.ndim != 3:
        raise PipelineError(
            ErrorCode.INPUT_FORMAT_INVALID,
            "Pyramid builder currently supports 3D arrays only",
            {"shape": list(current.shape)},
        )

    shapes = [list(current.shape)]
    level = 0
    while level + 1 < int(max_levels) and min(int(s) for s in current.shape) >= int(min_size) * 2:
        level += 1
        name = str(level)
        out_shape = tuple(int(s) // 2 for s in current.shape)
        chunks = tuple(min(int(c), s) for c, s in zip((32, 256, 256), out_shape))
        if name in root and overwrite_levels:
            del root[name]
        if name not in root:
            dst = _create_array(
                root,
                name,
                shape=out_shape,
                chunks=chunks,
                dtype=current.dtype,
                compressor=_copy_compressor(current),
            )
        else:
            dst = root[name]
        logger.info("Writing pyramid level %s shape=%s from %s", name, list(out_shape), getattr(current, "name", "?"))
        _fill_level(current, dst, z_block=z_block, label=name, method=method)
        current = dst
        shapes.append(list(out_shape))

    n_levels = level + 1
    root.attrs["multiscales"] = ngff_multiscales(n_levels, base_scale_zyx=scale_zyx)
    extra = {
        "success": True,
        "zarr_path": str(path),
        "levels": n_levels,
        "shapes": shapes,
        "method": method,
        "scale_zyx": list(scale_zyx),
    }
    if write_manifest:
        manifest_path = write_run_manifest(
            path,
            module="preprocessing",
            entrypoint="add_resolution_pyramid",
            inputs={
                "zarr_path": str(path),
                "dataset_name": dataset_name,
                "max_levels": max_levels,
                "min_size": min_size,
                "method": method,
                "scale_zyx": list(scale_zyx),
            },
            outputs=[path],
            started_at=started_at,
            extra=extra,
        )
        extra["manifest_path"] = str(manifest_path)
    extra["duration_seconds"] = time.time() - started_at
    return extra


def _preview_zarr_path(src: Path) -> Path:
    return src.with_name(f"{src.stem}_preview.zarr")


def write_preview_zarr(
    zarr_path: str | Path,
    output_zarr: str | Path | None = None,
    *,
    dataset_name: str = "0",
    factor: int = 4,
    z_block: int = 64,
) -> dict[str, Any]:
    """Write a single-array preview store so napari 3D can open without a pyramid plugin."""
    started_at = time.time()
    src_path = Path(zarr_path)
    if factor < 2:
        raise PipelineError(ErrorCode.ARGUMENT_INVALID, "preview factor must be >= 2", {"factor": factor})
    try:
        import zarr
    except ModuleNotFoundError as exc:
        raise PipelineError(
            ErrorCode.DEPENDENCY_MISSING,
            "zarr is required",
            {"dependency": "zarr", "error": str(exc)},
        ) from exc

    src_root = zarr.open_group(str(src_path), mode="r")
    if dataset_name not in src_root:
        raise PipelineError(
            ErrorCode.INPUT_FORMAT_INVALID,
            "Full-resolution dataset not found",
            {"zarr_path": str(src_path), "dataset": dataset_name},
        )
    src = src_root[dataset_name]
    if src.ndim != 3:
        raise PipelineError(ErrorCode.INPUT_FORMAT_INVALID, "Preview currently supports 3D arrays only", {"shape": list(src.shape)})

    out_path = Path(output_zarr) if output_zarr is not None else _preview_zarr_path(src_path)
    depth, height, width = (int(s) for s in src.shape)
    out_shape = (depth // factor, height // factor, width // factor)
    if min(out_shape) < 1:
        raise PipelineError(ErrorCode.ARGUMENT_INVALID, "Preview factor is too large for this volume", {"shape": list(src.shape), "factor": factor})

    root = _open_output_group(zarr, out_path)
    root.attrs["multiscales"] = [{"version": "0.4", "datasets": [{"path": "0"}]}]
    chunks = tuple(min(c, s) for c, s in zip((32, 256, 256), out_shape))
    dst = _create_array(
        root,
        "0",
        shape=out_shape,
        chunks=chunks,
        dtype=src.dtype,
        compressor=_copy_compressor(src),
    )
    z_block = max(int(factor), int(z_block) // int(factor) * int(factor))
    progress = tqdm(total=out_shape[0] * factor, desc=f"Preview {factor}x", unit="slice", file=sys.stderr)
    try:
        out_z = 0
        for z0 in range(0, out_shape[0] * factor, z_block):
            z1 = min(z0 + z_block, out_shape[0] * factor)
            slab = np.asarray(src[z0:z1])
            z_use = (slab.shape[0] // factor) * factor
            y_use = (slab.shape[1] // factor) * factor
            x_use = (slab.shape[2] // factor) * factor
            work = slab[:z_use, :y_use, :x_use]
            down = (
                work.reshape(z_use // factor, factor, y_use // factor, factor, x_use // factor, factor)
                .mean(axis=(1, 3, 5), dtype=np.float32)
                .astype(src.dtype, copy=False)
            )
            dst[out_z : out_z + down.shape[0]] = down
            out_z += down.shape[0]
            progress.update(z_use)
    finally:
        progress.close()

    extra = {
        "success": True,
        "input_zarr": str(src_path),
        "output_zarr": str(out_path),
        "factor": int(factor),
        "shape": list(out_shape),
    }
    extra["duration_seconds"] = time.time() - started_at
    logger.info("Wrote 3D preview %s shape=%s", out_path, list(out_shape))
    return extra


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add a 2x OME-NGFF resolution pyramid to a 3D Zarr group")
    parser.add_argument("--input", required=True, help="Existing Zarr group that already contains dataset 0")
    parser.add_argument("--max_levels", type=int, default=5, help="Maximum number of levels including full-res 0")
    parser.add_argument("--min_size", type=int, default=64, help="Stop when the shortest axis would drop below this")
    parser.add_argument(
        "--method",
        choices=("mean", "nearest"),
        default="mean",
        help="mean for intensity images; nearest for integer label zarrs",
    )
    parser.add_argument(
        "--scale_zyx",
        default="1,1,1",
        help="Full-res voxel size Z,Y,X (written into NGFF scale; level n is 2^n times this)",
    )
    parser.add_argument("--preview_factor", type=int, default=0, help="If >1, also write a sibling *_preview.zarr downsampled by this factor for napari 3D")
    parser.add_argument("--z_block", type=int, default=64, help="Z slices read per downsample pass (capped by slice RAM)")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    _configure_logging()
    try:
        results: dict[str, Any] = {}
        if args.preview_factor > 1:
            results["preview"] = write_preview_zarr(args.input, factor=args.preview_factor, z_block=args.z_block)
        else:
            scale_parts = [float(x.strip()) for x in str(args.scale_zyx).split(",") if x.strip()]
            if len(scale_parts) != 3:
                raise PipelineError(ErrorCode.ARGUMENT_INVALID, "--scale_zyx must be Z,Y,X", {"scale_zyx": args.scale_zyx})
            results = add_resolution_pyramid(
                args.input,
                max_levels=args.max_levels,
                min_size=args.min_size,
                z_block=args.z_block,
                method=str(args.method),
                scale_zyx=(scale_parts[0], scale_parts[1], scale_parts[2]),
            )
    except PipelineError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False), file=sys.stderr)
        return exc.exit_code
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
