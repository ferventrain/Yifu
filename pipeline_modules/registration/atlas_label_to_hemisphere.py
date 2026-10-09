from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from pipeline_modules.preprocessing.tiff_to_zarr import convert_tiff_to_zarr
    from pipeline_modules.utils.errors import ErrorCode, PipelineError
    from pipeline_modules.utils.run_manifest import write_run_manifest
except ImportError:  # pragma: no cover
    convert_tiff_to_zarr = None  # type: ignore[assignment]
    PipelineError = None  # type: ignore[assignment,misc]
    ErrorCode = None  # type: ignore[assignment]
    write_run_manifest = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

LEFT_ID = np.uint8(1)
RIGHT_ID = np.uint8(2)


def _configure_logging(json_logs: bool) -> None:
    if json_logs:
        class _JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                return json.dumps(
                    {
                        "level": record.levelname,
                        "logger": record.name,
                        "message": record.getMessage(),
                    },
                    ensure_ascii=False,
                )

        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonFormatter())
        logging.root.handlers.clear()
        logging.root.addHandler(handler)
        logging.root.setLevel(logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _coerce_chunk_size(value: str | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, tuple):
        return value
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 3:
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            "chunk_size must be three comma-separated integers",
            {"chunk_size": value},
        )
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def _list_tiff_stack(input_dir: Path) -> list[Path]:
    tiff_files = sorted(input_dir.glob("*.tif*"))
    if not tiff_files:
        raise PipelineError(
            ErrorCode.INPUT_NOT_FOUND,
            "No TIFF files found for hemisphere conversion",
            {"input_dir": str(input_dir)},
        )
    return tiff_files


def _open_zarr_dataset(path_like: Path, dataset_name: str):
    try:
        from pipeline_modules.utils.zarr_io import open_zarr_array
    except ImportError:  # pragma: no cover
        from ..utils.zarr_io import open_zarr_array
    try:
        return open_zarr_array(path_like, dataset_name=dataset_name)
    except ModuleNotFoundError as exc:
        raise PipelineError(
            ErrorCode.DEPENDENCY_MISSING,
            "zarr is required for hemisphere-label conversion",
            {"dependency": "zarr", "error": str(exc)},
        ) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            "Could not resolve a Zarr array from input",
            {"input": str(path_like), "dataset_name": dataset_name, "error": str(exc)},
        ) from exc


def _create_output_dataset(output_path: Path, dataset_name: str, shape, chunk_size, compressor):
    try:
        from pipeline_modules.utils.zarr_io import create_output_zarr
    except ImportError:  # pragma: no cover
        from ..utils.zarr_io import create_output_zarr
    try:
        root, dataset = create_output_zarr(
            output_path,
            shape,
            chunk_size,
            np.uint8,
            dataset_name=dataset_name,
            compressor=compressor,
        )
    except ModuleNotFoundError as exc:
        raise PipelineError(
            ErrorCode.DEPENDENCY_MISSING,
            "zarr and numcodecs are required for hemisphere-label conversion",
            {"dependency": "zarr/numcodecs", "error": str(exc)},
        ) from exc
    root.attrs["labels"] = {"0": "background", "1": "left", "2": "right"}
    return root, dataset


def _make_hemisphere_block_for_x_range(label_block: np.ndarray, x0: int, x1: int, split_x: int) -> np.ndarray:
    hemisphere_block = np.zeros(label_block.shape, dtype=np.uint8)
    positive_mask = label_block > 0
    if not np.any(positive_mask):
        return hemisphere_block

    if x1 <= split_x:
        hemisphere_block[positive_mask] = LEFT_ID
    elif x0 >= split_x:
        hemisphere_block[positive_mask] = RIGHT_ID
    else:
        local_split = int(split_x - x0)
        left_mask = label_block[..., :local_split] > 0
        right_mask = label_block[..., local_split:] > 0
        if np.any(left_mask):
            hemisphere_block[..., :local_split][left_mask] = LEFT_ID
        if np.any(right_mask):
            hemisphere_block[..., local_split:][right_mask] = RIGHT_ID
    return hemisphere_block


def _write_hemisphere_slice(dataset, z_index: int, label_slice: np.ndarray, split_x: int) -> None:
    if not np.any(label_slice > 0):
        return
    hemisphere_slice = np.zeros(label_slice.shape, dtype=np.uint8)
    left_mask = label_slice[:, :split_x] > 0
    right_mask = label_slice[:, split_x:] > 0
    if np.any(left_mask):
        hemisphere_slice[:, :split_x][left_mask] = LEFT_ID
    if np.any(right_mask):
        hemisphere_slice[:, split_x:][right_mask] = RIGHT_ID
    dataset[z_index, :, :] = hemisphere_slice


def _iter_3d_blocks(shape: tuple[int, int, int], block_shape: tuple[int, int, int]):
    for z0 in range(0, shape[0], block_shape[0]):
        z1 = min(z0 + block_shape[0], shape[0])
        for y0 in range(0, shape[1], block_shape[1]):
            y1 = min(y0 + block_shape[1], shape[1])
            for x0 in range(0, shape[2], block_shape[2]):
                x1 = min(x0 + block_shape[2], shape[2])
                yield z0, z1, y0, y1, x0, x1


def _label_x_extent_zarr(label_zarr, shape, read_block_shape) -> tuple[int, int] | None:
    """Stream the label volume and return the (min, max) x of nonzero voxels.

    The midline must bisect the registered brain, not the sample volume: the
    brain sits wherever mounting put it, so a volume-midplane split assigns
    near-100%% of voxels to one hemisphere whenever the brain is off-center.
    """
    xmin, xmax = None, None
    for z0, z1, y0, y1, x0, x1 in _iter_3d_blocks(shape, read_block_shape):
        block = np.asarray(label_zarr[z0:z1, y0:y1, x0:x1])
        if not np.any(block > 0):
            continue
        x_present = np.any(block > 0, axis=(0, 1))
        present = np.nonzero(x_present)[0]
        block_xmin, block_xmax = x0 + int(present[0]), x0 + int(present[-1])
        xmin = block_xmin if xmin is None else min(xmin, block_xmin)
        xmax = block_xmax if xmax is None else max(xmax, block_xmax)
    if xmin is None:
        return None
    return xmin, xmax


def _label_x_extent_tiff(tiff_files) -> tuple[int, int] | None:
    xmin, xmax = None, None
    for tiff_path in tiff_files:
        label_slice = tifffile.imread(str(tiff_path))
        if not np.any(label_slice > 0):
            continue
        x_present = np.any(label_slice > 0, axis=0)
        present = np.nonzero(x_present)[0]
        slice_xmin, slice_xmax = int(present[0]), int(present[-1])
        xmin = slice_xmin if xmin is None else min(xmin, slice_xmin)
        xmax = slice_xmax if xmax is None else max(xmax, slice_xmax)
    if xmin is None:
        return None
    return xmin, xmax


def convert_atlas_label_to_hemisphere(
    input_dir: str | Path,
    output_zarr: str | Path,
    chunk_size: tuple[int, int, int] = (128, 256, 256),
    compressor: Any = "default",
    *,
    dataset_name: str = "0",
) -> dict[str, Any]:
    started_at = time.time()
    input_path = Path(input_dir)
    output_path = Path(output_zarr)

    if not input_path.exists():
        raise PipelineError(
            ErrorCode.INPUT_NOT_FOUND,
            "Input atlas label path not found",
            {"input_dir": str(input_path)},
        )

    input_kind = "zarr" if input_path.suffix.lower() == ".zarr" else "tiff_dir"
    if input_kind == "zarr":
        label_zarr = _open_zarr_dataset(input_path, dataset_name)
        if len(label_zarr.shape) != 3:
            raise PipelineError(
                ErrorCode.ARGUMENT_INVALID,
                "Hemisphere conversion expects a 3D label Zarr",
                {"shape": list(label_zarr.shape)},
            )
        shape = tuple(int(value) for value in label_zarr.shape)
        input_chunks = getattr(label_zarr, "chunks", None)
        read_block_shape = (
            tuple(int(value) for value in input_chunks[:3])
            if input_chunks is not None
            else tuple(int(value) for value in chunk_size)
        )
        extent = _label_x_extent_zarr(label_zarr, shape, read_block_shape)
        if extent is None:
            split_x = int(np.ceil(shape[2] / 2.0))
        else:
            split_x = extent[0] + (extent[1] - extent[0] + 1) // 2
        logger.info(
            "Hemisphere midline: label x extent=%s, split_x=%d (volume midplane=%d)",
            extent,
            split_x,
            int(np.ceil(shape[2] / 2.0)),
        )
        root, dataset = _create_output_dataset(output_path, dataset_name, shape, chunk_size, compressor)

        block_specs = list(_iter_3d_blocks(shape, read_block_shape))
        for z0, z1, y0, y1, x0, x1 in tqdm(block_specs, desc="Hemisphere Zarr blocks", unit="block"):
            label_block = np.asarray(label_zarr[z0:z1, y0:y1, x0:x1])
            if not np.any(label_block > 0):
                continue
            dataset[z0:z1, y0:y1, x0:x1] = _make_hemisphere_block_for_x_range(
                label_block,
                x0=x0,
                x1=x1,
                split_x=split_x,
            )
    else:
        if not input_path.is_dir():
            raise PipelineError(
                ErrorCode.INPUT_NOT_FOUND,
                "Input atlas label directory not found",
                {"input_dir": str(input_path)},
            )
        tiff_files = _list_tiff_stack(input_path)
        first_slice = tifffile.imread(str(tiff_files[0]))
        if first_slice.ndim != 2:
            raise PipelineError(
                ErrorCode.ARGUMENT_INVALID,
                "Hemisphere conversion expects a 2D TIFF stack",
                {"first_slice_shape": list(first_slice.shape)},
            )
        shape = (len(tiff_files), int(first_slice.shape[0]), int(first_slice.shape[1]))
        extent = _label_x_extent_tiff(tiff_files)
        if extent is None:
            split_x = int(np.ceil(shape[2] / 2.0))
        else:
            split_x = extent[0] + (extent[1] - extent[0] + 1) // 2
        logger.info(
            "Hemisphere midline: label x extent=%s, split_x=%d (volume midplane=%d)",
            extent,
            split_x,
            int(np.ceil(shape[2] / 2.0)),
        )
        root, dataset = _create_output_dataset(output_path, dataset_name, shape, chunk_size, compressor)

        for z_index, tiff_path in tqdm(
            enumerate(tiff_files),
            total=len(tiff_files),
            desc="Hemisphere TIFF slices",
            unit="slice",
        ):
            label_slice = tifffile.imread(str(tiff_path))
            if label_slice.shape != first_slice.shape:
                raise PipelineError(
                    ErrorCode.ARGUMENT_INVALID,
                    "TIFF stack contains inconsistent slice shapes",
                    {
                        "expected_shape": list(first_slice.shape),
                        "actual_shape": list(label_slice.shape),
                        "path": str(tiff_path),
                    },
                )
            _write_hemisphere_slice(dataset, z_index, label_slice, split_x)

    root.attrs["source"] = str(input_path)
    root.attrs["input_kind"] = input_kind
    root.attrs["split_x"] = int(split_x)
    root.attrs["label_x_extent"] = list(extent) if extent is not None else None

    result = {
        "success": True,
        "input": str(input_path),
        "input_kind": input_kind,
        "output_zarr": str(output_path),
        "dataset_name": dataset_name,
        "shape": list(shape),
        "dtype": "uint8",
        "chunk_size": list(chunk_size),
        "label_x_extent": list(extent) if extent is not None else None,
        "split_x": int(split_x),
    }
    if write_run_manifest is not None:
        manifest_path = write_run_manifest(
            output_path,
            module="registration",
            entrypoint="convert_atlas_label_to_hemisphere",
            inputs={
                "input": str(input_path),
                "input_kind": input_kind,
                "output_zarr": str(output_path),
                "dataset_name": dataset_name,
                "chunk_size": chunk_size,
            },
            outputs=[output_path],
            started_at=started_at,
            extra=result,
        )
        result["manifest_path"] = str(manifest_path)
    return result


convert_atlas_label_to_hemisphere_zarr = convert_atlas_label_to_hemisphere


def _resolve_default_atlas_label() -> Path:
    import os

    explicit = os.environ.get("YIFU_ATLAS_LABEL")
    if explicit:
        return Path(explicit)
    data_dir = os.environ.get("YIFU_DATA_DIR")
    if data_dir:
        candidate = Path(data_dir) / "reference" / "atlas_label.tiff"
        if candidate.exists():
            return candidate
    return Path(r"H:\Yifu_data\reference\atlas_label.tiff")


def _standard_space_hemi(label_arr_xyz: np.ndarray) -> tuple[np.ndarray, float]:
    """Paint left/right on a standard-space annotation in ANTs order (X,Y,Z)
    around the annotation's x centroid. The reference annotation merges
    left/right structure pairs (every label straddles the midline), so the
    global x centroid IS the midline."""
    mask = label_arr_xyz > 0
    xs = np.nonzero(mask.any(axis=(1, 2)))[0]
    if xs.size == 0:
        raise ValueError("atlas label volume is empty")
    weights = mask.sum(axis=(1, 2)).astype(np.float64)
    mid = float((xs * weights[xs]).sum() / weights[xs].sum())
    hemi = np.zeros(label_arr_xyz.shape, dtype=np.uint8)
    left_plane = np.arange(label_arr_xyz.shape[0])[:, None, None] < mid
    hemi[mask & left_plane] = LEFT_ID
    hemi[mask & ~left_plane] = RIGHT_ID
    return hemi, mid


def _upsample_hemi_and_write(warped_zyx: np.ndarray, target_shape, output_zarr: Path,
                             chunk=(256, 256, 256)) -> np.ndarray:
    """Nearest-neighbour upsample to the sample grid (z: round-linspace index,
    xy: cv2 nearest) and write in z-batches so each zarr chunk compresses once.
    Returns the per-z left/right boundary curve."""
    import cv2

    from pipeline_modules.utils.zarr_io import create_output_zarr

    z_indices = np.round(np.linspace(0, warped_zyx.shape[0] - 1, target_shape[0])).astype(int)
    root, dataset = create_output_zarr(output_zarr, list(target_shape), chunk, np.dtype("uint8"))
    boundary = np.full(target_shape[0], np.nan)
    target_xy = (int(target_shape[2]), int(target_shape[1]))
    batch_z = int(chunk[0])
    for z0 in range(0, target_shape[0], batch_z):
        z1 = min(z0 + batch_z, target_shape[0])
        batch = np.empty((z1 - z0, target_shape[1], target_shape[2]), dtype=np.uint8)
        for i, sz in enumerate(z_indices[z0:z1]):
            plane = warped_zyx[sz]
            if plane.shape[1] != target_shape[1] or plane.shape[2] != target_shape[2]:
                plane = cv2.resize(plane, target_xy, interpolation=cv2.INTER_NEAREST)
            batch[i] = plane
            cols_l = np.nonzero((plane == int(LEFT_ID)).any(axis=0))[0]
            cols_r = np.nonzero((plane == int(RIGHT_ID)).any(axis=0))[0]
            if cols_l.size and cols_r.size:
                boundary[z0 + i] = (cols_l.max() + 1 + cols_r.min()) / 2.0
        dataset[z0:z1] = batch
    good = ~np.isnan(boundary)
    if good.any():
        idx = np.arange(boundary.shape[0])
        boundary = np.interp(idx, idx[good], boundary[good])
    root.attrs["midline_method"] = "warped standard-space midplane (nearest-neighbour)"
    root.attrs["split_x"] = int(np.ceil(np.nanmedian(boundary))) if good.any() else 0
    root.attrs["split_x_per_z"] = [round(float(v), 2) for v in boundary]
    root.attrs["source"] = "hemisphere_from_transform"
    return boundary


def convert_hemisphere_from_transform(
    sample_dir: str | Path,
    output_zarr: str | Path | None = None,
    atlas_label_tiff: str | Path | None = None,
    *,
    transformlist: list | None = None,
    fixed_image=None,
    atlas_image=None,
    target_shape=None,
) -> dict[str, Any]:
    """Preferred hemisphere method: paint the STANDARD-SPACE midline plane on
    the annotation and warp it with the exact transform that produced
    upsampled_atlas_label. The split surface is the registered midline itself
    (bends and yaws with the sample, poles included).

    Post-run CLI usage reads everything from sample_dir (transforms/ +
    ch0_downsample/volume.nii.gz). Inside the registration flow the in-memory
    objects can be passed instead: transformlist=reg_result['fwdtransforms'],
    fixed_image=register_image, atlas_image=atlas_image,
    target_shape=original_shape.
    """
    import ants

    from pipeline_modules.registration.label_codec import load_label_array_preserving_ids

    started_at = time.time()
    sample_dir = Path(sample_dir)
    output_zarr = Path(output_zarr) if output_zarr else sample_dir / "atlas_label_hemisphere.zarr"
    atlas_label_tiff = Path(atlas_label_tiff) if atlas_label_tiff else _resolve_default_atlas_label()
    atlas_tiff = atlas_label_tiff.with_name("atlas.tiff")
    if not atlas_label_tiff.exists():
        raise FileNotFoundError(str(atlas_label_tiff))

    if transformlist is None:
        transforms_dir = sample_dir / "transforms"
        transformlist = [
            str(p)
            for p in sorted(
                (q for q in transforms_dir.iterdir() if q.name.startswith("fwd_")),
                key=lambda q: int(q.name.split("_")[1]),
            )
        ]
        if not transformlist:
            raise FileNotFoundError(f"no fwd_* transforms under {transforms_dir}")
    if fixed_image is None:
        from pipeline_modules.registration.ANTs_registration import _ants_image_read

        reference_nii = sample_dir / "ch0_downsample" / "volume.nii.gz"
        if not reference_nii.exists():
            raise FileNotFoundError(str(reference_nii))
        fixed_image = _ants_image_read(reference_nii)
        fixed_image.set_direction(np.eye(3))  # same direction forcing as registration
    if target_shape is None:
        label_meta = sample_dir / "upsampled_atlas_label.zarr" / "0" / ".zarray"
        if not label_meta.exists():
            raise FileNotFoundError(str(label_meta))
        target_shape = tuple(json.loads(label_meta.read_text(encoding="utf-8"))["shape"])

    label_arr_xyz = load_label_array_preserving_ids(atlas_label_tiff)  # (X, Y, Z)
    hemi_xyz, mid = _standard_space_hemi(label_arr_xyz)
    if atlas_image is None:
        if not atlas_tiff.exists():
            raise FileNotFoundError(str(atlas_tiff))
        atlas_image = ants.image_read(str(atlas_tiff))
    hemi_img = ants.from_numpy(
        hemi_xyz,
        spacing=atlas_image.spacing,
        origin=atlas_image.origin,
        direction=atlas_image.direction,
    )
    logger.info("standard-space midline x=%.1f; warping with fwd transforms", mid)
    warped = ants.apply_transforms(
        fixed=fixed_image,
        moving=hemi_img,
        transformlist=[str(t) for t in transformlist],
        interpolator="nearestNeighbor",
    )
    warped_zyx = np.transpose(np.asarray(warped.numpy(), dtype=np.uint8), (2, 1, 0))  # -> z,y,x
    if not np.any(warped_zyx > 0):
        raise RuntimeError("warped hemisphere volume is empty - transform/grid mismatch")

    boundary = _upsample_hemi_and_write(warped_zyx, target_shape, output_zarr)
    logger.info(
        "hemisphere zarr written: %s (boundary median %d, curve %.0f..%.0f, %.0fs)",
        output_zarr,
        int(np.nanmedian(boundary)),
        np.nanmin(boundary),
        np.nanmax(boundary),
        time.time() - started_at,
    )
    return {
        "success": True,
        "output_zarr": str(output_zarr),
        "target_shape": list(target_shape),
        "standard_midline_x": round(mid, 2),
        "split_x": int(np.ceil(np.nanmedian(boundary))),
        "elapsed_s": round(time.time() - started_at, 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert atlas label to hemisphere label Zarr")
    parser.add_argument("--input", default="", help="Label method: warped atlas label .zarr or TIFF folder (single global split)")
    parser.add_argument("--sample_dir", default="", help="Transform method (preferred): sample dir with transforms/ + ch0_downsample/volume.nii.gz")
    parser.add_argument("--output", required=True, help="Output hemisphere .zarr path")
    parser.add_argument("--atlas_label", default="", help="Standard-space annotation tiff (transform method)")
    parser.add_argument("--chunk_size", default="256,256,256", help="Chunk size z,y,x")
    parser.add_argument(
        "--compressor",
        choices=("default", "none"),
        default="default",
        help="Output compression. Use none for faster writing and reading at the cost of larger files.",
    )
    parser.add_argument("--dataset_name", default="0", help="Dataset name inside the Zarr group")
    parser.add_argument("--json_logs", action="store_true", help="Emit NDJSON log records to stderr")
    args = parser.parse_args()
    if bool(args.input) == bool(args.sample_dir):
        parser.error("provide exactly one of --input (label method) or --sample_dir (transform method)")
    return args


def main() -> int:
    args = parse_args()
    _configure_logging(args.json_logs)
    try:
        if args.sample_dir:
            result = convert_hemisphere_from_transform(
                args.sample_dir,
                args.output,
                args.atlas_label or None,
            )
        else:
            result = convert_atlas_label_to_hemisphere(
                args.input,
                args.output,
                _coerce_chunk_size(args.chunk_size),
                compressor=args.compressor,
                dataset_name=args.dataset_name,
            )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:  # pragma: no cover
        if PipelineError is not None and isinstance(exc, PipelineError):
            print(json.dumps(exc.to_dict(), ensure_ascii=False), file=sys.stderr)
            return exc.exit_code
        logger.exception("Unhandled hemisphere conversion error: %s", exc)
        if PipelineError is not None and ErrorCode is not None:
            wrapped = PipelineError(ErrorCode.INTERNAL_ERROR, "Unhandled hemisphere conversion error", {"error": str(exc)})
            print(json.dumps(wrapped.to_dict(), ensure_ascii=False), file=sys.stderr)
            return wrapped.exit_code
        print(
            json.dumps(
                {
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "Unhandled hemisphere conversion error",
                        "context": {"error": str(exc)},
                    }
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 5


if __name__ == "__main__":
    sys.exit(main())
