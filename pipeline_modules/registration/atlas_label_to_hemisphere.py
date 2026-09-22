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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert atlas label Zarr or TIFF stack to hemisphere label Zarr")
    parser.add_argument("--input", required=True, help="Input atlas label .zarr or TIFF folder")
    parser.add_argument("--output", required=True, help="Output hemisphere .zarr path")
    parser.add_argument("--chunk_size", default="256,256,256", help="Chunk size z,y,x")
    parser.add_argument(
        "--compressor",
        choices=("default", "none"),
        default="default",
        help="Output compression. Use none for faster writing and reading at the cost of larger files.",
    )
    parser.add_argument("--dataset_name", default="0", help="Dataset name inside the Zarr group")
    parser.add_argument("--json_logs", action="store_true", help="Emit NDJSON log records to stderr")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _configure_logging(args.json_logs)
    try:
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
