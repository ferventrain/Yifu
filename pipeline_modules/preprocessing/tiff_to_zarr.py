from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm
from typing import Any

import numpy as np
import tifffile

try:
    from pipeline_modules.utils.errors import ErrorCode, PipelineError
    from pipeline_modules.utils.run_manifest import write_run_manifest
except ImportError:  # pragma: no cover - fallback for package-relative execution
    from ..utils.errors import ErrorCode, PipelineError
    from ..utils.run_manifest import write_run_manifest

logger = logging.getLogger(__name__)


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


def convert_tiff_to_zarr(
    input_dir: str | Path,
    output_zarr: str | Path,
    chunk_size: tuple[int, int, int] = (128, 256, 256),
    compressor: Any = "default",
    *,
    dataset_name: str = "0",
) -> dict[str, Any]:
    """Convert a folder of TIFF files into a directory-store Zarr volume."""
    started_at = time.time()
    input_path = Path(input_dir)
    output_path = Path(output_zarr)

    try:
        import zarr
        from numcodecs import Blosc
    except ModuleNotFoundError as exc:
        raise PipelineError(
            ErrorCode.DEPENDENCY_MISSING,
            "zarr and numcodecs are required for TIFF-to-Zarr conversion",
            {"dependency": "zarr/numcodecs", "error": str(exc)},
        ) from exc

    if not input_path.exists():
        raise PipelineError(
            ErrorCode.INPUT_NOT_FOUND,
            "Input TIFF directory not found",
            {"input_dir": str(input_path)},
        )
    if not input_path.is_dir():
        raise PipelineError(
            ErrorCode.INPUT_FORMAT_INVALID,
            "Input to TIFF-to-Zarr must be a directory",
            {"input_dir": str(input_path)},
        )

    tiff_files = sorted(input_path.glob("*.tif*"))
    if not tiff_files:
        raise PipelineError(
            ErrorCode.INPUT_NOT_FOUND,
            "No TIFF files found for TIFF-to-Zarr conversion",
            {"input_dir": str(input_path)},
        )

    sample = tifffile.imread(tiff_files[0])
    dtype = sample.dtype
    shape = (len(tiff_files),) + sample.shape
    logger.info("Converting %d TIFF slices from %s into %s", len(tiff_files), input_path, output_path)

    store = zarr.DirectoryStore(str(output_path))
    root = zarr.group(store=store, overwrite=True)
    if compressor == "default":
        compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)

    dataset = root.create_dataset(
        dataset_name,
        shape=shape,
        chunks=chunk_size,
        dtype=dtype,
        compressor=compressor,
    )
    root.attrs["multiscales"] = [{
        "version": "0.4",
        "datasets": [{"path": dataset_name}],
    }]

    z_chunk = chunk_size[0]
    n_chunks = math.ceil(shape[0] / z_chunk)
    n_workers = min(4, max(1, n_chunks))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for start in tqdm(range(0, shape[0], z_chunk), total=n_chunks, desc="Converting to Zarr", unit="chunk"):
            end = min(start + z_chunk, shape[0])
            slices_files = tiff_files[start:end]
            images = list(pool.map(tifffile.imread, [str(f) for f in slices_files]))
            dataset[start:end] = np.stack(images)

    result = {
        "success": True,
        "input_dir": str(input_path),
        "output_zarr": str(output_path),
        "dataset_name": dataset_name,
        "shape": list(shape),
        "dtype": str(dtype),
        "chunk_size": list(chunk_size),
    }
    manifest_path = write_run_manifest(
        output_path,
        module="preprocessing",
        entrypoint="convert_tiff_to_zarr",
        inputs={
            "input_dir": str(input_path),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a TIFF folder to a directory-store Zarr volume")
    parser.add_argument("--input", required=True, help="Input TIFF folder")
    parser.add_argument("--output", required=True, help="Output .zarr path")
    parser.add_argument("--chunk_size", default="256,256,256", help="Chunk size z,y,x")
    parser.add_argument("--dataset_name", default="0", help="Dataset name inside the Zarr group")
    parser.add_argument("--json_logs", action="store_true", help="Emit NDJSON log records to stderr")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _configure_logging(args.json_logs)

    try:
        result = convert_tiff_to_zarr(
            args.input,
            args.output,
            _coerce_chunk_size(args.chunk_size),
            dataset_name=args.dataset_name,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except PipelineError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False), file=sys.stderr)
        return exc.exit_code
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        logger.exception("Unhandled TIFF-to-Zarr error: %s", exc)
        wrapped = PipelineError(ErrorCode.INTERNAL_ERROR, "Unhandled TIFF-to-Zarr error", {"error": str(exc)})
        print(json.dumps(wrapped.to_dict(), ensure_ascii=False), file=sys.stderr)
        return wrapped.exit_code


if __name__ == "__main__":
    sys.exit(main())
