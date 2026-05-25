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


def _load_tiff_stack(input_dir: Path) -> np.ndarray:
    tiff_files = sorted(input_dir.glob("*.tif*"))
    if not tiff_files:
        raise PipelineError(
            ErrorCode.INPUT_NOT_FOUND,
            "No TIFF files found for hemisphere conversion",
            {"input_dir": str(input_dir)},
        )
    slices = [tifffile.imread(str(path)) for path in tiff_files]
    return np.stack(slices, axis=0)


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

    if not input_path.exists() or not input_path.is_dir():
        raise PipelineError(
            ErrorCode.INPUT_NOT_FOUND,
            "Input atlas label directory not found",
            {"input_dir": str(input_path)},
        )

    label_stack = _load_tiff_stack(input_path)
    hemisphere_stack = np.zeros_like(label_stack, dtype=np.uint8)
    hemisphere_stack[label_stack > 0] = np.where(
        np.asarray(np.arange(label_stack.shape[2])[None, None, :]) < (label_stack.shape[2] / 2.0),
        LEFT_ID,
        RIGHT_ID,
    ).astype(np.uint8)

    try:
        import zarr
        from numcodecs import Blosc
    except ModuleNotFoundError as exc:
        raise PipelineError(
            ErrorCode.DEPENDENCY_MISSING,
            "zarr and numcodecs are required for hemisphere-label conversion",
            {"dependency": "zarr/numcodecs", "error": str(exc)},
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.DirectoryStore(str(output_path))
    root = zarr.group(store=store, overwrite=True)
    if compressor == "default":
        compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)

    dataset = root.create_dataset(
        dataset_name,
        shape=hemisphere_stack.shape,
        chunks=chunk_size,
        dtype=np.uint8,
        compressor=compressor,
    )
    dataset[:] = hemisphere_stack
    root.attrs["source"] = str(input_path)
    root.attrs["labels"] = {"0": "background", "1": "left", "2": "right"}

    result = {
        "success": True,
        "input_dir": str(input_path),
        "output_zarr": str(output_path),
        "dataset_name": dataset_name,
        "shape": list(hemisphere_stack.shape),
        "dtype": "uint8",
        "chunk_size": list(chunk_size),
    }
    manifest_path = write_run_manifest(
        output_path,
        module="registration",
        entrypoint="convert_atlas_label_to_hemisphere",
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


convert_atlas_label_to_hemisphere_zarr = convert_atlas_label_to_hemisphere


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert atlas label TIFF stack to hemisphere label Zarr")
    parser.add_argument("--input", required=True, help="Input atlas label TIFF folder")
    parser.add_argument("--output", required=True, help="Output hemisphere .zarr path")
    parser.add_argument("--chunk_size", default="256,256,256", help="Chunk size z,y,x")
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
            dataset_name=args.dataset_name,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except PipelineError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False), file=sys.stderr)
        return exc.exit_code
    except Exception as exc:  # pragma: no cover
        logger.exception("Unhandled hemisphere conversion error: %s", exc)
        wrapped = PipelineError(ErrorCode.INTERNAL_ERROR, "Unhandled hemisphere conversion error", {"error": str(exc)})
        print(json.dumps(wrapped.to_dict(), ensure_ascii=False), file=sys.stderr)
        return wrapped.exit_code


if __name__ == "__main__":
    sys.exit(main())
