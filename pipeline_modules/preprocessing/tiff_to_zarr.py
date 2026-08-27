from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
from typing import Any, Iterable, Iterator, Literal

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


def resolve_tiff_workers(
    workers: int,
    *,
    read_z_chunk: int | None = None,
    zarr_z_chunk: int | None = None,
) -> int:
    if workers < 0:
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            "workers must be >= 0",
            {"workers": workers},
        )
    if workers == 0:
        # Network-backed TIFF stacks are usually I/O bound; cap auto workers to avoid
        # dozens of threads fighting over the same disk (Windows max is 61).
        worker_count = max(1, min(8, (os.cpu_count() or 4) // 4))
    else:
        worker_count = int(workers)
    if (
        read_z_chunk is not None
        and zarr_z_chunk is not None
        and read_z_chunk >= zarr_z_chunk
        and workers == 0
    ):
        # Full z-slab reads are memory-heavy (often tens of GB per batch).
        worker_count = min(worker_count, 2)
    if os.name == "nt" and worker_count > 61:
        worker_count = 61
    return worker_count


def resolve_read_z_chunk(zarr_z_chunk: int, read_z_chunk: int | None) -> int:
    if read_z_chunk is None or read_z_chunk <= 0:
        # Match the Zarr z-chunk so each batch fills complete chunks and avoids
        # read-modify-write amplification on partial z updates.
        return max(1, int(zarr_z_chunk))
    read_z = int(read_z_chunk)
    if read_z <= 0:
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            "read_z_chunk must be positive when explicitly set",
            {"read_z_chunk": read_z_chunk},
        )
    return read_z


def resolve_max_in_flight(
    worker_count: int,
    job_count: int,
    read_z_chunk: int,
    zarr_z_chunk: int,
) -> int:
    ceiling = max(1, min(worker_count + 1, job_count))
    if read_z_chunk >= zarr_z_chunk:
        return max(1, min(ceiling, 2))
    return ceiling


def resolve_compressor(compressor: Any) -> Any | None:
    if compressor is None or compressor == "none":
        return None
    if compressor == "default":
        from numcodecs import Blosc

        return Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
    if compressor == "fast":
        from numcodecs import Blosc

        return Blosc(cname="lz4", clevel=1, shuffle=Blosc.SHUFFLE)
    return compressor


def _open_output_group(zarr_mod: Any, output_path: Path) -> Any:
    """Create an empty Zarr group compatible with both zarr v2 and v3."""
    if hasattr(zarr_mod, "DirectoryStore"):
        store = zarr_mod.DirectoryStore(str(output_path))
        return zarr_mod.group(store=store, overwrite=True)
    # zarr>=3 removed DirectoryStore / create_dataset; keep format-2 on-disk layout
    # so the rest of the pipeline can still open the result.
    return zarr_mod.open_group(str(output_path), mode="w", zarr_format=2)


def _create_array(
    root: Any,
    dataset_name: str,
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: Any,
    compressor: Any,
) -> Any:
    if hasattr(root, "create_dataset"):
        return root.create_dataset(
            dataset_name,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressor=compressor,
        )
    return root.create_array(
        dataset_name,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressors=compressor,
    )


def _read_z_chunk(tiff_files: list[Path]) -> np.ndarray:
    return np.stack([tifffile.imread(str(path)) for path in tiff_files])


def _iter_read_jobs(
    depth: int,
    tiff_files: list[Path],
    read_z_chunk: int,
) -> Iterator[tuple[int, int, list[Path]]]:
    for start in range(0, depth, read_z_chunk):
        end = min(start + read_z_chunk, depth)
        yield start, end, tiff_files[start:end]


def _run_bounded_read_pipeline(
    dataset: Any,
    jobs: Iterable[tuple[int, int, list[Path]]],
    *,
    worker_count: int,
    executor_kind: Literal["thread", "process"] = "thread",
    max_in_flight: int | None = None,
) -> None:
    job_list = list(jobs)
    if not job_list:
        return

    executor_cls = ProcessPoolExecutor if executor_kind == "process" else ThreadPoolExecutor
    in_flight_limit = max_in_flight if max_in_flight is not None else max(1, min(worker_count + 1, len(job_list)))
    in_flight_limit = max(1, min(in_flight_limit, len(job_list)))
    pending: dict[Future[np.ndarray], tuple[int, int]] = {}
    job_iter = iter(job_list)

    def submit_next(pool: ThreadPoolExecutor | ProcessPoolExecutor) -> None:
        try:
            start, end, files = next(job_iter)
        except StopIteration:
            return
        pending[pool.submit(_read_z_chunk, files)] = (start, end)

    with executor_cls(max_workers=worker_count) as pool:
        for _ in range(in_flight_limit):
            submit_next(pool)

        progress = tqdm(total=len(job_list), desc="Converting to Zarr", unit="batch", leave=True)
        try:
            while pending:
                for future in as_completed(list(pending)):
                    start, end = pending.pop(future)
                    dataset[start:end] = future.result()
                    progress.update(1)
                    submit_next(pool)
                    break
        finally:
            progress.close()


def normalize_channel_label(value: str) -> str:
    text = str(value).strip()
    if not text:
        raise PipelineError(ErrorCode.ARGUMENT_INVALID, "Channel label must not be empty", {"channel": value})
    return text if text.startswith("ch") else f"ch{text}"


def parse_channel_list(channels: str | list[str]) -> list[str]:
    if isinstance(channels, list):
        parts = [str(part).strip() for part in channels if str(part).strip()]
    else:
        parts = [part.strip() for part in str(channels).split(",") if part.strip()]
    if not parts:
        raise PipelineError(ErrorCode.ARGUMENT_INVALID, "At least one channel is required", {"channels": channels})
    return [normalize_channel_label(part) for part in parts]


def convert_tiff_to_zarr(
    input_dir: str | Path,
    output_zarr: str | Path,
    chunk_size: tuple[int, int, int] = (256, 256, 256),
    compressor: Any = "default",
    *,
    dataset_name: str = "0",
    workers: int = 0,
    read_z_chunk: int | None = None,
    executor: Literal["thread", "process"] = "thread",
) -> dict[str, Any]:
    """Convert a folder of TIFF files into a directory-store Zarr volume."""
    started_at = time.time()
    input_path = Path(input_dir)
    output_path = Path(output_zarr)

    try:
        import zarr
    except ModuleNotFoundError as exc:
        raise PipelineError(
            ErrorCode.DEPENDENCY_MISSING,
            "zarr is required for TIFF-to-Zarr conversion",
            {"dependency": "zarr", "error": str(exc)},
        ) from exc

    try:
        resolved_compressor = resolve_compressor(compressor)
    except ModuleNotFoundError as exc:
        raise PipelineError(
            ErrorCode.DEPENDENCY_MISSING,
            "numcodecs is required for Zarr compression",
            {"dependency": "numcodecs", "error": str(exc)},
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

    root = _open_output_group(zarr, output_path)
    dataset = _create_array(
        root,
        dataset_name,
        shape=shape,
        chunks=chunk_size,
        dtype=dtype,
        compressor=resolved_compressor,
    )
    root.attrs["multiscales"] = [{
        "version": "0.4",
        "datasets": [{"path": dataset_name}],
    }]

    read_batch = resolve_read_z_chunk(chunk_size[0], read_z_chunk)
    worker_count = resolve_tiff_workers(
        workers,
        read_z_chunk=read_batch,
        zarr_z_chunk=chunk_size[0],
    )
    job_count = (int(shape[0]) + read_batch - 1) // read_batch
    in_flight_limit = resolve_max_in_flight(worker_count, job_count, read_batch, chunk_size[0])
    logger.info(
        "Using %d %s worker(s), read batch=%d slice(s), zarr z-chunk=%d, max in-flight batches=%d",
        worker_count,
        executor,
        read_batch,
        chunk_size[0],
        in_flight_limit,
    )

    _run_bounded_read_pipeline(
        dataset,
        _iter_read_jobs(shape[0], tiff_files, read_batch),
        worker_count=worker_count,
        executor_kind=executor,
        max_in_flight=in_flight_limit,
    )

    result = {
        "success": True,
        "input_dir": str(input_path),
        "output_zarr": str(output_path),
        "dataset_name": dataset_name,
        "shape": list(shape),
        "dtype": str(dtype),
        "chunk_size": list(chunk_size),
        "read_z_chunk": read_batch,
        "workers": worker_count,
        "executor": executor,
        "compressor": str(compressor),
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


def convert_sample_channels_to_zarr(
    sample_dir: str | Path,
    channels: str | list[str],
    *,
    chunk_size: tuple[int, int, int] = (256, 256, 256),
    workers: int = 0,
    read_z_chunk: int | None = None,
    executor: Literal["thread", "process"] = "thread",
    compressor: Any = "default",
    dataset_name: str = "0",
    skip_existing: bool = True,
) -> dict[str, Any]:
    sample_path = Path(sample_dir)
    if not sample_path.exists():
        raise PipelineError(
            ErrorCode.INPUT_NOT_FOUND,
            "Sample directory not found",
            {"sample_dir": str(sample_path)},
        )

    channel_labels = parse_channel_list(channels)
    converted: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}
    failed: dict[str, str] = {}

    for channel in channel_labels:
        output_zarr = sample_path / f"{channel}.zarr"
        if skip_existing and output_zarr.exists():
            skipped[channel] = str(output_zarr)
            continue
        input_dir = sample_path / channel
        if not input_dir.exists():
            failed[channel] = f"Missing TIFF folder and Zarr store: {input_dir}"
            continue
        try:
            converted[channel] = convert_tiff_to_zarr(
                input_dir,
                output_zarr,
                chunk_size=chunk_size,
                compressor=compressor,
                dataset_name=dataset_name,
                workers=workers,
                read_z_chunk=read_z_chunk,
                executor=executor,
            )
        except PipelineError as exc:
            failed[channel] = str(exc.message)
        except Exception as exc:  # pragma: no cover - defensive batch boundary
            failed[channel] = str(exc)

    if not converted and failed:
        raise PipelineError(
            ErrorCode.INPUT_NOT_FOUND,
            "No channels were converted to Zarr",
            {"failed": failed, "sample_dir": str(sample_path)},
        )

    return {
        "success": True,
        "sample_dir": str(sample_path),
        "channels_requested": channel_labels,
        "converted": converted,
        "skipped_existing": skipped,
        "failed": failed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a TIFF folder to a directory-store Zarr volume")
    parser.add_argument("--input", default="", help="Input TIFF folder")
    parser.add_argument("--output", default="", help="Output .zarr path")
    parser.add_argument("--sample_dir", default="", help="Sample root directory for --channels batch mode")
    parser.add_argument("--channels", default="", help="Comma-separated channels for batch mode, e.g. 0,1,2,3")
    parser.add_argument("--chunk_size", default="256,256,256", help="Chunk size z,y,x")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel workers for TIFF reads. 0 auto-uses min(8, cpu_count // 4).",
    )
    parser.add_argument(
        "--read-z-chunk",
        type=int,
        default=0,
        help="Slices read per worker batch. 0 matches the Zarr z-chunk (fastest; uses more RAM).",
    )
    parser.add_argument(
        "--executor",
        choices=("thread", "process"),
        default="thread",
        help="thread=low memory overhead (default); process=more CPU for TIFF decode, higher RAM.",
    )
    parser.add_argument(
        "--compressor",
        choices=("default", "fast", "none"),
        default="default",
        help="Zarr compression: default=zstd, fast=lz4, none=fastest write/largest store.",
    )
    parser.add_argument("--skip_existing", action="store_true", help="Skip channels whose .zarr already exists in batch mode")
    parser.add_argument("--dataset_name", default="0", help="Dataset name inside the Zarr group")
    parser.add_argument("--json_logs", action="store_true", help="Emit NDJSON log records to stderr")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _configure_logging(args.json_logs)

    try:
        if args.sample_dir and args.channels:
            result = convert_sample_channels_to_zarr(
                args.sample_dir,
                args.channels,
                chunk_size=_coerce_chunk_size(args.chunk_size),
                workers=int(args.workers),
                read_z_chunk=int(args.read_z_chunk) or None,
                executor=args.executor,
                compressor=args.compressor,
                dataset_name=args.dataset_name,
                skip_existing=bool(args.skip_existing),
            )
        else:
            if not args.input or not args.output:
                raise PipelineError(
                    ErrorCode.ARGUMENT_INVALID,
                    "Provide --input and --output, or use --sample_dir with --channels",
                    {"input": args.input, "output": args.output, "sample_dir": args.sample_dir},
                )
            result = convert_tiff_to_zarr(
                args.input,
                args.output,
                _coerce_chunk_size(args.chunk_size),
                compressor=args.compressor,
                dataset_name=args.dataset_name,
                workers=int(args.workers),
                read_z_chunk=int(args.read_z_chunk) or None,
                executor=args.executor,
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
