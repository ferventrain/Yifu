from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

try:
    from pipeline_modules.utils.errors import ErrorCode, PipelineError
    from pipeline_modules.utils.run_manifest import write_run_manifest
    from pipeline_modules.segmentation.zarr_utils import create_output_zarr, list_existing_chunk_indices, open_zarr_dataset
except ImportError:  # pragma: no cover
    from ..utils.errors import ErrorCode, PipelineError
    from ..utils.run_manifest import write_run_manifest
    from .zarr_utils import create_output_zarr, list_existing_chunk_indices, open_zarr_dataset

logger = logging.getLogger(__name__)


def _require_threshold_stack():
    try:
        import numpy as np
        from scipy import ndimage
        from skimage.filters import threshold_otsu
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "numpy, scipy, and scikit-image are required for threshold segmentation"
        ) from exc
    return np, ndimage, threshold_otsu


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


def segment_chunk(img, threshold, sigma, min_size, output_mode: str = "label"):
    np, ndimage, threshold_otsu = _require_threshold_stack()

    if sigma > 0:
        img = ndimage.gaussian_filter(img, sigma=sigma)

    if threshold == "otsu":
        try:
            thresh_val = threshold_otsu(img)
        except ValueError:
            thresh_val = 0
    else:
        thresh_val = float(threshold)

    binary_mask = img > thresh_val
    labeled, num_features = ndimage.label(binary_mask)

    if min_size > 0:
        sizes = ndimage.sum(binary_mask, labeled, range(num_features + 1))
        mask_size = sizes < min_size
        remove_pixel = mask_size[labeled]
        labeled[remove_pixel] = 0

    if output_mode == "binary":
        return (labeled > 0).astype(np.uint8)
    return labeled.astype(np.uint16)


def run_threshold_segmentation(
    *,
    input_zarr: str | Path,
    output_zarr: str | Path,
    threshold: str = "otsu",
    sigma: float = 0.0,
    min_size: int = 10,
    output_mode: str = "binary",
    dataset_name: str = "0",
    process_existing_only: bool = False,
) -> dict[str, Any]:
    started_at = time.time()
    input_path = Path(input_zarr)
    if not input_path.exists():
        raise PipelineError(
            ErrorCode.INPUT_NOT_FOUND,
            "Input Zarr not found",
            {"input_zarr": str(input_path)},
        )

    data_in = open_zarr_dataset(input_path, dataset_name=dataset_name)
    shape = tuple(int(v) for v in data_in.shape)
    chunks = tuple(int(v) for v in data_in.chunks)
    output_dtype = "uint8" if output_mode == "binary" else "uint16"
    _, data_out = create_output_zarr(output_zarr, shape, chunks, output_dtype, dataset_name=dataset_name)

    processed_chunks = 0
    if process_existing_only:
        existing_indices = list_existing_chunk_indices(data_in)
        if not existing_indices:
            raise PipelineError(
                ErrorCode.EMPTY_RESULT,
                "No physical chunks found in input store",
                {"input_zarr": str(input_path)},
            )
        for idx in existing_indices:
            slices = []
            for axis, chunk_idx in enumerate(idx):
                start = chunk_idx * chunks[axis]
                stop = min(start + chunks[axis], shape[axis])
                slices.append(slice(start, stop))
            slices = tuple(slices)
            data_out[slices] = segment_chunk(
                data_in[slices],
                threshold,
                sigma,
                min_size,
                output_mode=output_mode,
            )
            processed_chunks += 1
    else:
        z_chunk_size = chunks[0]
        for z in range(0, shape[0], z_chunk_size):
            z_end = min(z + z_chunk_size, shape[0])
            data_out[z:z_end, :, :] = segment_chunk(
                data_in[z:z_end, :, :],
                threshold,
                sigma,
                min_size,
                output_mode=output_mode,
            )
            processed_chunks += 1

    result = {
        "success": True,
        "input_zarr": str(input_path),
        "output_zarr": str(Path(output_zarr)),
        "dataset_name": dataset_name,
        "shape": list(shape),
        "chunks": list(chunks),
        "processed_chunks": processed_chunks,
        "threshold": str(threshold),
        "sigma": float(sigma),
        "min_size": int(min_size),
        "output_mode": output_mode,
    }
    manifest_path = write_run_manifest(
        output_zarr,
        module="segmentation",
        entrypoint="run_threshold_segmentation",
        inputs={
            "input_zarr": str(input_path),
            "output_zarr": str(output_zarr),
            "dataset_name": dataset_name,
            "threshold": threshold,
            "sigma": sigma,
            "min_size": min_size,
            "output_mode": output_mode,
            "process_existing_only": process_existing_only,
        },
        outputs=[output_zarr],
        started_at=started_at,
        extra=result,
    )
    result["manifest_path"] = str(manifest_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run threshold segmentation on a Zarr volume")
    parser.add_argument("--input_zarr", required=True, help="Path to input .zarr directory")
    parser.add_argument("--output_zarr", required=True, help="Path for output .zarr directory")
    parser.add_argument("--threshold", default="otsu", help='Threshold value or "otsu"')
    parser.add_argument("--sigma", type=float, default=0.0, help="Gaussian smoothing sigma")
    parser.add_argument("--min_size", type=int, default=10, help="Minimum object size")
    parser.add_argument("--output_mode", choices=["label", "binary"], default="binary")
    parser.add_argument("--dataset_name", default="0", help="Dataset name inside the Zarr group")
    parser.add_argument("--test", action="store_true", help="Only process chunks that physically exist in the input store")
    parser.add_argument("--json_logs", action="store_true", help="Emit NDJSON log records to stderr")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _configure_logging(args.json_logs)
    try:
        result = run_threshold_segmentation(
            input_zarr=args.input_zarr,
            output_zarr=args.output_zarr,
            threshold=args.threshold,
            sigma=args.sigma,
            min_size=args.min_size,
            output_mode=args.output_mode,
            dataset_name=args.dataset_name,
            process_existing_only=args.test,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except PipelineError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False), file=sys.stderr)
        return exc.exit_code
    except Exception as exc:  # pragma: no cover
        logger.exception("Unhandled threshold segmentation error: %s", exc)
        wrapped = PipelineError(ErrorCode.INTERNAL_ERROR, "Unhandled threshold segmentation error", {"error": str(exc)})
        print(json.dumps(wrapped.to_dict(), ensure_ascii=False), file=sys.stderr)
        return wrapped.exit_code


if __name__ == "__main__":
    sys.exit(main())
