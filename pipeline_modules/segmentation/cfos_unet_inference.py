from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

try:
    from pipeline_modules.utils.errors import ErrorCode, PipelineError
    from pipeline_modules.utils.run_manifest import write_run_manifest
    from pipeline_modules.segmentation.cfos_unet_model import load_cfos_unet_checkpoint
    from pipeline_modules.segmentation.zarr_utils import create_output_zarr, list_existing_chunk_indices, open_zarr_dataset
except ImportError:  # pragma: no cover
    from ..utils.errors import ErrorCode, PipelineError
    from ..utils.run_manifest import write_run_manifest
    from .cfos_unet_model import load_cfos_unet_checkpoint
    from .zarr_utils import create_output_zarr, list_existing_chunk_indices, open_zarr_dataset

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


def _resolve_device(torch_mod, requested: str) -> str:
    requested = (requested or "auto").lower()
    if requested == "auto":
        return "cuda" if torch_mod.cuda.is_available() else "cpu"
    return requested


def _compute_tile_starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, max(stride, 1)))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def _iter_chunk_slices(shape: tuple[int, int, int], chunks: tuple[int, int, int]):
    for z in range(0, shape[0], chunks[0]):
        z_end = min(z + chunks[0], shape[0])
        for y in range(0, shape[1], chunks[1]):
            y_end = min(y + chunks[1], shape[1])
            for x in range(0, shape[2], chunks[2]):
                x_end = min(x + chunks[2], shape[2])
                yield (slice(z, z_end), slice(y, y_end), slice(x, x_end))


def _infer_volume(
    volume_np,
    *,
    model,
    torch_mod,
    patch_size: tuple[int, int, int],
    overlap: float,
    batch_size: int,
    device: str,
):
    import numpy as np

    tile_d, tile_h, tile_w = patch_size
    stride_d = max(1, int(round(tile_d * (1.0 - overlap))))
    stride_h = max(1, int(round(tile_h * (1.0 - overlap))))
    stride_w = max(1, int(round(tile_w * (1.0 - overlap))))

    d, h, w = volume_np.shape
    pad_d = max(tile_d - d, 0)
    pad_h = max(tile_h - h, 0)
    pad_w = max(tile_w - w, 0)
    if pad_d or pad_h or pad_w:
        volume_np = np.pad(volume_np, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant")
    padded_shape = volume_np.shape

    logits_acc = None
    count_acc = None
    pending_patches = []
    pending_coords = []

    z_starts = _compute_tile_starts(padded_shape[0], tile_d, stride_d)
    y_starts = _compute_tile_starts(padded_shape[1], tile_h, stride_h)
    x_starts = _compute_tile_starts(padded_shape[2], tile_w, stride_w)

    use_amp = device.startswith("cuda")

    def flush_pending():
        nonlocal logits_acc, count_acc, pending_patches, pending_coords
        if not pending_patches:
            return
        batch_np = np.stack(pending_patches, axis=0)[:, None, ...].astype("float32")
        batch_tensor = torch_mod.from_numpy(batch_np).to(device)
        with torch_mod.no_grad():
            if use_amp:
                with torch_mod.autocast(device_type="cuda", dtype=torch_mod.float16):
                    logits = model(batch_tensor).detach().float()
            else:
                logits = model(batch_tensor).detach().float()
        if logits_acc is None:
            num_classes = int(logits.shape[1])
            logits_acc = torch_mod.zeros((num_classes,) + padded_shape, dtype=torch_mod.float32, device=device)
            count_acc = torch_mod.zeros((1,) + padded_shape, dtype=torch_mod.float32, device=device)
        for batch_index, (z0, y0, x0) in enumerate(pending_coords):
            logits_acc[:, z0:z0 + tile_d, y0:y0 + tile_h, x0:x0 + tile_w] += logits[batch_index]
            count_acc[:, z0:z0 + tile_d, y0:y0 + tile_h, x0:x0 + tile_w] += 1.0
        pending_patches = []
        pending_coords = []

    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                pending_patches.append(volume_np[z0:z0 + tile_d, y0:y0 + tile_h, x0:x0 + tile_w])
                pending_coords.append((z0, y0, x0))
                if len(pending_patches) >= max(1, batch_size):
                    flush_pending()
    flush_pending()

    averaged_logits = (logits_acc / count_acc.clamp(min=1.0))[:, :d, :h, :w]
    return averaged_logits.cpu()


def _compute_global_percentiles(
    data_in,
    chunk_slices_iter,
    low_pct: float,
    high_pct: float,
):
    """Compute global percentile values across all chunks via histogram accumulation."""
    import numpy as np

    # Sub-pass 1: determine global min / max
    global_min = float("inf")
    global_max = float("-inf")
    for slices in tqdm(chunk_slices_iter, desc="Global min/max", unit="chunk"):
        chunk = np.asarray(data_in[slices])
        global_min = min(global_min, float(chunk.min()))
        global_max = max(global_max, float(chunk.max()))

    if global_max <= global_min:
        return float(global_min), float(global_max)

    # Sub-pass 2: accumulate histogram and derive percentiles
    num_bins = 1 << 16
    bin_edges = np.linspace(global_min, global_max, num_bins + 1)
    hist = np.zeros(num_bins, dtype=np.int64)

    for slices in tqdm(chunk_slices_iter, desc="Global histogram", unit="chunk"):
        chunk = np.asarray(data_in[slices])
        h, _ = np.histogram(chunk.ravel(), bins=bin_edges)
        hist += h.astype(np.int64)

    total = hist.sum()
    if total == 0:
        return float(global_min), float(global_max)

    cumsum = np.cumsum(hist)
    bin_width = (global_max - global_min) / num_bins

    def _interp_percentile(pct: float) -> float:
        target = total * pct / 100.0
        idx = int(np.searchsorted(cumsum, target))
        idx = min(max(idx, 0), num_bins - 1)
        count_before = float(cumsum[idx - 1]) if idx > 0 else 0.0
        count_in_bin = float(hist[idx])
        if count_in_bin > 0:
            fraction = (target - count_before) / count_in_bin
        else:
            fraction = 0.0
        return float(bin_edges[idx]) + fraction * bin_width

    return _interp_percentile(low_pct), _interp_percentile(high_pct)


def _normalize_with_bounds(volume, low: float, high: float):
    """Normalize volume to [0, 1] using pre-computed percentile bounds.

    Mirrors the logic of ``normalize_volume`` but uses externally computed
    percentile bounds so that all chunks share the same normalisation.
    When ``high <= low`` the entire volume is effectively constant, so
    return zeros (consistent with clip-then-subtract on a flat signal).
    """
    import numpy as np

    volume = volume.astype(np.float32, copy=False)
    if high <= low:
        return np.zeros_like(volume)
    volume = np.clip(volume, low, high)
    volume = volume - low
    volume = volume / max(high - low, 1e-6)
    return volume


def run_cfos_unet_inference(
    *,
    input_zarr: str | Path,
    output_zarr: str | Path,
    checkpoint_path: str | Path,
    probability_zarr: str | Path | None = None,
    dataset_name: str = "0",
    patch_size: tuple[int, int, int] | None = None,
    overlap: float = 0.25,
    batch_size: int = 4,
    device: str = "auto",
    foreground_class: int = 1,
    probability_threshold: float = 0.5,
    process_existing_only: bool = False,
    output_mode: str = "binary",
    output_dtype: str = "uint8",
    probability_dtype: str = "float32",
    chunk_size: tuple[int, int, int] | None = None,
    normalize_percentiles: tuple[float, float] = (1.0, 99.5),
    skip_empty: bool = False,
    skip_eps: float = 0.0,
) -> dict[str, Any]:
    import numpy as np

    started_at = time.time()
    input_path = Path(input_zarr)
    checkpoint = Path(checkpoint_path)
    if not input_path.exists():
        raise PipelineError(ErrorCode.INPUT_NOT_FOUND, "Input Zarr not found", {"input_zarr": str(input_path)})
    if not checkpoint.exists():
        raise PipelineError(ErrorCode.INPUT_NOT_FOUND, "Checkpoint not found", {"checkpoint_path": str(checkpoint)})
    probability_path = Path(probability_zarr) if probability_zarr else None
    if probability_path and probability_path == Path(output_zarr):
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            "probability_zarr must be different from output_zarr",
            {"probability_zarr": str(probability_path), "output_zarr": str(output_zarr)},
        )

    data_in = open_zarr_dataset(input_path, dataset_name=dataset_name)
    model_bundle = load_cfos_unet_checkpoint(checkpoint, device="cpu")
    torch_mod = model_bundle["torch"]
    resolved_device = _resolve_device(torch_mod, device)
    if resolved_device != "cpu":
        model_bundle = load_cfos_unet_checkpoint(checkpoint, device=resolved_device)
        torch_mod = model_bundle["torch"]

    model = model_bundle["model"]
    if hasattr(torch_mod, "compile") and resolved_device.startswith("cuda"):
        try:
            import triton  # noqa: F401
            logger.info("Compiling model with torch.compile ...")
            model = torch_mod.compile(model)
        except ImportError:
            logger.info("Triton not available (Windows), skipping torch.compile")
    checkpoint_args = model_bundle["checkpoint_args"]
    inferred_patch_size = patch_size or tuple(int(v) for v in checkpoint_args.get("patch_size", [128, 128, 128]))
    inferred_chunk_size = chunk_size or tuple(int(v) for v in getattr(data_in, "chunks", inferred_patch_size))
    output_dtype_np = np.dtype(output_dtype)
    probability_dtype_np = np.dtype(probability_dtype)
    _, data_out = create_output_zarr(output_zarr, data_in.shape, inferred_chunk_size, output_dtype_np, dataset_name=dataset_name)
    data_prob = None
    if probability_path:
        _, data_prob = create_output_zarr(
            probability_path,
            data_in.shape,
            inferred_chunk_size,
            probability_dtype_np,
            dataset_name=dataset_name,
        )

    processed_regions = 0
    skipped_chunks = 0
    if process_existing_only:
        existing_indices = list_existing_chunk_indices(data_in)
        if not existing_indices:
            raise PipelineError(
                ErrorCode.EMPTY_RESULT,
                "No physical chunks found in input store",
                {"input_zarr": str(input_path)},
            )
        chunk_slices_iter = []
        shape = tuple(int(v) for v in data_in.shape)
        chunks = tuple(int(v) for v in getattr(data_in, "chunks", inferred_chunk_size))
        for idx in existing_indices:
            slices = []
            for axis, chunk_idx_val in enumerate(idx):
                start = chunk_idx_val * chunks[axis]
                stop = min(start + chunks[axis], shape[axis])
                slices.append(slice(start, stop))
            chunk_slices_iter.append(tuple(slices))
    else:
        chunk_slices_iter = list(_iter_chunk_slices(tuple(int(v) for v in data_in.shape), inferred_chunk_size))

    total_chunks = len(chunk_slices_iter)
    global_low, global_high = _compute_global_percentiles(
        data_in,
        chunk_slices_iter,
        float(normalize_percentiles[0]),
        float(normalize_percentiles[1]),
    )
    logger.info(
        "Starting inference on %d chunks, patch_size=%s, batch_size=%d, device=%s%s"
        " | global norm bounds: low=%.4f high=%.4f (p%.1f, p%.1f)",
        total_chunks, inferred_patch_size, batch_size, resolved_device,
        ", skip_empty=True eps=%.4f" % skip_eps if skip_empty else "",
        global_low, global_high,
        float(normalize_percentiles[0]), float(normalize_percentiles[1]),
    )
    for slices in tqdm(chunk_slices_iter, desc="Inference", unit="chunk"):
        volume_np = np.asarray(data_in[slices])
        if skip_empty and float(volume_np.max()) <= float(skip_eps):
            data_out[slices] = np.zeros(volume_np.shape, dtype=output_dtype_np)
            if data_prob is not None:
                data_prob[slices] = np.zeros(volume_np.shape, dtype=probability_dtype_np)
            skipped_chunks += 1
            continue
        volume_np = _normalize_with_bounds(volume_np, global_low, global_high)
        logits = _infer_volume(
            volume_np,
            model=model,
            torch_mod=torch_mod,
            patch_size=inferred_patch_size,
            overlap=float(overlap),
            batch_size=int(batch_size),
            device=resolved_device,
        )
        probs = torch_mod.softmax(logits, dim=0)
        fg_probs = probs[int(foreground_class)].cpu().numpy()
        if output_mode == "multiclass":
            pred_np = torch_mod.argmax(logits, dim=0).cpu().numpy().astype(output_dtype_np, copy=False)
        else:
            pred_np = (fg_probs >= float(probability_threshold)).astype(output_dtype_np, copy=False)
        data_out[slices] = pred_np
        if data_prob is not None:
            data_prob[slices] = fg_probs.astype(probability_dtype_np, copy=False)
        processed_regions += 1

    if skip_empty and skipped_chunks:
        logger.info("Skipped %d/%d empty chunks (max <= %.4f)", skipped_chunks, total_chunks, skip_eps)

    result = {
        "success": True,
        "input_zarr": str(input_path),
        "output_zarr": str(Path(output_zarr)),
        "probability_zarr": str(probability_path) if probability_path else "",
        "checkpoint_path": str(checkpoint),
        "dataset_name": dataset_name,
        "shape": list(data_in.shape),
        "chunks": list(inferred_chunk_size),
        "patch_size": list(inferred_patch_size),
        "processed_regions": processed_regions,
        "skipped_chunks": skipped_chunks,
        "resolved_device": resolved_device,
        "num_classes": int(model_bundle["num_classes"]),
        "base_channels": int(model_bundle["base_channels"]),
        "output_mode": output_mode,
        "output_dtype": output_dtype,
        "probability_dtype": probability_dtype,
    }
    manifest_path = write_run_manifest(
        output_zarr,
        module="segmentation",
        entrypoint="run_cfos_unet_inference",
        inputs={
            "input_zarr": str(input_path),
            "output_zarr": str(output_zarr),
            "probability_zarr": str(probability_path) if probability_path else "",
            "checkpoint_path": str(checkpoint),
            "dataset_name": dataset_name,
            "patch_size": inferred_patch_size,
            "overlap": overlap,
            "batch_size": batch_size,
            "device": resolved_device,
            "foreground_class": foreground_class,
            "probability_threshold": probability_threshold,
            "process_existing_only": process_existing_only,
            "output_mode": output_mode,
            "output_dtype": output_dtype,
            "probability_dtype": probability_dtype,
            "chunk_size": inferred_chunk_size,
            "skip_empty": skip_empty,
            "skip_eps": skip_eps,
        },
        outputs=[path for path in [output_zarr, probability_path] if path],
        started_at=started_at,
        extra=result,
    )
    result["manifest_path"] = str(manifest_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local cFos 3D U-Net checkpoint inference on a Zarr volume")
    parser.add_argument("--input_zarr", required=True, help="Input signal Zarr path")
    parser.add_argument("--output_zarr", required=True, help="Output mask Zarr path")
    parser.add_argument("--checkpoint_path", required=True, help="Checkpoint produced by train_cfos_3d_mlflow.py")
    parser.add_argument("--probability_zarr", default="", help="Optional output Zarr path for foreground probabilities")
    parser.add_argument("--dataset_name", default="0", help="Dataset name inside the input/output Zarr groups")
    parser.add_argument("--patch_size", default="", help="Override patch size as z,y,x")
    parser.add_argument("--overlap", type=float, default=0.125, help="Sliding-window overlap ratio")
    parser.add_argument("--batch_size", type=int, default=4, help="Inference batch size in number of tiles")
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda")
    parser.add_argument("--foreground_class", type=int, default=1, help="Foreground class index")
    parser.add_argument("--probability_threshold", type=float, default=0.5, help="Foreground probability threshold for binary output")
    parser.add_argument("--process_existing_only", action="store_true", help="Only process physically present input chunks")
    parser.add_argument("--output_mode", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--output_dtype", default="uint8", help="numpy dtype for the output mask")
    parser.add_argument("--probability_dtype", default="float32", help="numpy dtype for optional probability output")
    parser.add_argument("--chunk_size", default="", help="Override output chunk size as z,y,x")
    parser.add_argument("--normalize_percentiles", default="1.0,99.5", help="Percentile pair low,high for normalization")
    parser.add_argument("--json_logs", action="store_true", help="Emit NDJSON log records to stderr")
    parser.add_argument("--skip_empty", action="store_true", default=True, help="Skip chunks whose max pixel value <= skip_eps")
    parser.add_argument("--no_skip_empty", action="store_false", dest="skip_empty", help="Disable skipping empty chunks")
    parser.add_argument("--skip_eps", type=float, default=100.0, help="Max-value threshold for skip_empty (default 100.0)")
    return parser.parse_args()


def _parse_triplet(value: str) -> tuple[int, int, int] | None:
    if not str(value).strip():
        return None
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 3:
        raise PipelineError(ErrorCode.ARGUMENT_INVALID, "Expected three comma-separated integers", {"value": value})
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def _parse_percentiles(value: str) -> tuple[float, float]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 2:
        raise PipelineError(ErrorCode.ARGUMENT_INVALID, "Expected two comma-separated percentiles", {"value": value})
    return (float(parts[0]), float(parts[1]))


def main() -> int:
    args = parse_args()
    _configure_logging(args.json_logs)
    try:
        result = run_cfos_unet_inference(
            input_zarr=args.input_zarr,
            output_zarr=args.output_zarr,
            checkpoint_path=args.checkpoint_path,
            probability_zarr=args.probability_zarr or None,
            dataset_name=args.dataset_name,
            patch_size=_parse_triplet(args.patch_size),
            overlap=args.overlap,
            batch_size=args.batch_size,
            device=args.device,
            foreground_class=args.foreground_class,
            probability_threshold=args.probability_threshold,
            process_existing_only=args.process_existing_only,
            output_mode=args.output_mode,
            output_dtype=args.output_dtype,
            probability_dtype=args.probability_dtype,
            chunk_size=_parse_triplet(args.chunk_size),
            normalize_percentiles=_parse_percentiles(args.normalize_percentiles),
            skip_empty=args.skip_empty,
            skip_eps=args.skip_eps,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except PipelineError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False), file=sys.stderr)
        return exc.exit_code
    except Exception as exc:  # pragma: no cover
        logger.exception("Unhandled cfos_unet inference error: %s", exc)
        wrapped = PipelineError(ErrorCode.INTERNAL_ERROR, "Unhandled cfos_unet inference error", {"error": str(exc)})
        print(json.dumps(wrapped.to_dict(), ensure_ascii=False), file=sys.stderr)
        return wrapped.exit_code


if __name__ == "__main__":
    sys.exit(main())
