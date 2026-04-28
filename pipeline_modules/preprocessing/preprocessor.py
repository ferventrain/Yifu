from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import tifffile

try:
    from pipeline_modules.preprocessing.config import PreprocessingCfg, layout_for_sample
    from pipeline_modules.utils.errors import ErrorCode, PipelineError
    from pipeline_modules.utils.run_manifest import write_run_manifest
except ImportError:  # pragma: no cover - fallback for package-relative execution
    from .config import PreprocessingCfg, layout_for_sample
    from ..utils.errors import ErrorCode, PipelineError
    from ..utils.run_manifest import write_run_manifest

from .median_filter import apply_median_filter
from .rolling_ball_background import rolling_ball_background

logger = logging.getLogger(__name__)

_BOOKKEEPING_KEYS = {"downsample", "zarr", "channel_subtraction"}


def _configure_logging(json_logs: bool) -> None:
    if json_logs:
        class _JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                payload = {
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }
                if hasattr(record, "event"):
                    payload["event"] = getattr(record, "event")
                return json.dumps(payload, ensure_ascii=False)

        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonFormatter())
        logging.root.handlers.clear()
        logging.root.addHandler(handler)
        logging.root.setLevel(logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _pipeline_error_to_stderr(exc: PipelineError) -> None:
    print(json.dumps(exc.to_dict(), ensure_ascii=False), file=sys.stderr)


def _load_full_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise PipelineError(
            ErrorCode.INPUT_NOT_FOUND,
            "Config file not found",
            {"config_path": str(path)},
        )
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise PipelineError(
            ErrorCode.CONFIG_INVALID,
            "Invalid JSON config",
            {"config_path": str(path), "error": str(exc)},
        ) from exc


def normalize_channel_list(channels: Any) -> list[str]:
    """Normalize values like ``1`` / ``"1"`` / ``"ch1"`` to ``["ch1"]``."""
    if channels is None:
        return []

    if isinstance(channels, (str, int)):
        raw_channels = [channels]
    elif isinstance(channels, (list, tuple, set)):
        raw_channels = list(channels)
    else:
        return []

    normalized: list[str] = []
    for channel in raw_channels:
        channel_str = str(channel).strip()
        if not channel_str:
            continue
        normalized.append(channel_str.lower() if channel_str.lower().startswith("ch") else f"ch{channel_str}")
    return list(dict.fromkeys(normalized))


def parse_filename(filename: str) -> tuple[str | None, str | None]:
    """Extract channel and Z-index from filename like ``..._C1_Z0051``."""
    match = re.search(r"_(C\d+)_Z(\d+)", filename)
    if match:
        return match.group(1), match.group(2)
    return None, None


def apply_processing_steps(img: np.ndarray, steps: list[tuple[str, Mapping[str, Any]]]) -> np.ndarray:
    """Apply configured preprocessing steps to an in-memory image."""
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise PipelineError(
            ErrorCode.DEPENDENCY_MISSING,
            "opencv-python is required for preprocessing steps",
            {"dependency": "cv2", "error": str(exc)},
        ) from exc

    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]

    current_img = img
    dtype = img.dtype

    for func, kwargs in steps:
        if func == "tophat":
            kernel_size = int(kwargs.get("kernel_size", 21))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            gray = current_img[:, :, 0] if len(current_img.shape) == 3 else current_img
            current_img = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel).astype(dtype)

        elif func == "rolling_ball":
            radius = int(kwargs.get("radius", 50))
            gray = current_img[:, :, 0] if len(current_img.shape) == 3 else current_img
            corrected, _ = rolling_ball_background(gray, radius=radius)
            current_img = corrected.astype(dtype)

        elif func == "median_filter":
            kernel_size = int(kwargs.get("kernel_size", 3))
            current_img = apply_median_filter(current_img, kernel_size=kernel_size)

        elif func == "scattering_removal":
            sigma = float(kwargs.get("sigma", 50.0))
            weight = float(kwargs.get("weight", 1.0))
            img_float = current_img.astype(np.float64)
            background = cv2.GaussianBlur(img_float, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
            result = np.clip(img_float - weight * background, 0, None)
            if np.issubdtype(dtype, np.integer):
                max_val = np.iinfo(dtype).max
                current_img = np.clip(result, 0, max_val).astype(dtype)
            else:
                current_img = result.astype(dtype)

        elif func == "clahe":
            clip_limit = float(kwargs.get("clip_limit", 2.0))
            tile_grid_size = int(kwargs.get("tile_grid_size", 8))
            clahe = cv2.createCLAHE(
                clipLimit=clip_limit,
                tileGridSize=(tile_grid_size, tile_grid_size),
            )
            gray = current_img[:, :, 0] if len(current_img.shape) == 3 else current_img
            current_img = clahe.apply(gray.astype(np.uint16)).astype(dtype)

        else:
            raise PipelineError(
                ErrorCode.CONFIG_INVALID,
                "Unsupported preprocessing step",
                {"step": func},
            )

    return current_img


def process_single_image(
    input_path: Path,
    output_path: Path,
    steps: list[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Process a single TIFF image; designed to run inside a worker process."""
    try:
        img = tifffile.imread(str(input_path))
        current_img = apply_processing_steps(img, steps)
        tifffile.imwrite(str(output_path), current_img, compression=None)
        return {"success": True, "input": str(input_path), "output": str(output_path)}
    except Exception as exc:  # pragma: no cover - exercised via parent process
        return {
            "success": False,
            "input": str(input_path),
            "output": str(output_path),
            "error": str(exc),
        }


def channel_subtraction_worker(
    cx_file: Path,
    c0_file: Path,
    output_path: Path,
    weight: float,
    steps: list[tuple[str, Mapping[str, Any]]] | None = None,
    compression: str | None = None,
) -> dict[str, Any]:
    """Perform channel subtraction and optional enhancement for one slice pair."""
    try:
        img_cx = tifffile.imread(str(cx_file))
        img_c0 = tifffile.imread(str(c0_file))

        if img_cx.shape != img_c0.shape:
            return {"success": False, "input": str(cx_file), "error": f"Shape mismatch with {c0_file.name}"}

        dtype = img_cx.dtype
        max_val = np.iinfo(dtype).max
        subtracted = np.clip(img_cx.astype(np.int32) - weight * img_c0.astype(np.int32), 0, max_val).astype(dtype)
        final_img = apply_processing_steps(subtracted, steps or [])
        tifffile.imwrite(str(output_path), final_img, compression=compression)
        return {"success": True, "input": str(cx_file), "output": str(output_path)}
    except Exception as exc:  # pragma: no cover - exercised via parent process
        return {"success": False, "input": str(cx_file), "error": str(exc)}


def _default_workers(max_workers: int | None) -> int:
    workers = max_workers if max_workers is not None else max(1, (os.cpu_count() or 2) // 2)
    if os.name == "nt" and workers > 61:
        logger.info("Capping worker count to 61 due to Windows multiprocessing limits")
        workers = 61
    return max(1, int(workers))


def _resolve_preprocessed_output_dir(
    *,
    sample_dir: Path,
    layout: Any,
    channel: str,
    output_root: str | Path | None,
) -> Path:
    if output_root is None:
        if channel == layout.signal_ch:
            return layout.signal_tiff_preprocessed_dir
        return sample_dir / f"{channel}_preprocessed"
    return Path(output_root) / f"{channel}_preprocessed"


class Preprocessor:
    """Configurable preprocessor that applies an ordered enhancement pipeline."""

    def __init__(self, preprocessing_config: Mapping[str, Any] | PreprocessingCfg):
        self.model = (
            preprocessing_config
            if isinstance(preprocessing_config, PreprocessingCfg)
            else PreprocessingCfg.model_validate(preprocessing_config)
        )
        self.config = self.model.model_dump()
        self.steps = self._build_steps()

    def _build_steps(self) -> list[tuple[str, Mapping[str, Any]]]:
        steps: list[tuple[str, Mapping[str, Any]]] = []
        for step_name, step_config in self.config.items():
            if step_name in _BOOKKEEPING_KEYS:
                continue
            if not isinstance(step_config, dict):
                continue
            if not step_config.get("apply", False):
                continue
            steps.append((step_name, step_config))
        return steps

    def process_folder_result(
        self,
        input_folder: str | Path,
        output_folder: str | Path,
        *,
        max_workers: int | None = None,
        resume: bool = True,
        manifest_inputs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        started_at = time.time()
        input_path = Path(input_folder)
        output_path = Path(output_folder)

        if not input_path.exists():
            raise PipelineError(
                ErrorCode.INPUT_NOT_FOUND,
                "Input TIFF directory not found",
                {"input_folder": str(input_path)},
            )
        if not input_path.is_dir():
            raise PipelineError(
                ErrorCode.INPUT_FORMAT_INVALID,
                "Input preprocessing path must be a directory",
                {"input_folder": str(input_path)},
            )

        tiff_files = sorted(input_path.glob("*.tif*"))
        if not tiff_files:
            raise PipelineError(
                ErrorCode.INPUT_NOT_FOUND,
                "No TIFF files found for preprocessing",
                {"input_folder": str(input_path)},
            )

        output_path.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []
        logger.info("Found %d TIFF files in %s", len(tiff_files), input_path)
        logger.info("Enabled enhancement steps: %s", [name for name, _ in self.steps])

        tasks: list[tuple[Path, Path, list[tuple[str, Mapping[str, Any]]]]] = []
        skipped_existing = 0
        for tiff_file in tiff_files:
            output_file = output_path / tiff_file.name
            if resume and output_file.exists():
                skipped_existing += 1
                continue
            tasks.append((tiff_file, output_file, self.steps))

        processed = 0
        failed = 0

        if not self.steps:
            warnings.append("No enhancement steps enabled; output directory was created without image processing.")
            logger.info("No enhancement steps enabled; nothing to process")
        elif tasks:
            workers = _default_workers(max_workers)
            logger.info("Processing %d TIFF slices with %d workers", len(tasks), workers)
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(process_single_image, *task): task[0].name for task in tasks}
                for future in as_completed(futures):
                    result = future.result()
                    if result["success"]:
                        processed += 1
                    else:
                        failed += 1
                        logger.error("Failed processing %s: %s", futures[future], result.get("error", "unknown error"))
        else:
            logger.info("All TIFF slices already processed; resume skipped %d files", skipped_existing)

        result = {
            "success": failed == 0,
            "input_folder": str(input_path),
            "output_dir": str(output_path),
            "steps": [name for name, _ in self.steps],
            "total_files": len(tiff_files),
            "processed_files": processed,
            "skipped_existing": skipped_existing,
            "failed_files": failed,
            "warnings": warnings,
        }

        manifest_path = write_run_manifest(
            output_path,
            module="preprocessing",
            entrypoint="Preprocessor.process_folder_result",
            inputs=dict(manifest_inputs or {}, input_folder=str(input_path), steps=result["steps"], resume=resume),
            outputs=[output_path],
            started_at=started_at,
            warnings=warnings,
            extra={k: v for k, v in result.items() if k not in {"warnings", "success"}},
        )
        result["manifest_path"] = str(manifest_path)
        return result

    def process_folder(
        self,
        input_folder: str | Path,
        output_folder: str | Path,
        max_workers: int | None = None,
        resume: bool = True,
    ) -> bool:
        """Backwards-compatible boolean wrapper used by ``main.py``."""
        return self.process_folder_result(
            input_folder,
            output_folder,
            max_workers=max_workers,
            resume=resume,
        )["success"]


def run_channel_subtraction(
    *,
    root_path: str | Path,
    background_channel: str = "ch0",
    weight: float = 1.0,
    adaptive: bool = False,
    save_plots: bool = False,
    sample_ratio: float = 0.005,
    min_samples: int = 10,
    max_samples: int = 50,
    max_workers: int | None = None,
    compression: str | None = None,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    target_channels: list[str] | None = None,
    enhancement_steps: list[tuple[str, Mapping[str, Any]]] | None = None,
    layout: Any = None,
) -> dict[str, Any]:
    from .channel_subtraction import estimate_global_weight

    started_at = time.time()
    root = Path(root_path)
    if not root.is_dir():
        raise PipelineError(
            ErrorCode.INPUT_NOT_FOUND,
            "Sample directory not found",
            {"sample_dir": str(root)},
        )

    background_channel = normalize_channel_list([background_channel])[0]
    bg_folders = [
        folder for folder in root.iterdir()
        if folder.is_dir() and (folder.name.lower() == background_channel or folder.name.lower().startswith(f"{background_channel}_"))
    ]
    if not bg_folders:
        raise PipelineError(
            ErrorCode.INPUT_NOT_FOUND,
            "Background channel directory not found",
            {"background_channel": background_channel, "sample_dir": str(root)},
        )

    bg_dir = bg_folders[0]
    bg_files: dict[str, Path] = {}
    for file_path in bg_dir.glob("*.tif*"):
        _, z_idx = parse_filename(file_path.name)
        if z_idx:
            bg_files[z_idx] = file_path
    if not bg_files:
        raise PipelineError(
            ErrorCode.INPUT_NOT_FOUND,
            "No valid TIFF files found in background channel directory",
            {"background_dir": str(bg_dir)},
        )

    target_channels = normalize_channel_list(target_channels)
    channels_to_process: list[str] = []
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        name = folder.name.lower()
        if not name.startswith("ch"):
            continue
        if folder == bg_dir or name.endswith("_subtracted") or name.endswith("_preprocessed"):
            continue
        if target_channels and name not in target_channels:
            continue
        channels_to_process.append(name)

    if target_channels and not channels_to_process:
        raise PipelineError(
            ErrorCode.INPUT_NOT_FOUND,
            "No matching target channel directories found",
            {"target_channels": target_channels},
        )

    workers = _default_workers(max_workers)
    logger.info("Using %s as background channel", bg_dir.name)
    logger.info("Found %d reference TIFF slices in %s", len(bg_files), bg_dir)
    logger.info("Running channel subtraction for channels: %s", channels_to_process)

    estimated_weights: dict[str, float] = {}
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, encoding="utf-8") as fh:
                config_data = json.load(fh)
            estimated_weights = dict(
                config_data.get("preprocessing", {})
                .get("channel_subtraction", {})
                .get("estimated_weights", {})
            )
        except Exception as exc:
            logger.warning("Failed loading cached estimated weights from %s: %s", config_path, exc)

    channel_results: list[dict[str, Any]] = []
    all_warnings: list[str] = []

    for channel in channels_to_process:
        channel_started_at = time.time()
        cx_input_dir = root / channel
        target_output_dir = _resolve_preprocessed_output_dir(
            sample_dir=root,
            layout=layout or layout_for_sample(root, signal_ch="ch0", reg_ch="ch1"),
            channel=channel,
            output_root=output_dir,
        )
        target_output_dir.mkdir(parents=True, exist_ok=True)

        cx_files = sorted(cx_input_dir.glob("*.tif*"))
        if not cx_files:
            warning = f"No TIFF files found in {cx_input_dir}"
            all_warnings.append(warning)
            logger.warning(warning)
            continue

        matched_pairs: list[tuple[Path, Path, Path, float, list[tuple[str, Mapping[str, Any]]] | None, str | None]] = []
        for cx_file in cx_files:
            _, z_idx = parse_filename(cx_file.name)
            if z_idx in bg_files:
                matched_pairs.append(
                    (
                        cx_file,
                        bg_files[z_idx],
                        target_output_dir / cx_file.name,
                        weight,
                        enhancement_steps,
                        compression,
                    )
                )

        if not matched_pairs:
            warning = f"No matched TIFF slice pairs found for {channel}"
            all_warnings.append(warning)
            logger.warning(warning)
            continue

        if channel in estimated_weights:
            global_a = float(estimated_weights[channel])
            final_weight = weight * global_a
            logger.info("Using cached adaptive weight for %s: a=%.4f, effective_weight=%.4f", channel, global_a, final_weight)
        elif adaptive:
            n_total = len(matched_pairs)
            n_sample = max(min_samples, min(max_samples, int(n_total * sample_ratio)))
            step = max(1, n_total // max(1, n_sample))
            idx_sample = list(range(0, n_total, step))[:n_sample]
            sample_pairs = [matched_pairs[i] for i in idx_sample]
            sample_cx = [pair[0] for pair in sample_pairs]
            sample_c0 = [pair[1] for pair in sample_pairs]
            plot_path = target_output_dir / f"{channel}_global_fit.png" if save_plots else None
            global_a = estimate_global_weight(sample_cx, sample_c0, plot_path=plot_path)
            estimated_weights[channel] = float(global_a)
            final_weight = weight * float(global_a)
            logger.info("Estimated adaptive weight for %s: a=%.4f, effective_weight=%.4f", channel, global_a, final_weight)
        else:
            final_weight = weight

        for index in range(len(matched_pairs)):
            matched_pairs[index] = (
                matched_pairs[index][0],
                matched_pairs[index][1],
                matched_pairs[index][2],
                final_weight,
                matched_pairs[index][4],
                matched_pairs[index][5],
            )

        tasks = [pair for pair in matched_pairs if not pair[2].exists()]
        processed = 0
        failed = 0
        skipped_existing = len(matched_pairs) - len(tasks)

        if tasks:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(channel_subtraction_worker, *task): task[0].name for task in tasks}
                for future in as_completed(futures):
                    result = future.result()
                    if result["success"]:
                        processed += 1
                    else:
                        failed += 1
                        logger.error("Channel subtraction failed for %s: %s", futures[future], result.get("error", "unknown error"))

        channel_result = {
            "channel": channel,
            "input_folder": str(cx_input_dir),
            "background_folder": str(bg_dir),
            "output_dir": str(target_output_dir),
            "matched_pairs": len(matched_pairs),
            "processed_files": processed,
            "skipped_existing": skipped_existing,
            "failed_files": failed,
            "effective_weight": float(final_weight),
            "enhancement_steps": [name for name, _ in (enhancement_steps or [])],
        }
        manifest_path = write_run_manifest(
            target_output_dir,
            module="preprocessing",
            entrypoint="run_channel_subtraction",
            inputs={
                "sample_dir": str(root),
                "channel": channel,
                "background_channel": background_channel,
                "weight": weight,
                "adaptive": adaptive,
                "effective_weight": float(final_weight),
                "output_dir": str(target_output_dir),
            },
            outputs=[target_output_dir],
            started_at=channel_started_at,
            warnings=[],
            extra=channel_result,
        )
        channel_result["manifest_path"] = str(manifest_path)
        channel_result["success"] = failed == 0
        channel_results.append(channel_result)

    if config_path and adaptive and estimated_weights:
        try:
            with open(config_path, encoding="utf-8") as fh:
                config_data = json.load(fh)
            config_data.setdefault("preprocessing", {}).setdefault("channel_subtraction", {})["estimated_weights"] = estimated_weights
            with open(config_path, "w", encoding="utf-8") as fh:
                json.dump(config_data, fh, indent=2, ensure_ascii=False)
        except Exception as exc:
            warning = f"Failed saving estimated weights back to config: {exc}"
            all_warnings.append(warning)
            logger.warning(warning)

    return {
        "success": all(result["success"] for result in channel_results) if channel_results else False,
        "sample_dir": str(root),
        "background_channel": background_channel,
        "channels": channel_results,
        "warnings": all_warnings,
        "duration_seconds": time.time() - started_at,
    }


def run_preprocessing(
    *,
    sample_dir: str | Path,
    preprocessing_config: Mapping[str, Any] | PreprocessingCfg,
    channel: str | None = None,
    max_workers: int | None = None,
    resume: bool = True,
    output_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    signal_ch: str = "ch0",
    reg_ch: str = "ch1",
) -> dict[str, Any]:
    """Run standalone preprocessing and return a machine-readable summary."""
    started_at = time.time()
    sample_path = Path(sample_dir)
    if not sample_path.exists():
        raise PipelineError(
            ErrorCode.INPUT_NOT_FOUND,
            "Sample directory not found",
            {"sample_dir": str(sample_path)},
        )

    try:
        model = (
            preprocessing_config
            if isinstance(preprocessing_config, PreprocessingCfg)
            else PreprocessingCfg.model_validate(preprocessing_config)
        )
    except Exception as exc:
        raise PipelineError(
            ErrorCode.CONFIG_INVALID,
            "Invalid preprocessing config",
            {"error": str(exc)},
        ) from exc

    layout = layout_for_sample(sample_path, signal_ch=signal_ch, reg_ch=reg_ch, require_exists=True)
    preprocessor = Preprocessor(model)
    target_channels = normalize_channel_list(channel) if channel else list(model.channels)
    if not target_channels:
        raise PipelineError(
            ErrorCode.CONFIG_INVALID,
            "No target channels configured",
            {"hint": "Set preprocessing.channels or pass --channel"},
        )

    result: dict[str, Any] = {
        "success": True,
        "sample_dir": str(sample_path),
        "signal_ch": signal_ch,
        "reg_ch": reg_ch,
        "target_channels": target_channels,
        "channel_subtraction": bool(model.channel_subtraction.apply),
        "steps": [name for name, _ in preprocessor.steps],
        "channels": [],
        "manifest_paths": [],
        "duration_seconds": 0.0,
    }

    if model.channel_subtraction.apply:
        subtraction_result = run_channel_subtraction(
            root_path=sample_path,
            background_channel=model.channel_subtraction.background_channel,
            weight=model.channel_subtraction.weight,
            adaptive=model.channel_subtraction.adaptive,
            save_plots=model.channel_subtraction.save_plots,
            sample_ratio=model.channel_subtraction.sample_ratio,
            min_samples=model.channel_subtraction.min_samples,
            max_samples=model.channel_subtraction.max_samples,
            max_workers=max_workers,
            compression=None if model.channel_subtraction.compression == "none" else model.channel_subtraction.compression,
            config_path=config_path,
            output_dir=output_dir,
            target_channels=target_channels,
            enhancement_steps=preprocessor.steps,
            layout=layout,
        )
        result["success"] = subtraction_result["success"]
        result["channels"] = subtraction_result["channels"]
        result["warnings"] = subtraction_result.get("warnings", [])
        result["manifest_paths"] = [item["manifest_path"] for item in subtraction_result["channels"] if item.get("manifest_path")]
    else:
        channel_results: list[dict[str, Any]] = []
        manifest_paths: list[str] = []
        warnings: list[str] = []
        for target_channel in target_channels:
            input_folder = sample_path / target_channel
            output_folder = _resolve_preprocessed_output_dir(
                sample_dir=sample_path,
                layout=layout,
                channel=target_channel,
                output_root=output_dir,
            )
            channel_result = preprocessor.process_folder_result(
                input_folder=input_folder,
                output_folder=output_folder,
                max_workers=max_workers,
                resume=resume,
                manifest_inputs={
                    "sample_dir": str(sample_path),
                    "channel": target_channel,
                    "signal_ch": signal_ch,
                    "reg_ch": reg_ch,
                },
            )
            channel_result["channel"] = target_channel
            channel_results.append(channel_result)
            manifest_paths.append(channel_result["manifest_path"])
            warnings.extend(channel_result.get("warnings", []))
            result["success"] = result["success"] and bool(channel_result["success"])
        result["channels"] = channel_results
        result["warnings"] = warnings
        result["manifest_paths"] = manifest_paths

    result["duration_seconds"] = time.time() - started_at
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone configurable preprocessor with channel subtraction and enhancement"
    )
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument("--sample_dir", required=True, help="Sample root directory containing channel folders")
    parser.add_argument("--channel", help='Only process this specific channel (e.g. "1" or "ch1")')
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: CPU//2)")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume (reprocess all files)")
    parser.add_argument("--output_dir", help="Custom output directory root (default: sample_dir)")
    parser.add_argument("--json_logs", action="store_true", help="Emit NDJSON log records to stderr")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _configure_logging(args.json_logs)

    try:
        full_config = _load_full_config(args.config)
        preprocessing_cfg = full_config.get("preprocessing", {})
        input_cfg = full_config.get("input", {})
        channels_cfg = input_cfg.get("channels", {})
        signal_ch = normalize_channel_list([channels_cfg.get("signal", "0")])[0]
        reg_ch = normalize_channel_list([channels_cfg.get("registration", "1")])[0]

        result = run_preprocessing(
            sample_dir=args.sample_dir,
            preprocessing_config=preprocessing_cfg,
            channel=args.channel,
            max_workers=args.workers,
            resume=not args.no_resume,
            output_dir=args.output_dir,
            config_path=args.config,
            signal_ch=signal_ch,
            reg_ch=reg_ch,
        )
        if not result.get("success", False):
            raise PipelineError(
                ErrorCode.INTERNAL_ERROR,
                "Preprocessing completed with failures",
                {
                    "sample_dir": args.sample_dir,
                    "target_channels": result.get("target_channels", []),
                    "manifest_paths": result.get("manifest_paths", []),
                },
            )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except PipelineError as exc:
        _pipeline_error_to_stderr(exc)
        return exc.exit_code
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        logger.exception("Unhandled preprocessing error: %s", exc)
        wrapped = PipelineError(ErrorCode.INTERNAL_ERROR, "Unhandled preprocessing error", {"error": str(exc)})
        _pipeline_error_to_stderr(wrapped)
        return wrapped.exit_code


if __name__ == "__main__":
    sys.exit(main())
