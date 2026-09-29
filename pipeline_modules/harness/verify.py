"""Post-run verifier and artifact manifest for harness jobs.

Success is NEVER decided by "ALL DONE" text, exit codes alone, or log lines.
Minimal success condition (all four required):

1. the pipeline process exited normally (return code 0), AND
2. every required output exists, AND
3. every required output can actually be opened/read, AND
4. the artifact manifest was written.

Anything else is ``failed`` with a structured error explaining which of the
four legs broke, whether it is retryable, and what to do next.
"""

from __future__ import annotations

import gzip
import zipfile
from pathlib import Path
from typing import Any

from pipeline_modules.harness.jsonio import utc_now_iso, write_json_atomic
from pipeline_modules.harness.paths import artifacts_path
from pipeline_modules.harness.progress import PIPELINE_MODE_SPINAL_CORD, pipeline_mode, read_progress
from pipeline_modules.harness.results import collect_existing_results, layout_from_config, normalize_channel_label
from pipeline_modules.utils.errors import ErrorCode

ARTIFACTS_SCHEMA_VERSION = "1"

CUDA_OOM_MARKERS = (
    "CUDA out of memory",
    "torch.cuda.OutOfMemoryError",
    "CUDA error: out of memory",
    "RuntimeError: CUDA error: out of memory",
)


def required_outputs(sample_dir: str | Path, config: dict[str, Any], extra_args: list[str] | None = None) -> list[Path]:
    """Final deliverables whose presence defines success for this flow.

    Deliberately short: one authoritative file per standard flow. Partial or
    custom flows (attach_external drivers) are not verified here.
    """
    extras = [str(item) for item in (extra_args or [])]
    layout = layout_from_config(sample_dir, config)
    if "--only_tubule" in extras or "--only_region_analysis" in extras:
        return [layout.tubule_region_summary_csv]
    if pipeline_mode(config) == PIPELINE_MODE_SPINAL_CORD:
        return [layout.spinal_segment_stats_xlsx]
    seg_cfg = config.get("segmentation") or {}
    if str(seg_cfg.get("method") or "").lower() == "spotiflow":
        signal_ch = normalize_channel_label((config.get("input") or {}).get("channels", {}).get("signal", "0"))
        model_cfg = seg_cfg.get("spotiflow") or {}
        raw = model_cfg.get("output_csv") or f"ch{signal_ch[2:]}_spotiflow_points.csv"
        path = Path(raw)
        return [path if path.is_absolute() else Path(sample_dir) / path]
    outputs = [layout.brain_distribution_stats_xlsx]
    tubule_cfg = config.get("tubule_reconstruction") or {}
    region_cfg = tubule_cfg.get("region_analysis") or {}
    if tubule_cfg.get("enabled") and region_cfg.get("enabled"):
        outputs.append(layout.tubule_region_summary_csv)
    return outputs


def output_is_readable(path: Path) -> bool:
    """Open the file just enough to prove it is not truncated garbage."""
    try:
        if not path.exists():
            return False
        if path.is_dir():
            return _zarr_store_is_readable(path)
        suffix = "".join(path.suffixes[-2:]).lower()
        if suffix.endswith(".xlsx"):
            return zipfile.is_zipfile(path)
        if suffix.endswith(".csv") or suffix.endswith(".json"):
            with path.open("rb") as handle:
                return len(handle.read(64)) > 0
        if suffix.endswith(".nii.gz"):
            with gzip.open(path, "rb") as handle:
                return len(handle.read(16)) > 0
        with path.open("rb") as handle:
            return len(handle.read(16)) > 0
    except OSError:
        return False


def _zarr_store_is_readable(path: Path) -> bool:
    if (path / "zarr.json").is_file() or (path / ".zgroup").is_file():
        return True
    # Directory that simply is not a zarr store still counts as present-but-
    # opaque; only a missing marker AND no contents at all is unreadable.
    try:
        return any(path.iterdir())
    except OSError:
        return False


def verify_outputs(sample_dir: str | Path, config: dict[str, Any], extra_args: list[str] | None = None) -> dict[str, Any]:
    required = required_outputs(sample_dir, config, extra_args)
    rows = []
    for path in required:
        rows.append({"path": str(path), "exists": path.exists(), "readable": output_is_readable(path)})
    return {
        "ok": bool(rows) and all(row["exists"] and row["readable"] for row in rows),
        "required": rows,
    }


def write_artifact_manifest(
    job_id: str,
    active: Path,
    sample_dir: str | Path,
    config: dict[str, Any],
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Persist the artifact manifest (the 4th success condition)."""
    manifest = {
        "schema_version": ARTIFACTS_SCHEMA_VERSION,
        "run_id": job_id,
        "written_at": utc_now_iso(),
        "verification": verify_outputs(sample_dir, config, extra_args),
        "artifacts": collect_existing_results(sample_dir, config),
    }
    write_json_atomic(artifacts_path(job_id, active), manifest)
    return manifest


def structured_error(
    *,
    code: str,
    message: str,
    step_id: int | str | None = None,
    retryable: bool = False,
    suggestion: str = "",
) -> dict[str, Any]:
    return {
        "code": str(code),
        "message": str(message),
        "step_id": step_id,
        "retryable": bool(retryable),
        "suggestion": str(suggestion or ""),
    }


def normalize_error(value: Any) -> dict[str, Any] | None:
    """Coerce legacy string errors into the structured shape (read-only)."""
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        return structured_error(
            code=str(value.get("code") or "INTERNAL_ERROR"),
            message=str(value.get("message") or ""),
            step_id=value.get("step_id"),
            retryable=bool(value.get("retryable")),
            suggestion=str(value.get("suggestion") or ""),
        )
    return structured_error(code="INTERNAL_ERROR", message=str(value))


def classify_failure(returncode: int, log_tail: str, progress: dict[str, Any] | None) -> dict[str, Any]:
    """Turn a failed run into ONE structured error the user can act on.

    Categories: configuration / input / environment / program / output
    verification. The log tail is only used for error CLASSIFICATION (OOM),
    never for completion detection.
    """
    progress = progress or {}
    step_id = progress.get("step_index")
    message = str(progress.get("error") or "").strip()
    # SystemExit(1) and friends stringify to bare numbers/digits; that is no
    # information — fall through to the exit-code message instead.
    if message.isdigit():
        message = ""
    for marker in CUDA_OOM_MARKERS:
        if marker.lower() in (log_tail or "").lower():
            return structured_error(
                code=ErrorCode.CUDA_OOM.value,
                message="CUDA ran out of memory during the run.",
                step_id=step_id,
                retryable=False,
                suggestion="降低 cfos_unet.batch_size / cellpose.workers，或减小 patch_size 后重新提交。",
            )
    if message:
        return structured_error(
            code=ErrorCode.INTERNAL_ERROR.value,
            message=message,
            step_id=step_id,
            retryable=False,
            suggestion="查看日志尾部定位失败步骤；确认输入与配置后可重新提交。",
        )
    return structured_error(
        code=ErrorCode.INTERNAL_ERROR.value,
        message=f"main.py exited with code {returncode}.",
        step_id=step_id,
        retryable=False,
        suggestion="查看日志尾部定位失败步骤；确认输入与配置后可重新提交。",
    )


def verification_error(verification: dict[str, Any]) -> dict[str, Any]:
    missing = [row["path"] for row in verification.get("required", []) if not row["exists"]]
    unreadable = [row["path"] for row in verification.get("required", []) if row["exists"] and not row["readable"]]
    if missing:
        message = "required output is missing: " + "; ".join(missing)
    else:
        message = "required output exists but cannot be read: " + "; ".join(unreadable)
    return structured_error(
        code=ErrorCode.OUTPUT_INVALID.value,
        message=message,
        retryable=False,
        suggestion="先看日志确认该步骤是否真的执行；若是则重跑该样本，若配置只跑了部分流程请用 attach/旧入口并说明不校验输出。",
    )


def read_progress_safe(progress_file: Path) -> dict[str, Any]:
    """Half-written / corrupt progress.json must never fake a verdict."""
    return read_progress(progress_file)
