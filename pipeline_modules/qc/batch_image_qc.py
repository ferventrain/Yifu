from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from pipeline_modules.qc.image_qc import ImageQcConfig, run_image_qc
    from pipeline_modules.qc.ims_io import DEFAULT_IMS_CHANNEL
    from pipeline_modules.utils.errors import ErrorCode, PipelineError
except ImportError:  # pragma: no cover
    from .image_qc import ImageQcConfig, run_image_qc
    from .ims_io import DEFAULT_IMS_CHANNEL
    from ..utils.errors import ErrorCode, PipelineError


BATCH_STATE_FILENAME = "_batch_qc_state.json"
BATCH_SUMMARY_FILENAME = "batch_qc_summary.csv"
STATE_SCHEMA_VERSION = "1"


@dataclass
class QcJob:
    key: str
    ims_path: str
    channel: int
    output_json: str
    output_csv: str
    status: str = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    runtime_seconds: float | None = None
    overall_verdict: str | None = None
    error: str | None = None


@dataclass
class BatchQcState:
    schema_version: str = STATE_SCHEMA_VERSION
    scan_root: str = ""
    output_root: str = ""
    created_at: str = ""
    updated_at: str = ""
    jobs: list[QcJob] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scan_root": self.scan_root,
            "output_root": self.output_root,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "jobs": [job.__dict__ for job in self.jobs],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BatchQcState:
        jobs = [QcJob(**item) for item in payload.get("jobs", [])]
        return cls(
            schema_version=str(payload.get("schema_version", STATE_SCHEMA_VERSION)),
            scan_root=str(payload.get("scan_root", "")),
            output_root=str(payload.get("output_root", "")),
            created_at=str(payload.get("created_at", "")),
            updated_at=str(payload.get("updated_at", "")),
            jobs=jobs,
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_channels(value: str) -> list[int]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        raise PipelineError(ErrorCode.ARGUMENT_INVALID, "At least one channel is required", {"channels": value})
    return [int(part) for part in parts]


def discover_ims_files(scan_root: Path, *, recursive: bool = True) -> list[Path]:
    scan_root = scan_root.resolve()
    if not scan_root.exists():
        raise PipelineError(ErrorCode.INPUT_NOT_FOUND, "Scan root not found", {"scan_root": str(scan_root)})
    if scan_root.is_file():
        return [scan_root] if scan_root.suffix.lower() == ".ims" else []
    pattern = "**/*.ims" if recursive else "*.ims"
    files = sorted(path for path in scan_root.glob(pattern) if path.is_file())
    return files


def _job_key(ims_path: Path, channel: int) -> str:
    return f"{ims_path.resolve()}|ch{int(channel)}"


def _output_paths(
    ims_path: Path,
    *,
    scan_root: Path,
    output_root: Path,
    channel: int,
) -> tuple[Path, Path]:
    try:
        relative_parent = ims_path.parent.resolve().relative_to(scan_root.resolve())
    except ValueError:
        relative_parent = Path(ims_path.parent.name)
    job_dir = output_root / relative_parent / f"{ims_path.stem}_ch{channel}"
    return job_dir / "image_qc.json", job_dir / "image_qc_slices.csv"


def build_jobs(
    ims_files: list[Path],
    *,
    scan_root: Path,
    output_root: Path,
    channels: list[int],
) -> list[QcJob]:
    jobs: list[QcJob] = []
    for ims_path in ims_files:
        for channel in channels:
            output_json, output_csv = _output_paths(
                ims_path,
                scan_root=scan_root,
                output_root=output_root,
                channel=channel,
            )
            jobs.append(
                QcJob(
                    key=_job_key(ims_path, channel),
                    ims_path=str(ims_path.resolve()),
                    channel=int(channel),
                    output_json=str(output_json),
                    output_csv=str(output_csv),
                )
            )
    return jobs


def _is_completed_output(output_json: Path) -> bool:
    if not output_json.is_file():
        return False
    try:
        payload = json.loads(output_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and "slice_aggregate" in payload and "global_exposure_dynamic_range" in payload


def load_or_create_state(
    *,
    scan_root: Path,
    output_root: Path,
    channels: list[int],
    recursive: bool,
) -> BatchQcState:
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / BATCH_STATE_FILENAME
    ims_files = discover_ims_files(scan_root, recursive=recursive)
    fresh_jobs = build_jobs(ims_files, scan_root=scan_root, output_root=output_root, channels=channels)
    fresh_by_key = {job.key: job for job in fresh_jobs}

    if state_path.exists():
        try:
            existing = BatchQcState.from_dict(json.loads(state_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError):
            existing = BatchQcState()
        preserved = {job.key: job for job in existing.jobs if job.key in fresh_by_key}
        merged_jobs: list[QcJob] = []
        for job in fresh_jobs:
            if job.key in preserved:
                old = preserved[job.key]
                old.output_json = job.output_json
                old.output_csv = job.output_csv
                if old.status == "running":
                    old.status = "pending"
                if old.status == "completed" and not _is_completed_output(Path(old.output_json)):
                    old.status = "pending"
                merged_jobs.append(old)
            else:
                merged_jobs.append(job)
        state = BatchQcState(
            scan_root=str(scan_root.resolve()),
            output_root=str(output_root.resolve()),
            created_at=existing.created_at or _utc_now(),
            updated_at=_utc_now(),
            jobs=merged_jobs,
        )
    else:
        state = BatchQcState(
            scan_root=str(scan_root.resolve()),
            output_root=str(output_root.resolve()),
            created_at=_utc_now(),
            updated_at=_utc_now(),
            jobs=fresh_jobs,
        )

    save_batch_state(state_path, state)
    return state


def save_batch_state(state_path: Path, state: BatchQcState) -> None:
    state.updated_at = _utc_now()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = state_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(state.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(state_path)


def _write_summary_csv(output_root: Path, jobs: list[QcJob]) -> Path:
    summary_path = output_root / BATCH_SUMMARY_FILENAME
    fieldnames = [
        "ims_path",
        "channel",
        "status",
        "overall_verdict",
        "runtime_seconds",
        "output_json",
        "output_csv",
        "error",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for job in jobs:
            writer.writerow(
                {
                    "ims_path": job.ims_path,
                    "channel": job.channel,
                    "status": job.status,
                    "overall_verdict": job.overall_verdict or "",
                    "runtime_seconds": job.runtime_seconds if job.runtime_seconds is not None else "",
                    "output_json": job.output_json,
                    "output_csv": job.output_csv,
                    "error": job.error or "",
                }
            )
    return summary_path


def run_batch_image_qc(
    *,
    scan_root: str | Path,
    output_root: str | Path,
    channels: list[int] | None = None,
    recursive: bool = True,
    force: bool = False,
    config: ImageQcConfig | None = None,
) -> dict[str, Any]:
    scan_path = Path(scan_root)
    output_path = Path(output_root)
    channel_list = channels or [DEFAULT_IMS_CHANNEL]
    cfg = config or ImageQcConfig(nas_qc=True, show_progress=True)

    state_path = output_path / BATCH_STATE_FILENAME
    state = load_or_create_state(
        scan_root=scan_path,
        output_root=output_path,
        channels=channel_list,
        recursive=recursive,
    )

    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        tqdm = None

    iterator: Any = state.jobs
    if tqdm is not None and cfg.show_progress:
        iterator = tqdm(state.jobs, desc="Batch IMS QC", unit="job", file=sys.stderr)

    completed = 0
    failed = 0
    skipped = 0

    for job in iterator:
        if job.status == "completed" and not force and _is_completed_output(Path(job.output_json)):
            skipped += 1
            continue

        job.status = "running"
        job.started_at = _utc_now()
        job.error = None
        save_batch_state(state_path, state)

        try:
            results = run_image_qc(
                input_ims=job.ims_path,
                output_json=job.output_json,
                output_csv=job.output_csv,
                sample_id=f"{Path(job.ims_path).stem}_ch{job.channel}",
                config=cfg,
            )
            job.status = "completed"
            job.runtime_seconds = float(results.get("runtime_seconds", 0.0))
            grading = results.get("grading") or {}
            job.overall_verdict = str(grading.get("overall_verdict") or "")
            completed += 1
        except Exception as exc:
            job.status = "failed"
            job.error = repr(exc)
            failed += 1
        finally:
            job.finished_at = _utc_now()
            save_batch_state(state_path, state)

    summary_path = _write_summary_csv(output_path, state.jobs)
    return {
        "scan_root": str(scan_path.resolve()),
        "output_root": str(output_path.resolve()),
        "state_path": str(state_path.resolve()),
        "summary_csv": str(summary_path.resolve()),
        "total_jobs": len(state.jobs),
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "pending_remaining": sum(1 for job in state.jobs if job.status != "completed"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch NAS IMS image QC with resume support and per-sample qc/ outputs.",
    )
    parser.add_argument("--root", default="Z:\\", help="NAS scan root, e.g. Z:\\")
    parser.add_argument("--output_root", default="Z:\\qc", help="Batch QC output root, default Z:\\qc")
    parser.add_argument("--channels", default="0", help="Comma-separated IMS channels, e.g. 0 or 0,1,2")
    parser.add_argument("--no_recursive", action="store_true", help="Only scan --root itself, not subfolders")
    parser.add_argument("--force", action="store_true", help="Re-run jobs even if outputs already exist")
    parser.add_argument("--quiet", action="store_true", help="Disable per-job progress bars")
    parser.add_argument("--ims_resolution_level", type=int, default=None)
    parser.add_argument("--max_slices", type=int, default=None)
    parser.add_argument("--ims_histogram_z_chunks", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ImageQcConfig(nas_qc=True, show_progress=not bool(args.quiet))
    if args.ims_resolution_level is not None:
        config.ims_resolution_level = int(args.ims_resolution_level)
    if args.max_slices is not None:
        config.max_slices = int(args.max_slices)
    if args.ims_histogram_z_chunks is not None:
        config.ims_histogram_z_chunks = int(args.ims_histogram_z_chunks)

    try:
        summary = run_batch_image_qc(
            scan_root=args.root,
            output_root=args.output_root,
            channels=_parse_channels(args.channels),
            recursive=not bool(args.no_recursive),
            force=bool(args.force),
            config=config,
        )
    except PipelineError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False), file=sys.stderr)
        return exc.exit_code

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["failed"] == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
