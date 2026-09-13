"""File-backed Active job queue under YIFU_ACTIVE_DIR."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline_modules.harness.jsonio import read_json, utc_now_iso, write_json_atomic
from pipeline_modules.harness.paths import (
    active_dir,
    analysis_root,
    job_file,
    jobs_dir,
    progress_path,
    queue_path,
    run_dir,
    stdout_log_path,
)
from pipeline_modules.harness.progress import empty_progress, estimate_eta, estimate_unit_eta, planned_step_names, read_progress
from pipeline_modules.harness.proc import pid_is_alive
from pipeline_modules.harness.results import load_config
from pipeline_modules.harness.timing import load_timing_history
from pipeline_modules.utils.errors import ErrorCode, PipelineError

JOB_STATUSES = ("queued", "running", "done", "failed", "cancelled")
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")
_STEP_RE = re.compile(r"Step\s+(\d+)", re.IGNORECASE)
RESERVED_ACTIVE_NAMES = {"jobs", "runs"}
VESSEL_STEP_TOTAL = 7


def _norm(path: Path) -> Path:
    return path.expanduser().resolve()


def is_under(path: Path, root: Path) -> bool:
    try:
        _norm(path).relative_to(_norm(root))
        return True
    except (ValueError, OSError):
        child = str(_norm(path)).replace("/", "\\").rstrip("\\").lower()
        parent = str(_norm(root)).replace("/", "\\").rstrip("\\").lower()
        return child == parent or child.startswith(parent + "\\")


class ActiveStore:
    def __init__(self, *, root: Path | None = None, active: Path | None = None) -> None:
        self.root = _norm(root or analysis_root())
        self.active = _norm(active or active_dir(root=self.root))

    def ensure_layout(self) -> None:
        jobs_dir(self.active).mkdir(parents=True, exist_ok=True)
        (self.active / "runs").mkdir(parents=True, exist_ok=True)
        if not queue_path(self.active).exists():
            write_json_atomic(queue_path(self.active), {"ids": []})

    def _queue_ids(self) -> list[str]:
        payload = read_json(queue_path(self.active), default={"ids": []}) or {}
        ids = payload.get("ids") if isinstance(payload, dict) else None
        if not isinstance(ids, list):
            return []
        return [str(item) for item in ids]

    def _write_queue(self, ids: list[str]) -> None:
        write_json_atomic(queue_path(self.active), {"ids": ids})

    def validate_sample_dir(self, sample_dir: str | Path) -> Path:
        sample = _norm(Path(sample_dir))
        if not sample.exists() or not sample.is_dir():
            raise PipelineError(
                ErrorCode.INPUT_NOT_FOUND,
                "sample_dir does not exist",
                context={"sample_dir": str(sample)},
            )
        if not is_under(sample, self.root) or sample == self.root:
            raise PipelineError(
                ErrorCode.ARGUMENT_INVALID,
                f"sample_dir must be a subdirectory of {self.root}",
                context={"sample_dir": str(sample), "analysis_root": str(self.root)},
            )
        if self._is_reserved_active_path(sample):
            raise PipelineError(
                ErrorCode.ARGUMENT_INVALID,
                "sample_dir cannot be the harness jobs/runs bookkeeping folder",
                context={"sample_dir": str(sample), "active_dir": str(self.active)},
            )
        return sample

    def _is_reserved_active_path(self, sample: Path) -> bool:
        if sample == self.active:
            return True
        jobs = self.active / "jobs"
        runs = self.active / "runs"
        return is_under(sample, jobs) or is_under(sample, runs) or sample in {jobs, runs}

    def validate_config(self, config_path: str | Path) -> tuple[Path, dict[str, Any]]:
        path = _norm(Path(config_path))
        if not path.is_file():
            raise PipelineError(
                ErrorCode.INPUT_NOT_FOUND,
                "config.json not found",
                context={"config_path": str(path)},
            )
        try:
            cfg = load_config(path)
        except (OSError, json.JSONDecodeError) as exc:
            raise PipelineError(
                ErrorCode.CONFIG_INVALID,
                "config.json is not valid JSON",
                context={"config_path": str(path), "error": repr(exc)},
            ) from exc
        if not isinstance(cfg, dict) or "input" not in cfg:
            raise PipelineError(
                ErrorCode.CONFIG_INVALID,
                "config.json is missing an input section",
                context={"config_path": str(path)},
            )
        return path, cfg

    def _make_job_id(self, sample_dir: Path) -> str:
        slug = _SLUG_RE.sub("-", sample_dir.name).strip("-") or "sample"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        candidate = f"{stamp}-{slug}"
        if not job_file(candidate, self.active).exists():
            return candidate
        return f"{candidate}-{uuid.uuid4().hex[:6]}"

    def add_job(
        self,
        sample_dir: str | Path,
        config_path: str | Path | None = None,
        extra_args: list[str] | None = None,
    ) -> dict[str, Any]:
        self.ensure_layout()
        sample = self.validate_sample_dir(sample_dir)
        config_file = Path(config_path) if config_path else sample / "config.json"
        config_file, cfg = self.validate_config(config_file)
        job_id = self._make_job_id(sample)
        record = {
            "id": job_id,
            "kind": "main",
            "title": sample.name,
            "sample_dir": str(sample),
            "sample_name": sample.name,
            "config_path": str(config_file),
            "status": "queued",
            "created_at": utc_now_iso(),
            "started_at": None,
            "ended_at": None,
            "pid": None,
            "error": None,
            "project_name": cfg.get("project_name"),
            "extra_args": [str(item) for item in (extra_args or [])],
        }
        write_json_atomic(job_file(job_id, self.active), record)
        ids = self._queue_ids()
        ids.append(job_id)
        self._write_queue(ids)
        run_dir(job_id, self.active).mkdir(parents=True, exist_ok=True)
        write_json_atomic(progress_path(job_id, self.active), empty_progress())
        return record

    def attach_external(
        self,
        sample_dir: str | Path,
        *,
        title: str,
        pid: int | None = None,
        log_path: str | Path | None = None,
        status_path: str | Path | None = None,
        config_path: str | Path | None = None,
        module_progress_path: str | Path | None = None,
        command: str = "",
        step_total: int = VESSEL_STEP_TOTAL,
    ) -> dict[str, Any]:
        """Track an already-running pipeline that was started outside the harness."""
        self.ensure_layout()
        sample = self.validate_sample_dir(sample_dir)
        config_file = ""
        project_name = None
        if config_path:
            resolved, cfg = self.validate_config(config_path)
            config_file = str(resolved)
            project_name = cfg.get("project_name")
        existing = self._find_external(sample)
        if existing is not None:
            record = existing
            job_id = str(record["id"])
        else:
            job_id = self._make_job_id(sample)
            record = {
                "id": job_id,
                "kind": "external",
                "created_at": utc_now_iso(),
            }
            ids = self._queue_ids()
            ids.append(job_id)
            self._write_queue(ids)
            run_dir(job_id, self.active).mkdir(parents=True, exist_ok=True)
        record.update(
            {
                "kind": "external",
                "title": title,
                "sample_dir": str(sample),
                "sample_name": sample.name,
                "config_path": config_file or record.get("config_path") or "",
                "project_name": project_name or record.get("project_name"),
                "pid": int(pid) if pid else record.get("pid"),
                "log_path": str(Path(log_path)) if log_path else record.get("log_path"),
                "status_path": str(Path(status_path)) if status_path else record.get("status_path"),
                "module_progress_path": (
                    str(Path(module_progress_path))
                    if module_progress_path
                    else record.get("module_progress_path")
                ),
                "command": command or record.get("command") or "",
                "step_total": int(step_total),
                "error": None,
                "ended_at": None,
            }
        )
        if record.get("started_at") is None or record.get("status") in {"failed", "cancelled", "done"}:
            record["started_at"] = utc_now_iso()
        self.save_job(record)
        return self.sync_external_job(record)

    def _find_external(self, sample_dir: Path) -> dict[str, Any] | None:
        target = str(_norm(sample_dir)).replace("/", "\\").lower()
        for job in self.list_jobs():
            if str(job.get("kind") or "") != "external":
                continue
            if str(job.get("sample_dir") or "").replace("/", "\\").lower() == target:
                return job
        return None

    def has_running(self) -> bool:
        return any(str(job.get("status")) == "running" for job in self.list_jobs())

    def sync_external_jobs(self) -> None:
        for job in self.list_jobs():
            if str(job.get("kind") or "") == "external":
                self.sync_external_job(job)

    def sync_external_job(self, job: dict[str, Any]) -> dict[str, Any]:
        parsed = _parse_status_file(job.get("status_path"))
        alive = pid_is_alive(job.get("pid"))
        first = parsed.get("first_line") or ""
        previous = str(job.get("status") or "running")
        if first.startswith("ALL DONE"):
            job["status"] = "done"
            job["error"] = None
            job["ended_at"] = job.get("ended_at") or utc_now_iso()
            job["step_name"] = "全部完成"
        elif first.startswith("CANCELLED"):
            job["status"] = "cancelled"
            job["error"] = first
            job["ended_at"] = job.get("ended_at") or utc_now_iso()
            job["step_name"] = first
        elif first.startswith("FAILED"):
            job["status"] = "failed"
            job["error"] = first
            job["ended_at"] = job.get("ended_at") or utc_now_iso()
            job["step_name"] = first
        elif alive:
            job["status"] = "running"
            job["error"] = None
            job["ended_at"] = None
            job["step_name"] = first.replace("RUNNING", "", 1).strip() or first or "运行中"
            if parsed.get("started_at"):
                job["started_at"] = parsed["started_at"]
        elif first.startswith("RUNNING"):
            job["status"] = "failed"
            job["error"] = "process exited while still marked RUNNING"
            job["ended_at"] = utc_now_iso()
            job["step_name"] = first
        elif previous in {"done", "cancelled"}:
            return job
        elif previous == "queued":
            job["status"] = "queued"
        else:
            job["status"] = "failed"
            job["error"] = first or "process exited without ALL DONE"
            job["ended_at"] = utc_now_iso()
            job["step_name"] = job["error"]
        if parsed.get("step_index"):
            job["step_index"] = parsed["step_index"]
        job["status_line"] = first
        module_prog = _read_module_progress(job.get("module_progress_path"))
        if module_prog:
            job["module_progress"] = module_prog
            phase = str(module_prog.get("phase") or "").strip()
            done = int(module_prog.get("unit_done") or 0)
            total = int(module_prog.get("unit_total") or 0)
            if phase and total > 0 and job.get("status") == "running":
                job["step_name"] = f"{phase} {done}/{total}"
        self.save_job(job)
        return job

    def load_job(self, job_id: str) -> dict[str, Any]:
        payload = read_json(job_file(job_id, self.active), default=None)
        if not isinstance(payload, dict):
            raise PipelineError(
                ErrorCode.INPUT_NOT_FOUND,
                "job not found",
                context={"job_id": job_id},
            )
        return payload

    def save_job(self, record: dict[str, Any]) -> dict[str, Any]:
        job_id = str(record["id"])
        write_json_atomic(job_file(job_id, self.active), record)
        return record

    def list_jobs(self) -> list[dict[str, Any]]:
        self.ensure_layout()
        jobs: list[dict[str, Any]] = []
        for path in sorted(jobs_dir(self.active).glob("*.json")):
            payload = read_json(path, default=None)
            if isinstance(payload, dict) and payload.get("id"):
                jobs.append(payload)
        order = {job_id: index for index, job_id in enumerate(self._queue_ids())}

        def sort_key(job: dict[str, Any]) -> tuple[int, int, str]:
            status = str(job.get("status") or "")
            rank = {"running": 0, "queued": 1, "failed": 2, "cancelled": 3, "done": 4}.get(status, 5)
            queued_index = order.get(str(job.get("id")), 10_000)
            return (rank, queued_index, str(job.get("created_at") or ""))

        jobs.sort(key=sort_key)
        return jobs

    def next_queued(self) -> dict[str, Any] | None:
        by_id = {str(job.get("id")): job for job in self.list_jobs()}
        for job_id in self._queue_ids():
            job = by_id.get(job_id)
            if job and job.get("status") == "queued":
                return job
        for job in self.list_jobs():
            if job.get("status") == "queued":
                return job
        return None

    def running_job(self) -> dict[str, Any] | None:
        for job in self.list_jobs():
            if job.get("status") == "running":
                return job
        return None

    def remove_job(self, job_id: str) -> None:
        job = self.load_job(job_id)
        if job.get("status") == "running":
            raise PipelineError(
                ErrorCode.ARGUMENT_INVALID,
                "cannot remove a running job; cancel it first",
                context={"job_id": job_id},
            )
        job_path = job_file(job_id, self.active)
        if job_path.exists():
            job_path.unlink()
        ids = [item for item in self._queue_ids() if item != job_id]
        self._write_queue(ids)

    def job_view(self, job: dict[str, Any]) -> dict[str, Any]:
        if str(job.get("kind") or "") == "external":
            job = self.sync_external_job(job)
        job_id = str(job["id"])
        log_file = job.get("log_path") or str(stdout_log_path(job_id, self.active))
        if str(job.get("kind") or "") == "external":
            module_prog = job.get("module_progress") or _read_module_progress(job.get("module_progress_path"))
            unit_done = int((module_prog or {}).get("unit_done") or 0)
            unit_total = int((module_prog or {}).get("unit_total") or 0)
            phase = str((module_prog or {}).get("phase") or "")
            progress = {
                "status": job.get("status") or "running",
                "step_index": int(job.get("step_index") or 0),
                "step_total": int(job.get("step_total") or VESSEL_STEP_TOTAL),
                "step_name": job.get("step_name") or job.get("status_line") or "",
                "message": job.get("status_line") or "",
                "error": job.get("error"),
                "results": None,
                "unit_done": unit_done,
                "unit_total": unit_total,
                "unit_phase": phase,
            }
            if job.get("status") == "running" and unit_total > 0:
                eta = estimate_unit_eta(
                    unit_done=unit_done,
                    unit_total=unit_total,
                    phase_started_at=(module_prog or {}).get("phase_started_at") or job.get("started_at"),
                )
            else:
                eta = {
                    "current_step_remaining_s": None,
                    "total_remaining_s": None,
                    "current_step_elapsed_s": None,
                    "source": "collecting",
                    "collecting": True,
                    "unit_done": unit_done,
                    "unit_total": unit_total,
                }
            view = dict(job)
            view["progress"] = progress
            view["eta"] = eta
            view["log_path"] = str(log_file)
            view["progress_path"] = str(job.get("module_progress_path") or progress_path(job_id, self.active))
            return view

        progress = read_progress(progress_path(job_id, self.active))
        cfg: dict[str, Any] | None = None
        try:
            if job.get("config_path"):
                cfg = load_config(job["config_path"])
        except Exception:
            cfg = None
        unit_total = int(progress.get("unit_total") or 0)
        if progress.get("status") == "running" and unit_total > 0:
            eta = estimate_unit_eta(
                unit_done=int(progress.get("unit_done") or 0),
                unit_total=unit_total,
                phase_started_at=progress.get("phase_started_at") or progress.get("step_started_at"),
            )
        else:
            eta = estimate_eta(
                progress,
                load_timing_history(self.active),
                planned_names=planned_step_names(cfg),
            )
        view = dict(job)
        view["progress"] = progress
        view["eta"] = eta
        view["log_path"] = str(log_file)
        view["progress_path"] = str(progress_path(job_id, self.active))
        if progress.get("results"):
            view["results"] = progress["results"]
        return view

    def list_job_views(self) -> list[dict[str, Any]]:
        self.sync_external_jobs()
        return [self.job_view(job) for job in self.list_jobs()]


def _read_module_progress(path_value: str | None) -> dict[str, Any] | None:
    if not path_value:
        return None
    payload = read_json(Path(path_value), default=None)
    return payload if isinstance(payload, dict) else None


def _parse_status_file(path_value: str | None) -> dict[str, Any]:
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.is_file():
        return {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return {}
    if not text:
        return {}
    lines = text.splitlines()
    first = lines[0].strip()
    started_at = lines[1].strip() if len(lines) > 1 else None
    match = _STEP_RE.search(first)
    return {
        "first_line": first,
        "started_at": started_at,
        "step_index": int(match.group(1)) if match else None,
    }
