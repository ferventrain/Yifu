"""The one detached worker process that executes harness jobs.

Lifecycle: ``python -m pipeline_modules.harness run`` (or a POST to the
monitor) validates a RunSpec, appends a job record, then calls
:func:`ensure_worker_running`. That spawns ONE detached ``worker`` process
(if none is alive) which owns the queue from then on:

* claims queued ``kind=main`` jobs serially (never two at once),
* launches the standard pipeline command (Runner -> main.py -> modules),
* stamps ``heartbeat_at`` into progress.json while the pipeline runs,
* honours cooperative cancel flags,
* verifies required outputs and writes the artifact manifest,
* adopts orphaned still-alive pipelines after a worker crash, and marks
  dead ones ``stalled`` instead of guessing success.

The worker holds an exclusive lock file for its whole life, so a second
worker can never start and races on job files are impossible.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from pipeline_modules.harness.jsonio import read_json, utc_now_iso, write_json_atomic
from pipeline_modules.harness.paths import (
    cancel_flag_path,
    progress_path,
    run_dir,
    stdout_log_path,
    worker_lock_path,
    worker_log_path,
    worker_state_path,
)
from pipeline_modules.harness.proc import pid_created_at, pid_is_alive, pid_matches
from pipeline_modules.harness.progress import (
    _elapsed_seconds,
    read_progress,
    touch_heartbeat,
)
from pipeline_modules.harness.queue import ActiveStore
from pipeline_modules.harness.results import load_config
from pipeline_modules.harness.timing import record_step_timings
from pipeline_modules.harness.verify import (
    classify_failure,
    read_progress_safe,
    structured_error,
    verification_error,
    write_artifact_manifest,
)
from pipeline_modules.utils.errors import ErrorCode, PipelineError

REPO_ROOT = Path(__file__).resolve().parents[2]

HEARTBEAT_INTERVAL_S = 15.0
CANCEL_POLL_S = 2.0
IDLE_POLL_S = 5.0
#: Job heartbeat older than this (while marked running) is displayed as stalled.
STALE_HEARTBEAT_S = 120.0
#: An idle worker exits after this long so no zombie lingers; the next
#: submit/start simply spawns a fresh one.
WORKER_IDLE_EXIT_S = 1800.0
LOG_TAIL_BYTES = 8192


class HarnessWorker:
    """Serial queue worker; exactly one instance per machine (lock-guarded)."""

    def __init__(
        self,
        store: ActiveStore,
        *,
        python_exe: str | None = None,
        repo_root: Path | None = None,
    ) -> None:
        self.store = store
        self.python_exe = python_exe or sys.executable
        self.repo_root = Path(repo_root or REPO_ROOT)
        self._stop = threading.Event()
        self._lock_handle: Any = None
        self._started_at: str | None = None

    # ------------------------------------------------------------------ lock

    def _acquire_lock(self) -> bool:
        path = worker_lock_path(self.store.active)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path, "a+")
        try:
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except ImportError:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                handle.close()
                return False
        except OSError:
            handle.close()
            return False
        self._lock_handle = handle
        return True

    # ----------------------------------------------------------------- state

    def _write_state(self, current_job_id: str | None) -> None:
        self._started_at = self._started_at or utc_now_iso()
        write_json_atomic(
            worker_state_path(self.store.active),
            {
                "pid": os.getpid(),
                "pid_created_at": pid_created_at(os.getpid()),
                "started_at": self._started_at,
                "heartbeat_at": utc_now_iso(),
                "current_job_id": current_job_id,
                "stopped_at": None,
            },
        )

    # ------------------------------------------------------------------ loop

    def run_forever(self) -> int:
        if not self._acquire_lock():
            print('{"worker": "already-running"}')
            return 0
        self._write_state(None)
        try:
            self._adopt_orphans()
            idle_since = time.monotonic()
            while not self._stop.is_set():
                try:
                    self.store.sync_external_jobs()
                    if self.store.has_running():
                        self._write_state(None)
                        time.sleep(IDLE_POLL_S)
                        idle_since = time.monotonic()
                        continue
                    job = self._next_queued_main()
                    if job is None:
                        if time.monotonic() - idle_since >= WORKER_IDLE_EXIT_S:
                            return 0
                        self._write_state(None)
                        time.sleep(IDLE_POLL_S)
                        continue
                    idle_since = time.monotonic()
                    self.run_job(job)
                except Exception as exc:  # keep the worker alive no matter what
                    print(f"[worker] loop error: {exc!r}", flush=True)
                    time.sleep(IDLE_POLL_S)
            return 0
        finally:
            state = read_json(worker_state_path(self.store.active), default={}) or {}
            state["stopped_at"] = utc_now_iso()
            state["current_job_id"] = None
            try:
                write_json_atomic(worker_state_path(self.store.active), state)
            except OSError:
                pass
            if self._lock_handle is not None:
                self._lock_handle.close()
                self._lock_handle = None

    def _next_queued_main(self) -> dict[str, Any] | None:
        for job in self.store.list_jobs():
            if job.get("status") == "queued" and str(job.get("kind") or "main") == "main":
                return job
        return None

    # --------------------------------------------------------------- orphans

    def _adopt_orphans(self) -> None:
        """Reconcile jobs left 'running' by a dead previous worker.

        A still-alive pipeline pid is adopted (waited out, then finalized);
        a dead pid can only be marked ``stalled`` — never succeeded.
        """
        for job in self.store.list_jobs():
            if str(job.get("kind") or "main") != "main":
                continue
            if str(job.get("status")) not in {"running", "verifying"}:
                continue
            job_id = str(job["id"])
            pid = job.get("pid")
            if pid and pid_matches(int(pid), job.get("pid_created_at")):
                print(f"[worker] adopting orphan pid {pid} for job {job_id}", flush=True)
                self._write_state(job_id)
                returncode = _wait_pid_exit_code(int(pid))
                self._finalize(job_id, returncode)
            else:
                self._mark_stalled(job_id)

    def _mark_stalled(self, job_id: str) -> None:
        try:
            job = self.store.load_job(job_id)
        except PipelineError:
            return
        if str(job.get("status")) not in {"running", "verifying"}:
            return
        progress = read_progress_safe(progress_path(job_id, self.store.active))
        job["status"] = "stalled"
        job["ended_at"] = utc_now_iso()
        job["pid"] = None
        job["error"] = structured_error(
            code="STALLED",
            message="worker process disappeared; cannot confirm whether the run finished.",
            step_id=progress.get("step_index"),
            retryable=True,
            suggestion="检查产物是否完整后重新提交该任务（pipeline 内部会跳过已完成步骤）。",
        )
        self.store.save_job(job)

    # --------------------------------------------------------------- run job

    def run_job(self, job: dict[str, Any]) -> None:
        job_id = str(job["id"])
        run_dir(job_id, self.store.active).mkdir(parents=True, exist_ok=True)
        progress_file = progress_path(job_id, self.store.active)
        log_file = stdout_log_path(job_id, self.store.active)
        cancel_flag_path(job_id, self.store.active).unlink(missing_ok=True)

        job["status"] = "running"
        job["started_at"] = utc_now_iso()
        job["ended_at"] = None
        job["error"] = None
        job["cancel_requested"] = False
        self.store.save_job(job)
        touch_heartbeat(progress_path(job_id, self.store.active))

        command = [
            self.python_exe,
            str(self.repo_root / "main.py"),
            "--config",
            str(job["config_path"]),
            "--sample_dir",
            str(job["sample_dir"]),
            "--progress_file",
            str(progress_file),
        ]
        extras = job.get("extra_args") or []
        if isinstance(extras, list):
            command.extend(str(item) for item in extras)
        env = os.environ.copy()
        env["YIFU_PROGRESS_FILE"] = str(progress_file)
        popen_kwargs: dict[str, Any] = {
            "args": command,
            "cwd": str(self.repo_root),
            "stdout": open(log_file, "a", encoding="utf-8"),
            "stderr": subprocess.STDOUT,
            "text": True,
            "env": env,
            "stdin": subprocess.DEVNULL,
        }
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        log_handle = popen_kwargs["stdout"]
        try:
            proc = subprocess.Popen(**popen_kwargs)
            job["pid"] = proc.pid
            job["pid_created_at"] = pid_created_at(proc.pid)
            self.store.save_job(job)
            self._write_state(job_id)
            returncode = self._supervise(proc, job_id, progress_file)
        finally:
            log_handle.close()
        self._finalize(job_id, returncode)

    def _supervise(self, proc: subprocess.Popen[str], job_id: str, progress_file: Path) -> int:
        """Wait for the pipeline while stamping heartbeats and honouring cancel."""
        last_heartbeat = 0.0
        while proc.poll() is None:
            if cancel_flag_path(job_id, self.store.active).exists():
                _terminate(proc)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
                return proc.wait()
            now = time.monotonic()
            if now - last_heartbeat >= HEARTBEAT_INTERVAL_S:
                touch_heartbeat(progress_file)
                self._write_state(job_id)
                last_heartbeat = now
            time.sleep(CANCEL_POLL_S)
        return proc.wait()

    # --------------------------------------------------------------- finalize

    def _finalize(self, job_id: str, returncode: int | None) -> None:
        try:
            job = self.store.load_job(job_id)
        except PipelineError:
            return
        if str(job.get("status")) == "cancelled":
            cancel_flag_path(job_id, self.store.active).unlink(missing_ok=True)
            return

        progress_file = progress_path(job_id, self.store.active)
        progress = read_progress_safe(progress_file)
        job["status"] = "verifying"
        job["pid"] = None
        self.store.save_job(job)

        cancelled = cancel_flag_path(job_id, self.store.active).exists() or job.get("cancel_requested")
        ended_at = utc_now_iso()
        job["ended_at"] = ended_at
        job["duration_s"] = _elapsed_seconds(job.get("started_at"), ended_at)
        if cancelled:
            job["status"] = "cancelled"
            job["error"] = structured_error(
                code="CANCELLED",
                message="cancelled by user",
                step_id=progress.get("step_index"),
                retryable=True,
                suggestion="需要时可重新提交该任务。",
            )
            cancel_flag_path(job_id, self.store.active).unlink(missing_ok=True)
            self.store.save_job(job)
            return

        try:
            cfg = load_config(job["config_path"])
            manifest = write_artifact_manifest(
                job_id,
                self.store.active,
                job["sample_dir"],
                cfg,
                job.get("extra_args") or [],
            )
            verification = manifest["verification"]
        except Exception as exc:
            verification = {"ok": False, "required": [], "error": repr(exc)}

        exited_ok = returncode is None or returncode == 0
        if exited_ok and verification.get("ok"):
            job["status"] = "succeeded"
            job["error"] = None
            record_step_timings(progress, active=self.store.active)
        else:
            job["status"] = "failed"
            if not exited_ok:
                job["error"] = classify_failure(
                    int(returncode or 1),
                    _log_tail(stdout_log_path(job_id, self.store.active)),
                    progress,
                )
            else:
                job["error"] = verification_error(verification)
        self.store.save_job(job)


# ----------------------------------------------------------------------
# Worker state helpers shared by the CLI and the (read-only) monitor.
# ----------------------------------------------------------------------


def read_worker_state(active: Path | None = None) -> dict[str, Any] | None:
    payload = read_json(worker_state_path(active), default=None)
    return payload if isinstance(payload, dict) else None


def worker_is_alive(active: Path | None = None) -> bool:
    state = read_worker_state(active)
    if not state or state.get("stopped_at"):
        return False
    return pid_matches(state.get("pid"), state.get("pid_created_at"))


def worker_snapshot(active: Path | None = None) -> dict[str, Any]:
    state = read_worker_state(active) or {}
    alive = worker_is_alive(active)
    return {
        "alive": alive,
        "pid": state.get("pid"),
        "current_job_id": state.get("current_job_id") if alive else None,
        "started_at": state.get("started_at"),
        "heartbeat_at": state.get("heartbeat_at") if alive else None,
    }


def ensure_worker_running(store: ActiveStore, *, python_exe: str | None = None) -> dict[str, Any]:
    """Start the detached worker if (and only if) none is alive."""
    if worker_is_alive(store.active):
        return {"started": False, "worker": worker_snapshot(store.active)}
    log_file = worker_log_path(store.active)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    popen_kwargs: dict[str, Any] = {
        "args": [python_exe or sys.executable, "-m", "pipeline_modules.harness", "worker"],
        "cwd": str(REPO_ROOT),
        "stdout": open(log_file, "a", encoding="utf-8"),
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        # Detached from any console/job so closing the submitting terminal
        # (or the monitor) never kills the run; output goes to worker.log.
        popen_kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_BREAKAWAY_FROM_JOB
        )
    else:
        popen_kwargs["start_new_session"] = True
    log_handle = popen_kwargs["stdout"]
    try:
        subprocess.Popen(**popen_kwargs)
    finally:
        log_handle.close()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if worker_is_alive(store.active):
            break
        time.sleep(0.2)
    return {"started": True, "worker": worker_snapshot(store.active)}


def request_cancel(store: ActiveStore, job_id: str) -> dict[str, Any]:
    """Cooperative cancel: signal the worker; it kills the pipeline tree."""
    job = store.load_job(job_id)
    status = str(job.get("status"))
    if status == "queued":
        job["status"] = "cancelled"
        job["ended_at"] = utc_now_iso()
        job["error"] = structured_error(code="CANCELLED", message="cancelled by user", retryable=True)
        store.save_job(job)
        return {"job_id": job_id, "status": "cancelled"}
    if status in {"running", "verifying", "stalled"}:
        job["cancel_requested"] = True
        store.save_job(job)
        cancel_flag_path(job_id, store.active).write_text("cancel\n", encoding="utf-8")
        pid_alive = bool(job.get("pid")) and pid_is_alive(int(job["pid"]))
        if not worker_is_alive(store.active) and not pid_alive:
            job["status"] = "cancelled"
            job["ended_at"] = utc_now_iso()
            job["error"] = structured_error(code="CANCELLED", message="cancelled by user", retryable=True)
            store.save_job(job)
            return {"job_id": job_id, "status": "cancelled"}
        return {"job_id": job_id, "status": "cancel_requested"}
    raise PipelineError(
        ErrorCode.ARGUMENT_INVALID,
        "job is not cancellable in its current state",
        context={"job_id": job_id, "status": status},
    )


def decorate_views(views: list[dict[str, Any]], active: Path | None = None) -> list[dict[str, Any]]:
    """Add runtime flags (display status, heartbeat age) WITHOUT writing anything."""
    alive = worker_is_alive(active)
    now = utc_now_iso()
    for view in views:
        status = str(view.get("status"))
        if status == "done":
            status = "succeeded"
        display = status
        progress = view.get("progress") or {}
        heartbeat_age = _elapsed_seconds(progress.get("heartbeat_at"), now)
        if status in {"running", "verifying"}:
            if heartbeat_age is None or heartbeat_age > STALE_HEARTBEAT_S or not alive:
                display = "stalled"
        view["display_status"] = display
        view["worker_alive"] = alive
        view["heartbeat_age_s"] = None if heartbeat_age is None else round(heartbeat_age, 1)
        started = view.get("started_at")
        ended = view.get("ended_at")
        duration = _elapsed_seconds(started, ended or now)
        view["duration_s"] = None if duration is None else round(duration, 1)
        step_duration = _elapsed_seconds(progress.get("step_started_at"), ended or now) if status == "running" else None
        view["current_step_duration_s"] = None if step_duration is None else round(step_duration, 1)
    return views


def _wait_pid_exit_code(pid: int, *, timeout_s: float = 604800.0) -> int | None:
    """Exit code of a non-child process (Windows psutil/ctypes); None if unknown."""
    try:
        import psutil

        return int(psutil.Process(pid).wait(timeout=timeout_s))
    except Exception:
        pass
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        SYNCHRONIZE = 0x00100000
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        INFINITE = 0xFFFFFFFF
        handle = kernel32.OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return None
        try:
            if kernel32.WaitForSingleObject(handle, INFINITE) != 0:
                return None
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return None
            return int(code.value)
        finally:
            kernel32.CloseHandle(handle)
    return None


def _log_tail(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - LOG_TAIL_BYTES))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _terminate(proc: subprocess.Popen[str]) -> None:
    if sys.platform == "win32" and proc.pid:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=False)
        return
    proc.terminate()
