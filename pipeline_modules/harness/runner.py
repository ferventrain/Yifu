"""Serial subprocess runner for Active queue jobs."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from pipeline_modules.harness.jsonio import utc_now_iso
from pipeline_modules.harness.paths import progress_path, run_dir, stdout_log_path
from pipeline_modules.harness.progress import read_progress
from pipeline_modules.harness.queue import ActiveStore
from pipeline_modules.harness.results import collect_existing_results, load_config
from pipeline_modules.harness.timing import record_step_timings
from pipeline_modules.harness.proc import pid_is_alive
from pipeline_modules.utils.errors import ErrorCode, PipelineError

REPO_ROOT = Path(__file__).resolve().parents[2]


class QueueRunner:
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
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen[str] | None = None
        self._current_job_id: str | None = None

    def is_busy(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            proc = self._proc
            return {
                "busy": self.is_busy(),
                "current_job_id": self._current_job_id,
                "pid": None if proc is None else proc.pid,
            }

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self.is_busy():
                return self.snapshot()
            self.store.sync_external_jobs()
            if self.store.has_running():
                return self.snapshot()
            if self.store.next_queued() is None:
                raise PipelineError(
                    ErrorCode.ARGUMENT_INVALID,
                    "Active queue is empty",
                    context={"active_dir": str(self.store.active)},
                )
            self._stop.clear()
            self._thread = threading.Thread(target=self._run_loop, name="harness-queue", daemon=True)
            self._thread.start()
        return self.snapshot()

    def maybe_start(self) -> dict[str, Any]:
        self.store.sync_external_jobs()
        if self.is_busy():
            return self.snapshot()
        running = self.store.running_job()
        if running and str(running.get("kind") or "main") == "main":
            pid = running.get("pid")
            if pid and pid_is_alive(int(pid)):
                self._attach_existing(running)
                return self.snapshot()
        if self.store.has_running() or self.store.next_queued() is None:
            return self.snapshot()
        try:
            return self.start()
        except PipelineError:
            return self.snapshot()

    def _attach_existing(self, job: dict[str, Any]) -> None:
        with self._lock:
            if self.is_busy():
                return
            self._stop.clear()
            self._current_job_id = str(job["id"])
            self._thread = threading.Thread(
                target=self._wait_existing_pid,
                args=(job,),
                name="harness-queue",
                daemon=True,
            )
            self._thread.start()

    def _wait_existing_pid(self, job: dict[str, Any]) -> None:
        job_id = str(job["id"])
        pid = int(job["pid"])
        try:
            while pid_is_alive(pid) and not self._stop.is_set():
                time.sleep(1.0)
            if self._stop.is_set():
                return
            self._complete_job(job_id, 0)
            while not self._stop.is_set():
                self.store.sync_external_jobs()
                nxt = self.store.next_queued()
                if nxt is None:
                    return
                if str(nxt.get("kind") or "main") == "external":
                    return
                self._run_job(nxt)
        finally:
            with self._lock:
                self._proc = None
                self._current_job_id = None

    def cancel_current(self) -> dict[str, Any]:
        with self._lock:
            proc = self._proc
            job_id = self._current_job_id
            self._stop.set()
            if proc is not None and proc.poll() is None:
                _terminate(proc)
        if job_id:
            self._mark_cancelled(job_id)
        return self.snapshot()

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        job = self.store.load_job(job_id)
        if str(job.get("status")) != "running":
            raise PipelineError(
                ErrorCode.ARGUMENT_INVALID,
                "job is not running",
                context={"job_id": job_id, "status": job.get("status")},
            )
        if str(job.get("kind") or "main") == "external":
            pid = job.get("pid")
            if pid:
                _kill_pid(int(pid))
            self._mark_cancelled(job_id)
            return self.snapshot()
        if self._current_job_id == job_id:
            return self.cancel_current()
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            "cannot cancel this job from the current runner",
            context={"job_id": job_id},
        )

    def _mark_cancelled(self, job_id: str) -> None:
        try:
            job = self.store.load_job(job_id)
        except PipelineError:
            return
        if job.get("status") != "running":
            return
        job["status"] = "cancelled"
        job["ended_at"] = utc_now_iso()
        job["pid"] = None
        job["error"] = "cancelled by user"
        self.store.save_job(job)
        progress = read_progress(progress_path(job_id, self.store.active))
        if progress.get("status") == "running":
            from pipeline_modules.harness.progress import ProgressWriter

            writer = ProgressWriter(progress_path(job_id, self.store.active))
            writer.state = progress
            writer.fail("cancelled by user")
            writer.state["status"] = "cancelled"
            writer._flush()

    def _run_loop(self) -> None:
        try:
            while not self._stop.is_set():
                self.store.sync_external_jobs()
                if self.store.has_running():
                    return
                job = self.store.next_queued()
                if job is None:
                    return
                if str(job.get("kind") or "main") == "external":
                    return
                self._run_job(job)
        finally:
            with self._lock:
                self._proc = None
                self._current_job_id = None

    def _run_job(self, job: dict[str, Any]) -> None:
        job_id = str(job["id"])
        sample_dir = Path(job["sample_dir"])
        config_path = Path(job["config_path"])
        run_folder = run_dir(job_id, self.store.active)
        run_folder.mkdir(parents=True, exist_ok=True)
        progress_file = progress_path(job_id, self.store.active)
        log_file = stdout_log_path(job_id, self.store.active)

        job["status"] = "running"
        job["started_at"] = utc_now_iso()
        job["ended_at"] = None
        job["error"] = None
        self.store.save_job(job)

        command = [
            self.python_exe,
            str(self.repo_root / "main.py"),
            "--config",
            str(config_path),
            "--sample_dir",
            str(sample_dir),
            "--progress_file",
            str(progress_file),
        ]
        extra_args = job.get("extra_args") or []
        if isinstance(extra_args, list):
            command.extend(str(item) for item in extra_args)
        env = os.environ.copy()
        env["YIFU_PROGRESS_FILE"] = str(progress_file)
        popen_kwargs: dict[str, Any] = {
            "args": command,
            "cwd": str(self.repo_root),
            "stdout": open(log_file, "a", encoding="utf-8"),
            "stderr": subprocess.STDOUT,
            "text": True,
            "env": env,
        }
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        log_handle = popen_kwargs["stdout"]
        try:
            proc = subprocess.Popen(**popen_kwargs)
            with self._lock:
                self._proc = proc
                self._current_job_id = job_id
            job["pid"] = proc.pid
            self.store.save_job(job)
            returncode = proc.wait()
        finally:
            log_handle.close()
            with self._lock:
                self._proc = None

        self._complete_job(job_id, returncode)

    def _complete_job(self, job_id: str, returncode: int) -> None:
        job = self.store.load_job(job_id)
        if job.get("status") == "cancelled":
            job["pid"] = None
            job["ended_at"] = job.get("ended_at") or utc_now_iso()
            self.store.save_job(job)
            return

        sample_dir = Path(job["sample_dir"])
        config_path = Path(job["config_path"])
        progress_file = progress_path(job_id, self.store.active)
        progress = read_progress(progress_file)
        job["pid"] = None
        job["ended_at"] = utc_now_iso()
        if returncode == 0 and progress.get("status") != "failed":
            job["status"] = "done"
            job["error"] = None
            if not progress.get("results"):
                try:
                    cfg = load_config(config_path)
                    from pipeline_modules.harness.progress import ProgressWriter

                    writer = ProgressWriter(progress_file)
                    writer.state = progress
                    writer.finish(collect_existing_results(sample_dir, cfg))
                    progress = writer.state
                except Exception:
                    pass
            record_step_timings(progress, active=self.store.active)
        else:
            job["status"] = "failed"
            job["error"] = progress.get("error") or f"main.py exited with code {returncode}"
            if progress.get("status") not in {"failed", "cancelled"}:
                from pipeline_modules.harness.progress import ProgressWriter

                writer = ProgressWriter(progress_file)
                writer.state = progress
                writer.fail(str(job["error"]))
        self.store.save_job(job)
        with self._lock:
            if self._current_job_id == job_id:
                self._current_job_id = None


def _kill_pid(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
        return
    try:
        os.kill(pid, 15)
    except OSError:
        pass


def _terminate(proc: subprocess.Popen[str]) -> None:
    if sys.platform == "win32" and proc.pid:
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
        try:
            proc.wait(timeout=8)
        except (subprocess.TimeoutExpired, OSError):
            pass
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=8)
            return
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    except OSError:
        pass
