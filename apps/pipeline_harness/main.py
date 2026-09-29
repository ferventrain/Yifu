"""Pipeline Monitor (PM): READ-ONLY local watch UI for the LSFM pipeline queue.

Opening this page, refreshing it, or polling GET /api/jobs NEVER starts a
task. Jobs are only ever started by:

* submitting a task (``python -m pipeline_modules.harness run`` or POST /api/jobs),
* explicitly starting the worker (POST /api/queue/start),
* the worker itself advancing the queue,
* cancelling a task (POST /api/jobs/{id}/cancel).

Everything the page shows comes from job records, progress.json, stdout.log
and artifacts.json. No log parsing, no fabricated percentages.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles as StarletteStaticFiles

from pipeline_modules.harness.paths import open_path_in_file_manager, stdout_log_path
from pipeline_modules.harness.queue import ActiveStore, is_under
from pipeline_modules.harness.worker import (
    decorate_views,
    ensure_worker_running,
    request_cancel,
    worker_snapshot,
)
from pipeline_modules.utils.errors import ErrorCode, PipelineError

STATIC_DIR = APP_DIR / "static"
ASSET_VERSION = "5"

def _resolve_pipeline_python() -> str | None:
    for key in ("YIFU_PYTHON", "YIFU_PYTHON_EXE"):
        candidate = os.environ.get(key, "").strip()
        if candidate and Path(candidate).is_file():
            return candidate
    return None


store = ActiveStore()


class NoCacheStaticFiles(StarletteStaticFiles):
    async def get_response(self, path: str, scope):  # type: ignore[override]
        response = await super().get_response(path, scope)
        if path.endswith((".js", ".css", ".html", ".map")):
            response.headers["Cache-Control"] = "no-cache"
        return response


app = FastAPI(title="Yifu Pipeline Monitor (PM)", version="2.0")


class AddJobBody(BaseModel):
    sample_dir: str = Field(..., min_length=1)
    config_path: str = ""


class OpenPathBody(BaseModel):
    path: str = Field(..., min_length=1)


@app.exception_handler(PipelineError)
async def pipeline_error_handler(_request, exc: PipelineError):
    status = 404 if exc.code == ErrorCode.INPUT_NOT_FOUND else 400
    return JSONResponse(status_code=status, content=exc.to_dict())


def _jobs_payload() -> dict[str, Any]:
    """Assemble the job view. Read-only: never spawns, never mutates."""
    return {
        "jobs": decorate_views(store.list_job_views(), store.active),
        "worker": worker_snapshot(store.active),
        "analysis_root": str(store.root),
        "active_dir": str(store.active),
    }


@app.get("/api/meta")
def api_meta() -> dict[str, Any]:
    store.ensure_layout()
    return {
        "analysis_root": str(store.root),
        "active_dir": str(store.active),
        "worker": worker_snapshot(store.active),
        "asset_version": ASSET_VERSION,
    }


@app.get("/api/jobs")
def api_jobs() -> dict[str, Any]:
    return _jobs_payload()


@app.post("/api/jobs")
def api_add_job(body: AddJobBody) -> dict[str, Any]:
    record = store.add_job(body.sample_dir, body.config_path or None)
    worker = ensure_worker_running(store, python_exe=_resolve_pipeline_python())
    return {"job": decorate_views([store.job_view(record)], store.active)[0], "worker": worker["worker"]}


@app.delete("/api/jobs/{job_id}")
def api_remove_job(job_id: str) -> dict[str, str]:
    store.remove_job(job_id)
    return {"status": "removed", "id": job_id}


@app.post("/api/queue/start")
def api_start_queue() -> dict[str, Any]:
    """Explicit worker start (an allowed state change, unlike GET polling)."""
    ensure_worker_running(store, python_exe=_resolve_pipeline_python())
    return _jobs_payload()


@app.post("/api/queue/cancel")
def api_cancel_current() -> dict[str, Any]:
    running = store.running_job()
    if running is None:
        return {**_jobs_payload(), "cancelled": None}
    request_cancel(store, str(running["id"]))
    return {**_jobs_payload(), "cancelled": running["id"]}


@app.post("/api/jobs/{job_id}/cancel")
def api_cancel_job(job_id: str) -> dict[str, Any]:
    request_cancel(store, job_id)
    return _jobs_payload()


@app.get("/api/jobs/{job_id}/log")
def api_job_log(job_id: str, tail: int = Query(200, ge=1, le=2000)) -> dict[str, Any]:
    job = store.load_job(job_id)
    path = Path(job.get("log_path") or stdout_log_path(job_id, store.active))
    if not path.exists():
        return {"job_id": job_id, "text": "", "path": str(path)}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"job_id": job_id, "text": "\n".join(lines[-tail:]), "path": str(path)}


@app.get("/api/jobs/{job_id}/artifacts")
def api_job_artifacts(job_id: str) -> dict[str, Any]:
    """Artifact manifest written by the worker's verifier (read-only)."""
    from pipeline_modules.harness.paths import artifacts_path

    job = store.load_job(job_id)
    path = artifacts_path(job_id, store.active)
    if not path.exists():
        return {"job_id": job_id, "artifacts": None, "path": str(path)}
    try:
        payload = path.read_text(encoding="utf-8")
        import json as _json

        return {"job_id": job_id, **_json.loads(payload)}
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/open")
def api_open_path(body: OpenPathBody) -> dict[str, str]:
    raw = str(body.path).strip().strip('"').strip("'")
    target = Path(raw).expanduser()
    try:
        target = target.resolve()
    except OSError as exc:
        raise PipelineError(ErrorCode.ARGUMENT_INVALID, str(exc)) from exc
    if not (is_under(target, store.root) or is_under(target, store.active)):
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            f"路径不在分析根目录内: {target}",
            context={"path": str(target), "root": str(store.root)},
        )
    if not target.exists():
        raise PipelineError(
            ErrorCode.INPUT_NOT_FOUND,
            f"路径不存在: {target}",
            context={"path": str(target)},
        )
    try:
        folder = open_path_in_file_manager(target)
    except OSError as exc:
        raise PipelineError(ErrorCode.INTERNAL_ERROR, f"无法打开目录: {exc}") from exc
    return {"opened": str(folder)}


app.mount("/", NoCacheStaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LSFM pipeline harness operator UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--reload", action="store_true")
    return parser


def main() -> int:
    import uvicorn

    args = build_parser().parse_args()
    store.ensure_layout()
    uvicorn.run("apps.pipeline_harness.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
