"""Local operator UI for the LSFM pipeline harness."""

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
from pipeline_modules.harness.runner import QueueRunner
from pipeline_modules.utils.errors import ErrorCode, PipelineError

STATIC_DIR = APP_DIR / "static"
ASSET_VERSION = "4"

def _resolve_pipeline_python() -> str | None:
    for key in ("YIFU_PYTHON", "YIFU_PYTHON_EXE"):
        candidate = os.environ.get(key, "").strip()
        if candidate and Path(candidate).is_file():
            return candidate
    return None


store = ActiveStore()
runner = QueueRunner(store, python_exe=_resolve_pipeline_python())


class NoCacheStaticFiles(StarletteStaticFiles):
    async def get_response(self, path: str, scope):  # type: ignore[override]
        response = await super().get_response(path, scope)
        if path.endswith((".js", ".css", ".html", ".map")):
            response.headers["Cache-Control"] = "no-cache"
        return response


app = FastAPI(title="Yifu Pipeline Harness", version="1.0")


class AddJobBody(BaseModel):
    sample_dir: str = Field(..., min_length=1)
    config_path: str = ""


class OpenPathBody(BaseModel):
    path: str = Field(..., min_length=1)


@app.exception_handler(PipelineError)
async def pipeline_error_handler(_request, exc: PipelineError):
    status = 404 if exc.code == ErrorCode.INPUT_NOT_FOUND else 400
    return JSONResponse(status_code=status, content=exc.to_dict())


@app.get("/api/meta")
def api_meta() -> dict[str, Any]:
    store.ensure_layout()
    return {
        "analysis_root": str(store.root),
        "active_dir": str(store.active),
        "runner": runner.snapshot(),
        "asset_version": ASSET_VERSION,
    }


@app.get("/api/jobs")
def api_jobs() -> dict[str, Any]:
    runner.maybe_start()
    return {
        "jobs": store.list_job_views(),
        "runner": runner.snapshot(),
        "analysis_root": str(store.root),
        "active_dir": str(store.active),
    }


@app.post("/api/jobs")
def api_add_job(body: AddJobBody) -> dict[str, Any]:
    record = store.add_job(body.sample_dir, body.config_path or None)
    runner.maybe_start()
    return {"job": store.job_view(record), "runner": runner.snapshot()}


@app.delete("/api/jobs/{job_id}")
def api_remove_job(job_id: str) -> dict[str, str]:
    store.remove_job(job_id)
    return {"status": "removed", "id": job_id}


@app.post("/api/queue/start")
def api_start_queue() -> dict[str, Any]:
    return {"runner": runner.maybe_start(), "jobs": store.list_job_views()}


@app.post("/api/queue/cancel")
def api_cancel_current() -> dict[str, Any]:
    snapshot = runner.cancel_current()
    return {"runner": snapshot, "jobs": store.list_job_views()}


@app.post("/api/jobs/{job_id}/cancel")
def api_cancel_job(job_id: str) -> dict[str, Any]:
    snapshot = runner.cancel_job(job_id)
    return {"runner": snapshot, "jobs": store.list_job_views()}


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
