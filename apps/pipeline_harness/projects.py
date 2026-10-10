"""Project board aggregation for the Pipeline Monitor.

A project = one sample directory; a batch = the sample's parent directory
(experiment grouping). Everything here is READ-ONLY over existing job
records/views plus a few small files in the sample dir:

* ``<sample>/pm_project.json`` — human notes (requirements / retro / log),
  edited via POST /api/projects/notes (the only writer);
* ``config*.json`` in the sample dir — the run configs the card can start;
* the newest ``_run_manifest.json`` within depth 2 — provenance summary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline_modules.harness.jsonio import read_json, utc_now_iso, write_json_atomic

NOTES_FILENAME = "pm_project.json"
NOTES_SCHEMA_VERSION = "1"

_AGGREGATE_RANK = {
    "running": 0,
    "verifying": 0,
    "stalled": 1,
    "queued": 2,
    "failed": 3,
    "cancelled": 4,
    "succeeded": 5,
    "done": 5,
}

#: canonical pipeline step names (progress.py BRAIN_STEPS + SPINAL steps) used
#: to bound the timing-history based estimate to real pipeline stages only.
_CANONICAL_STEP_NAMES = {
    "Registration channel downsample",
    "Atlas registration and label outputs",
    "Signal preprocessing and Zarr conversion",
    "Segmentation",
    "Region density analysis",
    "Vessel network reconstruction and region morphology",
    "Spinal cord per-vertebra signal analysis",
    "Spotiflow signal count summary",
}

_MANIFEST_CANDIDATE_LIMIT = 60


def _norm_key(path_value: str) -> str:
    return str(path_value).replace("\\", "/").rstrip("/").lower()


def read_project_notes(sample_dir: Path) -> dict[str, Any] | None:
    payload = read_json(sample_dir / NOTES_FILENAME, default=None)
    return payload if isinstance(payload, dict) else None


def write_project_notes(
    sample_dir: Path,
    *,
    requirements: str | None = None,
    retro: str | None = None,
    append_note: str | None = None,
) -> dict[str, Any]:
    """Create/update ``<sample>/pm_project.json`` (the notes writer)."""
    path = sample_dir / NOTES_FILENAME
    state = read_json(path, default=None) or {}
    if not isinstance(state, dict):
        state = {}
    if requirements is not None:
        state["requirements"] = str(requirements)
    if retro is not None:
        state["retro"] = str(retro)
    if append_note:
        entry = {"at": utc_now_iso(), "text": str(append_note)}
        state.setdefault("log", []).append(entry)
    state["schema_version"] = NOTES_SCHEMA_VERSION
    state["updated_at"] = utc_now_iso()
    write_json_atomic(path, state)
    return state


def list_sample_configs(sample_dir: Path) -> list[str]:
    """config*.json names available for one-click start (config.json, config_1.json, ...)."""
    try:
        names = sorted(p.name for p in sample_dir.glob("config*.json") if p.is_file())
    except OSError:
        return []
    # config.json first, then config_1, config_2 ...
    def rank(name: str) -> tuple[int, str]:
        if name == "config.json":
            return (0, "")
        return (1, name)

    return sorted(names, key=rank)


def latest_run_manifest_summary(sample_dir: Path) -> dict[str, Any] | None:
    """Newest _run_manifest.json within depth 2, as a compact summary."""
    try:
        candidates = list(sample_dir.glob("_run_manifest.json"))
        candidates.extend(sample_dir.glob("*/_run_manifest.json"))
    except OSError:
        return None
    candidates = [p for p in candidates if p.is_file()][:_MANIFEST_CANDIDATE_LIMIT]
    if not candidates:
        return None
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    payload = read_json(newest, default=None)
    if not isinstance(payload, dict):
        return None
    outputs = payload.get("outputs") or []
    code = payload.get("code") or {}
    return {
        "path": str(newest),
        "module": payload.get("module"),
        "entrypoint": payload.get("entrypoint"),
        "ended_at": payload.get("ended_at"),
        "duration_seconds": payload.get("duration_seconds"),
        "git_commit": code.get("git_commit"),
        "git_dirty": code.get("git_dirty"),
        "outputs": [
            {"path": item.get("path"), "type": item.get("type")}
            for item in outputs
            if isinstance(item, dict)
        ][:12],
    }


def _error_summary(error: Any) -> str | None:
    if not error:
        return None
    if isinstance(error, dict):
        code = str(error.get("code") or "")
        message = str(error.get("message") or "")
        return f"[{code}] {message}" if code else message or None
    return str(error)[:300]


def _run_row(view: dict[str, Any]) -> dict[str, Any]:
    config_path = str(view.get("config_path") or "")
    code_version = view.get("code_version") or {}
    return {
        "id": view.get("id"),
        "title": view.get("title") or view.get("sample_name"),
        "kind": view.get("kind"),
        "status": view.get("status"),
        "display_status": view.get("display_status") or view.get("status"),
        "config_name": Path(config_path).name if config_path else "",
        "config_path": config_path or None,
        "created_at": view.get("created_at"),
        "started_at": view.get("started_at"),
        "ended_at": view.get("ended_at"),
        "duration_s": view.get("duration_s"),
        "error": _error_summary(view.get("error")),
        "git_commit": (code_version.get("git_commit") or "")[:12] or None,
        "qc_images": view.get("qc_images") or [],
        "results": view.get("results") or [],
        "progress": view.get("progress") or {},
        "eta": view.get("eta"),
        "log_path": view.get("log_path"),
        "worker_alive": view.get("worker_alive"),
        "heartbeat_age_s": view.get("heartbeat_age_s"),
        "current_step_duration_s": view.get("current_step_duration_s"),
        # full decorated view for the expanded rich card (local UI only)
        "full_view": view,
    }


def _project_sort_key(project: dict[str, Any]) -> tuple[int, str]:
    newest = project["runs"][0] if project["runs"] else {}
    stamp = newest.get("started_at") or newest.get("created_at") or ""
    return (0, str(stamp))


# ------------------------------------------------------- board order (priority)

BOARD_FILENAME = "pm_board.json"
BOARD_SCHEMA_VERSION = "1"


def read_board_order(active_dir: Path) -> list[str]:
    payload = read_json(active_dir / BOARD_FILENAME, default=None)
    if not isinstance(payload, dict):
        return []
    order = payload.get("sample_order")
    return [str(item) for item in order if isinstance(item, str)] if isinstance(order, list) else []


def write_board_order(active_dir: Path, ordered: list[str]) -> dict[str, Any]:
    path = active_dir / BOARD_FILENAME
    state = {
        "schema_version": BOARD_SCHEMA_VERSION,
        "sample_order": [str(item) for item in ordered],
        "updated_at": utc_now_iso(),
    }
    write_json_atomic(path, state)
    return state


# ---------------------------------------------------------------- estimations


def _median(values: list[float]) -> float | None:
    numbers = sorted(float(v) for v in values if v is not None and float(v) > 0)
    if not numbers:
        return None
    mid = len(numbers) // 2
    if len(numbers) % 2:
        return numbers[mid]
    return (numbers[mid - 1] + numbers[mid]) / 2.0


def estimate_project_seconds(
    project: dict[str, Any], timing_history: dict[str, list[float]]
) -> dict[str, Any] | None:
    """Rough estimate of a full pipeline run for this sample.

    Basis 1: the active dir's timing history (median seconds per pipeline
    step, recorded by the worker after each succeeded run) — summed over the
    canonical steps. Basis 2: the sample's own succeeded-run average. Honest
    null when neither exists.
    """
    if timing_history:
        totals = [
            _median(values)
            for name, values in timing_history.items()
            if name in _CANONICAL_STEP_NAMES
        ]
        totals = [value for value in totals if value]
        if len(totals) >= 3:
            return {"seconds": round(sum(totals), 1), "basis": "历史中位（全部步骤）"}
    durations = [
        float(run["duration_s"])
        for run in project.get("runs", [])
        if run.get("duration_s") and str(run.get("display_status") or run.get("status")) in {"succeeded", "done"}
    ]
    if durations:
        return {"seconds": round(sum(durations) / len(durations), 1), "basis": f"该样本 {len(durations)} 次完成 run 均值"}
    return None


# ------------------------------------------------------------- stage durations

_STAGE_CACHE: dict[str, Any] = {"at": 0.0, "payload": None}
_STAGE_CACHE_TTL_S = 300.0
_MANIFEST_SCAN_LIMIT = 600
#: single-manifest durations above this are pause/artifact noise (a hung
#: session writing its manifest days later), not real stage time.
_STAGE_MAX_DURATION_S = 3 * 86400.0


def collect_stage_times(root: Path, active_dir: Path) -> dict[str, Any]:
    """Aggregate per-module durations from run manifests (active + archive).

    Scans sample dirs under the analysis root (depth 2) and the active dir,
    reading ``_run_manifest.json`` at depth 0-2 inside each sample. Bounded
    and cached; read-only.
    """
    import time as _time

    now = _time.time()
    if _STAGE_CACHE["payload"] is not None and now - _STAGE_CACHE["at"] < _STAGE_CACHE_TTL_S:
        return _STAGE_CACHE["payload"]

    sample_dirs: list[Path] = []
    for parent in (root, active_dir):
        try:
            for child in parent.iterdir():
                if not child.is_dir() or child.name.startswith("_") or child.name.startswith("."):
                    continue
                sample_dirs.append(child)
                try:
                    sample_dirs.extend(grand for grand in child.iterdir() if grand.is_dir() and not grand.name.startswith(("_", ".")))
                except OSError:
                    continue
        except OSError:
            continue
    sample_dirs = list(dict.fromkeys(sample_dirs))

    manifests: list[Path] = []
    for sample in sample_dirs:
        if len(manifests) >= _MANIFEST_SCAN_LIMIT:
            break
        for pattern in ("_run_manifest.json", "*/_run_manifest.json", "*/*/_run_manifest.json"):
            try:
                manifests.extend(manifest for manifest in sample.glob(pattern) if manifest.is_file())
            except OSError:
                continue

    by_module: dict[str, list[float]] = {}
    skipped_outliers = 0
    for manifest in manifests[:_MANIFEST_SCAN_LIMIT]:
        payload = read_json(manifest, default=None)
        if not isinstance(payload, dict):
            continue
        duration = payload.get("duration_seconds")
        module = payload.get("module")
        if not module or not isinstance(duration, (int, float)) or float(duration) <= 0:
            continue
        if float(duration) > _STAGE_MAX_DURATION_S:
            skipped_outliers += 1
            continue
        by_module.setdefault(str(module), []).append(float(duration))

    stages = []
    for module, durations in by_module.items():
        stages.append(
            {
                "module": module,
                "count": len(durations),
                "median_s": round(_median(durations) or 0.0, 1),
                "mean_s": round(sum(durations) / len(durations), 1),
                "total_s": round(sum(durations), 1),
            }
        )
    stages.sort(key=lambda item: item["median_s"], reverse=True)
    payload = {
        "stages": stages,
        "manifests_scanned": min(len(manifests), _MANIFEST_SCAN_LIMIT),
        "outliers_skipped": skipped_outliers,
        "generated_at": utc_now_iso(),
    }
    _STAGE_CACHE["at"] = now
    _STAGE_CACHE["payload"] = payload
    return payload


# ------------------------------------------------------------------ grouping


def _sample_anomaly(sample_path: Path) -> str | None:
    """Why a sample dir looks broken (missing / no files under it)."""
    try:
        if not sample_path.exists():
            return "样本目录不存在"
        if not any(sample_path.iterdir()):
            return "样本目录为空"
    except OSError:
        return "样本目录不可读"
    return None


def group_into_batches(
    views: list[dict[str, Any]],
    *,
    active_dir: str | Path | None = None,
    board_order: list[str] | None = None,
    timing_history: dict[str, list[float]] | None = None,
) -> list[dict[str, Any]]:
    """Group decorated job views into 项目(batch) → 样本(sample) cards.

    Ordering inside a batch: user priority (board_order index) first, then
    newest activity; anomalous samples (dir missing/empty) always last.
    ``under_active`` marks batches living under the active dir so the UI can
    show them first by default.
    """
    by_sample: dict[str, dict[str, Any]] = {}
    for view in views:
        sample_dir = str(view.get("sample_dir") or "").strip()
        if not sample_dir:
            continue
        key = _norm_key(sample_dir)
        project = by_sample.setdefault(
            key,
            {
                "sample_dir": sample_dir,
                "sample_name": view.get("sample_name") or Path(sample_dir).name,
                "runs": [],
            },
        )
        row = _run_row(view)
        row["full_view"] = view  # stripped below
        project["runs"].append(row)

    order_rank = {_norm_key(item): index for index, item in enumerate(board_order or [])}
    active_key = _norm_key(str(active_dir)) if active_dir else None

    batches: dict[str, dict[str, Any]] = {}
    for project in by_sample.values():
        # newest run first (started_at falls back to created_at)
        project["runs"].sort(
            key=lambda run: str(run.get("started_at") or run.get("created_at") or ""),
            reverse=True,
        )
        # keep the newest run's full view for the expanded card; drop the rest
        latest_view = None
        for index, run in enumerate(project["runs"]):
            full = run.pop("full_view", None)
            if index == 0:
                latest_view = full
        project["latest_full_view"] = latest_view

        active = [r for r in project["runs"] if str(r["status"]) in {"queued", "running", "verifying"}]
        project["active_run_id"] = active[0]["id"] if active else None
        statuses = [str(r["display_status"] or r["status"]) for r in project["runs"]]
        project["status"] = (
            min(statuses, key=lambda s: _AGGREGATE_RANK.get(s, 9)) if statuses else "idle"
        )
        project["failed_count"] = sum(1 for s in statuses if s in {"failed", "stalled"})
        project["error_history"] = [
            {"run_id": r["id"], "error": r["error"], "ended_at": r["ended_at"]}
            for r in project["runs"]
            if r["error"]
        ][:20]

        sample_path = Path(project["sample_dir"])
        project["notes"] = read_project_notes(sample_path)
        project["configs"] = list_sample_configs(sample_path)
        project["last_manifest"] = latest_run_manifest_summary(sample_path)
        project["anomaly"] = _sample_anomaly(sample_path)
        project["estimate"] = estimate_project_seconds(project, timing_history or {})
        project["priority_rank"] = order_rank.get(_norm_key(project["sample_dir"]))

        batch_dir = sample_path.parent
        batch_key = _norm_key(str(batch_dir))
        batch = batches.setdefault(
            batch_key,
            {
                "batch": batch_dir.name or str(batch_dir),
                "batch_dir": str(batch_dir),
                "under_active": bool(active_key) and (batch_key == active_key or batch_key.startswith(active_key.rstrip("/") + "/")),
                "projects": [],
            },
        )
        batch["projects"].append(project)

    def _batch_sort(batch: dict[str, Any]) -> tuple[int, str]:
        return (0 if batch.get("under_active") else 1, str(batch["batch"]).lower())

    def _project_rank(project: dict[str, Any]) -> tuple[int, int, int, str]:
        priority = project.get("priority_rank")
        anomaly = 1 if project.get("anomaly") else 0
        return (
            anomaly,
            0 if priority is not None else 1,
            priority if priority is not None else 10**9,
            str(project["sample_name"]).lower(),
        )

    batch_list = sorted(batches.values(), key=_batch_sort)
    for batch in batch_list:
        batch["projects"].sort(key=_project_rank)
    return batch_list


def projects_payload(store: Any, views: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "batches": group_into_batches(views),
        "worker": None,  # filled by caller
        "analysis_root": str(store.root),
        "active_dir": str(store.active),
    }
