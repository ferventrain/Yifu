"""Project board aggregation for the Pipeline Monitor.

A project = one sample directory; a batch = the sample's parent directory
(experiment grouping). Everything here is READ-ONLY over existing job
records/views plus a few small files in the sample dir:

* ``config*.json`` in the sample dir — the run configs the card can start;
* the newest ``_run_manifest.json`` within depth 2 — provenance summary.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from pipeline_modules.harness.jsonio import read_json, utc_now_iso, write_json_atomic
from pipeline_modules.harness.progress import planned_step_names

# ------------------------------------------------------------- archive (归档)
#
# Archive = strip the heavy intermediates from a finished sample, then move
# the folder under <root>/_archived. Deleted: zarr stores (v2 group .zgroup /
# v2 array .zarray / v3 zarr.json markers at the store root), completion
# markers (*.done.json), raw ims/tiff, and the transforms folder. Kept:
# result*/visualization* (deliverables), qc (质量检查证据, tiny), and every
# small file (config, manifests, logs).

ARCHIVE_DIRNAME = "_archived"
ARCHIVE_KEEP_DIR_PREFIXES = ("result", "visualization")
ARCHIVE_TRANSFORMS_DIRNAME = "transforms"
_ARCHIVE_ZARR_MARKERS = (".zarray", ".zgroup", "zarr.json")
_ARCHIVE_DELETE_FILE_SUFFIXES = (".ims", ".tiff", ".tif")
_ARCHIVE_DELETE_MARKER_SUFFIXES = (".done.json",)


def _is_zarr_store(path: Path) -> bool:
    return path.is_dir() and any((path / marker).exists() for marker in _ARCHIVE_ZARR_MARKERS)


def _plan_archive_deletions(sample_dir: Path) -> list[str]:
    """Relative names (dirs with a trailing /) that archive would delete."""
    planned: list[str] = []
    for entry in sorted(sample_dir.iterdir()):
        name = entry.name
        if entry.is_dir():
            if any(name.startswith(prefix) for prefix in ARCHIVE_KEEP_DIR_PREFIXES):
                continue
            if _is_zarr_store(entry) or name == ARCHIVE_TRANSFORMS_DIRNAME:
                planned.append(f"{name}/")
            continue
        low = name.lower()
        if low.endswith(_ARCHIVE_DELETE_MARKER_SUFFIXES) or low.endswith(_ARCHIVE_DELETE_FILE_SUFFIXES):
            planned.append(name)
    return planned


def archive_sample(sample_dir: Path, archive_root: Path | None = None, dry_run: bool = False) -> dict[str, Any]:
    """Delete heavy intermediates in-place, then move the sample to _archived.

    ``dry_run=True`` only reports what would be deleted; nothing is removed
    and the folder stays put (used for the confirm dialog's file list).
    """
    planned = _plan_archive_deletions(sample_dir)
    if dry_run:
        return {
            "dry_run": True,
            "would_delete": planned,
            "deleted_count": len(planned),
        }
    dest_root = (
        Path(archive_root) if archive_root is not None else sample_dir.parents[1] / ARCHIVE_DIRNAME
    )
    deleted: list[str] = []
    for entry in sorted(sample_dir.iterdir()):
        name = entry.name
        if entry.is_dir():
            if any(name.startswith(prefix) for prefix in ARCHIVE_KEEP_DIR_PREFIXES):
                continue
            if _is_zarr_store(entry) or name == ARCHIVE_TRANSFORMS_DIRNAME:
                shutil.rmtree(entry)
                deleted.append(f"{name}/")
            continue
        low = name.lower()
        if low.endswith(_ARCHIVE_DELETE_MARKER_SUFFIXES) or low.endswith(_ARCHIVE_DELETE_FILE_SUFFIXES):
            entry.unlink()
            deleted.append(name)
    dest_root.mkdir(parents=True, exist_ok=True)
    dest = dest_root / sample_dir.name
    if dest.exists():
        dest = dest_root / f"{sample_dir.name}_arch_{utc_now_iso().replace('-', '').replace(':', '').replace('T', '_')[:15]}"
    shutil.move(str(sample_dir), str(dest))
    return {
        "archived_to": str(dest),
        "deleted": deleted,
        "deleted_count": len(deleted),
    }

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
        "status": payload.get("status") or "completed",
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


# ------------------------------------------------------- board priorities (P0-P3)

BOARD_FILENAME = "pm_board.json"
BOARD_SCHEMA_VERSION = "2"
PRIORITY_LEVELS = (0, 1, 2, 3)  # P0 highest
PRIORITY_LABELS = {0: "P0", 1: "P1", 2: "P2", 3: "P3"}


def read_board_priorities(active_dir: Path) -> dict[str, int]:
    """{normalized sample_dir: level 0..3} from <active>/pm_board.json."""
    payload = read_json(active_dir / BOARD_FILENAME, default=None)
    if not isinstance(payload, dict):
        return {}
    raw = payload.get("priorities")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for sample_dir, level in raw.items():
        try:
            level_int = int(level)
        except (TypeError, ValueError):
            continue
        if level_int in PRIORITY_LEVELS:
            out[_norm_key(str(sample_dir))] = level_int
    return out


def write_board_priority(active_dir: Path, sample_dir: str, level: int | None) -> dict[str, Any]:
    """Set one sample's priority level (level=None clears it)."""
    path = active_dir / BOARD_FILENAME
    state = read_json(path, default=None) or {}
    if not isinstance(state, dict):
        state = {}
    priorities = state.get("priorities")
    if not isinstance(priorities, dict):
        priorities = {}
    # index by the original stored casing when present, else by given path
    stored_key = None
    for key in list(priorities.keys()):
        if _norm_key(str(key)) == _norm_key(sample_dir):
            stored_key = key
            break
    key = stored_key if stored_key is not None else str(sample_dir)
    if level is None:
        priorities.pop(key, None)
    else:
        priorities[key] = int(level)
    state["priorities"] = priorities
    state["schema_version"] = BOARD_SCHEMA_VERSION
    state["updated_at"] = utc_now_iso()
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


def _percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile (0..1) over positive values."""
    numbers = sorted(float(v) for v in values if v is not None and float(v) > 0)
    if not numbers:
        return None
    index = min(len(numbers) - 1, max(0, int(round(q * (len(numbers) - 1)))))
    return numbers[index]


#: pipeline profiles for time estimates — vessel vs cfos(unet/threshold) vs
#: spinal run very different stage mixes, so peers must share the profile
PROFILE_LABELS = {
    "vessel": "血管",
    "cfos-unet": "cFos UNet",
    "cfos-threshold": "cFos 阈值",
    "cfos-spotiflow": "cFos Spotiflow",
    "spinal": "脊髓",
    "generic": "通用",
}

_CONFIG_CACHE: dict[str, tuple[float, dict[str, Any] | None]] = {}


def _read_config_cached(path_value: str | None) -> dict[str, Any] | None:
    """Mtime-cached config read (partial configs resolved via layering)."""
    if not path_value:
        return None
    path = Path(path_value)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    hit = _CONFIG_CACHE.get(str(path))
    if hit and hit[0] == mtime:
        return hit[1]
    from pipeline_modules.utils.config_layering import resolve_partial_config

    _p, cfg = resolve_partial_config(path)
    _CONFIG_CACHE[str(path)] = (mtime, cfg)
    return cfg


def pipeline_profile(cfg: dict[str, Any] | None) -> str:
    if not isinstance(cfg, dict):
        return "generic"
    if bool((cfg.get("tubule_reconstruction") or {}).get("enabled")):
        return "vessel"
    if str(cfg.get("mode") or "").lower() == "spinal_cord" or bool((cfg.get("spinal_cord") or {}).get("enabled")):
        return "spinal"
    method = str(((cfg.get("segmentation") or {}).get("method")) or "").strip().lower()
    if method == "cfos_unet":
        return "cfos-unet"
    if method in {"threshold", "intensity"}:
        return "cfos-threshold"
    if method == "spotiflow":
        return "cfos-spotiflow"
    return "cfos-unet"


def _project_profile(runs: list[dict[str, Any]]) -> str:
    for run in runs:
        cfg = _read_config_cached(run.get("config_path"))
        if cfg:
            return pipeline_profile(cfg)
    return "generic"


def _succeeded_durations(runs: list[dict[str, Any]]) -> list[float]:
    return [
        float(run["duration_s"])
        for run in runs
        if run.get("duration_s")
        and str(run.get("display_status") or run.get("status")) in {"succeeded", "done"}
    ]


# ------------------------------------------------------- config-driven estimates
#
# Estimate a run from the config it would use: sum per-stage medians from run
# manifests, selecting stages by config content (segmentation method,
# postprocess on/off, skip flags). Module names were too coarse historically
# ("segmentation" covered unet AND threshold; "preprocessing" several steps),
# so fine names (segmentation.cfos_unet_inference, ...) are preferred and the
# coarse bucket is the labelled fallback until samples accumulate.


def _norm_module(name: str) -> str:
    return str(name).removeprefix("pipeline_modules.")


def _stage_median_map(root: Path, active_dir: Path) -> dict[str, float]:
    payload = collect_stage_times(root, active_dir)
    return {_norm_module(stage["module"]): float(stage["median_s"] or 0.0) for stage in payload["stages"]}


def estimate_for_config(
    cfg: dict[str, Any] | None, stage_medians: dict[str, float]
) -> dict[str, Any] | None:
    """Sum per-stage medians selected by the config. Returns seconds + basis."""
    if not isinstance(cfg, dict):
        return None
    stages: list[tuple[str, float, str]] = []  # (label, seconds, note)

    def _pick(*names: str) -> tuple[float | None, str]:
        for name in names:
            if name in stage_medians and stage_medians[name] > 0:
                return stage_medians[name], name
        return None, names[-1]

    seg = (cfg.get("segmentation") or {})
    method = str(seg.get("method") or "").strip().lower()
    post_cfg = seg.get("postprocess") or {}
    post_on = bool(post_cfg.get("enabled"))

    picked, _ = _pick("preprocessing")
    if picked:
        stages.append(("预处理", picked, ""))
    picked, _ = _pick("preprocessing.zarr_to_registration_nii")
    if picked:
        stages.append(("配准下采样", picked, ""))
    if not cfg.get("skip_registration"):
        picked, _ = _pick("registration.ANTs_registration")
        if picked:
            stages.append(("配准", picked, ""))

    if method in {"", "cfos_unet"}:
        picked, src = _pick("segmentation.cfos_unet_inference", "segmentation")
        if picked:
            stages.append(("分割（UNet）" if src.startswith("segmentation.cf") else "分割（UNet，混合口径）", picked, ""))
    elif method in {"threshold", "intensity"}:
        picked, src = _pick("segmentation.intensity_threshold", "segmentation")
        if picked:
            stages.append(("分割（阈值）" if src.startswith("segmentation.in") else "分割（阈值，混合口径）", picked, ""))
    elif method == "spotiflow":
        picked, _ = _pick("segmentation.spotiflow_inference")
        if picked:
            stages.append(("分割（Spotiflow）", picked, ""))

    if post_on:
        if "segmentation.cfos_mask_postprocess" in stage_medians:
            stages.append(("后处理", stage_medians["segmentation.cfos_mask_postprocess"], ""))
        else:
            stages.append(("后处理", 0.0, "数据积累中"))
    else:
        stages.append(("后处理", 0.0, "未开启"))

    if not cfg.get("only_tubule") and not cfg.get("only_region_analysis"):
        picked, _ = _pick("registration.region_signal_analysis_zarr_graph")
        if picked:
            stages.append(("区域统计", picked, ""))
    if (cfg.get("tubule_reconstruction") or {}).get("enabled"):
        tubule_total = sum(
            median
            for name, median in stage_medians.items()
            if name.startswith("tubule_reconstruction.")
        )
        if tubule_total:
            stages.append(("血管重建", tubule_total, ""))
    picked, _ = _pick("visualization.pipeline_qc")
    if picked:
        stages.append(("QC", picked, ""))

    known = [seconds for _label, seconds, _note in stages if seconds > 0]
    if len(known) < 2:
        return None
    seconds = sum(seconds for _label, seconds, _note in stages)
    labels = "＋".join(label for label, seconds, _note in stages if seconds > 0)
    notes = [note for _l, _s, note in stages if note]
    basis = f"按 config 估算：{labels}"
    if notes:
        basis += f"（{'，'.join(notes)}）"
    return {"seconds": round(seconds, 1), "basis": basis}


def generic_estimate(timing_history: dict[str, list[float]]) -> dict[str, Any] | None:
    """No usable config yet: one number for everyone (step-median sum)."""
    totals = [
        _median(values)
        for name, values in timing_history.items()
        if name in _CANONICAL_STEP_NAMES
    ]
    totals = [value for value in totals if value]
    if len(totals) >= 3:
        return {"seconds": round(sum(totals), 1), "basis": "统一估时（config 未定，历史步骤中位）"}
    return None


def _project_estimate(
    runs: list[dict[str, Any]],
    sample_dir: str,
    stage_medians: dict[str, float],
    timing_history: dict[str, list[float]],
) -> dict[str, Any] | None:
    """Config-driven estimate; own run history is deliberately NOT used."""
    cfg = None
    for run in runs:
        cfg = _read_config_cached(run.get("config_path"))
        if cfg:
            break
    if cfg is None:
        cfg = _read_config_cached(str(Path(sample_dir) / "config.json"))
    estimate = estimate_for_config(cfg, stage_medians)
    if estimate:
        return estimate
    return generic_estimate(timing_history)


# ------------------------------------------------------------- resource model
#
# Rough v1 (user: rules can be simple now, refined later): each task draws a
# constant footprint by pipeline profile; capacity is auto-detected.


PROFILE_COSTS = {
    "cfos-unet": {"cpu": 8, "gpu": 1},
    "cfos-threshold": {"cpu": 4, "gpu": 0},
    "cfos-spotiflow": {"cpu": 8, "gpu": 1},
    "vessel": {"cpu": 8, "gpu": 0},
    "spinal": {"cpu": 8, "gpu": 0},
    "generic": {"cpu": 8, "gpu": 0},
}

_CAPACITY_CACHE: dict[str, Any] = {"at": 0.0, "payload": None}


def detect_capacity() -> dict[str, int]:
    """CPU cores + GPU count, auto-detected (cached 5 min)."""
    import os
    import subprocess
    import time as _time

    now = _time.time()
    if _CAPACITY_CACHE["payload"] is not None and now - _CAPACITY_CACHE["at"] < 300.0:
        return _CAPACITY_CACHE["payload"]
    cpu = None
    try:
        import psutil

        cpu = psutil.cpu_count(logical=False)  # physical cores
    except Exception:
        cpu = None
    cpu = cpu or os.cpu_count() or 8
    gpus = 0
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--list-gpus"], capture_output=True, text=True, timeout=10
        )
        if proc.returncode == 0:
            gpus = len([line for line in proc.stdout.splitlines() if line.strip()])
    except (OSError, subprocess.SubprocessError):
        gpus = 0
    payload = {"cpu": int(cpu), "gpu": int(gpus)}
    _CAPACITY_CACHE["at"] = now
    _CAPACITY_CACHE["payload"] = payload
    return payload


def build_resources(views: list[dict[str, Any]]) -> dict[str, Any]:
    """Live draw + capacity + reference costs for the decision panel."""
    capacity = detect_capacity()
    tasks = []
    for view in views:
        status = str(view.get("display_status") or view.get("status"))
        if status not in {"running", "verifying", "stalled"}:
            continue
        profile = pipeline_profile(_read_config_cached(view.get("config_path")))
        cost = PROFILE_COSTS.get(profile, PROFILE_COSTS["generic"])
        eta = view.get("eta") or {}
        progress = view.get("progress") or {}
        tasks.append(
            {
                "sample": view.get("sample_name") or view.get("title"),
                "profile": profile,
                "profile_label": PROFILE_LABELS.get(profile, profile),
                "cpu": cost["cpu"],
                "gpu": cost["gpu"],
                "remaining_s": eta.get("total_remaining_s"),
                "step": progress.get("step_name") or "",
            }
        )
    used_cpu = sum(task["cpu"] for task in tasks)
    used_gpu = sum(task["gpu"] for task in tasks)
    return {
        "capacity": capacity,
        "used": {"cpu": used_cpu, "gpu": used_gpu},
        "free": {"cpu": max(capacity["cpu"] - used_cpu, 0), "gpu": max(capacity["gpu"] - used_gpu, 0)},
        "active_tasks": tasks,
        "profile_costs": [
            {
                "profile": profile,
                "label": label,
                "cpu": PROFILE_COSTS[profile]["cpu"],
                "gpu": PROFILE_COSTS[profile]["gpu"],
            }
            for profile, label in PROFILE_LABELS.items()
            if profile != "generic"
        ],
    }


# ------------------------------------------------------------- stage durations

_STAGE_CACHE: dict[str, Any] = {"at": 0.0, "payload": None}
_STAGE_CACHE_TTL_S = 300.0
_MANIFEST_SCAN_LIMIT = 600
#: single-manifest durations above this are pause/artifact noise (a hung
#: session writing its manifest days later), not real stage time.
_STAGE_MAX_DURATION_S = 3 * 86400.0


def _is_sample_dir(path: Path) -> bool:
    """A dir holding config*.json is a sample (the board's own rule)."""
    try:
        return any(path.glob("config*.json"))
    except OSError:
        return False


def _iter_batch_dirs(root: Path, active_dir: Path) -> list[Path]:
    """Dirs that may contain samples: active children, root children, root
    grandchildren, and the batch dirs inside _archived."""
    dirs: list[Path] = []

    def append_all(parent: Path) -> None:
        try:
            dirs.extend(child for child in parent.iterdir() if child.is_dir() and not child.name.startswith(("_", ".")))
        except OSError:
            return

    append_all(Path(active_dir))
    append_all(Path(root))
    for child in list(dirs):
        append_all(child)
    archived = Path(root) / ARCHIVE_DIRNAME
    append_all(archived)
    return list(dict.fromkeys(dirs))


def _iter_manifest_entries(root: Path, active_dir: Path):
    """Yield (manifest payload, sample Path) for every ``_run_manifest.json``
    under the analysis root, attributed to the real sample dir.

    Walking starts at batch-level dirs; subdirs of a non-sample batch that
    carry no configs are treated as samples too (root tool folders), and a
    manifest inside a result subdir (``<sample>/result_x/...``) is credited
    to its top-level ancestor. Bounded, read-only."""
    scanned: set[str] = set()
    for start in _iter_batch_dirs(root, active_dir):
        if len(scanned) >= _MANIFEST_SCAN_LIMIT:
            break
        if _is_sample_dir(start):
            paths = [start]
        else:
            paths = [start]
            try:
                paths.extend(
                    sub for sub in start.iterdir()
                    if sub.is_dir() and not sub.name.startswith(("_", ".")) and not _is_sample_dir(sub)
                )
            except OSError:
                pass
        for path in paths:
            for pattern in ("_run_manifest.json", "*/_run_manifest.json", "*/*/_run_manifest.json"):
                try:
                    matches = [manifest for manifest in path.glob(pattern) if manifest.is_file()]
                except OSError:
                    continue
                for manifest in matches:
                    if len(scanned) >= _MANIFEST_SCAN_LIMIT:
                        break
                    if str(manifest) in scanned:
                        continue
                    scanned.add(str(manifest))
                    payload = read_json(manifest, default=None)
                    if not isinstance(payload, dict):
                        continue
                    try:
                        top = manifest.relative_to(path).parts[0]
                        candidate = path / top
                        attributed = candidate if candidate.is_dir() else path
                    except (IndexError, OSError):
                        attributed = path
                    yield payload, attributed


def _iter_progress_steps(active_dir: Path):
    """Yield (step dict, job record) for every step recorded in
    ``<active>/runs/*/progress.json`` (worker-side memory/duration samples)."""
    runs_root = Path(active_dir) / "runs" if active_dir is not None else None
    if runs_root is None or not runs_root.is_dir():
        return
    jobs_by_id: dict[str, dict[str, Any]] = {}
    jobs_dir = Path(active_dir) / "jobs"
    if jobs_dir.is_dir():
        for job_file in jobs_dir.glob("*.json"):
            record = read_json(job_file, default=None)
            if isinstance(record, dict) and record.get("id"):
                jobs_by_id[str(record["id"])] = record
    for progress_file in sorted(runs_root.glob("*/progress.json")):
        prog = read_json(progress_file, default=None)
        if not isinstance(prog, dict):
            continue
        record = jobs_by_id.get(progress_file.parent.name, {})
        for step in prog.get("steps") or []:
            if isinstance(step, dict):
                yield step, record


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

    by_module: dict[str, list[float]] = {}
    skipped_outliers = 0
    skipped_incomplete = 0
    scanned = 0
    for payload, _sample in _iter_manifest_entries(root, active_dir):
        scanned += 1
        # legacy manifests (written only on success) have no status field;
        # anything explicitly marked incomplete is a failed/interrupted run
        if str(payload.get("status") or "").lower() == "incomplete":
            skipped_incomplete += 1
            continue
        duration = payload.get("duration_seconds")
        module = payload.get("module")
        if not module or not isinstance(duration, (int, float)) or float(duration) <= 0:
            continue
        if float(duration) > _STAGE_MAX_DURATION_S:
            skipped_outliers += 1
            continue
        by_module.setdefault(str(module), []).append(float(duration))

    # Peak memory per module: run manifests don't carry it; harness progress
    # files record it per step (worker samples the pipeline process tree).
    peaks_by_module: dict[str, list[float]] = {}
    for step, _record in _iter_progress_steps(active_dir):
        if step.get("skipped"):
            continue
        module = step.get("module")
        peak = step.get("peak_mem_mb")
        if not module or not isinstance(peak, (int, float)) or float(peak) <= 0:
            continue
        peaks_by_module.setdefault(str(module), []).append(float(peak))

    stages = []
    for module, durations in by_module.items():
        peaks = peaks_by_module.get(module) or []
        stages.append(
            {
                "module": module,
                "count": len(durations),
                "median_s": round(_median(durations) or 0.0, 1),
                "mean_s": round(sum(durations) / len(durations), 1),
                "total_s": round(sum(durations), 1),
                "peak_mem_mb_median": round(_median(peaks), 0) or None if peaks else None,
                "peak_mem_count": len(peaks),
            }
        )
    stages.sort(key=lambda item: item["median_s"], reverse=True)
    payload = {
        "stages": stages,
        "manifests_scanned": scanned,
        "outliers_skipped": skipped_outliers,
        "incomplete_skipped": skipped_incomplete,
        "generated_at": utc_now_iso(),
    }
    _STAGE_CACHE["at"] = now
    _STAGE_CACHE["payload"] = payload
    return payload


# ------------------------------------------------------------------ analytics
#
# One read-only aggregation feeding the 数据分析 charts. Job records answer
# "what ran" (status/duration/wait); run manifests answer "which stage took
# how long, where" (they also cover archived samples, whose job records are
# deleted). 5-minute cache, same policy as collect_stage_times.

_ANALYTICS_CACHE: dict[str, Any] = {"at": 0.0, "payload": None}
_HIST_BINS_S = (300.0, 1800.0, 7200.0, 21600.0)
_HIST_BIN_LABELS = ("< 5 分", "5-30 分", "30 分-2 时", "2-6 时", "≥ 6 时")


def _norm_day(value: Any) -> str | None:
    text = str(value or "").strip()
    return text[:10] if len(text) >= 10 else None


def _iso_epoch(value: Any) -> float | None:
    from datetime import datetime

    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _infer_profile_from_modules(modules: set[str]) -> str:
    if any(str(module).startswith("tubule_reconstruction") for module in modules):
        return "vessel"
    if any("spinal" in str(module).lower() for module in modules):
        return "spinal"
    if "segmentation.spotiflow_inference" in modules:
        return "cfos-spotiflow"
    if "segmentation.cfos_unet_inference" in modules or "segmentation" in modules:
        return "cfos-unet"
    if "segmentation.intensity_threshold" in modules:
        return "cfos-threshold"
    return "generic"


def collect_analytics(root: Path, active_dir: Path) -> dict[str, Any]:
    """Aggregated stats for the analysis panel. Read-only, cached 5 min."""
    import time as _time

    now = _time.time()
    if _ANALYTICS_CACHE["payload"] is not None and now - _ANALYTICS_CACHE["at"] < _STAGE_CACHE_TTL_S:
        return _ANALYTICS_CACHE["payload"]

    records: list[dict[str, Any]] = []
    jobs_dir = Path(active_dir) / "jobs"
    if jobs_dir.is_dir():
        for job_file in sorted(jobs_dir.glob("*.json")):
            record = read_json(job_file, default=None)
            if isinstance(record, dict) and record.get("id"):
                records.append(record)

    def _ended_day(record: dict[str, Any]) -> str | None:
        return _norm_day(record.get("ended_at") or record.get("created_at"))

    status_counts: dict[str, int] = {}
    daily: dict[str, dict[str, Any]] = {}
    profile_map: dict[str, dict[str, Any]] = {}
    by_sample: dict[str, dict[str, Any]] = {}
    wait_values: list[float] = []
    window_totals = {"runs": 0.0, "success_s": 0.0}
    for record in records:
        status = str(record.get("status") or "queued")
        status_counts[status] = status_counts.get(status, 0) + 1
        started = _iso_epoch(record.get("started_at"))
        ended = _iso_epoch(record.get("ended_at"))
        created = _iso_epoch(record.get("created_at"))
        duration = float(record.get("duration_s") or 0.0)
        if not duration and started and ended and ended > started:
            duration = ended - started
        if started and created and started > created:
            wait_values.append(started - created)
        day = _ended_day(record)
        if day:
            bucket = daily.setdefault(
                day,
                {"date": day, "runs": 0, "success": 0, "failed": 0, "other": 0,
                 "duration_s": 0.0, "wait_s": [], "samples": {}},
            )
            bucket["runs"] += 1
            if status in {"succeeded", "done"}:
                bucket["success"] += 1
            elif status in {"failed", "stalled"}:
                bucket["failed"] += 1
            else:
                bucket["other"] += 1
            bucket["duration_s"] += duration
            if started and created and started > created:
                bucket["wait_s"].append(started - created)
            sample_name = str(record.get("sample_name") or Path(str(record.get("sample_dir") or "")).name or "?")
            bucket["samples"][sample_name] = round(bucket["samples"].get(sample_name, 0.0) + duration / 3600.0, 2)
        profile = pipeline_profile(_read_config_cached(record.get("config_path")))
        entry = profile_map.setdefault(
            profile, {"profile": profile, "label": PROFILE_LABELS.get(profile, profile), "durations": []}
        )
        if duration > 0:
            entry["durations"].append(duration)
        sample_dir = _norm_key(str(record.get("sample_dir") or ""))
        if sample_dir:
            sample = by_sample.setdefault(
                sample_dir,
                {"sample": str(record.get("sample_name") or Path(str(record.get("sample_dir") or "")).name),
                 "batch": Path(str(record.get("sample_dir") or "")).parent.name,
                 "statuses": {}},
            )
            sample["statuses"][status] = sample["statuses"].get(status, 0) + 1

    for cutoff in (7.0, 30.0):
        days_ago = now - cutoff * 86400.0
        window_runs = [r for r in records if (_iso_epoch(r.get("ended_at")) or 0.0) >= days_ago and r.get("ended_at")]
        total = len(window_runs)
        success = sum(1 for r in window_runs if str(r.get("status")) in {"succeeded", "done"})
        window_totals[str(int(cutoff))] = {
            "runs": total,
            "success_rate": round(success / total, 3) if total else None,
            "success_s": round(
                sum(
                    float(r.get("duration_s") or 0.0)
                    or max((_iso_epoch(r.get("ended_at")) or 0.0) - (_iso_epoch(r.get("started_at")) or 0.0), 0.0)
                    for r in window_runs
                    if str(r.get("status")) in {"succeeded", "done"}
                ), 1,
            ),
        }

    stages_payload = collect_stage_times(root, active_dir)
    durations_by_module: dict[str, list[float]] = {}
    sample_entries: dict[str, dict[str, Any]] = {}
    batches: dict[str, dict[str, Any]] = {}
    for payload, sample_path in _iter_manifest_entries(root, active_dir):
        module = str(payload.get("module") or "")
        duration = payload.get("duration_seconds")
        incomplete = str(payload.get("status") or "").lower() == "incomplete"
        valid = bool(module) and isinstance(duration, (int, float)) and 0 < float(duration) <= _STAGE_MAX_DURATION_S
        sample_key = _norm_key(str(sample_path))
        entry = sample_entries.setdefault(
            sample_key,
            {"sample": sample_path.name, "batch": sample_path.parent.name,
             "modules": set(), "bin_counts": [0] * len(_HIST_BIN_LABELS), "total_s": 0.0},
        )
        batch = batches.setdefault(entry["batch"], {"batch": entry["batch"], "completed": 0, "incomplete": 0})
        if incomplete:
            batch["incomplete"] += 1
            continue
        batch["completed"] += 1
        if not valid:
            continue
        durations_by_module.setdefault(module, []).append(float(duration))
        entry["modules"].add(module)
        bin_index = sum(1 for bound in _HIST_BINS_S if float(duration) >= bound)
        entry["bin_counts"][bin_index] += 1
        entry["total_s"] += float(duration)

    stage_rows = []
    for stage in stages_payload["stages"]:
        values = durations_by_module.get(str(stage["module"])) or []
        if values:
            stage_rows.append(
                {
                    "module": stage["module"],
                    "count": len(values),
                    "median_s": round(_median(values) or 0.0, 1),
                    "mean_s": round(sum(values) / len(values), 1),
                    "total_s": round(sum(values), 1),
                    "p25_s": round(_percentile(values, 0.25) or 0.0, 1),
                    "p75_s": round(_percentile(values, 0.75) or 0.0, 1),
                    "p95_s": round(_percentile(values, 0.95) or 0.0, 1),
                    "peak_mem_mb_median": stage.get("peak_mem_mb_median"),
                }
            )

    module_names = sorted(durations_by_module.keys(), key=lambda name: -sum(durations_by_module[name]))[:12]
    heatmap: list[dict[str, Any]] = []
    for info in sample_entries.values():
        if not info["total_s"]:
            continue
        inferred = _infer_profile_from_modules(info["modules"])
        heatmap.append(
            {
                "sample": info["sample"],
                "batch": info["batch"],
                "profile": inferred,
                "profile_label": PROFILE_LABELS.get(inferred, inferred),
                "bins": list(info["bin_counts"]),
                "total_s": round(info["total_s"], 1),
            }
        )
    # most expensive samples first, keep the grid readable
    heatmap.sort(key=lambda row: row["total_s"], reverse=True)
    heatmap = heatmap[:24]

    profiles = []
    for profile, entry in profile_map.items():
        values = entry["durations"]
        if not values:
            continue
        profiles.append(
            {
                "profile": profile,
                "label": entry["label"],
                "count": len(values),
                "durations": [round(v, 1) for v in sorted(values)],
                "median_s": round(_median(values) or 0.0, 1),
                "min_s": round(min(values), 1),
                "max_s": round(max(values), 1),
            }
        )

    sample_rows = []
    for sample in by_sample.values():
        counts = sample["statuses"]
        total = sum(counts.values())
        success = counts.get("succeeded", 0) + counts.get("done", 0)
        failed = counts.get("failed", 0) + counts.get("stalled", 0)
        sample_rows.append(
            {"sample": sample["sample"], "batch": sample["batch"], "runs": total,
             "success_rate": round(success / total, 3) if total else None, "failed": failed}
        )
    batch_rows = [
        {"batch": b["batch"], "completed": b["completed"], "incomplete": b["incomplete"],
         "total": b["completed"] + b["incomplete"],
         "success_rate": round(b["completed"] / (b["completed"] + b["incomplete"]), 3) if (b["completed"] + b["incomplete"]) else None}
        for b in batches.values()
    ]
    batch_rows.sort(key=lambda row: row["total"], reverse=True)

    memory_rows = []
    for step, record in _iter_progress_steps(active_dir):
        if step.get("skipped"):
            continue
        peak = step.get("peak_mem_mb")
        if not isinstance(peak, (int, float)) or float(peak) <= 0:
            continue
        memory_rows.append(
            {
                "job": str(record.get("id") or ""),
                "sample": str(record.get("sample_name") or "?"),
                "step": str(step.get("name") or step.get("module") or "?"),
                "peak_mem_mb": round(float(peak), 0),
                "ended_at": record.get("ended_at"),
            }
        )
    memory_rows = sorted(memory_rows, key=lambda row: row["peak_mem_mb"], reverse=True)[:80]

    daily_days = sorted(daily.keys())
    daily_rows = []
    for day in daily_days:
        bucket = daily[day]
        waits = bucket.pop("wait_s")
        bucket["wait_median_s"] = round(_median(waits) or 0.0, 0) if waits else None
        bucket["duration_s"] = round(bucket["duration_s"], 1)
        bucket["samples"] = {name: hours for name, hours in bucket["samples"].items() if hours > 0}
        daily_rows.append(bucket)

    wait_median = round(_median(wait_values) or 0.0, 0) if wait_values else None

    payload = {
        "generated_at": utc_now_iso(),
        "manifests_scanned": stages_payload.get("manifests_scanned", 0),
        "jobs_scanned": len(records),
        "kpi": {
            "total_runs": len(records),
            "success_rate": window_totals["30"]["success_rate"],
            "runs_7d": window_totals["7"]["runs"],
            "success_s_7d": window_totals["7"]["success_s"],
            "runs_30d": window_totals["30"]["runs"],
            "success_s_30d": window_totals["30"]["success_s"],
            "queue_wait_median_s": wait_median,
        },
        "status_counts": status_counts,
        "capacity": detect_capacity(),
        "daily": daily_rows,
        "stages": stage_rows,
        "modules": module_names,
        "hist_bins": list(_HIST_BIN_LABELS),
        "heatmap": heatmap,
        "profiles": profiles,
        "samples": sorted(sample_rows, key=lambda row: row["runs"], reverse=True),
        "batches": batch_rows,
        "memory": memory_rows,
    }
    _ANALYTICS_CACHE["at"] = now
    _ANALYTICS_CACHE["payload"] = payload
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
    board_priorities: dict[str, int] | None = None,
    timing_history: dict[str, list[float]] | None = None,
    stage_medians: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Group decorated job views into 项目(batch) → 样本(sample) cards.

    Ordering inside a batch: P0 < P1 < P2 < P3 < 无优先级, then newest
    activity; anomalous samples (dir missing/empty) always last.
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

    priority_map = {_norm_key(k): v for k, v in (board_priorities or {}).items()}
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
        all_statuses = [str(r["display_status"] or r["status"]) for r in project["runs"]]
        # Status reflects what is live right now: any active run, else the
        # NEWEST run only. Ranking the whole history kept a sample "failed"
        # forever after one failed attempt even when a later rerun succeeded;
        # past failures stay visible via failed_count / error_history.
        live_statuses = [str(r["display_status"] or r["status"]) for r in (active or project["runs"][:1])]
        project["status"] = (
            min(live_statuses, key=lambda s: _AGGREGATE_RANK.get(s, 9)) if live_statuses else "idle"
        )
        project["failed_count"] = sum(1 for s in all_statuses if s in {"failed", "stalled"})
        project["error_history"] = [
            {"run_id": r["id"], "error": r["error"], "ended_at": r["ended_at"]}
            for r in project["runs"]
            if r["error"]
        ][:20]

        sample_path = Path(project["sample_dir"])
        project["configs"] = list_sample_configs(sample_path)
        project["last_manifest"] = latest_run_manifest_summary(sample_path)
        project["anomaly"] = _sample_anomaly(sample_path)

        batch_dir = sample_path.parent
        batch_key = _norm_key(str(batch_dir))
        batch = batches.setdefault(
            batch_key,
            {
                "batch": batch_dir.name or str(batch_dir),
                "batch_dir": str(batch_dir),
                "under_active": bool(active_key) and (batch_key == active_key or batch_key.startswith(active_key.rstrip("/") + "/")),
                # priority lives on the PROJECT (batch), not the sample
                "priority": priority_map.get(batch_key),
                "projects": [],
            },
        )
        batch["projects"].append(project)

    # profile pass + config-driven estimates (own run history deliberately
    # NOT used — different configs cost different amounts)
    for project in by_sample.values():
        profile = _project_profile(project["runs"])
        project["profile"] = profile
        project["profile_label"] = PROFILE_LABELS.get(profile, profile)
        project["estimate"] = _project_estimate(
            project["runs"], project["sample_dir"], stage_medians or {}, timing_history or {}
        )

    def _batch_sort(batch: dict[str, Any]) -> tuple[int, int, int, str]:
        priority = batch.get("priority")
        return (
            0 if batch.get("under_active") else 1,
            0 if priority is not None else 1,
            priority if priority is not None else 10**9,
            str(batch["batch"]).lower(),
        )

    # per-batch sample progress + remaining-time estimate
    def _sample_outcome(project: dict[str, Any]) -> str:
        if project.get("active_run_id"):
            return "running"
        status = str(project.get("status") or "idle")
        if status in {"succeeded", "done"}:
            return "done"
        if status in {"failed", "stalled"}:
            return "failed"
        return "waiting"  # idle / cancelled / never run

    def _remaining_for_running(project: dict[str, Any]) -> float | None:
        """Remaining time for a running sample = sum of the HISTORICAL step
        medians for every pipeline step it has NOT finished yet (steps from
        its current one onward). Deliberately ignores the dynamic ETA — that
        reflects one inner sub-step (downsample/upsample inside a module) and
        swings wildly. Returns None when the step history can't say."""
        view = project.get("latest_full_view") or {}
        progress = view.get("progress") or {}
        step_index = int(progress.get("step_index") or 0)
        steps = progress.get("steps")
        if step_index <= 0 or not isinstance(steps, list) or not steps:
            return None
        cfg = None
        for run in project["runs"]:
            cfg = _read_config_cached(run.get("config_path"))
            if cfg:
                break
        planned = planned_step_names(cfg)
        names = [str(s.get("name") or "") for s in steps if isinstance(s, dict)]
        remaining = 0.0
        known = 0
        for idx in range(step_index, len(planned) + 1):
            name = planned[idx - 1] if idx - 1 < len(planned) else (names[idx - 1] if idx - 1 < len(names) else "")
            median = _median(timing_history.get(name) or [])
            if median:
                remaining += median
                known += 1
        if known == 0:
            return None
        return remaining

    for batch in batches.values():
        counts = {"done": 0, "running": 0, "failed": 0, "waiting": 0}
        remaining = 0.0
        has_remaining = False
        for project in batch["projects"]:
            outcome = _sample_outcome(project)
            counts[outcome] += 1
            if outcome == "running":
                step_remaining = _remaining_for_running(project)
                if step_remaining is not None:
                    remaining += step_remaining
                    has_remaining = True
                    continue
            if outcome != "done":
                estimate = project.get("estimate") or {}
                if estimate.get("seconds"):
                    remaining += float(estimate["seconds"])
                    has_remaining = True
        batch["progress"] = {**counts, "total": len(batch["projects"])}
        batch["remaining_s"] = round(remaining, 1) if has_remaining else None

    batch_list = sorted(batches.values(), key=_batch_sort)
    for batch in batch_list:
        # newest activity first, anomalous samples pushed last (stable 2-pass)
        batch["projects"].sort(key=lambda p: str(p["runs"][0].get("started_at") or p["runs"][0].get("created_at") or "") if p["runs"] else "", reverse=True)
        batch["projects"].sort(key=lambda p: 1 if p.get("anomaly") else 0)
    return batch_list


def projects_payload(store: Any, views: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "batches": group_into_batches(
            views,
            active_dir=store.active,
            board_priorities=read_board_priorities(store.active),
            timing_history={},
            stage_medians=_stage_median_map(store.root, store.active),
        ),
        "resources": build_resources(views),
        "worker": None,  # filled by caller
        "analysis_root": str(store.root),
        "active_dir": str(store.active),
    }
