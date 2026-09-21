"""Pre-attach a not-yet-started vessel driver as a queued PM job (visibility)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, r"S:\Yifu")
import os

os.environ.setdefault("YIFU_ANALYSIS_ROOT", r"S:\Arivis_Analysis")

from pipeline_modules.harness.queue import ActiveStore
from pipeline_modules.harness.progress import empty_progress
from pipeline_modules.harness.paths import run_dir, progress_path
from pipeline_modules.harness.jsonio import utc_now_iso

sample_dir = Path(sys.argv[1])
title = sys.argv[2]

store = ActiveStore()
store.ensure_layout()

status_path = sample_dir / "vessel_pipeline.status.txt"
log_path = sample_dir / "vessel_pipeline.log"
config_path = sample_dir / "config.json"
if not status_path.exists():
    status_path.write_text("QUEUED: waiting for a free harness slot\n", encoding="utf-8")

existing = store._find_external(sample_dir)
if existing is not None:
    print("already attached:", existing["id"], existing.get("status"))
    sys.exit(0)

record = {
    "id": store._make_job_id(sample_dir),
    "kind": "external",
    "title": title,
    "sample_dir": str(sample_dir),
    "sample_name": sample_dir.name,
    "config_path": str(config_path),
    "status": "queued",
    "created_at": utc_now_iso(),
    "started_at": None,
    "ended_at": None,
    "pid": None,
    "error": None,
    "command": "ims_to_zarr -> ims L2 zarr -> zarr_to_registration_nii -> main.py(threshold+hemisphere)",
    "step_total": 5,
    "status_path": str(status_path),
    "log_path": str(log_path),
}
store.save_job(record)
ids = store._queue_ids()
ids.append(record["id"])
store._write_queue(ids)
run_dir(record["id"], store.active).mkdir(parents=True, exist_ok=True)
progress_path(record["id"], store.active).write_text(json.dumps(empty_progress()), encoding="utf-8")
view = store.sync_external_job(record)
print("pre-attached:", view["id"], "->", view["status"])
