---
name: yifu-dev
description: Development guidance for large 3D LSFM pipeline modules in the Yifu repo — volume/chunking constraints, and structured progress/timing so the pipeline harness UI can show ETA. Use when writing or changing pipeline modules, standalone CLIs, or anything that should appear in the harness monitor.
---

# Yifu development mode

## Large-volume constraints

The harness **开发** tab is a placeholder. There is no separate crop/resource executor yet.

When editing pipeline code that touches large 3D volumes:

- Prefer existing chunked Zarr paths (`dataset` `0`, block/chunk sizes already in config).
- Do not add full-volume in-memory loads when a chunked reader already exists.
- Reuse `process_existing_only` / resume / skip-if-output-exists behavior.
- Keep new work streaming or block-wise; cap workers rather than defaulting to all cores.
- For a cheap check, operate on physically present chunks or a small ROI instead of the whole brain.

Point the user at the **开发** tab in `python -m apps.pipeline_harness --host 127.0.0.1 --port 8766` if they are looking for the UI entry. Do not implement a new limiter service unless asked.

---

## Harness progress and timing (required for ETA)

The harness **does not parse terminal stdout/stderr** for progress or ETA. A module must write **structured JSON** that the harness reads from disk.

### What the UI reads

| Run type | Progress file | Who writes it |
|----------|---------------|---------------|
| `main.py` job queued by harness | `<active>/jobs/<job_id>/progress.json` | `ProgressWriter` in `pipeline_modules/harness/progress.py` (wired from `main.py` via `--progress-file`) |
| Standalone module / external process | Path passed to `harness attach --module-progress` (usually `<output>/_progress.json`) | The module itself |
| Step-level ETA history | `<active>/timing_history.json` | Harness runner after a `main.py` job finishes (`record_step_timings`) |

If a script only `print()`s progress, the monitor will show **“收集耗时”** and no ETA until enough completed runs populate `timing_history.json`.

### Path A — code running under `main.py`

Reuse `ProgressWriter`; do not invent a parallel progress format.

```python
from pipeline_modules.harness.progress import ProgressWriter

writer = ProgressWriter(progress_path, step_total=len(step_names))
writer.start_run(sample_dir=..., config_path=...)
writer.begin_step(1, "Segmentation")          # title must match planned step name
writer.set_units(done, total, phase="chunks") # optional sub-step ETA
writer.note("optional status line")
writer.finish(results=[...])                  # or writer.fail("reason")
```

Rules:

- Step titles passed to `begin_step()` must match `planned_step_names()` / `CANONICAL_STEPS` in `progress.py` (e.g. `"Segmentation"`, `"Vessel network reconstruction and region morphology"`). ETA uses the **exact string** as the key in `timing_history.json`.
- Each completed step gets `steps[].seconds` when `finish()` / `fail()` closes it. Durations **> 1 s** and not `skipped` are appended to history (median of last 20 runs).
- For sub-step ETA inside one pipeline step, call `set_units(unit_done, unit_total, phase=..., phase_started_at=...)`.

`main.py` already receives `--progress-file` from the harness runner. Submodule scripts invoked by `main.py` should accept `YIFU_PROGRESS_FILE` and forward unit progress if they do long inner loops (see VesselExpress below).

### Path B — standalone CLI or external long job

Write a module progress JSON file and register it when attaching to the harness.

**File location:** `<output_root>/_progress.json` (convention used by tubule modules).

**Minimum payload** (update periodically during long phases):

```json
{
  "phase": "Native EDT radii",
  "unit_done": 42,
  "unit_total": 100,
  "phase_started_at": "2026-03-09T12:00:00+08:00",
  "updated_at": "2026-03-09T12:05:00+08:00"
}
```

| Field | Purpose |
|-------|---------|
| `phase` | Short label shown as `step_name` in the UI |
| `unit_done` / `unit_total` | Fractional progress within the current phase; drives **unit ETA** when `unit_total > 0` |
| `phase_started_at` | ISO timestamp when the current phase began; ETA = elapsed vs `unit_done/unit_total` |
| `updated_at` | Last write time (optional but useful for debugging) |

**Write pattern:** atomic replace (temp file + `os.replace`), same as `pipeline_modules/harness/jsonio.write_json_atomic` or `vessel_express_reconstruction._write_progress`.

**Register with harness:**

```bash
python -m pipeline_modules.harness attach --sample-dir "H:\path\to\sample" --title "my-job" --pid 12345 --module-progress "H:\path\to\output\_progress.json" --step-total 7
```

Or from Python: `ActiveStore.attach_external(..., module_progress_path=...)`.

**Optional bridge:** if `YIFU_PROGRESS_FILE` is set in the environment, also mirror `unit_done` / `unit_total` / `phase` into that harness `progress.json` via `ProgressWriter.set_units()` (see `vessel_express_reconstruction._write_progress`).

### Reference implementation

Copy the pattern from `pipeline_modules/tubule_reconstruction/vessel_express_reconstruction.py`:

- Inner `report(phase, done, total)` builds the `_progress.json` payload.
- Long loops pass `progress_callback=lambda d, t: report("...", d, t)`.
- `_write_progress` atomically writes `_progress.json` and optionally syncs `YIFU_PROGRESS_FILE`.

### Checklist for new modules

1. Identify whether the job runs under `main.py` (Path A) or standalone (Path B).
2. Emit JSON progress on a fixed interval or per work unit — not only at start/end.
3. Use atomic JSON writes; never leave a half-written `progress.json`.
4. For standalone jobs, document the `_progress.json` path and use `attach --module-progress` (or watcher scripts that call `attach_external`).
5. Keep **log `print()` ASCII-safe on Windows** (`PYTHONIOENCODING=utf-8` or avoid symbols like `→`); logging failures can kill the whole job even when progress JSON is correct.

### What does *not* count

- Unstructured `print()` / `logger.info()` lines
- Ad-hoc text status files unless the harness explicitly parses them (e.g. `_ims_to_tiff_status.txt` only updates the status line, not unit ETA)
- Progress files that are never referenced in the job record (`module_progress_path` empty)
