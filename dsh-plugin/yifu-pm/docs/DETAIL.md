# Yifu Pipeline Monitor tools

Brings the Yifu LSFM pipeline's Pipeline Monitor (PM) into DeepSeek Harness as
five agent tools. The Python harness remains the engine and the single owner of
state; this bundle only calls it.

## Architecture

- Engine: `pipeline_modules/harness` (file-backed queue, one detached serial
  worker, `progress.json` heartbeats, output verification, artifact manifest).
- Adapter: this bundle. Every read goes through
  `python -m pipeline_modules.harness list|preflight` or through the JSON files
  under `YIFU_ACTIVE_DIR`; submissions and cancellations go through the same
  CLI humans use, so validation, sample dedupe and structured errors are
  identical for agents and operators.
- The old web PM (`http://127.0.0.1:8766`) can keep running in parallel while
  this bundle is evaluated.

## Tools

| Tool | Kind | Purpose |
|---|---|---|
| `yifu_preflight` | read-only | Validate sample dir, config, pipeline mode, required outputs, active run and worker before submitting |
| `yifu_submit` | write | Submit one sample: validate, enqueue, start/reuse the single detached worker, return `run_id`. Dedupes by sample and still ensures the worker is online |
| `yifu_jobs` | read-only | Queue list ordered by status with step, percent, unit progress, ETA, heartbeat age, worker snapshot and structured error; filter by sample, project, state |
| `yifu_job` | read-only | One run: job record, `progress.json`, artifact manifest summary and log tail |
| `yifu_cancel` | write | Cooperative cancel: queued jobs cancel immediately, running jobs get a `cancel.flag` handled by the worker |

## Configuration

Set in this bundle's `cordis.patch.yml` row `config` (restart DSH after edits):

| Key | Meaning |
|---|---|
| `repoRoot` | Repository path (CLI working directory) |
| `pythonExe` | Interpreter that owns the pipeline environment; must be the micromamba `yifu` environment python, otherwise the spawned worker exits with `ModuleNotFoundError` |
| `analysisRoot` | Analysis root (`YIFU_ANALYSIS_ROOT`); the queue lives in its `_active` directory |
| `activeDir` | Optional explicit `YIFU_ACTIVE_DIR` override |
| `cliTimeoutMs` | Per-CLI-call timeout |
| `jobsLimit` / `logTailLines` | Default list size / log tail lines |
| `staleHeartbeatS` | Worker heartbeat age used for the approximate liveness flag |

Environment variables `YIFU_PM_REPO_ROOT`, `YIFU_PM_PYTHON`, `YIFU_PM_ANALYSIS_ROOT`
are not read: the patch config is the single source of truth.

## Projects

A project is the folder that owns the sample: `_active/<project>/<sample>` today,
or `<project>/<sample>` under the planned one-folder-per-project layout. An
optional `<project>/project.json` lets the operator park a project without
touching the queue:

```json
{ "display_name": "SQ_25_2610_0413 vessel batch", "state": "paused", "note": "waiting for customer payment" }
```

`state` is `active` (default), `paused` or `delivered`. The file is read-only for
the plugin; paused/delivered projects render dimmed with a tag and every job row
carries `project_state`.

## Behaviour and limits

- Read paths never write job records and never start tasks. External (`wrap`)
  job state is persisted by the worker's tick and by `wrap`'s own finalize.
- Success is decided by the harness verifier (exit code + required outputs exist
  and are readable + artifact manifest), never by log text.
- `harness wrap` is the supported path for programs `main.py` cannot express;
  the legacy two-line status-file `attach` protocol is compatibility only.
- Worker liveness in `yifu_jobs` is inferred from `worker.json` freshness and is
  labelled as approximate; the harness remains the authority.
- A run must outlive DSH: the detaching worker is spawned by the Python CLI, not
  by this plugin, because harness-managed child processes are terminated when
  the Host disposes them.

## Verification status

Verified in this workspace (2026-11-19):

- the Python harness test suite covering these tools: 67 passed; one
  pre-existing analytics test failure (a 30-day time window) is unrelated to
  this bundle and to the convergence changes;
- `harness list` against the real analysis root: 57 jobs, legacy `done`
  normalised to `succeeded`, job-record hash unchanged across the call;
- the installed bundle row `include:yifu-pm` is active and independent agent
  sessions could call `yifu_jobs` / `yifu_job` / `yifu_preflight` successfully;
- `GET /yifu-pm/jobs` and `GET /yifu-pm/job?run_id=...` answered HTTP 200 with
  live queue data (SHAM_1 running at 66.7%, heartbeat age ~7 s) and a bounded,
  line-cleaned log tail.

Not verified without a live page inspection: the rendered sidebar entry, the
panel view and the plugin row/icon/detail page in Plugin Manager. Reload the
page after installing; a stale page is the most likely reason a panel does not
appear.
