"""CLI for the pipeline harness.

THE standard entry for running an existing pipeline:

    python -m pipeline_modules.harness run \
      --sample-dir "S:/path/to/sample" \
      --config "S:/path/to/sample/config.json"

Validates, creates the job record, starts (or reuses) the single detached
worker, prints the run_id, and exits. Watch it in Pipeline Monitor:
http://127.0.0.1:8766 — no second script, no status-file editing.

Legacy subcommands (``enqueue``/``attach``/``list``) still work for
compatibility but are no longer the recommended path.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from pipeline_modules.harness.queue import ActiveStore
from pipeline_modules.harness.runspec import ALLOWED_EXTRA_FLAGS, RunSpec
from pipeline_modules.harness.verify import required_outputs
from pipeline_modules.utils.errors import ErrorCode, PipelineError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LSFM pipeline harness: one command to run a sample.")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    run = sub.add_parser("run", help="Submit a sample and start/reuse the worker; prints run_id")
    run.add_argument("--sample-dir", required=True, help="Sample directory under the analysis root")
    run.add_argument("--config", default="", help="config.json path; defaults to sample_dir/config.json")
    run.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        dest="extra_args",
        help=f"Pre-declared main.py flag (repeatable); allowed: {sorted(ALLOWED_EXTRA_FLAGS)}",
    )
    run.add_argument(
        "--no-start-worker",
        action="store_true",
        help="Queue the job without spawning the worker (tests / manual worker control)",
    )

    preflight = sub.add_parser("preflight", help="Validate inputs and config without submitting")
    preflight.add_argument("--sample-dir", required=True)
    preflight.add_argument("--config", default="")
    preflight.add_argument("--extra-arg", action="append", default=[], dest="extra_args")

    sub.add_parser("worker", help="Run the detached serial worker (started automatically by `run`)")

    cancel = sub.add_parser("cancel", help="Cancel a queued or running job")
    cancel.add_argument("run_id", help="run_id printed by the run command")

    enqueue = sub.add_parser("enqueue", help="LEGACY: add a sample to the queue without starting a worker")
    enqueue.add_argument("--sample-dir", required=True, help="Sample directory under the analysis root")
    enqueue.add_argument("--config", default="", help="config.json path; defaults to sample_dir/config.json")
    enqueue.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        dest="extra_args",
        help="Extra argument forwarded to main.py (repeatable), e.g. --extra-arg --only_tubule",
    )

    attach = sub.add_parser("attach", help="LEGACY: track an already-running external pipeline")
    attach.add_argument("--sample-dir", required=True)
    attach.add_argument("--title", required=True)
    attach.add_argument("--pid", type=int, default=0)
    attach.add_argument("--log", default="")
    attach.add_argument("--status", default="")
    attach.add_argument("--config", default="")
    attach.add_argument("--command", default="")
    attach.add_argument("--module-progress", default="", help="Path to module _progress.json for ETA")
    attach.add_argument("--step-total", type=int, default=7)

    sub.add_parser("list", help="Print jobs as JSON (read-only)")
    return parser


def _load_spec(args: argparse.Namespace) -> tuple[RunSpec, dict[str, Any]]:
    spec = RunSpec.from_args(args.sample_dir, args.config or None, extra_args=args.extra_args)
    from pipeline_modules.harness.results import load_config

    config = load_config(spec.config_path)
    return spec, config


def cmd_run(args: argparse.Namespace, store: ActiveStore) -> int:
    spec, config = _load_spec(args)
    store.ensure_layout()
    store.validate_sample_dir(spec.sample_dir)
    store.validate_config(spec.config_path)

    existing = store.find_active_for_sample(spec.sample_dir)
    if existing is not None:
        payload = _run_payload(store, existing, already_active=True)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    record = store.add_job(spec.sample_dir, spec.config_path, extra_args=list(spec.extra_args))
    payload = _run_payload(store, record, already_active=False)
    if not args.no_start_worker:
        from pipeline_modules.harness.worker import ensure_worker_running

        payload["worker"] = ensure_worker_running(store)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _run_payload(store: ActiveStore, record: dict[str, Any], *, already_active: bool) -> dict[str, Any]:
    from pipeline_modules.harness.paths import progress_path, stdout_log_path

    job_id = str(record["id"])
    return {
        "run_id": job_id,
        "status": record.get("status") or "queued",
        "already_active": already_active,
        "sample_dir": record.get("sample_dir"),
        "config_path": record.get("config_path"),
        "monitor": "http://127.0.0.1:8766",
        "log_path": str(stdout_log_path(job_id, store.active)),
        "progress_path": str(progress_path(job_id, store.active)),
    }


def cmd_preflight(args: argparse.Namespace, store: ActiveStore) -> int:
    spec, config = _load_spec(args)
    store.validate_sample_dir(spec.sample_dir)
    store.validate_config(spec.config_path)
    from pipeline_modules.harness.worker import worker_snapshot
    from pipeline_modules.utils.run_manifest import collect_code_version

    print(
        json.dumps(
            {
                "status": "ok",
                "sample_dir": str(spec.sample_dir),
                "config_path": str(spec.config_path),
                "pipeline": spec.pipeline,
                "extra_args": list(spec.extra_args),
                "required_outputs": [str(p) for p in required_outputs(spec.sample_dir, config, list(spec.extra_args))],
                "active_run": (store.find_active_for_sample(spec.sample_dir) or {}).get("id"),
                "worker": worker_snapshot(store.active),
                "code_version": collect_code_version(),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def cmd_worker(_args: argparse.Namespace, store: ActiveStore) -> int:
    from pipeline_modules.harness.worker import HarnessWorker

    store.ensure_layout()
    return HarnessWorker(store).run_forever()


def cmd_cancel(args: argparse.Namespace, store: ActiveStore) -> int:
    from pipeline_modules.harness.worker import request_cancel

    result = request_cancel(store, args.run_id)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_list(_args: argparse.Namespace, store: ActiveStore) -> int:
    from pipeline_modules.harness.worker import decorate_views

    views = decorate_views(store.list_job_views(), store.active)
    print(json.dumps(views, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = ActiveStore()
    try:
        if args.subcommand == "run":
            return cmd_run(args, store)
        if args.subcommand == "preflight":
            return cmd_preflight(args, store)
        if args.subcommand == "worker":
            return cmd_worker(args, store)
        if args.subcommand == "cancel":
            return cmd_cancel(args, store)
        if args.subcommand == "list":
            return cmd_list(args, store)
        if args.subcommand == "enqueue":
            record = store.add_job(args.sample_dir, args.config or None, extra_args=args.extra_args)
            print(json.dumps(record, indent=2, ensure_ascii=False))
            return 0
        if args.subcommand == "attach":
            record = store.attach_external(
                args.sample_dir,
                title=args.title,
                pid=args.pid or None,
                log_path=args.log or None,
                status_path=args.status or None,
                config_path=args.config or None,
                module_progress_path=args.module_progress or None,
                command=args.command,
                step_total=args.step_total,
            )
            print(json.dumps(record, indent=2, ensure_ascii=False))
            return 0
    except PipelineError as exc:
        print(json.dumps(exc.to_dict(), indent=2, ensure_ascii=False), file=sys.stderr)
        return exc.exit_code
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
