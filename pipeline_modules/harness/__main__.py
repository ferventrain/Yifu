"""CLI helpers for the pipeline harness queue."""

from __future__ import annotations

import argparse
import json
import sys

from pipeline_modules.harness.queue import ActiveStore
from pipeline_modules.utils.errors import PipelineError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LSFM pipeline harness queue helpers.")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    enqueue = sub.add_parser("enqueue", help="Add a sample to the Active queue")
    enqueue.add_argument("--sample-dir", required=True, help="Sample directory under the analysis root")
    enqueue.add_argument("--config", default="", help="config.json path; defaults to sample_dir/config.json")
    enqueue.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        dest="extra_args",
        help="Extra argument forwarded to main.py (repeatable), e.g. --extra-arg --only_tubule",
    )

    attach = sub.add_parser("attach", help="Track an already-running external pipeline")
    attach.add_argument("--sample-dir", required=True)
    attach.add_argument("--title", required=True)
    attach.add_argument("--pid", type=int, default=0)
    attach.add_argument("--log", default="")
    attach.add_argument("--status", default="")
    attach.add_argument("--config", default="")
    attach.add_argument("--command", default="")
    attach.add_argument("--module-progress", default="", help="Path to module _progress.json for ETA")
    attach.add_argument("--step-total", type=int, default=7)

    sub.add_parser("list", help="Print Active jobs as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = ActiveStore()
    try:
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
        if args.subcommand == "list":
            print(json.dumps(store.list_job_views(), indent=2, ensure_ascii=False))
            return 0
    except PipelineError as exc:
        print(json.dumps(exc.to_dict(), indent=2, ensure_ascii=False), file=sys.stderr)
        return exc.exit_code
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
