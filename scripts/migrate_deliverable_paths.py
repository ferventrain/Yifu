#!/usr/bin/env python3
"""Migrate legacy pipeline deliverables to standard naming rules.

Default is dry-run. Use --apply to create links/copies/moves.

Examples:
  python scripts/migrate_deliverable_paths.py S:/Arivis_Analysis/_delivered/YF2026032001zzj_cfos/sham
  python scripts/migrate_deliverable_paths.py S:/Arivis_Analysis/_delivered/YF2026032001zzj_cfos --apply --method copy
  python scripts/migrate_deliverable_paths.py S:/Arivis_Analysis/_active --recursive --apply
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DELIVERABLE_PATHS = ROOT / "pipeline_modules" / "utils" / "deliverable_paths.py"
_spec = importlib.util.spec_from_file_location("deliverable_paths", DELIVERABLE_PATHS)
_dp = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_dp)

brain_distribution_stats_xlsx = _dp.brain_distribution_stats_xlsx
heatmap_2d_dir = _dp.heatmap_2d_dir
heatmap_3d_colorbar_png = _dp.heatmap_3d_colorbar_png
heatmap_3d_png = _dp.heatmap_3d_png
heatmap_3d_stack_tiff = _dp.heatmap_3d_stack_tiff
heatmap_3d_summary_json = _dp.heatmap_3d_summary_json
heatmap_3d_volume_tiff = _dp.heatmap_3d_volume_tiff
legacy_brain_distribution_candidates = _dp.legacy_brain_distribution_candidates
normalize_channel = _dp.normalize_channel
results_dir = _dp.results_dir
sample_slug = _dp.sample_slug
visualization_dir = _dp.visualization_dir

LEGACY_2D_DIR_NAMES = (
    "cell_density_slices",
    "region_metric_slices",
    "signal_count_diff_slices",
)


@dataclass
class MigrationAction:
    sample: str
    kind: str
    source: str
    target: str
    method: str
    status: str
    note: str = ""


def infer_signal_channel(sample_dir: Path) -> str:
    for path in sorted(sample_dir.glob("ch*_mask.zarr")):
        match = re.fullmatch(r"(ch\d+)_mask\.zarr", path.name)
        if match:
            return match.group(1)

    for path in sorted(sample_dir.glob("sample_ch*_result.xlsx")):
        match = re.fullmatch(r"sample_(ch\d+)_result\.xlsx", path.name)
        if match:
            return match.group(1)

    for path in sorted(sample_dir.glob("density_results_ch*.xlsx")):
        match = re.fullmatch(r"density_results_(ch\d+)\.xlsx", path.name)
        if match:
            return match.group(1)

    for path in sorted(sample_dir.glob("ch*.zarr")):
        match = re.fullmatch(r"(ch\d+)\.zarr", path.name)
        if match and not path.name.endswith("_mask.zarr") and not path.name.endswith("_prob.zarr"):
            name = match.group(1)
            if name != "ch0":
                return name

    return "ch1"


def is_sample_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if any(path.glob("ch*_mask.zarr")):
        return True
    if (path / "upsampled_atlas_label.zarr").exists():
        return True
    if any(path.glob("ch[0-9]")) and any(path.glob("ch[0-9]*.zarr")):
        return True
    if any(path.glob("sample_ch*_result.xlsx")):
        return True
    vis = path / "visualization"
    if vis.is_dir() and any(vis.glob("*heatmap3d*")):
        return True
    return False


def has_nested_sample_dirs(path: Path) -> bool:
    skip_names = {"visualization", "results", "transforms", "__pycache__", "备份"}
    for child in path.iterdir():
        if not child.is_dir() or child.name in skip_names or child.name.endswith(".zarr"):
            continue
        if is_sample_dir(child):
            return True
    return False


def discover_sample_dirs(root: Path, *, recursive: bool) -> list[Path]:
    root = root.resolve()
    if is_sample_dir(root) and not has_nested_sample_dirs(root):
        return [root]

    if not recursive:
        return [
            child
            for child in sorted(root.iterdir())
            if child.is_dir() and is_sample_dir(child)
        ]

    found: list[Path] = []
    for dirpath, dirnames, _ in os.walk(root):
        current = Path(dirpath)
        if not is_sample_dir(current):
            continue
        if has_nested_sample_dirs(current):
            continue
        found.append(current)
        dirnames[:] = []
    return sorted(set(found))


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return False


def plan_file_action(
    actions: list[MigrationAction],
    *,
    sample_dir: Path,
    channel: str,
    kind: str,
    source: Path | None,
    target: Path,
    method: str,
) -> None:
    if source is None or not source.exists():
        return
    if _same_file(source, target):
        return
    if target.exists():
        actions.append(
            MigrationAction(
                sample=sample_dir.name,
                kind=kind,
                source=str(source),
                target=str(target),
                method=method,
                status="skip",
                note="target already exists",
            )
        )
        return
    actions.append(
        MigrationAction(
            sample=sample_dir.name,
            kind=kind,
            source=str(source),
            target=str(target),
            method=method,
            status="planned",
        )
    )


def plan_excel_migration(actions: list[MigrationAction], sample_dir: Path, channel: str, method: str) -> None:
    target = brain_distribution_stats_xlsx(sample_dir, channel)
    if target.exists():
        return
    for candidate in legacy_brain_distribution_candidates(sample_dir, channel):
        if not candidate.exists() or _same_file(candidate, target):
            continue
        if candidate.name == "merged_density.xlsx":
            continue
        plan_file_action(
            actions,
            sample_dir=sample_dir,
            channel=channel,
            kind="brain_distribution_stats",
            source=candidate,
            target=target,
            method=method,
        )
        return


def plan_heatmap_3d_migration(actions: list[MigrationAction], sample_dir: Path, channel: str, method: str) -> None:
    slug = sample_slug(sample_dir)
    vis = visualization_dir(sample_dir)
    legacy_map = {
        "heatmap_3d_volume": vis / f"{slug}_heatmap3d_volume.tiff",
        "heatmap_3d_stack": vis / f"{slug}_heatmap3d_stack.tiff",
        "heatmap_3d_colorbar": vis / f"{slug}_heatmap3d_colorbar.png",
        "heatmap_3d_summary": vis / f"{slug}_heatmap3d_stack.json",
    }
    target_map = {
        "heatmap_3d_volume": heatmap_3d_volume_tiff(sample_dir, channel),
        "heatmap_3d_stack": heatmap_3d_stack_tiff(sample_dir, channel),
        "heatmap_3d_colorbar": heatmap_3d_colorbar_png(sample_dir, channel),
        "heatmap_3d_summary": heatmap_3d_summary_json(sample_dir, channel),
    }
    for kind, source in legacy_map.items():
        plan_file_action(
            actions,
            sample_dir=sample_dir,
            channel=channel,
            kind=kind,
            source=source,
            target=target_map[kind],
            method=method,
        )

    png_target = heatmap_3d_png(sample_dir, channel)
    if not png_target.exists():
        legacy_colorbar = vis / f"{slug}_heatmap3d_colorbar.png"
        new_colorbar = target_map["heatmap_3d_colorbar"]
        source_for_png = legacy_colorbar if legacy_colorbar.exists() else (new_colorbar if new_colorbar.exists() else None)
        if source_for_png is not None:
            actions.append(
                MigrationAction(
                    sample=sample_dir.name,
                    kind="heatmap_3d_png",
                    source=str(source_for_png),
                    target=str(png_target),
                    method=method,
                    status="planned",
                    note="temporary link from colorbar until full 3D PNG is regenerated",
                )
            )
        else:
            actions.append(
                MigrationAction(
                    sample=sample_dir.name,
                    kind="heatmap_3d_png",
                    source="",
                    target=str(png_target),
                    method=method,
                    status="missing",
                    note="re-run heatmap sample-stack to generate preview PNG",
                )
            )


def plan_heatmap_2d_migration(actions: list[MigrationAction], sample_dir: Path, channel: str, method: str) -> None:
    target_dir = heatmap_2d_dir(sample_dir, channel)
    if target_dir.exists():
        return

    vis = visualization_dir(sample_dir)
    legacy_dirs = [vis / name for name in LEGACY_2D_DIR_NAMES if (vis / name).is_dir()]
    if not legacy_dirs:
        return

    if len(legacy_dirs) == 1:
        actions.append(
            MigrationAction(
                sample=sample_dir.name,
                kind="heatmap_2d_dir",
                source=str(legacy_dirs[0]),
                target=str(target_dir),
                method=method,
                status="planned",
            )
        )
        return

    actions.append(
        MigrationAction(
            sample=sample_dir.name,
            kind="heatmap_2d_dir_merge",
            source=";".join(str(path) for path in legacy_dirs),
            target=str(target_dir),
            method="merge",
            status="planned",
            note="merge PNG/SVG files from multiple legacy slice folders",
        )
    )


def plan_sample_migrations(sample_dir: Path, method: str) -> list[MigrationAction]:
    channel = infer_signal_channel(sample_dir)
    actions: list[MigrationAction] = []
    plan_excel_migration(actions, sample_dir, channel, method)
    plan_heatmap_3d_migration(actions, sample_dir, channel, method)
    plan_heatmap_2d_migration(actions, sample_dir, channel, method)
    return actions


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def apply_action(action: MigrationAction) -> MigrationAction:
    if action.status != "planned":
        return action

    source = Path(action.source) if action.source else None
    target = Path(action.target)

    try:
        if action.method == "merge":
            target.mkdir(parents=True, exist_ok=True)
            for legacy_dir in [Path(part) for part in action.source.split(";")]:
                for item in legacy_dir.iterdir():
                    dest = target / f"{legacy_dir.name}_{item.name}"
                    if dest.exists():
                        continue
                    if item.is_dir():
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)
            action.status = "applied"
            return action

        if source is None:
            action.status = "failed"
            action.note = "missing source"
            return action

        ensure_parent(target)

        if action.method == "symlink":
            os.symlink(source, target)
        elif action.method == "hardlink":
            os.link(source, target)
        elif action.method == "copy":
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
        elif action.method == "move":
            shutil.move(str(source), str(target))
        else:
            raise ValueError(f"Unknown method: {action.method}")

        action.status = "applied"
    except OSError as exc:
        action.status = "failed"
        action.note = str(exc)
    return action


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate legacy deliverable paths to standard naming rules.")
    parser.add_argument("path", help="Sample directory or project root")
    parser.add_argument("--recursive", action="store_true", help="Scan all sample-like directories under path")
    parser.add_argument(
        "--method",
        choices=("symlink", "hardlink", "copy", "move"),
        default="symlink",
        help="How to create the new deliverable path (default: symlink)",
    )
    parser.add_argument("--apply", action="store_true", help="Apply planned migrations (default: dry-run)")
    parser.add_argument(
        "--report",
        default="",
        help="Optional JSON report output path",
    )
    args = parser.parse_args(argv)

    root = Path(args.path)
    if not root.exists():
        print(f"Path not found: {root}", file=sys.stderr)
        return 2

    samples = discover_sample_dirs(root, recursive=args.recursive)
    if not samples:
        print(f"No sample directories found under: {root}", file=sys.stderr)
        return 1

    all_actions: list[MigrationAction] = []
    for sample_dir in samples:
        all_actions.extend(plan_sample_migrations(sample_dir, args.method))

    if args.apply:
        all_actions = [apply_action(action) for action in all_actions]

    payload = {
        "root": str(root),
        "apply": bool(args.apply),
        "method": args.method,
        "samples": [sample.name for sample in samples],
        "actions": [asdict(action) for action in all_actions],
        "summary": {
            "planned": sum(1 for action in all_actions if action.status == "planned"),
            "applied": sum(1 for action in all_actions if action.status == "applied"),
            "skipped": sum(1 for action in all_actions if action.status == "skip"),
            "missing": sum(1 for action in all_actions if action.status == "missing"),
            "failed": sum(1 for action in all_actions if action.status == "failed"),
        },
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote report: {report_path}")

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to execute.", file=sys.stderr)

    return 1 if payload["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
