"""Convert float32 Allen atlas_label.tiff to uint32 with CSV ID recovery.

Large Allen structure IDs (>2^24) cannot be stored exactly in float32. This script
rewrites each corrupted atlas value to the best-matching region id from the
ontology CSV (exact match when unique; nearest candidate when float32 collisions).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import tifffile


def build_float32_recovery_map(csv_ids: list[int]) -> dict[int, int]:
    """Map float32-rounded atlas values to a recovered Allen region id."""
    by_f32: dict[int, list[int]] = {}
    for rid in csv_ids:
        key = int(np.float32(rid))
        by_f32.setdefault(key, []).append(int(rid))

    recovery: dict[int, int] = {}
    csv_set = set(csv_ids)
    for rid in csv_ids:
        recovery[int(rid)] = int(rid)

    for key, candidates in by_f32.items():
        if key in csv_set and len(candidates) == 1:
            recovery[key] = candidates[0]
            continue
        # Prefer exact CSV id when the rounded key itself is a real id.
        if key in csv_set:
            recovery[key] = key
            continue
        recovery[key] = min(candidates, key=lambda cid: (abs(cid - key), cid))
    return recovery


def float32_collision_groups(csv_ids: list[int]) -> dict[int, tuple[int, ...]]:
    """Map float32(key) -> all ontology ids that round to that key."""
    by_f32: dict[int, list[int]] = {}
    for rid in csv_ids:
        key = int(np.float32(rid))
        by_f32.setdefault(key, []).append(int(rid))
    return {key: tuple(sorted(vals)) for key, vals in by_f32.items()}


def expand_region_ids_for_atlas_labels(
    region_ids: Iterable[int],
    cfg_path: str | Path,
) -> frozenset[int]:
    """Expand ontology ids so they match labels on a float32-repaired atlas.

    When several large Allen ids collapse to one float32 value, the repaired atlas
    keeps only one representative id. Selecting any collapsed sibling must still
    highlight those voxels, so include the representative and all collision aliases.
    """
    cfg_path = Path(cfg_path)
    csv = pd.read_csv(cfg_path)
    if "id" not in csv.columns:
        raise ValueError(f"Region CSV missing 'id' column: {cfg_path}")
    csv_ids = [int(x) for x in csv["id"]]
    recovery = build_float32_recovery_map(csv_ids)
    groups = float32_collision_groups(csv_ids)

    expanded: set[int] = set()
    for raw_id in region_ids:
        rid = int(raw_id)
        expanded.add(rid)
        key = int(np.float32(rid))
        expanded.add(int(recovery.get(key, rid)))
        for sibling in groups.get(key, (rid,)):
            expanded.add(int(sibling))
            sibling_key = int(np.float32(sibling))
            expanded.add(int(recovery.get(sibling_key, sibling)))
    return frozenset(expanded)


def convert_atlas_label(
    atlas_path: Path,
    cfg_path: Path,
    *,
    backup: bool = True,
    output_path: Path | None = None,
) -> dict[str, object]:
    atlas_path = Path(atlas_path)
    cfg_path = Path(cfg_path)
    output_path = Path(output_path) if output_path else atlas_path

    csv = pd.read_csv(cfg_path)
    if "id" not in csv.columns:
        raise ValueError(f"Region CSV missing 'id' column: {cfg_path}")
    csv_ids = [int(x) for x in csv["id"]]
    csv_set = set(csv_ids)
    recovery = build_float32_recovery_map(csv_ids)

    volume = np.asarray(tifffile.imread(str(atlas_path)))
    original_dtype = str(volume.dtype)
    as_int = np.rint(volume).astype(np.int64, copy=False)
    unique_vals = [int(x) for x in np.unique(as_int) if int(x) > 0]

    unmapped_before = [rid for rid in unique_vals if rid not in csv_set]
    remap = {0: 0}
    ambiguous = 0
    unique_recoveries = 0
    missing = 0
    by_f32: dict[int, list[int]] = {}
    for rid in csv_ids:
        by_f32.setdefault(int(np.float32(rid)), []).append(int(rid))

    for rid in unique_vals:
        if rid in csv_set:
            remap[rid] = rid
            continue
        if rid in recovery:
            remap[rid] = int(recovery[rid])
            cands = by_f32.get(rid, [])
            if len(cands) > 1:
                ambiguous += 1
            else:
                unique_recoveries += 1
        else:
            remap[rid] = 0
            missing += 1

    out = np.zeros(as_int.shape, dtype=np.uint32)
    for src, dst in remap.items():
        if src == 0:
            continue
        out[as_int == src] = np.uint32(dst)

    unmapped_after = [int(x) for x in np.unique(out) if int(x) > 0 and int(x) not in csv_set]

    if backup and output_path.resolve() == atlas_path.resolve():
        backup_path = atlas_path.with_suffix(atlas_path.suffix + ".float32.bak")
        if not backup_path.exists():
            shutil.copy2(atlas_path, backup_path)
    else:
        backup_path = None

    tifffile.imwrite(str(output_path), out, compression="zlib")
    return {
        "atlas_path": str(atlas_path),
        "output_path": str(output_path),
        "backup_path": str(backup_path) if backup_path else None,
        "original_dtype": original_dtype,
        "output_dtype": "uint32",
        "unique_labels_before": len(unique_vals),
        "unmapped_before": len(unmapped_before),
        "unique_recoveries": unique_recoveries,
        "ambiguous_recoveries": ambiguous,
        "unrecoverable_set_to_0": missing,
        "unmapped_after": len(unmapped_after),
        "unmapped_after_ids": unmapped_after[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--atlas",
        type=Path,
        default=Path(r"S:\Yifu_data\reference\atlas_label.tiff"),
    )
    parser.add_argument(
        "--cfg",
        type=Path,
        default=Path(r"S:\Yifu\pipeline_modules\registration\Region_Csv_Rev1_updated.CSV"),
    )
    parser.add_argument("--output", type=Path, default=None, help="Defaults to in-place overwrite")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    summary = convert_atlas_label(
        args.atlas,
        args.cfg,
        backup=not args.no_backup,
        output_path=args.output,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
