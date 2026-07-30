"""Build a reusable atlas region pairing reference (paired vs midline).

Scans an atlas label TIFF once with 3D connected components (cc3d, 26-connect)
and writes JSON:

    {
      "atlas_label": "...",
      "connectivity": 26,
      "paired_rule": "n_cc >= 2",
      "regions": {
        "123": {"n_cc": 2, "paired": true, "name": "..."},
        "456": {"n_cc": 1, "paired": false, "name": "..."}
      }
    }

Use this file with render_ab_hemisphere_ratio_heatmap.py so heatmap runs do
not re-scan the atlas.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from pipeline_modules.visualization.atlas_slice import (
    DEFAULT_ATLAS_LABEL,
    _read_label_volume,
    build_region_name_lookup,
    count_region_connected_components,
    default_region_pairing_reference_path,
    save_region_pairing_reference,
)
from pipeline_modules.visualization.heatmap import default_region_cfg_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan atlas label volume and write region pairing reference JSON.",
    )
    parser.add_argument(
        "--atlas_label",
        default=str(DEFAULT_ATLAS_LABEL),
        help="3D atlas label TIFF (DV, AP, ML). Default: project/data or YIFU_DATA_DIR reference.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output JSON path. Default: YIFU_DATA_DIR/reference/region_pairing.json",
    )
    parser.add_argument(
        "--cfg",
        default="",
        help="Optional region CSV to attach names to each region id.",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        default=26,
        choices=(6, 18, 26),
        help="cc3d connectivity (default 26).",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=25,
        help="Print progress every N regions (0 disables).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    atlas_label = Path(args.atlas_label)
    output = Path(args.output) if args.output else default_region_pairing_reference_path()
    cfg_path = Path(args.cfg) if args.cfg else default_region_cfg_path()

    print(f"Loading atlas label: {atlas_label}", flush=True)
    t0 = time.perf_counter()
    volume = np.asarray(_read_label_volume(atlas_label))
    if hasattr(volume, "filename"):
        volume = np.array(volume)
    print(f"  shape={tuple(volume.shape)} dtype={volume.dtype} ({time.perf_counter() - t0:.1f}s)", flush=True)

    name_by_region_id = None
    if cfg_path.exists():
        name_by_region_id = build_region_name_lookup(cfg_path)
        print(f"Loaded region names from: {cfg_path}", flush=True)
    else:
        print(f"Region CSV not found ({cfg_path}); writing ids only.", flush=True)

    print(f"Counting 3D connected components (connectivity={args.connectivity})...", flush=True)
    t1 = time.perf_counter()
    n_cc = count_region_connected_components(
        volume,
        connectivity=int(args.connectivity),
        progress_every=int(args.progress_every),
    )
    elapsed = time.perf_counter() - t1
    paired_n = sum(1 for count in n_cc.values() if count >= 2)
    unpaired_n = sum(1 for count in n_cc.values() if count == 1)
    empty_n = sum(1 for count in n_cc.values() if count <= 0)
    print(
        f"Done in {elapsed:.1f}s: {len(n_cc)} regions "
        f"(paired={paired_n}, unpaired={unpaired_n}, empty={empty_n})",
        flush=True,
    )

    out_path = save_region_pairing_reference(
        output,
        n_cc,
        atlas_label=atlas_label,
        connectivity=int(args.connectivity),
        name_by_region_id=name_by_region_id,
        extra_meta={"elapsed_sec": round(elapsed, 3)},
    )
    print(f"Wrote pairing reference: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
