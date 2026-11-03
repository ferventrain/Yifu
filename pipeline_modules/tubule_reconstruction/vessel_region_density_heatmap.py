"""Render coronal atlas-slice heatmaps of per-region vessel volume density.

Reads a region density CSV with ``query`` (region id / acronym / name) and
``density_vessel_fraction`` columns, maps each region's density fraction onto
every atlas label id in that region's subtree, then renders coronal slices
with render_region_metric_atlas_slice.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_modules.tubule_reconstruction.region_vessel_analysis import (
    _collect_subtree_ids,
    load_region_tree_with_lookups,
    resolve_region_query,
)
from pipeline_modules.utils.data_paths import resolve_atlas_label_path
from pipeline_modules.visualization.atlas_slice import (
    AtlasSliceSpec,
)
from pipeline_modules.visualization.heatmap import (
    render_region_metric_atlas_slice,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--density_csv",
        default="",
        help=(
            "Region density CSV with 'query' and 'density_vessel_fraction' columns; "
            "produced by region_vessel_analysis --volume_only "
            "(region_vessel_volume_density.csv). Omit when --id_value_csv covers all ids"
        ),
    )
    parser.add_argument("--id_value_csv", default="", help="Direct id->density CSV painted by exact atlas id (no subtree resolution)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cfg", required=True, help="Allen region CSV")
    parser.add_argument("--label_path", default=None, help="atlas label TIFF (default resolve_atlas_label_path)")
    parser.add_argument("--bregma_start", type=float, default=1.1)
    parser.add_argument("--bregma_end", type=float, default=-5.2)
    parser.add_argument("--slice_count", type=int, default=12)
    parser.add_argument("--vmax_percent", type=float, default=6.0)
    parser.add_argument("--cmap", default="white_orange_red_black")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    density = (pd.read_csv(args.density_csv) if args.density_csv
               else pd.DataFrame(columns=["query", "density_vessel_fraction"]))
    label_path = Path(args.label_path) if args.label_path else resolve_atlas_label_path()
    nodes_by_id, acronym_to_ids, name_to_ids = load_region_tree_with_lookups(args.cfg)

    value_by_label: dict[int, float] = {}
    if getattr(args, "id_value_csv", ""):
        # direct id -> value rows (e.g. 2022-edition fine regions absent from
        # the region CSV tree); painted by exact id, no subtree resolution
        id_value = pd.read_csv(args.id_value_csv)
        for _, row in id_value.iterrows():
            value_by_label[int(row["id"])] = float(row["density_vessel_fraction"]) * 100.0
        print(f"Mapped {len(id_value)} ids from id_value csv {args.id_value_csv}")
    for _, row in density.iterrows():
        # iterrows upcasts a fully numeric query column to float64, turning
        # region id 1 into "1.0"; restore integer ids before resolving
        raw_query = row["query"]
        if isinstance(raw_query, float) and float(raw_query).is_integer():
            query = str(int(raw_query))
        else:
            query = str(raw_query)
        fraction = float(row["density_vessel_fraction"])
        node = resolve_region_query(query, nodes_by_id, acronym_to_ids, name_to_ids)
        for label_id in _collect_subtree_ids(node):
            value_by_label[int(label_id)] = fraction * 100.0

    print(f"Mapped {len(value_by_label)} labels from {len(density)} regions")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coords = [
        round(args.bregma_start + (args.bregma_end - args.bregma_start) * i / (args.slice_count - 1), 4)
        for i in range(args.slice_count)
    ]
    outputs = []
    for ap_mm in coords:
        spec = AtlasSliceSpec(
            plane="coronal",
            coordinate_system="bregma-mm",
            coordinate=ap_mm,
            atlas_resolution_um=25.0,
            bregma_index=(18, 216, 228),
        )
        output_path = out_dir / f"bregma_{ap_mm}mm.png"
        render_region_metric_atlas_slice(
            label_path,
            spec,
            value_by_label,
            output_path,
            cmap_name=args.cmap,
            vmin=0.0,
            vmax=args.vmax_percent,
            dpi=args.dpi,
            colorbar_label="vessel volume density (%)",
        )
        outputs.append(str(output_path))
        print(f"wrote {output_path}")

    (out_dir / "heatmap_summary.json").write_text(
        json.dumps(
            {
                "density_csv": str(args.density_csv),
                "bregma_coords_mm": coords,
                "vmin_percent": 0.0,
                "vmax_percent": args.vmax_percent,
                "cmap": args.cmap,
                "outputs": outputs,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Done, {len(outputs)} slices in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
