"""CLI: sample A/B left-right ratio heatmap on a coronal atlas slice.

Paired regions (from a precomputed pairing reference JSON) are colored with
A_L/B_L on the ML-left half and A_R/B_R on the ML-right half. Single-CC /
midline regions use (A_L+A_R)/(B_L+B_R) for the whole region.

Generate the pairing reference once:

    python pipeline_modules/visualization/build_region_pairing_reference.py \\
        --atlas_label <atlas_label.tiff> --output <region_pairing.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from pipeline_modules.visualization.atlas_slice import (
    DEFAULT_ATLAS_LABEL,
    DEFAULT_BREGMA_INDEX,
    AtlasSliceSpec,
    build_hemisphere_ab_ratio_lookups,
    build_region_metric_lookup,
    compute_ratio_color_limits,
    default_region_pairing_reference_path,
    extract_atlas_slice,
    load_region_pairing_reference,
    paint_hemisphere_ratio_slice,
    resolve_slice_region_values,
)
from pipeline_modules.visualization.heatmap import default_region_cfg_path
from pipeline_modules.visualization.slice_heatmap_render import (
    DEFAULT_BRAIN_OUTLINE_WIDTH,
    DEFAULT_CONTOUR_SMOOTH,
    DEFAULT_PIXEL_SCALE,
    DEFAULT_REGION_LINE_WIDTH,
    DEFAULT_SUPERSAMPLE,
    save_slice_heatmap_png,
)

METRIC_COLUMNS = {
    "signal_count": ("Left Signal Count", "Right Signal Count"),
    "voxel_density": ("Left Voxel Density", "Right Voxel Density"),
}


def _parse_bregma_index(text: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("bregma_index must be DV,AP,ML")
    return int(parts[0]), int(parts[1]), int(parts[2])


def render_painted_ratio_slice_png(
    painted: np.ndarray,
    label_slice: np.ndarray,
    output_path: str | Path,
    *,
    vmin: float,
    vmax: float,
    vcenter: float = 1.0,
    cmap_name: str = "signal_count_diff",
    pixel_scale: int = DEFAULT_PIXEL_SCALE,
    line_width: float = DEFAULT_REGION_LINE_WIDTH,
    brain_outline_width: float = DEFAULT_BRAIN_OUTLINE_WIDTH,
    show_region_contours: bool = True,
    contour_smooth: float = DEFAULT_CONTOUR_SMOOTH,
    fill_softness: float = 0.0,
    supersample: int = DEFAULT_SUPERSAMPLE,
    colorbar_label: str = "A/B",
    background: str = "white",
    colorbar_width_px: int = 72,
) -> Path:
    """Render A/B ratio slice via the shared high-quality heatmap renderer."""
    _ = fill_softness
    _ = colorbar_width_px
    return save_slice_heatmap_png(
        painted,
        label_slice,
        output_path,
        mode="labeled",
        vmin=vmin,
        vmax=vmax,
        vcenter=vcenter,
        cmap_name=cmap_name,
        pixel_scale=pixel_scale,
        supersample=supersample,
        contour_smooth=contour_smooth,
        line_width=line_width,
        brain_outline_width=brain_outline_width,
        show_region_contours=show_region_contours,
        colorbar_label=colorbar_label,
        background=background,
        include_colorbar=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render A/B hemisphere ratio heatmap (paired L/R vs midline sum).",
    )
    parser.add_argument("--excel_a", required=True, help="Sample A density Excel workbook.")
    parser.add_argument("--excel_b", required=True, help="Sample B density Excel workbook.")
    parser.add_argument(
        "--pairing_reference",
        default="",
        help="Region pairing JSON from build_region_pairing_reference.py. "
        "Default: YIFU_DATA_DIR/reference/region_pairing.json",
    )
    parser.add_argument("--atlas_label", default=str(DEFAULT_ATLAS_LABEL), help="3D atlas label TIFF.")
    parser.add_argument("--cfg", default="", help="Region CSV path.")
    parser.add_argument(
        "--metric",
        default="signal_count",
        choices=sorted(METRIC_COLUMNS),
        help="Which L/R metric columns to ratio (default: signal_count).",
    )
    parser.add_argument("--plane", default="coronal", choices=("coronal",), help="Only coronal supported.")
    parser.add_argument("--bregma_mm", type=float, required=True, help="Coronal AP coordinate in mm from bregma.")
    parser.add_argument("--atlas_resolution_um", type=float, default=25.0)
    parser.add_argument("--bregma_index", type=_parse_bregma_index, default=",".join(str(v) for v in DEFAULT_BREGMA_INDEX))
    parser.add_argument("--pseudocount", type=float, default=1.0)
    parser.add_argument("--vmin", type=float, default=None, help="Optional explicit ratio color lower bound.")
    parser.add_argument("--vmax", type=float, default=None, help="Optional explicit ratio color upper bound.")
    parser.add_argument("--cmap", default="signal_count_diff")
    parser.add_argument("--pixel_scale", type=int, default=DEFAULT_PIXEL_SCALE)
    parser.add_argument("--supersample", type=int, default=DEFAULT_SUPERSAMPLE)
    parser.add_argument("--contour_smooth", type=float, default=DEFAULT_CONTOUR_SMOOTH)
    parser.add_argument("--brain_outline_width", type=float, default=DEFAULT_BRAIN_OUTLINE_WIDTH, help="0 disables outer brain outline.")
    parser.add_argument("--fill_softness", type=float, default=0.0, help="Deprecated/ignored.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--summary_json", default="", help="Optional sidecar JSON with lookups/limits.")
    parser.add_argument("--label_a", default="A", help="Short sample A name for colorbar.")
    parser.add_argument("--label_b", default="B", help="Short sample B name for colorbar.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    excel_a = Path(args.excel_a)
    excel_b = Path(args.excel_b)
    atlas_label = Path(args.atlas_label)
    cfg_path = Path(args.cfg) if args.cfg else default_region_cfg_path()
    pairing_path = Path(args.pairing_reference) if args.pairing_reference else default_region_pairing_reference_path()
    output_path = Path(args.output)

    if not pairing_path.exists():
        raise FileNotFoundError(
            f"Pairing reference not found: {pairing_path}\n"
            "Generate it once with:\n"
            f'  python pipeline_modules/visualization/build_region_pairing_reference.py --atlas_label "{atlas_label}" --output "{pairing_path}"'
        )

    left_metric, right_metric = METRIC_COLUMNS[str(args.metric)]
    a_left, path_by_region_id = build_region_metric_lookup(
        excel_a, cfg_path=cfg_path, metric=left_metric, direct_label_only=True
    )
    a_right, _ = build_region_metric_lookup(
        excel_a, cfg_path=cfg_path, metric=right_metric, direct_label_only=True
    )
    b_left, _ = build_region_metric_lookup(
        excel_b, cfg_path=cfg_path, metric=left_metric, direct_label_only=True
    )
    b_right, _ = build_region_metric_lookup(
        excel_b, cfg_path=cfg_path, metric=right_metric, direct_label_only=True
    )

    n_cc_by_region_id, paired_by_region_id = load_region_pairing_reference(pairing_path)
    left_ratios, right_ratios = build_hemisphere_ab_ratio_lookups(
        a_left=a_left,
        a_right=a_right,
        b_left=b_left,
        b_right=b_right,
        n_cc_by_region_id=n_cc_by_region_id,
        paired_by_region_id=paired_by_region_id,
        pseudocount=float(args.pseudocount),
    )

    bregma_index = args.bregma_index if isinstance(args.bregma_index, tuple) else _parse_bregma_index(str(args.bregma_index))
    spec = AtlasSliceSpec(
        plane=str(args.plane),
        coordinate_system="bregma-mm",
        coordinate=float(args.bregma_mm),
        atlas_resolution_um=float(args.atlas_resolution_um),
        bregma_index=bregma_index,
    )
    atlas_slice = extract_atlas_slice(atlas_label, spec)
    left_resolved = resolve_slice_region_values(
        atlas_slice.image, left_ratios, path_by_region_id, inherit_ancestors=False
    )
    right_resolved = resolve_slice_region_values(
        atlas_slice.image, right_ratios, path_by_region_id, inherit_ancestors=False
    )
    painted = paint_hemisphere_ratio_slice(
        atlas_slice.image,
        left_resolved,
        right_resolved,
        paired_by_region_id,
        ml_mid_index=int(bregma_index[2]),
    )

    finite_values = [float(v) for v in painted[np.isfinite(painted)].tolist() if float(v) > 0]
    if args.vmin is not None and args.vmax is not None:
        vmin, vmax = float(args.vmin), float(args.vmax)
    elif args.vmax is not None:
        vmin, vmax = compute_ratio_color_limits(finite_values, explicit_vmax=float(args.vmax))
    else:
        vmin, vmax = compute_ratio_color_limits(finite_values)

    colorbar_label = (
        f"{args.label_a}/{args.label_b} · {args.metric} "
        f"(L|R paired; midline summed)"
    )
    render_painted_ratio_slice_png(
        painted,
        atlas_slice.image,
        output_path,
        vmin=vmin,
        vmax=vmax,
        vcenter=1.0,
        cmap_name=str(args.cmap),
        pixel_scale=int(args.pixel_scale),
        supersample=int(args.supersample),
        contour_smooth=float(args.contour_smooth),
        brain_outline_width=float(args.brain_outline_width),
        colorbar_label=colorbar_label,
    )
    print(f"Wrote heatmap: {output_path}")
    print(f"  pairing_reference={pairing_path}")
    print(f"  slice={atlas_slice.coordinate_label}")
    print(f"  color limits=[{vmin:g}, {vmax:g}] (center=1)")

    summary_path = Path(args.summary_json) if args.summary_json else output_path.with_suffix(".json")
    summary = {
        "excel_a": str(excel_a),
        "excel_b": str(excel_b),
        "pairing_reference": str(pairing_path),
        "atlas_label": str(atlas_label),
        "metric": str(args.metric),
        "pseudocount": float(args.pseudocount),
        "bregma_mm": float(args.bregma_mm),
        "coordinate_label": atlas_slice.coordinate_label,
        "vmin": float(vmin),
        "vmax": float(vmax),
        "vcenter": 1.0,
        "output": str(output_path),
        "n_regions_left": len(left_ratios),
        "n_regions_right": len(right_ratios),
        "n_paired_in_reference": sum(1 for paired in paired_by_region_id.values() if paired),
        "n_unpaired_in_reference": sum(1 for paired in paired_by_region_id.values() if not paired),
        "brain_outline_width": float(args.brain_outline_width),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
