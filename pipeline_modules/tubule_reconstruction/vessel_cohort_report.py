"""Per-sample cohort vessel report: whole-brain + hippocampus summary sheets,
distribution figures (delegated to vessel_analysis_report builders), and an
atlas-coronal hippocampal vessel-volume-density heatmap.

Inputs are the six vessel-stat files of one sample (region_vessel_summary.*,
skeleton_*, vessel_branch_metrics.csv, vessel_network_summary.json) plus the
sample's upsampled atlas-label Zarr for tissue volumes / region mapping /
coronal heatmap. Runs without the label Zarr too (region-summary-only mode:
no tissue volumes, no branch->region mapping, no heatmap).

Usage:
    python -m pipeline_modules.tubule_reconstruction.vessel_cohort_report \
        --tubule-dir "<sample tubule_reconstruction dir>" \
        --label-zarr "<sample>/upsampled_atlas_label.zarr" \
        --sample-id 16week_dbdb2 --output-dir <out> \
        [--metrics length,diameter,degree,tortuosity] [--stride 8]
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from pipeline_modules.tubule_reconstruction.vessel_analysis_report import (
    bin_degree_histogram,
    build_diameter_rows,
    build_length_rows,
    build_tortuosity_length_rows,
    is_edt_polyline_edges,
    is_express_branch_metrics,
    load_express_branch_table,
    normalize_branch_table,
    plot_diameter_distribution,
    plot_length_distribution,
    plot_length_fraction_distribution,
    plot_branch_degree_distribution,
    plot_tortuosity_length_fraction,
    stream_vertices,
    stream_vertex_stats_from_edges,
)

HPF_ROOT_ID = 1089  # Allen "Hippocampal formation"


def read_structures(csv_path: Path) -> dict[int, dict]:
    structures: dict[int, dict] = {}
    with open(csv_path, encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            try:
                sid = int(row["id"])
            except (TypeError, ValueError):
                continue
            structures[sid] = {
                "acronym": row.get("acronym", ""),
                "name": row.get("name", ""),
                "path": row.get("structure_id_path", ""),
            }
    return structures


def subtree_ids(structures: dict[int, dict], root_id: int) -> set[int]:
    needle = f"/{root_id}/"
    ids = {root_id}
    ids.update(sid for sid, info in structures.items() if needle in info["path"])
    return ids


def load_region_rows(tubule_dir: Path) -> list[dict]:
    with open(tubule_dir / "region_vessel_summary.csv", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def coarse_label_pass(label_zarr: Path, stride: int) -> dict:
    """One strided pass over the label Zarr.

    Returns a coarse label grid (one label per stride^3 cell), per-label sample
    counts, and the label vocabulary. Chunk sizes in these Zarrs are multiples
    of 8 so stride-aligned sampling maps one-to-one onto grid cells.
    """
    group = zarr.open_group(str(label_zarr), mode="r")
    key = "0" if "0" in group.array_keys() else next(iter(group.array_keys()))
    array = group[key]
    shape = array.shape
    chunks = array.chunks
    grid = np.zeros(tuple(-(-dim // stride) for dim in shape), dtype=np.uint32)
    counts: dict[int, int] = {}
    chunk_rows = -(-shape[0] // chunks[0])
    chunk_cols = -(-shape[1] // chunks[1])
    chunk_depths = -(-shape[2] // chunks[2])
    for cz in range(chunk_rows):
        z0, z1 = cz * chunks[0], min((cz + 1) * chunks[0], shape[0])
        for cy in range(chunk_cols):
            y0, y1 = cy * chunks[1], min((cy + 1) * chunks[1], shape[1])
            for cx in range(chunk_depths):
                x0, x1 = cx * chunks[2], min((cx + 1) * chunks[2], shape[2])
                block = np.asarray(array[z0:z1, y0:y1, x0:x1])[::stride, ::stride, ::stride]
                grid[z0 // stride:(z0 + block.shape[0] * stride) // stride,
                     y0 // stride:(y0 + block.shape[1] * stride) // stride,
                     x0 // stride:(x0 + block.shape[2] * stride) // stride] = block
                labels, label_counts = np.unique(block, return_counts=True)
                for label, count in zip(labels.tolist(), label_counts.tolist()):
                    if label:
                        counts[label] = counts.get(label, 0) + count
        print(f"  label pass: z chunk {cz + 1}/{chunk_rows}", flush=True)
    return {"grid": grid, "counts": counts, "stride": stride, "shape": shape}


def region_tissue_volumes(counts: dict[int, int], stride: int, voxel_volume_um3: float) -> dict[int, float]:
    scale = (stride ** 3) * voxel_volume_um3
    return {label: count * scale for label, count in counts.items()}


def map_branches_to_label(branch_table: pd.DataFrame, grid: np.ndarray, stride: int,
                          resolution_xyz) -> np.ndarray:
    """Region label at each branch midpoint via the coarse grid (nearest cell)."""
    mid_z = (branch_table["source_z_um"].to_numpy(dtype=np.float64)
             + branch_table["target_z_um"].to_numpy(dtype=np.float64)) / 2.0
    mid_y = (branch_table["source_y_um"].to_numpy(dtype=np.float64)
             + branch_table["target_y_um"].to_numpy(dtype=np.float64)) / 2.0
    mid_x = (branch_table["source_x_um"].to_numpy(dtype=np.float64)
             + branch_table["target_x_um"].to_numpy(dtype=np.float64)) / 2.0
    res_z, res_y, res_x = resolution_xyz[2], resolution_xyz[1], resolution_xyz[0]
    cells = np.stack([
        np.clip(np.round(mid_z / (res_z * stride)).astype(np.int64), 0, grid.shape[0] - 1),
        np.clip(np.round(mid_y / (res_y * stride)).astype(np.int64), 0, grid.shape[1] - 1),
        np.clip(np.round(mid_x / (res_x * stride)).astype(np.int64), 0, grid.shape[2] - 1),
    ], axis=1)
    labels = grid[cells[:, 0], cells[:, 1], cells[:, 2]].astype(np.int64)
    empty = labels == 0
    if empty.any():
        pad = 1
        for index in np.nonzero(empty)[0]:
            z, y, x = cells[index]
            neighborhood = grid[max(0, z - pad):z + pad + 1,
                                 max(0, y - pad):y + pad + 1,
                                 max(0, x - pad):x + pad + 1]
            nonzero = neighborhood[neighborhood > 0]
            if nonzero.size:
                values, value_counts = np.unique(nonzero, return_counts=True)
                labels[index] = int(values[np.argmax(value_counts)])
    return labels


def branch_metrics(branch_table: pd.DataFrame) -> dict:
    length = pd.to_numeric(branch_table["branch_length_um"], errors="coerce").to_numpy(dtype=np.float64)
    length = np.where(np.isfinite(length), length, 0.0)
    radius = pd.to_numeric(branch_table["mean_radius_um"], errors="coerce").to_numpy(dtype=np.float64)
    tortuosity = pd.to_numeric(branch_table["tortuosity"], errors="coerce").to_numpy(dtype=np.float64)
    valid_radius = np.isfinite(radius) & (radius > 0)
    valid_tort = np.isfinite(tortuosity) & (tortuosity >= 1.0)
    return {
        "血管总长度_um": float(length.sum()),
        "血管段数": int(len(branch_table)),
        "平均直径_um": float(np.nanmean(radius[valid_radius]) * 2.0) if valid_radius.any() else float("nan"),
        "血管体积_圆柱法_um3": float(np.sum(np.pi * radius[valid_radius] ** 2 * length[valid_radius])),
        "总表面积_圆柱法_um2": float(np.sum(2.0 * np.pi * radius[valid_radius] * length[valid_radius])),
        "扭曲度_最小": float(np.nanmin(tortuosity[valid_tort])) if valid_tort.any() else float("nan"),
        "扭曲度_最大": float(np.nanmax(tortuosity[valid_tort])) if valid_tort.any() else float("nan"),
        "扭曲度_平均": float(np.nanmean(tortuosity[valid_tort])) if valid_tort.any() else float("nan"),
        "终末段数": int(branch_table["is_terminal_branch"].fillna(False).astype(bool).sum())
        if "is_terminal_branch" in branch_table.columns else None,
        "血管环数": int(branch_table["is_loop"].fillna(False).astype(bool).sum())
        if "is_loop" in branch_table.columns else None,
    }


def render_density_heatmap(tissue_volumes: dict[int, float], leaf_by_id: dict[int, dict],
                           output_dir: Path, figures_dir: Path) -> list[Path]:
    """Call the existing vessel_region_density_heatmap program (atlas-space,
    dpi-300 HD coronal slices). Empty regions are prefilled with 0 via a root
    row; leaf rows then overwrite with their real density."""
    density_csv = output_dir / "region_density_for_heatmap.csv"
    # root row prefills every label with 0 (empty regions read as 0); leaf rows
    # then overwrite with real densities. Queries use numeric region ids: some
    # acronyms are digit strings ("5") which the resolver treats as ids.
    rows = [{"query": "root", "density_vessel_fraction": 0.0}]
    percents = []
    for label, tissue_vol in tissue_volumes.items():
        row = leaf_by_id.get(int(label))
        if row is None:
            continue
        vessel_vol = float(row.get("vessel_volume_um3") or 0.0)
        fraction = vessel_vol / tissue_vol if tissue_vol > 0 else 0.0
        rows.append({"query": int(label), "density_vessel_fraction": fraction})
        if fraction > 0:
            percents.append(fraction * 100.0)
    pd.DataFrame(rows).to_csv(density_csv, index=False)

    # color ceiling ~95th percentile keeps the bulk of regions in a readable
    # range; a few tiny over-dense leaf regions (coarse tissue estimate) clip
    vmax = float(np.percentile(percents, 95)) if percents else 6.0
    vmax = max(min(vmax, 15.0), 1.0)
    heat_dir = figures_dir / "coronal_heatmap"
    cmd = [
        sys.executable, "-m", "pipeline_modules.tubule_reconstruction.vessel_region_density_heatmap",
        "--density_csv", str(density_csv),
        "--output_dir", str(heat_dir),
        "--cfg", str(Path(__file__).resolve().parents[1] / "registration" / "Region_Csv_Rev1_updated.CSV"),
        "--dpi", "300",
        "--cmap", "inferno",
        "--vmax_percent", f"{vmax:.2f}",
    ]
    repo_root = Path(__file__).resolve().parents[2]
    print(f"Rendering atlas coronal density heatmap (dpi 300, vmax {vmax:.2f}%) ...", flush=True)
    result = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("vessel_region_density_heatmap failed")
    slices = sorted(heat_dir.glob("bregma_*.png"))
    print(f"  {len(slices)} coronal slices written to {heat_dir}")
    return slices


def sheet_rows(pairs: list[tuple[str, object]]) -> pd.DataFrame:
    return pd.DataFrame([{"指标": name, "数值": value} for name, value in pairs])


def fmt_um3_to_mm3(value):
    return None if value is None else float(value) / 1e9


def main() -> int:
    parser = argparse.ArgumentParser(description="Whole-brain + hippocampus vessel cohort report for one sample")
    parser.add_argument("--tubule-dir", required=True, help="Sample tubule_reconstruction directory")
    parser.add_argument("--label-zarr", default="", help="upsampled_atlas_label.zarr (optional)")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resolution-xyz", default="1.8,1.8,2.0")
    parser.add_argument("--stride", type=int, default=8, help="Sampling stride for the label pass")
    parser.add_argument("--metrics", default="length,diameter,degree,tortuosity",
                        help="Sections: length,length_fraction,diameter,degree,tortuosity")
    parser.add_argument("--structures-csv", default="", help="Allen structures CSV (defaults to repo copy)")
    args = parser.parse_args()

    tubule_dir = Path(args.tubule_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    resolution = tuple(float(part) for part in args.resolution_xyz.split(","))
    voxel_volume = float(np.prod(resolution))
    structures_csv = Path(args.structures_csv) if args.structures_csv else \
        Path(__file__).resolve().parents[2] / "allen_mouse_25um_structures.csv"
    structures = read_structures(structures_csv)
    hpf_ids = subtree_ids(structures, HPF_ROOT_ID)

    metrics = set(part for part in args.metrics.split(",") if part)

    # --- region summary rows ---
    region_rows = load_region_rows(tubule_dir)
    by_acronym = {row["region_acronym"]: row for row in region_rows}
    leaf_by_id = {int(row["region_id"]): row for row in region_rows}
    root_row = by_acronym.get("root", {})
    hpf_row = by_acronym.get("HPF", {})
    whole_vessel_vol = float(root_row.get("vessel_volume_um3") or 0.0)
    hpf_vessel_vol = float(hpf_row.get("vessel_volume_um3") or 0.0)

    # --- label pass ---
    label_result = None
    tissue_volumes = None
    if args.label_zarr:
        print(f"Label pass over {args.label_zarr} (stride {args.stride}) ...", flush=True)
        label_result = coarse_label_pass(Path(args.label_zarr), args.stride)
        tissue_volumes = region_tissue_volumes(label_result["counts"], args.stride, voxel_volume)
        whole_tissue_vol = float(sum(tissue_volumes.values()))
        hpf_tissue_vol = float(sum(vol for label, vol in tissue_volumes.items() if label in hpf_ids))
        print(f"  brain tissue {whole_tissue_vol/1e9:.1f} mm3 | HPF tissue {hpf_tissue_vol/1e9:.2f} mm3")
    else:
        whole_tissue_vol = None
        hpf_tissue_vol = None

    # --- branch table ---
    edge_csv = tubule_dir / "skeleton_edges.csv"
    branch_csv = tubule_dir / "vessel_branch_metrics.csv"
    if is_edt_polyline_edges(edge_csv):
        raise SystemExit("EDT polyline edges not supported by the cohort report; use vessel_express runs")
    if is_express_branch_metrics(branch_csv):
        branch_table = load_express_branch_table(branch_csv)
    else:
        branch_table = normalize_branch_table(pd.read_csv(branch_csv, low_memory=False))

    whole_metrics = branch_metrics(branch_table)

    # --- vertex stats (branch points / endpoints / degree histogram) ---
    vertex_stats = stream_vertices(tubule_dir / "skeleton_vertices.csv")
    if vertex_stats is None:
        vertex_stats = stream_vertex_stats_from_edges(edge_csv)

    # --- hippocampus branch subset ---
    hpf_metrics = None
    branch_region_labels = None
    if label_result is not None:
        print("Mapping branch midpoints to atlas regions ...", flush=True)
        branch_region_labels = map_branches_to_label(branch_table, label_result["grid"],
                                                     label_result["stride"], resolution)
        in_hpf = np.isin(branch_region_labels, np.array(sorted(hpf_ids), dtype=np.int64))
        print(f"  {int(in_hpf.sum())} / {len(branch_table)} branches mapped into HPF")
        hpf_metrics = branch_metrics(branch_table.loc[in_hpf].reset_index(drop=True))

    # --- whole-brain summary sheet ---
    wb_rows = [
        ("样本", args.sample_id),
        ("脑组织体积_mm3", fmt_um3_to_mm3(whole_tissue_vol)),
        ("血管总体积_区域法_mm3", whole_vessel_vol / 1e9),
        ("血管总体积_圆柱法_mm3", whole_metrics["血管体积_圆柱法_um3"] / 1e9),
        ("血管总长度_mm", whole_metrics["血管总长度_um"] / 1e3),
        ("血管段数", whole_metrics["血管段数"]),
        ("分支点数", vertex_stats["num_branch_points"]),
        ("终末点数", vertex_stats["num_endpoints"]),
        ("终末点数_非边界", vertex_stats["num_endpoints_non_boundary"]),
        ("终末段数", whole_metrics["终末段数"]),
        ("扭曲度_最小", whole_metrics["扭曲度_最小"]),
        ("扭曲度_最大", whole_metrics["扭曲度_最大"]),
        ("扭曲度_平均", whole_metrics["扭曲度_平均"]),
        ("总表面积_圆柱法_mm2", whole_metrics["总表面积_圆柱法_um2"] / 1e6),
        ("血管环数", whole_metrics["血管环数"]),
        ("平均直径_um", whole_metrics["平均直径_um"]),
        ("血管体积占脑比_%", (whole_vessel_vol / whole_tissue_vol * 100.0) if whole_tissue_vol else None),
    ]

    hpf_rows = [
        ("样本", args.sample_id),
        ("海马组织体积_mm3", fmt_um3_to_mm3(hpf_tissue_vol)),
        ("海马血管体积_区域法_mm3", hpf_vessel_vol / 1e9),
    ]
    if hpf_metrics is not None:
        hpf_rows += [
            ("海马血管体积_圆柱法_mm3", hpf_metrics["血管体积_圆柱法_um3"] / 1e9),
            ("海马血管总长度_mm", hpf_metrics["血管总长度_um"] / 1e3),
            ("海马血管段数", hpf_metrics["血管段数"]),
            ("海马分支点数_区域法", int(hpf_row.get("num_branch_points") or 0)),
            ("海马终末段数", hpf_metrics["终末段数"]),
            ("海马扭曲度_最小", hpf_metrics["扭曲度_最小"]),
            ("海马扭曲度_最大", hpf_metrics["扭曲度_最大"]),
            ("海马扭曲度_平均", hpf_metrics["扭曲度_平均"]),
            ("海马总表面积_圆柱法_mm2", hpf_metrics["总表面积_圆柱法_um2"] / 1e6),
            ("海马血管环数", hpf_metrics["血管环数"]),
            ("海马平均直径_um", hpf_metrics["平均直径_um"]),
            ("海马血管体积占海马比_%",
             (hpf_vessel_vol / hpf_tissue_vol * 100.0) if hpf_tissue_vol else None),
        ]
    else:
        hpf_rows += [
            ("海马分支点数_区域法", int(hpf_row.get("num_branch_points") or 0)),
            ("说明", "未提供 label zarr：无组织体积 / 分支级海马统计"),
        ]

    # --- distributions (delegated to vessel_analysis_report builders) ---
    tables: dict[str, pd.DataFrame] = {}
    figure_placements: list[tuple[str, Path, str]] = []
    if "length" in metrics or "length_fraction" in metrics:
        length_df = build_length_rows(branch_table, branch_to_branch_only=True)
        tables["长度分布"] = length_df
        plot_length_distribution(length_df, figures_dir / "length_distribution.png")
        figure_placements.append(("长度分布", figures_dir / "length_distribution.png", "A20"))
        if "length_fraction" in metrics:
            plot_length_fraction_distribution(length_df, figures_dir / "length_fraction_distribution.png")
            figure_placements.append(("长度分布", figures_dir / "length_fraction_distribution.png", "A40"))
    if "diameter" in metrics:
        diam_df = build_diameter_rows(branch_table)
        tables["直径分布"] = diam_df
        plot_diameter_distribution(diam_df, figures_dir / "diameter_distribution.png")
        figure_placements.append(("直径分布", figures_dir / "diameter_distribution.png", "A20"))
    if "degree" in metrics:
        degree_df = pd.DataFrame(
            [{"连接血管数": label, "数量": count}
             for label, count in bin_degree_histogram(vertex_stats["branch_point_degree_histogram"])]
        )
        tables["分支点连接数"] = degree_df
        plot_branch_degree_distribution(vertex_stats["branch_point_degree_histogram"],
                                        figures_dir / "branch_point_degree_distribution.png")
        figure_placements.append(("分支点连接数", figures_dir / "branch_point_degree_distribution.png", "A20"))
    if "tortuosity" in metrics:
        tort_len_df = build_tortuosity_length_rows(branch_table)
        tables["扭曲度长度占比"] = tort_len_df
        plot_tortuosity_length_fraction(tort_len_df, figures_dir / "tortuosity_length_share.png")
        figure_placements.append(("扭曲度长度占比", figures_dir / "tortuosity_length_share.png", "A20"))

    # --- coronal heatmap ---
    density_table = None
    heatmap_slices: list[Path] = []
    if label_result is not None:
        density_rows = []
        for label, tissue_vol in tissue_volumes.items():
            row = leaf_by_id.get(int(label))
            if row is None or tissue_vol <= 0:
                continue
            vessel_vol = float(row.get("vessel_volume_um3") or 0.0)
            density = vessel_vol / tissue_vol * 100.0
            info = structures.get(int(label), {})
            density_rows.append({
                "region_id": int(label),
                "缩写": row.get("region_acronym", info.get("acronym", "")),
                "名称": row.get("region_name", info.get("name", "")),
                "组织体积_mm3": tissue_vol / 1e9,
                "血管体积_mm3": vessel_vol / 1e9,
                "血管体积密度_%": density,
            })
        density_table = pd.DataFrame(density_rows).sort_values("血管体积密度_%", ascending=False)
        heatmap_slices = render_density_heatmap(tissue_volumes, leaf_by_id, output_dir, figures_dir)

    # --- workbook ---
    from openpyxl.drawing.image import Image as XLImage

    xlsx_path = output_dir / f"vessel_cohort_report_{args.sample_id}.xlsx"
    try:
        with open(xlsx_path, "a", encoding="utf-8"):
            pass
    except OSError:
        # locked, e.g. open in Excel: fall back instead of failing the whole report
        xlsx_path = output_dir / f"vessel_cohort_report_{args.sample_id}_new.xlsx"
        print(f"NOTE: primary workbook is locked (open in Excel?); writing {xlsx_path.name}")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        sheet_rows(wb_rows).to_excel(writer, sheet_name="全脑汇总", index=False)
        sheet_rows(hpf_rows).to_excel(writer, sheet_name="海马汇总", index=False)
        for name, frame in tables.items():
            frame.to_excel(writer, sheet_name=name, index=False)
        if density_table is not None:
            density_table.to_excel(writer, sheet_name="区域密度", index=False)
        book = writer.book
        for sheet_name, png, anchor in figure_placements:
            book[sheet_name].add_image(XLImage(str(png)), anchor)
        if heatmap_slices and "区域密度" in book.sheetnames:
            sheet = book["区域密度"]
            anchor_row = 2
            for png in heatmap_slices:
                sheet.add_image(XLImage(str(png)), f"H{anchor_row}")
                anchor_row += 42
    print(f"Workbook written to {xlsx_path}")

    summary = {
        "sample_id": args.sample_id,
        "whole_brain": {name: (None if value is None else float(value) if isinstance(value, (int, float)) else value)
                        for name, value in wb_rows},
        "hippocampus": {name: (None if value is None else float(value) if isinstance(value, (int, float)) else value)
                        for name, value in hpf_rows},
        "label_pass": None if label_result is None else {
            "stride": label_result["stride"],
            "labels_found": len(label_result["counts"]),
        },
    }
    (output_dir / "cohort_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
