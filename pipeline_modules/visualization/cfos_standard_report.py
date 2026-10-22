"""Standard two-sample c-Fos descriptive report (HTML + PDF + Excel + figures).

Follows the "光片全脑 c-Fos 分析交付报告" spec (v1.0, 2026-10-11):
overview page + topic pages, fixed mutually exclusive default region set
(the same parcellation as top_level7_density_ratio --level auto), signed
density difference ranking, absolute-threshold 3D density isosurfaces with a
shared camera, atlas-space coronal slices, and an explicit missing-inputs
list instead of simulated data.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

from pipeline_modules.visualization.top_level7_density_ratio import (
    DEFAULT_PARCELLATION_CSV,
    build_default_parcellation,
    load_structure_tree,
)

ATLAS_VOXEL_UM = 25.0
ATLAS_VOXEL_MM3 = (ATLAS_VOXEL_UM ** 3) / 1e9
DEFAULT_REPORT_DIRNAME = "report_{a}_vs_{b}"
DEFAULT_COARSE_RAW_NAMES = [
    "Isocortex,Isocortex",
    "Hippocampal formation,HPF",
    "Olfactory areas,OLF",
    "Cortical subplate,CTXsp",
    "Striatum,STR",
    "Pallidum,PAL",
    "Thalamus,TH",
    "Hypothalamus,HY",
    "Midbrain,MB",
    "Pons,P",
    "Medulla,MY",
    "Cerebellum,CB",
    "fiber tracts,fiber tracts",
    "ventricular systems,VS",
]
COARSE_DISPLAY = {
    "Isocortex": "新皮层", "Hippocampal formation": "海马结构", "Olfactory areas": "嗅区",
    "Cortical subplate": "皮质下板", "Striatum": "纹状体", "Pallidum": "苍白球",
    "Thalamus": "丘脑", "Hypothalamus": "下丘脑", "Midbrain": "中脑", "Pons": "脑桥",
    "Medulla": "延髓", "Cerebellum": "小脑", "fiber tracts": "纤维束", "ventricular systems": "脑室",
}
DIFF_CMAP_NEG, DIFF_CMAP_POS = "#2f6f9f", "#b84a4a"
DENSITY_CMAP = "viridis"


@dataclasses.dataclass
class SampleInputs:
    sample_id: str
    sample_dir: Path
    workbook: Path
    count_volume: Path | None
    config: dict
    voxel_um3: float
    qc_images: list[Path]
    missing: list[str]


def find_sample_workbook(sample_dir: str | Path) -> Path:
    sample_dir = Path(sample_dir)
    candidates = sorted(sample_dir.glob("results/*.xlsx")) + sorted(sample_dir.glob("*_result.xlsx"))
    for path in candidates:
        if path.name.startswith("~$"):
            continue
        try:
            if any(str(s).startswith("Level_") for s in pd.ExcelFile(path).sheet_names):
                return path
        except Exception:
            continue
    raise FileNotFoundError(f"No region workbook (Level_* sheets) under {sample_dir} or its results/")


def find_count_volume(sample_dir: str | Path) -> Path | None:
    sample_dir = Path(sample_dir)
    for path in sorted(sample_dir.glob("visualization/*count_volume.tif")) + sorted(sample_dir.glob("visualization/*_count_volume.tiff")):
        return path
    return None


def load_sample_config(sample_dir: str | Path) -> dict:
    config_path = Path(sample_dir) / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_voxel_um3(config: dict) -> float:
    resolution = (config.get("input") or {}).get("resolution_xyz")
    if not resolution or len(resolution) != 3:
        raise ValueError("config.json input.resolution_xyz missing; cannot derive native voxel volume")
    return float(np.prod([float(v) for v in resolution]))


def load_workbook_metrics(workbook: str | Path) -> dict[str, dict[str, float]]:
    """Name -> {count, volume_voxels} summed over all Level_* sheets."""

    sheet_names = [s for s in pd.ExcelFile(workbook).sheet_names if str(s).startswith("Level_")]
    if not sheet_names:
        raise ValueError(f"No Level_* sheets in {workbook}")
    metrics: dict[str, dict[str, float]] = {}
    for sheet in sheet_names:
        frame = pd.read_excel(workbook, sheet_name=sheet)
        for name, count, voxels in zip(
            frame["Name"],
            pd.to_numeric(frame["Signal Count"], errors="coerce").fillna(0.0),
            pd.to_numeric(frame["Total Voxels"], errors="coerce").fillna(0.0),
        ):
            row = metrics.setdefault(str(name), {"count": 0.0, "volume_voxels": 0.0})
            row["count"] += float(count)
            row["volume_voxels"] += float(voxels)
    return metrics


def parcellation_table(
    metrics: dict[str, dict[str, float]],
    voxel_um3: float,
    parcellation: list[dict] | None = None,
) -> pd.DataFrame:
    """Default analysis layer: exclusive regions with remainder math on count AND volume."""

    parcellation = parcellation if parcellation is not None else build_default_parcellation()
    rows = []
    for region in parcellation:
        entry = metrics.get(region["name"], {"count": 0.0, "volume_voxels": 0.0})
        count = entry["count"]
        volume = entry["volume_voxels"]
        children = region["remainder_children"]
        if children is not None:
            count -= sum(metrics.get(child, {}).get("count", 0.0) for child in children)
            volume -= sum(metrics.get(child, {}).get("volume_voxels", 0.0) for child in children)
        count = max(0.0, count)
        volume = max(0.0, volume)
        volume_mm3 = volume * voxel_um3 / 1e9
        density = count / volume_mm3 if volume_mm3 > 0 else np.nan
        rows.append(
            {
                "region_name": region["name"],
                "acronym": region["acronym"],
                "region_id": region["id"],
                "count": count,
                "volume_mm3": volume_mm3,
                "density_cells_per_mm3": density,
                "is_remainder": children is not None,
            }
        )
    return pd.DataFrame(rows)


def coarse_table(metrics: dict[str, dict[str, float]], voxel_um3: float) -> pd.DataFrame:
    rows = []
    for raw_name in DEFAULT_COARSE_RAW_NAMES:
        entry = metrics.get(raw_name)
        if entry is None:
            continue
        volume_mm3 = entry["volume_voxels"] * voxel_um3 / 1e9
        density = entry["count"] / volume_mm3 if volume_mm3 > 0 else np.nan
        label = raw_name.rsplit(",", 1)[0]
        rows.append(
            {
                "system_en": label,
                "system_zh": COARSE_DISPLAY.get(label, label),
                "count": entry["count"],
                "volume_mm3": volume_mm3,
                "density_cells_per_mm3": density,
            }
        )
    return pd.DataFrame(rows)


def build_comparison(
    table_a: pd.DataFrame,
    table_b: pd.DataFrame,
    *,
    min_volume_mm3: float,
    low_count_warning: float,
) -> pd.DataFrame:
    merged = table_a.merge(table_b, on="region_name", suffixes=("_a", "_b"))
    merged["acronym"] = merged["acronym_a"]
    merged["delta_density"] = merged["density_cells_per_mm3_b"] - merged["density_cells_per_mm3_a"]
    merged["delta_count"] = merged["count_b"] - merged["count_a"]
    merged["abs_delta_density"] = merged["delta_density"].abs()
    merged["qc_usable"] = (merged["volume_mm3_a"] >= min_volume_mm3) & (merged["volume_mm3_b"] >= min_volume_mm3)
    merged["low_count"] = (merged["count_a"] < low_count_warning) | (merged["count_b"] < low_count_warning)
    return merged


def rank_top_regions(comparison: pd.DataFrame, top_n: int) -> pd.DataFrame:
    usable = comparison[comparison["qc_usable"]].copy()
    usable = usable.sort_values(["abs_delta_density", "region_name"], ascending=[False, True], kind="stable")
    usable["rank"] = np.arange(1, len(usable) + 1)
    return usable.head(top_n)


def system_of_regions(csv_path: str | Path = DEFAULT_PARCELLATION_CSV) -> dict[str, str]:
    """Default-layer region name -> coarse system label (概览层归属)."""

    coarse_acronyms = {raw.rsplit(",", 1)[1]: raw.rsplit(",", 1)[0] for raw in DEFAULT_COARSE_RAW_NAMES}
    nodes, _root = load_structure_tree(csv_path)
    mapping: dict[str, str] = {}
    for region in build_default_parcellation(csv_path):
        node = nodes.get(region["id"])
        while node is not None:
            if node["acronym"] in coarse_acronyms:
                mapping[region["name"]] = coarse_acronyms[node["acronym"]]
                break
            node = nodes.get(node["parent"]) if node["parent"] is not None else None
    return mapping


def load_density_fraction_grid(
    count_volume_path: str | Path,
    *,
    voxel_um3: float,
    bin_um: float = 100.0,
    sigma_um: float = 150.0,
) -> np.ndarray:
    """Count tif -> smoothed signal-volume-fraction grid, axes (AP, LR, SI).

    Per atlas voxel the count tif holds warped native mask-voxel counts;
    converting per-voxel to an occupancy fraction (native mask volume /
    atlas voxel volume, capped at 1) BEFORE binning keeps the quantity a
    true volume fraction (never above 100% regardless of interpolation).
    """

    counts = tifffile.imread(str(count_volume_path)).astype(np.float32)
    support = counts > 0
    aligned = _annotation_aligned(counts)
    if aligned is not None:
        support = support & aligned[0]

    occupancy = np.minimum(counts * voxel_um3 / ATLAS_VOXEL_UM**3, 1.0)
    factor = int(round(bin_um / ATLAS_VOXEL_UM))
    binned = _block_sum(occupancy, factor) / float(factor**3)
    support_binned = _block_any(support, factor)

    from scipy.ndimage import gaussian_filter

    smoothed = gaussian_filter(binned, sigma_um / bin_um)
    smoothed[~support_binned] = 0.0  # missing coverage stays missing, not zero-density tissue
    return np.transpose(smoothed, _AP_LR_SI_ORDER(counts))


def _block_sum(volume: np.ndarray, factor: int) -> np.ndarray:
    t = volume[: (volume.shape[0] // factor) * factor, : (volume.shape[1] // factor) * factor, : (volume.shape[2] // factor) * factor]
    return t.reshape(t.shape[0] // factor, factor, t.shape[1] // factor, factor, t.shape[2] // factor, factor).sum(axis=(1, 3, 5))


def _block_any(mask: np.ndarray, factor: int) -> np.ndarray:
    return _block_sum(mask.astype(np.float32), factor) > 0


def _annotation_aligned(counts: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]] | None:
    """Annotation permuted onto the count grid, chosen by real support overlap.

    BrainGlobe annotation axes are (AP, SI, LR). Returns (aligned_mask,
    permutation) where permutation[i] tells which annotation axis landed on
    count-grid axis i.
    """

    try:
        from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas

        annotation = np.asarray(BrainGlobeAtlas("allen_mouse_25um", check_latest=False).annotation) > 0
    except Exception:
        return None
    support = counts > 0
    best = None
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (2, 0, 1), (1, 2, 0), (2, 1, 0)):
        candidate = np.transpose(annotation, perm)
        if candidate.shape != counts.shape:
            continue
        overlap = float((candidate & support).sum())
        if best is None or overlap > best[2]:
            best = (candidate, perm, overlap)
    if best is None:
        return None
    return best[0], best[1]


def _AP_LR_SI_ORDER(counts: np.ndarray) -> tuple[int, int, int]:
    """Count-grid axis indices for (AP, LR, SI), from annotation alignment.

    Annotation perm[i] is the annotation axis on grid axis i; annotation
    axes are (AP=0, SI=1, LR=2).
    """

    aligned = _annotation_aligned(counts)
    if aligned is None:
        return (0, 1, 2)
    _mask, perm = aligned
    inverse = np.argsort(perm)  # inverse[ann_axis] = grid axis
    return int(inverse[0]), int(inverse[2]), int(inverse[1])


def derive_iso_thresholds(grid: np.ndarray) -> tuple[float, float, float]:
    """Fixed absolute thresholds derived once from sample A, rounded to 2 digits."""

    nonzero = grid[grid > 0]
    if nonzero.size == 0:
        return (0.002, 0.005, 0.015)
    mid = max(float(np.percentile(nonzero, 97)), 1e-4)

    def round2(value: float) -> float:
        exponent = np.floor(np.log10(max(value, 1e-9)))
        return float(round(value / 10**exponent, 1) * 10**exponent)

    return round2(mid / 2.0), round2(mid), round2(mid * 2.0)


ISO_LEVELS = (("#414487", 0.10), ("#21918c", 0.35), ("#fde725", 0.9))


def render_density_isosurfaces(
    grid: np.ndarray,
    *,
    thresholds: tuple[float, float, float],
    bin_um: float,
    output_path: str | Path,
    camera_view_path: str | Path | None = None,
) -> Path:
    """Nested density isosurfaces at fixed absolute thresholds inside the Allen brain."""

    from pipeline_modules.visualization import render_points_brainrender as rpb

    rpb.configure_brainrender(background="white", root_alpha=0.10, show_axes=False, offscreen=True)
    from brainrender import Scene
    import vedo

    scene = Scene(root=True, atlas_name="allen_mouse_25um", check_latest=False, inset=False)
    volume = vedo.Volume(np.ascontiguousarray(grid, dtype=np.float32))
    volume.spacing([float(bin_um)] * 3)
    for (color, alpha), value in zip(ISO_LEVELS, thresholds):
        if value <= 0:
            continue
        mesh = volume.isosurface(value=float(value))
        mesh.c(color)
        mesh.alpha(alpha)
        scene.add(mesh)
    camera = None
    if camera_view_path and Path(camera_view_path).exists():
        camera = json.loads(Path(camera_view_path).read_text(encoding="utf-8"))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved = rpb.save_scene_screenshot(scene, output_path, camera=camera, screenshot_scale=1)
    scene.close()
    return Path(saved)


def coronal_slice_figure(
    grid_a: np.ndarray,
    grid_b: np.ndarray,
    *,
    bin_um: float,
    vmax: float,
    positions_mm: list[float],
    label_a: str,
    label_b: str,
    output_path: str | Path,
) -> Path:
    """Equally spaced coronal density slices; A/B rows share one color scale."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(positions_mm)
    fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 4.2), constrained_layout=True)
    extent = [0.0, grid_a.shape[2] * bin_um / 1000.0, 0.0, grid_a.shape[1] * bin_um / 1000.0]
    for column, ap_mm in enumerate(positions_mm):
        index = min(max(int(round(ap_mm * 1000.0 / bin_um)), 0), grid_a.shape[0] - 1)
        for row, (grid, label) in enumerate(((grid_a, label_a), (grid_b, label_b))):
            ax = axes[row, column]
            ax.imshow(np.rot90(grid[index]), cmap=DENSITY_CMAP, vmin=0.0, vmax=vmax,
                      extent=extent, aspect="equal", origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"AP {ap_mm:.1f} mm", fontsize=9)
            if column == 0:
                ax.set_ylabel(label, fontsize=10)
    sm = plt.cm.ScalarMappable(cmap=DENSITY_CMAP, norm=plt.Normalize(vmin=0.0, vmax=vmax))
    fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.015, label="信号体积占比")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def ap_profile_figure(
    grid_a: np.ndarray,
    grid_b: np.ndarray,
    *,
    bin_um: float,
    label_a: str,
    label_b: str,
    output_path: str | Path,
) -> Path:
    ap_axis = np.arange(grid_a.shape[0]) * bin_um / 1000.0

    def profile(grid: np.ndarray) -> np.ndarray:
        valid = grid > 0
        total = valid.sum(axis=(1, 2))
        sums = np.where(valid, grid, 0.0).sum(axis=(1, 2))
        return np.divide(sums, total, out=np.zeros_like(sums), where=total > 0)

    fig, ax = plt.subplots(figsize=(7.0, 2.7), constrained_layout=True)
    ax.plot(ap_axis, profile(grid_a), color="#2f6f9f", label=label_a, lw=1.8)
    ax.plot(ap_axis, profile(grid_b), color="#b84a4a", label=label_b, lw=1.8)
    ax.set_xlabel("AP 距图谱前缘 (mm)")
    ax.set_ylabel("平均信号体积占比")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    output_path = Path(output_path)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def delta_dot_figure(top: pd.DataFrame, *, label_a: str, label_b: str, output_path: str | Path, title: str) -> Path:
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(7.6, 0.5 * len(top) + 1.6), constrained_layout=True)
    y = np.arange(len(top))
    values = top["delta_density"].to_numpy(float)
    colors = [DIFF_CMAP_POS if v >= 0 else DIFF_CMAP_NEG for v in values]
    ax.axvline(0.0, color="#263238", lw=1.0)
    ax.hlines(y, 0.0, values, color=colors, lw=1.6, alpha=0.65)
    ax.scatter(values, y, s=46, color=colors, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{row.acronym or row.region_name.rsplit(',', 1)[0]} ({int(row.count_a)}/{int(row.count_b)})" for row in top.itertuples()], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"Δ密度 = 密度({label_b}) − 密度({label_a})  (cells/mm³)")
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    abs_max = max(np.abs(values).max(), 1e-9) * 1.25
    ax.set_xlim(-abs_max, abs_max)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def count_density_dual_figure(top: pd.DataFrame, *, label_a: str, label_b: str, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    n = len(top)
    y = np.arange(n)
    fig, (ax_count, ax_density) = plt.subplots(1, 2, figsize=(9.4, 0.42 * n + 1.7), sharey=True, constrained_layout=True)
    labels = [row.acronym or row.region_name.rsplit(",", 1)[0] for row in top.itertuples()]
    counts_a = top["count_a"].to_numpy(float)
    counts_b = top["count_b"].to_numpy(float)
    dens_a = top["density_cells_per_mm3_a"].to_numpy(float)
    dens_b = top["density_cells_per_mm3_b"].to_numpy(float)
    bar = 0.36
    ax_count.barh(y - bar / 2, np.maximum(counts_a, 0.5), bar, color="#2f6f9f", label=label_a)
    ax_count.barh(y + bar / 2, np.maximum(counts_b, 0.5), bar, color="#b84a4a", label=label_b)
    ax_count.set_xscale("symlog", linthresh=10)
    ax_count.set_xlabel("检测计数 (symlog)")
    ax_density.barh(y - bar / 2, np.maximum(dens_a, 1e-6), bar, color="#2f6f9f")
    ax_density.barh(y + bar / 2, np.maximum(dens_b, 1e-6), bar, color="#b84a4a")
    ax_density.set_xlabel("密度 (cells/mm³)")
    for ax in (ax_count, ax_density):
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.25)
    ax_count.set_yticks(y)
    ax_count.set_yticklabels(labels, fontsize=9)
    ax_count.legend(frameon=False, fontsize=9)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def coarse_share_figure(coarse_a: pd.DataFrame, coarse_b: pd.DataFrame, *, label_a: str, label_b: str, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    merged = coarse_a.merge(coarse_b, on=["system_en", "system_zh"], suffixes=("_a", "_b"))
    merged = merged.sort_values("count_a", ascending=False)
    x = np.arange(len(merged))
    fig, ax = plt.subplots(figsize=(9.8, 3.1), constrained_layout=True)
    ax.bar(x - 0.2, merged["count_a"], 0.38, color="#2f6f9f", label=label_a)
    ax.bar(x + 0.2, merged["count_b"], 0.38, color="#b84a4a", label=label_b)
    ax.set_xticks(x)
    ax.set_xticklabels(merged["system_zh"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("检测计数")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def roi_parent_child_figure(
    region_name: str,
    parent_row: pd.Series,
    child_table: pd.DataFrame,
    *,
    label_a: str,
    label_b: str,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    fine = child_table
    n = len(fine)
    fig, (ax_parent, ax_child) = plt.subplots(1, 2, figsize=(9.6, max(1.6, 0.4 * n + 1.5)), constrained_layout=True,
                                              gridspec_kw={"width_ratios": [1, 1.6]})
    ax_parent.bar([0, 1], [parent_row["density_cells_per_mm3_a"], parent_row["density_cells_per_mm3_b"]],
                  color=["#2f6f9f", "#b84a4a"], width=0.6)
    ax_parent.set_xticks([0, 1])
    ax_parent.set_xticklabels([label_a, label_b], fontsize=9)
    ax_parent.set_ylabel("密度 (cells/mm³)")
    ax_parent.set_title(f"父区 {parent_row['acronym'] or region_name.rsplit(',', 1)[0]}", fontsize=10)
    ax_parent.spines[["top", "right"]].set_visible(False)
    if n:
        y = np.arange(n)
        deltas = fine["delta_density"].to_numpy(float)
        colors = [DIFF_CMAP_POS if v >= 0 else DIFF_CMAP_NEG for v in deltas]
        ax_child.axvline(0.0, color="#263238", lw=1.0)
        ax_child.barh(y, deltas, color=colors, alpha=0.85)
        ax_child.set_yticks(y)
        ax_child.set_yticklabels([f"{row.acronym or row.region_name.rsplit(',', 1)[0]} ({int(row.count_a)}/{int(row.count_b)})" for row in fine.itertuples()], fontsize=8.5)
        ax_child.invert_yaxis()
        ax_child.set_xlabel("Δ密度 (cells/mm³)")
        ax_child.set_title("子区 Δ密度（B − A）", fontsize=10)
        ax_child.spines[["top", "right", "left"]].set_visible(False)
        ax_child.grid(axis="x", alpha=0.25)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{{ title }}</title>
<style>
@page { size: A4 landscape; margin: 12mm 10mm; }
* { box-sizing: border-box; }
body { font-family: "Microsoft YaHei", "Noto Sans SC", "Segoe UI", sans-serif;
       color: #17212b; font-size: 10.5pt; margin: 0; background: #fff; }
.page { page-break-after: always; padding: 1mm 0.5mm; }
#p1 { height: 183mm; overflow: hidden; }
.page:last-child { page-break-after: auto; }
h1 { font-size: 15pt; margin: 0 0 1.5mm; }
h2 { font-size: 13pt; margin: 0 0 2mm; border-bottom: 1.5px solid #263238; padding-bottom: 0.8mm; }
h3 { font-size: 11pt; margin: 3mm 0 1.5mm; }
.header { display: flex; justify-content: space-between; align-items: baseline;
          border-bottom: 2px solid #17212b; padding-bottom: 2mm; margin-bottom: 3mm; }
.header .meta { font-size: 9pt; color: #46586a; text-align: right; }
.summary { background: #f2f6f9; border-left: 3.5px solid #2f6f9f; padding: 1.5mm 3mm;
           margin-bottom: 2mm; line-height: 1.35; font-size: 9.5pt; }
.grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 2.5mm; }
.c6 { grid-column: span 6; } .c4 { grid-column: span 4; } .c8 { grid-column: span 8; }
.c12 { grid-column: span 12; } .c5 { grid-column: span 5; } .c3 { grid-column: span 3; }
.panel { border: 1px solid #d8e1e8; border-radius: 2mm; padding: 1.8mm; }
.panel .cap { font-size: 9pt; color: #46586a; margin-top: 1.5mm; line-height: 1.45; }
img.fig { width: 100%; height: auto; display: block; object-fit: contain; }
img.cap3d { max-height: 50mm; }
img.capchart { max-height: 60mm; }
img.caproi { max-height: 60mm; }
img.capmid { max-height: 52mm; }
img.capqc { max-height: 58mm; }
img.caplow { max-height: 30mm; }
table { border-collapse: collapse; width: 100%; font-size: 8.6pt; }
th, td { border: 0.5pt solid #c6d2db; padding: 1.2mm 1.8mm; text-align: left; }
th { background: #eef3f7; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.pos { color: #b84a4a; } .neg { color: #2f6f9f; }
.note { font-size: 8.5pt; color: #46586a; line-height: 1.45; }
.missing { border-left: 3.5px solid #c98b2e; background: #fdf8ef; padding: 2.5mm; font-size: 9pt; }
.ab-label-a { color:#2f6f9f; font-weight:600; } .ab-label-b { color:#b84a4a; font-weight:600; }
ul { margin: 1mm 0 1mm 4mm; padding-left: 3mm; } li { margin: 0.3mm 0; }
</style></head><body>

<section class="page" id="p1">
  <div class="header"><h1>{{ title }}</h1>
    <div class="meta">样本 A：<span class="ab-label-a">{{ a.sample_id }}</span> ·
      样本 B：<span class="ab-label-b">{{ b.sample_id }}</span><br>
      图谱：{{ atlas_label }} · 报告版本：{{ report_version }} · {{ generated_at }}</div></div>
  <div class="summary">{{ summary_html }}</div>
  <div class="grid">
    <div class="panel c12"><div class="grid" style="gap:2mm">
      <div class="c4"><h3>A · 实验与样本</h3>{{ info_table_a }}</div>
      <div class="c8"><h3>B · 质量与有效覆盖</h3>{{ qc_block }}</div>
    </div></div>
    <div class="panel c12"><h3>C · 全脑空间密度（固定阈值等值面，同一视角）</h3>
      <div class="grid">
        <div class="c6"><img class="fig cap3d" src="{{ figs.overview_a }}"><div class="cap"><span class="ab-label-a">{{ a.sample_id }}</span> · 等值面阈值（信号体积占比）{{ iso_text }}</div></div>
        <div class="c6"><img class="fig cap3d" src="{{ figs.overview_b }}"><div class="cap"><span class="ab-label-b">{{ b.sample_id }}</span> · 同阈值、同视角、同色标</div></div>
      </div></div>
    <div class="panel c5"><h3>D · Top 5 密度差</h3><img class="fig caplow" src="{{ figs.top5 }}">
      <div class="cap">Δ密度 = B − A；括号内为 A/B 计数。完整 Top {{ top_n }} 见第 4 页。</div></div>
    <div class="panel c4"><h3>E · 大分区分担</h3><img class="fig caplow" src="{{ figs.coarse }}">
      <div class="cap">互不重叠大分区计数。</div></div>
    <div class="panel c3"><h3>F · 重点 ROI 预览</h3>{{ roi_preview_html }}</div>
  </div>
</section>

<section class="page" id="p2">
  <h2>第 2 页 · 样本与质量证据</h2>
  <div class="grid">
    <div class="panel c6"><h3>配准叠加 QC · <span class="ab-label-a">{{ a.sample_id }}</span></h3>{{ qc_img_a }}</div>
    <div class="panel c6"><h3>配准叠加 QC · <span class="ab-label-b">{{ b.sample_id }}</span></h3>{{ qc_img_b }}</div>
    <div class="panel c6"><h3>样本信息</h3>{{ info_table_pair }}</div>
    <div class="panel c6"><h3>配准参数摘要（来自样本 config 与运行记录）</h3>{{ reg_params_html }}</div>
    <div class="panel c12 missing"><b>质量证据的边界：</b>{{ qc_limits }}</div>
  </div>
</section>

<section class="page" id="p3">
  <h2>第 3 页 · 全脑空间分布</h2>
  <div class="grid">
    <div class="panel c12"><h3>固定阈值密度等值面（同视角）</h3>
      <div class="grid"><div class="c6"><img class="fig cap3d" src="{{ figs.overview_a }}"></div>
        <div class="c6"><img class="fig cap3d" src="{{ figs.overview_b }}"></div></div></div>
    <div class="panel c8"><h3>等间隔冠状切片（共享色标）</h3><img class="fig capmid" src="{{ figs.slices }}">
      <div class="cap">上排 <span class="ab-label-a">{{ a.sample_id }}</span>，下排 <span class="ab-label-b">{{ b.sample_id }}</span>；AP 为距图谱前缘距离；色标 = 信号体积占比 0–{{ slices_vmax_pct }}%。</div></div>
    <div class="panel c4"><h3>前后轴密度曲线</h3><img class="fig capmid" src="{{ figs.ap_profile }}">
      <div class="cap">每个 AP 位置上组织支持区内平均信号体积占比。</div></div>
    <div class="panel c12 note">空间差图（B−A）需要共同可靠覆盖掩膜的严格定义，本版未生成（见缺失清单）。</div>
  </div>
</section>

<section class="page" id="p4">
  <h2>第 4 页 · 脑区定量与 Top {{ top_n }} 密度差</h2>
  <div class="grid">
    <div class="panel c6"><h3>Top {{ top_n }} · Δ密度（B − A，按 |Δ| 排序）</h3><img class="fig capchart" src="{{ figs.topn }}"></div>
    <div class="panel c6"><h3>同组脑区 · 计数与密度（独立坐标轴）</h3><img class="fig capchart" src="{{ figs.dual }}"></div>
    <div class="panel c12"><h3>数值表与排除说明（默认分析层，互斥脑区集合）</h3>{{ topn_table }}
      <div class="note" style="margin-top:1.5mm"><b>未进入排名的区域：</b>{{ excluded_table }}</div>
      <div class="note">候选区 {{ n_candidates }} 个（默认分析层互斥集合）；质量规则：两样本体积 ≥ {{ min_volume_mm3 }} mm³。低计数（任一样本 &lt; {{ low_count_warning }}）仅标记不排除。</div></div>
  </div>
</section>

<section class="page" id="p5">
  <h2>第 5 页 · 重点区域与粒度敏感性</h2>
  <div class="grid">
    {% for roi in rois %}
    <div class="panel c6"><h3>ROI {{ loop.index }} · {{ roi.acronym }}</h3>
      <img class="fig caproi" src="{{ roi.figure }}">
      <div class="cap">{{ roi.caption }}</div>
      {{ roi.table }}</div>
    {% endfor %}
    <div class="panel c12 note"><b>粒度敏感性标签：</b>{{ granularity_note }}</div>
    <div class="panel c12 missing"><b>本页边界：</b>{{ roi_limits }}</div>
  </div>
</section>

<section class="page" id="p6">
  <h2>第 6 页 · 方法、参数与数据索引</h2>
  <div class="grid">
    <div class="panel c6"><h3>方法与定义</h3>{{ methods_html }}</div>
    <div class="panel c6"><h3>参数</h3>{{ params_html }}</div>
    <div class="panel c6"><h3>数据来源</h3>{{ provenance_html }}</div>
    <div class="panel c6"><h3>Excel 字段与缺失输入说明</h3>{{ fields_html }}{{ missing_html }}</div>
  </div>
</section>
</body></html>
"""


def _html_table(frame: pd.DataFrame, *, num_cols: set[str], max_rows: int | None = None) -> str:
    frame = frame if max_rows is None else frame.head(max_rows)
    header = "".join(f'<th class="{"num" if str(c) in num_cols else ""}">{c}</th>' for c in frame.columns)
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for col in frame.columns:
            value = row[col]
            if isinstance(value, float):
                if np.isnan(value):
                    text = "—"
                elif abs(value) >= 100:
                    text = f"{value:,.0f}"
                else:
                    text = f"{value:,.2f}"
            else:
                text = str(value)
            cls = "num" if str(col) in num_cols else ""
            if str(col) in ("Δ密度", "delta_density") and isinstance(value, float) and not np.isnan(value):
                cls += " pos" if value >= 0 else " neg"
                text = ("+" if value >= 0 else "") + text
            cells.append(f'<td class="{cls}">{text}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table><thead><tr>{header}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def html_pdf_from_edge(html_path: Path, pdf_path: Path, edge_exe: str, *, wait_seconds: int = 90) -> Path:
    import time

    pdf_path.unlink(missing_ok=True)
    command = [
        edge_exe, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri(),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    # msedge returns before the browser process finishes writing the PDF
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if pdf_path.exists() and pdf_path.stat().st_size >= 10_000:
            time.sleep(1.0)
            if pdf_path.stat().st_size == pdf_path.stat().st_size:
                return pdf_path
        time.sleep(1.5)
    raise RuntimeError(f"Edge print-to-pdf produced no usable file (exit {result.returncode}): {result.stderr[:400]}")


def write_excel(
    out_path: Path,
    *,
    info: pd.DataFrame,
    comparison: pd.DataFrame,
    coarse_pair: pd.DataFrame,
    roi_tables: dict[str, pd.DataFrame],
    parcellation_names: list[str],
    missing: list[str],
) -> Path:
    fields = pd.DataFrame(
        [
            ("region_name / acronym / region_id", "默认分析层脑区（Name,缩写 / 图谱结构 ID / 层级）"),
            ("count_a / count_b", "A/B 检测计数（工作簿子树聚合计数，互斥集合去重）"),
            ("volume_mm3_a / volume_mm3_b", "A/B 样本空间区域体积 = Total Voxels × 原生体素体积"),
            ("density_cells_per_mm3", "计数 ÷ 体积（cells/mm³，样本空间体积定义）"),
            ("delta_density", "B − A；abs_delta_density 仅用于排序"),
            ("is_remainder", "父节点直接标注体素的残余行（去子区后）"),
            ("qc_usable / low_count", "体积质量规则 / 低计数标记（不排除）"),
        ],
        columns=["字段", "说明"],
    )
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        info.to_excel(writer, sheet_name="样本信息", index=False)
        comparison.to_excel(writer, sheet_name="默认层对比与排名", index=False)
        coarse_pair.to_excel(writer, sheet_name="概览层大分区", index=False)
        for name, table in roi_tables.items():
            table.to_excel(writer, sheet_name=f"细分_{name}"[:31], index=False)
        pd.DataFrame({"region_name": parcellation_names}).to_excel(writer, sheet_name="region_sets默认层", index=False)
        fields.to_excel(writer, sheet_name="字段说明", index=False)
        pd.DataFrame({"missing": missing}).to_excel(writer, sheet_name="缺失项", index=False)
    return out_path


def generate_report(
    *,
    root_dir: str | Path,
    sample_a: str = "",
    sample_b: str = "",
    out_dir: str | Path | None = None,
    top_n: int = 10,
    min_volume_mm3: float = 0.005,
    low_count_warning: float = 20.0,
    bin_um: float = 100.0,
    sigma_um: float = 150.0,
    camera_view_path: str | Path | None = None,
    edge_exe: str = r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    parcellation_csv: str | Path = DEFAULT_PARCELLATION_CSV,
) -> dict[str, Path]:
    from jinja2 import Template

    root_dir = Path(root_dir)
    sample_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir() and not p.name.startswith("_")])
    with_workbook = [p for p in sample_dirs if _has_workbook(p)]
    if sample_a and sample_b:
        pair = [root_dir / sample_a, root_dir / sample_b]
    elif len(with_workbook) == 2:
        pair = with_workbook
    else:
        raise ValueError(f"Need exactly two sample dirs with region workbooks under {root_dir}; found {[p.name for p in with_workbook]}")

    samples = []
    for sample_dir in pair:
        config = load_sample_config(sample_dir)
        voxel_um3 = resolve_voxel_um3(config)
        count_volume = find_count_volume(sample_dir)
        qc_images = sorted(sample_dir.glob("qc_masked_reg_*.png")) or sorted(sample_dir.glob("qc_registration_*.png"))
        missing = []
        if count_volume is None:
            missing.append(f"{sample_dir.name}: 无 atlas 计数体积（visualization/*count_volume.tif），空间密度图缺失")
        if not qc_images:
            missing.append(f"{sample_dir.name}: 无配准叠加 QC 图")
        samples.append(
            SampleInputs(
                sample_id=sample_dir.name,
                sample_dir=sample_dir,
                workbook=find_sample_workbook(sample_dir),
                count_volume=count_volume,
                config=config,
                voxel_um3=voxel_um3,
                qc_images=qc_images,
                missing=missing,
            )
        )
    a, b = samples

    out_dir = Path(out_dir) if out_dir else root_dir / DEFAULT_REPORT_DIRNAME.format(a=a.sample_id, b=b.sample_id)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    parcellation = build_default_parcellation(parcellation_csv)
    metrics_a = load_workbook_metrics(a.workbook)
    metrics_b = load_workbook_metrics(b.workbook)
    table_a = parcellation_table(metrics_a, a.voxel_um3, parcellation)
    table_b = parcellation_table(metrics_b, b.voxel_um3, parcellation)
    comparison = build_comparison(table_a, table_b, min_volume_mm3=min_volume_mm3, low_count_warning=low_count_warning)
    top = rank_top_regions(comparison, top_n)
    coarse_a = coarse_table(metrics_a, a.voxel_um3)
    coarse_b = coarse_table(metrics_b, b.voxel_um3)

    grids = {}
    iso_thresholds = None
    if a.count_volume and b.count_volume:
        grid_a = load_density_fraction_grid(a.count_volume, voxel_um3=a.voxel_um3, bin_um=bin_um, sigma_um=sigma_um)
        grid_b = load_density_fraction_grid(b.count_volume, voxel_um3=b.voxel_um3, bin_um=bin_um, sigma_um=sigma_um)
        grids = {"a": grid_a, "b": grid_b}
        iso_thresholds = derive_iso_thresholds(grid_a)

    figs = {}
    overview_a = overview_b = None
    if grids:
        overview_a = render_density_isosurfaces(grids["a"], thresholds=iso_thresholds, bin_um=bin_um,
                                                output_path=fig_dir / "overview_3d_a.png", camera_view_path=camera_view_path)
        overview_b = render_density_isosurfaces(grids["b"], thresholds=iso_thresholds, bin_um=bin_um,
                                                output_path=fig_dir / "overview_3d_b.png", camera_view_path=camera_view_path)
    figs["top5"] = delta_dot_figure(top.head(5), label_a=a.sample_id, label_b=b.sample_id,
                                    output_path=fig_dir / "top5_delta.svg", title="Top 5 |Δ密度|")
    figs["topn"] = delta_dot_figure(top, label_a=a.sample_id, label_b=b.sample_id,
                                    output_path=fig_dir / "topn_delta.svg", title=f"Top {top_n} |Δ密度|（默认分析层）")
    figs["dual"] = count_density_dual_figure(top, label_a=a.sample_id, label_b=b.sample_id,
                                             output_path=fig_dir / "topn_count_density.svg")
    coarse_pair = coarse_a.merge(coarse_b, on=["system_en", "system_zh"], suffixes=("_a", "_b"))
    figs["coarse"] = coarse_share_figure(coarse_a, coarse_b, label_a=a.sample_id, label_b=b.sample_id,
                                         output_path=fig_dir / "coarse_share.svg")
    slices_vmax_pct = 3.0
    if grids:
        vmax = float(np.percentile(np.concatenate([grids["a"][grids["a"] > 0], grids["b"][grids["b"] > 0]]), 99.5))
        slices_vmax_pct = round(vmax * 100, 2)
        figs["slices"] = coronal_slice_figure(grids["a"], grids["b"], bin_um=bin_um, vmax=vmax,
                                              positions_mm=[1.5, 3.2, 4.9, 6.6, 8.3, 10.0],
                                              label_a=a.sample_id, label_b=b.sample_id,
                                              output_path=fig_dir / "coronal_slices.png")
        figs["ap_profile"] = ap_profile_figure(grids["a"], grids["b"], bin_um=bin_um,
                                               label_a=a.sample_id, label_b=b.sample_id,
                                               output_path=fig_dir / "ap_profile.svg")

    system_of = system_of_regions(parcellation_csv)
    comparison_with_system = comparison.copy()
    comparison_with_system["system"] = comparison_with_system["region_name"].map(system_of)
    coarse_delta = coarse_pair.copy()
    coarse_delta["delta_density"] = coarse_delta["density_cells_per_mm3_b"] - coarse_delta["density_cells_per_mm3_a"]
    rois = []
    roi_tables: dict[str, pd.DataFrame] = {}
    chosen_systems: list[str] = []
    ranked_all = comparison.sort_values(["abs_delta_density", "region_name"], ascending=[False, True], kind="stable")
    for row in ranked_all.itertuples():
        system = system_of.get(row.region_name)
        if system and system not in chosen_systems:
            chosen_systems.append(system)
        if len(chosen_systems) >= 2:
            break
    for system in chosen_systems:
        parent = coarse_delta[coarse_delta["system_en"] == system]
        if parent.empty:
            continue
        parent_row = parent.iloc[0]
        children = comparison_with_system[comparison_with_system["system"] == system].copy()
        children = children.sort_values("abs_delta_density", ascending=False).head(6)
        if children.empty:
            continue
        child = pd.DataFrame({
            "region_name": children["region_name"],
            "acronym": children["acronym"],
            "count_a": children["count_a"], "count_b": children["count_b"],
            "density_cells_per_mm3_a": children["density_cells_per_mm3_a"],
            "density_cells_per_mm3_b": children["density_cells_per_mm3_b"],
            "delta_density": children["delta_density"],
        })
        direction = "子区方向较一致" if (np.sign(child["delta_density"]) == np.sign(parent_row["delta_density"])).mean() >= 0.6 else "子区方向不一致"
        low = "；部分子区低计数" if (child[["count_a", "count_b"]].min(axis=1) < low_count_warning).any() else ""
        zh = COARSE_DISPLAY.get(system, system)
        parent_row_dict = {"acronym": zh, "density_cells_per_mm3_a": parent_row["density_cells_per_mm3_a"],
                           "density_cells_per_mm3_b": parent_row["density_cells_per_mm3_b"]}
        figure = roi_parent_child_figure(system, parent_row_dict, child,
                                         label_a=a.sample_id, label_b=b.sample_id,
                                         output_path=fig_dir / f"roi_{system.replace(' ', '_')}.svg")
        child_display = pd.DataFrame({
            "缩写": [row.acronym or row.region_name.rsplit(",", 1)[0] for row in child.itertuples()],
            "A 计数": child["count_a"].astype(int),
            "B 计数": child["count_b"].astype(int),
            "A 密度": child["density_cells_per_mm3_a"],
            "B 密度": child["density_cells_per_mm3_b"],
            "Δ密度": child["delta_density"],
        })
        rois.append({
            "acronym": zh,
            "short": f"{zh}：Δ密度 {parent_row['delta_density']:+,.0f}（详见第 5 页）",
            "figure": figure.relative_to(out_dir).as_posix(),
            "caption": f"{zh}（{system}）：系统级 Δ密度 {parent_row['delta_density']:+,.1f}；默认层子区中 |Δ| 最大的 6 个如图；{direction}{low}。标签为尺度敏感性说明，非统计检验。",
            "table": _html_table(child_display, num_cols={"A 计数", "B 计数", "A 密度", "B 密度", "Δ密度"}),
        })
        roi_tables[system] = child

    missing = list(a.missing) + list(b.missing) + [
        "count_direct 未区分（父区数值均为子树聚合计数）",
        "逐区域覆盖率/有效体积未定义（区域 QC 仅最小体积规则）",
        "空间差图（B−A）未生成（需共同覆盖掩膜定义）",
        "ROI 原图叠加缺失（本版以 QC 叠加图替代）",
    ]

    excel_path = write_excel(out_dir / "cfos_report_tables.xlsx", info=_info_frame(a, b),
                             comparison=comparison.sort_values("abs_delta_density", ascending=False),
                             coarse_pair=coarse_pair, roi_tables=roi_tables,
                             parcellation_names=[r["name"] for r in parcellation], missing=missing)

    params = {
        "mode": "two_sample_descriptive", "sign_convention": "B_minus_A",
        "ranking_metric": "absolute_density_difference", "top_n": top_n,
        "hemisphere_mode": "bilateral", "min_volume_mm3": min_volume_mm3,
        "low_count_warning": low_count_warning, "bin_um": bin_um, "sigma_um": sigma_um,
        "iso_thresholds": list(iso_thresholds) if iso_thresholds else None,
        "parcellation_csv": str(parcellation_csv), "voxel_um3": {"a": a.voxel_um3, "b": b.voxel_um3},
    }
    (out_dir / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_a": {"id": a.sample_id, "workbook": str(a.workbook), "count_volume": str(a.count_volume), "config": str(a.sample_dir / "config.json")},
        "sample_b": {"id": b.sample_id, "workbook": str(b.workbook), "count_volume": str(b.count_volume), "config": str(b.sample_dir / "config.json")},
        "atlas": "allen_mouse_25um（配准参考 atlas.tiff）",
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")

    top_text = "、".join(
        f"{(row.acronym or row.region_name.rsplit(',', 1)[0])}（{'B 高' if row.delta_density >= 0 else 'A 高'}）"
        for row in top.head(3).itertuples()
    )
    total_a = float(table_a["count"].sum())
    total_b = float(table_b["count"].sum())
    summary_html = (
        f"两样本全脑检测计数：<span class='ab-label-a'>{a.sample_id}</span> {total_a:,.0f} vs "
        f"<span class='ab-label-b'>{b.sample_id}</span> {total_b:,.0f}。"
        f"在固定互斥脑区集合（{len(comparison)} 区）上，密度差绝对值最大的区域为 {top_text}。"
        "以上为样本间描述性差异（每条件一只动物，无组间显著性推断）；密度与空间图定义见第 6 页。"
    )

    iso_text = "、".join(f"{v * 100:g}%" for v in iso_thresholds) if iso_thresholds else "未生成（缺计数体积）"
    placeholder = fig_dir / "missing.png"
    if not placeholder.exists():
        pd.DataFrame({"缺失": ["该图所需输入缺失，见缺失清单"]}).to_csv(placeholder.with_suffix(".csv"))
        fig_placeholder, ax_placeholder = plt.subplots(figsize=(4, 3))
        ax_placeholder.text(0.5, 0.5, "输入缺失\n见缺失清单", ha="center", va="center", fontsize=12, color="#c98b2e")
        ax_placeholder.axis("off")
        fig_placeholder.savefig(placeholder)
        plt.close(fig_placeholder)

    def fig_ref(path: Path | None) -> str:
        if path is None:
            return placeholder.relative_to(out_dir).as_posix()
        return path.relative_to(out_dir).as_posix()

    html = Template(HTML_TEMPLATE).render(
        title=f"全脑 c-Fos 分布与样本间比较：{a.sample_id} vs {b.sample_id}",
        a=a, b=b,
        figs={"overview_a": fig_ref(overview_a), "overview_b": fig_ref(overview_b),
              "top5": fig_ref(figs["top5"]), "topn": fig_ref(figs["topn"]),
              "dual": fig_ref(figs["dual"]), "coarse": fig_ref(figs["coarse"]),
              "slices": fig_ref(figs.get("slices")), "ap_profile": fig_ref(figs.get("ap_profile"))},
        top_n=top_n, iso_text=iso_text, slices_vmax_pct=slices_vmax_pct,
        summary_html=summary_html, generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        report_version="v1.0", atlas_label="Allen mouse 25µm（CCF，配准参考 atlas.tiff）",
        info_table_a=_info_frame(a, None).to_html(index=False) if False else _sample_kv_html(a),
        info_table_pair=_info_frame(a, b).to_html(index=False, border=0),
        qc_block=_qc_block_html(a, b), qc_img_a=_qc_img_html(a, out_dir), qc_img_b=_qc_img_html(b, out_dir),
        reg_params_html=_reg_params_html(a, b),
        qc_limits="QC 图来自管线配准/掩膜检查步骤；本版未做人工抽样核查，逐区域覆盖率未定义（见缺失清单）。两样本若成像或染色批次不同，报告未自动校正批次效应。",
        topn_table=_topn_table_html(top),
        excluded_table=_excluded_table_html(comparison),
        n_candidates=int(comparison["qc_usable"].sum()), min_volume_mm3=min_volume_mm3, low_count_warning=low_count_warning,
        rois=rois, roi_preview_html=("<br>".join(roi["short"] for roi in rois) if rois else "无（缺 Top 区域子区数据）"),
        granularity_note="「方向较一致」指 ≥60% 子区 Δ密度符号与父区一致；「子区方向相反」「低计数或覆盖敏感」按数据触发。这些标签描述粒度敏感性，不是统计检验或生物学重复验证。",
        roi_limits="父区与其子区来自图谱树的不同层级，不进入同一排名集合；ROI 选取规则 = 默认分析层 Top N（本报告 Top1/Top2）。原图叠加面板缺失（见缺失清单）。",
        methods_html=_methods_html(a, b), params_html=_params_html(params),
        provenance_html=_provenance_html(provenance), missing_html=_missing_html(missing),
        fields_html="<div class='note'>字段说明见 Excel「字段说明」工作表；region_sets 清单见 Excel「region_sets默认层」。</div>",
    )
    html_path = out_dir / "report.html"
    html_path.write_text(html, encoding="utf-8")
    pdf_path = html_pdf_from_edge(html_path, out_dir / "report.pdf", edge_exe)
    return {"out_dir": out_dir, "html": html_path, "pdf": pdf_path, "excel": excel_path}


def _has_workbook(sample_dir: Path) -> bool:
    try:
        find_sample_workbook(sample_dir)
        return True
    except FileNotFoundError:
        return False


def _info_frame(a: SampleInputs, b: SampleInputs | None) -> pd.DataFrame:
    rows = [("样本", a.sample_id, b.sample_id if b else ""),
            ("区域工作簿", a.workbook.name, b.workbook.name if b else ""),
            ("原生体素 (µm³)", f"{a.voxel_um3:g}", f"{b.voxel_um3:g}" if b else ""),
            ("计数体积", a.count_volume.name if a.count_volume else "缺失", b.count_volume.name if b and b.count_volume else "缺失")]
    return pd.DataFrame(rows, columns=["字段", "样本 A", "样本 B"])


def _sample_kv_html(sample: SampleInputs) -> str:
    resolution = (sample.config.get("input") or {}).get("resolution_xyz")
    return (f"<div class='kv'><b>样本</b>{sample.sample_id}</div>"
            f"<div class='kv'><b>体素间距</b>{resolution} µm (x,y,z)</div>"
            f"<div class='kv'><b>信号通道</b>ch{(sample.config.get('input') or {}).get('channels', {}).get('signal', '?')}</div>"
            f"<div class='kv'><b>配准通道</b>ch{(sample.config.get('input') or {}).get('channels', {}).get('registration', '?')}</div>")


def _qc_block_html(a: SampleInputs, b: SampleInputs) -> str:
    return ("<ul>"
            f"<li>{a.sample_id}：区域工作簿 {a.workbook.name}；计数体积 {'有' if a.count_volume else '缺失'}；QC 叠加图 {len(a.qc_images)} 张</li>"
            f"<li>{b.sample_id}：区域工作簿 {b.workbook.name}；计数体积 {'有' if b.count_volume else '缺失'}；QC 叠加图 {len(b.qc_images)} 张</li>"
            "</ul><div class='note'>空间图以组织支持区（计数&gt;0 且在图谱标注内）为可靠范围，支持区外不渲染。</div>")


def _qc_img_html(sample: SampleInputs, out_dir: Path) -> str:
    if not sample.qc_images:
        return "<div class='missing'>无配准叠加 QC 图（缺失清单已记录）</div>"
    target = out_dir / "figures" / f"qc_{sample.sample_id}_{sample.qc_images[0].name}"
    if not target.exists():
        shutil.copy(sample.qc_images[0], target)
    return f"<img class='fig capqc' src='{target.relative_to(out_dir).as_posix()}'><div class='cap'>{sample.qc_images[0].name}（管线配准检查步骤输出）</div>"


def _topn_table_html(top: pd.DataFrame) -> str:
    frame = pd.DataFrame({
        "脑区": [row.region_name.rsplit(",", 1)[0] for row in top.itertuples()],
        "缩写": [row.acronym for row in top.itertuples()],
        "A 计数": top["count_a"].astype(int),
        "B 计数": top["count_b"].astype(int),
        "A 密度": top["density_cells_per_mm3_a"],
        "B 密度": top["density_cells_per_mm3_b"],
        "Δ密度": top["delta_density"],
        "低计数": ["是" if v else "" for v in top["low_count"]],
    })
    return _html_table(frame, num_cols={"A 计数", "B 计数", "A 密度", "B 密度", "Δ密度"})


def _excluded_table_html(comparison: pd.DataFrame) -> str:
    excluded = comparison[~comparison["qc_usable"]].copy()
    if excluded.empty:
        return "无因体积规则排除的区域。"
    with_counts = int((excluded[["count_a", "count_b"]].sum(axis=1) > 0).sum())
    return (
        f"共 <b>{len(excluded)}</b> 个区域未满足两样本体积 ≥ 阈值的质量规则"
        f"（多为图谱中无标注体积的细分节点；其中 {with_counts} 个仍带非零计数，完整清单见 Excel qc_usable=False 行，未从数据中删除）。"
    )


def _reg_params_html(a: SampleInputs, b: SampleInputs) -> str:
    rows = []
    for sample in (a, b):
        reg = (sample.config.get("registration") or {})
        duration = ""
        manifest_path = sample.sample_dir / "_run_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                seconds = float(manifest.get("duration_seconds") or 0.0)
                duration = f"{seconds / 60:.0f} min"
            except Exception:
                duration = ""
        rows.append(
            f"<tr><td>{sample.sample_id}</td><td>{reg.get('method', '—')}/{reg.get('transform_type', '—')}</td>"
            f"<td>{reg.get('mode', '—')}</td><td>{reg.get('grad_step', '—')} / {reg.get('syn_iterations', '—')}</td>"
            f"<td>{'开' if reg.get('halo_mask') else '关'}</td><td>{duration or '—'}</td></tr>"
        )
    return ("<table><thead><tr><th>样本</th><th>方法</th><th>方向</th><th>grad_step / 迭代</th><th>halo mask</th><th>配准耗时</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
            "<div class='note'>参数与耗时取自样本 config.json 的 registration 块和 _run_manifest.json；n4 预处理与刚性预检按管线默认开启（qc 目录含 rigid/final 检查图）。</div>")


def _methods_html(a: SampleInputs, b: SampleInputs) -> str:
    return ("<ul>"
            "<li><b>图谱</b>：Allen mouse CCF 25 µm（配准参考 atlas.tiff；标签树 Region_Csv_Rev2_updated，2011 版体）。</li>"
            "<li><b>默认分析层</b>：固定互斥脑区集合（分到不分 layer 为止；皮层拆亚区、核团不拆部；父节点直接标注体素计入残余行），A/B 使用同一集合。</li>"
            "<li><b>密度（区域）</b>：检测计数 ÷ 样本空间区域体积（Total Voxels × 原生体素体积），单位 cells/mm³；属<b>样本空间密度</b>定义。</li>"
            "<li><b>密度（空间图）</b>：c-Fos 掩膜信号体积 ÷ 图谱空间分箱体积 = 信号体积占比；与区域密度分属两种定义，不可直接互比。</li>"
            "<li><b>Δ密度</b> = 密度(B) − 密度(A)；排序用 |Δ密度|，图中保留方向；不做倍数变化排名。</li>"
            "<li><b>统计边界</b>：每条件一只动物，仅描述性比较；无 p/q 值、显著性标记或置信区间。</li>"
            "</ul>")


def _params_html(params: dict) -> str:
    return "<pre style='font-size:8.2pt;white-space:pre-wrap'>" + json.dumps(params, ensure_ascii=False, indent=2) + "</pre>"


def _provenance_html(provenance: dict) -> str:
    lines = [f"<b>生成时间</b>：{provenance['generated_at']}"]
    for key in ("sample_a", "sample_b"):
        entry = provenance[key]
        lines.append(f"<b>{entry['id']}</b>：{entry['workbook']}<br>计数体积：{entry['count_volume']}")
    return "<div class='note'>" + "<br>".join(lines) + f"<br><b>图谱</b>：{provenance['atlas']}</div>"


def _missing_html(missing: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{item}</li>" for item in missing) + "</ul>"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the standard two-sample c-Fos descriptive report (HTML+PDF+Excel).")
    parser.add_argument("--root_dir", required=True, help="Batch root containing the two sample dirs")
    parser.add_argument("--sample_a", default="", help="Sample A dir name (default: auto-detect two workbook samples)")
    parser.add_argument("--sample_b", default="")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--min_volume_mm3", type=float, default=0.005)
    parser.add_argument("--low_count_warning", type=float, default=20.0)
    parser.add_argument("--bin_um", type=float, default=100.0)
    parser.add_argument("--sigma_um", type=float, default=150.0)
    parser.add_argument("--camera_view", default="S:/Arivis_Analysis/standard_brainrender_view.json")
    parser.add_argument("--edge_exe", default=r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        outputs = generate_report(
            root_dir=args.root_dir, sample_a=args.sample_a, sample_b=args.sample_b, out_dir=args.out_dir,
            top_n=args.top_n, min_volume_mm3=args.min_volume_mm3, low_count_warning=args.low_count_warning,
            bin_um=args.bin_um, sigma_um=args.sigma_um, camera_view_path=args.camera_view, edge_exe=args.edge_exe,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    for key, value in outputs.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
