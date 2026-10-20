"""Region-grid heatmap: parcellated regions x samples density matrix.

Distinct from ``heatmap.py`` (``render_signal_heatmap``), which warps a mask
into ATLAS SPACE and renders anatomical 2D/3D heatmaps: this module has no
spatial anatomy — rows are the ``top_level7_density_ratio`` default
parcellation regions (level auto: cortical subareas split, nuclei whole),
columns are samples, and each cell is the region's signal density
(Signal Voxels / Total Voxels, %). Use it to compare many samples
quantitatively per region; use ``render_signal_heatmap`` to view where signal
sits in the brain.

Layout (fixed in production 2026-10-14): grey-matter regions only, grouped by
major division (Isocortex / Cortical subplate / Striatum / Pallidum / Thalamus
/ Hypothalamus / Midbrain / Pons / Medulla / Cerebellum); groups stack
naturally into a few equal-width columns (square cells, top-aligned panels,
ragged bottoms); each group's full name is an inline centered band above its
rows; vertical sample labels under each panel; white->orange colormap with a
thick colorbar hugging the last panel.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_modules.visualization.top_level7_density_ratio import (
    DEFAULT_PARCELLATION_CSV,
    build_default_parcellation,
    load_all_level_counts,
    load_structure_tree,
    resolve_parcellation_counts,
)

GROUP_ORDER = ["Isocortex", "CTXsp", "STR", "PAL", "TH", "HY", "MB", "P", "MY", "CB"]
GROUP_FULLNAME = {
    "Isocortex": "Isocortex",
    "CTXsp": "Cortical\nsubplate",
    "STR": "Striatum",
    "PAL": "Pallidum",
    "TH": "Thalamus",
    "HY": "Hypothalamus",
    "MB": "Midbrain",
    "P": "Pons",
    "MY": "Medulla",
    "CB": "Cerebellum",
}
# chain-acronym -> major division (first match wins)
GROUP_RULES = [
    ("CTXpl", "Isocortex"), ("Isocortex", "Isocortex"),
    ("OLF", "OLF"), ("HPF", "HPF"), ("CTXsp", "CTXsp"),
    ("STR", "STR"), ("PAL", "PAL"),
    ("TH", "TH"), ("HY", "HY"),
    ("MBmot", "MB"), ("MBsen", "MB"), ("MBsta", "MB"), ("MB", "MB"),
    ("P", "P"), ("MY", "MY"),
    ("CB", "CB"), ("CBX", "CB"), ("CBN", "CB"), ("VERM", "CB"), ("HEM", "CB"),
]


def find_stats_workbook(sample_dir: str | Path) -> Path:
    """Newest stats workbook of a cfos run: prefers the atlas-v2 (*_v2.xlsx),
    falls back to the legacy name."""
    sample_dir = Path(sample_dir)
    results = sample_dir / "results"
    for pattern in ("*_brain_distribution_stats_v2.xlsx", "*_brain_distribution_stats.xlsx"):
        matches = sorted(
            (p for p in results.glob(pattern) if not p.name.startswith("~$")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"No results/*_brain_distribution_stats*.xlsx under sample dir: {sample_dir}"
    )


def grey_group_of(node_id: int, nodes: dict) -> str | None:
    """Major division of a region, or None for fiber tracts / ventricles."""
    chain = []
    cur = nodes[node_id]
    while True:
        chain.append(cur["acronym"] or cur["name"])
        if cur["parent"] is None:
            break
        cur = nodes[cur["parent"]]
    if "grey" not in chain:
        return None
    for key, label in GROUP_RULES:
        if key in chain:
            return label
    return None


def collect_region_density(
    workbook_paths: list[str | Path],
    labels: list[str],
    parcellation_csv: str | Path = DEFAULT_PARCELLATION_CSV,
) -> tuple[list[tuple[str, str, str]], np.ndarray, list[list[tuple[str, int, int]]]]:
    """Per-region density matrix over the default parcellation.

    Returns (rows, mat, group_sizes):
      rows      -- (acronym, region name, group) in parcellation tree order,
                   grey-matter regions present (Total Voxels > 0) in any sample
      mat       -- len(rows) x len(labels) density in percent; NaN where the
                   region has no voxels in that sample
      group_sizes -- (group, row_start, row_end_exclusive, layout_rows) per
                   group in GROUP_ORDER; layout_rows includes the 2 band rows
    """
    regions = build_default_parcellation(parcellation_csv)
    nodes, _root = load_structure_tree(parcellation_csv)

    per_sample = []
    for path in workbook_paths:
        sv = resolve_parcellation_counts(regions, load_all_level_counts(path, "Signal Voxels"))
        tv = resolve_parcellation_counts(regions, load_all_level_counts(path, "Total Voxels"))
        per_sample.append(
            {r["name"]: (float(s), float(t)) for r, s, t in zip(regions, sv["count"], tv["count"])}
        )

    rows: list[tuple[str, str, str]] = []
    for r in regions:
        g = grey_group_of(r["id"], nodes)
        if g is None:
            continue
        if any(per_sample[k][r["name"]][1] > 0 for k in range(len(per_sample))):
            rows.append((r["acronym"] or r["name"], r["name"], g))

    mat = np.full((len(rows), len(labels)), np.nan)
    for i, (_acro, name, _g) in enumerate(rows):
        for j in range(len(labels)):
            sv, tv = per_sample[j][name]
            if tv > 0:
                mat[i, j] = sv / tv * 100.0

    group_sizes = []
    for g in GROUP_ORDER:
        idx = [i for i, row in enumerate(rows) if row[2] == g]
        if idx:
            group_sizes.append((g, idx[0], idx[-1] + 1, idx[-1] + 1 - idx[0] + 2))
    return rows, mat, group_sizes


def pack_group_panels(
    group_sizes: list[tuple[str, int, int, int]], *, max_panels: int = 5
) -> list[list[tuple[str, int, int, int]]]:
    """Sequential groups into <= max_panels columns at group boundaries.

    Picks the smallest column capacity that fits within max_panels (whole
    groups only, canonical order), tie-broken on less blank layout space.
    """
    if not group_sizes:
        return []
    floor = max(it[3] for it in group_sizes)
    best = None
    for cap in range(floor, floor + 60):
        packed: list[list[tuple[str, int, int, int]]] = []
        cur: list[tuple[str, int, int, int]] = []
        cur_len = 0
        for it in group_sizes:
            if cur and cur_len + it[3] > cap:
                packed.append(cur)
                cur, cur_len = [], 0
            cur.append(it)
            cur_len += it[3]
        if cur:
            packed.append(cur)
        if len(packed) > max_panels:
            continue
        blank = len(packed) * cap - sum(it[3] for it in group_sizes)
        if best is None or len(packed) < len(best[0]) or (
            len(packed) == len(best[0]) and blank < best[1]
        ):
            best = (packed, blank)
        break  # smallest cap reaching <= max_panels wins
    if best is None:  # unreachable group bigger than any cap window
        return [list(group_sizes)]
    return best[0]


def render_region_grid(
    rows: list[tuple[str, str, str]],
    mat: np.ndarray,
    labels: list[str],
    panels: list[list[tuple[str, int, int, int]]],
    out_png: str | Path,
    out_csv: str | Path,
    *,
    vmin: float = 0.0,
    vmax: float = 1.0,
    dpi: int = 300,
) -> None:
    """Draw the finalized grid layout (square cells, inline group bands)."""
    cell = 0.170
    gutter = 0.48
    gap = 0.26
    sample_lbl = 0.32
    margin = 0.25
    cbar_w = 0.20
    cbar_gap = 0.14

    tallest = max(sum(n for _g, _s, _e, n in groups) for groups in panels)
    n_cols = len(labels)
    fig_h = 0.45 + tallest * cell + sample_lbl + 0.35
    fig_w = (
        margin + gutter
        + sum(n_cols * cell + gutter + gap for _ in panels)
        + cbar_gap + cbar_w + 0.55
    )
    fig = plt.figure(figsize=(fig_w, fig_h))
    cmap = matplotlib.colormaps["Oranges"].copy()
    cmap.set_bad("#ffffff")

    x_in = margin + gutter
    top_in = 0.45
    for groups in panels:
        n_rows = sum(n for _g, _s, _e, n in groups)
        ax = fig.add_axes([
            x_in / fig_w,
            (fig_h - top_in - n_rows * cell) / fig_h,
            n_cols * cell / fig_w,
            n_rows * cell / fig_h,
        ])
        sub = np.full((n_rows, n_cols), np.nan)
        tick_pos: list[int] = []
        tick_labels: list[str] = []
        cursor = 0
        last_region_row = 0
        for _g, s, e, _n in groups:
            band_y = cursor
            cursor += 2  # two-row band fits two-line names
            sub[cursor : cursor + (e - s), :] = mat[s:e, :]
            for i in range(s, e):
                tick_pos.append(cursor + (i - s))
                tick_labels.append(rows[i][0])
            cursor += e - s
            last_region_row = cursor
            if band_y > 0:
                ax.axhline(band_y - 0.5, color="#999999", lw=0.8)
            ax.axhspan(band_y - 0.5, band_y + 1.5, facecolor="#f6ecdf", zorder=2.0)
            ax.text(
                (n_cols - 1) / 2.0, band_y + 0.5, GROUP_FULLNAME[_g],
                ha="center", va="center", fontsize=8.2, fontweight="bold",
                color="#7a4a12", zorder=3.0, linespacing=1.0,
            )

        im = ax.imshow(sub, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
        ax.xaxis.tick_bottom()
        ax.tick_params(axis="x", length=0, pad=1.5)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_labels, fontsize=6.4)
        ax.tick_params(axis="y", length=0, pad=2)
        ax.set_xticks(np.arange(-0.5, n_cols), minor=True)
        ax.set_yticks(np.arange(-0.5, last_region_row), minor=True)
        ax.grid(which="minor", color="#f0f0f0", lw=0.4)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        x_in += n_cols * cell + gutter + gap

    cbar_x = x_in - gap + cbar_gap
    cax = fig.add_axes([
        cbar_x / fig_w,
        (fig_h - top_in - tallest * cell * 0.72) / fig_h,
        cbar_w / fig_w,
        tallest * cell * 0.72 / fig_h,
    ])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("cFos signal density (%)", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7)
    fig.text(0.5, 0.988, "cFos density heatmap", ha="center", va="top", fontsize=13.5, fontweight="bold")

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)

    csv = pd.DataFrame(
        {"acronym": [r[0] for r in rows], "region": [r[1] for r in rows], "group": [r[2] for r in rows]}
    )
    for j, s in enumerate(labels):
        csv[f"density_pct_{s}"] = np.round(mat[:, j], 4)
    csv.to_csv(out_csv, index=False, encoding="utf-8-sig")


def strip_common_prefix(names: list[str]) -> list[str]:
    """Shorten sample-dir names by their shared prefix (keep a distinguishing tail).

    The cut lands on the last '_' inside the common prefix so B_x_SY1/B_x_SY2
    -> SY1/SY2 (never a bare digit).
    """
    if len(names) < 2:
        return names
    prefix = names[0]
    for name in names[1:]:
        while prefix and not name.startswith(prefix):
            prefix = prefix[:-1]
    if not prefix:
        return names
    cut = prefix.rfind("_") + 1 if "_" in prefix else len(prefix)
    stripped = [name[cut:].lstrip("_- ") for name in names]
    if any(not s for s in stripped):
        return names
    return stripped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Region-grid heatmap: Top7 auto parcellation regions x samples density matrix"
    )
    parser.add_argument("--root_dir", required=True, help="Batch folder containing sample child dirs")
    parser.add_argument(
        "--samples", default="",
        help="Comma-separated sample dir names under root_dir (default: all with a stats workbook)",
    )
    parser.add_argument("--parcellation_csv", default=str(DEFAULT_PARCELLATION_CSV))
    parser.add_argument("--output_prefix", default="", help="Default <root_dir>/<root_name>_region_grid")
    parser.add_argument("--max_panels", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)

    root = Path(args.root_dir)
    if args.samples:
        names = [s.strip() for s in args.samples.split(",") if s.strip()]
    else:
        names = sorted(
            p.name for p in root.iterdir()
            if p.is_dir() and any((p / "results").glob("*_brain_distribution_stats.xlsx"))
        )
    if len(names) < 2:
        print("Need at least two sample dirs with a stats workbook", file=sys.stderr)
        return 1
    workbooks = [find_stats_workbook(root / n) for n in names]
    labels = strip_common_prefix(names)

    rows, mat, group_sizes = collect_region_density(workbooks, labels, args.parcellation_csv)
    panels = pack_group_panels(group_sizes, max_panels=args.max_panels)

    prefix = args.output_prefix or str(root / f"{root.name}_region_grid")
    render_region_grid(rows, mat, labels, panels, f"{prefix}_heatmap.png", f"{prefix}_density.csv", dpi=args.dpi)
    print(json_summary(prefix, rows, labels))
    return 0


def json_summary(prefix: str, rows: list, labels: list[str]) -> str:
    import json

    return json.dumps(
        {
            "png": f"{prefix}_heatmap.png",
            "csv": f"{prefix}_density.csv",
            "regions": len(rows),
            "samples": labels,
            "generated": date.today().isoformat(),
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
