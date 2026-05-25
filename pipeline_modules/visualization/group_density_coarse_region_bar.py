"""Build grouped coarse-region bar plots from multiple density Excel files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from pipeline_modules.visualization.coarse_region_metric_plot import (
        DEFAULT_CFG,
        build_coarse_region_table,
        load_level_sheets,
        load_region_names,
        parse_figsize,
    )
except ImportError:
    from coarse_region_metric_plot import (  # type: ignore[no-redef]
        DEFAULT_CFG,
        build_coarse_region_table,
        load_level_sheets,
        load_region_names,
        parse_figsize,
    )


def find_density_excels(root_dir: str | Path) -> list[Path]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input folder not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {root}")

    density_files = []
    for sample_dir in sorted([path for path in root.iterdir() if path.is_dir()], key=lambda path: path.name.lower()):
        matches = sorted(
            [
                path
                for path in sample_dir.glob("*.xlsx")
                if "density" in path.name.lower()
                and not path.name.startswith("~$")
                and "coarse_region_stats" not in path.name.lower()
            ],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if matches:
            density_files.append(matches[0])

    if not density_files:
        raise FileNotFoundError(f"No density *.xlsx files found under child sample folders of: {root}")
    return density_files


def collect_group_coarse_table(
    root_dir: str | Path,
    *,
    cfg: str | Path,
    metric: str,
) -> pd.DataFrame:
    region_table = load_region_names(cfg)
    sample_tables = []

    for excel_path in find_density_excels(root_dir):
        sample_name = excel_path.parent.name
        level_table = load_level_sheets(excel_path)
        coarse_table = build_coarse_region_table(level_table, region_table, metric, warn_missing=False)
        coarse_table["sample"] = sample_name
        coarse_table["density_excel"] = str(excel_path)
        sample_tables.append(coarse_table)

    return pd.concat(sample_tables, ignore_index=True)


def plot_grouped_bar(
    table: pd.DataFrame,
    *,
    output_path: str | Path,
    metric: str,
    title: str,
    ylabel: str,
    figsize: tuple[float, float],
    dpi: int,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    region_order = (
        table[["order", "region_name"]]
        .drop_duplicates()
        .sort_values("order", kind="stable")["region_name"]
        .astype(str)
        .tolist()
    )
    samples = table["sample"].drop_duplicates().astype(str).tolist()
    values_by_sample = []
    for sample in samples:
        sample_frame = table[table["sample"] == sample].set_index("region_name")
        values_by_sample.append(sample_frame.reindex(region_order)["value"].fillna(0.0).to_numpy(dtype=np.float64))

    x = np.arange(len(region_order), dtype=np.float64)
    sample_count = max(len(samples), 1)
    group_width = min(0.82, 0.14 * sample_count)
    bar_width = group_width / sample_count
    offsets = (np.arange(sample_count, dtype=np.float64) - (sample_count - 1) / 2.0) * bar_width
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, sample_count))

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#253746",
            "axes.linewidth": 1.0,
            "axes.labelcolor": "#17212b",
            "axes.titleweight": "semibold",
            "xtick.color": "#253746",
            "ytick.color": "#253746",
            "font.size": 10.5,
            "font.family": "DejaVu Sans",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for index, (sample, values) in enumerate(zip(samples, values_by_sample)):
        ax.bar(
            x + offsets[index],
            values,
            width=bar_width * 0.92,
            label=sample,
            color=colors[index],
            edgecolor="#ffffff",
            linewidth=0.6,
        )

    ax.set_title(title, fontsize=15, pad=12)
    ax.set_ylabel(ylabel, fontsize=11.5)
    ax.set_xticks(x)
    ax.set_xticklabels(region_order, rotation=42, ha="right")
    ax.grid(axis="y", color="#d8e1e8", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Sample", frameon=False, ncols=1, loc="best")

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def write_outputs(table: pd.DataFrame, output_prefix: str | Path) -> tuple[Path, Path]:
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_name(f"{output_prefix.name}_coarse_region_group_stats.csv")
    xlsx_path = output_prefix.with_name(f"{output_prefix.name}_coarse_region_group_stats.xlsx")
    table.to_csv(csv_path, index=False)
    table.to_excel(xlsx_path, index=False)
    return csv_path, xlsx_path


def generate_group_density_plot(
    *,
    root_dir: str | Path,
    cfg: str | Path = DEFAULT_CFG,
    metric: str = "Voxel Density",
    title: str | None = None,
    ylabel: str | None = None,
    output_prefix: str | Path | None = None,
    figsize: tuple[float, float] = (13.5, 6.8),
    dpi: int = 300,
) -> dict[str, Path | pd.DataFrame]:
    root_dir = Path(root_dir)
    title = title if title and str(title).strip() else f"Coarse-region {metric}"
    ylabel = ylabel if ylabel and str(ylabel).strip() else metric
    output_prefix = Path(output_prefix) if output_prefix else root_dir / f"{root_dir.name}_{metric.replace(' ', '_')}"

    table = collect_group_coarse_table(root_dir, cfg=cfg, metric=metric)
    csv_path, xlsx_path = write_outputs(table, output_prefix)
    plot_path = output_prefix.with_name(f"{output_prefix.name}_coarse_region_group_bar.png")
    plot_grouped_bar(
        table,
        output_path=plot_path,
        metric=metric,
        title=title,
        ylabel=ylabel,
        figsize=figsize,
        dpi=dpi,
    )
    return {
        "table": table,
        "csv_path": csv_path,
        "xlsx_path": xlsx_path,
        "plot_path": plot_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find density Excel files in sample subfolders and draw grouped coarse-region bar plots."
    )
    parser.add_argument("--root_dir", required=True, help="Folder whose direct child folders are sample_dir folders")
    parser.add_argument("--cfg", default=str(DEFAULT_CFG), help="Path to Region_Csv_Rev1_updated.CSV")
    parser.add_argument("--metric", default="Voxel Density", help="Metric column to compare")
    parser.add_argument("--title", default=None, help="Figure title")
    parser.add_argument("--ylabel", default=None, help="Y-axis label. Defaults to --metric.")
    parser.add_argument("--output_prefix", default=None, help="Output prefix. Defaults inside --root_dir.")
    parser.add_argument("--figsize", default="13.5,6.8", help="Figure size in inches as width,height")
    parser.add_argument("--dpi", type=int, default=300, help="Output image DPI")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        outputs = generate_group_density_plot(
            root_dir=args.root_dir,
            cfg=args.cfg,
            metric=args.metric,
            title=args.title,
            ylabel=args.ylabel,
            output_prefix=args.output_prefix,
            figsize=parse_figsize(args.figsize),
            dpi=args.dpi,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved group stats CSV to: {outputs['csv_path']}")
    print(f"Saved group stats Excel to: {outputs['xlsx_path']}")
    print(f"Saved grouped bar plot to: {outputs['plot_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
