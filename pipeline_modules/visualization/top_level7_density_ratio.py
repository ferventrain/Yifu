"""Compare two sample density workbooks and plot top Level_6 log count ratios."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RANK_BY_CHOICES = ("weighted_log_ratio", "abs_log_ratio", "abs_count_diff")


def parse_figsize(value: str) -> tuple[float, float]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"figsize must be width,height, got: {value}")
    width, height = float(parts[0]), float(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"figsize values must be positive, got: {value}")
    return width, height


def find_density_excel(sample_dir: str | Path) -> Path:
    sample_dir = Path(sample_dir)
    if not sample_dir.exists() or not sample_dir.is_dir():
        raise NotADirectoryError(f"Sample directory not found: {sample_dir}")

    exact = sample_dir / f"{sample_dir.name}_density_result.xlsx"
    if exact.exists():
        return exact

    matches = sorted(
        [
            path
            for path in sample_dir.glob("*.xlsx")
            if "density" in path.name.lower()
            and not path.name.startswith("~$")
            and "coarse_region" not in path.name.lower()
            and "level_ratio" not in path.name.lower()
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"No density Excel workbook found in sample directory: {sample_dir}")
    return matches[0]


def find_two_sample_excels(root_dir: str | Path) -> tuple[Path, Path]:
    root_dir = Path(root_dir)
    if not root_dir.exists() or not root_dir.is_dir():
        raise NotADirectoryError(f"Input root folder not found: {root_dir}")

    sample_dirs = sorted([path for path in root_dir.iterdir() if path.is_dir()], key=lambda path: path.name.lower())
    excels = []
    for sample_dir in sample_dirs:
        try:
            excels.append(find_density_excel(sample_dir))
        except FileNotFoundError:
            continue

    if len(excels) != 2:
        raise ValueError(
            f"Expected exactly 2 child sample folders with density Excel files under {root_dir}, found {len(excels)}."
        )
    return excels[0], excels[1]


def load_level_table(excel_path: str | Path, level: int, metric: str) -> pd.DataFrame:
    excel_path = Path(excel_path)
    sheet_name = f"Level_{int(level)}"
    try:
        frame = pd.read_excel(excel_path, sheet_name=sheet_name)
    except ValueError as exc:
        raise ValueError(f"{excel_path} does not contain sheet {sheet_name}") from exc

    required = {"Name", metric}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{excel_path}:{sheet_name} missing required column(s): {sorted(missing)}")

    output = frame[["Name", metric]].copy()
    output[metric] = pd.to_numeric(output[metric], errors="coerce").fillna(0.0)
    output = output.groupby("Name", as_index=False, sort=False)[metric].sum()
    return output


def build_level_ratio_table(
    excel_a: str | Path,
    excel_b: str | Path,
    *,
    metric: str = "Signal Count",
    level: int = 6,
    pseudocount: float = 1.0,
    min_count: float = 100.0,
    top_n: int = 10,
    rank_by: str = "weighted_log_ratio",
) -> pd.DataFrame:
    if pseudocount < 0:
        raise ValueError("pseudocount must be >= 0")
    if min_count < 0:
        raise ValueError("min_count must be >= 0")
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if rank_by not in RANK_BY_CHOICES:
        raise ValueError(f"rank_by must be one of {RANK_BY_CHOICES}, got: {rank_by}")

    table_a = load_level_table(excel_a, level, metric).rename(columns={metric: "count_a"})
    table_b = load_level_table(excel_b, level, metric).rename(columns={metric: "count_b"})
    merged = table_a.merge(table_b, on="Name", how="outer").fillna(0.0)
    merged["sample_a"] = Path(excel_a).parent.name
    merged["sample_b"] = Path(excel_b).parent.name
    merged["log_ratio"] = np.log((merged["count_a"].astype(float) + pseudocount) / (merged["count_b"].astype(float) + pseudocount))
    merged["abs_log_ratio"] = merged["log_ratio"].abs()
    merged["abs_count_diff"] = (merged["count_a"].astype(float) - merged["count_b"].astype(float)).abs()
    merged["total_count"] = merged["count_a"].astype(float) + merged["count_b"].astype(float)
    merged["weighted_log_ratio"] = merged["abs_log_ratio"] * np.log1p(merged["total_count"])
    merged = merged[merged["total_count"] >= float(min_count)].copy()
    merged["rank_by"] = rank_by
    merged = merged.sort_values([rank_by, "Name"], ascending=[False, True], kind="stable").reset_index(drop=True)
    return merged.head(top_n).copy()


def write_ratio_table(table: pd.DataFrame, output_prefix: str | Path) -> tuple[Path, Path]:
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    level = int(table["level"].iloc[0]) if "level" in table.columns and not table.empty else 6
    csv_path = output_prefix.with_name(f"{output_prefix.name}_level{level}_ratio_top.csv")
    xlsx_path = output_prefix.with_name(f"{output_prefix.name}_level{level}_ratio_top.xlsx")
    table.to_csv(csv_path, index=False)
    table.to_excel(xlsx_path, index=False)
    return csv_path, xlsx_path


def plot_level_ratio(
    table: pd.DataFrame,
    *,
    output_path: str | Path,
    metric: str,
    title: str,
    figsize: tuple[float, float],
    dpi: int,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = table["Name"].astype(str).tolist()
    values = table["log_ratio"].to_numpy(dtype=np.float64)
    colors = np.where(values >= 0, "#2f6f9f", "#b84a4a")

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#263238",
            "axes.linewidth": 1.0,
            "axes.labelcolor": "#17212b",
            "axes.titleweight": "semibold",
            "xtick.color": "#263238",
            "ytick.color": "#263238",
            "font.size": 10.0,
            "font.family": "DejaVu Sans",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    x = np.arange(len(labels), dtype=np.float64)
    ax.bar(x, values, color=colors, edgecolor="#ffffff", linewidth=0.7)
    ax.axhline(0.0, color="#263238", linewidth=0.9)
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_ylabel(f"log({metric} A / {metric} B)", fontsize=11)
    level = int(table["level"].iloc[0]) if "level" in table.columns and not table.empty else 6
    ax.set_xlabel(f"Top Level_{level} regions", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=42, ha="right")
    ax.grid(axis="y", color="#d8e1e8", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def generate_top_level_ratio_plot(
    *,
    root_dir: str | Path,
    metric: str = "Signal Count",
    level: int = 6,
    top_n: int = 10,
    pseudocount: float = 1.0,
    min_count: float = 100.0,
    rank_by: str = "weighted_log_ratio",
    output_prefix: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (12.0, 6.2),
    dpi: int = 300,
) -> dict[str, Path | pd.DataFrame]:
    root_dir = Path(root_dir)
    excel_a, excel_b = find_two_sample_excels(root_dir)
    output_prefix = Path(output_prefix) if output_prefix else root_dir / f"{root_dir.name}_level{int(level)}_ratio"
    title = title or f"Top {top_n} Level_{level} log ratio: {excel_a.parent.name} / {excel_b.parent.name}"

    table = build_level_ratio_table(
        excel_a,
        excel_b,
        metric=metric,
        level=level,
        pseudocount=pseudocount,
        min_count=min_count,
        top_n=top_n,
        rank_by=rank_by,
    )
    table["level"] = int(level)
    csv_path, xlsx_path = write_ratio_table(table, output_prefix)
    plot_path = output_prefix.with_name(f"{output_prefix.name}_level{int(level)}_ratio_top{top_n}.png")
    plot_level_ratio(
        table,
        output_path=plot_path,
        metric=metric,
        title=title,
        figsize=figsize,
        dpi=dpi,
    )
    return {"table": table, "csv_path": csv_path, "xlsx_path": xlsx_path, "plot_path": plot_path}


build_level7_ratio_table = build_level_ratio_table
generate_top_level7_ratio_plot = generate_top_level_ratio_plot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two child sample density workbooks and plot top Level_6 log count ratios."
    )
    parser.add_argument("--root_dir", required=True, help="Folder containing exactly two sample_dir child folders")
    parser.add_argument("--metric", default="Signal Count", help="Metric/count column to compare")
    parser.add_argument("--level", type=int, default=6, help="Level sheet to compare. Default: 6")
    parser.add_argument("--top_n", type=int, default=10, help="Number of largest absolute differences to plot")
    parser.add_argument("--pseudocount", type=float, default=1.0, help="Added to both counts before log ratio")
    parser.add_argument(
        "--min_count",
        type=float,
        default=100.0,
        help="Minimum combined count across both samples required before ranking. Default: 100",
    )
    parser.add_argument(
        "--rank_by",
        choices=RANK_BY_CHOICES,
        default="weighted_log_ratio",
        help="Ranking score for top regions. Default weights log ratio by log1p(countA + countB).",
    )
    parser.add_argument("--output_prefix", default=None, help="Output prefix. Defaults inside --root_dir.")
    parser.add_argument("--title", default=None, help="Figure title")
    parser.add_argument("--figsize", default="12,6.2", help="Figure size in inches as width,height")
    parser.add_argument("--dpi", type=int, default=300, help="Output image DPI")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        outputs = generate_top_level_ratio_plot(
            root_dir=args.root_dir,
            metric=args.metric,
            level=args.level,
            top_n=args.top_n,
            pseudocount=args.pseudocount,
            min_count=args.min_count,
            rank_by=args.rank_by,
            output_prefix=args.output_prefix,
            title=args.title,
            figsize=parse_figsize(args.figsize),
            dpi=args.dpi,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved ratio CSV to: {outputs['csv_path']}")
    print(f"Saved ratio Excel to: {outputs['xlsx_path']}")
    print(f"Saved ratio plot to: {outputs['plot_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
