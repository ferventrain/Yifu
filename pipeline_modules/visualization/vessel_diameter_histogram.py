"""Publication-ready vessel diameter histogram from kimimaro branch metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


DEFAULT_BRANCH_CSV = "vessel_branch_metrics.csv"
DEFAULT_OUTPUT = "vessel_diameter_histogram.png"


def _parse_pair(value: str, name: str) -> tuple[float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"{name} must be in 'a,b' format, got: {value}")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError(f"{name} must contain numeric values, got: {value}") from exc


def parse_figsize(value: str) -> tuple[float, float]:
    return _parse_pair(value, "figsize")


def parse_xlim(value: str | None) -> tuple[float, float] | None:
    if value is None:
        return None
    lo, hi = _parse_pair(value, "xlim")
    if hi <= lo:
        raise ValueError(f"xlim upper bound must be greater than lower bound, got: {value}")
    return lo, hi


def load_branch_diameters(branch_csv_path: str | Path) -> np.ndarray:
    branch_csv_path = Path(branch_csv_path)
    if not branch_csv_path.exists():
        raise FileNotFoundError(f"Branch CSV not found: {branch_csv_path}")

    table = pd.read_csv(branch_csv_path)
    required_column = "mean_radius_um"
    if required_column not in table.columns:
        raise ValueError(
            f"Branch CSV must contain '{required_column}' column: {branch_csv_path}"
        )

    radii = pd.to_numeric(table[required_column], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(radii) & (radii > 0)
    diameters = radii[valid] * 2.0
    if diameters.size == 0:
        raise ValueError(
            "No valid vessel diameters found after filtering NaN, inf, and non-positive values."
        )
    return diameters


def summarize_diameters(diameters_um: np.ndarray) -> dict[str, float]:
    return {
        "n": int(diameters_um.size),
        "mean": float(np.mean(diameters_um)),
        "median": float(np.median(diameters_um)),
        "std": float(np.std(diameters_um, ddof=1)) if diameters_um.size > 1 else 0.0,
        "min": float(np.min(diameters_um)),
        "max": float(np.max(diameters_um)),
    }


def resolve_output_path(branch_csv_path: Path, output: str | None) -> Path:
    if output:
        return Path(output)
    return branch_csv_path.with_name(DEFAULT_OUTPUT)


def plot_vessel_diameter_histogram(
    diameters_um: np.ndarray,
    *,
    output_path: str | Path,
    bins: int = 24,
    dpi: int = 300,
    title: str = "Vessel Diameter Distribution",
    xlabel: str = "Vessel diameter (um)",
    ylabel: str = "Section count",
    figsize: tuple[float, float] = (8.4, 5.6),
    xlim: tuple[float, float] | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = summarize_diameters(diameters_um)

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#243447",
            "axes.linewidth": 1.0,
            "axes.labelcolor": "#16202a",
            "axes.titleweight": "semibold",
            "xtick.color": "#243447",
            "ytick.color": "#243447",
            "font.size": 11,
            "font.family": "DejaVu Sans",
            "legend.labelcolor": "#243447",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    hist_color = "#325d88"
    edge_color = "#17324d"
    counts, bin_edges = np.histogram(diameters_um, bins=bins)

    ax.hist(
        diameters_um,
        bins=bin_edges,
        color=hist_color,
        edgecolor="#f4f7fa",
        linewidth=0.9,
        alpha=0.95,
    )
    ax.stairs(counts, bin_edges, color=edge_color, linewidth=1.5, alpha=0.95)

    ax.set_title(title, fontsize=16, pad=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    if xlim is not None:
        ax.set_xlim(*xlim)

    ax.grid(axis="y", color="#d7e3ea", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    stats_text = "\n".join(
        [
            f"n = {summary['n']}",
            f"std = {summary['std']:.2f} um",
            f"min = {summary['min']:.2f} um",
            f"max = {summary['max']:.2f} um",
        ]
    )
    ax.text(
        0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        color="#163042",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#fbfdfe",
            "edgecolor": "#ccd9e2",
            "linewidth": 0.9,
        },
    )

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a publication-ready vessel diameter histogram from kimimaro branch metrics."
    )
    parser.add_argument(
        "--branch_csv",
        default=DEFAULT_BRANCH_CSV,
        help="Path to vessel_branch_metrics.csv",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path. Defaults to vessel_diameter_histogram.png next to the CSV.",
    )
    parser.add_argument("--bins", type=int, default=24, help="Number of histogram bins")
    parser.add_argument("--dpi", type=int, default=300, help="Output image DPI")
    parser.add_argument("--title", default="Vessel Diameter Distribution")
    parser.add_argument("--xlabel", default="Vessel diameter (um)")
    parser.add_argument("--ylabel", default="Section count")
    parser.add_argument(
        "--figsize",
        default="8.4,5.6",
        help="Figure size in inches as width,height",
    )
    parser.add_argument(
        "--xlim",
        default=None,
        help="Optional x-axis limits as min,max",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        branch_csv_path = Path(args.branch_csv)
        diameters_um = load_branch_diameters(branch_csv_path)
        output_path = resolve_output_path(branch_csv_path, args.output)
        plot_vessel_diameter_histogram(
            diameters_um,
            output_path=output_path,
            bins=args.bins,
            dpi=args.dpi,
            title=args.title,
            xlabel=args.xlabel,
            ylabel=args.ylabel,
            figsize=parse_figsize(args.figsize),
            xlim=parse_xlim(args.xlim),
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved vessel diameter histogram to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
