#!/usr/bin/env python3
"""Plot per-vertebra spinal cord signal statistics.

Reads the flat CSV produced by ``spinal_segment_signal_stats`` (sheets flattened
with a Level column) and renders a three-panel figure per segment:

  panel 1: signal count bars
  panel 2: signal volume (mm3) bars
  panel 3: count density (count/mm3) bars + voxel density (%) line

Bars are colored by vertebral region group (C/T/L/S/Co).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pipeline_modules.registration.spinalj_labels_to_native import _segment_sort_key

GROUP_COLORS = {
    "C": "#4C72B0",
    "T": "#DD8452",
    "L": "#55A868",
    "S": "#C44E52",
    "Co": "#8172B3",
}


def group_of(name: str) -> str:
    prefix = "".join(char for char in str(name) if char.isalpha())
    return prefix if prefix in GROUP_COLORS else "Co"


def load_segment_frame(stats_csv: str | Path, level: int = 1) -> pd.DataFrame:
    frame = pd.read_csv(stats_csv)
    frame = frame[frame["Level"] == level].copy()
    frame["_sort_key"] = frame["Name"].map(_segment_sort_key)
    return frame.sort_values("_sort_key").reset_index(drop=True)


def plot_spinal_segment_stats(
    stats_csv: str | Path,
    output_png: str | Path,
    *,
    level: int = 1,
    title: str | None = None,
) -> Path:
    output_png = Path(output_png)
    segments = load_segment_frame(stats_csv, level=level)
    if segments.empty:
        raise ValueError(f"No level-{level} rows found in {stats_csv}")

    names = segments["Name"].tolist()
    colors = [GROUP_COLORS[group_of(name)] for name in names]
    x = range(len(names))

    fig, (ax_count, ax_volume, ax_density) = plt.subplots(
        3, 1, figsize=(max(12, len(names) * 0.42), 12.5), sharex=True
    )

    count_bars = ax_count.bar(x, segments["Signal Count"], color=colors, width=0.72)
    ax_count.set_ylabel("Signal count")
    ax_count.set_title(title or "Per-vertebra spinal cord signal statistics")

    ax_volume.bar(x, segments["Signal Volume (mm3)"], color=colors, width=0.72)
    ax_volume.set_ylabel("Signal volume (mm$^3$)")

    density_bars = ax_density.bar(
        x, segments["Count Density (count/mm3)"], color=colors, width=0.72
    )
    ax_density.set_ylabel("Count density (count/mm$^3$)")
    ax_density_voxel = ax_density.twinx()
    ax_density_voxel.plot(
        x, segments["Voxel Density"] * 100.0, color="#333333", marker="o", lw=1.4, ms=3.5
    )
    ax_density_voxel.set_ylabel("Voxel density (%)")
    ax_density.set_xticks(list(x))
    ax_density.set_xticklabels(names, rotation=90, fontsize=8)
    ax_density.set_xlabel("Vertebral segment")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=group)
        for group, color in GROUP_COLORS.items()
    ]
    ax_count.legend(handles=handles, title="Region", loc="upper right", fontsize=8)

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    return output_png


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats_csv", required=True, help="spinal_segment_stats flat CSV")
    parser.add_argument("--output_png", required=True, help="Output PNG path")
    parser.add_argument("--level", type=int, default=1, help="Hierarchy level to plot (1 = per vertebra)")
    parser.add_argument("--title", default="", help="Optional chart title")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = plot_spinal_segment_stats(
        args.stats_csv, args.output_png, level=args.level, title=args.title or None
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
