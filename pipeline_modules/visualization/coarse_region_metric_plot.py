"""Plot coarse brain-region metrics from region_signal_analysis_zarr_graph output."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_REGION_IDS = [
    315,  # Isocortex
    1089,  # Hippocampal formation
    698,  # Olfactory areas
    703,  # Cortical subplate
    477,  # Striatum
    803,  # Pallidum
    549,  # Thalamus
    1097,  # Hypothalamus
    313,  # Midbrain
    771,  # Pons
    354,  # Medulla
    512,  # Cerebellum
    1009,  # fiber tracts
    73,  # ventricular systems
]

EXCLUDED_REGION_IDS = {1024, 304325711}  # grooves, retina
DEFAULT_CFG = Path(__file__).resolve().parents[1] / "registration" / "Region_Csv_Rev1_updated.CSV"


def parse_figsize(value: str) -> tuple[float, float]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"figsize must be in width,height format, got: {value}")
    width, height = float(parts[0]), float(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"figsize values must be positive, got: {value}")
    return width, height


def split_name_and_acronym(name_text: str) -> tuple[str, str]:
    name_text = str(name_text).strip()
    if "," not in name_text:
        return name_text, ""
    base_name, acronym = name_text.rsplit(",", 1)
    return base_name.strip(), acronym.strip()


def parse_structure_id_path(path_text: str) -> list[int]:
    path_text = str(path_text).strip()
    if path_text.startswith("/"):
        return [int(part) for part in path_text.strip("/").split("/") if part]
    path_values = ast.literal_eval(path_text)
    return [int(value) for value in path_values]


def load_region_names(cfg_path: str | Path, region_ids: list[int] | None = None) -> pd.DataFrame:
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Region CSV not found: {cfg_path}")

    requested_ids = set(DEFAULT_REGION_IDS if region_ids is None else region_ids)
    rows = []
    region_df = pd.read_csv(cfg_path)
    for _, row in region_df.iterrows():
        region_id = int(row["id"])
        if region_id not in requested_ids:
            continue
        if region_id in EXCLUDED_REGION_IDS:
            continue

        full_name = str(row["name"])
        display_name, acronym = split_name_and_acronym(full_name)
        rows.append(
            {
                "region_id": region_id,
                "region_name": display_name,
                "region_acronym": acronym,
                "excel_name": full_name,
                "structure_id_path": parse_structure_id_path(row["structure_id_path"]),
            }
        )

    found_ids = {int(row["region_id"]) for row in rows}
    missing_ids = [region_id for region_id in requested_ids if region_id not in found_ids]
    if missing_ids:
        raise KeyError(f"Region id(s) not found in CSV: {missing_ids}")

    order_by_id = {region_id: index for index, region_id in enumerate(DEFAULT_REGION_IDS)}
    rows.sort(key=lambda item: order_by_id.get(int(item["region_id"]), len(order_by_id)))
    return pd.DataFrame(rows)


def load_level_sheets(input_excel: str | Path) -> pd.DataFrame:
    input_excel = Path(input_excel)
    if not input_excel.exists():
        raise FileNotFoundError(f"Input Excel not found: {input_excel}")

    sheets = pd.read_excel(input_excel, sheet_name=None)
    level_frames = []
    for sheet_name, frame in sheets.items():
        if not str(sheet_name).startswith("Level_"):
            continue
        level_frame = frame.copy()
        level_frame["source_sheet"] = sheet_name
        level_frames.append(level_frame)

    if not level_frames:
        raise ValueError(f"No Level_* sheets found in Excel workbook: {input_excel}")

    combined = pd.concat(level_frames, ignore_index=True)
    if "Name" not in combined.columns:
        raise ValueError("Input Excel must contain a 'Name' column in its Level_* sheets.")
    return combined


def build_coarse_region_table(
    level_table: pd.DataFrame,
    region_table: pd.DataFrame,
    metric: str,
    *,
    warn_missing: bool = True,
) -> pd.DataFrame:
    if metric not in level_table.columns:
        raise ValueError(
            f"Metric column not found: {metric}. Available columns: {', '.join(map(str, level_table.columns))}"
        )

    output_rows = []
    for order, region in region_table.reset_index(drop=True).iterrows():
        matches = level_table[level_table["Name"] == region["excel_name"]]
        if matches.empty:
            if warn_missing:
                print(
                    f"Warning: region '{region['excel_name']}' was not found in the Excel output; using 0.",
                    file=sys.stderr,
                )
            metric_value = 0.0
            source_sheet = ""
        else:
            metric_values = pd.to_numeric(matches[metric], errors="coerce")
            if metric_values.isna().any():
                raise ValueError(f"Metric column '{metric}' contains non-numeric values for {region['excel_name']}.")
            metric_value = float(metric_values.iloc[0])
            source_sheet = str(matches["source_sheet"].iloc[0]) if "source_sheet" in matches.columns else ""

        output_rows.append(
            {
                "order": int(order) + 1,
                "region_id": int(region["region_id"]),
                "region_name": str(region["region_name"]),
                "region_acronym": str(region["region_acronym"]),
                "excel_name": str(region["excel_name"]),
                "metric": metric,
                "value": metric_value,
                "source_sheet": source_sheet,
            }
        )

    return pd.DataFrame(output_rows)


def resolve_output_prefix(input_excel: str | Path, output_prefix: str | None) -> Path:
    if output_prefix:
        return Path(output_prefix)
    return Path(input_excel).with_suffix("")


def write_coarse_outputs(coarse_table: pd.DataFrame, output_prefix: str | Path) -> tuple[Path, Path]:
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_name(f"{output_prefix.name}_coarse_region_stats.csv")
    xlsx_path = output_prefix.with_name(f"{output_prefix.name}_coarse_region_stats.xlsx")

    coarse_table.to_csv(csv_path, index=False)
    coarse_table.to_excel(xlsx_path, index=False)
    return csv_path, xlsx_path


def plot_coarse_region_bar(
    coarse_table: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
    ylabel: str,
    figsize: tuple[float, float] = (10.5, 5.8),
    dpi: int = 300,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = coarse_table["region_name"].astype(str).tolist()
    values = coarse_table["value"].to_numpy(dtype=np.float64)

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
    colors = plt.cm.viridis(np.linspace(0.18, 0.82, len(values)))
    ax.bar(labels, values, color=colors, edgecolor="#ffffff", linewidth=0.8)
    ax.set_title(title, fontsize=15, pad=12)
    ax.set_ylabel(ylabel, fontsize=11.5)
    ax.grid(axis="y", color="#d8e1e8", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=42)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def generate_coarse_region_outputs(
    *,
    input_excel: str | Path,
    cfg: str | Path = DEFAULT_CFG,
    metric: str,
    title: str,
    ylabel: str | None = None,
    output_prefix: str | Path | None = None,
    figsize: tuple[float, float] = (10.5, 5.8),
    dpi: int = 300,
) -> dict[str, Path | pd.DataFrame]:
    level_table = load_level_sheets(input_excel)
    region_table = load_region_names(cfg)
    coarse_table = build_coarse_region_table(level_table, region_table, metric)

    prefix = resolve_output_prefix(input_excel, str(output_prefix) if output_prefix is not None else None)
    csv_path, xlsx_path = write_coarse_outputs(coarse_table, prefix)

    ylabel = metric if ylabel is None or not str(ylabel).strip() else str(ylabel)
    atlas_plot_path = prefix.with_name(f"{prefix.name}_coarse_region_bar_atlas_order.png")
    sorted_plot_path = prefix.with_name(f"{prefix.name}_coarse_region_bar_sorted.png")

    plot_coarse_region_bar(
        coarse_table,
        output_path=atlas_plot_path,
        title=title,
        ylabel=ylabel,
        figsize=figsize,
        dpi=dpi,
    )
    sorted_table = coarse_table.sort_values("value", ascending=False, kind="stable").reset_index(drop=True)
    plot_coarse_region_bar(
        sorted_table,
        output_path=sorted_plot_path,
        title=f"{title} (sorted)",
        ylabel=ylabel,
        figsize=figsize,
        dpi=dpi,
    )

    return {
        "table": coarse_table,
        "csv_path": csv_path,
        "xlsx_path": xlsx_path,
        "atlas_plot_path": atlas_plot_path,
        "sorted_plot_path": sorted_plot_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize and plot coarse brain-region metrics from region signal Excel output."
    )
    parser.add_argument("--input_excel", required=True, help="Excel output from region_signal_analysis_zarr_graph.py")
    parser.add_argument("--cfg", default=str(DEFAULT_CFG), help="Path to Region_Csv_Rev1_updated.CSV")
    parser.add_argument("--metric", required=True, help="Metric column to plot, e.g. 'Voxel Density'")
    parser.add_argument("--title", required=True, help="Figure title")
    parser.add_argument("--ylabel", default=None, help="Y-axis label. Defaults to --metric.")
    parser.add_argument("--output_prefix", default=None, help="Output prefix. Defaults to input Excel path without suffix.")
    parser.add_argument("--figsize", default="10.5,5.8", help="Figure size in inches as width,height")
    parser.add_argument("--dpi", type=int, default=300, help="Output image DPI")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        outputs = generate_coarse_region_outputs(
            input_excel=args.input_excel,
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

    print(f"Saved coarse region stats CSV to: {outputs['csv_path']}")
    print(f"Saved coarse region stats Excel to: {outputs['xlsx_path']}")
    print(f"Saved atlas-order bar plot to: {outputs['atlas_plot_path']}")
    print(f"Saved sorted bar plot to: {outputs['sorted_plot_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
