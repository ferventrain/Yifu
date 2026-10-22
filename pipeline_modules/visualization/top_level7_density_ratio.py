"""Compare two sample density workbooks and plot top log count ratios.

Default mode (``--level auto``) compares a fixed brain parcellation that
divides the Allen structure tree until further division would split regions
into layers: cortical areas keep their subareas (ACAd/ACAv, RSPd/RSPv/RSPagl,
ORBl/ORBm/ORBvl, AId/AIp/AIv, ENTl/ENTm), nuclei stay whole instead of
splitting into parts (BLA/BMA/EP/MEA/COA/BST/LGd...), and divided parents
contribute a remainder row for voxels annotated with the parent label itself.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RANK_BY_CHOICES = ("weighted_log_ratio", "abs_log_ratio", "abs_count_diff")

DEFAULT_PARCELLATION_CSV = Path(__file__).resolve().parents[1] / "registration" / "Region_Csv_Rev2_updated.CSV"

_LAYER_NAME_RE = re.compile(r"layer|stratum|subgranular zone", re.IGNORECASE)
_LAYER_SUFFIX_RE = re.compile(r",\s*(?:1|2/3|4|5|6a|6b)$", re.IGNORECASE)
_PART_SUFFIX_RE = re.compile(r"\b(?:part|parts|zone|zones|division|divisions|shell|core|blade|crest)$", re.IGNORECASE)
# Non-isocortex areas whose part children still split (agreed parcellation exceptions).
_PART_SPLIT_ACRO = {"ENT"}
_ROOT_ACRONYM = "root"
_GREY_ACRONYM = "grey"
_ISOCORTEX_ACRONYM = "Isocortex"


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

    matches = sorted(
        [
            path
            for path in sample_dir.glob("*.xlsx")
            if path.name.lower().endswith("_result.xlsx")
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


def split_raw_name(raw_name: str) -> tuple[str, str]:
    if "," in raw_name:
        base, acronym = raw_name.rsplit(",", 1)
        return base.strip(), acronym.strip()
    return raw_name.strip(), ""


def load_structure_tree(csv_path: str | Path) -> tuple[dict[int, dict], int]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Structure tree CSV not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    missing = {"id", "name", "structure_id_path"}.difference(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required column(s): {sorted(missing)}")

    nodes = {}
    for _, row in frame.iterrows():
        structure_id = int(row["id"])
        path_text = row["structure_id_path"]
        if isinstance(path_text, str):
            path = [int(value) for value in ast.literal_eval(path_text)]
        else:
            path = [int(value) for value in list(path_text)]
        base_name, acronym = split_raw_name(str(row["name"]))
        nodes[structure_id] = {
            "id": structure_id,
            "name": base_name,
            "acronym": acronym,
            "raw_name": str(row["name"]),
            "level": max(0, len(path) - 1),
            "parent": path[-2] if len(path) >= 2 else None,
            "children": [],
        }

    root_id = None
    for structure_id, node in nodes.items():
        parent = node["parent"]
        if parent is None or parent not in nodes:
            root_id = structure_id
            continue
        nodes[parent]["children"].append(node)
    if root_id is None:
        raise ValueError(f"Could not determine root node from CSV: {csv_path}")
    return nodes, root_id


def is_layer_child_name(name: str) -> bool:
    return bool(_LAYER_NAME_RE.search(name) or _LAYER_SUFFIX_RE.search(name.strip()))


def is_part_of_parent(parent_base: str, child_base: str) -> bool:
    parent = parent_base.strip().lower()
    child = child_base.strip()
    if not child.lower().startswith(parent):
        return False
    # children name the part either with a comma ("nucleus, anterior part")
    # or with a space ("Dentate gyrus lateral blade")
    for separator in (",", " "):
        if child.lower().startswith(parent + separator) and _PART_SUFFIX_RE.search(child):
            return True
    return False


def build_default_parcellation(csv_path: str | Path = DEFAULT_PARCELLATION_CSV) -> list[dict]:
    """Divide the structure tree until children would be layers or nucleus parts.

    Cortical subareas still split (isocortex areas + ENT); nuclei whose children
    are parts of the nucleus stay whole. Divided parents get a remainder entry
    that later subtracts the included children rows, so parent-labelled
    annotation voxels are not lost.
    """

    nodes, root_id = load_structure_tree(csv_path)
    root = nodes[root_id]
    regions: list[dict] = []

    def emit(node: dict, remainder_children: list[str] | None = None) -> None:
        regions.append(
            {
                "name": node["raw_name"],
                "acronym": node["acronym"],
                "id": node["id"],
                "level": node["level"],
                "remainder_children": remainder_children,
            }
        )

    def walk(node: dict, under_grey: bool, under_isocortex: bool) -> None:
        children = node["children"]
        if not children:
            emit(node)
            return

        next_isocortex = under_isocortex or node["acronym"] == _ISOCORTEX_ACRONYM
        if not under_grey:
            included_names = []
            for child in children:
                walk(child, under_grey or node["acronym"] == _GREY_ACRONYM, next_isocortex)
                included_names.append(child["raw_name"])
            if node["id"] != root["id"]:
                emit(node, remainder_children=included_names)
            return

        layer_flags = [is_layer_child_name(child["name"]) for child in children]
        if all(layer_flags):
            emit(node)
            return

        non_layer_children = [child for child, flag in zip(children, layer_flags) if not flag]
        part_split_allowed = under_isocortex or node["acronym"] in _PART_SPLIT_ACRO
        if non_layer_children and not part_split_allowed and all(
            is_part_of_parent(node["name"], child["name"]) for child in non_layer_children
        ):
            emit(node)
            return

        included_names = []
        for child in non_layer_children:
            walk(child, under_grey, next_isocortex)
            included_names.append(child["raw_name"])
        if node["id"] != root["id"]:
            emit(node, remainder_children=included_names)

    walk(root, under_grey=False, under_isocortex=False)
    return regions


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


def load_all_level_counts(excel_path: str | Path, metric: str) -> dict[str, float]:
    """Sum the metric over every Level_* sheet, keyed by workbook region Name."""

    excel_path = Path(excel_path)
    try:
        sheet_names = pd.ExcelFile(excel_path).sheet_names
    except Exception as exc:
        raise ValueError(f"Cannot read Excel workbook: {excel_path}") from exc

    level_sheets = [name for name in sheet_names if str(name).startswith("Level_")]
    if not level_sheets:
        raise ValueError(f"No Level_* sheets found in Excel workbook: {excel_path}")

    counts: dict[str, float] = {}
    for sheet_name in level_sheets:
        frame = pd.read_excel(excel_path, sheet_name=sheet_name)
        required = {"Name", metric}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{excel_path}:{sheet_name} missing required column(s): {sorted(missing)}")
        for name, value in zip(frame["Name"], pd.to_numeric(frame[metric], errors="coerce").fillna(0.0)):
            counts[str(name)] = counts.get(str(name), 0.0) + float(value)
    return counts


def resolve_parcellation_counts(regions: list[dict], counts: dict[str, float]) -> pd.DataFrame:
    """Map parcellation regions to counts; remainder rows subtract included children."""

    rows = []
    for region in regions:
        value = counts.get(region["name"], 0.0)
        if region["remainder_children"] is not None:
            value -= sum(counts.get(child_name, 0.0) for child_name in region["remainder_children"])
        rows.append({"Name": region["name"], "count": max(0.0, value)})
    return pd.DataFrame(rows, columns=["Name", "count"])


def _assemble_ratio_table(
    counts_a: pd.DataFrame,
    counts_b: pd.DataFrame,
    *,
    sample_a: str,
    sample_b: str,
    pseudocount: float,
    min_count: float,
    top_n: int,
    rank_by: str,
) -> pd.DataFrame:
    merged = counts_a.merge(counts_b, on="Name", how="outer").fillna(0.0)
    merged["sample_a"] = sample_a
    merged["sample_b"] = sample_b
    merged["log_ratio"] = np.log((merged["count_a"].astype(float) + pseudocount) / (merged["count_b"].astype(float) + pseudocount))
    merged["abs_log_ratio"] = merged["log_ratio"].abs()
    merged["abs_count_diff"] = (merged["count_a"].astype(float) - merged["count_b"].astype(float)).abs()
    merged["total_count"] = merged["count_a"].astype(float) + merged["count_b"].astype(float)
    merged["weighted_log_ratio"] = merged["abs_log_ratio"] * np.log1p(merged["total_count"])
    merged = merged[merged["total_count"] >= float(min_count)].copy()
    merged["rank_by"] = rank_by
    merged = merged.sort_values([rank_by, "Name"], ascending=[False, True], kind="stable").reset_index(drop=True)
    return merged.head(top_n).copy()


def _validate_ratio_args(pseudocount: float, min_count: float, top_n: int, rank_by: str) -> None:
    if pseudocount < 0:
        raise ValueError("pseudocount must be >= 0")
    if min_count < 0:
        raise ValueError("min_count must be >= 0")
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if rank_by not in RANK_BY_CHOICES:
        raise ValueError(f"rank_by must be one of {RANK_BY_CHOICES}, got: {rank_by}")


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
    _validate_ratio_args(pseudocount, min_count, top_n, rank_by)
    table_a = load_level_table(excel_a, level, metric).rename(columns={metric: "count_a"})
    table_b = load_level_table(excel_b, level, metric).rename(columns={metric: "count_b"})
    return _assemble_ratio_table(
        table_a,
        table_b,
        sample_a=Path(excel_a).parent.name,
        sample_b=Path(excel_b).parent.name,
        pseudocount=pseudocount,
        min_count=min_count,
        top_n=top_n,
        rank_by=rank_by,
    )


def build_auto_ratio_table(
    excel_a: str | Path,
    excel_b: str | Path,
    *,
    metric: str = "Signal Count",
    pseudocount: float = 1.0,
    min_count: float = 100.0,
    top_n: int = 10,
    rank_by: str = "weighted_log_ratio",
    csv_path: str | Path = DEFAULT_PARCELLATION_CSV,
) -> pd.DataFrame:
    _validate_ratio_args(pseudocount, min_count, top_n, rank_by)
    regions = build_default_parcellation(csv_path)
    table_a = resolve_parcellation_counts(regions, load_all_level_counts(excel_a, metric)).rename(columns={"count": "count_a"})
    table_b = resolve_parcellation_counts(regions, load_all_level_counts(excel_b, metric)).rename(columns={"count": "count_b"})
    return _assemble_ratio_table(
        table_a,
        table_b,
        sample_a=Path(excel_a).parent.name,
        sample_b=Path(excel_b).parent.name,
        pseudocount=pseudocount,
        min_count=min_count,
        top_n=top_n,
        rank_by=rank_by,
    )


def _format_level_label(level: int | str) -> str:
    if isinstance(level, str):
        if level.lower() != "auto":
            raise ValueError(f"level must be an integer or 'auto', got: {level!r}")
        return "auto"
    return f"level{int(level)}"


def write_ratio_table(table: pd.DataFrame, output_prefix: str | Path) -> tuple[Path, Path]:
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    label = _format_level_label(table["level"].iloc[0]) if "level" in table.columns and not table.empty else "level6"
    csv_path = output_prefix.with_name(f"{output_prefix.name}_{label}_ratio_top.csv")
    xlsx_path = output_prefix.with_name(f"{output_prefix.name}_{label}_ratio_top.xlsx")
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
    level = table["level"].iloc[0] if "level" in table.columns and not table.empty else 6
    if isinstance(level, str) and level.lower() == "auto":
        ax.set_xlabel("Top regions (default parcellation)", fontsize=11)
    else:
        ax.set_xlabel(f"Top Level_{int(level)} regions", fontsize=11)
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
    level: int | str = "auto",
    top_n: int = 10,
    pseudocount: float = 1.0,
    min_count: float = 100.0,
    rank_by: str = "weighted_log_ratio",
    output_prefix: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (12.0, 6.2),
    dpi: int = 300,
    csv_path: str | Path = DEFAULT_PARCELLATION_CSV,
) -> dict[str, Path | pd.DataFrame]:
    root_dir = Path(root_dir)
    excel_a, excel_b = find_two_sample_excels(root_dir)
    is_auto = isinstance(level, str) and level.lower() == "auto"
    label = _format_level_label(level)
    output_prefix = Path(output_prefix) if output_prefix else root_dir / root_dir.name
    if is_auto:
        title = title or f"Top {top_n} log ratio (default parcellation): {excel_a.parent.name} / {excel_b.parent.name}"
        table = build_auto_ratio_table(
            excel_a,
            excel_b,
            metric=metric,
            pseudocount=pseudocount,
            min_count=min_count,
            top_n=top_n,
            rank_by=rank_by,
            csv_path=csv_path,
        )
    else:
        title = title or f"Top {top_n} Level_{int(level)} log ratio: {excel_a.parent.name} / {excel_b.parent.name}"
        table = build_level_ratio_table(
            excel_a,
            excel_b,
            metric=metric,
            level=int(level),
            pseudocount=pseudocount,
            min_count=min_count,
            top_n=top_n,
            rank_by=rank_by,
        )
    table["level"] = "auto" if is_auto else int(level)
    csv_path_out, xlsx_path = write_ratio_table(table, output_prefix)
    plot_path = output_prefix.with_name(f"{output_prefix.name}_{label}_ratio_top{top_n}.png")
    plot_level_ratio(
        table,
        output_path=plot_path,
        metric=metric,
        title=title,
        figsize=figsize,
        dpi=dpi,
    )
    return {"table": table, "csv_path": csv_path_out, "xlsx_path": xlsx_path, "plot_path": plot_path}


build_level7_ratio_table = build_level_ratio_table
generate_top_level7_ratio_plot = generate_top_level_ratio_plot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two child sample density workbooks and plot top log count ratios."
    )
    parser.add_argument("--root_dir", required=True, help="Folder containing exactly two sample_dir child folders")
    parser.add_argument("--metric", default="Signal Count", help="Metric/count column to compare")
    parser.add_argument(
        "--level",
        default="auto",
        help="Level sheet number, or 'auto' for the default parcellation (divide until non-layer; "
        "cortical subareas split, nuclei stay whole, remainder rows keep parent-labelled voxels). Default: auto",
    )
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
        level: int | str
        if str(args.level).lower() == "auto":
            level = "auto"
        else:
            try:
                level = int(args.level)
            except ValueError as exc:
                raise ValueError(f"--level must be an integer or 'auto', got: {args.level}") from exc
        outputs = generate_top_level_ratio_plot(
            root_dir=args.root_dir,
            metric=args.metric,
            level=level,
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
