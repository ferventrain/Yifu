"""Summarize density workbook signal counts for configured region groups."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path


DEFAULT_CFG = Path(__file__).resolve().parents[1] / "registration" / "Region_Csv_Rev1_updated.CSV"
DEFAULT_GROUPS = Path(__file__).resolve().parents[2] / "config" / "region_groups.json"


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
            and "region_group_signal_count" not in path.name.lower()
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"No density Excel workbook found in sample directory: {sample_dir}")
    return matches[0]


def parse_acronym_text(acronym_text: object) -> str:
    try:
        acronym_values = ast.literal_eval(str(acronym_text))
        if isinstance(acronym_values, list) and acronym_values:
            return str(acronym_values[-1]).strip()
    except Exception:
        pass
    return str(acronym_text).strip()


def split_name_and_acronym(name_text: object) -> tuple[str, str]:
    text = str(name_text).strip()
    if "," not in text:
        return text, ""
    base_name, acronym = text.rsplit(",", 1)
    return base_name.strip(), acronym.strip()


def load_region_groups(path: str | Path) -> dict[str, list[str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Region groups JSON not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Region groups JSON must be an object, got: {type(payload).__name__}")

    groups: dict[str, list[str]] = {}
    for group_name, spec in payload.items():
        if isinstance(spec, list):
            raw_acronyms = spec
        elif isinstance(spec, dict):
            raw_acronyms = spec.get("acronyms", [])
        else:
            raise ValueError(f"Group {group_name!r} must be a list or an object with acronyms.")
        if not isinstance(raw_acronyms, list):
            raise ValueError(f"Group {group_name!r} acronyms must be a list.")

        acronyms = [str(item).strip() for item in raw_acronyms if str(item).strip()]
        if not acronyms:
            raise ValueError(f"Group {group_name!r} has no acronyms.")
        groups[str(group_name)] = acronyms
    return groups


def load_region_lookup(cfg_path: str | Path) -> pd.DataFrame:
    import pandas as pd

    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Region CSV not found: {cfg_path}")

    region_df = pd.read_csv(cfg_path)
    required = {"id", "name", "acronym"}
    missing = required.difference(region_df.columns)
    if missing:
        raise ValueError(f"Region CSV missing required column(s): {sorted(missing)}")

    rows = []
    for _, row in region_df.iterrows():
        display_name, name_acronym = split_name_and_acronym(row["name"])
        acronym = parse_acronym_text(row["acronym"]) or name_acronym
        rows.append(
            {
                "region_id": int(row["id"]),
                "region_name": display_name,
                "region_acronym": acronym,
                "excel_name": str(row["name"]),
            }
        )
    return pd.DataFrame(rows)


def load_density_metric_table(input_excel: str | Path, metric: str = "Signal Count") -> pd.DataFrame:
    import pandas as pd

    input_excel = Path(input_excel)
    if not input_excel.exists():
        raise FileNotFoundError(f"Input Excel not found: {input_excel}")

    sheets = pd.read_excel(input_excel, sheet_name=None)
    frames = []
    for sheet_name, frame in sheets.items():
        if not str(sheet_name).startswith("Level_"):
            continue
        required = {"Name", metric}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{input_excel}:{sheet_name} missing required column(s): {sorted(missing)}")
        level_frame = frame[["Name", metric]].copy()
        level_frame["source_sheet"] = str(sheet_name)
        frames.append(level_frame)

    if not frames:
        raise ValueError(f"No Level_* sheets found in Excel workbook: {input_excel}")

    table = pd.concat(frames, ignore_index=True)
    table[metric] = pd.to_numeric(table[metric], errors="coerce")
    if table[metric].isna().any():
        bad_names = table.loc[table[metric].isna(), "Name"].astype(str).head(5).tolist()
        raise ValueError(f"Metric column {metric!r} contains non-numeric values, examples: {bad_names}")
    return table


def build_region_group_signal_count_table(
    input_excel: str | Path,
    *,
    groups_json: str | Path = DEFAULT_GROUPS,
    cfg: str | Path = DEFAULT_CFG,
    metric: str = "Signal Count",
    warn_missing: bool = True,
) -> pd.DataFrame:
    import pandas as pd

    groups = load_region_groups(groups_json)
    region_lookup = load_region_lookup(cfg)
    density_table = load_density_metric_table(input_excel, metric)

    density_by_name = density_table.groupby("Name", sort=False).agg(
        value=(metric, "first"),
        source_sheet=("source_sheet", "first"),
    )
    region_by_acronym = region_lookup.drop_duplicates("region_acronym", keep="first").set_index("region_acronym")

    rows = []
    for order, (group_name, acronyms) in enumerate(groups.items(), start=1):
        group_total = 0.0
        matched = []
        missing = []
        for acronym in acronyms:
            if acronym not in region_by_acronym.index:
                missing.append(acronym)
                continue

            region = region_by_acronym.loc[acronym]
            excel_name = str(region["excel_name"])
            if excel_name not in density_by_name.index:
                missing.append(acronym)
                continue

            value = float(density_by_name.loc[excel_name, "value"])
            group_total += value
            matched.append(
                {
                    "acronym": acronym,
                    "region_id": int(region["region_id"]),
                    "excel_name": excel_name,
                    "source_sheet": str(density_by_name.loc[excel_name, "source_sheet"]),
                    "signal_count": value,
                }
            )

        if missing and warn_missing:
            print(f"Warning: group {group_name!r} missing acronym(s): {missing}", file=sys.stderr)

        rows.append(
            {
                "order": order,
                "sample": Path(input_excel).parent.name,
                "group": group_name,
                "metric": metric,
                "signal_count": group_total,
                "matched_acronyms": ",".join(item["acronym"] for item in matched),
                "missing_acronyms": ",".join(missing),
                "matched_regions": "; ".join(item["excel_name"] for item in matched),
                "density_excel": str(input_excel),
            }
        )
    return pd.DataFrame(rows)


def write_outputs(table: pd.DataFrame, output_prefix: str | Path) -> tuple[Path, Path]:
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_name(f"{output_prefix.name}_region_group_signal_count.csv")
    xlsx_path = output_prefix.with_name(f"{output_prefix.name}_region_group_signal_count.xlsx")
    table.to_csv(csv_path, index=False)
    table.to_excel(xlsx_path, index=False)
    return csv_path, xlsx_path


def summarize_sample_region_group_signal_count(
    *,
    sample_dir: str | Path,
    groups_json: str | Path = DEFAULT_GROUPS,
    cfg: str | Path = DEFAULT_CFG,
    metric: str = "Signal Count",
    input_excel: str | Path | None = None,
    output_prefix: str | Path | None = None,
) -> dict[str, Path | pd.DataFrame]:
    sample_dir = Path(sample_dir)
    excel_path = Path(input_excel) if input_excel else find_density_excel(sample_dir)
    output_prefix = Path(output_prefix) if output_prefix else sample_dir / sample_dir.name
    table = build_region_group_signal_count_table(excel_path, groups_json=groups_json, cfg=cfg, metric=metric)
    csv_path, xlsx_path = write_outputs(table, output_prefix)
    return {"table": table, "csv_path": csv_path, "xlsx_path": xlsx_path, "density_excel": excel_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize Signal Count for region groups from a sample_dir density workbook."
    )
    parser.add_argument("--sample_dir", required=True, help="Sample directory containing a density *.xlsx workbook")
    parser.add_argument("--groups_json", default=str(DEFAULT_GROUPS), help="Path to region_groups.json")
    parser.add_argument("--cfg", default=str(DEFAULT_CFG), help="Path to Region_Csv_Rev1_updated.CSV")
    parser.add_argument("--metric", default="Signal Count", help="Metric column to sum. Default: Signal Count")
    parser.add_argument("--input_excel", default=None, help="Optional explicit density workbook path")
    parser.add_argument("--output_prefix", default=None, help="Output prefix. Defaults to sample_dir/sample_dir_name")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        outputs = summarize_sample_region_group_signal_count(
            sample_dir=args.sample_dir,
            groups_json=args.groups_json,
            cfg=args.cfg,
            metric=args.metric,
            input_excel=args.input_excel,
            output_prefix=args.output_prefix,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Using density Excel: {outputs['density_excel']}")
    print(f"Saved region group CSV to: {outputs['csv_path']}")
    print(f"Saved region group Excel to: {outputs['xlsx_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
