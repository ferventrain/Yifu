"""Merge per-sample vessel bilateral results into one summary workbook.

Scans both vessel batch dirs (YF2025063002, YF2026051202). For each sample it
reads the run status file and the Level_0 root row of
results/*_brain_distribution_stats.xlsx (falling back to the quarantined copy
when the live one is absent), then writes an overview workbook plus a Markdown
status report into <YF2025063002 batch>/vessel_bilateral_summary/.

Run after reruns complete to refresh: python _summarize_vessel_results.py
"""
from __future__ import annotations

import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Font

sys.path.insert(0, r"S:\Yifu")

BATCHES = [
    Path(r"S:\Arivis_Analysis\_active\YF2025063002"),
    Path(r"S:\Arivis_Analysis\_active\YF2026051202"),
]
SUMMARY_DIR = BATCHES[0] / "vessel_bilateral_summary"
QUARANTINE = "_quarantine_broken_reg_20260922"
# ch1 vessel fraction measured on Imaris pyramid level 3 (all-voxel counts,
# independent of registration) — the only quantitative column that stays
# trustworthy regardless of registration quality.
LEVEL3_FRACTION = {
    "YF2025063002_MPTP_1": 6.14,
    "YF2025063002_MPTP_2": 6.03,
    "YF2025063002_SEBL_1": 3.78,
    "YF2025063002_SEBL_2": 8.00,
    "YF2025063002_PBS_1": 6.09,
    "YF2025063002_PBS_2": 2.50,
    "YF2026051202_A125": 4.00,
}


def status_first_line(sample_dir: Path) -> str:
    status = sample_dir / "vessel_pipeline.status.txt"
    try:
        return status.read_text(encoding="utf-8", errors="replace").splitlines()[0].strip()
    except (OSError, IndexError):
        return ""


def root_row(path: Path) -> dict:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = list(ws.iter_rows(min_row=1, max_row=2, values_only=True))
    wb.close()
    return dict(zip(rows[0], rows[1]))


def lr_balance(path: Path) -> float | None:
    """Share of hemisphere-labelled voxels assigned Left (None = no L/R data)."""
    data = root_row(path)
    left = float(data.get("Left Total Voxels") or 0)
    right = float(data.get("Right Total Voxels") or 0)
    if left + right <= 0:
        return None
    return 100 * left / (left + right)


def main() -> int:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    overview = []
    for batch in BATCHES:
        for sample_dir in sorted(batch.glob(f"{batch.name}_*")):
            if not (sample_dir / "config.json").exists():
                continue
            name = sample_dir.name
            status = status_first_line(sample_dir)
            live = sorted((sample_dir / "results").glob("*_brain_distribution_stats.xlsx"))
            quarantined = sorted((batch / QUARANTINE / name / "results").glob("*_brain_distribution_stats.xlsx"))
            valid = bool(live) and status.startswith("ALL DONE")
            src = live[0] if live else (quarantined[0] if quarantined else None)
            row = {
                "Sample": name,
                "Group": name.split("_", 1)[1].rsplit("_", 1)[0] if "_" in name else "",
                "Run Status": status or "(not started)",
                "Results Valid": "YES" if valid else "NO (quarantined/invalid)",
                "Source": ("live" if live else ("quarantine" if quarantined else "none")),
                "ch1 Vessel Fraction % (level-3)": LEVEL3_FRACTION.get(name),
                "Root Total Voxels": None,
                "Root Signal Voxels": None,
                "Root Voxel Density": None,
                "Left Share of Label %": None,
            }
            if src is not None:
                try:
                    data = root_row(src)
                    row["Root Total Voxels"] = int(data.get("Total Voxels") or 0)
                    row["Root Signal Voxels"] = int(data.get("Signal Voxels") or 0)
                    row["Root Voxel Density"] = round(float(data.get("Voxel Density") or 0), 4)
                    row["Left Share of Label %"] = (
                        round(lr_balance(src), 1) if lr_balance(src) is not None else None
                    )
                except Exception as exc:  # noqa: BLE001
                    row["Run Status"] += f" (xlsx unreadable: {exc})"
            overview.append(row)

    out = SUMMARY_DIR / f"vessel_bilateral_overview_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "overview"
    header_font = Font(bold=True)
    columns = list(overview[0].keys()) if overview else []
    ws.append(columns)
    for cell in ws[1]:
        cell.font = header_font
        cell.alignment = Alignment(horizontal="left")
    for row in overview:
        ws.append([row.get(col) for col in columns])
    for column, width in zip(range(1, len(columns) + 1), (26, 10, 46, 26, 12, 16, 18, 18, 16, 20)):
        ws.column_dimensions[openpyxl.utils.get_column_letter(column)].width = width
    wb.save(out)
    print("wrote", out)
    Path(SUMMARY_DIR / "overview.json").write_text(
        json.dumps(overview, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
