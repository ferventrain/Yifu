"""Quarantine registration products broken by the unit-affine / stride bugs.

Moves (never deletes) the label/hemisphere/warped/transforms/coarse-zarr/25um
nii/results-xlsx of each vessel sample into a quarantine tree, keeping
ch1.zarr, ch1_mask.zarr, config and logs so the rerun only redoes steps 2-5.
"""
import json
import shutil
from pathlib import Path

SAMPLES = [
    r"S:\Arivis_Analysis\_active\YF2025063002\YF2025063002_MPTP_1",
    r"S:\Arivis_Analysis\_active\YF2025063002\YF2025063002_MPTP_2",
    r"S:\Arivis_Analysis\_active\YF2025063002\YF2025063002_SEBL_1",
    r"S:\Arivis_Analysis\_active\YF2025063002\YF2025063002_SEBL_2",
    r"S:\Arivis_Analysis\_active\YF2025063002\YF2025063002_PBS_1",
    r"S:\Arivis_Analysis\_active\YF2025063002\YF2025063002_PBS_2",
    r"S:\Arivis_Analysis\_active\YF2026051202\YF2026051202_A125",
]
QUARANTINE_ROOTS = {
    r"S:\Arivis_Analysis\_active\YF2025063002": Path(r"S:\Arivis_Analysis\_active\YF2025063002\_quarantine_broken_reg_20260922"),
    r"S:\Arivis_Analysis\_active\YF2026051202": Path(r"S:\Arivis_Analysis\_active\YF2026051202\_quarantine_broken_reg_20260922"),
}
PRODUCTS = [
    "upsampled_atlas_label",
    "upsampled_atlas_label.zarr",
    "atlas_label_hemisphere.zarr",
    "ch0_warped_image",
    "transforms",
    "ch0_L2.zarr",
    "ch0_L2.zarr.done",
    "ch0_downsample",
    "results",
]
KEEP_NOTE = "ch1.zarr / ch1_mask.zarr / config.json / logs kept; rerun redoes steps 2-5 only"

for sample in SAMPLES:
    sample_dir = Path(sample)
    quarantine = QUARANTINE_ROOTS[str(sample_dir.parent)] / sample_dir.name
    moved = []
    for name in PRODUCTS:
        src = sample_dir / name
        if not src.exists():
            continue
        dst = quarantine / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True) if dst.is_dir() else dst.unlink()
        shutil.move(str(src), str(dst))
        moved.append(name)
    (sample_dir / "results").mkdir(exist_ok=True)
    print(sample_dir.name, "->", moved)
for root in QUARANTINE_ROOTS.values():
    (root / "_README.txt").write_text(
        "Quarantined 2026-09-22: registration products built from a 12.5um (should be 25um) "
        "fixed image carrying a physical affine; the warped atlas label was near-empty so all "
        "region/L-R statistics in these files are invalid. " + KEEP_NOTE,
        encoding="utf-8",
    )
print("done")
