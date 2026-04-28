#!/usr/bin/env python3
"""
Merge a fine-grained Allen atlas label volume into a coarser region set.

Default behavior uses a 20-region whole-brain preset (wb20) that provides
complete grey-matter coverage with biologically meaningful divisions.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import time
from pathlib import Path

import numpy as np

try:
    from pipeline_modules.utils.errors import ErrorCode, PipelineError
    from pipeline_modules.utils.run_manifest import write_run_manifest
except ImportError:
    PipelineError = None  # type: ignore[assignment,misc]
    ErrorCode = None  # type: ignore[assignment]
    write_run_manifest = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


PRESET_TARGETS = {
    # ------------------------------------------------------------------ #
    # wb20 — 20-region whole-brain preset                                 #
    # Provides complete grey-matter coverage with no unmapped voxels.     #
    # Region assignments are based on the Allen CCFv3 hierarchy.          #
    #                                                                     #
    # Isocortex is split into 5 functional groups:                        #
    #   MO+SS  (somatomotor / somatosensory)                              #
    #   VIS    (visual)                                                    #
    #   AUD    (auditory)                                                  #
    #   Medial (cingulate / prefrontal / orbital / insular / retrosplenial)#
    #   Other  (remaining isocortex: temporal, parietal, ectorhinal …)    #
    # Everything else maps to 15 standard structural divisions.           #
    # ------------------------------------------------------------------ #
    "wb20": [
        # --- Isocortex (5 subdivisions) ---
        # Somatomotor + somatosensory cortex
        {"id": 500,  "name": "Somatomotor areas",          "acronym": "MO"},
        {"id": 453,  "name": "Somatosensory areas",        "acronym": "SS"},
        # Visual cortex
        {"id": 669,  "name": "Visual areas",               "acronym": "VIS"},
        # Auditory cortex
        {"id": 247,  "name": "Auditory areas",             "acronym": "AUD"},
        # Medial / limbic / frontal cortex (ACA, PL, ILA, ORB, AI, RSP, GU, VISC, FRP, PTLp, TEa, PERI, ECT)
        {"id": 315,  "name": "Isocortex (other)",          "acronym": "ISOother"},
        # --- Other cortical regions ---
        # Olfactory areas (OB + piriform + cortical amygdala …)
        {"id": 698,  "name": "Olfactory areas",            "acronym": "OLF"},
        # Hippocampal formation (CA, DG, subiculum, entorhinal …)
        {"id": 1089, "name": "Hippocampal formation",      "acronym": "HPF"},
        # Cortical subplate (claustrum, amygdala …)
        {"id": 703,  "name": "Cortical subplate / Amygdala", "acronym": "CTXsp"},
        # --- Cerebral nuclei ---
        {"id": 477,  "name": "Striatum",                   "acronym": "STR"},
        {"id": 803,  "name": "Pallidum",                   "acronym": "PAL"},
        # --- Interbrain ---
        {"id": 549,  "name": "Thalamus",                   "acronym": "TH"},
        {"id": 1097, "name": "Hypothalamus",               "acronym": "HY"},
        # --- Midbrain (3 functional subdivisions) ---
        {"id": 339,  "name": "Midbrain sensory",           "acronym": "MBsen"},
        {"id": 323,  "name": "Midbrain motor",             "acronym": "MBmot"},
        {"id": 348,  "name": "Midbrain behavioral state",  "acronym": "MBsta"},
        # --- Hindbrain ---
        {"id": 771,  "name": "Pons",                       "acronym": "P"},
        {"id": 354,  "name": "Medulla",                    "acronym": "MY"},
        # --- Cerebellum (2 compartments) ---
        {"id": 528,  "name": "Cerebellar cortex",          "acronym": "CBX"},
        {"id": 519,  "name": "Cerebellar nuclei",          "acronym": "CBN"},
        # --- Catch-all for grey matter root not covered above ---
        # "Basic cell groups and regions" (id=8) is the ancestor of every
        # grey-matter structure, so it will receive any label that does not
        # fall under one of the 19 more specific targets above.
        {"id": 8,    "name": "Other grey matter",          "acronym": "grey"},
    ],
    "wb16": [
        {"id": 695, "name": "Cortical plate", "acronym": "CTXpl"},
        {"id": 315, "name": "Isocortex", "acronym": "Isocortex"},
        {"id": 698, "name": "Olfactory areas", "acronym": "OLF"},
        {"id": 1089, "name": "Hippocampal formation", "acronym": "HPF"},
        {"id": 703, "name": "Cortical subplate", "acronym": "CTXsp"},
        {"id": 477, "name": "Striatum", "acronym": "STR"},
        {"id": 803, "name": "Pallidum", "acronym": "PAL"},
        {"id": 549, "name": "Thalamus", "acronym": "TH"},
        {"id": 1097, "name": "Hypothalamus", "acronym": "HY"},
        {"id": 339, "name": "Midbrain, sensory related", "acronym": "MBsen"},
        {"id": 323, "name": "Midbrain, motor related", "acronym": "MBmot"},
        {"id": 348, "name": "Midbrain, behavioral state related", "acronym": "MBsta"},
        {"id": 771, "name": "Pons", "acronym": "P"},
        {"id": 354, "name": "Medulla", "acronym": "MY"},
        {"id": 528, "name": "Cerebellar cortex", "acronym": "CBX"},
        {"id": 519, "name": "Cerebellar nuclei", "acronym": "CBN"},
    ],
    "wb13": [
        {"id": 695, "name": "Cortical plate", "acronym": "CTXpl"},
        {"id": 703, "name": "Cortical subplate", "acronym": "CTXsp"},
        {"id": 477, "name": "Striatum", "acronym": "STR"},
        {"id": 803, "name": "Pallidum", "acronym": "PAL"},
        {"id": 549, "name": "Thalamus", "acronym": "TH"},
        {"id": 1097, "name": "Hypothalamus", "acronym": "HY"},
        {"id": 339, "name": "Midbrain, sensory related", "acronym": "MBsen"},
        {"id": 323, "name": "Midbrain, motor related", "acronym": "MBmot"},
        {"id": 348, "name": "Midbrain, behavioral state related", "acronym": "MBsta"},
        {"id": 771, "name": "Pons", "acronym": "P"},
        {"id": 354, "name": "Medulla", "acronym": "MY"},
        {"id": 528, "name": "Cerebellar cortex", "acronym": "CBX"},
        {"id": 519, "name": "Cerebellar nuclei", "acronym": "CBN"},
    ],
    "wb7": [
        {"id": 688, "name": "Cerebral cortex", "acronym": "CTX"},
        {"id": 623, "name": "Cerebral nuclei", "acronym": "CNU"},
        {"id": 1129, "name": "Interbrain", "acronym": "IB"},
        {"id": 313, "name": "Midbrain", "acronym": "MB"},
        {"id": 1065, "name": "Hindbrain", "acronym": "HB"},
        {"id": 528, "name": "Cerebellar cortex", "acronym": "CBX"},
        {"id": 519, "name": "Cerebellar nuclei", "acronym": "CBN"},
    ],
    "roi24": [
        {"id": 453, "name": "Somatosensory areas", "acronym": "SS"},
        {"id": 500, "name": "Somatomotor areas", "acronym": "MO"},
        {"id": 669, "name": "Visual areas", "acronym": "VIS"},
        {"id": 254, "name": "Retrosplenial area", "acronym": "RSP"},
        {"id": 507, "name": "Main olfactory bulb", "acronym": "MOB"},
        {"id": 31, "name": "Anterior cingulate area", "acronym": "ACA"},
        {"id": 1007, "name": "Simple lobule", "acronym": "SIM"},
        {"id": 95, "name": "Agranular insular area", "acronym": "AI"},
        {"id": 672, "name": "Caudoputamen", "acronym": "CP"},
        {"id": 1017, "name": "Ansiform lobule", "acronym": "AN"},
        {"id": 909, "name": "Entorhinal area", "acronym": "ENT"},
        {"id": 375, "name": "Ammon's horn", "acronym": "CA"},
        {"id": 247, "name": "Auditory areas", "acronym": "AUD"},
        {"id": 136, "name": "Intermediate reticular nucleus", "acronym": "IRN"},
        {"id": 714, "name": "Orbital area", "acronym": "ORB"},
        {"id": 961, "name": "Piriform area", "acronym": "PIR"},
        {"id": 4, "name": "Inferior colliculus", "acronym": "IC"},
        {"id": 852, "name": "Parvicellular reticular nucleus", "acronym": "PARN"},
        {"id": 159, "name": "Anterior olfactory nucleus", "acronym": "AON"},
        {"id": 1048, "name": "Gigantocellular reticular nucleus", "acronym": "GRN"},
        {"id": 395, "name": "Medullary reticular nucleus", "acronym": "MDRN"},
        {"id": 1057, "name": "Gustatory areas", "acronym": "GU"},
        {"id": 677, "name": "Visceral area", "acronym": "VISC"},
        {"id": 928, "name": "Culmen", "acronym": "CUL"},
    ],
}


def parse_args():
    default_cfg = Path(__file__).with_name("Region_Csv_Rev1_updated.CSV")
    parser = argparse.ArgumentParser(
        description="Merge fine-grained atlas labels into a coarser target-region atlas."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more atlas label volumes (.tif/.tiff/.nii/.nii.gz).",
    )
    parser.add_argument(
        "--cfg",
        default=str(default_cfg),
        help="Path to Allen region CSV. Default uses Region_Csv_Rev1_updated.CSV next to this script.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_TARGETS.keys()),
        default="wb20",
        help="Built-in coarse-region preset. Default is wb20 (20-region whole-brain).",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Optional output folder. Default writes next to each input volume.",
    )
    parser.add_argument(
        "--output_suffix",
        default="",
        help="Suffix appended to each output file stem. Default follows the preset name.",
    )
    parser.add_argument(
        "--target_ids",
        default="",
        help="Optional comma-separated parent IDs to override the preset.",
    )
    parser.add_argument(
        "--preserve_unmapped",
        action="store_true",
        help="Keep labels outside the selected target hierarchy instead of setting them to 0.",
    )
    parser.add_argument(
        "--json_logs",
        action="store_true",
        help="Emit NDJSON log records to stderr instead of plain text",
    )
    return parser.parse_args()


def parse_structure_id_path(path_text: str) -> list[int]:
    path_values = ast.literal_eval(str(path_text))
    return [int(value) for value in path_values]


def parse_acronym_text(acronym_text: str) -> str:
    try:
        acronym_values = ast.literal_eval(str(acronym_text))
        if isinstance(acronym_values, list) and acronym_values:
            return str(acronym_values[-1])
    except Exception:
        pass
    return str(acronym_text)


def split_name_and_acronym(name_text: str) -> tuple[str, str]:
    name_text = str(name_text).strip()
    if "," not in name_text:
        return name_text, ""
    base_name, suffix = name_text.rsplit(",", 1)
    return base_name.strip(), suffix.strip()


def load_region_tree(cfg_path: str | Path) -> dict[int, dict]:
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Region CSV not found: {cfg_path}")

    nodes_by_id: dict[int, dict] = {}
    with cfg_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            structure_id = int(row["id"])
            structure_path = parse_structure_id_path(row["structure_id_path"])
            parent_structure_id = structure_path[-2] if len(structure_path) >= 2 else None
            full_name = str(row["name"]) if row["name"] else str(structure_id)
            base_name, suffix_acronym = split_name_and_acronym(full_name)
            parsed_acronym = parse_acronym_text(row["acronym"])
            nodes_by_id[structure_id] = {
                "id": structure_id,
                "name": full_name,
                "base_name": base_name,
                "acronym": parsed_acronym or suffix_acronym,
                "structure_path": structure_path,
                "parent_structure_id": parent_structure_id,
                "children": [],
            }

    for node in nodes_by_id.values():
        parent_structure_id = node["parent_structure_id"]
        if parent_structure_id in nodes_by_id:
            nodes_by_id[parent_structure_id]["children"].append(node)

    return nodes_by_id


def parse_target_ids(target_ids_text: str, preset_name: str) -> list[int]:
    if not str(target_ids_text).strip():
        return [int(item["id"]) for item in PRESET_TARGETS[preset_name]]

    parts = [part.strip() for part in str(target_ids_text).split(",") if part.strip()]
    if not parts:
        raise ValueError("--target_ids was provided but no valid integers were found.")

    deduplicated = []
    seen_ids = set()
    for part in parts:
        target_id = int(part)
        if target_id in seen_ids:
            continue
        seen_ids.add(target_id)
        deduplicated.append(target_id)
    return deduplicated


def resolve_target_specs(nodes_by_id: dict[int, dict], preset_name: str, target_ids_text: str) -> list[dict]:
    requested_target_ids = parse_target_ids(target_ids_text, preset_name)
    preset_targets_by_id = {
        int(item["id"]): item
        for item in PRESET_TARGETS.get(preset_name, [])
    }

    resolved = []
    for target_id in requested_target_ids:
        if target_id not in nodes_by_id:
            raise KeyError(f"Target region id not found in CSV: {target_id}")
        node = nodes_by_id[target_id]
        preset_item = preset_targets_by_id.get(target_id, {})
        resolved.append(
            {
                "id": target_id,
                "name": preset_item.get("name", node["base_name"]),
                "acronym": preset_item.get("acronym", node["acronym"]),
                "node": node,
            }
        )
    return resolved


def build_nearest_ancestor_mapping(nodes_by_id: dict[int, dict], target_specs: list[dict]) -> tuple[dict[int, int], list[dict]]:
    selected_target_ids = {int(item["id"]) for item in target_specs}
    target_summaries_by_id = {
        int(item["id"]): {
            "target_id": int(item["id"]),
            "target_name": str(item["name"]),
            "target_acronym": str(item["acronym"]),
            "assigned_ids": [],
        }
        for item in target_specs
    }

    merge_mapping: dict[int, int] = {}
    for node in nodes_by_id.values():
        structure_id = int(node["id"])
        if structure_id <= 0:
            continue

        mapped_target_id = None
        for ancestor_id in reversed(node["structure_path"]):
            if ancestor_id in selected_target_ids:
                mapped_target_id = int(ancestor_id)
                break

        if mapped_target_id is None:
            continue

        merge_mapping[structure_id] = mapped_target_id
        target_summaries_by_id[mapped_target_id]["assigned_ids"].append(structure_id)

    target_summaries = []
    for target_spec in target_specs:
        summary = target_summaries_by_id[int(target_spec["id"])]
        summary["assigned_ids"] = sorted(summary["assigned_ids"])
        summary["assigned_label_count"] = int(len(summary["assigned_ids"]))
        target_summaries.append(summary)

    return merge_mapping, target_summaries


def split_volume_suffix(path: Path) -> tuple[str, str]:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes[-2:] == [".nii", ".gz"]:
        return path.name[:-7], ".nii.gz"
    if path.suffix:
        return path.stem, path.suffix
    raise ValueError(f"Unsupported volume path without extension: {path}")


def infer_output_paths(input_path: Path, output_dir: str, output_suffix: str, preset_name: str) -> tuple[Path, Path]:
    stem, extension = split_volume_suffix(input_path)
    target_dir = Path(output_dir) if str(output_dir).strip() else input_path.parent
    final_suffix = output_suffix if str(output_suffix).strip() else f"_merged_{preset_name}"
    output_volume_path = target_dir / f"{stem}{final_suffix}{extension}"
    output_summary_path = target_dir / f"{stem}{final_suffix}_summary.csv"
    return output_volume_path, output_summary_path


def load_volume(path_like: str | Path) -> tuple[np.ndarray, dict]:
    path = Path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"Input volume not found: {path}")

    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes[-2:] == [".nii", ".gz"] or path.suffix.lower() == ".nii":
        try:
            import nibabel as nib
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Reading NIfTI files requires nibabel. Please use the project environment from environment.yml."
            ) from exc

        image = nib.load(str(path))
        data = np.asanyarray(image.dataobj)
        metadata = {"format": "nifti", "image": image}
        return data, metadata

    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import tifffile
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Reading TIFF files requires tifffile. Please use the project environment from environment.yml."
            ) from exc

        data = tifffile.imread(str(path))
        metadata = {"format": "tiff"}
        return data, metadata

    raise ValueError(f"Unsupported input volume format: {path}")


def save_volume(path_like: str | Path, data: np.ndarray, metadata: dict):
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)

    if metadata["format"] == "nifti":
        import nibabel as nib

        source_image = metadata["image"]
        header = source_image.header.copy()
        header.set_data_dtype(np.int32)
        output_image = nib.Nifti1Image(np.asarray(data, dtype=np.int32), source_image.affine, header=header)
        nib.save(output_image, str(path))
        return

    if metadata["format"] == "tiff":
        import tifffile

        tifffile.imwrite(str(path), np.asarray(data, dtype=np.int32))
        return

    raise ValueError(f"Unsupported output metadata format: {metadata['format']}")


def remap_labels(label_volume: np.ndarray, merge_mapping: dict[int, int], preserve_unmapped: bool):
    label_volume = np.asarray(label_volume)
    if label_volume.ndim != 3:
        raise ValueError(f"Expected a 3D label volume, got shape={label_volume.shape}")

    output_volume = label_volume.astype(np.int32, copy=True) if preserve_unmapped else np.zeros(
        label_volume.shape, dtype=np.int32
    )
    target_voxel_counts: dict[int, int] = {}

    unique_labels, unique_counts = np.unique(label_volume, return_counts=True)
    for label_value, voxel_count in zip(unique_labels.tolist(), unique_counts.tolist()):
        label_id = int(label_value)
        if label_id <= 0:
            continue

        mapped_target_id = merge_mapping.get(label_id)
        if mapped_target_id is None:
            continue

        output_volume[label_volume == label_id] = int(mapped_target_id)
        target_voxel_counts[mapped_target_id] = target_voxel_counts.get(mapped_target_id, 0) + int(voxel_count)

    return output_volume, target_voxel_counts


def write_summary_csv(summary_path: Path, target_summaries: list[dict], target_voxel_counts: dict[int, int]):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target_id",
                "target_name",
                "target_acronym",
                "assigned_label_count",
                "voxel_count",
                "assigned_ids",
            ],
        )
        writer.writeheader()
        for target_summary in target_summaries:
            writer.writerow(
                {
                    "target_id": target_summary["target_id"],
                    "target_name": target_summary["target_name"],
                    "target_acronym": target_summary["target_acronym"],
                    "assigned_label_count": target_summary["assigned_label_count"],
                    "voxel_count": int(target_voxel_counts.get(target_summary["target_id"], 0)),
                    "assigned_ids": ";".join(str(item) for item in target_summary["assigned_ids"]),
                }
            )


def process_one_volume(
    input_path: Path,
    output_dir: str,
    output_suffix: str,
    preset_name: str,
    merge_mapping: dict[int, int],
    target_summaries: list[dict],
    preserve_unmapped: bool,
):
    output_volume_path, output_summary_path = infer_output_paths(input_path, output_dir, output_suffix, preset_name)
    label_volume, metadata = load_volume(input_path)
    output_volume, target_voxel_counts = remap_labels(label_volume, merge_mapping, preserve_unmapped)
    save_volume(output_volume_path, output_volume, metadata)
    write_summary_csv(output_summary_path, target_summaries, target_voxel_counts)

    kept_nonzero_labels = int(np.unique(output_volume[output_volume > 0]).size)
    logger.info("[done] %s", input_path)
    logger.info("  output volume : %s", output_volume_path)
    logger.info("  summary csv   : %s", output_summary_path)
    logger.info("  nonzero labels: %d", kept_nonzero_labels)


def main():
    import sys as _sys

    args = parse_args()

    if args.json_logs:
        class _JsonFormatter(logging.Formatter):
            def format(self, record):
                return json.dumps({
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                })
        _handler = logging.StreamHandler(_sys.stderr)
        _handler.setFormatter(_JsonFormatter())
        logging.root.addHandler(_handler)
        logging.root.setLevel(logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        _started_at = time.time()

        nodes_by_id = load_region_tree(args.cfg)
        target_specs = resolve_target_specs(nodes_by_id, args.preset, args.target_ids)
        merge_mapping, target_summaries = build_nearest_ancestor_mapping(nodes_by_id, target_specs)

        logger.info("Preset: %s", args.preset)
        logger.info("Resolved %d target regions.", len(target_specs))
        logger.info("Mapped %d source labels into nearest target ancestors.", len(merge_mapping))
        logger.info("Preserve unmapped labels: %s", bool(args.preserve_unmapped))

        for input_item in args.input:
            process_one_volume(
                input_path=Path(input_item),
                output_dir=args.output_dir,
                output_suffix=args.output_suffix,
                preset_name=args.preset,
                merge_mapping=merge_mapping,
                target_summaries=target_summaries,
                preserve_unmapped=args.preserve_unmapped,
            )

        if write_run_manifest is not None and args.output_dir:
            write_run_manifest(
                Path(args.output_dir),
                module="registration.merge_atlas_regions",
                entrypoint="main",
                inputs={
                    "preset": args.preset,
                    "input": args.input,
                    "preserve_unmapped": args.preserve_unmapped,
                    "target_ids": args.target_ids,
                },
                outputs=[],
                started_at=_started_at,
            )
    except Exception as exc:
        if PipelineError is not None and isinstance(exc, PipelineError):
            print(json.dumps({"error_code": exc.code.value, "message": str(exc.message)}), file=_sys.stderr)
            _sys.exit(exc.exit_code)
        logger.exception("Unhandled error: %s", exc)
        _sys.exit(1)


if __name__ == "__main__":
    main()
