from __future__ import annotations

import json
from pathlib import Path

from pipeline_modules.segmentation import (
    CfosUNetInferenceCfg,
    SegmentationCfg,
    ThresholdSegmentationCfg,
    export_json_schema,
    layout_for_sample,
    load_capability_manifest,
)


def test_segmentation_cfg_accepts_cfos_unet_fields():
    cfg = SegmentationCfg.model_validate(
        {
            "method": "cfos_unet",
            "cfos_unet": {
                "checkpoint_path": "models/best_model.pt",
                "save_probability": True,
                "probability_zarr": "sample/ch2_prob.zarr",
                "patch_size": [128, 128, 128],
                "chunk_size": [64, 64, 64],
                "process_existing_only": True,
                "rerun_if_model_updated": True,
            },
        }
    )
    dumped = cfg.model_dump()
    assert dumped["method"] == "cfos_unet"
    assert dumped["cfos_unet"]["checkpoint_path"] == "models/best_model.pt"
    assert dumped["cfos_unet"]["save_probability"] is True
    assert dumped["cfos_unet"]["probability_zarr"] == "sample/ch2_prob.zarr"
    assert dumped["cfos_unet"]["patch_size"] == (128, 128, 128)
    assert dumped["cfos_unet"]["chunk_size"] == (64, 64, 64)
    assert dumped["cfos_unet"]["process_existing_only"] is True
    assert dumped["cfos_unet"]["rerun_if_model_updated"] is True


def test_schema_export_contains_expected_models():
    schema = export_json_schema()
    assert "SegmentationCfg" in schema
    assert "ThresholdSegmentationCfg" in schema
    assert "CfosUNetInferenceCfg" in schema


def test_capability_manifest_contains_cfos_entrypoint():
    manifest = load_capability_manifest()
    entrypoint_ids = {entry["id"] for entry in manifest["entrypoints"]}
    assert "threshold_segmentation" in entrypoint_ids
    assert "cfos_unet_inference" in entrypoint_ids
    assert "cfos_unet_qc" in entrypoint_ids
    assert "export_zarr_to_tiff" in entrypoint_ids


def test_layout_for_sample_resolves_mask_paths(tmp_path: Path):
    layout = layout_for_sample(tmp_path, signal_ch="ch3", reg_ch="ch1", require_exists=True)
    assert Path(layout.signal_zarr) == tmp_path / "ch3.zarr"
    assert Path(layout.mask_zarr) == tmp_path / "ch3_mask.zarr"
    assert Path(layout.mask_tiff_dir) == tmp_path / "ch3_mask"


def test_capabilities_json_marks_segmentation_agent_native():
    capabilities = json.loads(Path("capabilities.json").read_text(encoding="utf-8"))
    segmentation = next(module for module in capabilities["modules"] if module["id"] == "segmentation")
    assert segmentation["agent_native"] is True
    assert segmentation["capability_manifest"] == "pipeline_modules/segmentation/capability_manifest.json"






def test_cfos_qc_manifest_uses_block_review_outputs():
    manifest = load_capability_manifest()
    qc_entry = next(entry for entry in manifest["entrypoints"] if entry["id"] == "cfos_unet_qc")
    assert "blocks" in qc_entry["description"].lower()
    assert "preview_dir" in qc_entry["inputs"]
    assert "preview_dir" in qc_entry["outputs"]


def test_package_exports_config_classes():
    assert SegmentationCfg is not None
    assert ThresholdSegmentationCfg is not None
    assert CfosUNetInferenceCfg is not None
