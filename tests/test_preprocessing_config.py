from __future__ import annotations

import json
from pathlib import Path

from pipeline_modules.preprocessing import (
    PreprocessingCfg,
    export_json_schema,
    layout_for_sample,
    load_capability_manifest,
)


def test_preprocessing_cfg_accepts_repo_template_fields():
    cfg = PreprocessingCfg.model_validate(
        {
            "channels": [1, "ch2"],
            "channel_subtraction": {
                "apply": True,
                "background_channel": "0",
                "estimated_weights": {"ch1": 0.5},
            },
            "downsample": {
                "target_resolution_xyz": [25, 25, 25],
                "chunk_size": 64,
            },
            "zarr": {
                "chunk_size": [32, 64, 64],
                "compressor": "default",
            },
        }
    )

    dumped = cfg.model_dump()
    assert dumped["channels"] == ("ch1", "ch2")
    assert dumped["channel_subtraction"]["background_channel"] == "ch0"
    assert dumped["downsample"]["chunk_size"] == 64
    assert dumped["zarr"]["chunk_size"] == (32, 64, 64)


def test_export_json_schema_contains_expected_models():
    schema = export_json_schema()
    assert "PreprocessingCfg" in schema
    assert "DownsampleCfg" in schema
    assert "ZarrCfg" in schema


def test_load_capability_manifest_has_preprocessing_entrypoint():
    manifest = load_capability_manifest()
    entrypoint_ids = {entry["id"] for entry in manifest["entrypoints"]}
    assert "run_preprocessing" in entrypoint_ids
    assert "convert_tiff_to_zarr" in entrypoint_ids
    assert "downsample_folder" in entrypoint_ids


def test_layout_for_sample_resolves_expected_paths(tmp_path: Path):
    layout = layout_for_sample(tmp_path, signal_ch="ch2", reg_ch="ch1", require_exists=True)
    assert Path(layout.signal_tiff_preprocessed_dir) == tmp_path / "ch2_preprocessed"
    assert Path(layout.signal_zarr) == tmp_path / "ch2.zarr"
    assert Path(layout.reg_downsample_nii) == tmp_path / "ch1_downsample" / "volume.nii.gz"


def test_capabilities_json_marks_preprocessing_agent_native():
    capabilities = json.loads(Path("capabilities.json").read_text(encoding="utf-8"))
    preprocessing = next(module for module in capabilities["modules"] if module["id"] == "preprocessing")
    assert preprocessing["agent_native"] is True
    assert preprocessing["capability_manifest"] == "pipeline_modules/preprocessing/capability_manifest.json"
