"""Tests for pipeline_modules/tubule_reconstruction/config.py."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from pipeline_modules.tubule_reconstruction.config import (
    RegionVesselAnalysisCfg,
    TeasarParams,
    TubuleReconstructionCfg,
    export_json_schema,
    layout_for_sample,
    load_capability_manifest,
    output_dir_for_sample,
)


# ---------------------------------------------------------------------------
# TeasarParams
# ---------------------------------------------------------------------------


class TestTeasarParams:
    def test_defaults(self):
        p = TeasarParams()
        assert p.scale == 1.5
        assert p.const == 300
        assert p.max_paths is None

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            TeasarParams(unknown_param=99)

    def test_frozen(self):
        p = TeasarParams()
        with pytest.raises(Exception):
            p.scale = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TubuleReconstructionCfg
# ---------------------------------------------------------------------------


class TestTubuleReconstructionCfg:
    def test_defaults(self):
        cfg = TubuleReconstructionCfg()
        assert cfg.enabled is False
        assert cfg.resolution_xyz == (1.0, 1.0, 1.0)
        assert cfg.chunkwise is False
        assert cfg.output_dirname == "tubule_reconstruction"

    def test_resolution_xyz_from_string(self):
        cfg = TubuleReconstructionCfg(resolution_xyz="0.5,0.5,1.0")
        assert cfg.resolution_xyz == (0.5, 0.5, 1.0)

    def test_resolution_xyz_from_list(self):
        cfg = TubuleReconstructionCfg(resolution_xyz=[2.0, 2.0, 3.0])
        assert cfg.resolution_xyz == (2.0, 2.0, 3.0)

    def test_halo_zyx_from_string(self):
        cfg = TubuleReconstructionCfg(halo_zyx="4,8,8")
        assert cfg.halo_zyx == (4, 8, 8)

    def test_invalid_resolution_too_few_values(self):
        with pytest.raises(ValidationError):
            TubuleReconstructionCfg(resolution_xyz="1,1")

    def test_parallel_must_be_positive(self):
        with pytest.raises(ValidationError):
            TubuleReconstructionCfg(parallel=0)

    def test_nested_teasar_params(self):
        cfg = TubuleReconstructionCfg(teasar_params={"scale": 2.0, "const": 500})
        assert cfg.teasar_params.scale == 2.0
        assert cfg.teasar_params.const == 500

    def test_json_schema_is_valid_json(self):
        schema = TubuleReconstructionCfg.model_json_schema()
        assert "properties" in schema
        json.dumps(schema)  # must not raise


# ---------------------------------------------------------------------------
# RegionVesselAnalysisCfg
# ---------------------------------------------------------------------------


class TestRegionVesselAnalysisCfg:
    def test_defaults(self):
        cfg = RegionVesselAnalysisCfg()
        assert cfg.annotation_resolution_xyz == (1.0, 1.0, 1.0)
        assert cfg.regions == ()

    def test_regions_from_comma_string(self):
        cfg = RegionVesselAnalysisCfg(regions="CA1, CA2, DG")
        assert cfg.regions == ("CA1", "CA2", "DG")

    def test_regions_from_semicolons(self):
        cfg = RegionVesselAnalysisCfg(regions="CA1;CA2;DG")
        assert cfg.regions == ("CA1", "CA2", "DG")

    def test_regions_from_list(self):
        cfg = RegionVesselAnalysisCfg(regions=["CA1", "CA2"])
        assert cfg.regions == ("CA1", "CA2")

    def test_annotation_resolution_from_string(self):
        cfg = RegionVesselAnalysisCfg(annotation_resolution_xyz="10,10,10")
        assert cfg.annotation_resolution_xyz == (10.0, 10.0, 10.0)


# ---------------------------------------------------------------------------
# export_json_schema
# ---------------------------------------------------------------------------


class TestExportJsonSchema:
    def test_returns_dict_with_both_models(self):
        schema = export_json_schema()
        assert "TubuleReconstructionCfg" in schema
        assert "RegionVesselAnalysisCfg" in schema

    def test_serialisable(self):
        schema = export_json_schema()
        json.dumps(schema)


# ---------------------------------------------------------------------------
# layout_for_sample / output_dir_for_sample
# ---------------------------------------------------------------------------


class TestLayoutForSample:
    def test_returns_sample_layout(self, tmp_path):
        from pipeline_modules.utils.sample_layout import SampleLayout
        layout = layout_for_sample(tmp_path)
        assert isinstance(layout, SampleLayout)

    def test_default_channels(self, tmp_path):
        layout = layout_for_sample(tmp_path)
        assert layout.signal_ch == "ch0"
        assert layout.reg_ch == "ch1"

    def test_custom_channels(self, tmp_path):
        layout = layout_for_sample(tmp_path, signal_ch="ch2", reg_ch="ch3")
        assert layout.signal_ch == "ch2"

    def test_tubule_dir_is_subdir(self, tmp_path):
        layout = layout_for_sample(tmp_path)
        assert layout.tubule_reconstruction_dir.parent == tmp_path

    def test_output_dir_for_sample(self, tmp_path):
        d = output_dir_for_sample(tmp_path, signal_ch="ch0")
        assert d == tmp_path / "tubule_reconstruction"


# ---------------------------------------------------------------------------
# load_capability_manifest
# ---------------------------------------------------------------------------


class TestLoadCapabilityManifest:
    def test_loads_without_error(self):
        manifest = load_capability_manifest()
        assert isinstance(manifest, dict)

    def test_has_entrypoints(self):
        manifest = load_capability_manifest()
        assert "entrypoints" in manifest
        assert len(manifest["entrypoints"]) >= 1

    def test_entrypoint_ids(self):
        manifest = load_capability_manifest()
        ids = {e["id"] for e in manifest["entrypoints"]}
        assert "analyze_binary_mask_zarr" in ids
        assert "analyze_regions_from_skeleton" in ids

    def test_schema_version_present(self):
        manifest = load_capability_manifest()
        assert "schema_version" in manifest
