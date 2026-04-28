"""Tests for the registration module's agent-native layer.

Covers:
- Pydantic config models (RegistrationCfg, AnalysisCfg)
- layout_for_sample helper
- export_json_schema / load_capability_manifest
- Smoke tests for check_region_coverage helpers (load_region_tree, resolve_target_node)
- Smoke test for merge_atlas_regions helpers (build_nearest_ancestor_mapping)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Config models
# ---------------------------------------------------------------------------


class TestRegistrationCfg:
    def test_defaults(self):
        from pipeline_modules.registration.config import RegistrationCfg

        cfg = RegistrationCfg()
        assert cfg.method == "ants"
        assert cfg.mode == "atlas2image"
        assert cfg.transform_type == "SyN"
        assert cfg.allow_reflection is False
        assert cfg.save_registered_image is False
        assert cfg.upsample_method == "nearest"
        assert cfg.chunk_size == 50

    def test_from_dict(self):
        from pipeline_modules.registration.config import RegistrationCfg

        d = {
            "method": "ants",
            "mode": "image2atlas",
            "atlas_path": "/data/atlas.tiff",
            "annotation_path": "/data/label.tiff",
            "transform_type": "Affine",
            "allow_reflection": False,
            "save_registered_image": True,
            "save_transforms": True,
            "upsample_method": "linear",
            "chunk_size": 100,
        }
        cfg = RegistrationCfg(**d)
        assert cfg.mode == "image2atlas"
        assert cfg.save_transforms is True
        assert cfg.chunk_size == 100

    def test_invalid_mode(self):
        from pipeline_modules.registration.config import RegistrationCfg

        with pytest.raises(ValueError, match="mode must be"):
            RegistrationCfg(mode="invalid")

    def test_invalid_transform(self):
        from pipeline_modules.registration.config import RegistrationCfg

        with pytest.raises(ValueError, match="transform_type must be"):
            RegistrationCfg(transform_type="BSpline")

    def test_frozen(self):
        from pipeline_modules.registration.config import RegistrationCfg

        cfg = RegistrationCfg()
        with pytest.raises(Exception):
            cfg.method = "something_else"  # type: ignore[misc]


class TestAnalysisCfg:
    def test_defaults(self):
        from pipeline_modules.registration.config import AnalysisCfg

        cfg = AnalysisCfg()
        assert cfg.foreground_mode == "equal"
        assert cfg.foreground_label == 1
        assert cfg.min_voxels == 10
        assert cfg.pass1_workers == 1
        assert cfg.block_size is None
        assert cfg.resolution_xyz == (1.0, 1.0, 1.0)

    def test_resolution_from_string(self):
        from pipeline_modules.registration.config import AnalysisCfg

        cfg = AnalysisCfg(resolution_xyz="1.8,1.8,2.0")
        assert cfg.resolution_xyz == (1.8, 1.8, 2.0)

    def test_block_size_from_string(self):
        from pipeline_modules.registration.config import AnalysisCfg

        cfg = AnalysisCfg(block_size="64,128,128")
        assert cfg.block_size == (64, 128, 128)

    def test_block_size_none_string(self):
        from pipeline_modules.registration.config import AnalysisCfg

        cfg = AnalysisCfg(block_size="")
        assert cfg.block_size is None

    def test_invalid_foreground_mode(self):
        from pipeline_modules.registration.config import AnalysisCfg

        with pytest.raises(ValueError, match="foreground_mode must be"):
            AnalysisCfg(foreground_mode="threshold")


# ---------------------------------------------------------------------------
# layout_for_sample
# ---------------------------------------------------------------------------


class TestLayoutForSample:
    def test_returns_layout(self, tmp_path):
        from pipeline_modules.registration.config import layout_for_sample

        sample = tmp_path / "mouse01"
        sample.mkdir()
        layout = layout_for_sample(str(sample), signal_ch="ch0", reg_ch="ch1")
        assert layout.sample_dir == sample
        assert layout.signal_ch == "ch0"
        assert layout.reg_ch == "ch1"


# ---------------------------------------------------------------------------
# export_json_schema / load_capability_manifest
# ---------------------------------------------------------------------------


class TestExportAndManifest:
    def test_export_json_schema(self):
        from pipeline_modules.registration.config import export_json_schema

        schema = export_json_schema()
        assert "RegistrationCfg" in schema
        assert "AnalysisCfg" in schema
        # Spot-check a property name
        assert "mode" in schema["RegistrationCfg"]["properties"]

    def test_load_capability_manifest(self):
        from pipeline_modules.registration.config import load_capability_manifest

        manifest = load_capability_manifest()
        assert manifest["module"] == "registration"
        assert len(manifest["entrypoints"]) == 4
        entry_ids = {e["id"] for e in manifest["entrypoints"]}
        assert "run_full_pipeline" in entry_ids
        assert "analyze_zarr_graph" in entry_ids
        assert "check_region_coverage" in entry_ids
        assert "merge_atlas_regions" in entry_ids


# ---------------------------------------------------------------------------
# Smoke tests for check_region_coverage_zarr helpers
# ---------------------------------------------------------------------------


class TestCheckRegionCoverageHelpers:
    def test_load_region_tree(self, tiny_region_csv):
        from pipeline_modules.registration.check_region_coverage_zarr import load_region_tree

        nodes_by_id, acronym_to_ids = load_region_tree(str(tiny_region_csv))
        assert 1 in nodes_by_id
        assert 10 in nodes_by_id
        assert 20 in nodes_by_id
        assert nodes_by_id[10]["name"] == "RegionA"
        assert "ra" in acronym_to_ids

    def test_resolve_target_node_by_id(self, tiny_region_csv):
        from pipeline_modules.registration.check_region_coverage_zarr import (
            load_region_tree,
            resolve_target_node,
        )

        nodes_by_id, acronym_to_ids = load_region_tree(str(tiny_region_csv))
        node = resolve_target_node(nodes_by_id, acronym_to_ids, region_id=10)
        assert node["id"] == 10
        assert node["acronym"] == "RA"

    def test_resolve_target_node_by_acronym(self, tiny_region_csv):
        from pipeline_modules.registration.check_region_coverage_zarr import (
            load_region_tree,
            resolve_target_node,
        )

        nodes_by_id, acronym_to_ids = load_region_tree(str(tiny_region_csv))
        node = resolve_target_node(nodes_by_id, acronym_to_ids, acronym="RB")
        assert node["id"] == 20

    def test_resolve_missing_raises(self, tiny_region_csv):
        from pipeline_modules.registration.check_region_coverage_zarr import (
            load_region_tree,
            resolve_target_node,
        )

        nodes_by_id, acronym_to_ids = load_region_tree(str(tiny_region_csv))
        with pytest.raises(KeyError):
            resolve_target_node(nodes_by_id, acronym_to_ids, region_id=999)


# ---------------------------------------------------------------------------
# Smoke tests for merge_atlas_regions helpers
# ---------------------------------------------------------------------------


class TestMergeAtlasRegionsHelpers:
    def test_load_region_tree(self, tiny_region_csv):
        from pipeline_modules.registration.merge_atlas_regions import load_region_tree

        nodes = load_region_tree(str(tiny_region_csv))
        assert 10 in nodes
        assert 20 in nodes

    def test_build_nearest_ancestor_mapping(self, tiny_region_csv):
        from pipeline_modules.registration.merge_atlas_regions import (
            build_nearest_ancestor_mapping,
            load_region_tree,
            resolve_target_specs,
        )

        nodes = load_region_tree(str(tiny_region_csv))
        target_specs = resolve_target_specs(nodes, "wb20", "")
        # wb20 may not match our tiny CSV, so test with explicit target_ids
        target_specs = resolve_target_specs(nodes, "", "1")
        merge_mapping, summaries = build_nearest_ancestor_mapping(nodes, target_specs)
        # Everything should map to root (id=1)
        assert len(merge_mapping) > 0
