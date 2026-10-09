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
        assert cfg.save_upsampled_label is True
        assert cfg.save_upsampled_label_zarr is True
        assert cfg.save_upsampled_label_hemisphere_zarr is False
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
            "save_upsampled_label": False,
            "save_upsampled_label_zarr": True,
            "upsample_method": "linear",
            "chunk_size": 100,
        }
        cfg = RegistrationCfg(**d)
        assert cfg.mode == "image2atlas"
        assert cfg.save_transforms is True
        assert cfg.save_upsampled_label is False
        assert cfg.save_upsampled_label_zarr is True
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
        assert cfg.use_hemisphere_label is False

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
# Atlas label id encoding
# ---------------------------------------------------------------------------


class TestAtlasLabelCodec:
    def test_tiff_stack_loads_in_ants_xyz_order(self, tmp_path):
        import numpy as np
        import tifffile

        from pipeline_modules.registration.label_codec import load_label_array_preserving_ids

        labels_zyx = np.arange(2 * 3 * 4, dtype=np.uint32).reshape(2, 3, 4)
        path = tmp_path / "atlas_label.tiff"
        tifffile.imwrite(path, labels_zyx)

        loaded = load_label_array_preserving_ids(path)
        assert loaded.shape == (4, 3, 2)
        np.testing.assert_array_equal(loaded, np.transpose(labels_zyx, (2, 1, 0)))

    def test_large_allen_ids_round_trip_without_float32_quantization(self):
        import numpy as np

        from pipeline_modules.registration.label_codec import (
            build_label_id_codec,
            decode_label_codes,
        )

        labels = np.asarray(
            [
                [[0, 589508447], [589508451, 589508455]],
                [[607344830, 607344834], [312782562, 182305697]],
            ],
            dtype=np.uint32,
        )

        encoded, lut = build_label_id_codec(labels)
        assert encoded.max() < 10
        assert int(np.float32(589508447)) == 589508416

        decoded = decode_label_codes(encoded.astype(np.float32), lut)
        np.testing.assert_array_equal(decoded, labels.astype(np.int64))


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
        assert len(manifest["entrypoints"]) == 6
        entry_ids = {e["id"] for e in manifest["entrypoints"]}
        assert "run_full_pipeline" in entry_ids
        assert "analyze_zarr_graph" in entry_ids
        assert "convert_atlas_label_to_hemisphere" in entry_ids
        assert "check_region_coverage" in entry_ids
        assert "merge_atlas_regions" in entry_ids
        assert "run_spinal_segment_analysis" in entry_ids


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
        # wb20 preset ids are absent from the tiny CSV -> validation raises
        with pytest.raises(KeyError):
            resolve_target_specs(nodes, "wb20", "")
        target_specs = resolve_target_specs(nodes, "", "1")
        merge_mapping, summaries = build_nearest_ancestor_mapping(nodes, target_specs)
        # Everything should map to root (id=1)
        assert len(merge_mapping) > 0


# ---------------------------------------------------------------------------
# N4 (SimpleITK masked) + alignment-check QC rendering
# ---------------------------------------------------------------------------


class TestN4AndAlignmentCheck:
    def test_n4_bias_correct_masked_flattens_gradient(self):
        ants = pytest.importorskip("ants")
        pytest.importorskip("SimpleITK")
        import numpy as np

        from pipeline_modules.registration.ANTs_registration import n4_bias_correct_masked

        shape = (24, 24, 24)
        zz, yy, xx = np.mgrid[0:24, 0:24, 0:24]
        blob = 500.0 * np.exp(-((zz - 12) ** 2 + (yy - 12) ** 2 + (xx - 12) ** 2) / 40.0)
        bias = 1.0 + 0.5 * xx / 24.0  # smooth x-ramp: classic bias field
        volume = (blob * bias).astype(np.float32)
        mask = blob > 50.0

        corrected = n4_bias_correct_masked(volume, mask, (1.0, 1.0, 1.0))
        assert corrected.shape == shape
        assert np.isfinite(corrected).all()
        # the corrected volume's x-dependence inside the blob must shrink
        def _x_slope(vol):
            vals = [vol[:, :, i][mask[:, :, i]].mean() for i in range(4, 20)]
            return abs(vals[-1] - vals[0]) / max(vals[0], 1e-6)
        assert _x_slope(corrected) < _x_slope(volume)

    def test_render_alignment_check_png_writes_grid(self, tmp_path: Path):
        ants = pytest.importorskip("ants")
        pytest.importorskip("PIL")
        import numpy as np
        from PIL import Image

        from pipeline_modules.registration.ANTs_registration import render_alignment_check_png

        shape = (16, 20, 24)  # x, y, z
        zz, yy, xx = np.mgrid[0:16, 0:20, 0:24]
        blob = 800 * np.exp(-((zz - 8) ** 2 + (yy - 10) ** 2 + (xx - 12) ** 2) / 30.0)
        fixed = ants.from_numpy(blob.astype(np.float32), spacing=(1.0, 1.0, 1.0))
        warped = ants.from_numpy(
            np.roll(blob, 2, axis=1).astype(np.float32), spacing=(1.0, 1.0, 1.0)
        )

        out_png = tmp_path / "qc" / "check.png"
        render_alignment_check_png(fixed, warped, out_png, "unit-test")
        assert out_png.exists()
        with Image.open(out_png) as img:
            # 3 rows x 4 columns of small planes on a dark sheet
            assert img.width > 4 * 10
            assert img.height > 3 * 10

    def test_rigid_preflight_qc_runs_and_renders(self, tmp_path: Path):
        """End-to-end smoke of the preflight path with tiny ANTs images,
        guarding against ants-version parameter drift (e.g. random_seed)."""
        ants = pytest.importorskip("ants")
        import numpy as np

        from pipeline_modules.registration.ANTs_registration import (
            BidirectionalRegistration,
        )

        registrator = BidirectionalRegistration.__new__(BidirectionalRegistration)
        registrator.sample_dir = tmp_path
        shape = (24, 24, 24)
        zz, yy, xx = np.mgrid[0:24, 0:24, 0:24]
        blob = 900 * np.exp(-((zz - 12) ** 2 + (yy - 12) ** 2 + (xx - 12) ** 2) / 30.0)
        fixed = ants.from_numpy(blob.astype(np.float32), spacing=(1.0, 1.0, 1.0))
        moved = ants.from_numpy(
            np.roll(blob, 2, axis=1).astype(np.float32), spacing=(1.0, 1.0, 1.0)
        )
        registrator._run_rigid_preflight_qc(fixed, moved)
        assert (tmp_path / "qc" / "registration_rigid_check.png").exists()
