"""Tests for pipeline_modules/utils: errors, run_manifest, sample_layout."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from pipeline_modules.utils.errors import ErrorCode, PipelineError
from pipeline_modules.utils.run_manifest import (
    MANIFEST_FILENAME,
    build_run_manifest,
    write_run_manifest,
)
from pipeline_modules.utils.sample_layout import SampleLayout


def _write_synthetic_ims_volume(path: Path, volume_zyx: np.ndarray) -> None:
    import h5py

    with h5py.File(path, "w") as handle:
        tp = handle.create_group("DataSet").create_group("ResolutionLevel 0").create_group("TimePoint 0")
        ch = tp.create_group("Channel 0")
        ch.create_dataset("Data", data=volume_zyx, chunks=True)


def _write_synthetic_multi_channel_ims(path: Path, channel_volumes: dict[int, np.ndarray]) -> None:
    import h5py

    with h5py.File(path, "w") as handle:
        tp = handle.create_group("DataSet").create_group("ResolutionLevel 0").create_group("TimePoint 0")
        for channel, volume_zyx in channel_volumes.items():
            ch = tp.create_group(f"Channel {int(channel)}")
            ch.create_dataset("Data", data=volume_zyx, chunks=True)


# ---------------------------------------------------------------------------
# PipelineError / ErrorCode
# ---------------------------------------------------------------------------


class TestPipelineError:
    def test_basic_attributes(self):
        err = PipelineError(ErrorCode.CONFIG_INVALID, "bad value")
        assert err.code is ErrorCode.CONFIG_INVALID
        assert "bad value" in str(err.message)

    def test_exit_code_config(self):
        err = PipelineError(ErrorCode.CONFIG_INVALID, "x")
        assert err.exit_code == 2

    def test_exit_code_io(self):
        err = PipelineError(ErrorCode.INPUT_NOT_FOUND, "missing file")
        assert err.exit_code == 3

    def test_exit_code_runtime(self):
        err = PipelineError(ErrorCode.SKELETONIZATION_FAILED, "oops")
        assert err.exit_code == 1

    def test_is_exception(self):
        err = PipelineError(ErrorCode.ARGUMENT_INVALID, "msg")
        with pytest.raises(PipelineError):
            raise err

    def test_all_codes_have_exit_code(self):
        for code in ErrorCode:
            err = PipelineError(code, "test")
            assert isinstance(err.exit_code, int)


# ---------------------------------------------------------------------------
# run_manifest
# ---------------------------------------------------------------------------


class TestBuildRunManifest:
    def _build(self, **kwargs):
        defaults = dict(
            module="test_module",
            entrypoint="test_fn",
            inputs={"a": 1},
            outputs=[],
            started_at=time.time(),
        )
        defaults.update(kwargs)
        return build_run_manifest(**defaults)

    def test_returns_dict(self):
        m = self._build()
        assert isinstance(m, dict)

    def test_required_keys(self):
        m = self._build()
        for key in ("schema_version", "module", "entrypoint", "started_at_iso", "duration_seconds"):
            assert key in m, f"missing key: {key}"

    def test_outputs_list(self, tmp_path):
        p = tmp_path / "out.csv"
        p.write_text("a,b\n1,2\n")
        m = self._build(outputs=[p])
        assert len(m["outputs"]) == 1
        assert m["outputs"][0]["exists"] is True

    def test_nonexistent_output_flagged(self, tmp_path):
        m = self._build(outputs=[tmp_path / "ghost.csv"])
        assert m["outputs"][0]["exists"] is False

    def test_warnings_included(self):
        m = self._build(warnings=["watch out"])
        assert "watch out" in m["warnings"]


class TestWriteRunManifest:
    def test_creates_file(self, tmp_path):
        path = write_run_manifest(
            tmp_path,
            module="m",
            entrypoint="f",
            inputs={},
            outputs=[],
            started_at=time.time(),
        )
        assert path == tmp_path / MANIFEST_FILENAME
        assert path.exists()

    def test_valid_json(self, tmp_path):
        write_run_manifest(
            tmp_path, module="m", entrypoint="f",
            inputs={"x": "y"}, outputs=[], started_at=time.time(),
        )
        manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
        assert manifest["entrypoint"] == "f"

    def test_overwrites_previous(self, tmp_path):
        for i in range(2):
            write_run_manifest(
                tmp_path, module="m", entrypoint=f"f{i}",
                inputs={}, outputs=[], started_at=time.time(),
            )
        manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
        assert manifest["entrypoint"] == "f1"


# ---------------------------------------------------------------------------
# SampleLayout
# ---------------------------------------------------------------------------


class TestSampleLayout:
    def test_default_channels(self, tmp_path):
        layout = SampleLayout(sample_dir=tmp_path)
        assert layout.signal_ch == "ch0"
        assert layout.reg_ch == "ch1"

    def test_signal_zarr_path(self, tmp_path):
        layout = SampleLayout(sample_dir=tmp_path, signal_ch="ch0")
        assert layout.signal_zarr == tmp_path / "ch0.zarr"

    def test_mask_zarr_path(self, tmp_path):
        layout = SampleLayout(sample_dir=tmp_path, signal_ch="ch0")
        assert layout.mask_zarr == tmp_path / "ch0_mask.zarr"

    def test_tubule_reconstruction_dir(self, tmp_path):
        layout = SampleLayout(sample_dir=tmp_path)
        assert layout.tubule_reconstruction_dir == tmp_path / "tubule_reconstruction"

    def test_tubule_sub_paths(self, tmp_path):
        layout = SampleLayout(sample_dir=tmp_path)
        assert layout.tubule_branch_csv.name == "vessel_branch_metrics.csv"
        assert layout.tubule_vertex_csv.name == "skeleton_vertices.csv"
        assert layout.tubule_edge_csv.name == "skeleton_edges.csv"
        assert layout.tubule_run_manifest.name == "_run_manifest.json"

    def test_density_xlsx_uses_standard_deliverable_name(self, tmp_path):
        sample_dir = tmp_path / "mouse01"
        sample_dir.mkdir()
        layout = SampleLayout(sample_dir=sample_dir, signal_ch="ch2")
        assert layout.brain_distribution_stats_xlsx == (
            sample_dir / "results" / "mouse01_ch2_brain_distribution_stats.xlsx"
        )
        assert layout.density_results_xlsx == layout.brain_distribution_stats_xlsx
        assert layout.heatmap_2d_dir == sample_dir / "visualization" / "mouse01_ch2_heatmap_2d"
        assert layout.heatmap_3d_png == sample_dir / "visualization" / "mouse01_ch2_heatmap_3d.png"

    def test_atlas_label_hemisphere_zarr(self, tmp_path):
        layout = SampleLayout(sample_dir=tmp_path)
        assert layout.atlas_label_hemisphere_zarr == tmp_path / "atlas_label_hemisphere.zarr"

    def test_require_exists_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SampleLayout(sample_dir=tmp_path / "nonexistent", require_exists=True)

    def test_require_exists_ok(self, tmp_path):
        layout = SampleLayout(sample_dir=tmp_path, require_exists=True)
        assert layout.sample_dir == tmp_path

    def test_as_dict_keys(self, tmp_path):
        layout = SampleLayout(sample_dir=tmp_path)
        d = layout.as_dict()
        assert "signal_zarr" in d
        assert "tubule_reconstruction_dir" in d
        assert all(isinstance(v, str) for v in d.values())

    def test_frozen(self, tmp_path):
        layout = SampleLayout(sample_dir=tmp_path)
        with pytest.raises(Exception):
            layout.signal_ch = "ch9"  # type: ignore[misc]


class TestDataPaths:
    def test_expand_config_path_env_placeholder(self, tmp_path, monkeypatch):
        data_root = tmp_path / "yifu_data"
        data_root.mkdir()
        monkeypatch.setenv("YIFU_DATA_DIR", str(data_root))

        from pipeline_modules.utils.data_paths import expand_config_path

        resolved = expand_config_path("${YIFU_DATA_DIR}/reference/atlas.tiff")
        assert resolved == (data_root / "reference" / "atlas.tiff").resolve()

    def test_expand_config_path_repo_relative(self, tmp_path, monkeypatch):
        monkeypatch.delenv("YIFU_DATA_DIR", raising=False)

        from pipeline_modules.utils.data_paths import expand_config_path

        resolved = expand_config_path("config/config.json", project_root_override=tmp_path)
        assert resolved == (tmp_path / "config" / "config.json").resolve()

    def test_get_yifu_data_dir_required(self, monkeypatch):
        monkeypatch.delenv("YIFU_DATA_DIR", raising=False)

        from pipeline_modules.utils.data_paths import get_yifu_data_dir

        with pytest.raises(RuntimeError, match="YIFU_DATA_DIR"):
            get_yifu_data_dir(required=True)


class TestImsToNrrd:
    def test_find_ims_files_accepts_directory(self, tmp_path):
        from pipeline_modules.utils.ims_to_nrrd import _find_ims_files

        ims_a = tmp_path / "a.ims"
        ims_b = tmp_path / "b.ims"
        txt = tmp_path / "note.txt"
        ims_a.write_bytes(b"")
        ims_b.write_bytes(b"")
        txt.write_text("x")

        found = _find_ims_files(tmp_path)
        assert found == [ims_a, ims_b]

    def test_single_nrrd_preserves_zyx_volume(self, tmp_path):
        import SimpleITK as sitk

        from pipeline_modules.utils.ims_to_nrrd import convert_ims_to_single_nrrd

        volume = np.arange(5 * 6 * 7, dtype=np.uint16).reshape(5, 6, 7)
        ims_path = tmp_path / "sample.ims"
        output_path = tmp_path / "sample.nrrd"
        _write_synthetic_ims_volume(ims_path, volume)

        result = convert_ims_to_single_nrrd(
            ims_path,
            output_path,
            spacing_xyz=(1.5, 2.0, 2.5),
            use_compression=False,
        )

        image = sitk.ReadImage(str(result))
        assert image.GetSize() == (7, 6, 5)
        assert image.GetSpacing() == pytest.approx((1.5, 2.0, 2.5))
        assert np.array_equal(sitk.GetArrayFromImage(image), volume)

    def test_fnt_catalog_writes_cubes_and_max_preview(self, tmp_path):
        import SimpleITK as sitk

        from pipeline_modules.utils.ims_to_nrrd import convert_ims_to_fnt_catalog

        volume = np.arange(5 * 6 * 7, dtype=np.uint16).reshape(5, 6, 7)
        ims_path = tmp_path / "sample.ims"
        output_dir = tmp_path / "fnt"
        _write_synthetic_ims_volume(ims_path, volume)

        catalog = convert_ims_to_fnt_catalog(
            ims_path,
            output_dir,
            cube_size_xyz=(8, 8, 4),
            downsample_factor_xyz=(2, 2, 2),
            use_compression=False,
        )

        assert catalog == output_dir / "catalog"
        catalog_text = catalog.read_text(encoding="utf-8")
        assert "size=7 6 5" in catalog_text
        assert "cubesize=8 8 4" in catalog_text

        first_cube = sitk.GetArrayFromImage(
            sitk.ReadImage(str(output_dir / "ch00" / "z00000000" / "y00000000.x00000000.nrrd"))
        )
        second_cube = sitk.GetArrayFromImage(
            sitk.ReadImage(str(output_dir / "ch00" / "z00000004" / "y00000000.x00000000.nrrd"))
        )
        downsampled = sitk.GetArrayFromImage(sitk.ReadImage(str(output_dir / "ch00ds.nrrd")))

        assert np.array_equal(first_cube, volume[:4, :, :])
        assert np.array_equal(second_cube, volume[4:, :, :])
        assert downsampled.shape == (3, 3, 4)
        assert int(downsampled[0, 0, 0]) == 50
        assert int(downsampled[-1, -1, -1]) == int(volume[-1, -1, -1])

    def test_fnt_catalog_all_channels_writes_every_channel(self, tmp_path):
        import SimpleITK as sitk

        from pipeline_modules.utils.ims_to_nrrd import convert_ims_to_fnt_catalog

        volume0 = np.arange(5 * 6 * 7, dtype=np.uint16).reshape(5, 6, 7)
        volume1 = volume0 + 1000
        ims_path = tmp_path / "multi.ims"
        output_dir = tmp_path / "fnt_all"
        _write_synthetic_multi_channel_ims(ims_path, {0: volume0, 1: volume1})

        catalog = convert_ims_to_fnt_catalog(
            ims_path,
            output_dir,
            channel="all",
            use_compression=False,
        )

        catalog_text = catalog.read_text(encoding="utf-8")
        assert "[CH00]" in catalog_text
        assert "[CH01]" in catalog_text
        assert "cubesize=256 256 256" in catalog_text
        assert "location=ch00ds.nrrd" in catalog_text
        assert "location=ch01ds.nrrd" in catalog_text

        ch0_ds = sitk.GetArrayFromImage(sitk.ReadImage(str(output_dir / "ch00ds.nrrd")))
        ch1_ds = sitk.GetArrayFromImage(sitk.ReadImage(str(output_dir / "ch01ds.nrrd")))
        ch0_cube = sitk.GetArrayFromImage(
            sitk.ReadImage(str(output_dir / "ch00" / "z00000000" / "y00000000.x00000000.nrrd"))
        )
        ch1_cube = sitk.GetArrayFromImage(
            sitk.ReadImage(str(output_dir / "ch01" / "z00000000" / "y00000000.x00000000.nrrd"))
        )

        assert np.array_equal(ch0_ds, volume0)
        assert np.array_equal(ch1_ds, volume1)
        assert np.array_equal(ch0_cube, volume0)
        assert np.array_equal(ch1_cube, volume1)
