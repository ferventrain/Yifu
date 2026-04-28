"""Tests for pipeline_modules/utils: errors, run_manifest, sample_layout."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from pipeline_modules.utils.errors import ErrorCode, PipelineError
from pipeline_modules.utils.run_manifest import (
    MANIFEST_FILENAME,
    build_run_manifest,
    write_run_manifest,
)
from pipeline_modules.utils.sample_layout import SampleLayout


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

    def test_density_xlsx_uses_signal_ch(self, tmp_path):
        layout = SampleLayout(sample_dir=tmp_path, signal_ch="ch2")
        assert layout.density_results_xlsx.name == "density_results_ch2.xlsx"

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
