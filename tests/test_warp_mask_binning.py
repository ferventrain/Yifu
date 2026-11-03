from __future__ import annotations

import numpy as np


def test_bin_mask_block_array_and_merge() -> None:
    from pipeline_modules.visualization.warp_mask_zarr_to_atlas_points import (
        _bin_mask_block_array,
        _merge_bin_partials,
        _volume_from_bin_counts,
    )

    block_a = np.zeros((2, 2, 2), dtype=np.uint8)
    block_a[0, 0, 0] = 1
    block_b = np.zeros((2, 2, 2), dtype=np.uint8)
    block_b[1, 1, 1] = 1

    partial_a = _bin_mask_block_array(
        block_a,
        z_start=0,
        y_start=0,
        x_start=0,
        resolution_xyz=(2.0, 2.0, 2.0),
        target_resolution_xyz=(4.0, 4.0, 4.0),
        grid_shape_zyx=(2, 2, 2),
        foreground_mode="nonzero",
        foreground_label=1,
    )
    partial_b = _bin_mask_block_array(
        block_b,
        z_start=2,
        y_start=2,
        x_start=2,
        resolution_xyz=(2.0, 2.0, 2.0),
        target_resolution_xyz=(4.0, 4.0, 4.0),
        grid_shape_zyx=(2, 2, 2),
        foreground_mode="nonzero",
        foreground_label=1,
    )

    foreground, clipped, merged = _merge_bin_partials([partial_a, partial_b])
    volume, occupied = _volume_from_bin_counts(
        merged,
        output_shape_zyx=(2, 2, 2),
        min_voxels_per_point=1,
        volume_mode="binary",
    )

    assert foreground == 2
    assert clipped == 0
    assert occupied == 2
    assert int(volume.sum()) == 2


def test_resolve_bin_workers_auto_and_cap() -> None:
    from pipeline_modules.visualization.warp_mask_zarr_to_atlas_points import resolve_bin_workers

    assert resolve_bin_workers(1) == 1
    assert resolve_bin_workers(4) == 4
    assert resolve_bin_workers(0) >= 1


def test_parse_and_axis_mapping_for_flip_atlas() -> None:
    from pipeline_modules.visualization.warp_mask_zarr_to_atlas_points import (
        atlas_flip_axes_zyx,
        parse_flip_atlas,
    )

    assert parse_flip_atlas("false,true,false") == [False, True, False]
    assert parse_flip_atlas("0,1,1") == [False, True, True]
    assert parse_flip_atlas([True, False, False]) == [True, False, False]
    # ML -> zyx axis 2, AP -> 1, DV -> 0.
    assert atlas_flip_axes_zyx([True, False, False]) == (2,)
    assert atlas_flip_axes_zyx([False, True, False]) == (1,)
    assert atlas_flip_axes_zyx([False, False, True]) == (0,)
    assert atlas_flip_axes_zyx([True, True, True]) == (0, 1, 2)
    assert atlas_flip_axes_zyx([False, False, False]) == ()

    try:
        parse_flip_atlas("true,false")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("two-element flip_atlas must be rejected")


def test_apply_atlas_flip_moves_marker_across_ap() -> None:
    import numpy as np

    from pipeline_modules.visualization.warp_mask_zarr_to_atlas_points import apply_atlas_flip

    volume = np.zeros((3, 5, 4), dtype=np.float32)
    volume[0, 0, 0] = 1.0
    flipped = apply_atlas_flip(volume, [False, True, False])
    assert flipped[0, 4, 0] == 1.0
    assert flipped.sum() == volume.sum()
    assert np.array_equal(apply_atlas_flip(volume, [False, False, False]), volume)


def test_resolve_flip_atlas_prefers_orientation_report(tmp_path) -> None:
    import json

    from pipeline_modules.visualization.warp_mask_zarr_to_atlas_points import resolve_flip_atlas

    sample = tmp_path / "sample"
    (sample / "qc").mkdir(parents=True)
    (sample / "qc" / "orientation_check.json").write_text(
        json.dumps({"source": "config", "applied_flip": [False, True, False]}), encoding="utf-8"
    )
    (sample / "config.json").write_text(
        json.dumps({"registration": {"flip_atlas": [True, False, False]}}), encoding="utf-8"
    )
    assert resolve_flip_atlas(sample) == [False, True, False]
    assert resolve_flip_atlas(sample, "0,0,0") == [False, False, False]
    (sample / "qc" / "orientation_check.json").unlink()
    assert resolve_flip_atlas(sample) == [True, False, False]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert resolve_flip_atlas(empty) == [False, False, False]
    assert resolve_flip_atlas(None) == [False, False, False]


def test_warp_sample_grid_to_atlas_mirrors_back_flip(monkeypatch) -> None:
    import sys
    import types

    import numpy as np

    from pipeline_modules.visualization.warp_mask_zarr_to_atlas_points import warp_sample_grid_to_atlas

    class _StubImage:
        def __init__(self, array):
            self._array = np.asarray(array, dtype=np.float32)
            self.origin = np.zeros(3)
            self.spacing = np.array([25.0, 25.0, 25.0])
            self.direction = np.eye(3)
            self.shape = self._array.shape

        def set_direction(self, direction):
            self.direction = np.asarray(direction)

        def numpy(self):
            return self._array

    def _image_read(_path):
        return _StubImage(np.zeros((4, 5, 6)))

    stub = types.SimpleNamespace(
        image_read=_image_read,
        from_numpy=lambda array, **kwargs: _StubImage(array),
        # Identity composition: the stub "registration" does not move anything.
        apply_transforms=lambda fixed, moving, **kwargs: _StubImage(moving.numpy()),
    )
    monkeypatch.setitem(sys.modules, "ants", stub)

    volume = np.zeros((3, 4, 5), dtype=np.float32)
    volume[0, 0, 0] = 7.0
    plain, spacing = warp_sample_grid_to_atlas(
        volume,
        sample_reference_nii="sample.nii.gz",
        atlas_image="atlas.tiff",
        transformlist=["inv.mat"],
        interpolator="linear",
        binarize=False,
    )
    assert spacing == (25.0, 25.0, 25.0)
    assert np.array_equal(plain, volume)
    mirrored, _ = warp_sample_grid_to_atlas(
        volume,
        sample_reference_nii="sample.nii.gz",
        atlas_image="atlas.tiff",
        transformlist=["inv.mat"],
        interpolator="linear",
        binarize=False,
        flip_atlas=[False, True, False],
    )
    assert mirrored[0, 3, 0] == 7.0
    assert mirrored[0, 0, 0] == 0.0


def test_resolve_resolution_xyz_precedence(tmp_path) -> None:
    import json

    from pipeline_modules.visualization.warp_mask_zarr_to_atlas_points import resolve_resolution_xyz

    sample = tmp_path / "sample"
    sample.mkdir()
    (sample / "config.json").write_text(
        json.dumps({"input": {"resolution_xyz": [2.0, 2.0, 3.0]}}), encoding="utf-8"
    )

    # (a) Config value is used when the flag is absent.
    triplet, source = resolve_resolution_xyz(sample_dir=sample)
    assert triplet == (2.0, 2.0, 3.0)
    assert source.startswith("config:")

    # Explicit flag wins over config.
    triplet, source = resolve_resolution_xyz(sample_dir=sample, explicit="1.5,1.5,2.5")
    assert triplet == (1.5, 1.5, 2.5)
    assert source == "flag"


def test_resolve_resolution_xyz_layers_run_config_over_base(tmp_path) -> None:
    import json

    from pipeline_modules.visualization.warp_mask_zarr_to_atlas_points import resolve_resolution_xyz

    sample = tmp_path / "sample"
    sample.mkdir()
    (sample / "config.json").write_text(
        json.dumps({"input": {"resolution_xyz": [0.5, 0.5, 1.0]}}), encoding="utf-8"
    )
    (sample / "config_1.json").write_text(json.dumps({"project_name": "rerun"}), encoding="utf-8")

    # A run config without input.resolution_xyz inherits it from the base config.
    triplet, source = resolve_resolution_xyz(sample_dir=sample, config_path=sample / "config_1.json")
    assert triplet == (0.5, 0.5, 1.0)
    assert "config_1.json" in source


def test_resolve_resolution_xyz_missing_raises_structured_error(tmp_path) -> None:
    import pytest

    from pipeline_modules.utils.errors import ErrorCode, PipelineError
    from pipeline_modules.visualization.warp_mask_zarr_to_atlas_points import resolve_resolution_xyz

    # (b) Missing config + missing flag -> structured error, never a silent
    # 1.8/1.8/2.0 fallback.
    sample = tmp_path / "empty"
    sample.mkdir()
    with pytest.raises(PipelineError) as excinfo:
        resolve_resolution_xyz(sample_dir=sample)
    assert excinfo.value.code == ErrorCode.CONFIG_INVALID
    assert "resolution" in excinfo.value.message.lower()
    assert "suggestion" in excinfo.value.context

    with pytest.raises(PipelineError):
        resolve_resolution_xyz(sample_dir=None)
