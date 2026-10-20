"""Hemisphere conversion: only the transform (warped standard-space midline) method remains."""

from __future__ import annotations

import numpy as np

from pipeline_modules.registration.atlas_label_to_hemisphere import (
    _standard_space_hemi,
    convert_hemisphere_from_transform,
)


def test_standard_space_hemi_midline():
    # axes are (X, Y, Z); the midline is the weighted x centroid of the label
    label = np.zeros((64, 32, 32), dtype=np.uint16)
    label[20:44, :, :] = 7
    hemi, mid = _standard_space_hemi(label)
    assert 20 < mid < 44
    xs = np.broadcast_to(np.arange(label.shape[0])[:, None, None], label.shape)
    assert np.all(xs[(hemi == 1) & (label > 0)] < mid)
    assert np.all(xs[(hemi == 2) & (label > 0)] >= mid)


def test_label_method_is_gone():
    import pipeline_modules.registration.atlas_label_to_hemisphere as m

    assert not hasattr(m, "convert_atlas_label_to_hemisphere")
    assert not hasattr(m, "convert_atlas_label_to_hemisphere_zarr")


def test_cli_requires_sample_dir():
    import subprocess
    import sys

    rc = subprocess.run(
        [sys.executable, "-m", "pipeline_modules.registration.atlas_label_to_hemisphere"],
        capture_output=True,
        text=True,
    )
    assert rc.returncode != 0
    assert "--sample_dir" in (rc.stderr + rc.stdout)


def test_transform_method_missing_inputs_raise(tmp_path):
    """A sample dir without transforms/nii fails with a clear message (no flat fallback)."""
    import pytest

    from pipeline_modules.utils.errors import ErrorCode, PipelineError

    with pytest.raises(FileNotFoundError):
        convert_hemisphere_from_transform(str(tmp_path), str(tmp_path / "hemi.zarr"))
