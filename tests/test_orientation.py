"""Orientation preflight: AP/DV flips are detectable from mask shape, ML is not."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pipeline_modules.registration.orientation import (
    axes_from_flip,
    detect_axis_flips,
    write_orientation_report,
)


def _synthetic_brain(shape: tuple[int, int, int] = (48, 96, 48)) -> np.ndarray:
    """AP/DV-asymmetric, ML-symmetric mask (x=ML, y=AP, z=DV)."""
    xx, yy, zz = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    x = xx - (shape[0] - 1) / 2
    y = yy - (shape[1] - 1) / 2
    z = zz - (shape[2] - 1) / 2
    mask = (x / 16) ** 2 + (y / 34) ** 2 + (z / 14) ** 2 <= 1
    mask |= (x / 8) ** 2 + ((y - 38) / 10) ** 2 + ((z + 2) / 8) ** 2 <= 1
    mask |= (x / 10) ** 2 + ((y + 36) / 8) ** 2 + ((z - 8) / 8) ** 2 <= 1
    mask |= (x / 7) ** 2 + ((y + 5) / 20) ** 2 + ((z - 18) / 5) ** 2 <= 1
    mask |= (x / 5) ** 2 + ((y - 8) / 8) ** 2 + ((z + 16) / 4) ** 2 <= 1
    return mask


@pytest.mark.parametrize(
    ("flip_axes", "expected"),
    [
        ((), [False, False, False]),
        ((1,), [False, True, False]),
        ((2,), [False, False, True]),
        ((1, 2), [False, True, True]),
    ],
)
def test_detects_ap_dv_flips(flip_axes, expected):
    atlas = _synthetic_brain()
    sample = np.flip(atlas, axis=flip_axes) if flip_axes else atlas
    result = detect_axis_flips(sample, atlas)
    assert result["detected_flip"] == expected
    assert result["reliable"] is True
    assert result["best_score"] > result["runner_up_score"]


def test_ml_flip_is_ambiguous_and_never_applied():
    atlas = _synthetic_brain()
    result = detect_axis_flips(np.flip(atlas, axis=0), atlas)
    assert result["detected_flip"] == [False, False, False]
    assert result["ml_ambiguous"] is True
    assert result["ml_score_gap"] <= 1e-3
    assert np.isclose(result["scores"]["0,0,0"], result["scores"]["1,0,0"], atol=1e-3)


def test_min_confidence_marks_result_unreliable():
    atlas = _synthetic_brain()
    result = detect_axis_flips(atlas, atlas, min_confidence=1.0)
    assert result["reliable"] is False


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((8, 8, 8), dtype=bool),
        np.zeros((8, 8), dtype=bool),
    ],
)
def test_invalid_masks_raise(mask):
    with pytest.raises(ValueError):
        detect_axis_flips(mask, _synthetic_brain())


def test_disconnected_specks_are_ignored():
    atlas = _synthetic_brain()
    sample = atlas.copy()
    sample[0, 0, :4] = True
    result = detect_axis_flips(sample, atlas)
    assert result["detected_flip"] == [False, False, False]


def test_axes_from_flip():
    assert axes_from_flip([False, True, False]) == (1,)


def test_write_orientation_report(tmp_path):
    result = detect_axis_flips(_synthetic_brain(), _synthetic_brain())
    out = tmp_path / "qc" / "orientation_check.json"
    write_orientation_report(out, result)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["detected_flip"] == [False, False, False]
    assert len(loaded["scores"]) == 8
