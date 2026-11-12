"""Channel shift correction + hex color seeding for the HE napari preview (fast, synthetic)."""

import numpy as np
import pytest
from scipy import ndimage
from skimage.draw import disk

from pipeline_modules.HE.napari_he_preview import (
    apply_channel_shift,
    estimate_channel_shift,
    parse_hex_rgb,
)


def _blobs(seed: int = 3, size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.random((size, size)) * 20.0
    for _ in range(40):
        cy, cx = rng.integers(20, size - 20, size=2)
        rr, cc = disk((cy, cx), rng.integers(3, 7))
        keep = (rr >= 0) & (rr < size) & (cc >= 0) & (cc < size)
        img[rr[keep], cc[keep]] += 400.0
    img = ndimage.gaussian_filter(img, 1.0)
    return img


def test_estimate_recovers_known_shift():
    base = _blobs()
    truth = (2.0, -3.0)  # cyto displaced by (dy, dx) relative to nuclei
    cyto = ndimage.shift(base, truth, order=1, mode="nearest")
    shift, ok = estimate_channel_shift(base, cyto)
    assert ok
    # returned shift is the CORRECTION to apply to cyto (= -displacement)
    np.testing.assert_allclose(shift, (-truth[0], -truth[1]), atol=0.35)


def test_guard_refuses_wild_shift():
    base = _blobs()
    cyto = np.roll(base, (60, -60), axis=(0, 1))  # beyond max_shift guard
    _shift, ok = estimate_channel_shift(base, cyto)
    assert not ok


def test_apply_round_trip_restores_base():
    base = _blobs()
    cyto = ndimage.shift(base, (1.5, -2.5), order=1, mode="nearest")
    shift, ok = estimate_channel_shift(base, cyto)
    assert ok
    aligned = apply_channel_shift(cyto, shift)
    err = np.abs(aligned - base).mean()
    assert err < 15.0  # bilinear resample leaves small residual


def test_parse_hex_rgb():
    assert parse_hex_rgb("9557C7") == (149, 87, 199)
    assert parse_hex_rgb("#F3A1F3") == (243, 161, 243)
    assert parse_hex_rgb("") is None
    assert parse_hex_rgb(None) is None
    with pytest.raises(ValueError):
        parse_hex_rgb("XYZ")
