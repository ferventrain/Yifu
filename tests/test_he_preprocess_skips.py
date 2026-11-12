"""HE preprocess step-skip flags: identity pass-through + skip_rolling_ball (fast, tiny arrays)."""

import numpy as np

from pipeline_modules.HE.falsecolor_batch import parse_args as batch_parse_args
from pipeline_modules.HE.preprocess import (
    DeconvParams,
    DestripeParams,
    HePreprocessParams,
    RollingBallParams,
    preprocess_nuclei,
)
from pipeline_modules.preprocessing.deconvolution import deconvolve_2d
from pipeline_modules.preprocessing.destripe import destripe_2d


def _rng_image():
    rng = np.random.default_rng(7)
    img = rng.integers(0, 400, size=(64, 64)).astype(np.float64)
    img += 50 * np.exp(-np.linspace(-3, 3, 64)[None, :] ** 2)  # soft gradient
    return img


def test_identity_backends_are_pass_through():
    img = _rng_image()
    out = deconvolve_2d(img, backend="identity", sigma=1.2, iterations=10)
    np.testing.assert_array_equal(out, img)
    out2 = destripe_2d(img, backend="identity", strength=0.85)
    np.testing.assert_array_equal(out2, img)


def test_skip_rolling_ball_passes_destripe_result_through():
    img = _rng_image()
    params = HePreprocessParams()
    params.deconv = DeconvParams(backend="identity")
    params.destripe = DestripeParams(backend="identity")
    params.rolling_ball = RollingBallParams(radius=5)

    on = preprocess_nuclei(img, params=params)
    assert not np.array_equal(on["after_rolling_ball"], on["after_destripe"])

    params.skip_rolling_ball = True
    off = preprocess_nuclei(img, params=params)
    np.testing.assert_array_equal(off["after_rolling_ball"], off["after_destripe"])
    np.testing.assert_array_equal(off["background"], np.zeros_like(img))


def test_batch_cli_rolling_ball_toggle():
    on = batch_parse_args(["--input_dir", "x"])
    assert on.use_rolling_ball is True
    off = batch_parse_args(["--input_dir", "x", "--no-use_rolling_ball"])
    assert off.use_rolling_ball is False
