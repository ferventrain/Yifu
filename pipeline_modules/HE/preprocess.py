"""
HE channel preprocessing chains.

Cyto:   deconv -> destripe
Nuclei: deconv -> destripe -> rolling_ball -> optional CLAHE (LCN)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pipeline_modules.preprocessing.deconvolution import deconvolve_2d
from pipeline_modules.preprocessing.destripe import destripe_2d
from pipeline_modules.preprocessing.masked_clahe import clahe_array
from pipeline_modules.preprocessing.rolling_ball_background import rolling_ball_background


@dataclass
class DeconvParams:
    backend: str = "rl"
    amount: float = 0.35
    sigma: float = 1.2
    iterations: int = 10


@dataclass
class DestripeParams:
    backend: str = "fft"
    strength: float = 0.85
    notch_width: float = 2.0
    keep_fraction: float = 0.04
    sigma: float = 40.0
    orientation: str = "horizontal"


@dataclass
class RollingBallParams:
    radius: int = 5


@dataclass
class LcnParams:
    clip_limit: float = 2.0
    grid_size: int = 16
    use_mask: bool = False


@dataclass
class HePreprocessParams:
    deconv: DeconvParams = field(default_factory=DeconvParams)
    destripe: DestripeParams = field(default_factory=DestripeParams)
    rolling_ball: RollingBallParams = field(default_factory=RollingBallParams)
    lcn: LcnParams = field(default_factory=LcnParams)
    # When True, skip placeholder backends that would alter intensities.
    skip_deconv: bool = False
    skip_destripe: bool = False


def _run_deconv(image: np.ndarray, params: DeconvParams) -> np.ndarray:
    return deconvolve_2d(
        image,
        backend=params.backend,
        amount=params.amount,
        sigma=params.sigma,
        iterations=params.iterations,
    )


def _run_destripe(image: np.ndarray, params: DestripeParams) -> np.ndarray:
    return destripe_2d(
        image,
        backend=params.backend,
        strength=params.strength,
        notch_width=params.notch_width,
        keep_fraction=params.keep_fraction,
        sigma=params.sigma,
        orientation=params.orientation,
    )


def preprocess_cyto(
    cyto: np.ndarray,
    params: HePreprocessParams | None = None,
) -> np.ndarray:
    """Cytoplasm / autofluorescence: deconv -> destripe."""
    cfg = params or HePreprocessParams()
    out = np.asarray(cyto)
    if not cfg.skip_deconv:
        out = _run_deconv(out, cfg.deconv)
    if not cfg.skip_destripe:
        out = _run_destripe(out, cfg.destripe)
    return out


def preprocess_nuclei(
    nuclei: np.ndarray,
    *,
    use_lcn: bool = False,
    params: HePreprocessParams | None = None,
) -> dict[str, Any]:
    """
    Nuclei: deconv -> destripe -> rolling_ball -> optional CLAHE.

    Returns dict with keys:
      nuclei, after_deconv, after_destripe, after_rolling_ball, background, used_lcn
    """
    cfg = params or HePreprocessParams()
    out = np.asarray(nuclei)
    after_deconv = out
    if not cfg.skip_deconv:
        after_deconv = _run_deconv(out, cfg.deconv)

    after_destripe = after_deconv
    if not cfg.skip_destripe:
        after_destripe = _run_destripe(after_deconv, cfg.destripe)

    corrected, background = rolling_ball_background(
        after_destripe,
        radius=int(cfg.rolling_ball.radius),
    )
    final = corrected
    if use_lcn:
        final = clahe_array(
            corrected,
            clip_limit=cfg.lcn.clip_limit,
            grid_size=cfg.lcn.grid_size,
            use_mask=cfg.lcn.use_mask,
        )

    return {
        "nuclei": final,
        "after_deconv": after_deconv,
        "after_destripe": after_destripe,
        "after_rolling_ball": corrected,
        "background": background,
        "used_lcn": bool(use_lcn),
    }
