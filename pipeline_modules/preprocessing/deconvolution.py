"""
Pluggable 2D deconvolution for LSFM / HE preprocessing.

Backends:
  - rl: Richardson-Lucy with Gaussian PSF (default, skimage)
  - placeholder: mild unsharp mask (legacy)
  - identity: pass-through
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.restoration import richardson_lucy

DeconvFn = Callable[..., np.ndarray]

_BACKENDS: dict[str, DeconvFn] = {}


def register_deconv_backend(name: str, fn: DeconvFn) -> None:
    _BACKENDS[name] = fn


def list_deconv_backends() -> list[str]:
    return sorted(_BACKENDS)


def _to_float(image: np.ndarray) -> tuple[np.ndarray, np.dtype]:
    if image.ndim != 2:
        raise ValueError(f"deconvolve_2d expects a 2D image, got shape={image.shape}")
    return image.astype(np.float64, copy=False), image.dtype


def _restore_dtype(image: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        max_val = np.iinfo(dtype).max
        return np.clip(image, 0, max_val).astype(dtype)
    return image.astype(dtype, copy=False)


def make_gaussian_psf(sigma: float, truncate: float = 3.0) -> np.ndarray:
    """Normalized 2D Gaussian PSF."""
    sigma = float(max(sigma, 1e-3))
    radius = max(1, int(truncate * sigma + 0.5))
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    psf = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    psf = psf.astype(np.float64)
    psf /= psf.sum()
    return psf


def deconvolve_identity(image: np.ndarray, **_kwargs) -> np.ndarray:
    """No-op backend (pass-through)."""
    arr, dtype = _to_float(image)
    return _restore_dtype(arr, dtype)


def deconvolve_placeholder(
    image: np.ndarray,
    *,
    amount: float = 0.35,
    sigma: float = 1.0,
    **_kwargs,
) -> np.ndarray:
    """Legacy mild unsharp mask."""
    arr, dtype = _to_float(image)
    if amount <= 0:
        return _restore_dtype(arr, dtype)
    blurred = ndi.gaussian_filter(arr, sigma=float(sigma))
    sharpened = arr + float(amount) * (arr - blurred)
    return _restore_dtype(np.maximum(sharpened, 0.0), dtype)


def deconvolve_richardson_lucy(
    image: np.ndarray,
    *,
    sigma: float = 1.2,
    iterations: int = 10,
    clip: bool = True,
    filter_epsilon: float = 1e-6,
    **_kwargs,
) -> np.ndarray:
    """
    Richardson-Lucy deconvolution with an isotropic Gaussian PSF.

    Parameters
    ----------
    sigma : float
        Gaussian PSF sigma in pixels (approximate lateral blur).
    iterations : int
        RL iterations (8-15 is typical for preview; higher = sharper but slower/noisier).
    """
    arr, dtype = _to_float(image)
    # skimage RL expects non-negative data; normalize to ~[0,1] for stability then rescale.
    vmin = float(arr.min())
    work = arr - vmin
    vmax = float(work.max())
    if vmax <= 0:
        return _restore_dtype(arr, dtype)
    work = work / vmax

    psf = make_gaussian_psf(sigma=sigma)
    restored = richardson_lucy(
        work,
        psf,
        num_iter=int(max(1, iterations)),
        clip=bool(clip),
        filter_epsilon=float(filter_epsilon),
    )
    restored = np.maximum(restored, 0.0) * vmax + vmin
    return _restore_dtype(restored, dtype)


register_deconv_backend("identity", deconvolve_identity)
register_deconv_backend("placeholder", deconvolve_placeholder)
register_deconv_backend("rl", deconvolve_richardson_lucy)


def deconvolve_2d(
    image: np.ndarray,
    *,
    backend: str = "rl",
    **kwargs,
) -> np.ndarray:
    """
    Run registered 2D deconvolution backend.

    Default backend is Richardson-Lucy (`rl`).
    """
    name = str(backend).lower()
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown deconvolution backend {backend!r}. "
            f"Available: {list_deconv_backends()}"
        )
    return _BACKENDS[name](image, **kwargs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D deconvolution (pluggable backends).")
    parser.add_argument("input", type=Path, help="Input TIFF path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output TIFF. Defaults to <stem>_deconv.tif beside input.",
    )
    parser.add_argument(
        "--backend",
        choices=tuple(list_deconv_backends()),
        default="rl",
    )
    parser.add_argument("--sigma", type=float, default=1.2, help="Gaussian PSF / unsharp sigma.")
    parser.add_argument("--iterations", type=int, default=10, help="RL iterations.")
    parser.add_argument("--amount", type=float, default=0.35, help="Placeholder unsharp amount.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    image = tifffile.imread(str(args.input))
    if image.ndim > 2:
        raise SystemExit(f"Expected 2D TIFF, got shape={image.shape}")
    out = deconvolve_2d(
        image,
        backend=args.backend,
        sigma=args.sigma,
        iterations=args.iterations,
        amount=args.amount,
    )
    output = args.output or args.input.with_name(f"{args.input.stem}_deconv{args.input.suffix}")
    tifffile.imwrite(str(output), out)
    print(f"Wrote {output} (backend={args.backend})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
