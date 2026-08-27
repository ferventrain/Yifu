"""
Pluggable 2D stripe artifact removal for LSFM / HE preprocessing.

Backends:
  - fft: soft Fourier notch along the stripe frequency axis (default)
  - smooth: estimate stripes by 1D smoothing and subtract residual
  - placeholder: legacy hard central-cross attenuation
  - identity: pass-through
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import tifffile
from scipy import ndimage as ndi

DestripeFn = Callable[..., np.ndarray]

_BACKENDS: dict[str, DestripeFn] = {}


def register_destripe_backend(name: str, fn: DestripeFn) -> None:
    _BACKENDS[name] = fn


def list_destripe_backends() -> list[str]:
    return sorted(_BACKENDS)


def _to_float(image: np.ndarray) -> tuple[np.ndarray, np.dtype]:
    if image.ndim != 2:
        raise ValueError(f"destripe_2d expects a 2D image, got shape={image.shape}")
    return image.astype(np.float64, copy=False), image.dtype


def _restore_dtype(image: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        max_val = np.iinfo(dtype).max
        return np.clip(image, 0, max_val).astype(dtype)
    return image.astype(dtype, copy=False)


def destripe_identity(image: np.ndarray, **_kwargs) -> np.ndarray:
    arr, dtype = _to_float(image)
    return _restore_dtype(arr, dtype)


def destripe_placeholder(
    image: np.ndarray,
    *,
    strength: float = 0.35,
    notch_width: int = 3,
    orientation: str = "horizontal",
    **_kwargs,
) -> np.ndarray:
    """Legacy hard attenuation of the FFT central cross."""
    arr, dtype = _to_float(image)
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return _restore_dtype(arr, dtype)

    spectrum = np.fft.fftshift(np.fft.fft2(arr))
    height, width = spectrum.shape
    cy, cx = height // 2, width // 2
    half = max(1, int(notch_width))
    damp = 1.0 - strength

    mask = np.ones((height, width), dtype=np.float64)
    orient = str(orientation).lower()
    if orient in {"horizontal", "both", "h"}:
        mask[cy - half : cy + half + 1, :] = damp
    if orient in {"vertical", "both", "v"}:
        mask[:, cx - half : cx + half + 1] = damp
    mask[cy - half : cy + half + 1, cx - half : cx + half + 1] = 1.0

    restored = np.fft.ifft2(np.fft.ifftshift(spectrum * mask)).real
    restored = np.maximum(restored, 0.0)
    return _restore_dtype(restored, dtype)


def _soft_notch_mask(
    shape: tuple[int, int],
    *,
    orientation: str,
    notch_width: float,
    keep_fraction: float,
    strength: float,
) -> np.ndarray:
    """
    Build a soft multiplicative FFT mask.

    Horizontal stripes -> energy along fx=0 (vertical line through DC).
    Vertical stripes   -> energy along fy=0 (horizontal line through DC).
    """
    height, width = shape
    cy, cx = height / 2.0, width / 2.0
    yy, xx = np.ogrid[0:height, 0:width]
    fy = yy - cy
    fx = xx - cx

    # Keep a low-frequency disk around DC.
    keep_fraction = float(np.clip(keep_fraction, 0.01, 0.5))
    r_keep = keep_fraction * min(height, width) / 2.0
    r2 = fx * fx + fy * fy
    keep = r2 <= (r_keep * r_keep)

    sigma = max(0.5, float(notch_width))
    strength = float(np.clip(strength, 0.0, 1.0))
    mask = np.ones((height, width), dtype=np.float64)
    orient = str(orientation).lower()

    if orient in {"horizontal", "both", "h"}:
        # Attenuate near fx ~= 0 (vertical line).
        w = np.exp(-(fx * fx) / (2.0 * sigma * sigma))
        mask *= 1.0 - strength * w
    if orient in {"vertical", "both", "v"}:
        w = np.exp(-(fy * fy) / (2.0 * sigma * sigma))
        mask *= 1.0 - strength * w

    mask[keep] = 1.0
    return mask


def destripe_fft(
    image: np.ndarray,
    *,
    strength: float = 0.85,
    notch_width: float = 2.0,
    keep_fraction: float = 0.04,
    orientation: str = "horizontal",
    **_kwargs,
) -> np.ndarray:
    """
    Soft Fourier notch destriping.

    Designed for LSFM stripe artifacts (default: horizontal stripes).
    Preserves a low-frequency core so overall shading is retained.
    """
    arr, dtype = _to_float(image)
    if float(strength) <= 0:
        return _restore_dtype(arr, dtype)

    spectrum = np.fft.fftshift(np.fft.fft2(arr))
    mask = _soft_notch_mask(
        spectrum.shape,
        orientation=orientation,
        notch_width=notch_width,
        keep_fraction=keep_fraction,
        strength=strength,
    )
    restored = np.fft.ifft2(np.fft.ifftshift(spectrum * mask)).real
    restored = np.maximum(restored, 0.0)
    # Match original mean intensity to avoid global dimming.
    m0 = float(arr.mean())
    m1 = float(restored.mean())
    if m1 > 1e-8:
        restored *= m0 / m1
    return _restore_dtype(restored, dtype)


def destripe_smooth(
    image: np.ndarray,
    *,
    strength: float = 1.0,
    sigma: float = 40.0,
    orientation: str = "horizontal",
    **_kwargs,
) -> np.ndarray:
    """
    Estimate stripe background with anisotropic smoothing and subtract.

    Horizontal stripes: smooth strongly along x, lightly along y, then
    subtract (smooth_x - smooth_xy) * strength.
    """
    arr, dtype = _to_float(image)
    strength = float(np.clip(strength, 0.0, 2.0))
    if strength <= 0:
        return _restore_dtype(arr, dtype)

    sigma = float(max(sigma, 1.0))
    orient = str(orientation).lower()
    if orient in {"horizontal", "h"}:
        # Stripes constant along x → smooth along x.
        along = ndi.gaussian_filter1d(arr, sigma=sigma, axis=1)
        both = ndi.gaussian_filter(arr, sigma=(max(1.0, sigma / 8.0), sigma))
    elif orient in {"vertical", "v"}:
        along = ndi.gaussian_filter1d(arr, sigma=sigma, axis=0)
        both = ndi.gaussian_filter(arr, sigma=(sigma, max(1.0, sigma / 8.0)))
    else:
        # both orientations: apply sequential
        tmp = destripe_smooth(
            arr, strength=strength, sigma=sigma, orientation="horizontal"
        )
        return destripe_smooth(
            tmp, strength=strength, sigma=sigma, orientation="vertical"
        )

    stripes = along - both
    restored = np.maximum(arr - strength * stripes, 0.0)
    return _restore_dtype(restored, dtype)


register_destripe_backend("identity", destripe_identity)
register_destripe_backend("placeholder", destripe_placeholder)
register_destripe_backend("fft", destripe_fft)
register_destripe_backend("smooth", destripe_smooth)


def destripe_2d(
    image: np.ndarray,
    *,
    backend: str = "fft",
    **kwargs,
) -> np.ndarray:
    """
    Run registered 2D destripe backend.

    Default backend is soft FFT notch (`fft`).
    """
    name = str(backend).lower()
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown destripe backend {backend!r}. "
            f"Available: {list_destripe_backends()}"
        )
    return _BACKENDS[name](image, **kwargs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D stripe removal (pluggable backends).")
    parser.add_argument("input", type=Path, help="Input TIFF path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output TIFF. Defaults to <stem>_destripe.tif beside input.",
    )
    parser.add_argument(
        "--backend",
        choices=tuple(list_destripe_backends()),
        default="fft",
    )
    parser.add_argument("--strength", type=float, default=0.85)
    parser.add_argument("--notch_width", type=float, default=2.0)
    parser.add_argument("--keep_fraction", type=float, default=0.04)
    parser.add_argument("--sigma", type=float, default=40.0, help="Smooth-backend sigma.")
    parser.add_argument(
        "--orientation",
        choices=("horizontal", "vertical", "both"),
        default="horizontal",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    image = tifffile.imread(str(args.input))
    if image.ndim > 2:
        raise SystemExit(f"Expected 2D TIFF, got shape={image.shape}")
    out = destripe_2d(
        image,
        backend=args.backend,
        strength=args.strength,
        notch_width=args.notch_width,
        keep_fraction=args.keep_fraction,
        sigma=args.sigma,
        orientation=args.orientation,
    )
    output = args.output or args.input.with_name(f"{args.input.stem}_destripe{args.input.suffix}")
    tifffile.imwrite(str(output), out)
    print(f"Wrote {output} (backend={args.backend})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
