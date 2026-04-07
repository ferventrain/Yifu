import argparse
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi


def _build_ball_structure(radius: int) -> np.ndarray:
    """Build a non-flat spherical-cap structuring element for rolling-ball opening."""
    if radius < 1:
        raise ValueError(f"radius must be >= 1, got {radius}")

    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    distance_sq = xx * xx + yy * yy
    mask = distance_sq <= radius * radius

    structure = np.full((2 * radius + 1, 2 * radius + 1), np.inf, dtype=np.float64)
    heights = np.sqrt((radius * radius - distance_sq[mask]).astype(np.float64))
    structure[mask] = heights.max() - heights
    return structure


def rolling_ball_background(image: np.ndarray, radius: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Estimate background with a rolling-ball opening and subtract it from the image."""
    if image.ndim != 2:
        raise ValueError(f"rolling_ball_background expects a 2D image, got shape={image.shape}")

    original_dtype = image.dtype
    image_float = image.astype(np.float64, copy=False)
    structure = _build_ball_structure(radius)

    eroded = ndi.grey_erosion(image_float, footprint=np.isfinite(structure), structure=structure)
    background = ndi.grey_dilation(eroded, footprint=np.isfinite(structure), structure=structure)
    corrected = np.clip(image_float - background, 0, None)

    if np.issubdtype(original_dtype, np.integer):
        max_val = np.iinfo(original_dtype).max
        corrected = np.clip(corrected, 0, max_val).astype(original_dtype)
        background = np.clip(background, 0, max_val).astype(original_dtype)
    else:
        corrected = corrected.astype(original_dtype, copy=False)
        background = background.astype(original_dtype, copy=False)

    return corrected, background


def rolling_ball_background_correction(image_path: str, output_path: str = "", radius: int = 50) -> Path:
    """Read one TIFF image, apply rolling-ball background subtraction, and save the result."""
    input_path = Path(image_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    image = tifffile.imread(str(input_path))
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if image.ndim != 2:
        raise ValueError(f"rolling_ball_background_correction expects a 2D TIFF, got shape={image.shape}")

    corrected, _ = rolling_ball_background(image, radius=radius)

    if output_path:
        save_path = Path(output_path)
    else:
        save_path = input_path.with_name(f"{input_path.stem}_rolling_ball{input_path.suffix}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(save_path), corrected, compression=None)
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Apply rolling-ball background subtraction to a TIFF image.")
    parser.add_argument("image_path", help="Input TIFF image path")
    parser.add_argument("--output", default="", help="Output TIFF path")
    parser.add_argument("--radius", type=int, default=50, help="Rolling ball radius in pixels")
    args = parser.parse_args()

    output_path = rolling_ball_background_correction(
        image_path=args.image_path,
        output_path=args.output,
        radius=args.radius,
    )
    print(f"Rolling-ball result saved to: {output_path}")


if __name__ == "__main__":
    main()
