import argparse
from pathlib import Path

import cv2
import numpy as np
import tifffile


def apply_median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply a 2D median filter and preserve the input dtype."""
    if kernel_size <= 1:
        return image.copy()
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")

    original_dtype = image.dtype
    filtered = cv2.medianBlur(image, kernel_size)
    return filtered.astype(original_dtype, copy=False)


def median_filter_image(image_path: str, output_path: str = "", kernel_size: int = 3) -> Path:
    """Read one TIFF image, apply median filtering, and save the result."""
    input_path = Path(image_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    image = tifffile.imread(str(input_path))
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if image.ndim != 2:
        raise ValueError(f"median_filter_image expects a 2D TIFF, got shape={image.shape}")

    filtered = apply_median_filter(image, kernel_size=kernel_size)

    if output_path:
        save_path = Path(output_path)
    else:
        save_path = input_path.with_name(f"{input_path.stem}_median{input_path.suffix}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(save_path), filtered, compression=None)
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Apply a median filter to a TIFF image.")
    parser.add_argument("image_path", help="Input TIFF image path")
    parser.add_argument("--output", default="", help="Output TIFF path")
    parser.add_argument("--kernel_size", type=int, default=3, help="Odd median kernel size")
    args = parser.parse_args()

    output_path = median_filter_image(
        image_path=args.image_path,
        output_path=args.output,
        kernel_size=args.kernel_size,
    )
    print(f"Median filter result saved to: {output_path}")


if __name__ == "__main__":
    main()
