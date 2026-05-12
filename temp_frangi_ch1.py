from pathlib import Path

import numpy as np
import tifffile
from skimage.filters import frangi


def main() -> None:
    input_dir = Path("S:/\u53ef\u89c6\u5316\u7d20\u6750/\u8840\u7ba1/ch2")
    output_dir = Path("S:/\u53ef\u89c6\u5316\u7d20\u6750/\u8840\u7ba1/ch2_frangi")
    sigmas = [2.0, 4.0]
    black_ridges = False

    files = sorted(input_dir.glob("*.tif"))
    if not files:
        files = sorted(input_dir.glob("*.tiff"))
    if not files:
        raise FileNotFoundError(f"No TIFF files found in {input_dir}")

    print(f"Reading {len(files)} slices from {input_dir}")
    volume = np.stack([tifffile.imread(str(path)) for path in files], axis=0)

    print(f"Running 3D Frangi with sigmas={sigmas}")
    enhanced = frangi(
        volume,
        sigmas=sigmas,
        black_ridges=black_ridges,
        mode="constant",
        cval=0,
    )

    if np.issubdtype(volume.dtype, np.integer):
        info = np.iinfo(volume.dtype)
        enhanced = np.clip(enhanced * info.max, 0, info.max).astype(volume.dtype)
    else:
        enhanced = enhanced.astype(volume.dtype)

    output_dir.mkdir(parents=True, exist_ok=True)
    for index, src in enumerate(files):
        tifffile.imwrite(
            str(output_dir / src.name),
            enhanced[index],
            compression=None,
        )

    print(f"Done: {output_dir}")


if __name__ == "__main__":
    main()
