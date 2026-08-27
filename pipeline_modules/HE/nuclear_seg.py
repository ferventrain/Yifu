"""Nuclear segmentation preview for HE QC (not used inside Beer-Lambert)."""

from __future__ import annotations

import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects


def segment_nuclei_preview(
    nuclei: np.ndarray,
    *,
    method: str = "otsu",
    threshold: float | None = None,
    min_size: int = 16,
) -> tuple[np.ndarray, float]:
    """
    Return (binary_mask uint8 {0,1}, threshold_used).

    method:
      - 'otsu': Otsu on finite pixels
      - 'fixed': use provided threshold (required)
    """
    if nuclei.ndim != 2:
        raise ValueError(f"segment_nuclei_preview expects 2D, got {nuclei.shape}")

    arr = np.asarray(nuclei, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8), 0.0

    method_l = str(method).lower()
    if method_l == "otsu":
        thr = float(threshold_otsu(finite)) if threshold is None else float(threshold)
    elif method_l == "fixed":
        if threshold is None:
            raise ValueError("fixed segmentation requires threshold=")
        thr = float(threshold)
    else:
        raise ValueError(f"Unknown segmentation method: {method!r}")

    mask = arr > thr
    if min_size > 0:
        mask = remove_small_objects(mask.astype(bool), min_size=int(min_size))
    return mask.astype(np.uint8), thr
