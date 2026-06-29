from __future__ import annotations

from pathlib import Path

import numpy as np


def ensure_label_storage_dtype(array: np.ndarray) -> np.dtype:
    """Pick a TIFF-safe integer dtype that preserves atlas label ids."""
    if np.issubdtype(array.dtype, np.integer):
        max_value = int(array.max()) if array.size > 0 else 0
        min_value = int(array.min()) if array.size > 0 else 0
        if min_value >= 0:
            if max_value <= np.iinfo(np.uint16).max:
                return np.uint16
            if max_value <= np.iinfo(np.uint32).max:
                return np.uint32
        if min_value >= np.iinfo(np.int32).min and max_value <= np.iinfo(np.int32).max:
            return np.int32
        return np.int64

    if np.issubdtype(array.dtype, np.floating):
        if np.allclose(array, np.round(array), atol=0):
            rounded = np.rint(array)
            return ensure_label_storage_dtype(rounded.astype(np.int64))

    raise ValueError(f"Label array must contain integer ids, got dtype={array.dtype}")


def load_label_array_preserving_ids(path: str | Path) -> np.ndarray:
    """Load atlas labels without routing large integer ids through float32."""
    label_path = Path(path)
    suffixes = "".join(label_path.suffixes).lower()
    if suffixes.endswith((".nii", ".nii.gz")):
        try:
            import nibabel as nib
        except ModuleNotFoundError as exc:
            raise RuntimeError("nibabel is required to load NIfTI atlas labels") from exc

        nii = nib.load(str(label_path))
        data = np.asanyarray(nii.dataobj)
    elif label_path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile

        data = tifffile.imread(str(label_path))
        if data.ndim == 3:
            # tifffile reads image stacks as (Z, Y, X), while ants.from_numpy
            # expects image arrays in ANTs order (X, Y, Z). Match the previous
            # ants.image_read(...).numpy() orientation without losing large IDs.
            data = np.transpose(data, (2, 1, 0))
    else:
        raise ValueError(f"Unsupported atlas label format: {label_path}")

    rounded = np.rint(data)
    if not np.allclose(data, rounded, atol=0):
        raise ValueError(f"Atlas label image contains non-integer values: {label_path}")
    return rounded.astype(ensure_label_storage_dtype(rounded), copy=False)


def build_label_id_codec(label_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode large atlas ids as small integers before ANTs warping.

    ANTs images commonly travel through float32 storage. Allen structure ids can
    be hundreds of millions, which float32 cannot represent exactly. Warping a
    dense 0..N code image preserves nearest-neighbor labels, then we decode back
    to the original structure ids before saving.
    """
    rounded = np.rint(label_array)
    if not np.allclose(label_array, rounded, atol=0):
        raise ValueError("Atlas label image contains non-integer values before encoding")

    original_ids = np.unique(rounded.astype(np.int64, copy=False))
    code_dtype = np.uint32 if original_ids.size > np.iinfo(np.uint16).max else np.uint16
    encoded = np.searchsorted(original_ids, rounded.astype(np.int64, copy=False)).astype(code_dtype, copy=False)
    return encoded, original_ids


def decode_label_codes(encoded_array: np.ndarray, label_id_lut: np.ndarray | None) -> np.ndarray:
    if label_id_lut is None:
        return encoded_array

    rounded_codes = np.rint(encoded_array).astype(np.int64, copy=False)
    if rounded_codes.size and (rounded_codes.min() < 0 or rounded_codes.max() >= len(label_id_lut)):
        raise ValueError(
            "Warped atlas label code is outside the label lookup table range: "
            f"min={int(rounded_codes.min())}, max={int(rounded_codes.max())}, lut_size={len(label_id_lut)}"
        )
    return label_id_lut[rounded_codes]
