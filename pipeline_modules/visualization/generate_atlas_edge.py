from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
import tifffile

try:
    from pipeline_modules.segmentation.zarr_utils import open_zarr_dataset
except ModuleNotFoundError:  # pragma: no cover - optional zarr support
    open_zarr_dataset = None


def read_label_volume(path: str | Path, *, dataset_name: str = "0") -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".zarr":
        if open_zarr_dataset is None:
            raise ModuleNotFoundError("Zarr input requires pipeline_modules.segmentation.zarr_utils and zarr dependencies.")
        return np.asarray(open_zarr_dataset(path, dataset_name=dataset_name)[:])
    return np.asarray(tifffile.imread(str(path)))


def build_atlas_edge(label_volume: np.ndarray, *, include_brain_outline: bool = True, value: int = 255) -> np.ndarray:
    labels = np.asarray(label_volume)
    if labels.ndim != 3:
        raise ValueError(f"Atlas label volume must be 3D, got shape: {labels.shape}")

    edge = np.zeros(labels.shape, dtype=bool)
    edge[:-1, :, :] |= labels[:-1, :, :] != labels[1:, :, :]
    edge[1:, :, :] |= labels[1:, :, :] != labels[:-1, :, :]
    edge[:, :-1, :] |= labels[:, :-1, :] != labels[:, 1:, :]
    edge[:, 1:, :] |= labels[:, 1:, :] != labels[:, :-1, :]
    edge[:, :, :-1] |= labels[:, :, :-1] != labels[:, :, 1:]
    edge[:, :, 1:] |= labels[:, :, 1:] != labels[:, :, :-1]

    brain_mask = labels > 0
    if include_brain_outline:
        edge |= brain_mask ^ ndimage.binary_erosion(brain_mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    else:
        edge &= brain_mask

    return edge.astype(np.uint8) * np.uint8(value)


def generate_atlas_edge(
    label: str | Path,
    output: str | Path,
    *,
    dataset_name: str = "0",
    include_brain_outline: bool = True,
    value: int = 255,
) -> Path:
    labels = read_label_volume(label, dataset_name=dataset_name)
    edge = build_atlas_edge(labels, include_brain_outline=include_brain_outline, value=value)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(output_path), edge, compression="lzw")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a 3D edge TIFF from an atlas label volume.")
    parser.add_argument("--label", required=True, help="Atlas label TIFF or Zarr path")
    parser.add_argument("--output", required=True, help="Output edge TIFF path")
    parser.add_argument("--dataset-name", default="0", help="Dataset name when --label is a Zarr group")
    parser.add_argument("--no-brain-outline", action="store_true", help="Only keep label-to-label edges inside the brain")
    parser.add_argument("--value", type=int, default=255, help="Output edge intensity")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output_path = generate_atlas_edge(
            args.label,
            args.output,
            dataset_name=args.dataset_name,
            include_brain_outline=not args.no_brain_outline,
            value=args.value,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    print(json.dumps({"label": args.label, "output": str(output_path)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
