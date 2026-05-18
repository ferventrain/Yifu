import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
import tifffile


TIFF_SUFFIXES = {".tif", ".tiff"}
IDENTITY_DIRECTION_3D = (
    1.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    1.0,
)


def natural_sort_key(path: Path) -> List[object]:
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def list_tiff_files(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    files = [path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES]
    return sorted(files, key=natural_sort_key)


def select_file_range(files: Sequence[Path], start_name: Optional[str], end_name: Optional[str]) -> List[Path]:
    if not files:
        raise ValueError("No TIFF files found in the input directory")

    if start_name is None and end_name is None:
        return list(files)

    if start_name is None or end_name is None:
        raise ValueError("--start-file and --end-file must be provided together, or both omitted")

    name_to_index = {path.name: idx for idx, path in enumerate(files)}

    if start_name not in name_to_index:
        raise ValueError(f"Start file not found in directory: {start_name}")
    if end_name not in name_to_index:
        raise ValueError(f"End file not found in directory: {end_name}")

    start_idx = name_to_index[start_name]
    end_idx = name_to_index[end_name]
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    return list(files[start_idx : end_idx + 1])


def load_tiff_as_zyx(input_path: Path) -> np.ndarray:
    array = tifffile.imread(str(input_path))
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    elif array.ndim != 3:
        raise ValueError(f"Only 2D or 3D TIFF is supported, got shape {array.shape} for {input_path}")
    return np.asarray(array)


def save_array_as_nifti(
    array_zyx: np.ndarray,
    output_path: Path,
    spacing: Tuple[float, float, float],
    origin: Tuple[float, float, float],
    direction: Tuple[float, ...],
) -> None:
    image = sitk.GetImageFromArray(array_zyx)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_path))


def stack_tiff_files(selected_files: Sequence[Path]) -> np.ndarray:
    slices: List[np.ndarray] = []
    reference_shape = None

    for tiff_path in selected_files:
        array_zyx = load_tiff_as_zyx(tiff_path)
        if array_zyx.shape[0] != 1:
            raise ValueError(
                f"Expected each TIFF to contain a single 2D slice, got shape {array_zyx.shape} for {tiff_path}"
            )
        slice_yx = array_zyx[0]
        if reference_shape is None:
            reference_shape = slice_yx.shape
        elif slice_yx.shape != reference_shape:
            raise ValueError(
                f"All TIFF slices must have the same shape. Expected {reference_shape}, got {slice_yx.shape} for {tiff_path}"
            )
        slices.append(slice_yx)

    if not slices:
        raise ValueError("No TIFF files were selected for conversion")

    return np.stack(slices, axis=0)


def convert_tiff_range(
    input_dir: Path,
    start_name: Optional[str],
    end_name: Optional[str],
    output_path: Path,
    spacing: Tuple[float, float, float],
    origin: Tuple[float, float, float],
    direction: Tuple[float, ...],
) -> Path:
    files = list_tiff_files(input_dir)
    selected_files = select_file_range(files, start_name, end_name)
    volume_zyx = stack_tiff_files(selected_files)
    save_array_as_nifti(volume_zyx, output_path, spacing, origin, direction)
    if start_name is None and end_name is None:
        print(
            f"Converted all {len(selected_files)} TIFF files ({selected_files[0].name} -> {selected_files[-1].name}) into {output_path.name}"
        )
    else:
        print(
            f"Converted {len(selected_files)} TIFF files ({selected_files[0].name} -> {selected_files[-1].name}) into {output_path.name}"
        )
    return output_path


def parse_triplet(values: Iterable[float], argument_name: str) -> Tuple[float, float, float]:
    values = tuple(values)
    if len(values) != 3:
        raise ValueError(f"{argument_name} requires exactly 3 values, got {len(values)}")
    return float(values[0]), float(values[1]), float(values[2])


def parse_direction(values: Sequence[float]) -> Tuple[float, ...]:
    if len(values) != 9:
        raise ValueError(f"--direction requires exactly 9 values, got {len(values)}")
    return tuple(float(value) for value in values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a selected TIFF file range, or all TIFF files when no range is given, into a single 3D NIfTI .nii.gz volume."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing TIFF files.")
    parser.add_argument(
        "--start-file",
        help="Start TIFF filename, inclusive. If omitted together with --end-file, all TIFF files are used.",
    )
    parser.add_argument(
        "--end-file",
        help="End TIFF filename, inclusive. If omitted together with --start-file, all TIFF files are used.",
    )
    parser.add_argument(
        "--output-path",
        help="Output .nii.gz file path. Defaults to input-dir/volume.nii.gz.",
    )
    parser.add_argument(
        "--spacing",
        nargs=3,
        type=float,
        default=(1.0, 1.0, 1.0),
        metavar=("SX", "SY", "SZ"),
        help="Voxel spacing in X Y Z order. Default: 1 1 1",
    )
    parser.add_argument(
        "--origin",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("OX", "OY", "OZ"),
        help="Image origin in X Y Z order. Default: 0 0 0",
    )
    parser.add_argument(
        "--direction",
        nargs=9,
        type=float,
        default=IDENTITY_DIRECTION_3D,
        metavar=("D00", "D01", "D02", "D10", "D11", "D12", "D20", "D21", "D22"),
        help="3x3 direction matrix in row-major order. Default: identity",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve() if args.output_path else input_dir / "volume.nii.gz"
    spacing = parse_triplet(args.spacing, "--spacing")
    origin = parse_triplet(args.origin, "--origin")
    direction = parse_direction(args.direction)

    generated_file = convert_tiff_range(
        input_dir=input_dir,
        start_name=args.start_file,
        end_name=args.end_file,
        output_path=output_path,
        spacing=spacing,
        origin=origin,
        direction=direction,
    )
    print(f"Finished. Generated NIfTI volume: {generated_file}")


if __name__ == "__main__":
    main()
