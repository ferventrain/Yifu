from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from scipy import ndimage

try:
    from pipeline_modules.utils.errors import ErrorCode, PipelineError
    from pipeline_modules.utils.run_manifest import write_run_manifest
except ImportError:  # pragma: no cover - fallback for package-relative execution
    from ..utils.errors import ErrorCode, PipelineError
    from ..utils.run_manifest import write_run_manifest

logger = logging.getLogger(__name__)


def _configure_logging(json_logs: bool) -> None:
    if json_logs:
        class _JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                return json.dumps(
                    {
                        "level": record.levelname,
                        "logger": record.name,
                        "message": record.getMessage(),
                    },
                    ensure_ascii=False,
                )

        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonFormatter())
        logging.root.handlers.clear()
        logging.root.addHandler(handler)
        logging.root.setLevel(logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _coerce_manual_factors(factor_text: str | tuple[float, float, float] | None) -> tuple[float, float, float] | None:
    if factor_text is None:
        return None
    if isinstance(factor_text, tuple):
        return factor_text
    parts = [part.strip() for part in str(factor_text).split(",") if part.strip()]
    if len(parts) == 1:
        value = float(parts[0])
        return (value, value, value)
    if len(parts) != 3:
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            "factor must be one number or three comma-separated numbers",
            {"factor": factor_text},
        )
    return tuple(float(part) for part in parts)


class ImageDownsampler:
    """Downsample 3D TIFF stacks into TIFF + NIfTI outputs."""

    def __init__(
        self,
        resolution_config_path: str | Path | None = None,
        manual_factors: tuple[float, float, float] | None = None,
    ) -> None:
        self.manual_factors = manual_factors
        self.target_resolution = (1.0, 1.0, 1.0)

        if manual_factors:
            self.downsample_factors = manual_factors
            self.source_resolution = None
            logger.info("Using manual downsample factors (z, y, x): %s", self.downsample_factors)
        else:
            if not resolution_config_path:
                raise PipelineError(
                    ErrorCode.ARGUMENT_INVALID,
                    "Must provide either resolution_config_path or manual_factors",
                )

            self.config_path = Path(resolution_config_path)
            if not self.config_path.exists():
                raise PipelineError(
                    ErrorCode.INPUT_NOT_FOUND,
                    "Resolution config file not found",
                    {"resolution_config": str(self.config_path)},
                )

            try:
                with open(self.config_path, encoding="utf-8") as fh:
                    self.config = json.load(fh)
            except json.JSONDecodeError as exc:
                raise PipelineError(
                    ErrorCode.CONFIG_INVALID,
                    "Invalid JSON in resolution config",
                    {"resolution_config": str(self.config_path), "error": str(exc)},
                ) from exc

            if "input" in self.config and "resolution_xyz" in self.config["input"]:
                self.source_resolution = self.config["input"]["resolution_xyz"]
            elif "source_resolution" in self.config:
                self.source_resolution = self.config["source_resolution"]
            else:
                raise PipelineError(
                    ErrorCode.CONFIG_INVALID,
                    "Config missing source resolution",
                    {"required": ["input.resolution_xyz", "source_resolution"]},
                )

            if "preprocessing" in self.config and "downsample" in self.config["preprocessing"] and "target_resolution_xyz" in self.config["preprocessing"]["downsample"]:
                self.target_resolution = tuple(self.config["preprocessing"]["downsample"]["target_resolution_xyz"])
            elif "target_resolution" in self.config:
                self.target_resolution = tuple(self.config["target_resolution"])
            else:
                raise PipelineError(
                    ErrorCode.CONFIG_INVALID,
                    "Config missing target resolution",
                    {"required": ["preprocessing.downsample.target_resolution_xyz", "target_resolution"]},
                )

            self.downsample_factors = self._calculate_downsample_factors()
            logger.info("Calculated downsample factors (z, y, x): %s", self.downsample_factors)

    def _calculate_downsample_factors(self) -> tuple[float, float, float]:
        factors = [source / target for source, target in zip(self.source_resolution, self.target_resolution)]
        return tuple(round(factor, 3) for factor in factors[::-1])

    def downsample_folder(
        self,
        input_folder: str | Path,
        output_folder: str | Path | None = None,
        *,
        is_mask: bool = False,
        chunk_size: int = 100,
    ) -> dict[str, Any]:
        started_at = time.time()
        input_path = Path(input_folder)
        if not input_path.exists():
            raise PipelineError(
                ErrorCode.INPUT_NOT_FOUND,
                "Input folder not found",
                {"input_folder": str(input_path)},
            )

        if output_folder is None:
            suffix = "_downsample_mask" if is_mask else "_downsample"
            if is_mask and input_path.stem.endswith("_mask"):
                output_path = input_path.parent / f"{input_path.stem.replace('_mask', '')}{suffix}"
            else:
                output_path = input_path.parent / f"{input_path.stem}{suffix}"
        else:
            output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        tiff_files = sorted(input_path.glob("*.tif*"))
        if not tiff_files:
            ome_tiffs = list(input_path.glob("*.ome.tiff"))
            if ome_tiffs:
                result = self._process_single_file(ome_tiffs[0], output_path, is_mask)
            else:
                raise PipelineError(
                    ErrorCode.INPUT_NOT_FOUND,
                    "No TIFF files found for downsampling",
                    {"input_folder": str(input_path)},
                )
        else:
            result = self._process_stack(tiff_files, output_path, is_mask, chunk_size)

        manifest_path = write_run_manifest(
            output_path,
            module="preprocessing",
            entrypoint="downsample_folder",
            inputs={
                "input_folder": str(input_path),
                "output_folder": str(output_path),
                "is_mask": is_mask,
                "chunk_size": chunk_size,
                "downsample_factors_zyx": self.downsample_factors,
                "target_resolution_xyz": self.target_resolution,
            },
            outputs=[output_path, output_path / "volume.nii.gz", output_path / "original_shape.json"],
            started_at=started_at,
            extra=result,
        )
        result["manifest_path"] = str(manifest_path)
        return result

    def _process_stack(
        self,
        tiff_files: list[Path],
        output_path: Path,
        is_mask: bool,
        chunk_size: int,
    ) -> dict[str, Any]:
        logger.info("Found %d TIFF slices in %s", len(tiff_files), tiff_files[0].parent)
        first_img = tifffile.imread(tiff_files[0])
        original_shape = (len(tiff_files),) + first_img.shape
        dtype = first_img.dtype

        with open(output_path / "original_shape.json", "w", encoding="utf-8") as fh:
            json.dump({"original_shape": original_shape}, fh)

        interp_order = 0
        all_downsampled_slices: list[np.ndarray] = []
        num_chunks = (len(tiff_files) + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(tiff_files))
            chunk_data = [tifffile.imread(file_path) for file_path in tiff_files[start_idx:end_idx]]
            chunk_volume = np.stack(chunk_data, axis=0)
            downsampled_chunk = ndimage.zoom(
                chunk_volume,
                self.downsample_factors,
                order=interp_order,
                mode="nearest",
                prefilter=not is_mask,
            )
            for index in range(downsampled_chunk.shape[0]):
                all_downsampled_slices.append(downsampled_chunk[index])

        full_downsampled_volume = np.stack(all_downsampled_slices, axis=0)
        for index in range(full_downsampled_volume.shape[0]):
            tifffile.imwrite(output_path / f"ds_{index:04d}.tiff", full_downsampled_volume[index].astype(dtype))

        nifti_path = output_path / "volume.nii.gz"
        self._save_as_nifti(full_downsampled_volume, nifti_path)
        logger.info("Downsampled shape: %s", full_downsampled_volume.shape)

        return {
            "success": True,
            "output_dir": str(output_path),
            "nifti_path": str(nifti_path),
            "original_shape": list(original_shape),
            "downsampled_shape": list(full_downsampled_volume.shape),
            "dtype": str(dtype),
            "is_mask": is_mask,
        }

    def _process_single_file(self, file_path: Path, output_path: Path, is_mask: bool) -> dict[str, Any]:
        volume = tifffile.imread(file_path)
        if volume.ndim == 2:
            volume = volume[np.newaxis, ...]

        interp_order = 0
        downsampled_volume = ndimage.zoom(
            volume,
            self.downsample_factors,
            order=interp_order,
            mode="nearest",
            prefilter=not is_mask,
        )

        with open(output_path / "original_shape.json", "w", encoding="utf-8") as fh:
            json.dump({"original_shape": list(volume.shape)}, fh)

        for index in range(downsampled_volume.shape[0]):
            tifffile.imwrite(output_path / f"ds_{index:04d}.tiff", downsampled_volume[index].astype(volume.dtype))

        nifti_path = output_path / "volume.nii.gz"
        self._save_as_nifti(downsampled_volume, nifti_path)

        return {
            "success": True,
            "output_dir": str(output_path),
            "nifti_path": str(nifti_path),
            "original_shape": list(volume.shape),
            "downsampled_shape": list(downsampled_volume.shape),
            "dtype": str(volume.dtype),
            "is_mask": is_mask,
        }

    def _save_as_nifti(self, volume: np.ndarray, output_path: Path) -> None:
        try:
            import nibabel as nib
        except ModuleNotFoundError as exc:
            raise PipelineError(
                ErrorCode.DEPENDENCY_MISSING,
                "nibabel is required to write NIfTI outputs",
                {"dependency": "nibabel", "error": str(exc)},
            ) from exc

        volume_xyz = np.transpose(volume, (2, 1, 0))
        affine = np.eye(4)
        affine[0, 0] = self.target_resolution[0]
        affine[1, 1] = self.target_resolution[1]
        affine[2, 2] = self.target_resolution[2]
        nib.save(nib.Nifti1Image(volume_xyz, affine), output_path)


def downsample_folder(
    input_folder: str | Path,
    *,
    resolution_config_path: str | Path | None = None,
    manual_factors: tuple[float, float, float] | None = None,
    output_folder: str | Path | None = None,
    is_mask: bool = False,
    chunk_size: int = 100,
) -> dict[str, Any]:
    downsampler = ImageDownsampler(resolution_config_path, manual_factors)
    return downsampler.downsample_folder(
        input_folder,
        output_folder=output_folder,
        is_mask=is_mask,
        chunk_size=chunk_size,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downsample LSFM TIFF stacks into TIFF + NIfTI outputs")
    parser.add_argument("--input_folder", required=True, help="Input folder containing TIFF files")
    parser.add_argument("--resolution_config", help="JSON file with resolution info")
    parser.add_argument("--factor", type=str, help='Manual downsample factors "z,y,x"')
    parser.add_argument("--output_folder", help="Optional output folder path")
    parser.add_argument("--is_mask", action="store_true", help="Treat input as a mask")
    parser.add_argument("--chunk_size", type=int, default=100, help="Z-slices per chunk for processing")
    parser.add_argument("--json_logs", action="store_true", help="Emit NDJSON log records to stderr")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _configure_logging(args.json_logs)

    try:
        result = downsample_folder(
            args.input_folder,
            resolution_config_path=args.resolution_config,
            manual_factors=_coerce_manual_factors(args.factor),
            output_folder=args.output_folder,
            is_mask=args.is_mask,
            chunk_size=args.chunk_size,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except PipelineError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False), file=sys.stderr)
        return exc.exit_code
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        logger.exception("Unhandled downsample error: %s", exc)
        wrapped = PipelineError(ErrorCode.INTERNAL_ERROR, "Unhandled downsample error", {"error": str(exc)})
        print(json.dumps(wrapped.to_dict(), ensure_ascii=False), file=sys.stderr)
        return wrapped.exit_code


if __name__ == "__main__":
    sys.exit(main())
