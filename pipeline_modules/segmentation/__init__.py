"""Segmentation module public API."""

from .config import (
    CfosUNetInferenceCfg,
    SegmentationCfg,
    ThresholdSegmentationCfg,
    export_json_schema,
    layout_for_sample,
    load_capability_manifest,
)

try:
    from .zarr_utils import export_zarr_to_tiff, open_zarr_dataset
except ModuleNotFoundError as exc:  # pragma: no cover - optional runtime dependency missing
    _zarr_import_error = str(exc)

    def export_zarr_to_tiff(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError(f"segmentation Zarr utilities are unavailable: {_zarr_import_error}")

    def open_zarr_dataset(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError(f"segmentation Zarr utilities are unavailable: {_zarr_import_error}")

try:
    from .intensity_threshold_segmentor import run_threshold_segmentation, segment_chunk
except ModuleNotFoundError as exc:  # pragma: no cover
    _threshold_import_error = str(exc)

    def run_threshold_segmentation(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError(f"threshold segmentation is unavailable: {_threshold_import_error}")

    def segment_chunk(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError(f"threshold segmentation is unavailable: {_threshold_import_error}")

try:
    from .cfos_unet_inference import run_cfos_unet_inference
    from .cfos_unet_model import build_cfos_unet_classes, load_cfos_unet_checkpoint, normalize_volume
except ModuleNotFoundError as exc:  # pragma: no cover
    _cfos_import_error = str(exc)

    def run_cfos_unet_inference(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError(f"cfos_unet inference is unavailable: {_cfos_import_error}")

    def build_cfos_unet_classes(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError(f"cfos_unet inference is unavailable: {_cfos_import_error}")

    def load_cfos_unet_checkpoint(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError(f"cfos_unet inference is unavailable: {_cfos_import_error}")

    def normalize_volume(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError(f"cfos_unet inference is unavailable: {_cfos_import_error}")

__all__ = [
    "CfosUNetInferenceCfg",
    "SegmentationCfg",
    "ThresholdSegmentationCfg",
    "build_cfos_unet_classes",
    "export_json_schema",
    "export_zarr_to_tiff",
    "layout_for_sample",
    "load_capability_manifest",
    "load_cfos_unet_checkpoint",
    "normalize_volume",
    "open_zarr_dataset",
    "run_cfos_unet_inference",
    "run_threshold_segmentation",
    "segment_chunk",
]
