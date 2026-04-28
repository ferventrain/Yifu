"""Preprocessing module public API."""

from .config import (
    ChannelSubtractionCfg,
    ClaheCfg,
    DownsampleCfg,
    MedianFilterCfg,
    PreprocessingCfg,
    RollingBallCfg,
    ScatteringRemovalCfg,
    TophatCfg,
    ZarrCfg,
    export_json_schema,
    layout_for_sample,
    load_capability_manifest,
)
try:
    from .preprocessor import (
        Preprocessor,
        apply_processing_steps,
        normalize_channel_list,
        run_preprocessing,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - optional runtime dependency missing
    Preprocessor = None  # type: ignore[assignment]
    _preprocessor_import_error = str(exc)

    def apply_processing_steps(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError(f"preprocessing support is unavailable: {_preprocessor_import_error}")

    def normalize_channel_list(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError(f"preprocessing support is unavailable: {_preprocessor_import_error}")

    def run_preprocessing(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError(f"preprocessing support is unavailable: {_preprocessor_import_error}")

try:
    from .downsample import ImageDownsampler, downsample_folder
except ModuleNotFoundError as exc:  # pragma: no cover - optional runtime dependency missing
    ImageDownsampler = None  # type: ignore[assignment]
    _downsample_import_error = str(exc)

    def downsample_folder(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError(f"downsample support is unavailable: {_downsample_import_error}")

try:
    from .tiff_to_zarr import convert_tiff_to_zarr
except ModuleNotFoundError as exc:  # pragma: no cover - optional runtime dependency missing
    _tiff_to_zarr_import_error = str(exc)

    def convert_tiff_to_zarr(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError(f"TIFF-to-Zarr support is unavailable: {_tiff_to_zarr_import_error}")

__all__ = [
    "ChannelSubtractionCfg",
    "ClaheCfg",
    "DownsampleCfg",
    "ImageDownsampler",
    "MedianFilterCfg",
    "PreprocessingCfg",
    "Preprocessor",
    "RollingBallCfg",
    "ScatteringRemovalCfg",
    "TophatCfg",
    "ZarrCfg",
    "apply_processing_steps",
    "convert_tiff_to_zarr",
    "downsample_folder",
    "export_json_schema",
    "layout_for_sample",
    "load_capability_manifest",
    "normalize_channel_list",
    "run_preprocessing",
]
