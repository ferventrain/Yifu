"""Configuration models for the preprocessing module.

This module intentionally uses stdlib dataclasses instead of a hard dependency
on Pydantic so preprocessing remains importable in lightweight environments.
The public classes still expose ``model_validate()``, ``model_dump()``, and
``model_json_schema()`` to match the rest of the repo's Agent-Native surface.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from pipeline_modules.utils.sample_layout import SampleLayout
except ImportError:  # pragma: no cover - direct script execution fallback
    SampleLayout = None  # type: ignore[assignment,misc]


def _coerce_triplet(value: Any) -> tuple[float, float, float]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, Sequence):
        parts = list(value)
    else:
        raise TypeError(f"Cannot coerce {value!r} to a 3-value tuple")
    if len(parts) != 3:
        raise ValueError(f"Expected 3 values, got {len(parts)}: {value!r}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _coerce_int_triplet(value: Any) -> tuple[int, int, int]:
    coerced = _coerce_triplet(value)
    return (int(coerced[0]), int(coerced[1]), int(coerced[2]))


def _coerce_channel_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, (str, int)):
        raw_values = [value]
    elif isinstance(value, Sequence):
        raw_values = list(value)
    else:
        raise TypeError(f"Cannot coerce {value!r} to a channel list")

    normalized: list[str] = []
    for raw in raw_values:
        text = str(raw).strip()
        if not text:
            continue
        normalized.append(text if text.lower().startswith("ch") else f"ch{text}")
    return tuple(dict.fromkeys(ch.lower() for ch in normalized))


class _ModelMixin:
    @classmethod
    def model_validate(cls, data: Any):
        if isinstance(data, cls):
            return data
        if not isinstance(data, Mapping):
            raise TypeError(f"{cls.__name__}.model_validate expects a mapping, got {type(data)!r}")
        return cls(**dict(data))

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def model_json_schema(cls) -> dict[str, Any]:
        return {
            "title": cls.__name__,
            "type": "object",
            "description": cls.__doc__ or "",
        }


@dataclass(frozen=True)
class ChannelSubtractionCfg(_ModelMixin):
    apply: bool = False
    background_channel: str = "ch0"
    weight: float = 1.0
    adaptive: bool = False
    save_plots: bool = False
    sample_ratio: float = 0.005
    min_samples: int = 10
    max_samples: int = 50
    compression: str = "lzw"
    estimated_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        text = str(self.background_channel).strip()
        if not text:
            raise ValueError("background_channel cannot be empty")
        object.__setattr__(self, "background_channel", text if text.lower().startswith("ch") else f"ch{text}")


@dataclass(frozen=True)
class TophatCfg(_ModelMixin):
    apply: bool = False
    kernel_size: int = 21


@dataclass(frozen=True)
class RollingBallCfg(_ModelMixin):
    apply: bool = False
    radius: int = 50


@dataclass(frozen=True)
class MedianFilterCfg(_ModelMixin):
    apply: bool = False
    kernel_size: int = 3


@dataclass(frozen=True)
class ScatteringRemovalCfg(_ModelMixin):
    apply: bool = False
    sigma: float = 50.0
    weight: float = 1.0


@dataclass(frozen=True)
class ClaheCfg(_ModelMixin):
    apply: bool = False
    clip_limit: float = 2.0
    tile_grid_size: int = 8


@dataclass(frozen=True)
class EdgeSignalRemovalCfg(_ModelMixin):
    apply: bool = False
    inward_px: int = 50
    outward_px: int = 0
    prior_curve_scale: float = 1.15
    prior_curve_smooth_sigma: float = 2.0
    contour_min_object_area_px: int = 50
    contour_morph_radius_px: int = 2
    contour_dilation_px: int = 2
    brightness_pct: float = 90.0
    min_area_px: int = 50
    suppression_weight: float = 1.0
    max_workers: int = 8
    adaptive_block_size: int | None = None


@dataclass(frozen=True)
class DownsampleCfg(_ModelMixin):
    target_resolution_xyz: tuple[float, float, float] = (25.0, 25.0, 25.0)
    chunk_size: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_resolution_xyz", _coerce_triplet(self.target_resolution_xyz))


@dataclass(frozen=True)
class ZarrCfg(_ModelMixin):
    chunk_size: tuple[int, int, int] = (256, 256, 256)
    compressor: str = "default"

    def __post_init__(self) -> None:
        object.__setattr__(self, "chunk_size", _coerce_int_triplet(self.chunk_size))


@dataclass(frozen=True)
class PreprocessingCfg(_ModelMixin):
    """Top-level preprocessing configuration."""

    channels: tuple[str, ...] = field(default_factory=tuple)
    channel_subtraction: ChannelSubtractionCfg = field(default_factory=ChannelSubtractionCfg)
    tophat: TophatCfg = field(default_factory=TophatCfg)
    rolling_ball: RollingBallCfg = field(default_factory=RollingBallCfg)
    median_filter: MedianFilterCfg = field(default_factory=MedianFilterCfg)
    scattering_removal: ScatteringRemovalCfg = field(default_factory=ScatteringRemovalCfg)
    clahe: ClaheCfg = field(default_factory=ClaheCfg)
    edge_signal_removal: EdgeSignalRemovalCfg = field(default_factory=EdgeSignalRemovalCfg)
    downsample: DownsampleCfg = field(default_factory=DownsampleCfg)
    zarr: ZarrCfg = field(default_factory=ZarrCfg)

    def __post_init__(self) -> None:
        object.__setattr__(self, "channels", _coerce_channel_list(self.channels))
        object.__setattr__(self, "channel_subtraction", ChannelSubtractionCfg.model_validate(self.channel_subtraction))
        object.__setattr__(self, "tophat", TophatCfg.model_validate(self.tophat))
        object.__setattr__(self, "rolling_ball", RollingBallCfg.model_validate(self.rolling_ball))
        object.__setattr__(self, "median_filter", MedianFilterCfg.model_validate(self.median_filter))
        object.__setattr__(self, "scattering_removal", ScatteringRemovalCfg.model_validate(self.scattering_removal))
        object.__setattr__(self, "clahe", ClaheCfg.model_validate(self.clahe))
        object.__setattr__(self, "edge_signal_removal", EdgeSignalRemovalCfg.model_validate(self.edge_signal_removal))
        object.__setattr__(self, "downsample", DownsampleCfg.model_validate(self.downsample))
        object.__setattr__(self, "zarr", ZarrCfg.model_validate(self.zarr))

    @classmethod
    def model_json_schema(cls) -> dict[str, Any]:
        return {
            "title": cls.__name__,
            "type": "object",
            "properties": {
                "channels": {"type": "array", "items": {"type": "string"}},
                "channel_subtraction": {"type": "object"},
                "tophat": {"type": "object"},
                "rolling_ball": {"type": "object"},
                "median_filter": {"type": "object"},
                "scattering_removal": {"type": "object"},
                "clahe": {"type": "object"},
                "edge_signal_removal": {"type": "object"},
                "downsample": {"type": "object"},
                "zarr": {"type": "object"},
            },
        }


@dataclass(frozen=True)
class _FallbackSampleLayout:
    sample_dir: Path
    signal_ch: str = "ch0"
    reg_ch: str = "ch1"
    require_exists: bool = False

    def __post_init__(self) -> None:
        if self.require_exists and not self.sample_dir.exists():
            raise FileNotFoundError(f"sample_dir does not exist: {self.sample_dir}")

    @property
    def signal_tiff_preprocessed_dir(self) -> Path:
        return self.sample_dir / f"{self.signal_ch}_preprocessed"

    @property
    def signal_tiff_dir(self) -> Path:
        return self.sample_dir / self.signal_ch

    @property
    def reg_tiff_dir(self) -> Path:
        return self.sample_dir / self.reg_ch

    @property
    def signal_zarr(self) -> Path:
        return self.sample_dir / f"{self.signal_ch}.zarr"

    @property
    def reg_downsample_dir(self) -> Path:
        return self.sample_dir / f"{self.reg_ch}_downsample"

    @property
    def reg_downsample_nii(self) -> Path:
        return self.reg_downsample_dir / "volume.nii.gz"


def layout_for_sample(
    sample_dir: str | Path,
    signal_ch: str = "ch0",
    reg_ch: str = "ch1",
    *,
    require_exists: bool = False,
):
    """Build a sample layout for *sample_dir*."""
    layout_cls = SampleLayout if SampleLayout is not None else _FallbackSampleLayout
    return layout_cls(
        sample_dir=Path(sample_dir),
        signal_ch=signal_ch,
        reg_ch=reg_ch,
        require_exists=require_exists,
    )


def export_json_schema() -> dict[str, Any]:
    """Return a combined JSON Schema for the module's configuration surface."""
    return {
        "PreprocessingCfg": PreprocessingCfg.model_json_schema(),
        "DownsampleCfg": DownsampleCfg.model_json_schema(),
        "ZarrCfg": ZarrCfg.model_json_schema(),
    }


def load_capability_manifest() -> dict[str, Any]:
    """Load and return the module's capability manifest as a plain dict."""
    manifest_path = Path(__file__).parent / "capability_manifest.json"
    with open(manifest_path, encoding="utf-8") as fh:
        return json.load(fh)


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(export_json_schema(), indent=2, ensure_ascii=False))
