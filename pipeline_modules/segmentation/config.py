"""Configuration models for the segmentation module.

This module mirrors the lightweight dataclass-based Agent-Native surface used
by preprocessing so it remains importable even when the full inference stack
is not installed on the current machine.
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
class ThresholdSegmentationCfg(_ModelMixin):
    apply: bool = True
    value: str = "otsu"
    sigma: float = 0.0
    min_object_size: int = 10
    output_mode: str = "binary"
    dataset_name: str = "0"
    process_existing_only: bool = False


@dataclass(frozen=True)
class CfosUNetInferenceCfg(_ModelMixin):
    apply: bool = True
    checkpoint_path: str = ""
    save_probability: bool = False
    probability_zarr: str = ""
    dataset_name: str = "0"
    patch_size: tuple[int, int, int] | None = None
    overlap: float = 0.25
    batch_size: int = 4
    device: str = "auto"
    foreground_class: int = 1
    probability_threshold: float = 0.5
    process_existing_only: bool = False
    rerun_if_model_updated: bool = False
    output_mode: str = "binary"
    output_dtype: str = "uint8"
    probability_dtype: str = "float16"
    chunk_size: tuple[int, int, int] | None = None
    normalize_percentiles: tuple[float, float] = (1.0, 99.5)

    def __post_init__(self) -> None:
        if self.patch_size is not None:
            object.__setattr__(self, "patch_size", _coerce_int_triplet(self.patch_size))
        if self.chunk_size is not None:
            object.__setattr__(self, "chunk_size", _coerce_int_triplet(self.chunk_size))
        if isinstance(self.normalize_percentiles, str):
            parts = [part.strip() for part in self.normalize_percentiles.split(",") if part.strip()]
            if len(parts) != 2:
                raise ValueError("normalize_percentiles must contain 2 values")
            object.__setattr__(self, "normalize_percentiles", (float(parts[0]), float(parts[1])))


@dataclass(frozen=True)
class SegmentationCfg(_ModelMixin):
    """Top-level segmentation configuration."""

    method: str = "threshold"
    export_mask_tiff: bool = False
    threshold: ThresholdSegmentationCfg = field(default_factory=ThresholdSegmentationCfg)
    cfos_unet: CfosUNetInferenceCfg = field(default_factory=CfosUNetInferenceCfg)

    def __post_init__(self) -> None:
        object.__setattr__(self, "threshold", ThresholdSegmentationCfg.model_validate(self.threshold))
        object.__setattr__(self, "cfos_unet", CfosUNetInferenceCfg.model_validate(self.cfos_unet))

    @classmethod
    def model_json_schema(cls) -> dict[str, Any]:
        return {
            "title": cls.__name__,
            "type": "object",
            "properties": {
                "method": {"type": "string", "enum": ["threshold", "cfos_unet", "cellpose"]},
                "export_mask_tiff": {"type": "boolean", "default": False},
                "threshold": {"type": "object"},
                "cfos_unet": {"type": "object"},
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
    def signal_zarr(self) -> Path:
        return self.sample_dir / f"{self.signal_ch}.zarr"

    @property
    def mask_zarr(self) -> Path:
        return self.sample_dir / f"{self.signal_ch}_mask.zarr"

    @property
    def mask_tiff_dir(self) -> Path:
        return self.sample_dir / f"{self.signal_ch}_mask"


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
        "SegmentationCfg": SegmentationCfg.model_json_schema(),
        "ThresholdSegmentationCfg": ThresholdSegmentationCfg.model_json_schema(),
        "CfosUNetInferenceCfg": CfosUNetInferenceCfg.model_json_schema(),
    }


def load_capability_manifest() -> dict[str, Any]:
    """Load and return the module's capability manifest as a plain dict."""
    manifest_path = Path(__file__).parent / "capability_manifest.json"
    with open(manifest_path, encoding="utf-8") as fh:
        return json.load(fh)


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(export_json_schema(), indent=2, ensure_ascii=False))
