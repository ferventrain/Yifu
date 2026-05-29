"""Pydantic configuration models for the registration module.

These models are the ground truth for:

- validating ``registration`` / ``analysis`` sections of ``config.json``
- generating JSON Schema for agents / IDE integrations
- documenting what each parameter means (docstring + ``Field(description=...)``)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

try:
    from typing import Annotated
except ImportError:  # pragma: no cover - Python < 3.9
    from typing_extensions import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
    from pipeline_modules.utils.errors import ErrorCode, PipelineError
    from pipeline_modules.utils.run_manifest import write_run_manifest
    from pipeline_modules.utils.sample_layout import SampleLayout
except ImportError:
    PipelineError = None  # type: ignore[assignment,misc]
    ErrorCode = None  # type: ignore[assignment]
    write_run_manifest = None  # type: ignore[assignment]
    SampleLayout = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Shared type aliases
# ---------------------------------------------------------------------------

_Triplet = Annotated[
    Tuple[float, float, float],
    Field(description="Three-value tuple in (x, y, z) order"),
]


def _coerce_triplet(value: Any) -> tuple[float, float, float]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, Sequence):
        parts = list(value)
    else:
        raise TypeError(f"Cannot coerce {value!r} to (x, y, z) triplet")
    if len(parts) != 3:
        raise ValueError(f"Expected 3 values, got {len(parts)}: {value!r}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _coerce_int_triplet(value: Any) -> tuple[int, int, int]:
    coerced = _coerce_triplet(value)
    return (int(coerced[0]), int(coerced[1]), int(coerced[2]))


# ---------------------------------------------------------------------------
# RegistrationCfg  —  corresponds to config.json "registration" section
# ---------------------------------------------------------------------------


class RegistrationCfg(BaseModel):
    """Configuration for ANTs-based atlas ↔ sample registration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: str = Field("ants", description="Registration backend; currently only 'ants'")
    mode: str = Field(
        "atlas2image",
        description="Direction: 'atlas2image' for density analysis, 'image2atlas' for heatmaps",
    )
    atlas_path: str = Field("", description="Path to Allen atlas template image (.tiff / .nii.gz)")
    annotation_path: str = Field("", description="Path to Allen atlas annotation label image")
    transform_type: str = Field(
        "SyN",
        description="ANTs transform type: Rigid, Affine, SyN, SyNRA",
    )
    allow_reflection: bool = Field(
        False,
        description="Allow ANTs reflection during registration (usually disabled to avoid flipping)",
    )
    save_registered_image: bool = Field(False, description="Save the warped image as TIFF + NIfTI")
    save_transforms: bool = Field(False, description="Copy forward/inverse transforms to sample_dir/transforms/")
    save_upsampled_label: bool = Field(
        True,
        description="Save warped atlas annotation as sample-space TIFF stack: sample_dir/upsampled_atlas_label/",
    )
    save_upsampled_label_zarr: bool = Field(
        True,
        description="Also convert the sample-space atlas annotation to sample_dir/upsampled_atlas_label.zarr",
    )
    save_upsampled_label_hemisphere_zarr: bool = Field(
        False,
        description="Also derive sample_dir/atlas_label_hemisphere.zarr for hemisphere-aware analysis",
    )
    flip_atlas: List[bool] = Field(
        [False, False, False],
        description="Flip atlas before registration: [flip_x, flip_y, flip_z]",
    )
    upsample_method: str = Field("nearest", description="Interpolation for upsampling: nearest, linear, cubic, quintic")
    chunk_size: int = Field(50, ge=1, description="Chunk size (slices) for chunked label upsampling")

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, v: str) -> str:
        if v not in ("atlas2image", "image2atlas"):
            raise ValueError(f"mode must be 'atlas2image' or 'image2atlas', got '{v}'")
        return v

    @field_validator("transform_type")
    @classmethod
    def _validate_transform(cls, v: str) -> str:
        allowed = {"Rigid", "Affine", "SyN", "SyNRA", "ElasticSyN"}
        if v not in allowed:
            raise ValueError(f"transform_type must be one of {allowed}, got '{v}'")
        return v


# ---------------------------------------------------------------------------
# AnalysisCfg  —  corresponds to config.json "analysis" section
# ---------------------------------------------------------------------------


class AnalysisCfg(BaseModel):
    """Configuration for Zarr-native block-graph region signal analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    density_config: str = Field(
        "pipeline_modules/registration/Region_Csv_Rev1_updated.CSV",
        description="Path to Allen region CSV (absolute or relative to project root)",
    )
    output_format: str = Field("xlsx", description="Output format: 'xlsx'")
    dataset_name: str = Field("0", description="Dataset name inside Zarr groups")
    block_size: Optional[Tuple[int, int, int]] = Field(
        None,
        description="Override block size as (z, y, x); None infers from Zarr chunks",
    )
    foreground_mode: str = Field(
        "equal",
        description="Mask interpretation: 'equal' (mask == foreground_label) or 'nonzero'",
    )
    foreground_label: int = Field(1, description="Foreground label value when foreground_mode='equal'")
    min_voxels: int = Field(10, ge=1, description="Minimum merged component size to keep")
    flush_every: int = Field(25, ge=0, description="Rewrite Excel after every N rows (0 = only at end)")
    resolution_xyz: _Triplet = Field((1.0, 1.0, 1.0), description="Voxel size in um as (x, y, z)")
    pass1_workers: int = Field(1, ge=1, description="Worker processes for Pass 1 block scanning")
    use_hemisphere_label: bool = Field(
        False,
        description="Use sample_dir/atlas_label_hemisphere.zarr for left/right analysis instead of x-midpoint splitting",
    )

    @field_validator("foreground_mode")
    @classmethod
    def _validate_fg_mode(cls, v: str) -> str:
        if v not in ("equal", "nonzero"):
            raise ValueError(f"foreground_mode must be 'equal' or 'nonzero', got '{v}'")
        return v

    @field_validator("resolution_xyz", mode="before")
    @classmethod
    def _validate_resolution(cls, value: Any) -> tuple[float, float, float]:
        return _coerce_triplet(value)

    @field_validator("block_size", mode="before")
    @classmethod
    def _validate_block_size(cls, value: Any) -> tuple[int, int, int] | None:
        if value is None or value == "" or value == "None":
            return None
        return _coerce_int_triplet(value)

    @field_validator("use_hemisphere_label")
    @classmethod
    def _validate_use_hemisphere_label(cls, value: bool) -> bool:
        return bool(value)


# ---------------------------------------------------------------------------
# layout_for_sample  —  convenience helper
# ---------------------------------------------------------------------------


def layout_for_sample(
    sample_dir: str | Path,
    signal_ch: str = "ch0",
    reg_ch: str = "ch1",
    *,
    require_exists: bool = False,
) -> "SampleLayout":
    """Build a :class:`SampleLayout` for *sample_dir*.

    This is the preferred way for registration code to resolve file paths.
    """
    if SampleLayout is None:
        raise RuntimeError(
            "pipeline_modules.utils is not importable. "
            "Ensure the project root is on sys.path before calling layout_for_sample()."
        )
    return SampleLayout(
        sample_dir=Path(sample_dir),
        signal_ch=signal_ch,
        reg_ch=reg_ch,
        require_exists=require_exists,
    )


# ---------------------------------------------------------------------------
# export_json_schema / load_capability_manifest
# ---------------------------------------------------------------------------


def export_json_schema() -> dict[str, Any]:
    """Return a combined JSON Schema for the module's configuration surface."""
    return {
        "RegistrationCfg": RegistrationCfg.model_json_schema(),
        "AnalysisCfg": AnalysisCfg.model_json_schema(),
    }


def load_capability_manifest() -> dict[str, Any]:
    """Load and return the module's capability manifest as a plain dict."""
    manifest_path = Path(__file__).parent / "capability_manifest.json"
    with open(manifest_path, encoding="utf-8") as fh:
        return json.load(fh)


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(export_json_schema(), indent=2, ensure_ascii=False))
