"""Pydantic configuration models for the tubule reconstruction module.

These models are the ground truth for:

- validating ``tubule_reconstruction`` sections of ``config.json``
- generating JSON Schema for agents / IDE integrations
- documenting what each parameter means (docstring + ``Field(description=...)``)

Model design notes
------------------
- All parameters have sensible defaults so the module can run from a bare
  ``TubuleReconstructionCfg()`` instance.
- ``resolution_xyz`` accepts either a 3-tuple or a ``"x,y,z"`` string, matching
  the existing CLI ergonomics.
- Models are frozen (``model_config = ConfigDict(frozen=True)``) so they are
  safe to pass around as value objects; callers mutate via ``model_copy``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

try:
    from typing import Annotated
except ImportError:  # pragma: no cover - Python < 3.9
    from typing_extensions import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
    from pipeline_modules.utils.sample_layout import SampleLayout
except ImportError:  # running the file directly without project root on sys.path
    SampleLayout = None  # type: ignore[assignment,misc]


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


class TeasarParams(BaseModel):
    """TEASAR skeletonization parameters passed to ``kimimaro.skeletonize``.

    Defaults mirror ``DEFAULT_TEASAR_PARAMS`` in ``kimimaro_reconstruction``.
    See the module README for guidance on which parameters are worth tuning.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    scale: float = Field(1.5, description="TEASAR scale term")
    const: int = Field(300, description="TEASAR const term")
    pdrf_scale: int = Field(100_000, description="Path distance penalty scale")
    pdrf_exponent: int = Field(4, description="Path distance penalty exponent")
    soma_acceptance_threshold: int = Field(3500, description="Soma-like region acceptance threshold")
    soma_detection_threshold: int = Field(750, description="Soma-like region detection threshold")
    soma_invalidation_scale: float = Field(1.0, description="Soma invalidation scale")
    soma_invalidation_const: int = Field(300, description="Soma invalidation const")
    max_paths: Optional[int] = Field(None, description="Per-component path cap; None disables it")


class TubuleReconstructionCfg(BaseModel):
    """Top-level configuration for vessel-skeleton reconstruction.

    Corresponds to the ``tubule_reconstruction`` section of ``config.json``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = Field(False, description="Whether the main pipeline should run this stage")
    mask_dataset_name: str = Field("0", description="Dataset name inside the mask Zarr group")
    foreground_label: int = Field(1, description="Voxel value treated as vessel foreground")
    resolution_xyz: _Triplet = Field((1.0, 1.0, 1.0), description="Mask voxel size in um as (x, y, z)")
    dust_threshold: int = Field(0, ge=0, description="Minimum connected-component size passed to kimimaro")
    fix_borders: bool = Field(True, description="Enable kimimaro border fixing")
    parallel: int = Field(1, ge=1, description="kimimaro worker count per chunk")

    save_skeleton: bool = Field(False, description="Export skeleton vertex/edge CSV tables")
    save_swc: bool = Field(False, description="Export one SWC file per skeleton (requires save_skeleton)")

    chunkwise: bool = Field(False, description="Process the mask chunk-by-chunk instead of loading it whole")
    chunk_workers: int = Field(1, ge=1, description="Worker processes for chunkwise mode")
    process_existing_only: bool = Field(
        False, description="In chunkwise mode, only process chunks that physically exist in the store"
    )
    halo_zyx: Tuple[int, int, int] = Field((0, 0, 0), description="Halo overlap in voxels as (z, y, x)")
    stitch: bool = Field(True, description="Stitch endpoints across chunk boundaries in chunkwise mode")
    stitch_max_distance_um: float = Field(5.0, ge=0.0, description="Max distance for cross-chunk endpoint stitching")

    merge_branch_points_distance_um: float = Field(0.0, ge=0.0, description="Merge branch points within this distance (um). 0=disabled")
    prune_spurs_max_length_um: float = Field(0.0, ge=0.0, description="Prune terminal branches shorter than this (um). 0=disabled")

    teasar_params: TeasarParams = Field(default_factory=TeasarParams)

    output_dirname: str = Field(
        "tubule_reconstruction",
        description="Subdirectory under sample_dir that receives this module's outputs",
    )

    @field_validator("resolution_xyz", mode="before")
    @classmethod
    def _validate_resolution_xyz(cls, value: Any) -> tuple[float, float, float]:
        return _coerce_triplet(value)

    @field_validator("halo_zyx", mode="before")
    @classmethod
    def _validate_halo_zyx(cls, value: Any) -> tuple[int, int, int]:
        return _coerce_int_triplet(value)


class RegionVesselAnalysisCfg(BaseModel):
    """Configuration for ``region_vessel_analysis.analyze_regions_from_skeleton``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    annotation_dataset_name: str = Field("0", description="Dataset name inside the annotation Zarr group")
    annotation_resolution_xyz: _Triplet = Field(
        (1.0, 1.0, 1.0),
        description="Annotation voxel size in um as (x, y, z); default should match config input.resolution_xyz",
    )
    regions: Tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Region queries; each entry is an Allen acronym, full name, or integer id "
            "(as string). Sub-tree is automatically included."
        ),
    )

    @field_validator("annotation_resolution_xyz", mode="before")
    @classmethod
    def _validate_annotation_resolution(cls, value: Any) -> tuple[float, float, float]:
        return _coerce_triplet(value)

    @field_validator("regions", mode="before")
    @classmethod
    def _coerce_regions(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return tuple()
        if isinstance(value, str):
            import re

            parts = re.split(r"[,;\n]", value)
        elif isinstance(value, Sequence):
            parts = list(value)
        else:
            raise TypeError(f"Cannot coerce {value!r} to a region list")
        return tuple(str(p).strip() for p in parts if str(p).strip())


def layout_for_sample(
    sample_dir: str | Path,
    signal_ch: str = "ch0",
    reg_ch: str = "ch1",
    *,
    require_exists: bool = False,
) -> "SampleLayout":
    """Build a :class:`~pipeline_modules.utils.SampleLayout` for *sample_dir*.

    This is the preferred way for ``tubule_reconstruction`` code to resolve
    file paths: call this once and reference paths via the layout object.

    Parameters
    ----------
    sample_dir:
        Root directory for the sample.
    signal_ch:
        Short channel label for the signal / mask channel (default ``"ch0"``).
    reg_ch:
        Short channel label for the registration channel (default ``"ch1"``).
    require_exists:
        Forward to :class:`SampleLayout`; raise if *sample_dir* does not exist.

    Raises
    ------
    RuntimeError
        If ``pydantic`` / ``pipeline_modules.utils`` is not importable.
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


def output_dir_for_sample(
    sample_dir: str | Path,
    signal_ch: str = "ch0",
    reg_ch: str = "ch1",
) -> Path:
    """Return the tubule reconstruction output directory for *sample_dir*.

    Shorthand for ``layout_for_sample(...).tubule_reconstruction_dir``.
    """
    return layout_for_sample(sample_dir, signal_ch, reg_ch).tubule_reconstruction_dir


def export_json_schema() -> dict[str, Any]:
    """Return a combined JSON Schema for the module's configuration surface."""
    return {
        "TubuleReconstructionCfg": TubuleReconstructionCfg.model_json_schema(),
        "RegionVesselAnalysisCfg": RegionVesselAnalysisCfg.model_json_schema(),
    }


def load_capability_manifest() -> dict[str, Any]:
    """Load and return the module's capability manifest as a plain dict.

    The manifest file ``capability_manifest.json`` lives alongside this module
    and is the machine-readable description of all entrypoints, inputs,
    outputs, and configuration models exposed by ``tubule_reconstruction``.
    """
    import json as _json

    manifest_path = Path(__file__).parent / "capability_manifest.json"
    with open(manifest_path, encoding="utf-8") as fh:
        return _json.load(fh)


if __name__ == "__main__":  # pragma: no cover - small CLI helper
    import json

    print(json.dumps(export_json_schema(), indent=2, ensure_ascii=False))
