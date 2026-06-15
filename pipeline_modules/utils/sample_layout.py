"""Centralised path definitions for a single-sample directory.

``SampleLayout`` is a frozen, Pydantic-validated descriptor that maps every
conventionally-named artefact produced by the pipeline to a ``Path`` object.
It is the single source of truth for file naming so that modules, agents, and
tests can refer to paths symbolically without hard-coding string literals.

Usage::

    layout = SampleLayout(sample_dir="/data/mouse01", signal_ch="ch0", reg_ch="ch1")
    layout.signal_zarr          # Path(".../mouse01/ch0.zarr")
    layout.mask_zarr            # Path(".../mouse01/ch0_mask.zarr")
    layout.tubule_reconstruction_dir  # Path(".../mouse01/tubule_reconstruction")
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_modules.utils.deliverable_paths import (
    brain_distribution_stats_xlsx as deliverable_brain_distribution_stats_xlsx,
    heatmap_2d_dir as deliverable_heatmap_2d_dir,
    heatmap_3d_png as deliverable_heatmap_3d_png,
    results_dir as deliverable_results_dir,
    visualization_dir as deliverable_visualization_dir,
)


class SampleLayout(BaseModel):
    """Canonical paths for one sample processed through the full pipeline.

    Parameters
    ----------
    sample_dir:
        Root directory for this sample (must already exist on disk when
        ``require_exists=True``).
    signal_ch:
        Short channel label for the signal channel, e.g. ``"ch0"``.
    reg_ch:
        Short channel label for the registration channel, e.g. ``"ch1"``.
    require_exists:
        If ``True`` (default ``False``), raise ``FileNotFoundError`` when
        ``sample_dir`` does not exist on disk.  Set to ``False`` for dry-run
        or agent-planning contexts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_dir: Path
    signal_ch: str = Field("ch0", description="Signal channel label")
    reg_ch: str = Field("ch1", description="Registration channel label")
    require_exists: bool = Field(False, description="Raise if sample_dir does not exist")

    @model_validator(mode="after")
    def _check_dir(self) -> "SampleLayout":
        if self.require_exists and not self.sample_dir.exists():
            raise FileNotFoundError(f"sample_dir does not exist: {self.sample_dir}")
        return self

    # ------------------------------------------------------------------
    # Input / raw data
    # ------------------------------------------------------------------

    @property
    def signal_tiff_dir(self) -> Path:
        """Raw signal TIFF stack directory."""
        return self.sample_dir / self.signal_ch

    @property
    def signal_tiff_preprocessed_dir(self) -> Path:
        """Pre-processed (e.g. contrast-enhanced) TIFF stack directory."""
        return self.sample_dir / f"{self.signal_ch}_preprocessed"

    @property
    def reg_tiff_dir(self) -> Path:
        """Raw registration-channel TIFF stack directory."""
        return self.sample_dir / self.reg_ch

    # ------------------------------------------------------------------
    # Zarr volumes
    # ------------------------------------------------------------------

    @property
    def signal_zarr(self) -> Path:
        """Signal channel Zarr store."""
        return self.sample_dir / f"{self.signal_ch}.zarr"

    @property
    def mask_zarr(self) -> Path:
        """Binary segmentation mask Zarr store."""
        return self.sample_dir / f"{self.signal_ch}_mask.zarr"

    @property
    def atlas_label_zarr(self) -> Path:
        """Warped atlas annotation label Zarr store."""
        return self.sample_dir / "upsampled_atlas_label.zarr"

    @property
    def atlas_label_hemisphere_zarr(self) -> Path:
        """Warped atlas hemisphere label Zarr store."""
        return self.sample_dir / "atlas_label_hemisphere.zarr"

    # ------------------------------------------------------------------
    # Registration outputs
    # ------------------------------------------------------------------

    @property
    def reg_downsample_dir(self) -> Path:
        """Registration downsampled volume directory."""
        return self.sample_dir / f"{self.reg_ch}_downsample"

    @property
    def reg_downsample_nii(self) -> Path:
        """Registration downsampled NIfTI volume."""
        return self.reg_downsample_dir / "volume.nii.gz"

    @property
    def atlas_label_tiff_dir(self) -> Path:
        """Warped atlas label TIFF stack directory."""
        return self.sample_dir / "upsampled_atlas_label"

    # ------------------------------------------------------------------
    # Segmentation exports
    # ------------------------------------------------------------------

    @property
    def mask_tiff_dir(self) -> Path:
        """Exported segmentation mask TIFF stack directory."""
        return self.sample_dir / f"{self.signal_ch}_mask"

    # ------------------------------------------------------------------
    # Analysis outputs
    # ------------------------------------------------------------------

    @property
    def results_dir(self) -> Path:
        """Directory for analysis Excel deliverables."""
        return deliverable_results_dir(self.sample_dir)

    @property
    def visualization_dir(self) -> Path:
        """Directory for visualization deliverables."""
        return deliverable_visualization_dir(self.sample_dir)

    @property
    def brain_distribution_stats_xlsx(self) -> Path:
        """Whole-brain distribution statistics Excel workbook."""
        return deliverable_brain_distribution_stats_xlsx(self.sample_dir, self.signal_ch)

    @property
    def density_results_xlsx(self) -> Path:
        """Alias for :attr:`brain_distribution_stats_xlsx`."""
        return self.brain_distribution_stats_xlsx

    @property
    def heatmap_2d_dir(self) -> Path:
        """2D heatmap slice output directory."""
        return deliverable_heatmap_2d_dir(self.sample_dir, self.signal_ch)

    @property
    def heatmap_3d_png(self) -> Path:
        """Primary 3D heatmap deliverable PNG."""
        return deliverable_heatmap_3d_png(self.sample_dir, self.signal_ch)

    # ------------------------------------------------------------------
    # Tubule reconstruction outputs
    # ------------------------------------------------------------------

    @property
    def tubule_reconstruction_dir(self) -> Path:
        """Root output directory for tubule / vessel reconstruction."""
        return self.sample_dir / "tubule_reconstruction"

    @property
    def tubule_branch_csv(self) -> Path:
        return self.tubule_reconstruction_dir / "vessel_branch_metrics.csv"

    @property
    def tubule_summary_json(self) -> Path:
        return self.tubule_reconstruction_dir / "vessel_network_summary.json"

    @property
    def tubule_vertex_csv(self) -> Path:
        return self.tubule_reconstruction_dir / "skeleton_vertices.csv"

    @property
    def tubule_edge_csv(self) -> Path:
        return self.tubule_reconstruction_dir / "skeleton_edges.csv"

    @property
    def tubule_region_summary_csv(self) -> Path:
        return self.tubule_reconstruction_dir / "region_vessel_summary.csv"

    @property
    def tubule_region_summary_json(self) -> Path:
        return self.tubule_reconstruction_dir / "region_vessel_summary.json"

    @property
    def tubule_run_manifest(self) -> Path:
        return self.tubule_reconstruction_dir / "_run_manifest.json"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def as_dict(self) -> dict[str, str]:
        """Return all named paths as a ``{name: str}`` mapping (for manifests)."""
        props = [
            "signal_tiff_dir", "signal_tiff_preprocessed_dir", "reg_tiff_dir",
            "signal_zarr", "mask_zarr", "atlas_label_zarr",
            "atlas_label_hemisphere_zarr",
            "reg_downsample_dir", "reg_downsample_nii", "atlas_label_tiff_dir",
            "mask_tiff_dir", "results_dir", "visualization_dir",
            "brain_distribution_stats_xlsx", "density_results_xlsx",
            "heatmap_2d_dir", "heatmap_3d_png",
            "tubule_reconstruction_dir", "tubule_branch_csv", "tubule_summary_json",
            "tubule_vertex_csv", "tubule_edge_csv",
            "tubule_region_summary_csv", "tubule_region_summary_json",
            "tubule_run_manifest",
        ]
        return {p: str(getattr(self, p)) for p in props}
