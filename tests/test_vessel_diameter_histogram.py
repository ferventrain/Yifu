from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline_modules.visualization.vessel_diameter_histogram import (
    load_branch_diameters,
    plot_vessel_diameter_histogram,
    summarize_diameters,
)


class TestLoadBranchDiameters:
    def test_computes_diameter_and_filters_invalid_values(self, tmp_path):
        csv_path = tmp_path / "vessel_branch_metrics.csv"
        pd.DataFrame(
            {
                "mean_radius_um": [1.0, 2.5, np.nan, np.inf, -1.0, 0.0, 3.0],
            }
        ).to_csv(csv_path, index=False)

        diameters = load_branch_diameters(csv_path)

        assert np.allclose(diameters, np.array([2.0, 5.0, 6.0]))

    def test_requires_mean_radius_column(self, tmp_path):
        csv_path = tmp_path / "vessel_branch_metrics.csv"
        pd.DataFrame({"radius_um": [1.0, 2.0]}).to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="mean_radius_um"):
            load_branch_diameters(csv_path)

    def test_rejects_empty_valid_data(self, tmp_path):
        csv_path = tmp_path / "vessel_branch_metrics.csv"
        pd.DataFrame({"mean_radius_um": [np.nan, np.inf, -2.0, 0.0]}).to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="No valid vessel diameters"):
            load_branch_diameters(csv_path)


class TestPlotVesselDiameterHistogram:
    def test_writes_output_image(self, tmp_path):
        output_path = tmp_path / "histogram.png"

        saved = plot_vessel_diameter_histogram(
            np.array([2.0, 3.5, 4.0, 5.5, 8.0], dtype=np.float64),
            output_path=output_path,
            bins=4,
        )

        assert saved == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_summary_values(self):
        summary = summarize_diameters(np.array([2.0, 4.0, 6.0], dtype=np.float64))

        assert summary["n"] == 3
        assert summary["mean"] == pytest.approx(4.0)
        assert summary["median"] == pytest.approx(4.0)
        assert summary["min"] == pytest.approx(2.0)
        assert summary["max"] == pytest.approx(6.0)
