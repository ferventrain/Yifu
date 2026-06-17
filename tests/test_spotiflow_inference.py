from __future__ import annotations

import json
import sys
from types import ModuleType

import numpy as np
import pandas as pd
import zarr

from pipeline_modules.segmentation.spotiflow_inference import run_spotiflow_inference


class _FakeSpotiflow:
    @classmethod
    def from_folder(cls, *args, **kwargs):
        return cls()

    def predict(self, img, **kwargs):
        return np.asarray([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32), object()


def test_run_spotiflow_inference_counts_regions(tmp_path, monkeypatch):
    model_module = ModuleType("spotiflow.model")
    model_module.Spotiflow = _FakeSpotiflow
    package_module = ModuleType("spotiflow")
    package_module.model = model_module
    monkeypatch.setitem(sys.modules, "spotiflow", package_module)
    monkeypatch.setitem(sys.modules, "spotiflow.model", model_module)

    input_zarr = tmp_path / "signal.zarr"
    root = zarr.group(store=zarr.DirectoryStore(str(input_zarr)), overwrite=True)
    root.create_dataset("0", data=np.ones((4, 4, 4), dtype=np.uint16), chunks=(4, 4, 4))

    label_zarr = tmp_path / "labels.zarr"
    label_root = zarr.group(store=zarr.DirectoryStore(str(label_zarr)), overwrite=True)
    labels = np.zeros((4, 4, 4), dtype=np.uint16)
    labels[1, 1, 1] = 7
    labels[2, 2, 2] = 8
    label_root.create_dataset("0", data=labels, chunks=(4, 4, 4))

    cfg = tmp_path / "regions.csv"
    pd.DataFrame(
        [
            {"id": 7, "name": "Region A", "acronym": "RA"},
            {"id": 8, "name": "Region B", "acronym": "RB"},
        ]
    ).to_csv(cfg, index=False)

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text("is_3d: true\n", encoding="utf-8")
    (model_dir / "best.pt").write_bytes(b"fake")

    result = run_spotiflow_inference(
        input_zarr=input_zarr,
        model_dir=model_dir,
        label_zarr=label_zarr,
        cfg=cfg,
        output_csv=tmp_path / "points.csv",
        region_counts_csv=tmp_path / "counts.csv",
        summary_json=tmp_path / "summary.json",
        tile_overlap=0,
        device="cpu",
    )

    assert result["total_signal_count"] == 2
    counts = pd.read_csv(tmp_path / "counts.csv").set_index("region_id")
    assert counts.loc[7, "signal_count"] == 1
    assert counts.loc[8, "signal_count"] == 1
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["total_signal_count"] == 2
