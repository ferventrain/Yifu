from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline_modules.qc.batch_image_qc import load_or_create_state, run_batch_image_qc
from pipeline_modules.qc.image_qc import ImageQcConfig


def _write_synthetic_ims(path: Path, *, shape: tuple[int, int, int] = (128, 128, 128)) -> None:
    import h5py

    volume = np.linspace(0, 1000, num=int(np.prod(shape)), dtype=np.uint16).reshape(shape)
    with h5py.File(path, "w") as handle:
        tp = handle.create_group("DataSet").create_group("ResolutionLevel 0").create_group("TimePoint 0")
        ch = tp.create_group("Channel 0")
        ch.create_dataset("Data", data=volume, chunks=(32, 64, 64), compression="gzip")
        tp.create_group("Channel 1")
        tp["Channel 1"].create_dataset("Data", data=volume + 100, chunks=(32, 64, 64), compression="gzip")


def test_batch_qc_resumes_completed_jobs(tmp_path: Path, monkeypatch):
    scan_root = tmp_path / "nas"
    output_root = tmp_path / "qc"
    sample_a = scan_root / "mouse_a"
    sample_b = scan_root / "mouse_b"
    sample_a.mkdir(parents=True)
    sample_b.mkdir(parents=True)
    _write_synthetic_ims(sample_a / "a.ims")
    _write_synthetic_ims(sample_b / "b.ims")

    cfg = ImageQcConfig(
        nas_qc=False,
        ims_resolution_level=0,
        max_slices=2,
        ims_histogram_z_chunks=2,
        show_progress=False,
        grading_enabled=False,
    )

    first = run_batch_image_qc(
        scan_root=scan_root,
        output_root=output_root,
        channels=[0],
        config=cfg,
    )
    assert first["completed"] == 2
    assert first["failed"] == 0

    second = run_batch_image_qc(
        scan_root=scan_root,
        output_root=output_root,
        channels=[0],
        config=cfg,
    )
    assert second["skipped"] == 2
    assert second["completed"] == 0

    state = load_or_create_state(
        scan_root=scan_root,
        output_root=output_root,
        channels=[0],
        recursive=True,
    )
    assert len(state.jobs) == 2
    assert all(job.status == "completed" for job in state.jobs)
