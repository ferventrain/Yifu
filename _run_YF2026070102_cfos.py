"""Unattended cFos run for YF2026070102 without the TIFF layer.

Streams IMS channel 1 (488 cFos) directly to Zarr, runs cfos_unet
segmentation, then writes a local .ims where channel 1 is replaced by the
segmentation mask (channel 0 is streamed from the NAS source). Registers the
run as an external job in the pipeline harness GUI.

Safe to re-run: step 1 resumes at chunk level; steps 2/3 use marker files.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

REPO = Path(r"S:\Yifu")
SAMPLE_DIR = Path(r"S:\Arivis_Analysis\YF2026070102")
ANALYSIS_ROOT = Path(r"S:\Arivis_Analysis")
os.environ["YIFU_DATA_DIR"] = r"S:\Yifu_data"
os.environ["YIFU_ANALYSIS_ROOT"] = str(ANALYSIS_ROOT)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, str(REPO))
os.environ["PYTHONPATH"] = str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", "")

IMS_PATH = Path(
    r"\\192.168.110.17\MegaSpim\YF2026070102"
    r"\20260723_09_36_13_YF2026070102_AYD_nao_SICM_Destripe_DONE"
    r"\YF2026070102_AYD_nao_SICM.ims"
)
CONFIG = SAMPLE_DIR / "config.json"
SIGNAL_ZARR = SAMPLE_DIR / "ch1.zarr"
MASK_ZARR = SAMPLE_DIR / "ch1_mask.zarr"
MASKED_SIGNAL_ZARR = SAMPLE_DIR / "ch1_masked.zarr"
MASKED_IMS = SAMPLE_DIR / "YF2026070102_AYD_nao_SICM_cfos_masked.ims"
STATUS_PATH = SAMPLE_DIR / "cfos_pipeline.status.txt"
LOG_PATH = SAMPLE_DIR / "cfos_pipeline.log"
CHECKPOINT = Path(os.environ["YIFU_DATA_DIR"]) / "models" / "cfos" / "best_model.pt"

PYTHON = sys.executable
CFOS_CH = 1
MAX_ATTEMPTS = 4

STEPS = [
    (1, "IMS ch1 -> Zarr", "ims_to_zarr"),
    (2, "cfos_unet segmentation", "segmentation"),
    (3, "masked Zarr -> IMS (replace ch1)", "masked_ims"),
    (4, "verify outputs", "verify"),
]


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def write_status(first_line: str) -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(f"{first_line}\n{STARTED_AT_ISO}\n", encoding="utf-8")


STARTED_AT_ISO = ""


def marker_path(target: Path) -> Path:
    return target.with_name(target.name + ".done")


def run_step(cmd: list[str], desc: str) -> None:
    env = os.environ.copy()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        log(f"{desc} attempt {attempt}/{MAX_ATTEMPTS}: {' '.join(str(c) for c in cmd)}")
        # Subprocesses bypass the Python-level Tee; route their output into the
        # log file explicitly so the harness GUI log view shows step progress.
        with open(LOG_PATH, "a", encoding="utf-8") as log_file:
            result = subprocess.run(
                [str(part) for part in cmd],
                cwd=str(REPO),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        if result.returncode == 0:
            return
        log(f"{desc} failed with code {result.returncode}")
        if attempt < MAX_ATTEMPTS:
            time.sleep(min(60 * attempt, 180))
    raise RuntimeError(f"{desc} failed after {MAX_ATTEMPTS} attempts")


def step1_ims_to_zarr() -> None:
    if marker_path(SIGNAL_ZARR).exists():
        log("Step 1 already complete (marker present), skipping")
        return
    write_status("RUNNING Step 1: IMS ch1 -> Zarr")
    run_step(
        [
            PYTHON, "-m", "pipeline_modules.preprocessing.ims_to_zarr",
            "--input", IMS_PATH,
            "--output", SIGNAL_ZARR,
            "--channels", str(CFOS_CH),
            "--chunk_size", "32,256,256",
            "--gzip_level", "1",
        ],
        "IMS->Zarr",
    )
    marker_path(SIGNAL_ZARR).write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")


def step2_segmentation() -> None:
    if marker_path(MASK_ZARR).exists():
        log("Step 2 already complete (marker present), skipping")
        return
    write_status("RUNNING Step 2: cfos_unet segmentation")
    run_step(
        [
            PYTHON, "-m", "pipeline_modules.segmentation.cfos_unet_inference",
            "--input_zarr", SIGNAL_ZARR,
            "--output_zarr", MASK_ZARR,
            "--checkpoint_path", CHECKPOINT,
            "--dataset_name", "0",
            "--patch_size", "256,256,256",
            "--overlap", "0.25",
            "--batch_size", "16",
            "--chunk_size", "256,256,256",
            "--probability_threshold", "0.5",
            "--skip_below_threshold", "100",
            "--output_mode", "binary",
            "--output_dtype", "uint8",
            "--device", "auto",
        ],
        "cfos_unet",
    )
    marker_path(MASK_ZARR).write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")


def step2b_masked_signal() -> None:
    """Filter the mask (per-chunk CC: aspect>3 or >200 voxels dropped) and
    produce the masked signal zarr (signal values kept, background 0) that
    step 3 writes into the ims as channel 1."""
    if marker_path(MASKED_SIGNAL_ZARR).exists():
        log("Step 2b already complete (marker present), skipping")
        return
    write_status("RUNNING Step 2b: filter mask + masked signal zarr")
    run_step(
        [
            PYTHON, "-m", "pipeline_modules.segmentation.apply_mask_to_signal",
            "--signal_zarr", SIGNAL_ZARR,
            "--mask_zarr", MASK_ZARR,
            "--masked_signal_zarr", MASKED_SIGNAL_ZARR,
            "--max_aspect_ratio", "3.0",
            "--max_voxels", "200",
        ],
        "apply_mask",
    )
    marker_path(MASKED_SIGNAL_ZARR).write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")


def step3_masked_ims() -> None:
    if marker_path(MASKED_IMS).exists():
        log("Step 3 already complete (marker present), skipping")
        return
    write_status("RUNNING Step 3: masked Zarr -> IMS (replace ch1)")
    run_step(
        [
            PYTHON, "-m", "pipeline_modules.utils.masked_zarr_to_ims",
            "--input", MASKED_SIGNAL_ZARR,
            "--output", MASKED_IMS,
            "--from_source_ims", IMS_PATH,
            "--channel", str(CFOS_CH),
            "--pyramid", "1",
            "--chunk_size", "64,256,256",
            "--gzip_level", "1",
            "--z_block", "32",
            "--overwrite",
        ],
        "Zarr->IMS",
    )
    marker_path(MASKED_IMS).write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")


def step4_verify() -> None:
    write_status("RUNNING Step 4: verify outputs")
    import h5py
    import numpy as np

    from pipeline_modules.utils.zarr_io import open_zarr_array

    signal = open_zarr_array(SIGNAL_ZARR)
    mask = open_zarr_array(MASK_ZARR)
    assert tuple(mask.shape) == tuple(signal.shape), (mask.shape, signal.shape)

    with h5py.File(MASKED_IMS, "r") as handle:
        base = "DataSet/ResolutionLevel 0/TimePoint 0"
        ch0 = handle[f"{base}/Channel 0/Data"]
        ch1 = handle[f"{base}/Channel 1/Data"]
        assert tuple(ch0.shape) == tuple(signal.shape), ch0.shape
        assert tuple(ch1.shape) == tuple(signal.shape), ch1.shape
        assert ch1.compression == "gzip" and ch1.chunks is not None
        chunk_bytes = int(np.prod(ch1.chunks)) * ch1.dtype.itemsize
        assert chunk_bytes <= 4 * 1024 * 1024, chunk_bytes
        assert "ResolutionLevel 1" in handle["DataSet"]

        for z in (0, signal.shape[0] // 2, signal.shape[0] - 1):
            want = np.asarray(mask[z : z + 1, : 512, : 512])
            got = ch1[z : z + 1, : 512, : 512]
            assert np.array_equal(np.asarray(got), want.astype(got.dtype)), f"mask mismatch at z={z}"

        with h5py.File(str(IMS_PATH), "r") as source:
            src_ch0 = source["DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data"]
            for z in (0, signal.shape[0] // 2):
                want = np.asarray(src_ch0[z : z + 1, : 512, : 512])
                got = ch0[z : z + 1, : 512, : 512]
                assert np.array_equal(np.asarray(got), want), f"ch0 mismatch at z={z}"

        info = handle["DataSetInfo"]
        report = {
            "masked_ims": str(MASKED_IMS),
            "shape_zyx": [int(v) for v in ch1.shape],
            "channels": ["source copy (640)", "cfos_unet mask (488)"],
            "level1_present": "ResolutionLevel 1" in handle["DataSet"],
            "chunk_zyx": [int(v) for v in ch1.chunks],
            "gzip": True,
        }
    log("VERIFY OK: " + json.dumps(report, ensure_ascii=False))


def attach_to_harness() -> None:
    try:
        from pipeline_modules.harness.queue import ActiveStore

        store = ActiveStore()
        record = store.attach_external(
            SAMPLE_DIR,
            title="YF2026070102 cFos: ims->zarr->unet->masked ims",
            pid=os.getpid(),
            log_path=LOG_PATH,
            status_path=STATUS_PATH,
            config_path=CONFIG,
            command="ims_to_zarr -> cfos_unet -> masked_zarr_to_ims",
            step_total=len(STEPS),
        )
        log(f"Attached to harness GUI as job {record.get('id')} (analysis_root={store.root})")
    except Exception:
        log("Failed to attach to harness GUI (continuing anyway):\n" + traceback.format_exc())


def main() -> int:
    global STARTED_AT_ISO
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(LOG_PATH, "a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    STARTED_AT_ISO = datetime.now().isoformat(timespec="seconds")
    log(f"=== YF2026070102 cFos (no-TIFF flow) start pid={os.getpid()} ===")
    log(f"Python={PYTHON} IMS={IMS_PATH}")
    free_gb = shutil.disk_usage(SAMPLE_DIR).free / 1024**3
    log(f"free on S: {free_gb:.0f} GB")
    try:
        if not IMS_PATH.exists():
            raise FileNotFoundError(IMS_PATH)
        if not CHECKPOINT.exists():
            raise FileNotFoundError(CHECKPOINT)
        if not CONFIG.exists():
            raise FileNotFoundError(CONFIG)
        attach_to_harness()
        write_status("RUNNING Step 1: IMS ch1 -> Zarr")

        step1_ims_to_zarr()
        step2_segmentation()
        step2b_masked_signal()
        step3_masked_ims()
        step4_verify()

        write_status("ALL DONE")
        log("=== ALL STEPS DONE ===")
        return 0
    except Exception:
        log("=== FAILED ===")
        log(traceback.format_exc())
        try:
            write_status(f"FAILED: {traceback.format_exc(limit=1).strip()[:200]}")
        except Exception:
            pass
        return 1
    finally:
        log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
