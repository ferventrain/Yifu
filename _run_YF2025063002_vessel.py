"""Unattended vessel bilateral-statistics run for one YF2025063002 sample.

Streams IMS channel 1 (561 vessel) directly to Zarr, builds the registration
NIfTI from the channel-0 pyramid level 2 (no TIFF layer), then runs main.py
brain mode: ANTs registration -> hemisphere label -> threshold segmentation ->
per-region Left/Right vessel density statistics. Vessel skeleton
reconstruction is disabled in the config.

Registers the run as an external job in the pipeline harness GUI.

Usage: python _run_YF2025063002_vessel.py <SAMPLE>   (e.g. MPTP_1, PBS_2)

Safe to re-run: steps 1-3 use marker files; step 4 (main.py) skips existing
outputs.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

REPO = Path(r"S:\Yifu")
ANALYSIS_ROOT = Path(r"S:\Arivis_Analysis")
os.environ["YIFU_DATA_DIR"] = r"S:\Yifu_data"
os.environ["YIFU_ANALYSIS_ROOT"] = str(ANALYSIS_ROOT)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, str(REPO))
os.environ["PYTHONPATH"] = str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", "")

# dataset id -> (analysis batch dir under the analysis root, NAS root of the raw ims folders)
DATASETS = {
    "YF2025063002": {
        "batch_dir": Path(r"S:\Arivis_Analysis\_active\YF2025063002"),
        "nas_root": Path(r"//192.168.110.4/Yifu/YF2025063002"),
    },
    "YF2026051202": {
        "batch_dir": Path(r"S:\Arivis_Analysis\_active\YF2026051202"),
        "nas_root": Path(r"//192.168.110.17/MegaSpim/YF2026051202"),
    },
}
DEFAULT_DATASET = "YF2025063002"

PYTHON = sys.executable
MAX_ATTEMPTS = 4
PYRAMID_LEVEL_STRIDE = 8  # Imaris pyramid levels are powers of two

STEPS = [
    (1, "IMS ch1 (vessel) -> Zarr", "ims_to_zarr"),
    (2, "IMS ch0 pyramid L2 -> Zarr", "ims_to_zarr_coarse"),
    (3, "coarse ch0 Zarr -> registration NIfTI (25 um)", "reg_nii"),
    (4, "main.py: registration -> hemisphere -> threshold -> L/R stats", "main_pipeline"),
    (5, "verify outputs", "verify"),
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


def marker_path(target: Path) -> Path:
    return target.with_name(target.name + ".done")


def run_step(cmd: list[str], desc: str, log_path: Path) -> None:
    env = os.environ.copy()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        log(f"{desc} attempt {attempt}/{MAX_ATTEMPTS}: {' '.join(str(c) for c in cmd)}")
        with open(log_path, "a", encoding="utf-8") as log_file:
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


def find_sample(spec: str) -> tuple[Path, Path]:
    """Resolve 'YF<id>_<sample>' (or bare '<sample>' = default dataset) to sample dir + ims."""
    match = re.match(r"(YF\d+)_(.+)$", spec)
    if match:
        dataset, sample = match.group(1), match.group(2)
    else:
        dataset, sample = DEFAULT_DATASET, spec
    dataset_cfg = DATASETS.get(dataset)
    if dataset_cfg is None:
        raise KeyError(f"unknown dataset {dataset}; known: {sorted(DATASETS)}")
    sample_dir = dataset_cfg["batch_dir"] / f"{dataset}_{sample}"
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"sample dir not found: {sample_dir}")
    candidates = [
        root for root in glob.glob(str(dataset_cfg["nas_root"] / "*_Destripe_DONE"))
        if re.search(rf"_{re.escape(sample)}(?=_Destripe_DONE$)", root)
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected 1 NAS folder for {sample}, found {candidates}")
    ims_files = glob.glob(candidates[0] + "/*.ims")
    if len(ims_files) != 1:
        raise FileNotFoundError(f"expected 1 ims file in {candidates[0]}, found {ims_files}")
    return sample_dir, Path(ims_files[0])


def step1_signal_zarr(ims_path: Path, signal_zarr: Path, signal_ch: str, log_path: Path) -> None:
    if marker_path(signal_zarr).exists():
        log("Step 1 already complete (marker present), skipping")
        return
    run_step(
        [
            PYTHON, "-m", "pipeline_modules.preprocessing.ims_to_zarr",
            "--input", ims_path,
            "--output", signal_zarr,
            "--channels", signal_ch,
            "--chunk_size", "32,256,256",
            "--gzip_level", "1",
        ],
        "IMS->Zarr signal",
        log_path,
    )
    marker_path(signal_zarr).write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")


def write_original_shape_json(sample_dir: Path, signal_zarr: Path) -> None:
    """ANTs_registration upsamples the warped label to this full-res grid."""
    from pipeline_modules.utils.zarr_io import open_zarr_array

    target = sample_dir / "original_shape.json"
    shape = [int(v) for v in open_zarr_array(signal_zarr).shape]
    target.write_text(json.dumps({"original_shape": shape}), encoding="utf-8")
    log(f"original_shape.json: {shape}")


def step2_coarse_reg_zarr(ims_path: Path, coarse_zarr: Path, reg_ch: str, log_path: Path) -> None:
    if marker_path(coarse_zarr).exists():
        log("Step 2 already complete (marker present), skipping")
        return
    run_step(
        [
            PYTHON, "-m", "pipeline_modules.preprocessing.ims_to_zarr",
            "--input", ims_path,
            "--output", coarse_zarr,
            "--channels", reg_ch,
            "--resolution_level", "2",
            "--chunk_size", "32,256,256",
        ],
        "IMS->Zarr registration L2",
        log_path,
    )
    marker_path(coarse_zarr).write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")


def step3_registration_nii(
    sample_dir: Path,
    config: dict,
    coarse_zarr: Path,
    reg_nii: Path,
    log_path: Path,
    keep_coarse: bool,
) -> None:
    if marker_path(reg_nii).exists() or reg_nii.exists():
        # Marker missing but output present = the driver was killed between
        # module completion and marker write; the module refuses to overwrite,
        # so adopt the finished volume instead of rerunning.
        log("Step 3 output already present, adopting")
    else:
        res_xyz = [float(v) for v in config["input"]["resolution_xyz"]]
        level2_xyz = [v * PYRAMID_LEVEL_STRIDE for v in res_xyz]
        run_step(
            [
                PYTHON, "-m", "pipeline_modules.preprocessing.zarr_to_registration_nii",
                "--input_zarr", coarse_zarr,
                "--output_nii", reg_nii,
                "--input_resolution_xyz", ",".join(f"{v:.4f}" for v in level2_xyz),
                "--target_resolution_xyz", ",".join(
                    f"{float(v):.1f}" for v in config["preprocessing"]["downsample"]["target_resolution_xyz"]
                ),
            ],
            "Zarr->NIfTI",
            log_path,
        )
    marker_path(reg_nii).write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
    if not keep_coarse:
        shutil.rmtree(coarse_zarr, ignore_errors=True)
        marker_path(coarse_zarr).unlink(missing_ok=True)
        log("Removed intermediate coarse ch0 Zarr")


def step4_main_pipeline(sample_dir: Path, config_path: Path, log_path: Path) -> None:
    run_step(
        [
            PYTHON, "main.py",
            "--config", config_path,
            "--sample_dir", sample_dir,
        ],
        "main.py",
        log_path,
    )


def step5_verify(sample_dir: Path, config: dict) -> None:
    from pipeline_modules.utils.zarr_io import open_zarr_array

    signal_ch = config["input"]["channels"]["signal"]
    mask_zarr = sample_dir / f"ch{signal_ch}_mask.zarr"
    hemisphere_zarr = sample_dir / "atlas_label_hemisphere.zarr"
    label_zarr = sample_dir / "upsampled_atlas_label.zarr"
    signal_zarr = sample_dir / f"ch{signal_ch}.zarr"
    results_dir = sample_dir / "results"
    excel_files = sorted(results_dir.glob("*.xlsx"))

    for path in (signal_zarr, mask_zarr, label_zarr, hemisphere_zarr):
        assert path.exists(), f"missing output: {path}"
    assert excel_files, f"missing results xlsx under {results_dir}"

    mask_shape = tuple(int(v) for v in open_zarr_array(mask_zarr).shape)
    signal_shape = tuple(int(v) for v in open_zarr_array(signal_zarr).shape)
    assert mask_shape == signal_shape, (mask_shape, signal_shape)

    import openpyxl

    left_right_found = False
    for excel in excel_files:
        workbook = openpyxl.load_workbook(excel, read_only=True)
        for sheet in workbook.sheetnames:
            for row in workbook[sheet].iter_rows(min_row=1, max_row=1, values_only=True):
                if "Left Signal Voxels" in (row or ()):
                    left_right_found = True
        workbook.close()
        if left_right_found:
            break
    assert left_right_found, f"no Left/Right columns found in {excel_files}"

    report = {
        "sample": sample_dir.name,
        "shape_zyx": list(signal_shape),
        "mask_zarr": str(mask_zarr),
        "hemisphere_zarr": str(hemisphere_zarr),
        "results_xlsx": [str(p) for p in excel_files],
        "left_right_columns": True,
    }
    log("VERIFY OK: " + json.dumps(report, ensure_ascii=False))


def attach_to_harness(sample_dir: Path, dataset: str, config_path: Path, log_path: Path, status_path: Path) -> None:
    try:
        from pipeline_modules.harness.queue import ActiveStore

        store = ActiveStore()
        record = store.attach_external(
            sample_dir,
            title=f"{dataset} vessel bilateral: {sample_dir.name.replace(dataset + '_', '')}",
            pid=os.getpid(),
            log_path=log_path,
            status_path=status_path,
            config_path=config_path,
            command="ims_to_zarr -> ims L2 zarr -> zarr_to_registration_nii -> main.py(threshold+hemisphere)",
            step_total=len(STEPS),
        )
        log(f"Attached to harness GUI as job {record.get('id')} (analysis_root={store.root})")
    except Exception:
        log("Failed to attach to harness GUI (continuing anyway):\n" + traceback.format_exc())


def main() -> int:
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    keep_coarse = "--keep_level2_zarr" in sys.argv
    if len(args) != 1:
        print(__doc__)
        return 2
    sample = args[0]
    dataset_match = re.match(r"(YF\d{8})_", sample)
    dataset = dataset_match.group(1) if dataset_match else DEFAULT_DATASET
    try:
        sample_dir, ims_path = find_sample(sample)
    except Exception:
        # No per-sample log yet (it lives in the sample dir); the traceback
        # goes to the orchestrator's queue log instead of being lost.
        traceback.print_exc()
        return 1
    config_path = sample_dir / "config.json"
    status_path = sample_dir / "vessel_pipeline.status.txt"
    log_path = sample_dir / "vessel_pipeline.log"

    sample_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")
    # Under the scheduled task the inherited stdout/stderr handles are invalid;
    # python.exe then exits with code 120 (stdout flush failure at shutdown)
    # even after a fully successful run. Point fd 1/2 at the log so this
    # process and every child it spawns carry valid handles.
    os.dup2(log_file.fileno(), 1)
    os.dup2(log_file.fileno(), 2)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    def write_status(first_line: str) -> None:
        status_path.write_text(f"{first_line}\n{datetime.now().isoformat(timespec='seconds')}\n", encoding="utf-8")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    channels = config["input"]["channels"]
    signal_ch = str(channels["signal"])
    reg_ch = str(channels["registration"])
    signal_zarr = sample_dir / f"ch{signal_ch}.zarr"
    coarse_zarr = sample_dir / f"ch{reg_ch}_L2.zarr"
    reg_nii = sample_dir / f"ch{reg_ch}_downsample" / "volume.nii.gz"

    log(f"=== {dataset} {sample} vessel bilateral start pid={os.getpid()} ===")
    log(f"Python={PYTHON} IMS={ims_path}")
    free_gb = shutil.disk_usage(sample_dir).free / 1024**3
    log(f"free on S: {free_gb:.0f} GB")
    started = datetime.now()
    try:
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        if not ims_path.exists():
            raise FileNotFoundError(ims_path)
        attach_to_harness(sample_dir, dataset, config_path, log_path, status_path)

        signal_step_title = f"IMS ch{signal_ch} (vessel) -> Zarr"
        for step_number, title, _tag in STEPS[:-1]:
            write_status(f"RUNNING Step {step_number}: {title}")
            if step_number == 1:
                step1_signal_zarr(ims_path, signal_zarr, signal_ch, log_path)
                write_status(f"RUNNING Step 1: {signal_step_title} (writing original_shape.json)")
                write_original_shape_json(sample_dir, signal_zarr)
            elif step_number == 2:
                step2_coarse_reg_zarr(ims_path, coarse_zarr, reg_ch, log_path)
            elif step_number == 3:
                step3_registration_nii(sample_dir, config, coarse_zarr, reg_nii, log_path, keep_coarse)
            elif step_number == 4:
                step4_main_pipeline(sample_dir, config_path, log_path)

        write_status("RUNNING Step 5: verify outputs")
        step5_verify(sample_dir, config)

        write_status("ALL DONE")
        log(f"=== ALL STEPS DONE (elapsed {(datetime.now() - started).total_seconds() / 3600:.2f} h) ===")
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
