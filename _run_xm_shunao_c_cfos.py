"""Unattended cFos pipeline for xm_shunao_c.

Extracts IMS channel 1 (488 cFos) and channel 2 (639 registration),
then runs main.py. Safe to re-run: TIFF export and main.py skip existing outputs.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

os.environ.setdefault("YIFU_DATA_DIR", r"S:\Yifu_data")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

REPO = Path(r"S:\Yifu")
sys.path.insert(0, str(REPO))
os.environ["PYTHONPATH"] = str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", "")

IMS_PATH = Path(r"E:\20260901_13_41_55_xm_shunao_c_Destripe_DONE\xm_shunao_c.ims")
DEFAULT_SAMPLE_DIR = Path(r"E:\20260901_13_41_55_xm_shunao_c_Destripe_DONE")
FALLBACK_SAMPLE_DIR = Path(r"S:\Arivis_Analysis\_active\xm_shunao_c")
CONFIG = REPO / "config" / "config_cfos_xm_shunao_c.json"
PYTHON = sys.executable
CFOS_CH = 1
REG_CH = 2
MAX_EXTRACT_RETRIES = 4


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


def count_tiffs(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in {".tif", ".tiff"})


def free_bytes(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def inspect_ims(ims_path: Path) -> dict[int, tuple[int, int, int, str]]:
    import h5py

    info: dict[int, tuple[int, int, int, str]] = {}
    with h5py.File(str(ims_path), "r", rdcc_nbytes=64 * 1024 * 1024, rdcc_nslots=10007) as handle:
        tp0 = handle["DataSet"]["ResolutionLevel 0"]["TimePoint 0"]
        for key in tp0.keys():
            if not str(key).startswith("Channel"):
                continue
            idx = int(str(key).split()[-1])
            data = tp0[key]["Data"]
            info[idx] = (int(data.shape[0]), int(data.shape[1]), int(data.shape[2]), str(data.dtype))
    return info


def extract_channel(ims_path: Path, sample_dir: Path, channel: int, expected_z: int) -> Path:
    from pipeline_modules.utils.ims_to_tiff import process_ims

    out_dir = sample_dir / f"ch{channel}"
    for attempt in range(1, MAX_EXTRACT_RETRIES + 1):
        have = count_tiffs(out_dir)
        log(f"Extract IMS channel {channel} attempt {attempt}/{MAX_EXTRACT_RETRIES}: {have}/{expected_z} TIFFs in {out_dir}")
        if have >= expected_z:
            log(f"Channel {channel} already complete")
            return out_dir
        try:
            process_ims(str(ims_path), str(sample_dir), channel)
        except Exception:
            log(f"process_ims raised for channel {channel}:\n{traceback.format_exc()}")
        have = count_tiffs(out_dir)
        log(f"After attempt {attempt}: {have}/{expected_z} TIFFs")
        if have >= expected_z:
            return out_dir
        time.sleep(min(30 * attempt, 120))
    raise RuntimeError(f"IMS channel {channel} incomplete: {count_tiffs(out_dir)}/{expected_z} in {out_dir}")


def ensure_ch0_from_reg(sample_dir: Path, expected_z: int) -> None:
    ch0 = sample_dir / "ch0"
    ch2 = sample_dir / "ch2"
    n0 = count_tiffs(ch0)
    n2 = count_tiffs(ch2)
    if n0 >= expected_z:
        log(f"ch0 already has {n0} TIFFs, leaving it in place")
        return
    if n2 < expected_z:
        raise RuntimeError(f"Cannot promote ch2 -> ch0: ch2 has {n2}/{expected_z}")
    if n0 > 0:
        backup = sample_dir / "ch0_not_reg_backup"
        if backup.exists():
            shutil.rmtree(backup)
        log(f"Renaming incomplete/wrong ch0 ({n0} files) -> {backup.name}")
        ch0.rename(backup)
    log("Renaming ch2 (639) -> ch0 for main.py registration")
    ch2.rename(ch0)


def run_main(sample_dir: Path) -> None:
    cmd = [PYTHON, str(REPO / "main.py"), "--config", str(CONFIG), "--sample_dir", str(sample_dir)]
    log("Running: " + " ".join(cmd))
    env = os.environ.copy()
    env["YIFU_DATA_DIR"] = os.environ["YIFU_DATA_DIR"]
    env["PYTHONPATH"] = str(REPO)
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(cmd, cwd=str(REPO), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"main.py exited with code {result.returncode}")
    log("main.py finished successfully")


def choose_sample_dir(bytes_needed: int) -> Path:
    e_free = free_bytes(DEFAULT_SAMPLE_DIR)
    log(f"E: free = {e_free / 1024**3:.1f} GB; need ~{bytes_needed / 1024**3:.1f} GB")
    if e_free >= bytes_needed:
        return DEFAULT_SAMPLE_DIR
    FALLBACK_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    s_free = free_bytes(FALLBACK_SAMPLE_DIR)
    log(f"E: not enough space. S: fallback free = {s_free / 1024**3:.1f} GB -> {FALLBACK_SAMPLE_DIR}")
    if s_free < bytes_needed:
        raise RuntimeError(
            f"Not enough disk space. Need ~{bytes_needed / 1024**3:.1f} GB, "
            f"E: {e_free / 1024**3:.1f} GB, S: {s_free / 1024**3:.1f} GB"
        )
    return FALLBACK_SAMPLE_DIR


def write_status(sample_dir: Path, text: str) -> None:
    (sample_dir / "cfos_pipeline.status.txt").write_text(
        f"{datetime.now().isoformat(timespec='seconds')}\n{text}\n",
        encoding="utf-8",
    )


def main() -> int:
    sample_dir = DEFAULT_SAMPLE_DIR
    log_path = DEFAULT_SAMPLE_DIR / "cfos_pipeline.log"
    log_file = open(log_path, "a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    log(f"=== xm_shunao_c cFos unattended start pid={os.getpid()} ===")
    log(f"Python={PYTHON}")
    log(f"Log={log_path}")
    try:
        if not IMS_PATH.exists():
            raise FileNotFoundError(IMS_PATH)
        if not CONFIG.exists():
            raise FileNotFoundError(CONFIG)
        model = Path(os.environ["YIFU_DATA_DIR"]) / "models" / "cfos" / "best_model.pt"
        if not model.exists():
            raise FileNotFoundError(model)
        log(f"IMS={IMS_PATH}")
        log(f"Model={model}")

        channels = inspect_ims(IMS_PATH)
        log(f"IMS channels: {channels}")
        if CFOS_CH not in channels:
            raise RuntimeError(f"cFos IMS channel {CFOS_CH} not found. Available={sorted(channels)}")
        if REG_CH not in channels:
            raise RuntimeError(f"Registration IMS channel {REG_CH} not found. Available={sorted(channels)}")

        z, y, x, dtype = channels[CFOS_CH]
        bytes_per_ch = z * y * x * (2 if "16" in dtype else 4)
        # Two TIFF stacks + Zarr/mask/registration headroom
        bytes_needed = int(bytes_per_ch * 3.2)
        sample_dir = choose_sample_dir(bytes_needed)
        if sample_dir != DEFAULT_SAMPLE_DIR:
            extra = sample_dir / "cfos_pipeline.log"
            log(f"Switching sample_dir to {sample_dir}; also appending {extra}")
        write_status(sample_dir, f"running pid={os.getpid()}\nsample_dir={sample_dir}")

        log(f"Step A: extract cFos IMS channel {CFOS_CH} -> ch{CFOS_CH}")
        extract_channel(IMS_PATH, sample_dir, CFOS_CH, z)
        log(f"Step B: extract registration IMS channel {REG_CH} -> ch{REG_CH}")
        extract_channel(IMS_PATH, sample_dir, REG_CH, channels[REG_CH][0])
        ensure_ch0_from_reg(sample_dir, channels[REG_CH][0])
        log(f"TIFF counts: ch0={count_tiffs(sample_dir / 'ch0')} ch1={count_tiffs(sample_dir / 'ch1')}")

        log("Step C: main.py (downsample, ANTs, Zarr, cfos_unet, density)")
        run_main(sample_dir)
        write_status(sample_dir, f"done pid={os.getpid()}\nsample_dir={sample_dir}")
        (sample_dir / "cfos_pipeline.done").write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
        log("=== ALL STEPS DONE ===")
        return 0
    except Exception:
        log("=== FAILED ===")
        log(traceback.format_exc())
        try:
            write_status(sample_dir, f"failed pid={os.getpid()}\n{traceback.format_exc()}")
        except Exception:
            pass
        return 1
    finally:
        log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
