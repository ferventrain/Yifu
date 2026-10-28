"""Deep-learning whole-brain mask for the registration channel (fusion of 3 planes).

Runs the Allen Institute ``aind-brain-segmentation`` 2D SwinUNETR tissue model
(Neuratt, SmartSPIM 639-channel training; weights bundled under
``pipeline_modules/segmentation/models/``) on three orthogonal planes of the
coarse registration Zarr ``ch{channel}_downsampled.zarr``, averages the three
probability volumes and thresholds them into a whole-brain mask on the SAME
grid as the input Zarr. ``zarr_to_registration_nii`` then maps the mask onto
the registration grid (``--brain_mask``) and uses it exactly where the Otsu
halo mask used to go: zeroing the registration NIfTI, ``brain_mask.nii.gz``
(N4 + orientation preflight consumers).

Why fusion of three planes: the model is plane-agnostic (trained on mixed
coronal/sagittal/axial slices), and a probability crack that opens the mask
in one plane is usually closed in another. On YF2026082501_AYD_nao_d21 this
rescued slices where the single-plane mask retained only a rim of the brain
(qc/brain_mask_dl.png vs the trial comparisons in
_active/aind-brain-segmentation/trial_out/).

Memory-safe by design: slice batches go to the GPU one at a time and each
plane's probability volume is cached as a uint8 Zarr (prob*255) under
``ch{channel}_brain_mask_probs/``; the caches are deleted after fusion unless
``--keep_probs`` is set. Runtime is ~40-60 min per sample on a 24 GB GPU
(~896+2688+1920 slices at 1024x1024).

The model was trained at 14.4/14.4/16 um; the coarse Zarr (7.2/7.2/8.0 um on
MegaSpim level-2 exports) feeds slices at half that sampling, which validated
without resampling.
"""
from __future__ import annotations
import json

import argparse
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "smartspim_tissue_segmentation.safetensors"
IMG_SIZE = 1024
BATCH_Z = 4
BATCH_YX = 8
PROB_SCALE = 255.0


class _RamVolume:
    """内存卷垫片：兼容三轴 slab 赋值语义（zarr3 v2 异步写在本机不可靠，先入内存）。"""

    def __init__(self, shape):
        self.buf = np.zeros(shape, dtype=np.uint8)

    def __setitem__(self, key, value):
        self.buf[key] = np.asarray(value, dtype=np.uint8)


def _dump_volume_v2(path, buf, chunk_yx=(512, 512), log=print):
    """手工 v2 落盘（zstd、全零 chunk 跳过、边缘补零）——zarr 写路径的可靠替代。"""
    import json
    import shutil
    import time

    from numcodecs import Blosc

    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    (path / ".zarray").write_text(json.dumps({
        "shape": [int(v) for v in buf.shape], "chunks": [1, chunk_yx[0], chunk_yx[1]],
        "dtype": buf.dtype.str, "fill_value": 0, "order": "C", "filters": None,
        "dimension_separator": ".", "zarr_format": 2,
        "compressor": {"id": "blosc", "cname": "zstd", "clevel": 5, "shuffle": 1, "blocksize": 0},
    }), encoding="utf-8")
    (path / ".zattrs").write_text(json.dumps({"multiscales": [{"version": "0.4", "datasets": [
        {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0]}]}]}]}), encoding="utf-8")
    arr_dir = path
    codec = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
    t0 = time.time()
    count = 0
    for z in range(buf.shape[0]):
        plane = np.ascontiguousarray(buf[z])
        for yo in range(0, buf.shape[1], chunk_yx[0]):
            for xo in range(0, buf.shape[2], chunk_yx[1]):
                sub = plane[yo:yo + chunk_yx[0], xo:xo + chunk_yx[1]]
                if not sub.any():
                    continue
                if sub.shape != (chunk_yx[0], chunk_yx[1]):
                    pad = np.zeros((chunk_yx[0], chunk_yx[1]), dtype=sub.dtype)
                    pad[: sub.shape[0], : sub.shape[1]] = sub
                    sub = pad
                if not sub.flags["C_CONTIGUOUS"]:
                    sub = np.ascontiguousarray(sub)
                (arr_dir / f"{z}.{yo // chunk_yx[0]}.{xo // chunk_yx[1]}").write_bytes(codec.encode(sub))
                count += 1
    log(f"  dumped {buf.shape} -> {count} chunks in {time.time() - t0:.0f}s")


def _load_volume_v2(path):
    """读回手工 v2 卷（缺 chunk 补零）。"""
    import json

    import numpy as np

    path = Path(path)
    meta = json.loads((path / "0" / ".zarray").read_text(encoding="utf-8"))
    shape = tuple(meta["shape"])
    out = np.zeros(shape, dtype=np.dtype(meta["dtype"]))
    from numcodecs import Blosc

    codec = Blosc.from_config(meta["compressor"]) if meta.get("compressor") else None
    arr_dir = path / "0"
    for f in arr_dir.iterdir():
        if f.name.startswith("."):
            continue
        idx = tuple(int(v) for v in f.name.split("."))
        raw = codec.decode(f.read_bytes())
        zz = slice(idx[0], min(idx[0] + meta["chunks"][0], shape[0]))
        yy = slice(idx[1], min(idx[1] + meta["chunks"][1], shape[1]))
        xx = slice(idx[2], min(idx[2] + meta["chunks"][2], shape[2]))
        block = np.frombuffer(raw, dtype=np.dtype(meta["dtype"])).reshape(
            min(meta["chunks"][0], shape[0] - idx[0]),
            min(meta["chunks"][1], shape[1] - idx[1]),
            min(meta["chunks"][2], shape[2] - idx[2]))
        out[zz, yy, xx] = block
    return out


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def build_model(model_path: str | Path, device: str = "cuda"):
    """Neuratt inference skeleton: monai SwinUNETR + checkpoint weights.

    Matches the upstream ``Neuratt`` attribute layout (``model.*`` state-dict
    prefix). Inference behaviour equals upstream ``Neuratt.predict``:
    per-slice min-max normalisation, forward, sigmoid.
    """
    import torch
    from monai.networks.nets import SwinUNETR
    from safetensors.torch import load_file

    class _NeurattInference(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            kwargs = dict(
                in_channels=1,
                out_channels=1,
                feature_size=12,
                spatial_dims=2,
            )
            try:
                # monai >=1.4 移除了 img_size（全卷积，本就未使用）
                self.model = SwinUNETR(img_size=(IMG_SIZE, IMG_SIZE), **kwargs)
            except TypeError:
                self.model = SwinUNETR(**kwargs)

        def forward(self, x):
            return self.model(x)

    model = _NeurattInference()
    state_dict = load_file(str(model_path))
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model


def infer_probabilities(model, slices_np: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Raw probabilities (0..1) for a batch of (B, H, W) slices at 1024^2."""
    import torch

    batch = torch.from_numpy(slices_np.astype(np.float32))[:, None].to(device)
    batch = torch.nn.functional.interpolate(
        batch, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
    )
    normed = torch.empty_like(batch)
    for i in range(batch.shape[0]):  # per-slice min-max, like upstream predict
        slab = batch[i]
        lo, hi = float(slab.min()), float(slab.max())
        normed[i] = (slab - lo) / (hi - lo) if hi > lo else 0.0
    with torch.no_grad():
        logits = model(normed)
        prob = torch.sigmoid(logits)
    return prob[:, 0].float().cpu().numpy()


# ---------------------------------------------------------------------------
# fusion postprocess (pure, unit-tested)
# ---------------------------------------------------------------------------
def fuse_slab(pa: np.ndarray, pb: np.ndarray, pc: np.ndarray, threshold: float,
              min_area: int = 500) -> np.ndarray:
    """Fuse one slab of the three uint8 probability planes into a binary mask.

    Mean of the three (prob*255) planes -> threshold -> per-slice hole filling
    -> per-slice keep of components >= max(0.2 * largest, ``min_area`` px).
    The 3D largest-component cleanup happens later, on the registration grid,
    inside zarr_to_registration_nii.
    """
    from scipy.ndimage import binary_fill_holes, label

    mean = (pa.astype(np.float32) + pb.astype(np.float32) + pc.astype(np.float32)) / (3.0 * PROB_SCALE)
    binary = mean > threshold
    squeezed = binary.ndim == 2
    if squeezed:
        binary = binary[None]
    out = np.zeros(binary.shape, dtype=bool)
    for k in range(binary.shape[0]):
        slab = binary_fill_holes(binary[k])
        labels, n = label(slab)
        if n == 0:
            continue
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        keep = sizes >= max(0.2 * int(sizes.max()), min_area)
        out[k] = keep[labels]
    return out[0] if squeezed else out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run_brain_mask(
    sample_dir: str | Path,
    channel: int = 0,
    *,
    input_zarr: str | Path | None = None,
    output_zarr: str | Path | None = None,
    model_path: str | Path | None = None,
    prob_dir: str | Path | None = None,
    threshold: float = 0.7,
    keep_probs: bool = False,
    device: str = "cuda",
    min_area: int = 500,
) -> dict:
    sample_dir = Path(sample_dir)
    input_zarr = Path(input_zarr or sample_dir / f"ch{channel}_downsampled.zarr")
    output_zarr = Path(output_zarr or sample_dir / f"ch{channel}_brain_mask_dl.zarr")
    model_path = Path(model_path or DEFAULT_MODEL_PATH)
    prob_dir = Path(prob_dir or sample_dir / f"ch{channel}_brain_mask_probs")
    if not input_zarr.exists():
        raise FileNotFoundError(f"Coarse registration Zarr not found: {input_zarr}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {model_path} (see pipeline_modules/segmentation/models/)"
        )

    started_at = time.time()
    model = build_model(model_path, device=device)

    from pipeline_modules.utils.zarr_io import create_output_zarr, open_zarr_array

    volume = open_zarr_array(input_zarr, dataset_name="0")
    depth, height, width = (int(v) for v in volume.shape)
    total = depth + height + width
    done = 0

    def progress(tag: str) -> None:
        print(f"PROGRESS {done}/{total} {tag}", flush=True)

    prob_paths = {tag: prob_dir / f"prob_{tag}.zarr" for tag in "ABC"}
    prob_dir.mkdir(parents=True, exist_ok=True)

    def compute_plane(tag: str, n_slices: int, batch: int, chunk, setter, get_slice, dsize):
        """Infer one orientation into its uint8 probability cache (RAM, dumped at end)."""
        nonlocal done
        ram = _RamVolume((depth, height, width))
        for i in range(0, n_slices, batch):
            idxs = list(range(i, min(i + batch, n_slices)))
            slabs = np.stack([np.asarray(get_slice(k), np.float32) for k in idxs])
            probs = infer_probabilities(model, slabs, device=device)
            for j, k in enumerate(idxs):
                full = np.clip(
                    _resize_prob(probs[j], dsize) * PROB_SCALE, 0, 255
                ).astype(np.uint8)
                if tag == "A":
                    ram.buf[k] = full              # (h, w)
                elif tag == "B":
                    ram.buf[:, k, :] = full        # cv2.resize 已返回 (depth, width)
                else:
                    ram.buf[:, :, k] = full        # (depth, height)
            done += len(idxs)
            progress(f"prob_{tag}")
        _dump_volume_v2(prob_paths[tag], ram.buf, log=progress)

    def _resize_prob(prob: np.ndarray, dsize) -> np.ndarray:
        import cv2

        return cv2.resize(prob, dsize, interpolation=cv2.INTER_LINEAR)

    compute_plane("A", depth, BATCH_Z, (1, height, width),
                  lambda s, k, v: None, lambda k: volume[k], (width, height))
    compute_plane("B", height, BATCH_YX, (depth, 1, width),
                  lambda s, k, v: None, lambda k: volume[:, k, :], (width, depth))
    compute_plane("C", width, BATCH_YX, (depth, height, 1),
                  lambda s, k, v: None, lambda k: volume[:, :, k], (height, depth))

    # fuse: stream output z-slices; plane B/C rows touch every y/x chunk, so
    # this pass re-reads their full volumes once (a few minutes on NVMe).
    pa_store = _open_read(prob_paths["A"])
    pb_store = _open_read(prob_paths["B"])
    pc_store = _open_read(prob_paths["C"])
    mask_ram = _RamVolume((depth, height, width))
    fractions: list[float] = []
    for k in range(depth):
        slab = fuse_slab(pa_store[k], pb_store[k], pc_store[k], threshold,
                         min_area=min_area).astype(np.uint8)
        mask_ram[(slice(k, k + 1),)] = slab[None]
        done += 1
        fractions.append(float(slab.mean()))
        progress("fuse")
    _dump_volume_v2(output_zarr, mask_ram.buf, log=progress)
    out_meta = Path(output_zarr) / ".zattrs"
    meta = json.loads(out_meta.read_text(encoding="utf-8"))
    meta["brain_mask_aind"] = {
        "source_zarr": str(input_zarr),
        "model_path": str(model_path),
        "threshold": threshold,
        "planes": ["z", "y", "x"],
        "mean_fraction": round(float(np.mean(fractions)), 5),
    }
    out_meta.write_text(json.dumps(meta, indent=1), encoding="utf-8")

    if not keep_probs:
        import shutil

        for path in prob_paths.values():
            shutil.rmtree(path, ignore_errors=True)
        prob_dir.rmdir() if prob_dir.is_dir() and not any(prob_dir.iterdir()) else None

    _write_qc_montage(sample_dir, volume, output_zarr)
    return {
        "input_zarr": str(input_zarr),
        "output_zarr": str(output_zarr),
        "model_path": str(model_path),
        "threshold": threshold,
        "elapsed_s": round(time.time() - started_at, 1),
        "mean_fraction": float(np.mean(fractions)),
    }


def _open_read(path: Path):
    from pipeline_modules.utils.zarr_io import open_zarr_array

    return open_zarr_array(path, dataset_name="0")


def _write_qc_montage(sample_dir: Path, volume, mask_zarr: Path, n_slices: int = 6) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # QC is best-effort
        logger.warning("matplotlib unavailable; skipping brain mask QC montage")
        return

    mask = _open_read(mask_zarr)
    depth = int(mask.shape[0])
    picks = sorted({int(depth * f) for f in np.linspace(0.1, 0.9, n_slices)})
    fig, axes = plt.subplots(1, len(picks), figsize=(4 * len(picks), 4.4))
    for ax, k in zip(np.atleast_1d(axes), picks):
        image = np.asarray(volume[k], np.float32)
        mk = np.asarray(mask[k]) > 0
        ax.imshow(image, cmap="gray", vmin=np.percentile(image, 1), vmax=np.percentile(image, 99.5))
        ax.contour(mk, levels=[0.5], colors="cyan", linewidths=0.6)
        ax.set_title(f"z={k} frac={mk.mean():.3f}", fontsize=9)
        ax.axis("off")
    qc_dir = sample_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(qc_dir / "brain_mask_dl.png", dpi=110)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DL whole-brain mask (aind fusion3) for the registration channel"
    )
    parser.add_argument("--sample_dir", required=True, help="Sample directory")
    parser.add_argument("--channel", type=int, default=0, help="Registration channel number")
    parser.add_argument("--input_zarr", default=None, help="Coarse Zarr (default ch{channel}_downsampled.zarr)")
    parser.add_argument("--output_zarr", default=None, help="Mask Zarr (default ch{channel}_brain_mask_dl.zarr)")
    parser.add_argument("--model_path", default=None, help=f"Checkpoint (default {DEFAULT_MODEL_PATH.name} in models/)")
    parser.add_argument("--prob_dir", default=None, help="Probability cache dir (default ch{channel}_brain_mask_probs)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Fused probability threshold")
    parser.add_argument("--min_area", type=int, default=500,
                        help="Per-slice minimum component area in px (default 500)")
    parser.add_argument("--keep_probs", action="store_true", help="Keep the three uint8 probability caches")
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("--json_logs", action="store_true", help="NDJSON logs to stderr")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    started_at = time.time()
    result = run_brain_mask(
        args.sample_dir,
        args.channel,
        input_zarr=args.input_zarr,
        output_zarr=args.output_zarr,
        model_path=args.model_path,
        prob_dir=args.prob_dir,
        threshold=args.threshold,
        keep_probs=args.keep_probs,
        device=args.device,
        min_area=args.min_area,
    )
    from pipeline_modules.utils.run_manifest import write_run_manifest

    write_run_manifest(
        Path(args.sample_dir),
        module="pipeline_modules.segmentation.brain_mask_aind",
        entrypoint="run_brain_mask",
        inputs={
            "input_zarr": result["input_zarr"],
            "model_path": result["model_path"],
            "threshold": args.threshold,
        },
        outputs=[Path(result["output_zarr"])],
        started_at=started_at,
        extra=result,
    )
    logger.info("Brain mask written: %s (%.1f min)", result["output_zarr"], result["elapsed_s"] / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
