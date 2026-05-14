from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline_modules.segmentation.cfos_unet_model import load_cfos_unet_checkpoint, normalize_volume
from pipeline_modules.segmentation.zarr_utils import open_zarr_dataset

logger = logging.getLogger(__name__)


def _parse_int_tuple(value: Any, default: Sequence[int]) -> Tuple[int, int, int]:
    if value is None:
        return tuple(int(v) for v in default)
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError(f"Expected 3 comma-separated values, got: {value}")
        return tuple(int(p) for p in parts)
    if isinstance(value, Sequence):
        if len(value) != 3:
            raise ValueError(f"Expected length-3 sequence, got: {value}")
        return tuple(int(v) for v in value)
    raise ValueError(f"Unsupported tuple value: {value}")


def _compute_tile_starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, max(stride, 1)))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _ensure_channel_first(volume: np.ndarray) -> np.ndarray:
    if volume.ndim == 3:
        return volume
    if volume.ndim == 4 and volume.shape[0] == 1:
        return volume[0]
    if volume.ndim == 4 and volume.shape[-1] == 1:
        return volume[..., 0]
    raise ValueError(f"Expected a 3D scalar volume, got shape {tuple(volume.shape)}")


def _load_volume_and_meta(path_like: str | os.PathLike[str]) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(path_like)
    suffixes = {s.lower() for s in path.suffixes}

    if path.is_dir() and path.suffix.lower() == ".zarr":
        zarr_arr = open_zarr_dataset(path)
        volume = np.asarray(zarr_arr[:])
        return _ensure_channel_first(volume), {"format": "zarr", "source": str(path)}

    if ".nii" in suffixes or ".gz" in suffixes:
        import nibabel as nib
        import SimpleITK as sitk

        img = nib.load(str(path))
        volume = np.asarray(img.get_fdata(dtype=np.float32))
        sitk_img = sitk.ReadImage(str(path))
        return _ensure_channel_first(volume), {
            "format": "nifti",
            "source": str(path),
            "nifti": img,
            "sitk": sitk_img,
        }

    if ".tif" in suffixes or ".tiff" in suffixes:
        import tifffile

        volume = tifffile.imread(str(path))
        return _ensure_channel_first(np.asarray(volume)), {"format": "tiff", "source": str(path)}

    raise ValueError(f"Unsupported image path: {path}")


def _save_mask(mask: np.ndarray, meta: dict[str, Any], output_path: Path, dtype: np.dtype | type | str = np.uint8) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = meta.get("format")
    np_dtype = np.dtype(dtype)
    mask = mask.astype(np_dtype, copy=False)

    if output_path.suffix.lower() == ".nrrd":
        import SimpleITK as sitk

        # Internal inference uses nibabel-style XYZ arrays for NIfTI volumes,
        # while SimpleITK expects ZYX array ordering when creating an image.
        mask_zyx = np.transpose(mask, (2, 1, 0)) if mask.ndim == 3 else mask
        image = sitk.GetImageFromArray(mask_zyx)
        ref_sitk = meta.get("sitk")
        if ref_sitk is not None:
            image.CopyInformation(ref_sitk)
        else:
            ref = meta.get("nifti")
            if ref is not None:
                zooms = tuple(float(v) for v in ref.header.get_zooms()[:3])
                image.SetSpacing((zooms[2], zooms[1], zooms[0]))
        sitk.WriteImage(image, str(output_path))
        return str(output_path)

    if fmt == "nifti":
        import nibabel as nib

        ref = meta["nifti"]
        out = nib.Nifti1Image(mask, affine=ref.affine, header=ref.header)
        nib.save(out, str(output_path))
        return str(output_path)

    if fmt == "tiff":
        import tifffile

        tifffile.imwrite(str(output_path), mask)
        return str(output_path)

    if fmt == "zarr":
        import zarr
        from numcodecs import Blosc

        root = zarr.group(store=zarr.DirectoryStore(str(output_path)), overwrite=True)
        root.create_dataset(
            "0",
            data=mask,
            chunks=mask.shape if all(v <= 256 for v in mask.shape) else (64, 128, 128),
            compressor=Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE),
        )
        root.attrs["multiscales"] = [{"version": "0.4", "datasets": [{"path": "0"}]}]
        return str(output_path)

    raise ValueError(f"Unsupported output format: {fmt}")


def _resolve_output_file(image_path: str, output_dir: Path, suffix: str) -> Path:
    src = Path(image_path)
    if src.is_dir() and src.suffix.lower() == ".zarr":
        return output_dir / f"{src.stem}{suffix}.zarr"
    if src.suffix.lower() == ".gz" and len(src.suffixes) >= 2 and src.suffixes[-2].lower() == ".nii":
        return output_dir / f"{src.name[:-7]}{suffix}.nii.gz"
    return output_dir / f"{src.stem}{suffix}{src.suffix}"


def _resolve_requested_output_file(
    image_path: str,
    output_dir: Path,
    suffix: str,
    result_extension: str | None,
) -> Path:
    if result_extension:
        normalized = result_extension.strip()
        if not normalized.startswith("."):
            normalized = f".{normalized}"
        src = Path(image_path)
        base_name = src.stem
        if src.suffix.lower() == ".gz" and len(src.suffixes) >= 2 and src.suffixes[-2].lower() == ".nii":
            base_name = src.name[:-7]
        if src.is_dir() and src.suffix.lower() == ".zarr":
            base_name = src.stem
        return output_dir / f"{base_name}{suffix}{normalized}"
    return _resolve_output_file(image_path, output_dir, suffix)


def _resolve_numpy_dtype(value: Any, default: str = "uint8") -> np.dtype:
    try:
        return np.dtype(str(value or default))
    except TypeError:
        return np.dtype(default)


class CFOSActiveLearningInfer(InferTask):
    def __init__(self, conf: Dict[str, Any]):
        self.conf = dict(conf)
        self.checkpoint_path = Path(self.conf.get("checkpoint", r"S:\Yifu\best_model.pt"))
        self.patch_size = _parse_int_tuple(self.conf.get("patch_size"), default=(128, 128, 128))
        self.batch_size = int(self.conf.get("infer_batch_size", 2))
        self.overlap = float(self.conf.get("overlap", 0.25))
        self.threshold = float(self.conf.get("threshold", 0.5))
        self.foreground_class = int(self.conf.get("foreground_class", 1))
        self.output_dir = Path(self.conf.get("output_dir", Path.cwd() / "output"))
        self.device_request = str(self.conf.get("device", "cuda"))

        self._bundle: dict[str, Any] | None = None
        self._resolved_device: str | None = None
        super().__init__(
            type=InferType.SEGMENTATION,
            labels=["background", "cfos"],
            dimension=3,
            description="3D cFos segmentation inference using best_model.pt",
            config={
                # Keep MONAI Label UI config scalar-only; Slicer treats lists as choice sets.
                "patch_size": ",".join(str(v) for v in self.patch_size),
                "infer_batch_size": self.batch_size,
                "overlap": self.overlap,
                "threshold": self.threshold,
            },
        )

    def is_valid(self) -> bool:
        return self.checkpoint_path.exists()

    def get_path(self, validate=True):
        if validate and not self.is_valid():
            return None
        return str(self.checkpoint_path)

    def _load_model(self):
        if self._bundle is not None:
            return self._bundle, self._resolved_device

        bundle = load_cfos_unet_checkpoint(self.checkpoint_path, device="cpu")
        torch_mod = bundle["torch"]
        requested = self.device_request.lower()
        resolved = "cuda" if requested in {"auto", "cuda"} and torch_mod.cuda.is_available() else "cpu"
        bundle["model"].to(resolved)
        bundle["model"].eval()
        self._bundle = bundle
        self._resolved_device = resolved
        return bundle, resolved

    def _infer_logits(self, volume: np.ndarray) -> np.ndarray:
        bundle, device = self._load_model()
        torch_mod = bundle["torch"]
        model = bundle["model"]

        tile_d, tile_h, tile_w = self.patch_size
        stride_d = max(1, int(round(tile_d * (1.0 - self.overlap))))
        stride_h = max(1, int(round(tile_h * (1.0 - self.overlap))))
        stride_w = max(1, int(round(tile_w * (1.0 - self.overlap))))

        d, h, w = volume.shape
        pad_d = max(tile_d - d, 0)
        pad_h = max(tile_h - h, 0)
        pad_w = max(tile_w - w, 0)
        if pad_d or pad_h or pad_w:
            volume = np.pad(volume, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant")
        padded_shape = volume.shape

        logits_acc = None
        count_acc = None
        patches = []
        coords = []
        use_amp = device.startswith("cuda")

        def flush():
            nonlocal logits_acc, count_acc, patches, coords
            if not patches:
                return
            batch_np = np.stack(patches, axis=0)[:, None, ...].astype(np.float32, copy=False)
            batch_tensor = torch_mod.from_numpy(batch_np).to(device)
            with torch_mod.no_grad():
                if use_amp:
                    with torch_mod.autocast(device_type="cuda", dtype=torch_mod.float16):
                        logits = model(batch_tensor).detach().float()
                else:
                    logits = model(batch_tensor).detach().float()
            if logits_acc is None:
                num_classes = int(logits.shape[1])
                logits_acc = torch_mod.zeros((num_classes,) + padded_shape, dtype=torch_mod.float32, device=device)
                count_acc = torch_mod.zeros((1,) + padded_shape, dtype=torch_mod.float32, device=device)
            for idx, (z0, y0, x0) in enumerate(coords):
                logits_acc[:, z0:z0 + tile_d, y0:y0 + tile_h, x0:x0 + tile_w] += logits[idx]
                count_acc[:, z0:z0 + tile_d, y0:y0 + tile_h, x0:x0 + tile_w] += 1.0
            patches = []
            coords = []

        for z0 in _compute_tile_starts(padded_shape[0], tile_d, stride_d):
            for y0 in _compute_tile_starts(padded_shape[1], tile_h, stride_h):
                for x0 in _compute_tile_starts(padded_shape[2], tile_w, stride_w):
                    patches.append(volume[z0:z0 + tile_d, y0:y0 + tile_h, x0:x0 + tile_w])
                    coords.append((z0, y0, x0))
                    if len(patches) >= max(1, self.batch_size):
                        flush()
        flush()

        averaged = (logits_acc / count_acc.clamp(min=1.0))[:, :d, :h, :w]
        return averaged.cpu().numpy()

    def infer_array(self, image_path: str | os.PathLike[str]) -> dict[str, Any]:
        started = time.time()
        volume, meta = _load_volume_and_meta(image_path)
        volume = normalize_volume(volume)
        logits = self._infer_logits(volume)

        bundle, device = self._load_model()
        torch_mod = bundle["torch"]
        probs = torch_mod.softmax(torch_mod.from_numpy(logits), dim=0).cpu().numpy()
        foreground_prob = probs[self.foreground_class]
        prediction = (foreground_prob >= self.threshold).astype(np.uint8)
        entropy = -np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0)), axis=0)

        return {
            "prediction": prediction,
            "probability": foreground_prob.astype(np.float32),
            "entropy": entropy.astype(np.float32),
            "meta": meta,
            "device": device,
            "latency": time.time() - started,
        }

    def __call__(self, request):
        image_path = request.get("image")
        if not image_path:
            raise ValueError("request['image'] is required")

        output_dir = Path(request.get("output_dir", self.output_dir))
        save_mask = _as_bool(request.get("save_mask", True), True)
        save_entropy = _as_bool(request.get("save_entropy", False), False)
        save_prob = _as_bool(request.get("save_prob", False), False)
        result_extension = request.get("result_extension")
        result_dtype = _resolve_numpy_dtype(request.get("result_dtype"), default="uint8")

        result = self.infer_array(image_path)
        output_file = None
        params = {
            "device": result["device"],
            "latency": round(float(result["latency"]), 4),
            "shape": list(result["prediction"].shape),
            "mean_entropy": float(result["entropy"].mean()),
            "max_entropy": float(result["entropy"].max()),
            "foreground_ratio": float(result["prediction"].mean()),
            "label_names": ["background", "cfos"],
        }

        if save_mask:
            output_path = _resolve_requested_output_file(str(image_path), output_dir, "_pred", result_extension)
            output_file = _save_mask(result["prediction"], result["meta"], output_path, dtype=result_dtype)
            params["output_label"] = output_file

        if save_prob:
            prob_path = _resolve_requested_output_file(str(image_path), output_dir, "_prob", result_extension)
            params["output_prob"] = _save_mask(
                (result["probability"] * 255.0).clip(0, 255),
                result["meta"],
                prob_path,
                dtype=result_dtype,
            )

        if save_entropy:
            entropy_path = _resolve_requested_output_file(str(image_path), output_dir, "_entropy", result_extension)
            entropy_norm = result["entropy"] / max(float(result["entropy"].max()), 1e-8)
            params["output_entropy"] = _save_mask(
                (entropy_norm * 255.0).clip(0, 255),
                result["meta"],
                entropy_path,
                dtype=result_dtype,
            )

        return output_file, params
