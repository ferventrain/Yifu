#!/usr/bin/env python3
"""Train a 3D U-Net segmentation model on cFos whole-brain volumes.

Features:
  - 5-level 3D U-Net with residual blocks and attention gates
  - Focal + Dice combined loss for extreme class imbalance (~0.8% foreground)
  - Foreground-aware sampling and cropping
  - Sliding-window validation on full volumes
  - MLflow experiment tracking

Quick start:
    python train_cfos_3d_mlflow.py \\
        --data-root /data/cfos --patch-size 128 128 128 \\
        --batch-size 16 --base-channels 24 --lr 2e-3 --epochs 100 \\
        --use-fg-aware-sampler --cache-data

Outputs:
    outputs/<experiment>/best_model.pt   - best val-dice checkpoint
    outputs/<experiment>/train_config.json - full config for reproducibility
    outputs/<experiment>/data_split.json  - train/val case lists
    mlruns/                               - MLflow tracking data
"""
import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

try:
    import mlflow
except ImportError:
    mlflow = None

try:
    import tifffile
except ImportError:
    tifffile = None

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# =====================================================================
#  CLI Arguments
# =====================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser with all training hyper-parameters."""
    parser = argparse.ArgumentParser(
        description="Train a 3D segmentation model on cFos data with MLflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=str, default="/data/cfos", help="Root folder containing image/ and mask/.")
    parser.add_argument("--image-dir", type=str, default=None, help="Override image directory. Default: <data-root>/image")
    parser.add_argument("--mask-dir", type=str, default=None, help="Override mask directory. Default: <data-root>/mask")
    parser.add_argument("--output-dir", type=str, default="./outputs/cfos_3d_mlflow", help="Folder for checkpoints and logs.")
    parser.add_argument("--experiment-name", type=str, default="cfos_3d_segmentation", help="MLflow experiment name.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional MLflow run name.")
    parser.add_argument("--tracking-uri", type=str, default="file:./mlruns", help="MLflow tracking URI.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers. Use 0 first in Docker/debug mode.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warmup epochs for learning rate.")
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.05,
        help="Minimum LR as a ratio of base lr for cosine decay, e.g. 0.05 means min_lr = 0.05 * lr.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of segmentation classes.")
    parser.add_argument("--base-channels", type=int, default=8, help="Base channel width for the 3D U-Net.")
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        default=[256, 256, 256],
        metavar=("D", "H", "W"),
        help="Training patch size. Use 256 256 256 for full-volume training.",
    )
    parser.add_argument(
        "--cls-loss",
        type=str,
        default="focal",
        choices=["ce", "focal"],
        help="Classification loss type combined with Dice.",
    )
    parser.add_argument("--ce-weight", type=float, default=0.5, help="Classification loss weight in total loss.")
    parser.add_argument("--dice-weight", type=float, default=0.5, help="Dice loss weight in total loss.")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma.")
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=0.6,
        help="Foreground alpha for binary focal loss. Background alpha will be 1-alpha.",
    )
    parser.add_argument("--save-every", type=int, default=5, help="Save an epoch checkpoint every N epochs.")
    parser.add_argument("--cache-data", action="store_true", help="Cache loaded volumes in memory.")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    parser.add_argument(
        "--accumulate-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps. Use batch-size=1 plus accumulation to simulate a larger batch.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=0.0,
        help="Global grad norm clipping. Set > 0 to enable (for stability).",
    )
    parser.add_argument(
        "--p-spatial-aug",
        type=float,
        default=0.6,
        help="Probability to apply spatial augmentation block on a training sample.",
    )
    parser.add_argument(
        "--p-intensity-aug",
        type=float,
        default=0.85,
        help="Probability to apply intensity augmentation block on a training sample.",
    )
    parser.add_argument(
        "--fg-aware-crop-prob",
        type=float,
        default=0.6,
        help="Probability of foreground-centered crop when random crop is possible.",
    )
    parser.add_argument(
        "--use-fg-aware-sampler",
        action="store_true",
        help="Enable weighted sampling by foreground ratio to mitigate class imbalance.",
    )
    parser.add_argument(
        "--fg-sampler-power",
        type=float,
        default=0.5,
        help="Power for foreground-ratio weighting in sampler; lower is smoother.",
    )
    parser.add_argument(
        "--fg-sampler-min-weight",
        type=float,
        default=0.2,
        help="Minimum normalized sampling weight for low-foreground samples.",
    )
    parser.add_argument(
        "--patches-per-volume",
        type=int,
        default=2,
        help="Training virtual dataset length multiplier: each volume is sampled this many patches per epoch.",
    )
    parser.add_argument(
        "--val-mode",
        type=str,
        default="sliding",
        choices=["center", "sliding"],
        help="Validation mode. 'center' = single center crop. 'sliding' = tile whole volume with overlap and aggregate.",
    )
    parser.add_argument(
        "--val-overlap",
        type=float,
        default=0,
        help="Overlap ratio between sliding-window tiles during validation (0 means no overlap).",
    )
    parser.add_argument(
        "--log-every-n-batches",
        type=int,
        default=0,
        help="Print batch-level progress every N batches. Set 0 to disable batch-level prints.",
    )
    parser.add_argument(
        "--show-progress-bar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tqdm progress bar for batches. Use --no-show-progress-bar to disable.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


# =====================================================================
#  Utilities: Dependencies, Seed, LR Schedule, AMP
# =====================================================================

def ensure_dependencies() -> None:
    missing = []
    if mlflow is None:
        missing.append("mlflow")
    if tifffile is None and sitk is None:
        missing.append("tifffile or SimpleITK")
    if missing:
        raise ImportError(
            "Missing dependencies: {}. Install them before running this script.".format(", ".join(missing))
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device_type: str, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device_type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def compute_lr_for_epoch(
    epoch: int,
    total_epochs: int,
    base_lr: float,
    warmup_epochs: int,
    min_lr_ratio: float,
) -> float:
    min_lr = base_lr * min(max(min_lr_ratio, 0.0), 1.0)
    total_epochs = max(int(total_epochs), 1)
    warmup_epochs = max(0, min(int(warmup_epochs), total_epochs))

    # No warmup: cosine from base_lr to min_lr across all epochs.
    if warmup_epochs == 0:
        progress = (epoch - 1) / max(total_epochs - 1, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine

    # Warmup stage: linearly ramp from min_lr to base_lr.
    if epoch <= warmup_epochs:
        warmup_progress = epoch / max(warmup_epochs, 1)
        return min_lr + (base_lr - min_lr) * warmup_progress

    # Post-warmup cosine decay.
    decay_epochs = max(total_epochs - warmup_epochs, 1)
    decay_progress = (epoch - warmup_epochs - 1) / max(decay_epochs - 1, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_lr + (base_lr - min_lr) * cosine


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# =====================================================================
#  Data I/O: Read, Normalize, Prepare Mask, Discover Samples
# =====================================================================

def read_volume(path: Path) -> np.ndarray:
    """Read a 3D volume from .tif/.tiff (via tifffile or SimpleITK)."""
    if tifffile is not None:
        array = tifffile.imread(str(path))
    elif sitk is not None:
        array = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    else:
        raise RuntimeError("Neither tifffile nor SimpleITK is available.")

    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError("Expected a 3D volume at {}, got shape {}.".format(path, array.shape))
    return array


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Percentile-based normalization to [0, 1]."""
    volume = volume.astype(np.float32, copy=False)
    low, high = np.percentile(volume, (1.0, 99.5))
    if high <= low:
        volume = volume - volume.min()
        denom = volume.max()
        return volume / denom if denom > 0 else volume
    volume = np.clip(volume, low, high)
    volume = volume - low
    volume = volume / max(high - low, 1e-6)
    return volume


def prepare_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    mask = np.asarray(mask)
    if num_classes == 2:
        return (mask > 0).astype(np.int64)

    if np.issubdtype(mask.dtype, np.floating):
        mask = np.rint(mask)
    mask = mask.astype(np.int64, copy=False)
    if mask.min() < 0:
        raise ValueError("Mask contains negative labels, which is not supported.")
    if mask.max() >= num_classes:
        raise ValueError("Mask max label {} exceeds num_classes={}.".format(mask.max(), num_classes))
    return mask


def find_mask_path(mask_dir: Path, image_path: Path) -> Path:
    candidates = [
        mask_dir / "{}_mask{}".format(image_path.stem, image_path.suffix),
        mask_dir / "{}_mask.tif".format(image_path.stem),
        mask_dir / "{}_mask.tiff".format(image_path.stem),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Mask for {} not found in {}.".format(image_path.name, mask_dir))


def collect_samples(image_dir: Path, mask_dir: Path) -> List[Dict[str, str]]:
    image_paths = sorted(list(image_dir.glob("*.tif")) + list(image_dir.glob("*.tiff")))
    if not image_paths:
        raise FileNotFoundError("No .tif or .tiff files found in {}.".format(image_dir))

    samples = []
    for image_path in image_paths:
        mask_path = find_mask_path(mask_dir, image_path)
        samples.append(
            {
                "case_id": image_path.stem,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
            }
        )
    return samples


def split_samples(samples: List[Dict[str, str]], val_ratio: float, seed: int) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if len(samples) == 1:
        return samples, samples

    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)
    val_count = max(1, int(round(len(samples) * val_ratio)))
    val_count = min(val_count, len(samples) - 1)
    val_indices = set(indices[:val_count])
    train_samples = [sample for idx, sample in enumerate(samples) if idx not in val_indices]
    val_samples = [sample for idx, sample in enumerate(samples) if idx in val_indices]
    return train_samples, val_samples


# =====================================================================
#  Data Splitting & Foreground-aware Sampling
# =====================================================================

def compute_mask_foreground_ratio(mask_path: str, num_classes: int) -> float:
    mask = prepare_mask(read_volume(Path(mask_path)), num_classes=num_classes)
    return float((mask > 0).mean())


def build_fg_aware_sampling_weights(
    samples: List[Dict[str, str]],
    num_classes: int,
    power: float,
    min_weight: float,
) -> Tuple[np.ndarray, np.ndarray]:
    ratios = []
    for sample in samples:
        ratios.append(compute_mask_foreground_ratio(sample["mask_path"], num_classes=num_classes))

    ratio_arr = np.asarray(ratios, dtype=np.float32)
    base = np.power(ratio_arr + 1e-6, max(power, 0.0))
    base = base / max(float(base.mean()), 1e-6)
    weights = np.maximum(base, max(min_weight, 1e-6))
    return weights.astype(np.float64), ratio_arr


# =====================================================================
#  Cropping, Padding & Augmentation
# =====================================================================

def compute_crop_slices(
    volume_shape: Sequence[int],
    target_shape: Sequence[int],
    random_crop: bool,
    fg_center: Optional[Tuple[int, int, int]] = None,
) -> Tuple[slice, slice, slice]:
    slices = []
    for axis, (current, target) in enumerate(zip(volume_shape, target_shape)):
        if current <= target:
            start = 0
        elif random_crop:
            if fg_center is None:
                start = random.randint(0, current - target)
            else:
                center = int(fg_center[axis])
                start = max(0, min(center - target // 2, current - target))
        else:
            start = (current - target) // 2
        slices.append(slice(start, min(start + target, current)))
    return slices[0], slices[1], slices[2]


def crop_or_pad(
    image: np.ndarray,
    mask: np.ndarray,
    target_shape: Sequence[int],
    random_crop: bool,
    fg_aware_crop_prob: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    if image.shape != mask.shape:
        raise ValueError("Image and mask shapes do not match: {} vs {}.".format(image.shape, mask.shape))

    fg_center = None
    if random_crop and fg_aware_crop_prob > 0 and random.random() < fg_aware_crop_prob:
        fg_points = np.argwhere(mask > 0)
        if fg_points.size > 0:
            fg_center = tuple(fg_points[random.randrange(len(fg_points))].tolist())

    target_d, target_h, target_w = target_shape
    src_slices = compute_crop_slices(image.shape, target_shape, random_crop, fg_center=fg_center)
    cropped_image = image[src_slices]
    cropped_mask = mask[src_slices]

    pad_d = max(target_d - cropped_image.shape[0], 0)
    pad_h = max(target_h - cropped_image.shape[1], 0)
    pad_w = max(target_w - cropped_image.shape[2], 0)
    if pad_d or pad_h or pad_w:
        padding = (
            (pad_d // 2, pad_d - pad_d // 2),
            (pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2),
        )
        cropped_image = np.pad(cropped_image, padding, mode="constant", constant_values=0)
        cropped_mask = np.pad(cropped_mask, padding, mode="constant", constant_values=0)

    return cropped_image, cropped_mask


def _shift_with_zeros(volume: np.ndarray, shift_d: int, shift_h: int, shift_w: int) -> np.ndarray:
    out = np.zeros_like(volume)
    d, h, w = volume.shape

    src_d0 = max(0, -shift_d)
    src_h0 = max(0, -shift_h)
    src_w0 = max(0, -shift_w)
    src_d1 = d - max(0, shift_d)
    src_h1 = h - max(0, shift_h)
    src_w1 = w - max(0, shift_w)

    dst_d0 = max(0, shift_d)
    dst_h0 = max(0, shift_h)
    dst_w0 = max(0, shift_w)
    dst_d1 = dst_d0 + (src_d1 - src_d0)
    dst_h1 = dst_h0 + (src_h1 - src_h0)
    dst_w1 = dst_w0 + (src_w1 - src_w0)

    if src_d1 <= src_d0 or src_h1 <= src_h0 or src_w1 <= src_w0:
        return out

    out[dst_d0:dst_d1, dst_h0:dst_h1, dst_w0:dst_w1] = volume[src_d0:src_d1, src_h0:src_h1, src_w0:src_w1]
    return out


def apply_spatial_augmentations(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    for axis in range(3):
        if random.random() < 0.5:
            image = np.flip(image, axis=axis).copy()
            mask = np.flip(mask, axis=axis).copy()

    # Light morphology variance: rotate in-plane by 90-degree steps.
    if random.random() < 0.35:
        k = random.randint(1, 3)
        image = np.rot90(image, k=k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k=k, axes=(1, 2)).copy()

    # Small translation with zero padding.
    if random.random() < 0.25:
        max_shift_d = max(1, int(image.shape[0] * 0.03))
        max_shift_h = max(1, int(image.shape[1] * 0.03))
        max_shift_w = max(1, int(image.shape[2] * 0.03))
        shift_d = random.randint(-max_shift_d, max_shift_d)
        shift_h = random.randint(-max_shift_h, max_shift_h)
        shift_w = random.randint(-max_shift_w, max_shift_w)
        image = _shift_with_zeros(image, shift_d, shift_h, shift_w)
        mask = _shift_with_zeros(mask, shift_d, shift_h, shift_w)

    return image, mask


def apply_intensity_augmentations(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)

    # Contrast and brightness jitter.
    if random.random() < 0.6:
        contrast = random.uniform(0.7, 1.3)
        brightness = random.uniform(-0.12, 0.12)
        image = image * contrast + brightness

    # Gamma jitter.
    if random.random() < 0.45:
        gamma = random.uniform(0.7, 1.6)
        image = np.power(np.clip(image, 0.0, 1.0), gamma).astype(np.float32, copy=False)

    # Additive Gaussian noise.
    if random.random() < 0.35:
        sigma = random.uniform(0.0, 0.06)
        image = image + np.random.normal(0.0, sigma, size=image.shape).astype(np.float32)

    # Multiplicative noise to mimic background/sensor noise fluctuation.
    if random.random() < 0.3:
        sigma_mul = random.uniform(0.0, 0.08)
        image = image + image * np.random.normal(0.0, sigma_mul, size=image.shape).astype(np.float32)

    # Low-frequency background drift along one axis.
    if random.random() < 0.3:
        low = random.uniform(-0.15, 0.0)
        high = random.uniform(0.0, 0.15)
        axis = random.randint(0, 2)
        if axis == 0:
            drift = np.linspace(low, high, image.shape[0], dtype=np.float32)[:, None, None]
        elif axis == 1:
            drift = np.linspace(low, high, image.shape[1], dtype=np.float32)[None, :, None]
        else:
            drift = np.linspace(low, high, image.shape[2], dtype=np.float32)[None, None, :]
        image = image + drift

    # Dynamic range squeeze/expand.
    if random.random() < 0.35:
        low = random.uniform(0.0, 0.12)
        high = random.uniform(0.88, 1.0)
        if high - low > 1e-6:
            image = (image - low) / (high - low)

    return np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)


# =====================================================================
#  Dataset
# =====================================================================

class VolumeDataset(Dataset):
    """Yields (image_patch, mask_patch) pairs with optional augmentation.

    Each volume is randomly cropped ``patches_per_volume`` times per epoch.
    Foreground-aware cropping centers patches on foreground voxels with
    probability ``fg_aware_crop_prob``.
    """
    def __init__(
        self,
        samples: List[Dict[str, str]],
        patch_size: Sequence[int],
        num_classes: int,
        training: bool,
        cache_data: bool = False,
        p_spatial_aug: float = 0.6,
        p_intensity_aug: float = 0.85,
        fg_aware_crop_prob: float = 0.0,
        patches_per_volume: int = 1,
    ) -> None:
        self.samples = samples
        self.patch_size = tuple(int(x) for x in patch_size)
        self.num_classes = num_classes
        self.training = training
        self.cache_data = cache_data
        self.p_spatial_aug = float(np.clip(p_spatial_aug, 0.0, 1.0))
        self.p_intensity_aug = float(np.clip(p_intensity_aug, 0.0, 1.0))
        self.fg_aware_crop_prob = float(np.clip(fg_aware_crop_prob, 0.0, 1.0))
        self.patches_per_volume = max(1, int(patches_per_volume)) if training else 1
        self.cache = {}

    def __len__(self) -> int:
        return len(self.samples) * self.patches_per_volume

    def load_sample(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.cache_data and index in self.cache:
            return self.cache[index]

        sample = self.samples[index]
        image = normalize_volume(read_volume(Path(sample["image_path"])))
        mask = prepare_mask(read_volume(Path(sample["mask_path"])), self.num_classes)
        if self.cache_data:
            self.cache[index] = (image, mask)
        return image, mask

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        real_index = index % len(self.samples)
        image, mask = self.load_sample(real_index)
        image, mask = crop_or_pad(
            image,
            mask,
            self.patch_size,
            random_crop=self.training,
            fg_aware_crop_prob=self.fg_aware_crop_prob if self.training else 0.0,
        )

        if self.training:
            if random.random() < self.p_spatial_aug:
                image, mask = apply_spatial_augmentations(image, mask)
            if random.random() < self.p_intensity_aug:
                image = apply_intensity_augmentations(image)

        image_tensor = torch.from_numpy(image[None, ...].astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.int64))
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "case_id": self.samples[real_index]["case_id"],
        }


# =====================================================================
#  Model Architecture: 5-level Residual U-Net with Attention Gates
# =====================================================================

class ConvBlock3D(nn.Module):
    """Double 3×3×3 conv block with residual (1×1) skip connection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ConvBlock3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
        )
        self.skip = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + self.skip(x))


class AttentionGate3D(nn.Module):
    """Additive attention gate: uses decoder (gate) signal to highlight
    relevant regions in the encoder (skip) features."""

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
        super(AttentionGate3D, self).__init__()
        self.W_gate = nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=True)
        self.W_skip = nn.Conv3d(skip_channels, inter_channels, kernel_size=1, bias=True)
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        g = match_tensor_shape(g, s)
        alpha = self.psi(self.relu(g + s))
        return skip * alpha


def match_tensor_shape(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.shape[-3:] == target.shape[-3:]:
        return source
    return F.interpolate(source, size=target.shape[-3:], mode="trilinear", align_corners=False)


class SimpleUNet3D(nn.Module):
    """5-level 3D U-Net with residual ConvBlocks and Attention Gates."""

    def __init__(self, in_channels: int = 1, num_classes: int = 2, base_channels: int = 8) -> None:
        super(SimpleUNet3D, self).__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        # ---------- Encoder ----------
        self.enc1 = ConvBlock3D(in_channels, c1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc2 = ConvBlock3D(c1, c2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc3 = ConvBlock3D(c2, c3)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc4 = ConvBlock3D(c3, c4)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc5 = ConvBlock3D(c4, c5)  # bottleneck

        # ---------- Decoder ----------
        self.up4 = nn.ConvTranspose3d(c5, c4, kernel_size=2, stride=2)
        self.ag4 = AttentionGate3D(gate_channels=c4, skip_channels=c4, inter_channels=c4 // 2)
        self.dec4 = ConvBlock3D(c4 + c4, c4)

        self.up3 = nn.ConvTranspose3d(c4, c3, kernel_size=2, stride=2)
        self.ag3 = AttentionGate3D(gate_channels=c3, skip_channels=c3, inter_channels=c3 // 2)
        self.dec3 = ConvBlock3D(c3 + c3, c3)

        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.ag2 = AttentionGate3D(gate_channels=c2, skip_channels=c2, inter_channels=c2 // 2)
        self.dec2 = ConvBlock3D(c2 + c2, c2)

        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.ag1 = AttentionGate3D(gate_channels=c1, skip_channels=c1, inter_channels=c1 // 2)
        self.dec1 = ConvBlock3D(c1 + c1, c1)

        self.head = nn.Conv3d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        s4 = self.enc4(self.pool3(s3))
        btm = self.enc5(self.pool4(s4))

        # Decoder with attention-gated skip connections
        d4 = self.up4(btm)
        d4 = match_tensor_shape(d4, s4)
        d4 = self.dec4(torch.cat([d4, self.ag4(d4, s4)], dim=1))

        d3 = self.up3(d4)
        d3 = match_tensor_shape(d3, s3)
        d3 = self.dec3(torch.cat([d3, self.ag3(d3, s3)], dim=1))

        d2 = self.up2(d3)
        d2 = match_tensor_shape(d2, s2)
        d2 = self.dec2(torch.cat([d2, self.ag2(d2, s2)], dim=1))

        d1 = self.up1(d2)
        d1 = match_tensor_shape(d1, s1)
        d1 = self.dec1(torch.cat([d1, self.ag1(d1, s1)], dim=1))

        return self.head(d1)


# =====================================================================
#  Loss Functions
# =====================================================================

class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_background: bool = True) -> None:
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_background = ignore_background and num_classes > 1

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()

        start_class = 1 if self.ignore_background else 0
        dice_losses = []
        for class_index in range(start_class, self.num_classes):
            pred_flat = probs[:, class_index].contiguous().view(probs.size(0), -1)
            target_flat = target_one_hot[:, class_index].contiguous().view(target.size(0), -1)
            intersection = (pred_flat * target_flat).sum(dim=1)
            union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
            dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
            dice_losses.append(1.0 - dice)

        if not dice_losses:
            return logits.new_tensor(0.0)
        return torch.stack(dice_losses, dim=0).mean()


class FocalLoss(nn.Module):
    def __init__(self, num_classes: int, gamma: float = 2.0, alpha: float = 0.85) -> None:
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma

        if num_classes == 2:
            fg_alpha = float(np.clip(alpha, 0.0, 1.0))
            alpha_tensor = torch.tensor([1.0 - fg_alpha, fg_alpha], dtype=torch.float32)
        else:
            alpha_tensor = torch.ones(num_classes, dtype=torch.float32)
            if num_classes > 1:
                fg_alpha = float(max(alpha, 0.0))
                alpha_tensor[1:] = fg_alpha
                alpha_tensor[0] = 1.0
        self.register_buffer("alpha_tensor", alpha_tensor)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        focal = torch.pow(1.0 - pt, self.gamma) * ce

        alpha = self.alpha_tensor.to(device=logits.device, dtype=logits.dtype)
        alpha_t = alpha[target]
        loss = alpha_t * focal
        return loss.mean()


# =====================================================================
#  Metrics (batch-level, used during training)
# =====================================================================

def compute_metrics(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict[str, float]:
    pred = torch.argmax(logits, dim=1)
    start_class = 1 if num_classes > 1 else 0

    dice_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []

    for class_index in range(start_class, num_classes):
        pred_mask = pred == class_index
        target_mask = target == class_index

        intersection = (pred_mask & target_mask).sum().float()
        pred_sum = pred_mask.sum().float()
        target_sum = target_mask.sum().float()
        union = pred_mask.logical_or(target_mask).sum().float()

        dice = (2.0 * intersection + 1e-5) / (pred_sum + target_sum + 1e-5)
        iou = (intersection + 1e-5) / (union + 1e-5)
        precision = (intersection + 1e-5) / (pred_sum + 1e-5)
        recall = (intersection + 1e-5) / (target_sum + 1e-5)

        dice_scores.append(dice.item())
        iou_scores.append(iou.item())
        precision_scores.append(precision.item())
        recall_scores.append(recall.item())

    if not dice_scores:
        return {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

    return {
        "dice": float(np.mean(dice_scores)),
        "iou": float(np.mean(iou_scores)),
        "precision": float(np.mean(precision_scores)),
        "recall": float(np.mean(recall_scores)),
    }


# =====================================================================
#  Training & Validation Loops
# =====================================================================

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    cls_loss_fn: nn.Module,
    dice_loss: DiceLoss,
    device: torch.device,
    num_classes: int,
    ce_weight: float,
    dice_weight: float,
    use_amp: bool,
    accumulate_steps: int,
    phase_name: str,
    epoch: int,
    log_every_n_batches: int,
    show_progress_bar: bool,
    grad_clip_norm: float,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    scaler = create_grad_scaler(enabled=is_train and use_amp)
    device_type = "cuda" if device.type == "cuda" else "cpu"

    total_loss = 0.0
    total_cls = 0.0
    total_dice_loss = 0.0
    total_metrics = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    iterator = tqdm(loader, desc="{} epoch {}".format(phase_name, epoch), leave=False) if show_progress_bar else loader
    for batch_index, batch in enumerate(iterator, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with autocast_context(device_type=device_type, enabled=use_amp):
                logits = model(images)
                loss_cls = cls_loss_fn(logits, masks)
                loss_dice = dice_loss(logits, masks)
                loss = ce_weight * loss_cls + dice_weight * loss_dice

            if is_train:
                scaled_loss = loss / max(accumulate_steps, 1)
                scaler.scale(scaled_loss).backward()
                should_step = (batch_index % max(accumulate_steps, 1) == 0) or (batch_index == len(loader))
                if should_step:
                    if grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        batch_metrics = compute_metrics(logits.detach(), masks.detach(), num_classes)
        total_loss += loss.item()
        total_cls += loss_cls.item()
        total_dice_loss += loss_dice.item()
        for key in total_metrics:
            total_metrics[key] += batch_metrics[key]

        should_print_batch = (
            log_every_n_batches > 0
            and (batch_index == 1 or batch_index % log_every_n_batches == 0 or batch_index == len(loader))
        )
        if should_print_batch:
            if show_progress_bar:
                iterator.set_postfix(
                    loss="{:.4f}".format(loss.item()),
                    dice="{:.4f}".format(batch_metrics["dice"]),
                )
            print(
                "[{}][epoch {}][batch {}/{}] loss={:.4f} dice={:.4f}".format(
                    phase_name,
                    epoch,
                    batch_index,
                    len(loader),
                    loss.item(),
                    batch_metrics["dice"],
                ),
                flush=True,
            )

    num_batches = max(len(loader), 1)
    results = {
        "loss": total_loss / num_batches,
        "loss_cls": total_cls / num_batches,
        "loss_dice": total_dice_loss / num_batches,
    }
    for key in total_metrics:
        results[key] = total_metrics[key] / num_batches
    return results


def _compute_tile_starts(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, max(stride, 1)))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def validate_sliding_window(
    model: nn.Module,
    val_samples: List[Dict[str, str]],
    patch_size: Sequence[int],
    num_classes: int,
    cls_loss_fn: nn.Module,
    dice_loss: DiceLoss,
    device: torch.device,
    ce_weight: float,
    dice_weight: float,
    overlap: float,
    phase_name: str,
    epoch: int,
    show_progress_bar: bool,
) -> Dict[str, float]:
    model.eval()
    tile_d, tile_h, tile_w = [int(p) for p in patch_size]
    stride_d = max(1, int(round(tile_d * (1.0 - overlap))))
    stride_h = max(1, int(round(tile_h * (1.0 - overlap))))
    stride_w = max(1, int(round(tile_w * (1.0 - overlap))))

    dice_scores, iou_scores, prec_scores, recall_scores = [], [], [], []
    loss_list, loss_cls_list, loss_dice_list = [], [], []

    iterator = tqdm(val_samples, desc="{} epoch {}".format(phase_name, epoch), leave=False) if show_progress_bar else val_samples
    for sample in iterator:
        image_np = normalize_volume(read_volume(Path(sample["image_path"])))
        mask_np = prepare_mask(read_volume(Path(sample["mask_path"])), num_classes)

        d, h, w = image_np.shape
        pad_d = max(tile_d - d, 0)
        pad_h = max(tile_h - h, 0)
        pad_w = max(tile_w - w, 0)
        if pad_d or pad_h or pad_w:
            image_np = np.pad(image_np, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant")
            mask_np = np.pad(mask_np, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant")
            d, h, w = image_np.shape

        image_t = torch.from_numpy(image_np[None, None, ...].astype(np.float32)).to(device)
        logits_acc = torch.zeros((1, num_classes, d, h, w), dtype=torch.float32, device=device)
        count_acc = torch.zeros((1, 1, d, h, w), dtype=torch.float32, device=device)

        for z0 in _compute_tile_starts(d, tile_d, stride_d):
            for y0 in _compute_tile_starts(h, tile_h, stride_h):
                for x0 in _compute_tile_starts(w, tile_w, stride_w):
                    tile = image_t[:, :, z0:z0 + tile_d, y0:y0 + tile_h, x0:x0 + tile_w]
                    with torch.no_grad():
                        with autocast_context(device_type=device.type, enabled=False):
                            out = model(tile)
                    logits_acc[:, :, z0:z0 + tile_d, y0:y0 + tile_h, x0:x0 + tile_w] += out.float()
                    count_acc[:, :, z0:z0 + tile_d, y0:y0 + tile_h, x0:x0 + tile_w] += 1.0

        logits_full = logits_acc / count_acc.clamp(min=1.0)
        mask_t = torch.from_numpy(mask_np.astype(np.int64))[None, ...].to(device)

        with torch.no_grad():
            loss_cls = cls_loss_fn(logits_full, mask_t)
            loss_dice_val = dice_loss(logits_full, mask_t)
            loss = ce_weight * loss_cls + dice_weight * loss_dice_val

        m = compute_metrics(logits_full, mask_t, num_classes)
        dice_scores.append(m["dice"])
        iou_scores.append(m["iou"])
        prec_scores.append(m["precision"])
        recall_scores.append(m["recall"])
        loss_list.append(loss.item())
        loss_cls_list.append(loss_cls.item())
        loss_dice_list.append(loss_dice_val.item())

        del image_t, logits_acc, count_acc, logits_full, mask_t

    return {
        "loss": float(np.mean(loss_list)) if loss_list else 0.0,
        "loss_cls": float(np.mean(loss_cls_list)) if loss_cls_list else 0.0,
        "loss_dice": float(np.mean(loss_dice_list)) if loss_dice_list else 0.0,
        "dice": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "iou": float(np.mean(iou_scores)) if iou_scores else 0.0,
        "precision": float(np.mean(prec_scores)) if prec_scores else 0.0,
        "recall": float(np.mean(recall_scores)) if recall_scores else 0.0,
    }


# =====================================================================
#  Main Entry Point
# =====================================================================

def save_json(path: Path, data: Dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def print_startup_summary(
    args: argparse.Namespace,
    device: torch.device,
    train_samples: List[Dict[str, str]],
    val_samples: List[Dict[str, str]],
    train_dataset: VolumeDataset,
) -> None:
    print("==== Startup Summary ====", flush=True)
    print("device: {}".format(device), flush=True)
    print("torch.cuda.is_available(): {}".format(torch.cuda.is_available()), flush=True)
    print("torch.version.cuda: {}".format(torch.version.cuda), flush=True)
    if torch.cuda.is_available():
        print("gpu_count: {}".format(torch.cuda.device_count()), flush=True)
        print("gpu_name: {}".format(torch.cuda.get_device_name(0)), flush=True)
    else:
        print("warning: CUDA is not available, so training will run on CPU and nvidia-smi will show nothing.", flush=True)

    print("train_cases: {} | val_cases: {}".format(len(train_samples), len(val_samples)), flush=True)
    print(
        "batch_size: {} | accumulate_steps: {} | num_workers: {} | patch_size: {} | warmup_epochs: {} | min_lr_ratio: {}".format(
            args.batch_size,
            args.accumulate_steps,
            args.num_workers,
            tuple(args.patch_size),
            args.warmup_epochs,
            args.min_lr_ratio,
        ),
        flush=True,
    )
    print(
        "cls_loss: {} | focal_gamma: {} | focal_alpha: {} | grad_clip_norm: {}".format(
            args.cls_loss, args.focal_gamma, args.focal_alpha, args.grad_clip_norm
        ),
        flush=True,
    )
    print(
        "p_spatial_aug: {} | p_intensity_aug: {} | fg_aware_crop_prob: {} | use_fg_aware_sampler: {}".format(
            args.p_spatial_aug, args.p_intensity_aug, args.fg_aware_crop_prob, args.use_fg_aware_sampler
        ),
        flush=True,
    )
    print(
        "fg_sampler_power: {} | fg_sampler_min_weight: {}".format(
            args.fg_sampler_power, args.fg_sampler_min_weight
        ),
        flush=True,
    )
    print("loading first training sample for shape check...", flush=True)
    sample = train_dataset[0]
    print(
        "first_sample image_shape={} mask_shape={} case_id={}".format(
            tuple(sample["image"].shape),
            tuple(sample["mask"].shape),
            sample["case_id"],
        ),
        flush=True,
    )
    print("=========================", flush=True)


def main() -> None:
    args = parse_args()
    ensure_dependencies()
    seed_everything(args.seed)

    image_dir = Path(args.image_dir) if args.image_dir else Path(args.data_root) / "image"
    mask_dir = Path(args.mask_dir) if args.mask_dir else Path(args.data_root) / "mask"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(image_dir, mask_dir)
    train_samples, val_samples = split_samples(samples, args.val_ratio, args.seed)

    if not val_samples:
        raise RuntimeError("Validation split is empty. Please increase the dataset size or val ratio.")

    train_dataset = VolumeDataset(
        samples=train_samples,
        patch_size=args.patch_size,
        num_classes=args.num_classes,
        training=True,
        cache_data=args.cache_data,
        p_spatial_aug=args.p_spatial_aug,
        p_intensity_aug=args.p_intensity_aug,
        fg_aware_crop_prob=args.fg_aware_crop_prob,
        patches_per_volume=args.patches_per_volume,
    )
    val_dataset = VolumeDataset(
        samples=val_samples,
        patch_size=args.patch_size,
        num_classes=args.num_classes,
        training=False,
        cache_data=args.cache_data,
        p_spatial_aug=0.0,
        p_intensity_aug=0.0,
        fg_aware_crop_prob=0.0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and not args.no_amp
    print_startup_summary(args, device, train_samples, val_samples, train_dataset)

    train_sampler = None
    train_shuffle = True
    if args.use_fg_aware_sampler:
        sampling_weights, fg_ratios = build_fg_aware_sampling_weights(
            samples=train_samples,
            num_classes=args.num_classes,
            power=args.fg_sampler_power,
            min_weight=args.fg_sampler_min_weight,
        )
        dataset_size = len(train_samples) * max(1, args.patches_per_volume)
        tiled_weights = np.tile(sampling_weights, max(1, args.patches_per_volume))
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(tiled_weights, dtype=torch.double),
            num_samples=dataset_size,
            replacement=True,
        )
        train_shuffle = False
        print(
            "fg-aware sampler enabled: fg_ratio train mean={:.6f}, median={:.6f}, p10={:.6f}, p90={:.6f}".format(
                float(np.mean(fg_ratios)),
                float(np.median(fg_ratios)),
                float(np.percentile(fg_ratios, 10)),
                float(np.percentile(fg_ratios, 90)),
            ),
            flush=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=torch.cuda.is_available(),
    )

    model = SimpleUNet3D(in_channels=1, num_classes=args.num_classes, base_channels=args.base_channels).to(device)
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs (DataParallel)".format(torch.cuda.device_count()), flush=True)
        model = nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.cls_loss == "focal":
        cls_loss_fn = FocalLoss(
            num_classes=args.num_classes,
            gamma=args.focal_gamma,
            alpha=args.focal_alpha,
        ).to(device)
    else:
        cls_loss_fn = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes=args.num_classes)

    run_name = args.run_name or "cfos3d_{}".format(time.strftime("%Y%m%d_%H%M%S"))
    train_cases = [sample["case_id"] for sample in train_samples]
    val_cases = [sample["case_id"] for sample in val_samples]

    split_path = output_dir / "data_split.json"
    save_json(split_path, {"train_cases": train_cases, "val_cases": val_cases})
    config_path = output_dir / "train_config.json"
    save_json(config_path, vars(args))

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    best_dice = -1.0
    best_ckpt_path = output_dir / "best_model.pt"
    last_ckpt_path = output_dir / "last_model.pt"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "data_root": str(Path(args.data_root).resolve()),
                "image_dir": str(image_dir.resolve()),
                "mask_dir": str(mask_dir.resolve()),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "cls_loss": args.cls_loss,
                "focal_gamma": args.focal_gamma,
                "focal_alpha": args.focal_alpha,
                "warmup_epochs": args.warmup_epochs,
                "min_lr_ratio": args.min_lr_ratio,
                "val_ratio": args.val_ratio,
                "seed": args.seed,
                "num_classes": args.num_classes,
                "base_channels": args.base_channels,
                "grad_clip_norm": args.grad_clip_norm,
                "patch_size": "x".join(str(v) for v in args.patch_size),
                "train_cases": len(train_cases),
                "val_cases": len(val_cases),
                "amp": use_amp,
                "accumulate_steps": args.accumulate_steps,
                "p_spatial_aug": args.p_spatial_aug,
                "p_intensity_aug": args.p_intensity_aug,
                "fg_aware_crop_prob": args.fg_aware_crop_prob,
                "use_fg_aware_sampler": args.use_fg_aware_sampler,
                "fg_sampler_power": args.fg_sampler_power,
                "fg_sampler_min_weight": args.fg_sampler_min_weight,
                "patches_per_volume": args.patches_per_volume,
                "val_mode": args.val_mode,
                "val_overlap": args.val_overlap,
            }
        )
        mlflow.log_artifact(str(config_path))
        mlflow.log_artifact(str(split_path))

        for epoch in range(1, args.epochs + 1):
            current_lr = compute_lr_for_epoch(
                epoch=epoch,
                total_epochs=args.epochs,
                base_lr=args.lr,
                warmup_epochs=args.warmup_epochs,
                min_lr_ratio=args.min_lr_ratio,
            )
            set_optimizer_lr(optimizer, current_lr)
            train_stats = run_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                cls_loss_fn=cls_loss_fn,
                dice_loss=dice_loss,
                device=device,
                num_classes=args.num_classes,
                ce_weight=args.ce_weight,
                dice_weight=args.dice_weight,
                use_amp=use_amp,
                accumulate_steps=args.accumulate_steps,
                phase_name="train",
                epoch=epoch,
                log_every_n_batches=args.log_every_n_batches,
                show_progress_bar=args.show_progress_bar,
                grad_clip_norm=args.grad_clip_norm,
            )
            if args.val_mode == "sliding":
                val_stats = validate_sliding_window(
                    model=model,
                    val_samples=val_samples,
                    patch_size=args.patch_size,
                    num_classes=args.num_classes,
                    cls_loss_fn=cls_loss_fn,
                    dice_loss=dice_loss,
                    device=device,
                    ce_weight=args.ce_weight,
                    dice_weight=args.dice_weight,
                    overlap=args.val_overlap,
                    phase_name="val_sw",
                    epoch=epoch,
                    show_progress_bar=args.show_progress_bar,
                )
            else:
                val_stats = run_epoch(
                    model=model,
                    loader=val_loader,
                    optimizer=None,
                    cls_loss_fn=cls_loss_fn,
                    dice_loss=dice_loss,
                    device=device,
                    num_classes=args.num_classes,
                    ce_weight=args.ce_weight,
                    dice_weight=args.dice_weight,
                    use_amp=False,
                    accumulate_steps=1,
                    phase_name="val",
                    epoch=epoch,
                    log_every_n_batches=args.log_every_n_batches,
                    show_progress_bar=args.show_progress_bar,
                    grad_clip_norm=0.0,
                )

            mlflow.log_metrics(
                {
                    "train_loss": train_stats["loss"],
                    "train_loss_cls": train_stats["loss_cls"],
                    "train_loss_ce": train_stats["loss_cls"],
                    "train_loss_dice": train_stats["loss_dice"],
                    "train_dice": train_stats["dice"],
                    "train_iou": train_stats["iou"],
                    "val_loss": val_stats["loss"],
                    "val_loss_cls": val_stats["loss_cls"],
                    "val_loss_ce": val_stats["loss_cls"],
                    "val_loss_dice": val_stats["loss_dice"],
                    "val_dice": val_stats["dice"],
                    "val_iou": val_stats["iou"],
                    "val_precision": val_stats["precision"],
                    "val_recall": val_stats["recall"],
                    "lr": current_lr,
                },
                step=epoch,
            )

            print(
                "Epoch [{}/{}] "
                "train_loss={:.4f} train_dice={:.4f} "
                "val_loss={:.4f} val_dice={:.4f}".format(
                    epoch,
                    args.epochs,
                    train_stats["loss"],
                    train_stats["dice"],
                    val_stats["loss"],
                    val_stats["dice"],
                )
            )

            if val_stats["dice"] > best_dice:
                best_dice = val_stats["dice"]
                is_best = True
            else:
                is_best = False

            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_dice": best_dice,
                "args": vars(args),
            }
            torch.save(checkpoint, str(last_ckpt_path))

            if is_best:
                torch.save(checkpoint, str(best_ckpt_path))

            if epoch % max(args.save_every, 1) == 0:
                epoch_ckpt = output_dir / "epoch_{:03d}.pt".format(epoch)
                torch.save(checkpoint, str(epoch_ckpt))

        mlflow.log_metric("best_val_dice", best_dice)
        mlflow.log_artifact(str(best_ckpt_path), artifact_path="checkpoints")
        mlflow.log_artifact(str(last_ckpt_path), artifact_path="checkpoints")

    print("Training finished. Best val dice: {:.4f}".format(best_dice))
    print("Best checkpoint: {}".format(best_ckpt_path.resolve()))
    print("MLflow tracking URI: {}".format(args.tracking_uri))


if __name__ == "__main__":
    main()
