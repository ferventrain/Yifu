from __future__ import annotations

from pathlib import Path
from typing import Any


def normalize_volume(volume, low_pct: float = 1.0, high_pct: float = 99.5):
    import numpy as np

    volume = volume.astype(np.float32, copy=False)
    low, high = np.percentile(volume, (float(low_pct), float(high_pct)))
    if high <= low:
        volume = volume - volume.min()
        denom = volume.max()
        return volume / denom if denom > 0 else volume
    volume = np.clip(volume, low, high)
    volume = volume - low
    volume = volume / max(high - low, 1e-6)
    return volume


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "torch is required for cfos_unet segmentation inference"
        ) from exc
    return torch, nn, F


def build_cfos_unet_classes():
    torch, nn, F = _require_torch()

    def match_tensor_shape(source, target):
        if source.shape[-3:] == target.shape[-3:]:
            return source
        return F.interpolate(source, size=target.shape[-3:], mode="trilinear", align_corners=False)

    class ConvBlock3D(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
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

        def forward(self, x):
            return self.act(self.block(x) + self.skip(x))

    class AttentionGate3D(nn.Module):
        def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
            super().__init__()
            self.W_gate = nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=True)
            self.W_skip = nn.Conv3d(skip_channels, inter_channels, kernel_size=1, bias=True)
            self.psi = nn.Sequential(
                nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, gate, skip):
            g = self.W_gate(gate)
            s = self.W_skip(skip)
            g = match_tensor_shape(g, s)
            alpha = self.psi(self.relu(g + s))
            return skip * alpha

    class SimpleUNet3D(nn.Module):
        def __init__(self, in_channels: int = 1, num_classes: int = 2, base_channels: int = 8) -> None:
            super().__init__()
            c1 = base_channels
            c2 = base_channels * 2
            c3 = base_channels * 4
            c4 = base_channels * 8
            c5 = base_channels * 16

            self.enc1 = ConvBlock3D(in_channels, c1)
            self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
            self.enc2 = ConvBlock3D(c1, c2)
            self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
            self.enc3 = ConvBlock3D(c2, c3)
            self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
            self.enc4 = ConvBlock3D(c3, c4)
            self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
            self.enc5 = ConvBlock3D(c4, c5)

            self.up4 = nn.ConvTranspose3d(c5, c4, kernel_size=2, stride=2)
            self.ag4 = AttentionGate3D(gate_channels=c4, skip_channels=c4, inter_channels=max(1, c4 // 2))
            self.dec4 = ConvBlock3D(c4 + c4, c4)

            self.up3 = nn.ConvTranspose3d(c4, c3, kernel_size=2, stride=2)
            self.ag3 = AttentionGate3D(gate_channels=c3, skip_channels=c3, inter_channels=max(1, c3 // 2))
            self.dec3 = ConvBlock3D(c3 + c3, c3)

            self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
            self.ag2 = AttentionGate3D(gate_channels=c2, skip_channels=c2, inter_channels=max(1, c2 // 2))
            self.dec2 = ConvBlock3D(c2 + c2, c2)

            self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
            self.ag1 = AttentionGate3D(gate_channels=c1, skip_channels=c1, inter_channels=max(1, c1 // 2))
            self.dec1 = ConvBlock3D(c1 + c1, c1)

            self.head = nn.Conv3d(c1, num_classes, kernel_size=1)

        def forward(self, x):
            s1 = self.enc1(x)
            s2 = self.enc2(self.pool1(s1))
            s3 = self.enc3(self.pool2(s2))
            s4 = self.enc4(self.pool3(s3))
            btm = self.enc5(self.pool4(s4))

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

    return torch, SimpleUNet3D


def load_cfos_unet_checkpoint(checkpoint_path: str | Path, device: str = "cpu"):
    torch, SimpleUNet3D = build_cfos_unet_classes()
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    args = checkpoint.get("args", {}) or {}
    base_channels = int(args.get("base_channels", 8))
    num_classes = int(args.get("num_classes", 2))
    model = SimpleUNet3D(in_channels=1, num_classes=num_classes, base_channels=base_channels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return {
        "model": model,
        "checkpoint": checkpoint,
        "checkpoint_args": args,
        "base_channels": base_channels,
        "num_classes": num_classes,
        "torch": torch,
    }
