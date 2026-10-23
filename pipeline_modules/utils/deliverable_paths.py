"""Canonical deliverable paths aligned with Arivis_Analysis naming rules.

Excel:  {sample}/results/{sample}_{channel}_brain_distribution_stats.xlsx
2D map: {sample}/visualization/{sample}_{channel}_heatmap_2d/
3D map: {sample}/visualization/{sample}_{channel}_heatmap_3d.png
"""
from __future__ import annotations

import re
from pathlib import Path


def normalize_channel(channel: str) -> str:
    if not channel or channel in ("-", "all"):
        return "all"
    text = channel.strip().lower().replace(" ", "")
    if text == "all":
        return "all"
    parts = re.split(r"[_+&,]", text)
    ch_nums: list[int] = []
    for part in parts:
        if not part:
            continue
        match = re.fullmatch(r"ch?(\d+)", part)
        if match:
            ch_nums.append(int(match.group(1)))
        else:
            return text.replace("+", "_")
    if not ch_nums:
        return text.replace("+", "_")
    if len(ch_nums) == 1:
        return f"ch{ch_nums[0]}"
    return "ch" + "_ch".join(str(n) for n in sorted(set(ch_nums)))


def sample_slug(sample_dir: str | Path) -> str:
    return Path(sample_dir).name


def results_dir(sample_dir: str | Path) -> Path:
    return Path(sample_dir) / "results"


def visualization_dir(sample_dir: str | Path) -> Path:
    return Path(sample_dir) / "visualization"


def brain_distribution_stats_xlsx(sample_dir: str | Path, channel: str) -> Path:
    channel_token = normalize_channel(channel)
    slug = sample_slug(sample_dir)
    return results_dir(sample_dir) / f"{slug}_{channel_token}_brain_distribution_stats.xlsx"


def find_stats_workbook(sample_dir: str | Path) -> Path:
    """Newest region-stats workbook for one sample.

    Order: atlas-v2 standard (results/*_brain_distribution_stats_v2.xlsx),
    main.py standard (results/*_brain_distribution_stats.xlsx), then legacy
    root-level *_result.xlsx from pre-results runs.
    """
    sample_dir = Path(sample_dir)
    if not sample_dir.is_dir():
        raise NotADirectoryError(f"Sample directory not found: {sample_dir}")
    searches = [
        (sample_dir / "results", "*_brain_distribution_stats_v2.xlsx"),
        (sample_dir / "results", "*_brain_distribution_stats.xlsx"),
        (sample_dir, "*_result.xlsx"),
    ]
    for directory, pattern in searches:
        matches = sorted(
            (path for path in directory.glob(pattern) if not path.name.startswith("~$")),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"No region stats workbook under {sample_dir}: expected "
        "results/*_brain_distribution_stats[_v2].xlsx or a legacy *_result.xlsx"
    )


def has_stats_workbook(sample_dir: str | Path) -> bool:
    try:
        find_stats_workbook(sample_dir)
    except (FileNotFoundError, NotADirectoryError):
        return False
    return True


def spinal_segment_stats_xlsx(sample_dir: str | Path, channel: str) -> Path:
    channel_token = normalize_channel(channel)
    slug = sample_slug(sample_dir)
    return results_dir(sample_dir) / f"{slug}_{channel_token}_spinal_segment_stats.xlsx"


def spinal_segment_stats_csv(sample_dir: str | Path, channel: str) -> Path:
    return spinal_segment_stats_xlsx(sample_dir, channel).with_suffix(".csv")


def spinal_segment_region_csv(sample_dir: str | Path) -> Path:
    return results_dir(sample_dir) / f"{sample_slug(sample_dir)}_spinal_segment_regions.csv"


def heatmap_2d_dir(sample_dir: str | Path, channel: str) -> Path:
    channel_token = normalize_channel(channel)
    slug = sample_slug(sample_dir)
    return visualization_dir(sample_dir) / f"{slug}_{channel_token}_heatmap_2d"


def heatmap_3d_png(sample_dir: str | Path, channel: str) -> Path:
    channel_token = normalize_channel(channel)
    slug = sample_slug(sample_dir)
    return visualization_dir(sample_dir) / f"{slug}_{channel_token}_heatmap_3d.png"


def heatmap_3d_stack_tiff(sample_dir: str | Path, channel: str) -> Path:
    channel_token = normalize_channel(channel)
    slug = sample_slug(sample_dir)
    return visualization_dir(sample_dir) / f"{slug}_{channel_token}_heatmap_3d_stack.tiff"


def heatmap_3d_volume_tiff(sample_dir: str | Path, channel: str) -> Path:
    channel_token = normalize_channel(channel)
    slug = sample_slug(sample_dir)
    return visualization_dir(sample_dir) / f"{slug}_{channel_token}_heatmap_3d_volume.tiff"


def heatmap_3d_colorbar_png(sample_dir: str | Path, channel: str) -> Path:
    channel_token = normalize_channel(channel)
    slug = sample_slug(sample_dir)
    return visualization_dir(sample_dir) / f"{slug}_{channel_token}_heatmap_3d_colorbar.png"


def heatmap_3d_summary_json(sample_dir: str | Path, channel: str) -> Path:
    return heatmap_3d_png(sample_dir, channel).with_suffix(".json")


def legacy_brain_distribution_candidates(sample_dir: str | Path, channel: str | None = None) -> list[Path]:
    root = Path(sample_dir)
    slug = sample_slug(root)
    channel_token = normalize_channel(channel) if channel else None
    candidates = [
        brain_distribution_stats_xlsx(root, channel or "ch1"),
        root / f"sample_{channel_token}_result.xlsx" if channel_token else None,
        root / f"{slug}_density_result.xlsx",
        root / f"{slug}_result.xlsx",
        root / "density_results_ch1.xlsx",
    ]
    if channel_token:
        candidates.append(root / f"density_results_{channel_token}.xlsx")
    candidates.extend(sorted(root.glob("*density*.xlsx")))
    candidates.extend(sorted(root.glob("*brain_distribution*.xlsx")))
    candidates.extend(sorted((root / "results").glob("*.xlsx")) if (root / "results").is_dir() else [])
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in candidates:
        if path is None:
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(path)
    return ordered


def legacy_heatmap_3d_volume_candidates(sample_dir: str | Path, channel: str | None = None) -> list[Path]:
    root = Path(sample_dir)
    slug = sample_slug(root)
    candidates: list[Path] = []
    if channel:
        candidates.append(heatmap_3d_volume_tiff(root, channel))
    candidates.extend(sorted((root / "visualization").glob(f"{slug}_*_heatmap_3d_volume.tiff")))
    candidates.append(root / "visualization" / f"{slug}_heatmap3d_volume.tiff")
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(path)
    return ordered
