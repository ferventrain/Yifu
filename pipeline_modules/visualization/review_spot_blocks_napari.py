"""Browse cFos review blocks in napari: image, GT mask, RS-FISH points.

Keys: n next block, p previous, s save current points CSV (after you edit the points layer).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import PIL.Image  # noqa: F401
import tifffile


def _load_points(csv_path: Path) -> np.ndarray:
    if not csv_path.exists():
        return np.zeros((0, 3), dtype=np.float32)
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((float(row["z"]), float(row["y"]), float(row["x"])))
    if not rows:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _save_points(csv_path: Path, points: np.ndarray) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["z", "y", "x"])
        writer.writeheader()
        for z, y, x in np.asarray(points, dtype=np.float32).reshape(-1, 3):
            writer.writerow({"z": f"{z:.4f}", "y": f"{y:.4f}", "x": f"{x:.4f}"})


def run_viewer(root: Path) -> int:
    import napari

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cases = [str(item["case_id"]) for item in manifest["cases"]]
    image_dir = root / "image"
    mask_dir = root / "mask_gt"
    spots_dir = root / "rs_fish"
    reviewed_dir = root / "reviewed_points"
    reviewed_dir.mkdir(exist_ok=True)

    state = {"index": 0}

    viewer = napari.Viewer(title="cFos RS-FISH review")
    image_layer = viewer.add_image(np.zeros((8, 8, 8), dtype=np.uint16), name="image", colormap="gray")
    mask_layer = viewer.add_labels(np.zeros((8, 8, 8), dtype=np.uint8), name="gt_mask", opacity=0.35)
    points_layer = viewer.add_points(
        np.zeros((0, 3), dtype=np.float32),
        name="rs_fish",
        size=4.0,
        ndim=3,
        face_color="magenta",
        border_color="white",
    )

    def case_id() -> str:
        return cases[state["index"]]

    def load_current() -> None:
        cid = case_id()
        img = tifffile.imread(str(image_dir / f"{cid}.tiff"))
        mask_path = mask_dir / f"{cid}_mask.tiff"
        mask = tifffile.imread(str(mask_path)) if mask_path.exists() else np.zeros(img.shape, dtype=np.uint8)
        reviewed = reviewed_dir / f"{cid}.csv"
        src = reviewed if reviewed.exists() else spots_dir / f"{cid}.csv"
        pts = _load_points(src)
        image_layer.data = img
        image_layer.contrast_limits = (
            float(np.percentile(img, 1)),
            float(np.percentile(img, 99.8)),
        )
        mask_layer.data = (mask > 0).astype(np.uint8)
        points_layer.data = pts
        viewer.title = f"cFos review [{state['index']+1}/{len(cases)}] {cid}  spots={len(pts)}"
        print(viewer.title, f"loaded {src.name}", flush=True)

    def save_current() -> None:
        cid = case_id()
        out = reviewed_dir / f"{cid}.csv"
        _save_points(out, np.asarray(points_layer.data))
        print(f"saved {out} n={len(points_layer.data)}", flush=True)

    def next_case() -> None:
        save_current()
        state["index"] = (state["index"] + 1) % len(cases)
        load_current()

    def prev_case() -> None:
        save_current()
        state["index"] = (state["index"] - 1) % len(cases)
        load_current()

    viewer.bind_key("n")(lambda _v: next_case())
    viewer.bind_key("p")(lambda _v: prev_case())
    viewer.bind_key("s")(lambda _v: save_current())
    load_current()
    print("Keys: n next, p prev, s save points. Add/delete points on layer 'rs_fish'.", flush=True)
    napari.run()
    save_current()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review RS-FISH spots on cFos blocks in napari.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"S:\Yifu_data\datasets\cfos_review_v1"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_viewer(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
