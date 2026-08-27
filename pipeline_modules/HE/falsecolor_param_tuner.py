"""
Interactive / grid preview for falsecolor colleague-style parameters.

Sweeps or interactively tunes:
  - flatfield_nuc_scale
  - flatfield_cyto_scale
  - bg_nuc_factor
  - bg_cyto_factor
  - sharpen_alpha

Example (interactive, middle slice, center crop for speed):
  python pipeline_modules/visualization/falsecolor_param_tuner.py --middle 1 --crop 1536

Example (save parameter grid):
  python pipeline_modules/visualization/falsecolor_param_tuner.py --middle 1 --crop 1536 --mode grid --output_dir "D:/fc_tune_grid"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from skimage.io import imread, imsave
from skimage.transform import downscale_local_mean

from pipeline_modules.HE.falsecolor_batch import (
    _gpu_available,
    apply_background_mask,
    discover_channel_pairs,
    preprocess_colleague_tricks,
    render_falsecolor,
    select_slice_pairs,
)


def parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated float list")
    return values


def center_crop(image: np.ndarray, size: int) -> np.ndarray:
    if size <= 0:
        return image
    height, width = image.shape[:2]
    crop = min(size, height, width)
    r0 = max(0, (height - crop) // 2)
    c0 = max(0, (width - crop) // 2)
    return image[r0 : r0 + crop, c0 : c0 + crop]


def maybe_downscale(image: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return image
    if image.ndim == 2:
        return downscale_local_mean(image, (factor, factor)).astype(image.dtype, copy=False)
    return downscale_local_mean(image, (factor, factor, 1)).astype(image.dtype, copy=False)


def render_preview(
    nuclei_raw: np.ndarray,
    cyto_raw: np.ndarray,
    *,
    flatfield_nuc_scale: float,
    flatfield_cyto_scale: float,
    bg_nuc_factor: float,
    bg_cyto_factor: float,
    sharpen_alpha: float,
    args: argparse.Namespace,
) -> np.ndarray:
    nuclei, cyto = preprocess_colleague_tricks(
        nuclei_raw,
        cyto_raw,
        nuc_threshold=args.nuc_threshold,
        cyto_threshold=args.cyto_threshold,
        flatfield=True,
        bg_nuc_factor=bg_nuc_factor,
        bg_cyto_factor=bg_cyto_factor,
        sharpen_alpha=sharpen_alpha,
        clahe=args.clahe,
        clahe_clip_limit=args.clahe_clip_limit,
    )
    rgb = render_falsecolor(
        nuclei,
        cyto,
        backend=args.backend,
        color_key=args.color_key,
        nuclei_hue=args.nuclei_hue,
        nuc_threshold=args.nuc_threshold,
        cyto_threshold=args.cyto_threshold,
        nuc_normfactor=args.nuc_normfactor,
        cyto_normfactor=args.cyto_normfactor,
        flatfield=True,
        flatfield_tile_size=args.flatfield_tile_size,
        flatfield_scale=args.flatfield_scale,
        flatfield_nuc_scale=flatfield_nuc_scale,
        flatfield_cyto_scale=flatfield_cyto_scale,
    )
    if args.mask_background != "none":
        rgb = apply_background_mask(
            rgb,
            nuclei_raw,
            cyto_raw,
            mode=args.mask_background,
            background_threshold=args.background_threshold,
            white_threshold=args.white_threshold,
            pale_signal_threshold=args.pale_signal_threshold,
            hsv_mask_val=0.05,
            hsv_min_size=150,
        )
    return rgb


def load_preview_pair(args: argparse.Namespace) -> tuple[str, np.ndarray, np.ndarray]:
    if args.nuclei_tiff is not None and args.cyto_tiff is not None:
        nuclei_path = args.nuclei_tiff.resolve()
        cyto_path = args.cyto_tiff.resolve()
        prefix = nuclei_path.name
        match = __import__("re").match(
            r"^(?P<prefix>.+)_ch\d+\.tiff?$", nuclei_path.name, __import__("re").I
        )
        if match:
            prefix = match.group("prefix")
        print(f"Using explicit TIFF pair prefix={prefix}", flush=True)
    else:
        input_dir = args.input_dir.resolve()
        print(f"Scanning {input_dir} ...", flush=True)
        t0 = time.time()
        pairs = discover_channel_pairs(
            input_dir,
            nuclei_channel=args.nuclei_channel,
            cyto_channel=args.cyto_channel,
        )
        print(f"Found {len(pairs)} pairs in {time.time() - t0:.1f}s", flush=True)
        if not pairs:
            raise FileNotFoundError(f"No paired TIFF channels in {input_dir}")

        if args.prefix:
            matched = [item for item in pairs if item[0] == args.prefix]
            if not matched:
                raise FileNotFoundError(f"Prefix not found: {args.prefix}")
            prefix, nuclei_path, cyto_path = matched[0]
            index = pairs.index(matched[0])
            print(f"Using slice [{index}] prefix={prefix}", flush=True)
        else:
            selected, start, end = select_slice_pairs(
                pairs,
                slice_start=args.slice_start,
                slice_count=1,
                middle=max(args.middle, 0),
            )
            if not selected:
                selected = [pairs[len(pairs) // 2]]
                start = pairs.index(selected[0])
                end = start + 1
            prefix, nuclei_path, cyto_path = selected[0]
            print(f"Using slice [{start}:{end}] prefix={prefix}", flush=True)

    print(f"  nuclei: {nuclei_path}", flush=True)
    print(f"  cyto:   {cyto_path}", flush=True)
    print("Reading TIFFs ...", flush=True)
    t0 = time.time()
    nuclei = imread(str(nuclei_path))
    cyto = imread(str(cyto_path))
    print(f"Read done in {time.time() - t0:.1f}s, raw shape={nuclei.shape}", flush=True)
    nuclei = center_crop(nuclei, args.crop)
    cyto = center_crop(cyto, args.crop)
    nuclei = maybe_downscale(nuclei, args.downscale)
    cyto = maybe_downscale(cyto, args.downscale)
    print(f"Preview shape: {nuclei.shape}, backend={args.backend}", flush=True)
    return prefix, nuclei, cyto


def run_interactive(prefix: str, nuclei: np.ndarray, cyto: np.ndarray, args: argparse.Namespace) -> int:
    state = {
        "nuc_scale": args.init_nuc_scale,
        "cyto_scale": args.init_cyto_scale,
        "bg_nuc": args.init_bg_nuc_factor,
        "bg_cyto": args.init_bg_cyto_factor,
        "sharpen_alpha": args.init_sharpen_alpha,
        "busy": False,
    }

    fig, ax = plt.subplots(figsize=(9, 9.5))
    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.38)
    image_artist = ax.imshow(np.zeros((*nuclei.shape, 3), dtype=np.uint8))
    ax.set_axis_off()
    title = ax.set_title(prefix)

    ax_nuc = plt.axes([0.14, 0.28, 0.62, 0.03])
    ax_cyto = plt.axes([0.14, 0.23, 0.62, 0.03])
    ax_bgn = plt.axes([0.14, 0.18, 0.62, 0.03])
    ax_bgc = plt.axes([0.14, 0.13, 0.62, 0.03])
    ax_sh = plt.axes([0.14, 0.08, 0.62, 0.03])
    ax_btn = plt.axes([0.82, 0.14, 0.14, 0.08])

    s_nuc = Slider(ax_nuc, "nuc_scale", 0.2, 8.0, valinit=state["nuc_scale"], valstep=0.05)
    s_cyto = Slider(ax_cyto, "cyto_scale", 0.2, 10.0, valinit=state["cyto_scale"], valstep=0.05)
    s_bgn = Slider(ax_bgn, "bg_nuc", 0.0, 6.0, valinit=state["bg_nuc"], valstep=0.05)
    s_bgc = Slider(ax_bgc, "bg_cyto", 0.0, 6.0, valinit=state["bg_cyto"], valstep=0.05)
    s_sh = Slider(ax_sh, "sharpen", 0.0, 2.0, valinit=state["sharpen_alpha"], valstep=0.05)
    btn = Button(ax_btn, "Render")

    status = fig.text(0.12, 0.02, "", fontsize=10)

    def do_render(_event=None) -> None:
        if state["busy"]:
            return
        state["busy"] = True
        state["nuc_scale"] = float(s_nuc.val)
        state["cyto_scale"] = float(s_cyto.val)
        state["bg_nuc"] = float(s_bgn.val)
        state["bg_cyto"] = float(s_bgc.val)
        state["sharpen_alpha"] = float(s_sh.val)
        status.set_text("Rendering...")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        try:
            t0 = time.time()
            rgb = render_preview(
                nuclei,
                cyto,
                flatfield_nuc_scale=state["nuc_scale"],
                flatfield_cyto_scale=state["cyto_scale"],
                bg_nuc_factor=state["bg_nuc"],
                bg_cyto_factor=state["bg_cyto"],
                sharpen_alpha=state["sharpen_alpha"],
                args=args,
            )
            image_artist.set_data(rgb)
            title.set_text(
                f"{prefix} | nuc={state['nuc_scale']:.2f} cyto={state['cyto_scale']:.2f} "
                f"bgN={state['bg_nuc']:.2f} bgC={state['bg_cyto']:.2f} "
                f"sh={state['sharpen_alpha']:.2f} ({time.time() - t0:.1f}s)"
            )
            status.set_text("Done. Drag sliders then click Render (or press R).")
            if args.autosave_dir is not None:
                out_dir = args.autosave_dir.resolve()
                out_dir.mkdir(parents=True, exist_ok=True)
                name = (
                    f"{prefix}_nuc{state['nuc_scale']:.2f}_cyto{state['cyto_scale']:.2f}"
                    f"_bgN{state['bg_nuc']:.2f}_bgC{state['bg_cyto']:.2f}"
                    f"_sh{state['sharpen_alpha']:.2f}.tif"
                )
                imsave(str(out_dir / name), rgb, check_contrast=False)
                status.set_text(f"Saved {name}")
        except Exception as exc:  # noqa: BLE001 - show in UI
            status.set_text(f"Error: {exc}")
        finally:
            state["busy"] = False
            fig.canvas.draw_idle()

    def on_key(event) -> None:
        if event.key in {"r", "R", "enter"}:
            do_render()
        elif event.key in {"s", "S"} and args.autosave_dir is not None:
            do_render()

    btn.on_clicked(do_render)
    fig.canvas.mpl_connect("key_press_event", on_key)
    do_render()
    plt.show()
    return 0


def run_grid(prefix: str, nuclei: np.ndarray, cyto: np.ndarray, args: argparse.Namespace) -> int:
    output_dir = args.output_dir
    if output_dir is None:
        raise ValueError("--mode grid requires --output_dir")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    nuc_scales = args.nuc_scales
    cyto_scales = args.cyto_scales
    bg_nucs = args.bg_nuc_factors
    bg_cytos = args.bg_cyto_factors
    sharpen_alphas = args.sharpen_alphas

    total = (
        len(nuc_scales)
        * len(cyto_scales)
        * len(bg_nucs)
        * len(bg_cytos)
        * len(sharpen_alphas)
    )
    print(f"Rendering grid: {total} combinations -> {output_dir}")
    done = 0
    for nuc_scale in nuc_scales:
        for cyto_scale in cyto_scales:
            for bg_nuc in bg_nucs:
                for bg_cyto in bg_cytos:
                    for sharpen_alpha in sharpen_alphas:
                        done += 1
                        t0 = time.time()
                        rgb = render_preview(
                            nuclei,
                            cyto,
                            flatfield_nuc_scale=nuc_scale,
                            flatfield_cyto_scale=cyto_scale,
                            bg_nuc_factor=bg_nuc,
                            bg_cyto_factor=bg_cyto,
                            sharpen_alpha=sharpen_alpha,
                            args=args,
                        )
                        name = (
                            f"{prefix}_nuc{nuc_scale:.2f}_cyto{cyto_scale:.2f}"
                            f"_bgN{bg_nuc:.2f}_bgC{bg_cyto:.2f}"
                            f"_sh{sharpen_alpha:.2f}.tif"
                        )
                        imsave(str(output_dir / name), rgb, check_contrast=False)
                        print(
                            f"[{done}/{total}] {name} ({time.time() - t0:.1f}s)",
                            flush=True,
                        )
    print(f"Done. Wrote {total} images to {output_dir}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive/grid tuner for falsecolor nuc/cyto scales and bg factors.",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path(
            r"Z:\YF2026061901\20260701_09_44_49_YF2026061901_CHYY_fei_Destripe_DONE\All_Channels"
        ),
    )
    parser.add_argument(
        "--nuclei_tiff",
        type=Path,
        default=None,
        help="Optional explicit nuclei TIFF; skip folder scan when used with --cyto_tiff.",
    )
    parser.add_argument(
        "--cyto_tiff",
        type=Path,
        default=None,
        help="Optional explicit cyto TIFF; skip folder scan when used with --nuclei_tiff.",
    )
    parser.add_argument("--output_dir", type=Path, default=None, help="Required for --mode grid.")
    parser.add_argument(
        "--autosave_dir",
        type=Path,
        default=None,
        help="If set in interactive mode, each Render also saves a TIFF here.",
    )
    parser.add_argument("--mode", choices=("interactive", "grid"), default="interactive")
    parser.add_argument("--prefix", default="", help="Exact slice prefix to use.")
    parser.add_argument("--middle", type=int, default=1, help="Pick 1 slice from stack center.")
    parser.add_argument("--slice_start", type=int, default=0)
    parser.add_argument("--nuclei_channel", type=int, default=1)
    parser.add_argument("--cyto_channel", type=int, default=0)
    parser.add_argument(
        "--crop",
        type=int,
        default=1536,
        help="Center crop size for faster preview (0 = full plane).",
    )
    parser.add_argument(
        "--downscale",
        type=int,
        default=1,
        help="Integer downscale after crop (1 = none).",
    )
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--color_key", choices=("HE", "IHC"), default="HE")
    parser.add_argument(
        "--nuclei_hue",
        choices=("default", "blue", "purple"),
        default="default",
    )
    parser.add_argument("--nuc_threshold", type=int, default=100)
    parser.add_argument("--cyto_threshold", type=int, default=100)
    parser.add_argument("--nuc_normfactor", type=int, default=8200)
    parser.add_argument("--cyto_normfactor", type=int, default=2100)
    parser.add_argument("--flatfield_tile_size", type=int, default=256)
    parser.add_argument("--flatfield_scale", type=float, default=1.0)
    parser.add_argument("--clahe", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clahe_clip_limit", type=float, default=1.5)
    parser.add_argument(
        "--mask_background",
        choices=("none", "channels", "rgb", "channels+rgb", "hsv"),
        default="none",
    )
    parser.add_argument("--background_threshold", type=int, default=250)
    parser.add_argument("--white_threshold", type=int, default=250)
    parser.add_argument("--pale_signal_threshold", type=int, default=2000)

    parser.add_argument("--init_nuc_scale", type=float, default=0.7)
    parser.add_argument("--init_cyto_scale", type=float, default=2.6)
    parser.add_argument("--init_bg_nuc_factor", type=float, default=3.0)
    parser.add_argument("--init_bg_cyto_factor", type=float, default=1.0)
    parser.add_argument("--init_sharpen_alpha", type=float, default=0.05)

    parser.add_argument(
        "--nuc_scales",
        type=parse_float_list,
        default=parse_float_list("1.0,1.5,2.5"),
        help="Grid values for flatfield_nuc_scale.",
    )
    parser.add_argument(
        "--cyto_scales",
        type=parse_float_list,
        default=parse_float_list("2.0,3.72,5.0"),
        help="Grid values for flatfield_cyto_scale.",
    )
    parser.add_argument(
        "--bg_nuc_factors",
        type=parse_float_list,
        default=parse_float_list("0.0,0.5,1.0"),
        help="Grid values for bg_nuc_factor.",
    )
    parser.add_argument(
        "--bg_cyto_factors",
        type=parse_float_list,
        default=parse_float_list("0.0,3.0,4.5"),
        help="Grid values for bg_cyto_factor.",
    )
    parser.add_argument(
        "--sharpen_alphas",
        type=parse_float_list,
        default=parse_float_list("0.0"),
        help="Grid values for sharpen_alpha.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.backend == "gpu" and not _gpu_available():
        print("CUDA unavailable; falling back to CPU (flatfield requires GPU).", file=sys.stderr)
        args.backend = "cpu"
    if args.mode in {"interactive", "grid"} and args.backend != "gpu":
        print("This tuner uses flatfield and requires --backend gpu.", file=sys.stderr)
        return 1

    prefix, nuclei, cyto = load_preview_pair(args)
    if args.mode == "grid":
        return run_grid(prefix, nuclei, cyto, args)
    return run_interactive(prefix, nuclei, cyto, args)


if __name__ == "__main__":
    raise SystemExit(main())
