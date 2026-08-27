"""
HE preprocessing + falsecolor CLI.

Examples:
  # Open Napari preview on middle slice
  python pipeline_modules/HE/pipeline.py --middle 1 --crop 2048 --preview

  # Preprocess + export HE without UI
  python pipeline_modules/HE/pipeline.py --middle 1 --crop 2048 --output_he he_preview.tif --use_lcn
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from skimage.io import imsave

from pipeline_modules.HE.napari_he_preview import build_he_rgb, load_pair
from pipeline_modules.HE.nuclear_seg import segment_nuclei_preview
from pipeline_modules.HE.preprocess import HePreprocessParams, preprocess_cyto, preprocess_nuclei


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HE preprocess pipeline + optional Napari preview.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path(
            r"Z:\YF2026061901\20260701_09_44_49_YF2026061901_CHYY_fei_Destripe_DONE\All_Channels"
        ),
    )
    parser.add_argument("--nuclei_tiff", type=Path, default=None)
    parser.add_argument("--cyto_tiff", type=Path, default=None)
    parser.add_argument("--nuclei_channel", type=int, default=1)
    parser.add_argument("--cyto_channel", type=int, default=0)
    parser.add_argument("--middle", type=int, default=1)
    parser.add_argument("--slice_start", type=int, default=0)
    parser.add_argument("--crop", type=int, default=2048)
    parser.add_argument("--preview", action="store_true", help="Open Napari HE preview UI.")
    parser.add_argument("--use_lcn", action="store_true", help="Enable CLAHE on nuclei before HE.")
    parser.add_argument("--output_he", type=Path, default=None, help="Write HE RGB TIFF.")
    parser.add_argument("--output_nuclei", type=Path, default=None, help="Write preprocessed nuclei.")
    parser.add_argument("--output_cyto", type=Path, default=None, help="Write preprocessed cyto.")
    parser.add_argument("--output_seg", type=Path, default=None, help="Write nuclei seg mask.")
    parser.add_argument(
        "--deconv_backend",
        choices=("rl", "placeholder", "identity"),
        default="rl",
    )
    parser.add_argument("--deconv_sigma", type=float, default=1.2)
    parser.add_argument("--deconv_iterations", type=int, default=10)
    parser.add_argument(
        "--destripe_backend",
        choices=("fft", "smooth", "placeholder", "identity"),
        default="fft",
    )
    parser.add_argument("--destripe_strength", type=float, default=0.85)
    parser.add_argument("--destripe_notch_width", type=float, default=2.0)
    parser.add_argument("--destripe_keep_fraction", type=float, default=0.04)
    parser.add_argument("--destripe_smooth_sigma", type=float, default=40.0)
    parser.add_argument(
        "--destripe_orientation",
        choices=("horizontal", "vertical", "both"),
        default="horizontal",
    )
    parser.add_argument("--rolling_ball_radius", type=int, default=5)
    parser.add_argument("--lcn_clip_limit", type=float, default=2.0)
    parser.add_argument("--lcn_grid_size", type=int, default=16)
    parser.add_argument("--seg_min_size", type=int, default=16)
    parser.add_argument("--nuclei_hue", choices=("default", "blue", "purple"), default="default")
    parser.add_argument("--nuc_threshold", type=int, default=100)
    parser.add_argument("--cyto_threshold", type=int, default=100)
    parser.add_argument("--flatfield_tile_size", type=int, default=256)
    parser.add_argument("--flatfield_nuc_scale", type=float, default=0.7)
    parser.add_argument("--flatfield_cyto_scale", type=float, default=2.6)
    parser.add_argument("--bg_nuc_factor", type=float, default=3.0)
    parser.add_argument("--bg_cyto_factor", type=float, default=1.0)
    parser.add_argument("--sharpen_alpha", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.preview:
        from pipeline_modules.HE.napari_he_preview import main as napari_main

        # Map overlapping flags into napari CLI argv.
        napari_argv = [
            "--input_dir",
            str(args.input_dir),
            "--middle",
            str(args.middle),
            "--slice_start",
            str(args.slice_start),
            "--crop",
            str(args.crop),
            "--deconv_backend",
            args.deconv_backend,
            "--deconv_sigma",
            str(args.deconv_sigma),
            "--deconv_iterations",
            str(args.deconv_iterations),
            "--destripe_backend",
            args.destripe_backend,
            "--destripe_strength",
            str(args.destripe_strength),
            "--destripe_notch_width",
            str(args.destripe_notch_width),
            "--destripe_keep_fraction",
            str(args.destripe_keep_fraction),
            "--destripe_smooth_sigma",
            str(args.destripe_smooth_sigma),
            "--destripe_orientation",
            args.destripe_orientation,
            "--rolling_ball_radius",
            str(args.rolling_ball_radius),
            "--lcn_clip_limit",
            str(args.lcn_clip_limit),
            "--lcn_grid_size",
            str(args.lcn_grid_size),
            "--nuclei_hue",
            args.nuclei_hue,
            "--init_nuc_scale",
            str(args.flatfield_nuc_scale),
            "--init_cyto_scale",
            str(args.flatfield_cyto_scale),
            "--init_bg_nuc_factor",
            str(args.bg_nuc_factor),
            "--init_bg_cyto_factor",
            str(args.bg_cyto_factor),
            "--init_sharpen_alpha",
            str(args.sharpen_alpha),
        ]
        if args.nuclei_tiff is not None:
            napari_argv += ["--nuclei_tiff", str(args.nuclei_tiff)]
        if args.cyto_tiff is not None:
            napari_argv += ["--cyto_tiff", str(args.cyto_tiff)]
        return napari_main(napari_argv)

    prefix, nuclei_raw, cyto_raw = load_pair(args)
    params = HePreprocessParams()
    params.deconv.backend = args.deconv_backend
    params.deconv.sigma = args.deconv_sigma
    params.deconv.iterations = args.deconv_iterations
    params.destripe.backend = args.destripe_backend
    params.destripe.strength = args.destripe_strength
    params.destripe.notch_width = args.destripe_notch_width
    params.destripe.keep_fraction = args.destripe_keep_fraction
    params.destripe.sigma = args.destripe_smooth_sigma
    params.destripe.orientation = args.destripe_orientation
    params.rolling_ball.radius = args.rolling_ball_radius
    params.lcn.clip_limit = args.lcn_clip_limit
    params.lcn.grid_size = args.lcn_grid_size

    print(f"Preprocessing {prefix} ...", flush=True)
    t0 = time.time()
    cyto = preprocess_cyto(cyto_raw, params)
    nuc_pack = preprocess_nuclei(nuclei_raw, use_lcn=args.use_lcn, params=params)
    nuclei = nuc_pack["nuclei"]
    seg, thr = segment_nuclei_preview(nuclei, method="otsu", min_size=args.seg_min_size)
    print(f"Preprocess done in {time.time() - t0:.1f}s (seg_thr={thr:.1f})", flush=True)

    if args.output_nuclei is not None:
        imsave(str(args.output_nuclei), nuclei, check_contrast=False)
        print(f"Wrote nuclei -> {args.output_nuclei}")
    if args.output_cyto is not None:
        imsave(str(args.output_cyto), cyto, check_contrast=False)
        print(f"Wrote cyto -> {args.output_cyto}")
    if args.output_seg is not None:
        imsave(str(args.output_seg), seg, check_contrast=False)
        print(f"Wrote seg -> {args.output_seg}")

    if args.output_he is not None:
        print("Rendering HE ...", flush=True)
        t1 = time.time()
        he = build_he_rgb(
            nuclei,
            cyto,
            flatfield_nuc_scale=args.flatfield_nuc_scale,
            flatfield_cyto_scale=args.flatfield_cyto_scale,
            bg_nuc_factor=args.bg_nuc_factor,
            bg_cyto_factor=args.bg_cyto_factor,
            sharpen_alpha=args.sharpen_alpha,
            nuclei_hue=args.nuclei_hue,
            nuc_threshold=args.nuc_threshold,
            cyto_threshold=args.cyto_threshold,
            flatfield_tile_size=args.flatfield_tile_size,
        )
        imsave(str(args.output_he), he, check_contrast=False)
        print(f"Wrote HE -> {args.output_he} ({time.time() - t1:.1f}s)")

    if (
        args.output_he is None
        and args.output_nuclei is None
        and args.output_cyto is None
        and args.output_seg is None
    ):
        print("Nothing to write. Pass --preview and/or --output_he / --output_nuclei / ...")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
