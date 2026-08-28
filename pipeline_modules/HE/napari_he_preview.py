"""
Napari preview for HE preprocessing + nuclear segmentation + falsecolor.

Layers:
  - nuclei (after rolling-ball, optional LCN)
  - nuclei_seg (Otsu / fixed threshold mask)
  - cyto (after deconv+destripe)
  - HE (RGB falsecolor)

Example:
  python pipeline_modules/HE/napari_he_preview.py --middle 1 --crop 2048
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from skimage.io import imread, imsave

from pipeline_modules.HE.falsecolor_batch import (
    _gpu_available,
    discover_channel_pairs,
    preprocess_colleague_tricks,
    render_falsecolor,
    select_slice_pairs,
)
from pipeline_modules.HE.nuclear_seg import segment_nuclei_preview
from pipeline_modules.HE.preprocess import HePreprocessParams, preprocess_cyto, preprocess_nuclei


def center_crop(image: np.ndarray, size: int) -> np.ndarray:
    if size <= 0:
        return image
    height, width = image.shape[:2]
    crop = min(size, height, width)
    r0 = max(0, (height - crop) // 2)
    c0 = max(0, (width - crop) // 2)
    return image[r0 : r0 + crop, c0 : c0 + crop]


def load_pair(args: argparse.Namespace) -> tuple[str, np.ndarray, np.ndarray]:
    if args.nuclei_tiff is not None and args.cyto_tiff is not None:
        nuclei_path = args.nuclei_tiff.resolve()
        cyto_path = args.cyto_tiff.resolve()
        prefix = nuclei_path.stem.rsplit("_ch", 1)[0]
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
            raise FileNotFoundError(f"No paired channels in {input_dir}")
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
    nuclei = center_crop(imread(str(nuclei_path)), args.crop)
    cyto = center_crop(imread(str(cyto_path)), args.crop)
    return prefix, nuclei, cyto


def build_he_rgb(
    nuclei: np.ndarray,
    cyto: np.ndarray,
    *,
    flatfield_nuc_scale: float,
    flatfield_cyto_scale: float,
    bg_nuc_factor: float,
    bg_cyto_factor: float,
    sharpen_alpha: float,
    nuclei_hue: str,
    nuc_threshold: int,
    cyto_threshold: int,
    flatfield_tile_size: int,
) -> np.ndarray:
    use_flatfield = _gpu_available()
    if not use_flatfield:
        print("CUDA unavailable; HE preview falls back to CPU without flatfield.", flush=True)

    nuc_p, cyto_p = preprocess_colleague_tricks(
        nuclei,
        cyto,
        nuc_threshold=nuc_threshold,
        cyto_threshold=cyto_threshold,
        flatfield=use_flatfield,
        bg_nuc_factor=bg_nuc_factor if use_flatfield else 0.0,
        bg_cyto_factor=bg_cyto_factor if use_flatfield else 0.0,
        sharpen_alpha=sharpen_alpha,
        clahe=False,
        clahe_clip_limit=1.5,
    )
    return render_falsecolor(
        nuc_p,
        cyto_p,
        backend="gpu" if use_flatfield else "cpu",
        color_key="HE",
        nuclei_hue=nuclei_hue,
        nuc_threshold=nuc_threshold,
        cyto_threshold=cyto_threshold,
        nuc_normfactor=8200,
        cyto_normfactor=2100,
        flatfield=use_flatfield,
        flatfield_tile_size=flatfield_tile_size,
        flatfield_scale=1.0,
        flatfield_nuc_scale=flatfield_nuc_scale,
        flatfield_cyto_scale=flatfield_cyto_scale,
    )


def run_napari(prefix: str, nuclei_raw: np.ndarray, cyto_raw: np.ndarray, args: argparse.Namespace) -> int:
    try:
        import napari
        from magicgui import magicgui
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "napari and magicgui are required. Activate an env with both installed."
        ) from exc

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

    state: dict = {
        "prefix": prefix,
        "nuclei_raw": nuclei_raw,
        "cyto_raw": cyto_raw,
        "params": params,
        "use_lcn": False,
        "cyto": None,
        "nuclei": None,
        "after_deconv": None,
        "after_destripe": None,
        "after_rolling_ball": None,
        "seg": None,
        "he": None,
        "seg_thr": 0.0,
    }

    def apply_preprocess_params(
        *,
        deconv_backend: str,
        deconv_sigma: float,
        deconv_iterations: int,
        destripe_backend: str,
        destripe_strength: float,
        destripe_notch_width: float,
        destripe_keep_fraction: float,
        destripe_smooth_sigma: float,
        destripe_orientation: str,
        rolling_ball_radius: int,
        lcn_clip_limit: float,
        lcn_grid_size: int,
    ) -> None:
        p = state["params"]
        p.deconv.backend = deconv_backend
        p.deconv.sigma = float(deconv_sigma)
        p.deconv.iterations = int(deconv_iterations)
        p.destripe.backend = destripe_backend
        p.destripe.strength = float(destripe_strength)
        p.destripe.notch_width = float(destripe_notch_width)
        p.destripe.keep_fraction = float(destripe_keep_fraction)
        p.destripe.sigma = float(destripe_smooth_sigma)
        p.destripe.orientation = destripe_orientation
        p.rolling_ball.radius = int(rolling_ball_radius)
        p.lcn.clip_limit = float(lcn_clip_limit)
        p.lcn.grid_size = int(lcn_grid_size)

    def recompute_channels(use_lcn: bool) -> None:
        print(f"Preprocessing channels (use_lcn={use_lcn}) ...", flush=True)
        t0 = time.time()
        state["cyto"] = preprocess_cyto(state["cyto_raw"], state["params"])
        nuc_result = preprocess_nuclei(
            state["nuclei_raw"],
            use_lcn=use_lcn,
            params=state["params"],
        )
        state["nuclei"] = nuc_result["nuclei"]
        state["after_deconv"] = nuc_result["after_deconv"]
        state["after_destripe"] = nuc_result["after_destripe"]
        state["after_rolling_ball"] = nuc_result["after_rolling_ball"]
        state["use_lcn"] = use_lcn
        print(f"Preprocess done in {time.time() - t0:.1f}s", flush=True)

    def recompute_seg(method: str, threshold: float | None) -> None:
        mask, thr = segment_nuclei_preview(
            state["nuclei"],
            method=method,
            threshold=threshold,
            min_size=args.seg_min_size,
        )
        state["seg"] = mask
        state["seg_thr"] = thr

    def recompute_he(
        flatfield_nuc_scale: float,
        flatfield_cyto_scale: float,
        bg_nuc_factor: float,
        bg_cyto_factor: float,
        sharpen_alpha: float,
    ) -> None:
        print("Rendering HE ...", flush=True)
        t0 = time.time()
        state["he"] = build_he_rgb(
            state["nuclei"],
            state["cyto"],
            flatfield_nuc_scale=flatfield_nuc_scale,
            flatfield_cyto_scale=flatfield_cyto_scale,
            bg_nuc_factor=bg_nuc_factor,
            bg_cyto_factor=bg_cyto_factor,
            sharpen_alpha=sharpen_alpha,
            nuclei_hue=args.nuclei_hue,
            nuc_threshold=args.nuc_threshold,
            cyto_threshold=args.cyto_threshold,
            flatfield_tile_size=args.flatfield_tile_size,
        )
        print(f"HE done in {time.time() - t0:.1f}s", flush=True)

    recompute_channels(use_lcn=False)
    recompute_seg("otsu", None)
    recompute_he(
        args.init_nuc_scale,
        args.init_cyto_scale,
        args.init_bg_nuc_factor,
        args.init_bg_cyto_factor,
        args.init_sharpen_alpha,
    )

    viewer = napari.Viewer(title=f"HE preview: {prefix}", ndisplay=2)
    viewer.add_image(state["nuclei_raw"], name="nuclei_raw", colormap="gray", visible=False)
    viewer.add_image(state["after_deconv"], name="nuclei_deconv", colormap="gray", visible=False)
    viewer.add_image(state["after_destripe"], name="nuclei_destripe", colormap="gray", visible=False)
    viewer.add_image(
        state["after_rolling_ball"], name="nuclei_rolling_ball", colormap="gray", visible=False
    )
    viewer.add_image(state["nuclei"], name="nuclei", colormap="magenta", blending="additive")
    viewer.add_labels(state["seg"].astype(np.int32), name="nuclei_seg", opacity=0.35)
    viewer.add_image(state["cyto"], name="cyto", colormap="green", blending="additive", visible=False)
    viewer.add_image(state["he"], name="HE", rgb=True)

    @magicgui(
        call_button="Update preview",
        deconv_backend={"choices": ["rl", "placeholder", "identity"]},
        deconv_sigma={"min": 0.3, "max": 5.0, "step": 0.1},
        deconv_iterations={"min": 1, "max": 40, "step": 1},
        destripe_backend={"choices": ["fft", "smooth", "placeholder", "identity"]},
        destripe_strength={"min": 0.0, "max": 1.0, "step": 0.05},
        destripe_notch_width={"min": 0.5, "max": 10.0, "step": 0.1},
        destripe_keep_fraction={"min": 0.01, "max": 0.2, "step": 0.01},
        destripe_smooth_sigma={"min": 5.0, "max": 120.0, "step": 1.0},
        destripe_orientation={"choices": ["horizontal", "vertical", "both"]},
        rolling_ball_radius={"min": 5, "max": 200, "step": 1},
        use_lcn={"label": "use LCN (CLAHE)"},
        lcn_clip_limit={"min": 0.5, "max": 8.0, "step": 0.1},
        lcn_grid_size={"min": 4, "max": 64, "step": 2},
        seg_method={"choices": ["otsu", "fixed"]},
        seg_threshold={"min": 0.0, "max": 65535.0, "step": 1.0},
        flatfield_nuc_scale={"min": 0.1, "max": 8.0, "step": 0.05},
        flatfield_cyto_scale={"min": 0.1, "max": 10.0, "step": 0.05},
        bg_nuc_factor={"min": 0.0, "max": 6.0, "step": 0.05},
        bg_cyto_factor={"min": 0.0, "max": 6.0, "step": 0.05},
        sharpen_alpha={"min": 0.0, "max": 2.0, "step": 0.05},
        layout="vertical",
        scrollable=True,
    )
    def controls(
        deconv_backend: str = args.deconv_backend,
        deconv_sigma: float = args.deconv_sigma,
        deconv_iterations: int = args.deconv_iterations,
        destripe_backend: str = args.destripe_backend,
        destripe_strength: float = args.destripe_strength,
        destripe_notch_width: float = args.destripe_notch_width,
        destripe_keep_fraction: float = args.destripe_keep_fraction,
        destripe_smooth_sigma: float = args.destripe_smooth_sigma,
        destripe_orientation: str = args.destripe_orientation,
        rolling_ball_radius: int = args.rolling_ball_radius,
        use_lcn: bool = False,
        lcn_clip_limit: float = args.lcn_clip_limit,
        lcn_grid_size: int = args.lcn_grid_size,
        seg_method: str = "otsu",
        seg_threshold: float = 0.0,
        flatfield_nuc_scale: float = args.init_nuc_scale,
        flatfield_cyto_scale: float = args.init_cyto_scale,
        bg_nuc_factor: float = args.init_bg_nuc_factor,
        bg_cyto_factor: float = args.init_bg_cyto_factor,
        sharpen_alpha: float = args.init_sharpen_alpha,
        save_he_path: str = "he_preview.tif",
        do_save: bool = False,
    ) -> None:
        apply_preprocess_params(
            deconv_backend=deconv_backend,
            deconv_sigma=deconv_sigma,
            deconv_iterations=deconv_iterations,
            destripe_backend=destripe_backend,
            destripe_strength=destripe_strength,
            destripe_notch_width=destripe_notch_width,
            destripe_keep_fraction=destripe_keep_fraction,
            destripe_smooth_sigma=destripe_smooth_sigma,
            destripe_orientation=destripe_orientation,
            rolling_ball_radius=rolling_ball_radius,
            lcn_clip_limit=lcn_clip_limit,
            lcn_grid_size=lcn_grid_size,
        )
        recompute_channels(use_lcn=use_lcn)
        viewer.layers["nuclei_deconv"].data = state["after_deconv"]
        viewer.layers["nuclei_destripe"].data = state["after_destripe"]
        viewer.layers["nuclei_rolling_ball"].data = state["after_rolling_ball"]
        viewer.layers["nuclei"].data = state["nuclei"]
        viewer.layers["cyto"].data = state["cyto"]

        thr_arg = None if seg_method == "otsu" else float(seg_threshold)
        recompute_seg(seg_method, thr_arg)
        viewer.layers["nuclei_seg"].data = state["seg"].astype(np.int32)

        recompute_he(
            flatfield_nuc_scale,
            flatfield_cyto_scale,
            bg_nuc_factor,
            bg_cyto_factor,
            sharpen_alpha,
        )
        viewer.layers["HE"].data = state["he"]
        print(
            f"Updated | deconv={deconv_backend}({deconv_sigma:.1f},{deconv_iterations}) "
            f"destripe={destripe_backend}/{destripe_orientation} "
            f"rb={rolling_ball_radius} LCN={use_lcn} seg_thr={state['seg_thr']:.1f}",
            flush=True,
        )
        if do_save:
            out = Path(save_he_path)
            imsave(str(out), state["he"], check_contrast=False)
            print(f"Saved HE -> {out.resolve()}", flush=True)

    viewer.window.add_dock_widget(controls, area="right", name="HE preprocess + falsecolor")
    napari.run()
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Napari HE preprocess + falsecolor preview.")
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
    parser.add_argument("--init_nuc_scale", type=float, default=0.7)
    parser.add_argument("--init_cyto_scale", type=float, default=2.6)
    parser.add_argument("--init_bg_nuc_factor", type=float, default=3.0)
    parser.add_argument("--init_bg_cyto_factor", type=float, default=1.0)
    parser.add_argument("--init_sharpen_alpha", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prefix, nuclei, cyto = load_pair(args)
    return run_napari(prefix, nuclei, cyto, args)


if __name__ == "__main__":
    raise SystemExit(main())
