"""Render the ROI rotation video: raw volume (vtk MIP blend) -> mask -> vessels,
each rotating 360 deg in side view with crossfades. cv2 mp4v writer.

Usage (yifu env, repo root on sys.path):
  python roi_video.py --image_zarr <ch1.zarr> --mask_zarr <filtered_mask.zarr> \
      --swc_dir <vessel_recon_*/swc> --center 234,1867,2087 --side_um 1000 \
      --resolution 3.5006,2.0,2.0 --out <out.mp4>
"""
import argparse
import numpy as np
import zarr
import pyvista as pv
from skimage.measure import marching_cubes

from pipeline_modules.visualization import render_vessels_pyvista as rv


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image_zarr", required=True)
    ap.add_argument("--mask_zarr", required=True)
    ap.add_argument("--swc_dir", required=True)
    ap.add_argument("--center", required=True, help="cube center voxel z,y,x")
    ap.add_argument("--side_um", type=float, default=1000.0)
    ap.add_argument("--resolution", default="3.5006,2.0,2.0")
    ap.add_argument("--out", required=True, help="output mp4 path")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--rot_s", type=float, default=10.0, help="seconds per 360 deg")
    ap.add_argument("--fade_s", type=float, default=2.0)
    ap.add_argument("--window", type=int, default=2000)
    ap.add_argument("--tube_width", type=float, default=9.0)
    ap.add_argument("--branch_sphere_um", type=float, default=7.0)
    ap.add_argument("--mask_min_voxels", type=int, default=640,
                    help="drop mask components smaller than this (native voxels); 0 keeps everything")
    args = ap.parse_args()

    RES = np.array([float(v) for v in args.resolution.split(",")])
    CENTER_VOX = np.array([int(v) for v in args.center.split(",")])
    SIDE_UM = args.side_um
    FPS, T_ROT, T_FADE = args.fps, args.rot_s, args.fade_s
    WINDOW = (args.window, args.window)
    AZ_RATE = 360.0 / T_ROT

    background_color = (0.0, 0.0, 0.0)  # pure black
    pv.global_theme.background = background_color
    pv.OFF_SCREEN = True

    hz, hy, hx = int(SIDE_UM / 2 / RES[0]), int(SIDE_UM / 2 / RES[1]), int(SIDE_UM / 2 / RES[2])
    cz, cy, cx = CENTER_VOX
    sl = (slice(cz - hz, cz + hz), slice(cy - hy, cy + hy), slice(cx - hx, cx + hx))
    off_um = np.array([sl[0].start * RES[0], sl[1].start * RES[1], sl[2].start * RES[2]])
    center_xyz = (off_um + np.array([sl[0].stop - sl[0].start, sl[1].stop - sl[1].start,
                                     sl[2].stop - sl[2].start]) * RES / 2.0)[::-1]
    lo3, hi3 = center_xyz - SIDE_UM / 2, center_xyz + SIDE_UM / 2

    def set_cam_side(p, azimuth_deg):
        a = np.radians(azimuth_deg)
        d = np.array([np.cos(a), np.sin(a), 0.0])
        p.camera_position = [tuple(center_xyz + d * SIDE_UM * 2.5), tuple(center_xyz), (0.0, 0.0, 1.0)]
        p.enable_parallel_projection()
        p.parallel_scale = SIDE_UM * 0.95

    def add_box(p):
        # shift wireframe toward the camera each frame: projection unchanged, never occluded
        return p.add_mesh(pv.Box(bounds=[lo3[0], hi3[0], lo3[1], hi3[1], lo3[2], hi3[2]]),
                          style="wireframe", color=rv.frame_color_for_background("dark"), line_width=3)

    box_actors = []

    # ---- panel A: raw volume with true MIP blend ----
    raw = zarr.open(args.image_zarr + "/0", mode="r")[sl].astype(np.uint16)
    grid = pv.ImageData(dimensions=np.array(raw.shape[::-1]), spacing=tuple(RES[::-1]),
                        origin=tuple(off_um[::-1]))
    grid.point_data["intensity"] = raw.ravel()  # C order (x fastest) — order="F" scrambles the volume
    vlo, vhi = np.percentile(raw, [50, 99.5])
    pA = pv.Plotter(off_screen=True, window_size=WINDOW)
    pA.set_background(background_color, top=background_color)
    volA = pA.add_volume(grid, cmap="gray", clim=(float(vlo), float(vhi)))
    volA.mapper.SetBlendModeToMaximumIntensity()  # the only working MIP switch; mapper="maximum_intensity" doesn't exist
    box_actors.append(add_box(pA))
    print("panel A ready", flush=True)

    # ---- panel B: mask surface ----
    mask = zarr.open(args.mask_zarr + "/0", mode="r")[sl] > 0
    # drop small fragments (same padded-crop rule as roi_trio: components reaching the
    # padded border are kept, purely-local ones below the threshold are removed)
    from scipy import ndimage
    pad = 60
    full_shape = zarr.open(args.mask_zarr + "/0", mode="r").shape
    psl = tuple(slice(max(s.start - pad, 0), min(s.stop + pad, n)) for s, n in zip(sl, full_shape))
    padded = zarr.open(args.mask_zarr + "/0", mode="r")[psl] > 0
    lab, n = ndimage.label(padded)
    sizes = np.bincount(lab.ravel())
    border = np.zeros(n + 1, dtype=bool)
    for axis in range(3):
        for idx in (0, -1):
            b = np.take(lab, idx, axis=axis)
            border[np.unique(b)] = True
    small = (sizes < args.mask_min_voxels) & ~border
    small[0] = False
    padded[small[lab]] = False
    inner = tuple(slice(s.start - p.start, s.stop - p.start) for s, p in zip(sl, psl))
    mask = padded[inner]
    print("mask fragments removed in/around cube:", int(small.sum()))
    verts, faces, _, _ = marching_cubes(mask.astype(np.uint8), level=0.5, spacing=tuple(RES))
    verts = verts[:, ::-1] + off_um[::-1]  # (z,y,x) -> (x,y,z)
    mesh = pv.PolyData(verts)
    mesh.faces = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64)
    pB = pv.Plotter(off_screen=True, window_size=WINDOW)
    pB.set_background(background_color, top=background_color)
    pB.add_mesh(mesh, color="#ff4646", smooth_shading=True,
                ambient=0.3, diffuse=0.7, specular=0.4, specular_power=24)
    box_actors.append(add_box(pB))
    print("panel B ready", flush=True)

    # ---- panel C: vessels ----
    branches, bps = rv.load_branches(args.swc_dir, max_files=0)
    lo_b, hi_b = lo3 - 1, hi3 + 1
    keep = [b for b in branches if ((b.points >= lo_b).all(axis=1) & (b.points <= hi_b).all(axis=1)).all()]
    clim = rv.diameter_clim(keep, clip_percentiles=(2.0, 98.0))
    cmap = rv.resolve_colormap("bright")
    pC = pv.Plotter(off_screen=True, window_size=WINDOW)
    pC.set_background(background_color, top=background_color)
    try:
        pC.enable_anti_aliasing("ssaa")
    except Exception:
        pass
    for index, branch in enumerate(keep):
        points = rv.prepare_branch_points(branch.points, min_spacing=1.0, smooth_window=5)
        if len(points) < 2:
            continue
        m = pv.Spline(points, max(len(points) * 12, len(points))) if len(points) >= 3 else pv.lines_from_points(points)
        m.point_data["diameter_um"] = np.full(m.n_points, np.clip(branch.mean_diameter_um, clim[0], clim[1]))
        pC.add_mesh(m, scalars="diameter_um", cmap=cmap, clim=clim, line_width=args.tube_width,
                    render_lines_as_tubes=True, show_scalar_bar=(index == 0),
                    scalar_bar_args={"title": "Diameter (um)", "position_x": 0.80, "position_y": 0.055,
                                     "width": 0.14, "height": 0.022, "n_labels": 4,
                                     "color": rv.text_color_for_background("dark"),
                                     "title_font_size": 10, "label_font_size": 8, "fmt": "%.2g"},
                    ambient=0.12, diffuse=0.78, specular=0.32, specular_power=32)
    bpk = bps[((bps >= lo_b).all(axis=1)) & ((bps <= hi_b).all(axis=1))]
    cloud = pv.PolyData(bpk)
    cloud["s"] = np.full(len(bpk), 1.0)  # glyph scale multiplies geom radius — keep 1.0
    pC.add_mesh(cloud.glyph(geom=pv.Sphere(radius=args.branch_sphere_um), scale="s", orient=False, factor=1.0),
                color=rv.branch_point_color_for_background("dark"), smooth_shading=True,
                ambient=0.18, diffuse=0.68, specular=0.38, specular_power=36)
    box_actors.append(add_box(pC))
    print("panel C ready, branches:", len(keep), "bps:", len(bpk), flush=True)

    # ---- timeline with crossfades ----
    panels = [pA, pB, pC]
    starts = [0.0, T_ROT, 2 * T_ROT]
    TOTAL = starts[2] + T_ROT + T_FADE
    n_frames = int(TOTAL * FPS)
    import cv2
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, WINDOW)
    last_img = None
    for f in range(n_frames):
        t = f / FPS
        weights = np.zeros(3)
        for i, s in enumerate(starts):
            e = s + T_ROT + T_FADE
            if s <= t < e:
                weights[i] = 1.0
                if i < 2 and t > e - T_FADE:
                    weights[i] = max(0.0, (e - t) / T_FADE)  # fade out
                if i > 0 and t < s + T_FADE:
                    weights[i] = min(weights[i], (t - s) / T_FADE)  # fade in
        img = None
        for i, w in enumerate(weights):
            if w <= 0:
                continue
            az = AZ_RATE * (t - starts[i])
            d = np.array([np.cos(np.radians(az)), np.sin(np.radians(az)), 0.0])
            box_actors[i].SetPosition(*(d * (SIDE_UM / 2)))
            set_cam_side(panels[i], az)
            panels[i].render()  # screenshot() returns a CACHED frame without this
            frame = panels[i].screenshot(return_img=True)
            img = frame.astype(np.float64) * w if img is None else img + frame.astype(np.float64) * w
        if img is None:
            img = last_img if last_img is not None else np.zeros((WINDOW[1], WINDOW[0], 3), np.uint8)
        last_img = img.astype(np.uint8)
        writer.write(cv2.cvtColor(last_img, cv2.COLOR_RGB2BGR))
        if f % 100 == 0:
            print(f"frame {f}/{n_frames} t={t:.1f}s", flush=True)
    writer.release()
    print("video done:", args.out, flush=True)


if __name__ == "__main__":
    main()
