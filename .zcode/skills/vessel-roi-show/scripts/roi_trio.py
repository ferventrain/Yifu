"""Render the ROI trio (corner-view MIP / mask isosurface / diameter-colored skeleton)
for one cube of a vessel sample. All the axis-order / row-order / occlusion traps of
the original 37L session are baked in — do not "simplify" them away.

Usage (yifu env, repo root on sys.path for pipeline_modules imports):
  python roi_trio.py --image_zarr <ch1.zarr> --mask_zarr <filtered_mask.zarr> \
      --swc_dir <vessel_recon_*/swc> --center 234,1867,2087 --side_um 1000 \
      --resolution 3.5006,2.0,2.0 --out_dir <figures_dir>
"""
import argparse
import numpy as np
import zarr
from skimage.measure import marching_cubes
from scipy.ndimage import map_coordinates

import pyvista as pv
from pipeline_modules.visualization import render_vessels_pyvista as rv


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image_zarr", required=True, help="raw signal zarr (dataset 0), z,y,x")
    ap.add_argument("--mask_zarr", required=True, help="binary vessel mask zarr (dataset 0)")
    ap.add_argument("--swc_dir", required=True, help="whole-brain SWC dir from vessel_swc_export")
    ap.add_argument("--center", required=True, help="cube center voxel z,y,x (napari point order)")
    ap.add_argument("--side_um", type=float, default=1000.0)
    ap.add_argument("--resolution", default="3.5006,2.0,2.0", help="voxel size z,y,x um (read from ims ExtMax if unsure)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--window", type=int, default=1500, help="square canvas px")
    ap.add_argument("--tube_width", type=float, default=9.0)
    ap.add_argument("--branch_sphere_um", type=float, default=7.0, help="radius; 7um ~ 12px at default zoom")
    ap.add_argument("--mask_min_voxels", type=int, default=640,
                    help="drop mask components smaller than this (native voxels); 640 = skeleton dust rule "
                         "(10 voxels of the 4x-downsampled mask). 0 keeps everything")
    args = ap.parse_args()

    RES = np.array([float(v) for v in args.resolution.split(",")])  # z,y,x um
    CENTER_VOX = np.array([int(v) for v in args.center.split(",")])
    SIDE_UM = args.side_um
    WINDOW = (args.window, args.window)

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    shape = zarr.open(args.image_zarr + "/0", mode="r").shape
    half = SIDE_UM / 2.0
    hz, hy, hx = int(half / RES[0]), int(half / RES[1]), int(half / RES[2])
    cz, cy, cx = CENTER_VOX
    sl = (slice(cz - hz, cz + hz), slice(cy - hy, cy + hy), slice(cx - hx, cx + hx))
    print("ROI voxels:", sl[0], sl[1], sl[2])
    off_um = np.array([sl[0].start * RES[0], sl[1].start * RES[1], sl[2].start * RES[2]])
    center_um = off_um + np.array([sl[0].stop - sl[0].start, sl[1].stop - sl[1].start,
                                   sl[2].stop - sl[2].start]) * RES / 2.0  # z,y,x um
    center_xyz = center_um[::-1]  # x,y,z um — pyvista camera + SWC order

    background_color = (0.0, 0.0, 0.0)  # pure black
    text_color = rv.text_color_for_background("dark")
    pv.global_theme.background = background_color
    pv.OFF_SCREEN = True

    # shared camera: from the top-left-front corner vertex toward center, parallel projection
    dir_v = np.array([-1.0, -1.0, 1.0]) / np.sqrt(3.0)
    cam_pos = center_xyz + dir_v * (SIDE_UM * 2.5)
    viewup = (0.0, 0.0, 1.0)

    def set_cam(p):
        p.camera_position = [tuple(cam_pos), tuple(center_xyz), viewup]
        p.enable_parallel_projection()
        p.parallel_scale = SIDE_UM * 0.95  # fixed identical scale on every panel

    # cube bounds x,y,z um (used by every panel)
    lo3 = off_um[[2, 1, 0]]
    hi3 = lo3 + np.array([sl[2].stop - sl[2].start, sl[1].stop - sl[1].start,
                          sl[0].stop - sl[0].start]) * RES[[2, 1, 0]]

    # ---- panel 1: raw corner-view MIP as a textured quad in the SAME 3D scene ----
    raw = zarr.open(args.image_zarr + "/0", mode="r")[sl].astype(np.float32)
    u_xyz = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)   # screen right
    v_xyz = np.cross(dir_v, u_xyz)                       # screen up
    step = 2.0                                           # output pixel size um
    # corner-view silhouette is a TALL hexagon: half-width side/sqrt2, half-height 2*side/sqrt6
    u_half = SIDE_UM / np.sqrt(2)
    v_half = 2 * SIDE_UM / np.sqrt(6)
    n_u, n_v = int(2 * u_half / step) + 1, int(2 * v_half / step) + 1
    ix, iy = np.meshgrid(np.arange(n_u), np.arange(n_v))
    base_xyz = center_xyz + (ix - n_u / 2)[:, :, None] * step * u_xyz + (iy - n_v / 2)[:, :, None] * step * v_xyz
    mip = None
    for t in np.arange(-SIDE_UM * 0.87, SIDE_UM * 0.87, step):
        zyx_idx = (base_xyz + t * dir_v)[..., ::-1] / RES - np.array([s.start for s in sl])  # LOCAL voxel index
        plane = map_coordinates(raw, np.moveaxis(zyx_idx, -1, 0), order=1, mode="constant", cval=0.0)
        mip = plane if mip is None else np.maximum(mip, plane)
    lo, hi = np.percentile(mip, [1, 99.5])
    # alpha mask = cube silhouette (ray-box intersection per axis)
    t_lo = np.full(mip.shape, -np.inf)
    t_hi = np.full(mip.shape, np.inf)
    axis_mask = np.ones(mip.shape, dtype=bool)
    for i in range(3):
        di = dir_v[i]
        pi = base_xyz[..., i]
        if abs(di) > 1e-9:
            t1, t2 = (lo3[i] - pi) / di, (hi3[i] - pi) / di
            t_lo = np.maximum(t_lo, np.minimum(t1, t2))
            t_hi = np.minimum(t_hi, np.maximum(t1, t2))
        else:
            axis_mask &= np.abs(pi - center_xyz[i]) <= SIDE_UM / 2
    inside_hex = axis_mask & (t_hi >= t_lo)
    # texture draws row 0 at screen TOP but iy=0 is screen BOTTOM: flip rows before packing
    g = np.flipud(np.where(inside_hex, (np.clip(mip, lo, hi) - lo) / max(hi - lo, 1e-6), 0.0))
    rgba = np.dstack([g * 255, g * 255, g * 255, np.flipud(inside_hex) * 255]).astype(np.uint8)
    quad = pv.PolyData()
    quad.points = np.array([center_xyz - u_xyz * u_half - v_xyz * v_half,
                            center_xyz + u_xyz * u_half - v_xyz * v_half,
                            center_xyz + u_xyz * u_half + v_xyz * v_half,
                            center_xyz - u_xyz * u_half + v_xyz * v_half])
    quad.faces = np.array([4, 0, 1, 2, 3])
    quad.active_texture_coordinates = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    p1 = pv.Plotter(off_screen=True, window_size=WINDOW)
    p1.set_background(background_color, top=background_color)
    p1.add_mesh(quad, texture=pv.Texture(rgba), lighting=False)
    box1 = p1.add_mesh(pv.Box(bounds=[lo3[0], hi3[0], lo3[1], hi3[1], lo3[2], hi3[2]]),
                       style="wireframe", color=rv.frame_color_for_background("dark"), line_width=5)
    box1.SetPosition(*(dir_v * 600.0))  # toward camera: outline identical, never occluded
    set_cam(p1)
    p1.screenshot(args.out_dir + "/roi3d_cube_raw.png")
    p1.close()
    print("raw corner-view MIP panel done")

    # ---- panel 2: mask isosurface, bright red ----
    mask = zarr.open(args.mask_zarr + "/0", mode="r")[sl] > 0
    # drop small fragments with the same rule the skeleton uses: vessel_express removes
    # 6-connected components < dust_threshold(10) voxels of the 4x-downsampled mask,
    # i.e. < 10 * 64 native voxels. Components are sized in a padded crop so that pieces
    # near the cube face are not cut and misjudged; anything reaching the padded border is kept.
    if args.mask_min_voxels > 0:
        from scipy import ndimage
        pad = 60
        psl = tuple(slice(max(s.start - pad, 0), min(s.stop + pad, n)) for s, n in zip(sl, shape))
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
        print(f"mask fragments removed in/around cube: {int(small.sum())} components")
    verts, faces, _, _ = marching_cubes(mask.astype(np.uint8), level=0.5, spacing=tuple(RES))
    verts = verts[:, ::-1] + off_um[::-1]  # marching_cubes emits (z,y,x); pyvista wants (x,y,z)
    mesh = pv.PolyData(verts)
    mesh.faces = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64)
    p2 = pv.Plotter(off_screen=True, window_size=WINDOW)
    p2.set_background(background_color, top=background_color)
    p2.add_mesh(mesh, color="#ff4646", smooth_shading=True,
                ambient=0.3, diffuse=0.7, specular=0.4, specular_power=24)
    box2 = p2.add_mesh(pv.Box(bounds=[lo3[0], hi3[0], lo3[1], hi3[1], lo3[2], hi3[2]]),
                       style="wireframe", color=rv.frame_color_for_background("dark"), line_width=5)
    box2.SetPosition(*(dir_v * 600.0))
    set_cam(p2)
    p2.screenshot(args.out_dir + "/roi3d_cube_mask.png")
    p2.close()
    print("mask panel done, mesh", len(verts))

    # ---- panel 3: skeleton, diameter-colored tubes + branch point spheres ----
    branches, branch_points = rv.load_branches(args.swc_dir, max_files=0)
    def in_box(pts):
        return (pts >= lo3).all(axis=1) & (pts <= hi3).all(axis=1)
    branches = [b for b in branches if in_box(b.points).all()]  # fully-inside only
    bp = branch_points[np.array([in_box(pt[None, :]).any() for pt in branch_points])] if len(branch_points) else branch_points
    print("branches in cube:", len(branches), "branch pts:", len(bp))

    clim = rv.diameter_clim(branches, clip_percentiles=(2.0, 98.0))
    cmap = rv.resolve_colormap("bright")
    scalar_bar_args = {"title": "Diameter (um)", "position_x": 0.80, "position_y": 0.055,
                       "width": 0.14, "height": 0.022, "n_labels": 4, "color": text_color,
                       "title_font_size": 10, "label_font_size": 8, "fmt": "%.2g"}
    p3 = pv.Plotter(off_screen=True, window_size=WINDOW)
    p3.set_background(background_color, top=background_color)
    try:
        p3.enable_anti_aliasing("ssaa")
    except Exception:
        pass
    for index, branch in enumerate(branches):
        points = rv.prepare_branch_points(branch.points, min_spacing=1.0, smooth_window=5)
        if len(points) < 2:
            continue
        m = pv.Spline(points, max(len(points) * 12, len(points))) if len(points) >= 3 else pv.lines_from_points(points)
        m.point_data["diameter_um"] = np.full(m.n_points, np.clip(branch.mean_diameter_um, clim[0], clim[1]))
        p3.add_mesh(m, scalars="diameter_um", cmap=cmap, clim=clim, line_width=args.tube_width,
                    render_lines_as_tubes=True, show_scalar_bar=(index == 0),
                    scalar_bar_args=scalar_bar_args,
                    ambient=0.12, diffuse=0.78, specular=0.32, specular_power=32)
    if len(bp):
        cloud = pv.PolyData(bp)
        cloud["s"] = np.full(len(bp), 1.0)  # glyph scale must be 1.0: it MULTIPLIES the geom radius
        sphere = pv.Sphere(radius=args.branch_sphere_um, theta_resolution=16, phi_resolution=16)
        p3.add_mesh(cloud.glyph(geom=sphere, scale="s", orient=False, factor=1.0),
                    color=rv.branch_point_color_for_background("dark"), smooth_shading=True,
                    ambient=0.18, diffuse=0.68, specular=0.38, specular_power=36)
    box3 = p3.add_mesh(pv.Box(bounds=[lo3[0], hi3[0], lo3[1], hi3[1], lo3[2], hi3[2]]),
                       style="wireframe", color=rv.frame_color_for_background("dark"), line_width=5)
    box3.SetPosition(*(dir_v * 600.0))
    set_cam(p3)
    p3.screenshot(args.out_dir + "/roi3d_cube_vessels.png")
    p3.close()
    print("vessel panel done")


if __name__ == "__main__":
    main()
