from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

from pipeline_modules.tubule_reconstruction.mask_downsample import (
    block_reduce_binary,
    default_occupancy_threshold,
    estimate_downsampled_memory,
)
from pipeline_modules.tubule_reconstruction.vessel_express_reconstruction import (
    clean_skeleton_topology,
    despike_thick_vessels,
    fill_small_holes,
    merge_nearby_branch_points,
    prune_terminal_spurs,
    reconstruct_vessel_express,
    resolve_triple_cliques,
    run_preview_chunks,
    skeletonize_lee,
    suppress_collinear_kinks,
    upsample_coarse_coords,
)


def _xy_width(binary: np.ndarray) -> float:
    widths = []
    for plane in binary:
        ys, xs = np.where(plane)
        if len(ys) == 0:
            continue
        widths.append(int(ys.max() - ys.min()) + 1)
        widths.append(int(xs.max() - xs.min()) + 1)
    return float(np.mean(widths)) if widths else 0.0


def test_occupancy_threshold_keeps_axis_aligned_filament():
    assert default_occupancy_threshold(4) == 0.0625
    block = np.zeros((4, 4, 4), dtype=bool)
    block[:, 1, 1] = True
    assert float(block.mean()) == default_occupancy_threshold(4)
    assert bool(block_reduce_binary(block, 4, method="occupancy")[0, 0, 0])
    assert not bool(block_reduce_binary(block, 4, method="majority")[0, 0, 0])


def test_occupancy_drops_grazing_voxels_that_max_pool_keeps():
    volume = np.zeros((48, 32, 32), dtype=bool)
    yy, xx = np.ogrid[:32, :32]
    volume[:, (yy - 16) ** 2 + (xx - 16) ** 2 <= 6**2] = True
    max_pool = block_reduce_binary(volume, 4, method="max_pool")
    occupancy = block_reduce_binary(volume, 4, method="occupancy")
    true_width = _xy_width(volume)
    max_width = _xy_width(max_pool) * 4
    occ_width = _xy_width(occupancy) * 4
    assert max_width >= occ_width
    assert abs(occ_width - true_width) <= abs(max_width - true_width)


def test_estimate_4x_whole_brain_fits_uint8_array():
    # Allen-like FOV at 1.8,1.8,2.0 um: ~13.2 x 11.4 x 8.0 mm
    native = (4000, 6333, 7333)
    report = estimate_downsampled_memory(native, 4)
    assert report["uint8_gib"] < 8.0
    assert report["lee_peak_gib_est"] < 24.0
    assert report["fits_128gib_uint8_array"]
    native_report = estimate_downsampled_memory(native, 1)
    assert native_report["uint8_gib"] > 100.0


def test_lee_skeleton_is_thin_tube(tmp_path: Path):
    mask = np.zeros((24, 24, 40), dtype=bool)
    mask[8:17, 8:17, 2:38] = True
    skeleton = skeletonize_lee(mask)
    coords = np.argwhere(skeleton)
    assert len(coords) > 10
    assert np.abs(coords[:, 0] - 12).mean() < 2.0
    assert np.abs(coords[:, 1] - 12).mean() < 2.0


def test_native_radius_recovers_cylinder_radius(tmp_path: Path):
    root = zarr.open_group(str(tmp_path / "mask.zarr"), mode="w")
    volume = np.zeros((48, 32, 32), dtype=np.uint8)
    yy, xx = np.ogrid[:32, :32]
    volume[:, (yy - 16) ** 2 + (xx - 16) ** 2 <= 5**2] = 1
    root.create_dataset("0", data=volume, chunks=(16, 16, 16), dtype=np.uint8)

    out_dir = tmp_path / "ve"
    summary = reconstruct_vessel_express(
        tmp_path / "mask.zarr",
        out_dir,
        resolution_xyz=(1.0, 1.0, 1.0),
        downsample_factor=4,
        downsample_method="occupancy",
        dust_threshold=0,
        keep_downsampled_mask=True,
    )
    vertices = np.loadtxt(out_dir / "skeleton_vertices.csv", delimiter=",", skiprows=1)
    if vertices.ndim == 1:
        vertices = vertices[None, :]
    radii = vertices[:, 5]
    assert summary["num_skeleton_voxels"] > 5
    assert 3.5 <= float(np.nanmean(radii)) <= 6.5
    written = json.loads((out_dir / "vessel_network_summary.json").read_text(encoding="utf-8"))
    assert written["radius_source"] == "native_edt"
    assert written["downsample_method"] == "occupancy"


def test_upsample_coarse_coords_lands_in_native_cell():
    pts = upsample_coarse_coords(np.array([[1, 2, 3]]), (40, 40, 40), 4)
    assert list(pts[0]) == [6, 10, 14]


def test_preview_chunks_max_pool_4x(tmp_path: Path):
    root = zarr.open_group(str(tmp_path / "mask.zarr"), mode="w")
    volume = np.zeros((16, 16, 16), dtype=np.uint8)
    volume[:, 8, 8] = 1
    root.create_dataset("0", data=volume, chunks=(8, 8, 8), dtype=np.uint8)
    summary = run_preview_chunks(
        tmp_path / "mask.zarr",
        tmp_path / "preview",
        ["0.1.1"],
        downsample_factor=4,
        downsample_method="max_pool",
        dust_threshold=1,
        view=False,
    )
    assert summary["num_skeleton_voxels"] >= 1
    assert Path(tmp_path / "preview" / "skeleton_native.npy").is_file()


def test_prune_terminal_spurs_drops_one_voxel_tick():
    coords = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 0, 3],
            [0, 1, 1],
        ],
        dtype=np.int32,
    )
    edges = np.array([[0, 1], [1, 2], [2, 3], [1, 4]], dtype=np.int32)
    degrees = np.array([1, 3, 2, 1, 1], dtype=np.int32)
    scale = (8.0, 7.2, 7.2)
    new_coords, new_edges, new_degrees, n_pruned = prune_terminal_spurs(
        coords, edges, degrees, scale, max_length_um=10.0
    )
    assert n_pruned == 1
    assert len(new_coords) == 4
    assert int((new_degrees >= 3).sum()) == 0


def test_fill_small_holes_fills_enclosed_cavity_only():
    binary = np.ones((7, 7, 7), dtype=bool)
    binary[3, 3, 3] = False
    filled, n_holes, n_vox = fill_small_holes(binary, max_voxels=4)
    assert n_holes == 1
    assert n_vox == 1
    assert bool(filled[3, 3, 3])
    open_hole = np.ones((7, 7, 7), dtype=bool)
    open_hole[0, 3, 3] = False
    filled_open, n_open, _ = fill_small_holes(open_hole, max_voxels=4)
    assert n_open == 0
    assert not bool(filled_open[0, 3, 3])


def test_despike_thick_keeps_one_voxel_filament():
    vol = np.zeros((12, 12, 16), dtype=bool)
    vol[4:10, 4:10, :] = True
    vol[0, 6, :] = True
    vol[6, 3, 8] = True
    out, removed = despike_thick_vessels(vol, thin_max_voxels=2)
    assert bool(out[0, 6, 8])
    assert removed >= 1
    assert not bool(out[6, 3, 8])


def test_triple_clique_loses_longest_edge():
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.int32)
    edges = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int32)
    degrees = np.array([2, 2, 2], dtype=np.int32)
    new_coords, new_edges, new_degrees, n_removed = resolve_triple_cliques(
        coords, edges, degrees, (1.0, 1.0, 1.0)
    )
    assert n_removed == 1
    assert len(new_edges) == 2
    assert int((new_degrees >= 3).sum()) == 0


def test_collinear_kink_is_not_a_branch():
    coords = np.array([[0, 0, 0], [0, 0, -1], [0, 0, 1], [0, 1, 1]], dtype=np.int32)
    edges = np.array([[0, 1], [0, 2], [0, 3]], dtype=np.int32)
    degrees = np.array([3, 1, 1, 1], dtype=np.int32)
    new_coords, new_edges, new_degrees, n_removed = suppress_collinear_kinks(
        coords, edges, degrees, (1.0, 1.0, 1.0)
    )
    assert n_removed == 1
    assert int((new_degrees >= 3).sum()) == 0


def test_clean_topology_merges_close_branch_points():
    coords = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 2],
            [0, 1, 2],
            [1, 0, 2],
        ],
        dtype=np.int32,
    )
    edges = np.array(
        [[0, 1], [0, 2], [0, 3], [1, 4], [4, 5], [4, 6]],
        dtype=np.int32,
    )
    degrees = np.array([3, 2, 1, 1, 3, 1, 1], dtype=np.int32)
    _, _, new_degrees, stats = clean_skeleton_topology(
        coords,
        edges,
        degrees,
        (8.0, 7.2, 7.2),
        prune_spurs_max_length_um=0.0,
        merge_branch_points_distance_um=20.0,
    )
    assert stats["merged_branch_points"] >= 1
    assert int((new_degrees >= 3).sum()) == 1


def test_native_edt_radius_on_cylinder(tmp_path: Path):
    from pipeline_modules.tubule_reconstruction.vessel_express_reconstruction import sample_native_edt_radii

    vol = np.zeros((40, 40, 40), dtype=np.uint8)
    yy, xx = np.ogrid[:40, :40]
    vol[:, (yy - 20) ** 2 + (xx - 20) ** 2 <= 6**2] = 1
    root = zarr.open_group(str(tmp_path / "mask.zarr"), mode="w")
    root.create_dataset("0", data=vol, chunks=(16, 16, 16))
    pts = np.array([[20, 20, 20]], dtype=np.int64)
    radii = sample_native_edt_radii(
        root["0"],
        pts,
        factor=1,
        resolution_xyz=(1.0, 1.0, 1.0),
        halo_zyx=(4, 4, 4),
        workers=1,
        mask_zarr_path=str(tmp_path / "mask.zarr"),
    )
    assert float(radii[0]) > 4.5
    assert float(radii[0]) < 8.0
