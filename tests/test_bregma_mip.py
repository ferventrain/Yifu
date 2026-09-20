import unittest
from pathlib import Path

import numpy as np

from pipeline_modules.utils.zarr_io import create_output_zarr
from pipeline_modules.visualization.bregma_mip import (
    AnchorError,
    ApAnchor,
    compute_coronal_slab_mip,
    coronal_row_ap_centers,
    label_boundary_mask,
    render_bregma_mips,
    resolve_manual_anchor,
    resolve_voxel_xyz_um,
)


def _make_signal_zarr(path: Path, shape=(8, 32, 16), ap_vox_um=2.0):
    """Volume whose value encodes its AP index; a bright slab at ap=20."""
    _, arr = create_output_zarr(path, shape, (4, 8, 8), "uint16")
    ramp = np.arange(shape[1], dtype=np.uint16)[None, :, None]
    arr[:, :, :] = np.broadcast_to(ramp, shape)
    arr[:, 20, :] = 1000
    return arr


class SlabMipTests(unittest.TestCase):
    def test_window_max_uses_own_ap_window_per_row(self):
        n_dv, n_ap, n_ml = 6, 40, 5
        volume = np.zeros((n_dv, n_ap, n_ml), dtype=np.uint16)
        volume[:, 10, :] = 10
        volume[:, 30, :] = 30
        centers = np.full(n_dv, 20.0)
        # half window 5 -> rows cover AP 15..25 -> only the 10-valued plane is excluded,
        # so MIP must be 0 unless the window reaches it.
        mip = compute_coronal_slab_mip(volume, row_ap_centers=centers, half_window_vox=5.0)
        self.assertEqual(mip.shape, (n_dv, n_ml))
        self.assertTrue(np.all(mip == 0))
        # window 9..31 now includes both planes; the max wins.
        mip = compute_coronal_slab_mip(volume, row_ap_centers=centers, half_window_vox=10.5)
        self.assertTrue(np.all(mip == 30))

    def test_tilt_shears_window_per_row(self):
        n_dv, n_ap, n_ml = 5, 40, 1
        volume = np.zeros((n_dv, n_ap, n_ml), dtype=np.uint16)
        volume[:, 35, :] = 35  # only reachable when a row's window shears far anterior/posterior
        centers = coronal_row_ap_centers(
            n_dv=n_dv, base_ap_index=20.0, anterior="low", tilt_deg=45.0,
            dv_vox_um=1.0, ap_vox_um=1.0, tilt_ref_row=0.0,
        )
        # anterior="low" + positive tilt -> ventral rows sample lower AP indices,
        # so the 35-plane is never inside any window.
        self.assertTrue(np.all(centers <= 20.0))
        centers = coronal_row_ap_centers(
            n_dv=n_dv, base_ap_index=20.0, anterior="low", tilt_deg=-45.0,
            dv_vox_um=1.0, ap_vox_um=1.0, tilt_ref_row=0.0,
        )
        self.assertAlmostEqual(float(centers[-1]), 24.0)
        # window center 24 with the 35-plane still outside; widen to reach it
        mip = compute_coronal_slab_mip(volume, row_ap_centers=centers, half_window_vox=11.5)
        self.assertTrue(np.all(mip[-1] == 35))
        self.assertTrue(np.all(mip[0] == 0))

    def test_window_clips_to_volume_bounds(self):
        volume = np.full((3, 10, 2), 7, dtype=np.uint16)
        centers = np.array([-50.0, 5.0, 60.0])
        mip = compute_coronal_slab_mip(volume, row_ap_centers=centers, half_window_vox=4.0)
        self.assertTrue(np.all(mip == 7))

    def test_zero_thickness_takes_single_slice(self):
        volume = np.zeros((2, 10, 1), dtype=np.uint16)
        volume[:, 3, :] = 3
        volume[:, 4, :] = 4
        centers = np.array([3.4, 3.4])
        mip = compute_coronal_slab_mip(volume, row_ap_centers=centers, half_window_vox=0.0)
        self.assertTrue(np.all(mip == 3))


class AnchorTests(unittest.TestCase):
    def test_index_for_sign_conventions(self):
        anchor_low = ApAnchor(index=100.0, anterior="low", source="test")
        anchor_high = ApAnchor(index=100.0, anterior="high", source="test")
        # anterior positive: low-end anterior means the index decreases.
        self.assertAlmostEqual(anchor_low.index_for(1.0, 2.0), 100.0 - 500.0)
        self.assertAlmostEqual(anchor_high.index_for(1.0, 2.0), 100.0 + 500.0)
        # level halves everything on the rendered grid.
        self.assertAlmostEqual(anchor_low.index_for(1.0, 2.0, level=1), (100.0 - 500.0) / 2.0)

    def test_manual_anchor_offset_and_errors(self):
        anchor = resolve_manual_anchor(
            n_ap_level=100, n_ap_level0=100, level=0, ap_vox_um=2.0,
            bregma_index=None, bregma_offset_mm=0.1, anterior="low",
        )
        self.assertAlmostEqual(anchor.index, 50.0)
        anchor = resolve_manual_anchor(
            n_ap_level=100, n_ap_level0=100, level=0, ap_vox_um=2.0,
            bregma_index=None, bregma_offset_mm=0.02, anterior="high",
        )
        self.assertAlmostEqual(anchor.index, 89.0)
        with self.assertRaises(AnchorError):
            resolve_manual_anchor(
                n_ap_level=100, n_ap_level0=100, level=0, ap_vox_um=2.0,
                bregma_index=None, bregma_offset_mm=None, anterior="low",
            )
        with self.assertRaises(AnchorError):
            resolve_manual_anchor(
                n_ap_level=100, n_ap_level0=100, level=0, ap_vox_um=2.0,
                bregma_index=None, bregma_offset_mm=10.0, anterior="low",
            )


class HelperTests(unittest.TestCase):
    def test_label_boundary_mask_marks_transitions(self):
        labels = np.array([[1, 1, 2], [1, 1, 2], [0, 0, 0]], dtype=np.int32)
        mask = label_boundary_mask(labels)
        self.assertTrue(mask[0, 1] and mask[0, 2])  # 1|2 transition
        self.assertTrue(mask[1, 1] and mask[1, 2])
        self.assertTrue(mask[1, 0] and mask[2, 0])  # 1|0 transition
        self.assertFalse(mask[0, 0])

    def test_resolve_voxel_explicit_flag(self):
        voxel, source = resolve_voxel_xyz_um(Path("nowhere.zarr"), explicit="1.0, 2.0, 3.0")
        self.assertEqual(voxel, (1.0, 2.0, 3.0))
        self.assertEqual(source, "flag")
        with self.assertRaises(ValueError):
            resolve_voxel_xyz_um(Path("nowhere.zarr"), explicit="1,2")


class RenderEndToEndTests(unittest.TestCase):
    def test_manual_render_produces_png_and_payload(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            sample_dir = tmp / "sampleA"
            sample_dir.mkdir()
            _make_signal_zarr(sample_dir / "ch1.zarr")

            payload = render_bregma_mips(
                zarr_path=sample_dir / "ch1.zarr",
                bregma_mm=[0.0],
                thickness_mm=0.16,
                bregma_offset_mm=0.04,  # 20 voxels * 2 um -> AP index 20
                anterior="low",
                voxel_xyz_um="1.8, 2.0, 2.0",
                use_transforms=False,
                output_dir=sample_dir / "vis",
                dv_block=4,
            )
            self.assertEqual(payload["anchor_source"], "bregma-offset")
            self.assertEqual(len(payload["outputs"]), 1)
            out = Path(str(payload["outputs"][0]["output"]))
            self.assertTrue(out.exists() and out.stat().st_size > 0)
            # Anchor lands at AP index 20 where the bright slab lives.
            self.assertAlmostEqual(float(payload["outputs"][0]["ap_index_center"]), 20.0)

    def test_bregma_out_of_range_raises(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            sample_dir = Path(tmp) / "sampleB"
            sample_dir.mkdir()
            _make_signal_zarr(sample_dir / "ch1.zarr")
            with self.assertRaises(AnchorError):
                render_bregma_mips(
                    zarr_path=sample_dir / "ch1.zarr",
                    bregma_mm=[2.0],
                    thickness_mm=0.1,
                    bregma_offset_mm=0.04,
                    anterior="low",
                    voxel_xyz_um="1.8, 2.0, 2.0",
                    use_transforms=False,
                    output_dir=sample_dir / "vis",
                )

    def test_label_overlay_used_when_shapes_match(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            sample_dir = Path(tmp) / "sampleC"
            sample_dir.mkdir()
            shape = (8, 32, 16)
            _make_signal_zarr(sample_dir / "ch1.zarr", shape=shape)
            _, label_arr = create_output_zarr(
                sample_dir / "upsampled_atlas_label.zarr", shape, (4, 8, 8), "uint32"
            )
            label_arr[:, :, :] = 0
            label_arr[:, :16, :] = 100
            label_arr[:, 16:, :] = 200

            payload = render_bregma_mips(
                zarr_path=sample_dir / "ch1.zarr",
                bregma_mm=[0.0],
                thickness_mm=0.16,
                bregma_index=20.0,
                anterior="low",
                voxel_xyz_um="1.8, 2.0, 2.0",
                use_transforms=False,
                output_dir=sample_dir / "vis",
                dv_block=4,
            )
            self.assertTrue(payload["outputs"][0]["boundary_overlay"])
            self.assertTrue(Path(str(payload["outputs"][0]["output"])).exists())


if __name__ == "__main__":
    unittest.main()
