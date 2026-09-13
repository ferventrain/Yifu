import numpy as np

from pipeline_modules.segmentation.rs_fish import (
    detect_spots_rsfish,
    match_points,
    radial_center_2d,
    radial_center_3d,
)


def _gaussian_blob(shape, center, sigma=1.6, amp=1.0):
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    z, y, x = center
    return amp * np.exp(-((zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma * sigma))


def test_radial_center_2d_finds_blob_peak():
    yy, xx = np.mgrid[0:15, 0:15]
    img = np.exp(-((yy - 7.4) ** 2 + (xx - 8.1) ** 2) / (2 * 1.5**2))
    row, col = radial_center_2d(img)
    assert abs(row - 7.4) < 0.6
    assert abs(col - 8.1) < 0.6


def test_radial_center_3d_finds_blob_peak():
    vol = _gaussian_blob((15, 15, 15), (6.4, 7.2, 8.1), sigma=1.5, amp=1.0)
    z, y, x = radial_center_3d(vol)
    assert abs(z - 6.4) < 0.8
    assert abs(y - 7.2) < 0.8
    assert abs(x - 8.1) < 0.8


def test_detect_synthetic_spots():
    vol = np.random.default_rng(0).normal(0.05, 0.02, size=(32, 32, 32)).astype(np.float32)
    centers = [(10, 12, 14), (20, 18, 9)]
    for c in centers:
        vol += _gaussian_blob(vol.shape, c, sigma=1.4, amp=1.0)
    spots = detect_spots_rsfish(vol, sigma=1.4, dog_threshold=0.02, min_distance=3, min_inlier_ratio=0.05)
    assert len(spots) >= 1
    gt = np.asarray(centers, dtype=np.float32)
    tp, fp, fn = match_points(spots, gt, max_dist=3.0)
    assert tp >= 1
