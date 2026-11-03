"""Atlas-space bregma mapping must undo registration.flip_atlas."""
from __future__ import annotations

import sys
import types

import numpy as np

from pipeline_modules.visualization.bregma_mip import AtlasNativeMapper, _mirror_atlas_voxel


def test_mirror_atlas_voxel_only_flips_flagged_axes():
    ml, ap, dv = _mirror_atlas_voxel(
        10.0, 100.0, 200.0, atlas_shape_xyz=(456, 528, 320), flip_atlas=[False, True, False]
    )
    assert float(ml) == 10.0
    assert float(ap) == 427.0
    assert float(dv) == 200.0


def test_mapper_sends_mirrored_atlas_coordinates(monkeypatch):
    captured = []

    fake = types.ModuleType("ants")

    def apply_transforms_to_points(dim, points, transforms):
        captured.append(points.copy())
        return points

    fake.apply_transforms_to_points = apply_transforms_to_points
    monkeypatch.setitem(sys.modules, "ants", fake)

    mapper = AtlasNativeMapper(
        transformlist=[],
        convention="test",
        atlas_origin_xyz=(0.0, 0.0, 0.0),
        atlas_spacing_xyz=(1.0, 1.0, 1.0),
        fixed_origin_xyz=(0.0, 0.0, 0.0),
        fixed_spacing_xyz=(1.0, 1.0, 1.0),
        fixed_shape_xyz=(10, 10, 10),
        native_shape_zyx=(40, 40, 40),
        flip_atlas=(False, True, False),
        atlas_shape_xyz=(456, 528, 320),
    )
    mapper.map(np.array([10.0]), np.array([100.0]), np.array([200.0]))

    points = captured[0]
    assert float(points["x"][0]) == 200.0
    assert float(points["y"][0]) == 427.0
    assert float(points["z"][0]) == 10.0
