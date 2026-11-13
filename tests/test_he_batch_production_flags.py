"""Production-flag contract tests for HE falsecolor_batch + stack_to_ims (fast)."""

import numpy as np
import pytest

from pipeline_modules.HE import stack_to_ims
from pipeline_modules.HE.falsecolor_batch import (
    display_rgb_to_od,
    parse_args,
    resolve_custom_colors,
)


def test_batch_cli_new_flags_defaults():
    args = parse_args(["--input_dir", "x"])
    assert args.align_channels is False
    assert args.align_max_shift == 32.0
    assert args.nuclei_rgb is None and args.cyto_rgb is None
    assert args.nuclei_strength is None and args.cyto_strength is None
    assert args.to_ims is None and args.from_source_ims is None


def test_batch_cli_new_flags_toggle():
    args = parse_args(
        [
            "--input_dir", "x",
            "--align_channels",
            "--nuclei_rgb", "9557C7", "--nuclei_strength", "0.4",
            "--cyto_rgb", "F3A1F3", "--cyto_strength", "0.8",
            "--to_ims", "out.ims", "--from_source_ims", "src.ims",
        ]
    )
    assert args.align_channels is True
    assert args.nuclei_rgb == "9557C7" and args.nuclei_strength == 0.4
    assert args.cyto_rgb == "F3A1F3" and args.cyto_strength == 0.8
    assert args.to_ims == "out.ims"


def test_resolve_custom_colors_full_and_partial():
    task = {
        "color_key": "HE", "nuclei_hue": "default",
        "nuclei_rgb": "9557C7", "nuclei_strength": 0.4,
        "cyto_rgb": "F3A1F3", "cyto_strength": 0.8,
    }
    colors = resolve_custom_colors(task)
    assert np.allclose(colors["nuclei"], display_rgb_to_od((149, 87, 199), 0.4))
    assert np.allclose(colors["cyto"], display_rgb_to_od((243, 161, 243), 0.8))

    partial = resolve_custom_colors(
        {"color_key": "HE", "nuclei_hue": "purple", "cyto_rgb": "FAB5EE"}
    )
    # missing side falls back to the hue preset
    assert np.allclose(partial["nuclei"], [0.085, 0.519, 0.051])

    assert resolve_custom_colors({"color_key": "HE", "nuclei_hue": "default"}) is None

    with pytest.raises(ValueError):
        resolve_custom_colors({"color_key": "IHC", "nuclei_rgb": "FFFFFF"})


class _FakeImageGroup:
    def __init__(self, attrs: dict):
        self.attrs = attrs

    def get(self, _key):
        return self


class _FakeHandle:
    def __init__(self, attrs: dict | None):
        self._attrs = attrs

    def get(self, _key):
        return None if self._attrs is None else _FakeImageGroup(self._attrs)


def test_inherit_geometry_scales_to_stack_dims():
    # source: X=5760 px over 10245.6 um, Y=8960 over 15903, Z=3520 over 7012
    s = lambda v: np.frombuffer(v.encode(), dtype="S1")
    handle = _FakeHandle(
        {
            "X": s("5760"), "Y": s("8960"), "Z": s("3520"),
            "ExtMin0": s("0.000000000"), "ExtMax0": s("10245.599609375"),
            "ExtMin1": s("0.000000000"), "ExtMax1": s("15903.000000000"),
            "ExtMin2": s("0.000000000"), "ExtMax2": s("7012.000000000"),
        }
    )
    attrs = stack_to_ims._inherit_source_geometry(handle, width=5692, height=8835, depth=3506)
    assert attrs["ExtMin0"] == "0.000000000"
    # 5692 * (10245.6/5760) = 10121.6...; 3506 * (7012/3520) = 6986.0
    assert abs(float(attrs["ExtMax0"]) - 5692 * 10245.599609375 / 5760) < 1e-3
    assert abs(float(attrs["ExtMax2"]) - 3506 * 7012.0 / 3520) < 1e-3


def test_discover_he_slices(tmp_path):
    (tmp_path / "a_000000_HE.tif").write_bytes(b"")
    (tmp_path / "a_000010_HE.tiff").write_bytes(b"")
    (tmp_path / "a_000010_ch1.tiff").write_bytes(b"")  # non-HE must be ignored
    (tmp_path / "notes.txt").write_bytes(b"")
    found = stack_to_ims.discover_he_slices(tmp_path)
    assert [p.name for _pre, p in found] == ["a_000000_HE.tif", "a_000010_HE.tiff"]
