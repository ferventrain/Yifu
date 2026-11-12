"""Custom stain-color plumbing for HE falsecolor (fast, no CUDA render needed)."""

import numpy as np
import pytest

import pipeline_modules.HE.falsecolor_batch as fb


def test_display_rgb_to_od_reproduces_purple_preset():
    """Presets were derived from reference display colors via OD=-ln(c/255):
    cyto max-normalized to 1.0, nuclei scaled ~0.52 (moderate stain).
    The picker conversion must invert back onto the same vectors."""
    nuc = fb.display_rgb_to_od((129, 4, 170), strength=0.519)
    assert np.allclose(nuc, fb.HE_COLOR_PRESETS["purple"]["nuclei"], atol=2e-3)
    cyto = fb.display_rgb_to_od((245, 179, 249), strength=1.0)
    assert np.allclose(cyto, fb.HE_COLOR_PRESETS["purple"]["cyto"], atol=2e-3)


def test_display_od_round_trip_keeps_channel_order():
    rgb = (200, 90, 40)  # descending R>G>B -> most absorbed channel is B
    od = fb.display_rgb_to_od(rgb)
    assert int(np.argmax(od)) == 2
    back = fb.od_to_display_rgb(od)
    assert back[0] > back[1] > back[2]
    assert all(0 <= c <= 255 for c in back)


@pytest.fixture()
def capture_falsecolor(monkeypatch):
    captured = {}

    def fake_rapid(nuclei, cyto, nuc_color, cyto_color, **kwargs):
        captured["nuc"] = [float(v) for v in nuc_color]
        captured["cyto"] = [float(v) for v in cyto_color]
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def fake_cpu(nuclei, cyto, **kwargs):
        captured["cpu_color_settings"] = kwargs.get("color_settings")
        return np.zeros((4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr(fb.falsecolor, "rapidFalseColor", fake_rapid)
    monkeypatch.setattr(fb.falsecolor, "falseColor", fake_cpu)
    return captured


def _render_kwargs():
    return dict(
        backend="gpu",
        color_key="HE",
        nuclei_hue="purple",
        nuc_threshold=100,
        cyto_threshold=100,
        nuc_normfactor=8200,
        cyto_normfactor=2100,
    )


def test_render_falsecolor_custom_settings_override(capture_falsecolor):
    custom = {"nuclei": [0.1, 0.2, 0.3], "cyto": [0.4, 0.5, 0.6]}
    fb.render_falsecolor(
        np.zeros((4, 4), np.float32),
        np.zeros((4, 4), np.float32),
        color_settings=custom,
        **_render_kwargs(),
    )
    assert capture_falsecolor["nuc"] == [0.1, 0.2, 0.3]
    assert capture_falsecolor["cyto"] == [0.4, 0.5, 0.6]


def test_render_falsecolor_preset_path_unchanged(capture_falsecolor):
    """Without color_settings the hue-preset vectors must reach the renderer as before."""
    fb.render_falsecolor(
        np.zeros((4, 4), np.float32), np.zeros((4, 4), np.float32), **_render_kwargs()
    )
    assert capture_falsecolor["nuc"] == list(fb.HE_COLOR_PRESETS["purple"]["nuclei"])
    assert capture_falsecolor["cyto"] == list(fb.HE_COLOR_PRESETS["purple"]["cyto"])


def test_render_falsecolor_cpu_path_receives_resolved_settings(capture_falsecolor):
    kw = _render_kwargs()
    kw["backend"] = "cpu"
    fb.render_falsecolor(np.zeros((4, 4), np.float32), np.zeros((4, 4), np.float32), **kw)
    settings = capture_falsecolor["cpu_color_settings"]
    assert settings is not None
    assert list(settings["nuclei"]) == list(fb.HE_COLOR_PRESETS["purple"]["nuclei"])
