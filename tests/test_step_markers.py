from __future__ import annotations

import os
from pathlib import Path

import pytest

from pipeline_modules.utils.step_markers import (
    LEGACY_REASONS,
    mark_step_complete,
    step_is_complete,
    step_or_legacy_complete,
)


def _touch(path: Path, text: str = "x") -> None:
    path.write_text(text, encoding="utf-8")


def test_marker_valid_after_write(tmp_path: Path):
    out = tmp_path / "out.zarr"
    out.mkdir()
    inp = tmp_path / "input.tif"
    _touch(inp, "data")

    mark_step_complete([out], [inp], {"step": "demo", "threshold": 3})
    complete, reason = step_is_complete([out], [inp], {"step": "demo", "threshold": 3})
    assert complete, reason
    assert reason == "completion marker valid"


def test_incomplete_without_marker(tmp_path: Path):
    out = tmp_path / "out.zarr"
    out.mkdir()
    inp = tmp_path / "input.tif"
    _touch(inp)
    complete, reason = step_is_complete([out], [inp], {"step": "demo"})
    assert not complete
    assert "no completion marker" in reason


def test_input_change_invalidates(tmp_path: Path):
    out = tmp_path / "out.bin"
    _touch(out, "result")
    inp = tmp_path / "input.tif"
    _touch(inp, "v1")
    mark_step_complete([out], [inp], {"step": "demo"})

    complete, _ = step_is_complete([out], [inp], {"step": "demo"})
    assert complete

    _touch(inp, "v2")  # content AND mtime/size change
    complete, reason = step_is_complete([out], [inp], {"step": "demo"})
    assert not complete
    assert "input changed" in reason


def test_directory_input_change_invalidates(tmp_path: Path):
    out = tmp_path / "out.bin"
    _touch(out)
    src = tmp_path / "stack"
    src.mkdir()
    _touch(src / "a.tif", "a")
    mark_step_complete([out], [src], {"step": "demo"})

    complete, _ = step_is_complete([out], [src], {"step": "demo"})
    assert complete

    _touch(src / "b.tif", "b")  # new chunk in the store
    complete, reason = step_is_complete([out], [src], {"step": "demo"})
    assert not complete
    assert "input changed" in reason


def test_payload_change_invalidates(tmp_path: Path):
    out = tmp_path / "out.bin"
    _touch(out)
    inp = tmp_path / "input.tif"
    _touch(inp)
    mark_step_complete([out], [inp], {"step": "demo", "sigma": 1.0})

    complete, reason = step_is_complete([out], [inp], {"step": "demo", "sigma": 2.0})
    assert not complete
    assert "parameters changed" in reason


def test_deleted_output_invalidates(tmp_path: Path):
    out = tmp_path / "out.bin"
    _touch(out)
    inp = tmp_path / "input.tif"
    _touch(inp)
    mark_step_complete([out], [inp], {"step": "demo"})

    out.unlink()
    complete, reason = step_is_complete([out], [inp], {"step": "demo"})
    assert not complete
    assert "output missing" in reason


def test_deleted_input_invalidates(tmp_path: Path):
    out = tmp_path / "out.bin"
    _touch(out)
    inp = tmp_path / "input.tif"
    _touch(inp)
    mark_step_complete([out], [inp], {"step": "demo"})

    inp.unlink()
    complete, reason = step_is_complete([out], [inp], {"step": "demo"})
    assert not complete
    assert "input missing" in reason


def test_missing_input_recorded_then_invalid(tmp_path: Path):
    out = tmp_path / "out.bin"
    _touch(out)
    gone = tmp_path / "not_there.tif"
    marker = mark_step_complete([out], [gone], {"step": "demo"})
    assert marker.exists()

    complete, reason = step_is_complete([out], [gone], {"step": "demo"})
    assert not complete
    assert "input missing" in reason


def test_marker_records_code_version(tmp_path: Path):
    import json

    out = tmp_path / "out.bin"
    _touch(out)
    marker = mark_step_complete([out], [], {"step": "demo"})
    record = json.loads(marker.read_text(encoding="utf-8"))
    assert "code_version" in record
    assert "payload" in record and record["payload"] == {"step": "demo"}


def test_same_mtime_rewrite_detection(tmp_path: Path):
    """Directory signature must notice an in-place rewrite even when the dir
    mtime itself would not change (chunk file mtime is part of the signature)."""
    out = tmp_path / "out.bin"
    _touch(out)
    src = tmp_path / "stack"
    src.mkdir()
    chunk = src / "0.0.0"
    _touch(chunk, "v1")
    mark_step_complete([out], [src], {"step": "demo"})

    # rewrite in place; force identical size and (nearly) same mtime
    _touch(chunk, "v2")
    os.utime(chunk, ns=(chunk.stat().st_atime_ns, chunk.stat().st_mtime_ns - 1))
    complete, reason = step_is_complete([out], [src], {"step": "demo"})
    assert not complete


def test_legacy_outputs_backfill_marker(tmp_path: Path):
    """Outputs produced before markers existed are recorded, not redone."""
    out = tmp_path / "out.bin"
    _touch(out)
    inp = tmp_path / "input.tif"
    _touch(inp)

    complete, reason = step_or_legacy_complete([out], [inp], {"step": "demo"}, lambda: out.exists())
    assert complete
    assert "legacy" in reason
    # marker was backfilled; a direct check now passes cleanly
    complete, reason = step_is_complete([out], [inp], {"step": "demo"})
    assert complete, reason


def test_legacy_lane_does_not_rescue_invalidated_marker(tmp_path: Path):
    """A marker that exists but no longer matches must rerun — the legacy
    lane only applies when there is no usable marker at all."""
    out = tmp_path / "out.bin"
    _touch(out)
    inp = tmp_path / "input.tif"
    _touch(inp)
    mark_step_complete([out], [inp], {"step": "demo", "sigma": 1.0})

    complete, reason = step_or_legacy_complete([out], [inp], {"step": "demo", "sigma": 2.0}, lambda: out.exists())
    assert not complete
    assert "parameters changed" in reason


def test_legacy_lane_respects_outputs_present_false(tmp_path: Path):
    """Half-written legacy outputs (present()=False) still rerun."""
    out = tmp_path / "out.zarr"
    out.mkdir()  # exists but caller knows it is incomplete
    inp = tmp_path / "input.tif"
    _touch(inp)

    complete, reason = step_or_legacy_complete([out], [inp], {"step": "demo"}, lambda: False)
    assert not complete
