"""Interactive cFos visual report web application."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from starlette.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pipeline_modules.utils.data_paths import get_yifu_data_dir
from pipeline_modules.visualization.cfos_report_data import (
    DEFAULT_CFG,
    build_report_bundle,
    collect_subtree_region_ids,
    export_region_metrics_csv,
    export_slice_bookmarks_zip,
    render_metric_slice_png_bytes,
    render_metric_slice_png_with_layout,
    resolve_display_region_id,
)
from pipeline_modules.visualization.cfos_report_summary import (
    build_summary_payload,
    read_summary_json_if_exists,
    summary_json_path,
    write_summary_json,
)
from pipeline_modules.visualization.cfos_report_group_stats import (
    build_group_analysis_payload,
    build_pairwise_manifest,
    export_differential_regions_csv,
    load_group_manifest,
    parse_group_manifest_json,
)
from pipeline_modules.visualization.cfos_report_spatial import (
    ALLEN_ROOT_REGION_ID,
    build_brain_outline_surface_payload,
    build_points_viewer_payload,
    build_region_pick_payload,
    build_region_slice_focus_payload,
    build_atlas_region_centroids_payload,
    build_region_surface_payload,
    build_spatial_payload,
)
from pipeline_modules.visualization.heatmap import sample_has_density_excel

STATIC_DIR = APP_DIR / "static"
ASSET_VERSION = "29"


class NoCacheStaticFiles(StaticFiles):
    """Serve UI assets without aggressive browser caching during development."""

    async def get_response(self, path: str, scope):  # type: ignore[override]
        response = await super().get_response(path, scope)
        if path.endswith((".js", ".css", ".html", ".map")):
            response.headers["Cache-Control"] = "no-cache"
        return response


def _prewarm_default_sample() -> None:
    defaults = resolve_bootstrap_defaults()
    sample_dir = defaults.get("sample_dir")
    if not sample_dir:
        return
    signal_ch = str(defaults.get("signal_ch") or "ch1")
    try:
        _load_bundle(sample_dir, signal_ch=signal_ch, refresh=False)
        print(f"Prewarmed report cache for: {sample_dir} ({signal_ch})", file=sys.stderr)
    except Exception as exc:
        print(f"Prewarm skipped for {sample_dir}: {exc}", file=sys.stderr)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _prewarm_default_sample()
    yield


app = FastAPI(title="cFos Visual Report", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_report_cache: dict[str, Any] = {}
_group_analysis_cache: dict[str, Any] = {}
_startup_sample_dir: str | None = os.environ.get("CFOS_DEFAULT_SAMPLE_DIR", "").strip() or None
_startup_signal_ch: str = os.environ.get("CFOS_DEFAULT_SIGNAL_CH", "ch1").strip() or "ch1"


def _explicit_sample_dir_candidates() -> list[Path]:
    candidates: list[Path] = []
    for key in ("CFOS_DEFAULT_SAMPLE_DIR", "CFOS_SAMPLE_DIR"):
        raw = os.environ.get(key, "").strip()
        if raw:
            candidates.append(Path(raw))
    if _startup_sample_dir:
        candidates.append(Path(_startup_sample_dir))
    return candidates


def _sample_scan_roots() -> list[Path]:
    roots: list[Path] = []
    for key in ("CFOS_SAMPLES_ROOT", "CFOS_SAMPLES_DIR"):
        raw = os.environ.get(key, "").strip()
        if raw:
            roots.append(Path(raw))
    data_dir = get_yifu_data_dir(required=False)
    if data_dir is not None:
        roots.append(data_dir)
        samples_sub = data_dir / "samples"
        if samples_sub.is_dir():
            roots.append(samples_sub)
    return roots


def _find_newest_sample_with_density(roots: list[Path], signal_ch: str) -> tuple[Path | None, str]:
    best: Path | None = None
    best_mtime = 0.0
    best_signal_ch = signal_ch

    def consider(path: Path, channel: str) -> None:
        nonlocal best, best_mtime, best_signal_ch
        if not path.is_dir():
            return
        if not sample_has_density_excel(path, channel):
            return
        mtime = path.stat().st_mtime
        if mtime > best_mtime:
            best = path
            best_mtime = mtime
            best_signal_ch = channel

    channels = [signal_ch]
    if signal_ch != "ch2":
        channels.append("ch2")
    if signal_ch != "ch1":
        channels.append("ch1")

    for root in roots:
        if not root.is_dir():
            continue
        for channel in channels:
            consider(root, channel)
            try:
                children = list(root.iterdir())
            except OSError:
                children = []
            for child in children:
                if child.is_dir():
                    consider(child, channel)
    if best is None:
        return None, signal_ch
    return best.resolve(), best_signal_ch


def resolve_bootstrap_defaults() -> dict[str, Any]:
    signal_ch = os.environ.get("CFOS_DEFAULT_SIGNAL_CH", _startup_signal_ch).strip() or "ch1"
    for candidate in _explicit_sample_dir_candidates():
        if not candidate.is_dir():
            continue
        for channel in (signal_ch, "ch1", "ch2"):
            if sample_has_density_excel(candidate, channel):
                return {
                    "sample_dir": str(candidate.resolve()),
                    "signal_ch": channel,
                    "source": "default_sample_dir",
                }

    discovered, resolved_channel = _find_newest_sample_with_density(_sample_scan_roots(), signal_ch)
    if discovered is not None:
        return {
            "sample_dir": str(discovered),
            "signal_ch": resolved_channel,
            "source": "samples_root_scan",
        }
    return {"sample_dir": None, "signal_ch": signal_ch, "source": None}


class GroupAnalyzeRequest(BaseModel):
    manifest_path: str | None = None
    manifest_json: str | None = None
    sample_a_dir: str | None = None
    sample_b_dir: str | None = None
    sample_a_label: str | None = None
    sample_b_label: str | None = None
    signal_ch_a: str | None = None
    signal_ch_b: str | None = None
    level: str = "Level_8"
    metric: str = "cfos_count"
    group_a: str | None = None
    group_b: str | None = None
    cfg_path: str | None = None
    top_n: int = Field(default=36, ge=8, le=120)
    focus_region_id: int | None = None
    heatmap_mode: str = "differential"


class SliceBookmark(BaseModel):
    plane: str = "coronal"
    coordinate_system: str = "index"
    coordinate: float
    label: str = ""
    bregma_mm: float | None = None
    region_id: int | None = None
    color_modes: list[str] | None = None


class ExportSliceBookmarksRequest(BaseModel):
    sample_dir: str
    input_excel: str | None = None
    signal_ch: str = "ch1"
    metric: str = "cfos_count"
    level: str | None = None
    color_modes: list[str] = Field(default_factory=lambda: ["region", "signal"])
    bookmarks: list[SliceBookmark]
    focus_region_id: int | None = None
    dpi: int = Field(default=150, ge=72, le=300)


def _resolve_group_manifest(
    *,
    manifest_path: str | None,
    manifest_json: str | None,
    sample_a_dir: str | None = None,
    sample_b_dir: str | None = None,
    sample_a_label: str | None = None,
    sample_b_label: str | None = None,
    signal_ch_a: str | None = None,
    signal_ch_b: str | None = None,
) -> list[dict[str, Any]]:
    provided = [
        bool(manifest_path),
        bool(manifest_json),
        bool(sample_a_dir and sample_b_dir),
    ]
    if sum(provided) != 1:
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of: manifest_path, manifest_json, or sample_a_dir + sample_b_dir.",
        )
    if manifest_path and manifest_json:
        raise HTTPException(status_code=400, detail="Provide either manifest_path or manifest_json, not both.")
    if sample_a_dir and sample_b_dir:
        try:
            return build_pairwise_manifest(
                sample_a_dir,
                sample_b_dir,
                signal_ch_a=signal_ch_a or "ch1",
                signal_ch_b=signal_ch_b,
                group_a=sample_a_label,
                group_b=sample_b_label,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    if manifest_path:
        try:
            return load_group_manifest(manifest_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    if manifest_json:
        try:
            return parse_group_manifest_json(manifest_json)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    raise HTTPException(status_code=400, detail="manifest_path or manifest_json is required.")


def _cache_key(sample_dir: str, input_excel: str | None) -> str:
    return f"{Path(sample_dir).resolve()}|{input_excel or ''}"


def _load_bundle(
    sample_dir: str,
    *,
    input_excel: str | None = None,
    signal_ch: str = "ch1",
    group: str | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    key = _cache_key(sample_dir, input_excel)
    if not refresh and key in _report_cache:
        return _report_cache[key]

    try:
        bundle = build_report_bundle(
            sample_dir,
            input_excel=input_excel,
            signal_ch=signal_ch,
            group_label=group,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _report_cache[key] = bundle
    return bundle


@app.get("/api/bootstrap")
def bootstrap() -> dict[str, Any]:
    """Resolved sample for product mode (CLI/env or scanned data roots)."""
    payload = resolve_bootstrap_defaults()
    sample_dir = payload.get("sample_dir")
    if not sample_dir:
        return payload
    signal_ch = str(payload.get("signal_ch") or "ch1")
    try:
        bundle = _load_bundle(sample_dir, signal_ch=signal_ch)
        payload["sample_id"] = bundle["sample"]["sample_id"]
        payload["group"] = bundle["sample"].get("group")
    except Exception:
        pass
    return payload


@app.get("/favicon.ico", include_in_schema=False)
def favicon_ico() -> FileResponse:
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


@app.get("/favicon.svg", include_in_schema=False)
def favicon_svg() -> FileResponse:
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


@app.get("/api/report")
def get_report(
    sample_dir: str = Query(..., description="Sample directory path"),
    input_excel: str | None = Query(None),
    signal_ch: str = Query("ch1"),
    group: str | None = Query(None),
    refresh: bool = Query(False),
) -> dict[str, Any]:
    return _load_bundle(
        sample_dir,
        input_excel=input_excel,
        signal_ch=signal_ch,
        group=group,
        refresh=refresh,
    )


def ensure_summary_json(
    sample_dir: str,
    *,
    input_excel: str | None = None,
    signal_ch: str = "ch1",
    group: str | None = None,
    level: str | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """Return summary JSON from disk, or build report + write summary if missing."""
    summary_path = summary_json_path(sample_dir, signal_ch)
    if not refresh:
        cached = read_summary_json_if_exists(summary_path)
        headline = (cached or {}).get("headline_stats") or {}
        if (
            cached is not None
            and cached.get("systems")
            and headline.get("signal_volume_um3") is not None
            and headline.get("leaf_region_scope") == "all_levels_finest_available"
            and headline.get("systems_scope") == "coarse_excel_lookup"
        ):
            return cached

    bundle = _load_bundle(
        sample_dir,
        input_excel=input_excel,
        signal_ch=signal_ch,
        group=group,
        refresh=True,
    )
    summary = bundle.get("summary")
    if summary is None or level:
        summary = build_summary_payload(bundle, sample_dir=sample_dir, level=level)
    try:
        write_summary_json(summary_path, summary)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write summary JSON: {exc}") from exc
    return summary


@app.get("/api/summary")
def get_summary(
    sample_dir: str = Query(..., description="Sample directory path"),
    input_excel: str | None = Query(None),
    signal_ch: str = Query("ch1"),
    group: str | None = Query(None),
    level: str | None = Query(None, description="Allen level override for headline stats"),
    refresh: bool = Query(False),
) -> dict[str, Any]:
    return ensure_summary_json(
        sample_dir,
        input_excel=input_excel,
        signal_ch=signal_ch,
        group=group,
        level=level,
        refresh=refresh,
    )


@app.get("/api/slice.png")
def get_slice_png(
    sample_dir: str = Query(...),
    input_excel: str | None = Query(None),
    signal_ch: str = Query("ch1"),
    plane: str = Query("coronal"),
    coordinate_system: str = Query("index"),
    coordinate: float = Query(216.0),
    metric: str = Query("cfos_count"),
    level: str | None = Query(None),
    color_mode: str = Query("region"),
    compare_sample_dir: str | None = Query(None),
    compare_signal_ch: str | None = Query(None),
    highlight_region_id: int | None = Query(None),
    focus_only: bool = Query(False),
    interactive: bool = Query(False, description="Pixel-accurate slice for web region picking (no colorbar)"),
) -> Response:
    if level is not None and not str(level).strip():
        level = None
    bundle = _load_bundle(sample_dir, input_excel=input_excel, signal_ch=signal_ch)
    density_excel = input_excel or bundle["sample"]["density_excel"]
    if not bundle["parameters"].get("atlas_label_available", False):
        detail = bundle["parameters"].get("atlas_label_error") or "Allen atlas label TIFF not found."
        raise HTTPException(status_code=400, detail=detail)
    params = bundle["parameters"]
    bregma_index = tuple(int(value) for value in params.get("bregma_index", [18, 216, 228]))
    resolution_um = float(params.get("atlas_resolution_um_dv_ap_ml", [25.0, 25.0, 25.0])[1])
    compare_excel = None
    compare_sample_id = None
    if compare_sample_dir:
        compare_bundle = _load_bundle(
            compare_sample_dir,
            input_excel=input_excel,
            signal_ch=compare_signal_ch or signal_ch,
        )
        compare_excel = compare_bundle["sample"]["density_excel"]
        compare_sample_id = compare_bundle["sample"]["sample_id"]
    try:
        png_bytes, slice_layout = render_metric_slice_png_with_layout(
            input_excel=density_excel,
            plane=plane,
            coordinate_system=coordinate_system,
            coordinate=coordinate,
            metric=metric,
            level=level,
            sample_id=bundle["sample"]["sample_id"],
            cfg_path=bundle["parameters"]["cfg_path"],
            atlas_label=bundle["parameters"]["atlas_label_tiff"],
            atlas_volume_tiff=bundle["sample"].get("atlas_volume_tiff"),
            color_mode=color_mode,
            compare_input_excel=compare_excel,
            compare_sample_id=compare_sample_id,
            focus_region_id=highlight_region_id,
            focus_only=focus_only,
            bregma_index=bregma_index,
            resolution_um=resolution_um,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    headers = {
        "X-Slice-Atlas-Width": str(slice_layout.get("atlas_width", 0)),
        "X-Slice-Atlas-Height": str(slice_layout.get("atlas_height", 0)),
        "X-Slice-Image-Width": str(slice_layout.get("image_width", 0)),
        "X-Slice-Image-Height": str(slice_layout.get("image_height", 0)),
        "X-Slice-Left": str(slice_layout.get("slice_left", 0)),
        "X-Slice-Top": str(slice_layout.get("slice_top", 0)),
        "X-Slice-Width": str(slice_layout.get("slice_width", 0)),
        "X-Slice-Height": str(slice_layout.get("slice_height", 0)),
    }
    return Response(content=png_bytes, media_type="image/png", headers=headers)


@app.get("/api/region/subtree")
def get_region_subtree(
    sample_dir: str = Query(...),
    region_id: int = Query(...),
    input_excel: str | None = Query(None),
    signal_ch: str = Query("ch1"),
) -> dict[str, Any]:
    bundle = _load_bundle(sample_dir, input_excel=input_excel, signal_ch=signal_ch)
    member_ids = collect_subtree_region_ids(region_id, bundle["parameters"]["cfg_path"])
    return {
        "region_id": int(region_id),
        "member_region_ids": sorted(member_ids),
    }


@app.get("/api/region/slice-focus")
def get_region_slice_focus(
    sample_dir: str = Query(...),
    region_id: int = Query(...),
    input_excel: str | None = Query(None),
    signal_ch: str = Query("ch1"),
    plane: str = Query("coronal"),
) -> dict[str, Any]:
    bundle = _load_bundle(sample_dir, input_excel=input_excel, signal_ch=signal_ch)
    if not bundle["parameters"].get("atlas_label_available", False):
        detail = bundle["parameters"].get("atlas_label_error") or "Allen atlas label TIFF not found."
        raise HTTPException(status_code=400, detail=detail)
    member_ids = collect_subtree_region_ids(region_id, bundle["parameters"]["cfg_path"])
    params = bundle["parameters"]
    bregma_index = tuple(int(value) for value in params.get("bregma_index", [18, 216, 228]))
    resolution_um = float(params.get("atlas_resolution_um_dv_ap_ml", [25.0, 25.0, 25.0])[1])
    try:
        payload = build_region_slice_focus_payload(
            member_ids,
            atlas_label=bundle["parameters"]["atlas_label_tiff"],
            bregma_index=bregma_index,
            resolution_um=resolution_um,
            plane=plane,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    payload["region_id"] = int(region_id)
    return payload


@app.get("/api/region/resolve-display")
def get_region_resolve_display(
    sample_dir: str = Query(...),
    region_id: int = Query(...),
    input_excel: str | None = Query(None),
    signal_ch: str = Query("ch1"),
    level: str | None = Query(None),
) -> dict[str, Any]:
    bundle = _load_bundle(sample_dir, input_excel=input_excel, signal_ch=signal_ch)
    resolved_level = level or bundle["parameters"].get("default_level") or bundle["overview"].get("level")
    display_id = resolve_display_region_id(
        int(region_id),
        level=str(resolved_level),
        cfg_path=bundle["parameters"]["cfg_path"],
    )
    return {
        "region_id": int(region_id),
        "display_region_id": int(display_id),
        "level": str(resolved_level),
    }


@app.get("/api/region/at-slice")
def get_region_at_slice(
    sample_dir: str = Query(...),
    region_id: int | None = Query(None),
    input_excel: str | None = Query(None),
    signal_ch: str = Query("ch1"),
    level: str | None = Query(None),
    plane: str = Query("coronal"),
    coordinate: float = Query(...),
    coordinate_system: str = Query("index"),
    pixel_x: float = Query(...),
    pixel_y: float = Query(...),
    image_width: float = Query(...),
    image_height: float = Query(...),
) -> dict[str, Any]:
    bundle = _load_bundle(sample_dir, input_excel=input_excel, signal_ch=signal_ch)
    if not bundle["parameters"].get("atlas_label_available", False):
        detail = bundle["parameters"].get("atlas_label_error") or "Allen atlas label TIFF not found."
        raise HTTPException(status_code=400, detail=detail)
    try:
        payload = build_region_pick_payload(
            atlas_label=bundle["parameters"]["atlas_label_tiff"],
            plane=plane,
            coordinate=coordinate,
            coordinate_system=coordinate_system,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            image_width=image_width,
            image_height=image_height,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not payload.get("available"):
        return payload
    resolved_level = level or bundle["parameters"].get("default_level") or bundle["overview"].get("level")
    display_id = resolve_display_region_id(
        int(payload["region_id"]),
        level=str(resolved_level),
        cfg_path=bundle["parameters"]["cfg_path"],
    )
    payload["display_region_id"] = int(display_id)
    return payload


@app.get("/api/spatial/axes")
def get_spatial_axes(
    sample_dir: str = Query(...),
    input_excel: str | None = Query(None),
    signal_ch: str = Query("ch1"),
    bins: int = Query(32, ge=8, le=128),
) -> dict[str, Any]:
    bundle = _load_bundle(sample_dir, input_excel=input_excel, signal_ch=signal_ch)
    params = bundle["parameters"]
    bregma_index = tuple(int(value) for value in params.get("bregma_index", [18, 216, 228]))
    resolution_um = float(params.get("atlas_resolution_um_dv_ap_ml", [25.0, 25.0, 25.0])[1])
    try:
        return build_spatial_payload(
            bundle["sample"],
            bins=bins,
            atlas_label=bundle["parameters"]["atlas_label_tiff"],
            bregma_index=bregma_index,
            resolution_um=resolution_um,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/spatial/region-centroids")
def get_spatial_region_centroids(
    sample_dir: str = Query(...),
    input_excel: str | None = Query(None),
    signal_ch: str = Query("ch1"),
) -> dict[str, Any]:
    bundle = _load_bundle(sample_dir, input_excel=input_excel, signal_ch=signal_ch)
    if not bundle["parameters"].get("atlas_label_available", False):
        detail = bundle["parameters"].get("atlas_label_error") or "Allen atlas label TIFF not found."
        raise HTTPException(status_code=400, detail=detail)
    try:
        return build_atlas_region_centroids_payload(bundle["parameters"]["atlas_label_tiff"])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/spatial/points")
def get_spatial_points(
    sample_dir: str = Query(...),
    input_excel: str | None = Query(None),
    signal_ch: str = Query("ch1"),
    max_points: int = Query(50_000, ge=1000, le=150_000),
    in_brain_only: bool = Query(True, description="Drop points outside atlas brain label (region_id=0)"),
) -> dict[str, Any]:
    bundle = _load_bundle(sample_dir, input_excel=input_excel, signal_ch=signal_ch)
    try:
        return build_points_viewer_payload(
            bundle["sample"],
            max_points=max_points,
            atlas_label=bundle["parameters"]["atlas_label_tiff"],
            in_brain_only=in_brain_only,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/spatial/brain-outline-surface")
def get_spatial_brain_outline_surface(
    sample_dir: str = Query(...),
    input_excel: str | None = Query(None),
    signal_ch: str = Query("ch1"),
    stride: int = Query(2, ge=1, le=8),
    smooth_sigma: float = Query(1.4, ge=0.0, le=4.0),
) -> dict[str, Any]:
    bundle = _load_bundle(sample_dir, input_excel=input_excel, signal_ch=signal_ch)
    if not bundle["parameters"].get("atlas_label_available", False):
        detail = bundle["parameters"].get("atlas_label_error") or "Allen atlas label TIFF not found."
        raise HTTPException(status_code=400, detail=detail)
    try:
        return build_brain_outline_surface_payload(
            atlas_label=bundle["parameters"]["atlas_label_tiff"],
            stride=stride,
            smooth_sigma=smooth_sigma,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/spatial/region-surface")
def get_spatial_region_surface(
    sample_dir: str = Query(...),
    region_id: int = Query(...),
    input_excel: str | None = Query(None),
    signal_ch: str = Query("ch1"),
    stride: int = Query(2, ge=1, le=8),
    smooth_sigma: float = Query(1.2, ge=0.0, le=4.0),
) -> dict[str, Any]:
    bundle = _load_bundle(sample_dir, input_excel=input_excel, signal_ch=signal_ch)
    if not bundle["parameters"].get("atlas_label_available", False):
        detail = bundle["parameters"].get("atlas_label_error") or "Allen atlas label TIFF not found."
        raise HTTPException(status_code=400, detail=detail)
    member_ids = collect_subtree_region_ids(region_id, bundle["parameters"]["cfg_path"])
    try:
        return build_region_surface_payload(
            atlas_label=bundle["parameters"]["atlas_label_tiff"],
            region_ids=member_ids,
            stride=stride,
            smooth_sigma=smooth_sigma,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/group/analyze")
def analyze_group_post(request: GroupAnalyzeRequest) -> dict[str, Any]:
    manifest = _resolve_group_manifest(
        manifest_path=request.manifest_path,
        manifest_json=request.manifest_json,
        sample_a_dir=request.sample_a_dir,
        sample_b_dir=request.sample_b_dir,
        sample_a_label=request.sample_a_label,
        sample_b_label=request.sample_b_label,
        signal_ch_a=request.signal_ch_a,
        signal_ch_b=request.signal_ch_b,
    )
    cfg_path = request.cfg_path or DEFAULT_CFG
    resolved_a = request.group_a or manifest[0]["group"]
    resolved_b = request.group_b or manifest[1]["group"] if len(manifest) > 1 else request.group_b
    cache_key = "|".join(
        [
            str(cfg_path),
            request.level,
            request.metric,
            request.heatmap_mode,
            str(request.focus_region_id or ""),
            request.group_a or "",
            request.group_b or "",
            str(request.top_n),
            json.dumps(manifest, sort_keys=True),
        ]
    )
    if cache_key in _group_analysis_cache:
        return _group_analysis_cache[cache_key]
    try:
        payload = build_group_analysis_payload(
            manifest,
            cfg_path=cfg_path,
            level=request.level,
            metric=request.metric,
            group_a=resolved_a,
            group_b=resolved_b,
            top_n=request.top_n,
            focus_region_id=request.focus_region_id,
            heatmap_mode=request.heatmap_mode if request.heatmap_mode in {"differential", "absolute"} else "differential",
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _group_analysis_cache[cache_key] = payload
    return payload


@app.get("/api/group/export/differential-regions.csv")
def export_group_differential_regions(
    manifest_path: str | None = Query(None),
    sample_a_dir: str | None = Query(None),
    sample_b_dir: str | None = Query(None),
    level: str = Query("Level_8"),
    metric: str = Query("cfos_count"),
    group_a: str | None = Query(None),
    group_b: str | None = Query(None),
) -> StreamingResponse:
    payload = analyze_group_post(
        GroupAnalyzeRequest(
            manifest_path=manifest_path,
            sample_a_dir=sample_a_dir,
            sample_b_dir=sample_b_dir,
            level=level,
            metric=metric,
            group_a=group_a,
            group_b=group_b,
        )
    )
    csv_text = export_differential_regions_csv(payload["differential_regions"])
    return StreamingResponse(
        iter([csv_text]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="differential_regions.csv"'},
    )


@app.post("/api/export/slice-bookmarks.zip")
def export_slice_bookmarks(
    request: ExportSliceBookmarksRequest,
) -> Response:
    bundle = _load_bundle(
        request.sample_dir,
        input_excel=request.input_excel,
        signal_ch=request.signal_ch,
    )
    if not bundle["parameters"].get("atlas_label_available", False):
        detail = bundle["parameters"].get("atlas_label_error") or "Allen atlas label TIFF not found."
        raise HTTPException(status_code=400, detail=detail)
    density_excel = request.input_excel or bundle["sample"]["density_excel"]
    params = bundle["parameters"]
    bregma_index = tuple(int(value) for value in params.get("bregma_index", [18, 216, 228]))
    resolution_um = float(params.get("atlas_resolution_um_dv_ap_ml", [25.0, 25.0, 25.0])[1])
    try:
        zip_bytes = export_slice_bookmarks_zip(
            sample_id=bundle["sample"]["sample_id"],
            input_excel=density_excel,
            cfg_path=params["cfg_path"],
            atlas_label=params["atlas_label_tiff"],
            atlas_volume_tiff=bundle["sample"].get("atlas_volume_tiff"),
            bookmarks=[bookmark.model_dump() for bookmark in request.bookmarks],
            color_modes=request.color_modes,
            metric=request.metric,
            level=request.level or params["default_level"],
            bregma_index=bregma_index,
            resolution_um=resolution_um,
            dpi=request.dpi,
            focus_region_id=request.focus_region_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    filename = f"{bundle['sample']['sample_id']}_slice_bookmarks.zip"
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/export/regions.csv")
def export_regions_csv(
    sample_dir: str = Query(...),
    input_excel: str | None = Query(None),
    signal_ch: str = Query("ch1"),
    region_ids: str | None = Query(None, description="Comma-separated Allen region IDs"),
    level: str | None = Query(None),
) -> StreamingResponse:
    bundle = _load_bundle(sample_dir, input_excel=input_excel, signal_ch=signal_ch)
    parsed_ids = [int(value.strip()) for value in region_ids.split(",") if value.strip()] if region_ids else None
    csv_text = export_region_metrics_csv(
        bundle["region_metrics"],
        region_ids=parsed_ids,
        level=level or bundle["parameters"]["default_level"],
        sample_id=bundle["sample"]["sample_id"],
        group=bundle["sample"].get("group"),
        atlas_version=bundle["sample"]["atlas_version"],
        source_paths={"density_excel": bundle["sample"]["density_excel"]},
    )
    filename = f"{bundle['sample']['sample_id']}_region_metrics.csv"
    return StreamingResponse(
        iter([csv_text]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


app.mount("/", NoCacheStaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the cFos interactive visual report web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--sample-dir",
        default=None,
        help="Default sample directory to auto-load on first page open",
    )
    parser.add_argument(
        "--signal-ch",
        default=None,
        help="Signal channel for --sample-dir (default: ch1)",
    )
    return parser


def main() -> int:
    import uvicorn

    global _startup_sample_dir, _startup_signal_ch
    args = build_parser().parse_args()
    if args.sample_dir:
        _startup_sample_dir = str(Path(args.sample_dir).expanduser())
        os.environ.setdefault("CFOS_DEFAULT_SAMPLE_DIR", _startup_sample_dir)
    if args.signal_ch:
        _startup_signal_ch = str(args.signal_ch).strip() or "ch1"
        os.environ.setdefault("CFOS_DEFAULT_SIGNAL_CH", _startup_signal_ch)
    uvicorn.run("apps.cfos_visual_report.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
