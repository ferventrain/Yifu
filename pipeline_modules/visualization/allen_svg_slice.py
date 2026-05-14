"""Download and restyle Allen Mouse Brain Atlas official SVG plates.

This module uses Allen Institute's public API to fetch the 2D coronal
reference-atlas SVG outlines, then maps bregma AP coordinates to the nearest
official atlas plate via ``reference_to_image``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any


API_BASE = "https://api.brain-map.org/api/v2"
CORONAL_REFERENCE_SPACE_ID = 9
ADULT_MOUSE_ATLAS_DATA_SET_ID = 100048576
DEFAULT_BREGMA_AP_UM = 5400.0
DEFAULT_REFERENCE_Y_UM = 4000.0
DEFAULT_REFERENCE_Z_UM = 5700.0
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "reference" / "allen_mouse_atlas_svg"

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


@dataclass(frozen=True)
class AllenSvgPlate:
    image_id: int
    section_number: int
    width: int
    height: int
    resolution_um: float
    raw_svg_path: Path


def _fetch_json(url: str, *, retries: int = 3, delay_seconds: float = 0.75) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(delay_seconds * (attempt + 1))
    raise RuntimeError(f"Failed to fetch JSON from {url}: {last_error}")


def _fetch_text(url: str, *, retries: int = 3, delay_seconds: float = 0.75) -> str:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read().decode("utf-8")
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(delay_seconds * (attempt + 1))
    raise RuntimeError(f"Failed to fetch text from {url}: {last_error}")


def _atlas_index_url() -> str:
    criteria = (
        "model::AtlasImage,"
        f"rma::criteria,[data_set_id$eq{ADULT_MOUSE_ATLAS_DATA_SET_ID}],[annotated$eqtrue],"
        "rma::options[order$eqsection_number][num_rows$eqall]"
    )
    return f"{API_BASE}/data/query.json?criteria={urllib.parse.quote(criteria, safe=':$,[]')}"


def _reference_to_image_url(*, ccf_x_um: float, ccf_y_um: float, ccf_z_um: float) -> str:
    query = urllib.parse.urlencode(
        {
            "x": f"{ccf_x_um:.3f}",
            "y": f"{ccf_y_um:.3f}",
            "z": f"{ccf_z_um:.3f}",
            "section_data_set_ids": str(ADULT_MOUSE_ATLAS_DATA_SET_ID),
        }
    )
    return f"{API_BASE}/reference_to_image/{CORONAL_REFERENCE_SPACE_ID}.json?{query}"


def _raw_svg_url(image_id: int) -> str:
    return f"{API_BASE}/svg/{int(image_id)}"


def _raw_svg_filename(image_id: int, section_number: int) -> str:
    return f"section_{int(section_number):03d}_image_{int(image_id)}.svg"


def load_or_fetch_plate_index(cache_dir: str | Path = DEFAULT_CACHE_DIR, *, force: bool = False) -> list[dict[str, Any]]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "index.json"
    if index_path.exists() and not force:
        return json.loads(index_path.read_text(encoding="utf-8"))

    payload = _fetch_json(_atlas_index_url())
    rows = payload.get("msg", [])
    if not rows:
        raise RuntimeError("Allen API returned no atlas SVG plate rows.")

    plates = [
        {
            "image_id": int(row["id"]),
            "section_number": int(row["section_number"]),
            "width": int(row["width"]),
            "height": int(row["height"]),
            "resolution_um": float(row.get("resolution", 0.0)),
        }
        for row in rows
    ]
    plates.sort(key=lambda item: int(item["section_number"]))
    index_path.write_text(json.dumps(plates, indent=2), encoding="utf-8")
    return plates


def download_all_svgs(
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    *,
    force: bool = False,
) -> list[AllenSvgPlate]:
    cache_dir = Path(cache_dir)
    raw_dir = cache_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    plates = load_or_fetch_plate_index(cache_dir, force=force)

    outputs: list[AllenSvgPlate] = []
    for plate in plates:
        image_id = int(plate["image_id"])
        section_number = int(plate["section_number"])
        raw_path = raw_dir / _raw_svg_filename(image_id, section_number)
        if force or not raw_path.exists() or raw_path.stat().st_size == 0:
            raw_path.write_text(_fetch_text(_raw_svg_url(image_id)), encoding="utf-8")
        outputs.append(
            AllenSvgPlate(
                image_id=image_id,
                section_number=section_number,
                width=int(plate["width"]),
                height=int(plate["height"]),
                resolution_um=float(plate.get("resolution_um", 0.0)),
                raw_svg_path=raw_path,
            )
        )
    return outputs


def ap_mm_to_ccf_x_um(ap_mm: float, *, bregma_ap_um: float = DEFAULT_BREGMA_AP_UM) -> float:
    """Convert bregma AP mm to Allen coronal reference-space x um.

    Allen ReferenceSpace id 9 uses a posterior-positive x axis, so anterior
    bregma AP values subtract from the estimated bregma x coordinate.
    """

    return float(bregma_ap_um) - float(ap_mm) * 1000.0


def find_plate_for_ap(
    ap_mm: float,
    *,
    bregma_ap_um: float = DEFAULT_BREGMA_AP_UM,
    reference_y_um: float = DEFAULT_REFERENCE_Y_UM,
    reference_z_um: float = DEFAULT_REFERENCE_Z_UM,
) -> dict[str, Any]:
    ccf_x_um = ap_mm_to_ccf_x_um(ap_mm, bregma_ap_um=bregma_ap_um)
    payload = _fetch_json(
        _reference_to_image_url(
            ccf_x_um=ccf_x_um,
            ccf_y_um=reference_y_um,
            ccf_z_um=reference_z_um,
        )
    )
    image_sync = payload.get("msg", [{}])[0].get("image_sync")
    if not image_sync:
        raise RuntimeError(f"Allen API did not return a plate for AP {ap_mm:g} mm.")
    return {
        "ap_mm": float(ap_mm),
        "ccf_x_um": float(ccf_x_um),
        "image_id": int(image_sync["section_image_id"]),
        "section_number": int(image_sync["section_number"]),
        "image_x": float(image_sync["x"]),
        "image_y": float(image_sync["y"]),
    }


def _find_raw_svg_path(cache_dir: Path, image_id: int, section_number: int) -> Path:
    raw_dir = cache_dir / "raw"
    expected = raw_dir / _raw_svg_filename(image_id, section_number)
    if expected.exists():
        return expected
    matches = sorted(raw_dir.glob(f"section_*_image_{int(image_id)}.svg"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Cached Allen SVG not found for image_id={image_id}. Run with --download-all first.")


def restyle_svg_outline(
    input_svg: str | Path,
    output_svg: str | Path,
    *,
    stroke_width: float = 1.0,
    background: str = "white",
) -> Path:
    input_svg = Path(input_svg)
    output_svg = Path(output_svg)
    output_svg.parent.mkdir(parents=True, exist_ok=True)

    root = ET.fromstring(input_svg.read_text(encoding="utf-8"))
    width = root.get("width", "100%")
    height = root.get("height", "100%")

    background_rect = ET.Element(
        f"{{{SVG_NS}}}rect",
        {
            "x": "0",
            "y": "0",
            "width": str(width),
            "height": str(height),
            "fill": background,
        },
    )
    root.insert(0, background_rect)

    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag in {"path", "polygon", "polyline", "line"}:
            element.attrib.pop("style", None)
            element.set("fill", "none")
            element.set("stroke", "black")
            element.set("stroke-width", f"{float(stroke_width):g}")
            element.set("stroke-linecap", "round")
            element.set("stroke-linejoin", "round")
            element.set("vector-effect", "non-scaling-stroke")

    tree = ET.ElementTree(root)
    tree.write(output_svg, encoding="utf-8", xml_declaration=True)
    return output_svg


def render_bregma_ap_svg(
    ap_mm: float,
    output_svg: str | Path,
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    stroke_width: float = 0.7,
    ensure_cache: bool = True,
    bregma_ap_um: float = DEFAULT_BREGMA_AP_UM,
) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    if ensure_cache:
        download_all_svgs(cache_dir)
    plate = find_plate_for_ap(ap_mm, bregma_ap_um=bregma_ap_um)
    raw_svg_path = _find_raw_svg_path(cache_dir, plate["image_id"], plate["section_number"])
    output_svg = restyle_svg_outline(raw_svg_path, output_svg, stroke_width=stroke_width)
    return {
        **plate,
        "raw_svg_path": str(raw_svg_path),
        "output_svg": str(output_svg),
    }


def parse_ap_range(values: list[str]) -> list[float]:
    if len(values) != 3:
        raise argparse.ArgumentTypeError("--ap-range requires START STOP STEP")
    start, stop, step = (float(value) for value in values)
    if step == 0:
        raise argparse.ArgumentTypeError("--ap-range STEP cannot be 0")
    aps: list[float] = []
    current = start
    epsilon = abs(step) / 1000.0
    if step > 0:
        while current <= stop + epsilon:
            aps.append(round(current, 6))
            current += step
    else:
        while current >= stop - epsilon:
            aps.append(round(current, 6))
            current += step
    return aps


def ap_name(ap_mm: float) -> str:
    if ap_mm >= 0:
        return f"AP_plus_{ap_mm:g}".replace(".", "p")
    return f"AP_minus_{abs(ap_mm):g}".replace(".", "p")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and render official Allen Mouse Atlas SVG coronal plates.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Directory for cached raw Allen SVG plates")
    parser.add_argument("--download-all", action="store_true", help="Download all official Allen coronal SVG plates")
    parser.add_argument("--force", action="store_true", help="Refresh cached index/SVG files")
    parser.add_argument("--ap-mm", type=float, action="append", help="Bregma AP coordinate in mm; can be repeated")
    parser.add_argument("--ap-range", nargs=3, metavar=("START", "STOP", "STEP"), help="Batch AP range in mm")
    parser.add_argument("--output", help="Output SVG path for a single --ap-mm")
    parser.add_argument("--output-dir", default="outputs/allen_svg_slices", help="Output directory for batch rendering")
    parser.add_argument("--stroke-width", type=float, default=0.7, help="Black outline stroke width in SVG units")
    parser.add_argument("--bregma-ap-um", type=float, default=DEFAULT_BREGMA_AP_UM, help="Estimated bregma AP x in CCF um")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.download_all:
            plates = download_all_svgs(args.cache_dir, force=args.force)
        else:
            plates = []

        ap_values: list[float] = []
        if args.ap_mm:
            ap_values.extend(float(value) for value in args.ap_mm)
        if args.ap_range:
            ap_values.extend(parse_ap_range(args.ap_range))

        outputs = []
        if ap_values:
            if len(ap_values) == 1 and args.output:
                output_paths = [Path(args.output)]
            else:
                output_dir = Path(args.output_dir)
                output_paths = [output_dir / f"{ap_name(ap)}.svg" for ap in ap_values]

            for ap, output_path in zip(ap_values, output_paths):
                outputs.append(
                    render_bregma_ap_svg(
                        ap,
                        output_path,
                        cache_dir=args.cache_dir,
                        stroke_width=args.stroke_width,
                        ensure_cache=True,
                        bregma_ap_um=args.bregma_ap_um,
                    )
                )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "cache_dir": str(Path(args.cache_dir)),
                "downloaded_or_cached_plates": len(plates),
                "outputs": outputs,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
