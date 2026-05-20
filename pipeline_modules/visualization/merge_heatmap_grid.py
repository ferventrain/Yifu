from __future__ import annotations

import argparse
import copy
import re
import xml.etree.ElementTree as ET
from pathlib import Path


SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

DEFAULT_INPUT_DIR = Path(
    "S:\\"
    "\u53ef\u89c6\u5316\u7d20\u6750\\"
    "heatmap\\"
    "\u5206\u8111\u533a"
)
DEFAULT_OUTPUT_NAME = "merged_heatmap_3x3.svg"

URL_REF_PATTERN = re.compile(r"url\(#([^)]+)\)")
XLINK_HREF_KEYS = ("href", "{http://www.w3.org/1999/xlink}href")
PATH_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
HEATMAP_NAME_PATTERN = re.compile(
    r"^density_results_(ch\d+)_AP_(minus|plus)_(\d+)_heatmap\.svg$",
    re.IGNORECASE,
)


def _parse_length(value: str | None, fallback: float) -> float:
    if not value:
        return fallback
    match = re.match(r"^\s*([0-9.]+)", value)
    if not match:
        return fallback
    return float(match.group(1))


def _read_svg(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


def _svg_size(root: ET.Element) -> tuple[float, float, str]:
    view_box = root.get("viewBox")
    if view_box:
        parts = view_box.replace(",", " ").split()
        if len(parts) == 4:
            _, _, width, height = map(float, parts)
            return width, height, view_box

    width = _parse_length(root.get("width"), 0.0)
    height = _parse_length(root.get("height"), 0.0)
    if width <= 0 or height <= 0:
        raise ValueError("SVG is missing a usable viewBox/width/height.")
    return width, height, f"0 0 {width} {height}"


def _prefix_svg_ids(root: ET.Element, prefix: str) -> None:
    id_map: dict[str, str] = {}
    for elem in root.iter():
        elem_id = elem.get("id")
        if elem_id:
            new_id = f"{prefix}_{elem_id}"
            id_map[elem_id] = new_id
            elem.set("id", new_id)

    if not id_map:
        return

    for elem in root.iter():
        for attr_name, attr_value in list(elem.attrib.items()):
            new_value = attr_value

            if attr_value.startswith("#") and attr_value[1:] in id_map:
                new_value = f"#{id_map[attr_value[1:]]}"
            else:
                new_value = URL_REF_PATTERN.sub(
                    lambda match: f"url(#{id_map.get(match.group(1), match.group(1))})",
                    attr_value,
                )

            if attr_name in XLINK_HREF_KEYS and new_value.startswith("#") and new_value[1:] in id_map.values():
                elem.set(attr_name, new_value)
            elif new_value != attr_value:
                elem.set(attr_name, new_value)


def _path_bounds(path_d: str) -> tuple[float, float, float, float] | None:
    values = [float(match.group(0)) for match in PATH_NUMBER_PATTERN.finditer(path_d)]
    if len(values) < 2:
        return None

    xs = values[0::2]
    ys = values[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def _collect_visual_paths(root: ET.Element) -> tuple[list[ET.Element], tuple[float, float, float, float]]:
    paths: list[ET.Element] = []
    bounds: list[tuple[float, float, float, float]] = []

    inherited_keys = {
        "clip-rule",
        "fill",
        "fill-opacity",
        "fill-rule",
        "opacity",
        "stroke",
        "stroke-dasharray",
        "stroke-linecap",
        "stroke-linejoin",
        "stroke-opacity",
        "stroke-width",
        "vector-effect",
    }

    def visit(elem: ET.Element, inherited: dict[str, str]) -> None:
        current = inherited.copy()
        for key in inherited_keys:
            if key in elem.attrib:
                current[key] = elem.attrib[key]

        if elem.tag == f"{{{SVG_NS}}}path":
            path_d = elem.get("d", "")
            path_bound = _path_bounds(path_d)
            if path_bound is not None:
                merged_attrib = current.copy()
                merged_attrib.update(elem.attrib)
                paths.append(ET.Element(f"{{{SVG_NS}}}path", merged_attrib))
                bounds.append(path_bound)
            return

        for child in list(elem):
            visit(child, current)

    visit(root, {})
    if not bounds:
        raise ValueError("No visual path elements found in SVG.")

    min_x = min(bound[0] for bound in bounds)
    min_y = min(bound[1] for bound in bounds)
    max_x = max(bound[2] for bound in bounds)
    max_y = max(bound[3] for bound in bounds)
    return paths, (min_x, min_y, max_x, max_y)


def _parse_heatmap_metadata(path: Path) -> tuple[str, int]:
    match = HEATMAP_NAME_PATTERN.match(path.name)
    if not match:
        raise ValueError(f"Filename does not match expected heatmap pattern: {path.name}")

    channel = match.group(1).lower()
    sign = -1 if match.group(2).lower() == "minus" else 1
    ap_value = sign * int(match.group(3))
    return channel, ap_value


def collect_sorted_heatmaps(input_dir: Path, channel: str) -> list[Path]:
    normalized_channel = channel.lower()
    matched: list[tuple[int, Path]] = []

    for path in input_dir.glob("*.svg"):
        try:
            file_channel, ap_value = _parse_heatmap_metadata(path)
        except ValueError:
            continue
        if file_channel != normalized_channel:
            continue
        matched.append((ap_value, path))

    matched.sort(key=lambda item: (item[0], item[1].name))
    return [path for _, path in matched]


def merge_svgs_to_grid(
    svg_paths: list[Path],
    output_path: Path,
    *,
    rows: int = 3,
    cols: int = 3,
    cell_padding: float = 0.0,
    visual_only: bool = True,
) -> Path:
    if len(svg_paths) < rows * cols:
        raise ValueError(f"Need at least {rows * cols} SVG files, got {len(svg_paths)}.")

    selected = svg_paths[: rows * cols]
    roots = [_read_svg(path) for path in selected]
    sizes = []
    visual_paths_by_tile: list[list[ET.Element]] = []
    for root in roots:
        if visual_only:
            visual_paths, (min_x, min_y, max_x, max_y) = _collect_visual_paths(root)
            width = max_x - min_x
            height = max_y - min_y
            view_box = f"{min_x} {min_y} {width} {height}"
            visual_paths_by_tile.append(visual_paths)
        else:
            width, height, view_box = _svg_size(root)
            visual_paths_by_tile.append([])
        sizes.append((width, height, view_box))

    cell_width = max(width for width, _, _ in sizes)
    cell_height = max(height for _, height, _ in sizes)
    total_width = cols * cell_width + max(cols - 1, 0) * cell_padding
    total_height = rows * cell_height + max(rows - 1, 0) * cell_padding

    merged_root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "width": str(total_width),
            "height": str(total_height),
            "viewBox": f"0 0 {total_width} {total_height}",
        },
    )

    for index, (root, (width, height, view_box), visual_paths) in enumerate(
        zip(roots, sizes, visual_paths_by_tile)
    ):
        prefixed_root = copy.deepcopy(root)
        _prefix_svg_ids(prefixed_root, f"tile{index}")

        row = index // cols
        col = index % cols
        x = col * (cell_width + cell_padding) + (cell_width - width) / 2.0
        y = row * (cell_height + cell_padding) + (cell_height - height) / 2.0

        tile_svg = ET.SubElement(
            merged_root,
            f"{{{SVG_NS}}}svg",
            {
                "x": str(x),
                "y": str(y),
                "width": str(width),
                "height": str(height),
                "viewBox": view_box,
            },
        )

        if visual_only:
            for path in visual_paths:
                tile_svg.append(copy.deepcopy(path))
        else:
            for child in list(prefixed_root):
                tile_svg.append(copy.deepcopy(child))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(merged_root).write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge heatmap SVG files into a 3x3 grid.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing source SVG heatmaps. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output SVG path. Defaults to merged_heatmap_3x3.svg inside input_dir.",
    )
    parser.add_argument("--rows", type=int, default=3, help="Grid rows")
    parser.add_argument("--cols", type=int, default=3, help="Grid columns")
    parser.add_argument(
        "--limit",
        type=int,
        default=9,
        help="How many sorted SVG files to use. Default: 9",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.0,
        help="Padding between cells in SVG units",
    )
    parser.add_argument(
        "--channel",
        default="ch1",
        help="Channel name to include, e.g. ch1 or ch2. Default: ch1",
    )
    parser.add_argument(
        "--keep_full_svg",
        action="store_true",
        help="Keep the full source SVG instead of only the visual path elements.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_dir = args.input_dir
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    svg_paths = collect_sorted_heatmaps(input_dir, args.channel)
    if not svg_paths:
        raise FileNotFoundError(
            f"No SVG heatmap files found in {input_dir} for channel {args.channel}."
        )

    selected = svg_paths[: args.limit]
    needed = args.rows * args.cols
    if len(selected) < needed:
        raise ValueError(
            f"Found only {len(selected)} SVG files after applying --limit={args.limit}, need {needed}."
        )

    output_path = args.output or input_dir / DEFAULT_OUTPUT_NAME
    merged_path = merge_svgs_to_grid(
        selected,
        output_path,
        rows=args.rows,
        cols=args.cols,
        cell_padding=args.padding,
        visual_only=not args.keep_full_svg,
    )

    print(f"Merged {needed} heatmaps into: {merged_path}")
    print(f"Channel: {args.channel}")
    print(f"Visual only: {not args.keep_full_svg}")
    for path in selected[:needed]:
        print(f" - {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
