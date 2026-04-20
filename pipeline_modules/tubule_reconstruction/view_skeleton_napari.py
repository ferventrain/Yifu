import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .kimimaro_reconstruction import open_zarr_dataset, parse_resolution_xyz
except ImportError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from pipeline_modules.tubule_reconstruction.kimimaro_reconstruction import (
        open_zarr_dataset,
        parse_resolution_xyz,
    )


def parse_zyx_text(text):
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected z,y,x text with 3 values, got: {text}")
    return tuple(int(part) for part in parts)


def load_optional_zarr(path_like, dataset_name):
    if not path_like:
        return None
    return open_zarr_dataset(path_like, dataset_name=dataset_name)


def choose_chunk_index(edge_table, requested_chunk_index=None):
    if "chunk_index" not in edge_table.columns or edge_table.empty:
        return None

    available = edge_table["chunk_index"].dropna().astype(str).unique().tolist()
    if not available:
        return None

    if requested_chunk_index:
        if requested_chunk_index not in available:
            raise ValueError(
                f"Requested chunk_index={requested_chunk_index} not found. "
                f"Available: {available}"
            )
        return requested_chunk_index

    return available[0]


def load_chunk_roi_from_edges(edge_table, chunk_index):
    chunk_rows = edge_table.loc[edge_table["chunk_index"].astype(str) == str(chunk_index)]
    if chunk_rows.empty:
        raise ValueError(f"No skeleton edges found for chunk_index={chunk_index}")

    start_zyx = parse_zyx_text(chunk_rows.iloc[0]["chunk_start_zyx"])
    stop_zyx = parse_zyx_text(chunk_rows.iloc[0]["chunk_stop_zyx"])
    roi = tuple(slice(start, stop) for start, stop in zip(start_zyx, stop_zyx))
    return chunk_rows, start_zyx, stop_zyx, roi


def read_roi(array, roi):
    if array is None:
        return None
    if roi is None:
        return np.asarray(array[:])
    return np.asarray(array[roi])


def build_edge_vectors(edge_table, resolution_xyz, start_zyx=None):
    # resolution_xyz is x,y,z; convert to z,y,x for the exported skeleton coordinates
    scale_zyx = np.array([resolution_xyz[2], resolution_xyz[1], resolution_xyz[0]], dtype=np.float64)
    offset = np.array(start_zyx if start_zyx is not None else (0, 0, 0), dtype=np.float64)

    vectors = []
    for _, row in edge_table.iterrows():
        source_um = np.array([row["source_z_um"], row["source_y_um"], row["source_x_um"]], dtype=np.float64)
        target_um = np.array([row["target_z_um"], row["target_y_um"], row["target_x_um"]], dtype=np.float64)

        source_px = source_um / scale_zyx - offset
        target_px = target_um / scale_zyx - offset
        vectors.append(np.stack([source_px, target_px - source_px], axis=0))

    if not vectors:
        return np.empty((0, 2, 3), dtype=np.float64)

    return np.asarray(vectors, dtype=np.float64)


def build_argparser():
    parser = argparse.ArgumentParser(description="Visualize reconstructed skeleton edges in napari.")
    parser.add_argument("--skeleton_edges_csv", required=True, help="Path to skeleton_edges.csv")
    parser.add_argument("--image_zarr", default="", help="Optional image Zarr path")
    parser.add_argument("--mask_zarr", default="", help="Optional mask Zarr path")
    parser.add_argument("--dataset_name", default="0", help="Dataset name inside the image/mask Zarr group")
    parser.add_argument("--resolution_xyz", default="1,1,1", help="Voxel size in microns as x,y,z")
    parser.add_argument("--chunk_index", default="", help="Optional chunk index like 0.11.8")
    parser.add_argument("--max_edges", type=int, default=0, help="Limit the number of displayed edges (0 = no limit)")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    try:
        import napari
    except ImportError as exc:
        raise ImportError("napari is required to run this visualization script.") from exc

    edge_csv_path = Path(args.skeleton_edges_csv)
    if not edge_csv_path.exists():
        raise FileNotFoundError(f"Skeleton edge CSV not found: {edge_csv_path}")

    edge_table = pd.read_csv(edge_csv_path)
    if edge_table.empty:
        raise ValueError(f"Skeleton edge CSV is empty: {edge_csv_path}")

    resolution_xyz = parse_resolution_xyz(args.resolution_xyz)
    selected_chunk_index = choose_chunk_index(edge_table, args.chunk_index or None)
    start_zyx = None
    roi = None

    if selected_chunk_index is not None:
        edge_table, start_zyx, _, roi = load_chunk_roi_from_edges(edge_table, selected_chunk_index)
        print(f"Viewing chunk_index={selected_chunk_index}")
        print(f"Chunk start zyx={start_zyx}")

    if args.max_edges and args.max_edges > 0:
        edge_table = edge_table.head(int(args.max_edges))

    image_zarr = load_optional_zarr(args.image_zarr, args.dataset_name)
    mask_zarr = load_optional_zarr(args.mask_zarr, args.dataset_name)

    image_data = read_roi(image_zarr, roi)
    mask_data = read_roi(mask_zarr, roi)
    edge_vectors = build_edge_vectors(edge_table, resolution_xyz, start_zyx=start_zyx)

    viewer = napari.Viewer(ndisplay=3 if image_data is not None and image_data.ndim == 3 else 2)

    if image_data is not None:
        viewer.add_image(image_data, name="image")

    if mask_data is not None:
        viewer.add_labels(mask_data, name="mask", opacity=0.35)

    viewer.add_vectors(
        edge_vectors,
        name="skeleton_edges",
        edge_width=1.0,
        edge_color="yellow",
        vector_style="line",
        length=1.0,
    )

    napari.run()


if __name__ == "__main__":
    main()
