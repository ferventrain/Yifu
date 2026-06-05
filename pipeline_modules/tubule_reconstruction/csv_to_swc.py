"""Convert existing skeleton_vertices.csv / skeleton_edges.csv to SWC files."""
import argparse
import sys
from pathlib import Path

import pandas as pd

from pipeline_modules.tubule_reconstruction.kimimaro_reconstruction import write_swc_files


def main():
    parser = argparse.ArgumentParser(description="Convert skeleton CSV files to SWC format")
    parser.add_argument("--vertex_csv", required=True)
    parser.add_argument("--edge_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    vertex_table = pd.read_csv(args.vertex_csv)
    edge_table = pd.read_csv(args.edge_csv)

    output_root = Path(args.output_dir)
    swc_dir, paths = write_swc_files(vertex_table, edge_table, output_root)

    print(f"Wrote {len(paths)} SWC files to {swc_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
