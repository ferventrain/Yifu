"""Print IMS/HDF5 resolution levels, channels, shapes, and chunking."""
import sys

import h5py


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_ims_h5.py <ims_or_h5_path>")
        sys.exit(1)

    path = sys.argv[1]
    with h5py.File(path, "r") as f:
        if "DataSet" not in f:
            print(f"No DataSet group in {path}")
            print("Top-level keys:", list(f.keys()))
            sys.exit(1)

        ds = f["DataSet"]
        for res in sorted(ds.keys()):
            if not res.startswith("ResolutionLevel"):
                continue
            tp = ds[res]["TimePoint 0"]
            for ch in sorted(tp.keys()):
                if not ch.startswith("Channel") or "Data" not in tp[ch]:
                    continue
                d = tp[ch]["Data"]
                print(
                    f"{res} | {ch} | shape={d.shape} | chunks={d.chunks} "
                    f"| dtype={d.dtype} | compression={d.compression}"
                )


if __name__ == "__main__":
    main()
