import zarr
import numpy as np
from pathlib import Path

for name in ["ch2.zarr", "ch2_mask_hyst.zarr"]:
    p = Path(r"S:\可视化素材\血管") / name
    print(f"\n{'='*50}")
    print(f"{name}:")
    print(f"  Path exists: {p.exists()}")
    z = zarr.open(str(p), mode="r")
    print(f"  Type: {type(z).__name__}")
    if isinstance(z, zarr.Array):
        print(f"  Direct array: shape={z.shape}, dtype={z.dtype}, chunks={z.chunks}")
    else:
        print(f"  Group keys: {list(z.keys()) if hasattr(z, 'keys') else 'N/A'}")
        for k in (z.array_keys() if hasattr(z, 'array_keys') else []):
            arr = z[k]
            print(f"  Array '{k}': shape={arr.shape}, dtype={arr.dtype}, chunks={arr.chunks}")
        if hasattr(z, 'group_keys'):
            for k in z.group_keys():
                print(f"  Group '{k}'")
