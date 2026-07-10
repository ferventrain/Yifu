import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
import zarr

from pipeline_modules.segmentation.zarr_utils import export_zarr_to_tiff


class ExportZarrToTiffBatchTests(unittest.TestCase):
    def test_export_zarr_to_tiff_writes_all_slices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zarr_path = root / "mask.zarr"
            out_dir = root / "mask_tiff"

            store = zarr.open(str(zarr_path), mode="w")
            data = np.arange(24, dtype=np.uint16).reshape(3, 2, 4)
            store[:] = data

            export_zarr_to_tiff(
                zarr_path,
                out_dir,
                prefix="mask_",
                workers=2,
                slice_batch=2,
                compression="none",
            )

            self.assertTrue((out_dir / "mask_0000.tiff").exists())
            self.assertTrue((out_dir / "mask_0002.tiff").exists())
            self.assertEqual(int(tifffile.imread(str(out_dir / "mask_0001.tiff"))[0, 0]), 8)


if __name__ == "__main__":
    unittest.main()
