import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile

from pipeline_modules.preprocessing.tiff_to_zarr import (
    convert_sample_channels_to_zarr,
    convert_tiff_to_zarr,
    normalize_channel_label,
    parse_channel_list,
    resolve_tiff_workers,
)


class TiffToZarrParallelTests(unittest.TestCase):
    def test_resolve_tiff_workers_auto(self):
        with mock.patch("pipeline_modules.preprocessing.tiff_to_zarr.os.cpu_count", return_value=16):
            self.assertEqual(resolve_tiff_workers(0), 8)
        with mock.patch("pipeline_modules.preprocessing.tiff_to_zarr.os.cpu_count", return_value=2):
            self.assertEqual(resolve_tiff_workers(0), 1)

    def test_parse_channel_list(self):
        self.assertEqual(parse_channel_list("0,1,ch2"), ["ch0", "ch1", "ch2"])
        self.assertEqual(normalize_channel_label("3"), "ch3")

    def test_convert_tiff_to_zarr_parallel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "ch0"
            output_zarr = root / "ch0.zarr"
            input_dir.mkdir()
            for index in range(8):
                tifffile.imwrite(str(input_dir / f"C1{index:06d}.tif"), np.full((4, 5), index, dtype=np.uint16))

            result = convert_tiff_to_zarr(
                input_dir,
                output_zarr,
                chunk_size=(2, 4, 5),
                workers=2,
            )
            self.assertTrue(result["success"])
            self.assertEqual(result["workers"], 2)
            self.assertTrue(output_zarr.exists())

            import zarr

            arr = zarr.open(str(output_zarr / "0"), mode="r")
            self.assertEqual(arr.shape, (8, 4, 5))
            self.assertEqual(int(arr[3, 0, 0]), 3)

    def test_convert_sample_channels_skip_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ch0.zarr").mkdir()
            ch1_dir = root / "ch1"
            ch1_dir.mkdir()
            tifffile.imwrite(str(ch1_dir / "C1000001.tif"), np.ones((2, 2), dtype=np.uint8))

            summary = convert_sample_channels_to_zarr(
                root,
                "0,1",
                chunk_size=(1, 2, 2),
                workers=1,
                skip_existing=True,
            )
            self.assertIn("ch0", summary["skipped_existing"])
            self.assertIn("ch1", summary["converted"])
            self.assertTrue((root / "ch1.zarr").exists())


if __name__ == "__main__":
    unittest.main()
