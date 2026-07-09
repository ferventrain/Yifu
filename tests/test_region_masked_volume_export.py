import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile

from pipeline_modules.visualization.region_masked_volume_export import (
    _build_region_slice,
    _masked_signal_slice,
    _pair_tiff_stacks,
    export_region_masked_volume_tiffs_for_channels,
    export_region_masked_volume_tiffs_from_tiff,
)
class RegionMaskedVolumeExportTests(unittest.TestCase):
    def test_masked_signal_slice_applies_region_and_mask(self):
        signal = np.array([[10, 20], [30, 40]], dtype=np.uint16)
        labels = np.array([[0, 7], [7, 0]], dtype=np.uint16)
        mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        region_ids = np.asarray([7], dtype=np.uint16)

        out = _masked_signal_slice(
            signal,
            labels,
            region_ids,
            mask_slice=mask,
            foreground_label=1,
        )
        self.assertEqual(int(out[0, 1]), 20)
        self.assertEqual(int(out[1, 0]), 30)
        self.assertEqual(int(out.sum()), 50)

    def test_build_region_slice(self):
        labels = np.array([[1, 2], [7, 0]], dtype=np.uint16)
        region_ids = np.asarray([7], dtype=np.uint16)
        mask = _build_region_slice(labels, region_ids)
        self.assertTrue(mask[1, 0])
        self.assertFalse(mask[0, 1])

    @mock.patch("pipeline_modules.visualization.region_masked_volume_export.export_region_masked_volume_tiffs")
    def test_export_for_channels_dispatches_each_channel(self, mock_export):
        mock_export.side_effect = lambda **kwargs: {
            "output_dir": str(kwargs["output_dir"]),
            "signal_ch": kwargs["signal_ch"],
            "source": kwargs.get("source", "auto"),
        }

        payload = export_region_masked_volume_tiffs_for_channels(
            sample_dir="S:/sample",
            region_query="HIP",
            channels="0,1",
            source="tiff",
        )

        self.assertEqual(mock_export.call_count, 2)
        self.assertIn("ch0", payload["channels"])
        self.assertIn("ch1", payload["channels"])
        self.assertEqual(payload["channels"]["ch0"]["signal_ch"], "ch0")

    def test_export_from_tiff_masks_signal_inside_region(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            signal_dir = root / "ch3"
            label_dir = root / "upsampled_atlas_label"
            signal_dir.mkdir()
            label_dir.mkdir()

            signal = np.array([[100, 200], [300, 400]], dtype=np.uint16)
            labels = np.array([[0, 7], [7, 0]], dtype=np.uint16)
            tifffile.imwrite(str(signal_dir / "C1000001.tiff"), signal)
            tifffile.imwrite(str(label_dir / "C1000001.tiff"), labels)

            with mock.patch(
                "pipeline_modules.visualization.region_masked_volume_export.resolve_region_subtree_ids",
                return_value=({7}, "HIP", "Hippocampus"),
            ):
                summary = export_region_masked_volume_tiffs_from_tiff(
                    sample_dir=root,
                    region_query="HIP",
                    signal_ch="ch3",
                    signal_tiff_dir=signal_dir,
                    label_tiff_dir=label_dir,
                    export_mode="signal",
                    workers=1,
                )

            out_path = Path(summary["output_dir"]) / "C1000000.tiff"
            self.assertTrue(out_path.exists())
            out = tifffile.imread(str(out_path))
            self.assertEqual(int(out[0, 1]), 200)
            self.assertEqual(int(out[1, 0]), 300)
            self.assertEqual(int(out.sum()), 500)

    def test_pair_tiff_stacks_requires_matching_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            signal_dir = root / "signal"
            label_dir = root / "label"
            signal_dir.mkdir()
            label_dir.mkdir()
            tifffile.imwrite(str(signal_dir / "a.tiff"), np.zeros((2, 2), dtype=np.uint8))
            tifffile.imwrite(str(label_dir / "a.tiff"), np.zeros((2, 2), dtype=np.uint8))
            tifffile.imwrite(str(label_dir / "b.tiff"), np.zeros((2, 2), dtype=np.uint8))
            with self.assertRaises(ValueError):
                _pair_tiff_stacks(signal_dir, label_dir)

if __name__ == "__main__":
    unittest.main()
