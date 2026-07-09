import unittest
from unittest import mock

import numpy as np

from pipeline_modules.visualization.region_masked_volume_export import (
    _build_region_slice,
    _masked_signal_slice,
    export_region_masked_volume_tiffs_for_channels,
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
    @mock.patch("pipeline_modules.visualization.region_masked_volume_export.convert_sample_channels_to_zarr")
    def test_export_for_channels_dispatches_each_channel(self, mock_convert, mock_export):
        mock_convert.return_value = {"converted": {}, "skipped_existing": {"ch0": "x"}, "failed": {}}
        mock_export.side_effect = lambda **kwargs: {"output_dir": str(kwargs["output_dir"]), "signal_ch": kwargs["signal_ch"]}

        with mock.patch("pathlib.Path.exists", return_value=True):
            payload = export_region_masked_volume_tiffs_for_channels(
                sample_dir="S:/sample",
                region_query="HIP",
                channels="0,1",
            )

        self.assertEqual(mock_convert.call_count, 1)
        self.assertEqual(mock_export.call_count, 2)
        self.assertIn("ch0", payload["channels"])
        self.assertIn("ch1", payload["channels"])
        self.assertEqual(payload["channels"]["ch0"]["signal_ch"], "ch0")


if __name__ == "__main__":
    unittest.main()
