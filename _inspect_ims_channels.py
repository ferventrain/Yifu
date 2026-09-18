"""Fast targeted ims inspection: DataSetInfo + DataSet tree only (no data reads)."""
import json
import sys

import h5py

out = {}
for path in sys.argv[1:]:
    entry = {}
    with h5py.File(path, "r") as f:
        di = f["DataSetInfo"]
        entry["dataset_info_keys"] = list(di)
        if "Channel" in di:
            ch_group = di["Channel"]
            n = int(ch_group["NumberOfChannels"][0, 0])
            entry["n_channels"] = n
            channels = {}
            for ch in range(n):
                attrs = dict(ch_group[f"Channel {ch}"].attrs)
                channels[f"ch{ch}"] = {
                    k: (v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v))
                    for k, v in attrs.items()
                    if k in ("Name", "Description", "Color", "EmissionWavelength", "ExcitationWavelength")
                }
            entry["channels"] = channels
        if "Image" in di:
            img = dict(di["Image"].attrs)
            entry["image_attrs"] = {
                k: (v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v))
                for k, v in list(img.items())[:20]
            }
        entry["resolution_levels"] = sorted(f["DataSet"], key=lambda s: int(s.split()[-1]))
        lv0 = "DataSet/ResolutionLevel 0/TimePoint 0"
        entry["level0_channels"] = sorted(f[lv0], key=lambda s: int(s.split()[-1]))
        entry["shapes"] = {
            c: [int(v) for v in f[f"{lv0}/{c}/Data"].shape] for c in entry["level0_channels"]
        }
        entry["dtypes"] = {c: f[f"{lv0}/{c}/Data"].dtype.name for c in entry["level0_channels"]}
        if len(entry["resolution_levels"]) > 1:
            lv_last = entry["resolution_levels"][-1]
            key = f"DataSet/{lv_last}/TimePoint 0"
            first = sorted(f[key], key=lambda s: int(s.split()[-1]))[0]
            entry["coarsest_shape"] = [int(v) for v in f[f"{key}/{first}/Data"].shape]
    out[path] = entry
print(json.dumps(out, indent=1, ensure_ascii=False))
