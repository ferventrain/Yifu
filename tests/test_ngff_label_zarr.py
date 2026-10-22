import json

from pipeline_modules.utils.zarr_io import write_ngff_label_sibling


def _make_label_zarr(root, with_multiscales=True):
    root.mkdir(parents=True)
    (root / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    attrs = {}
    if with_multiscales:
        attrs["multiscales"] = [{
            "version": "0.4",
            "axes": [{"name": a, "type": "space"} for a in ("z", "y", "x")],
            "datasets": [{
                "path": "0",
                "coordinateTransformations": [{"type": "scale", "scale": [2.0, 1.8, 1.8]}],
            }],
        }]
    (root / ".zattrs").write_text(json.dumps(attrs))
    arr = root / "0"
    arr.mkdir()
    (arr / ".zarray").write_text(json.dumps({
        "shape": [4, 4, 4], "chunks": [2, 2, 2], "dtype": "<u4",
        "compressor": None, "fill_value": 0, "filters": None,
        "order": "C", "zarr_format": 2, "dimension_separator": ".",
    }))
    for idx in ((0, 0, 0), (1, 1, 1)):
        (arr / ".".join(map(str, idx))).write_bytes(b"\x00" * 8)
    return root


def test_ngff_sibling_carries_image_label_and_chunks(tmp_path):
    src = _make_label_zarr(tmp_path / "upsampled_atlas_label.zarr")
    dst = write_ngff_label_sibling(src, tmp_path / "upsampled_label_ngff.zarr")

    attrs = json.loads((dst / ".zattrs").read_text(encoding="utf-8"))
    assert attrs["image-label"] == {"version": "0.4"}
    scale = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
    assert scale == [2.0, 1.8, 1.8]
    assert json.loads((dst / ".zgroup").read_text()) == {"zarr_format": 2}
    assert (dst / "0" / "0.0.0").exists() and (dst / "0" / "1.1.1").exists()


def test_ngff_sibling_builds_multiscales_when_missing(tmp_path):
    src = _make_label_zarr(tmp_path / "lab.zarr", with_multiscales=False)
    dst = write_ngff_label_sibling(src, tmp_path / "lab_ngff.zarr")

    attrs = json.loads((dst / ".zattrs").read_text(encoding="utf-8"))
    assert "image-label" in attrs
    assert attrs["multiscales"][0]["datasets"][0]["path"] == "0"


def test_ngff_sibling_replaces_existing_destination(tmp_path):
    src = _make_label_zarr(tmp_path / "lab.zarr")
    dst = tmp_path / "lab_ngff.zarr"
    dst.mkdir()
    (dst / "stale.txt").write_text("old")
    write_ngff_label_sibling(src, dst)
    assert not (dst / "stale.txt").exists()
    assert (dst / "0" / "0.0.0").exists()
