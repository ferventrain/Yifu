"""Smoke tests for region_vessel_analysis.analyze_regions_from_skeleton.

All tests use only numpy/pandas/zarr synthetic data — no kimimaro, no ANTs,
no GPU required.
"""

from __future__ import annotations

import pytest

from pipeline_modules.tubule_reconstruction.region_vessel_analysis import (
    analyze_regions_from_skeleton,
    load_region_tree_with_lookups,
    parse_region_list,
    resolve_region_query,
)


# ---------------------------------------------------------------------------
# parse_region_list
# ---------------------------------------------------------------------------


class TestParseRegionList:
    def test_comma_separated(self):
        result = parse_region_list("CA1, CA2, DG")
        assert result == ["CA1", "CA2", "DG"]

    def test_semicolon_separated(self):
        result = parse_region_list("CA1;CA2")
        assert result == ["CA1", "CA2"]

    def test_list_passthrough(self):
        result = parse_region_list(["CA1", "CA2"])
        assert result == ["CA1", "CA2"]

    def test_strips_whitespace(self):
        result = parse_region_list("  CA1 ,  CA2  ")
        assert result == ["CA1", "CA2"]

    def test_empty_string_returns_empty(self):
        assert parse_region_list("") == []


# ---------------------------------------------------------------------------
# load_region_tree_with_lookups
# ---------------------------------------------------------------------------


class TestLoadRegionTree:
    def test_returns_three_dicts(self, tiny_region_csv):
        nodes_by_id, acronym_to_ids, name_to_ids = load_region_tree_with_lookups(tiny_region_csv)
        assert isinstance(nodes_by_id, dict)
        assert isinstance(acronym_to_ids, dict)
        assert isinstance(name_to_ids, dict)

    def test_ids_present(self, tiny_region_csv):
        nodes_by_id, _, _ = load_region_tree_with_lookups(tiny_region_csv)
        assert 1 in nodes_by_id
        assert 10 in nodes_by_id
        assert 20 in nodes_by_id

    def test_acronym_lookup(self, tiny_region_csv):
        _, acronym_to_ids, _ = load_region_tree_with_lookups(tiny_region_csv)
        assert 10 in acronym_to_ids.get("ra", [])

    def test_name_lookup(self, tiny_region_csv):
        _, _, name_to_ids = load_region_tree_with_lookups(tiny_region_csv)
        assert 10 in name_to_ids.get("regiona", [])


# ---------------------------------------------------------------------------
# resolve_region_query
# ---------------------------------------------------------------------------


class TestResolveRegionQuery:
    def _tree(self, tiny_region_csv):
        return load_region_tree_with_lookups(tiny_region_csv)

    def test_resolve_by_acronym(self, tiny_region_csv):
        nodes, acr, name = self._tree(tiny_region_csv)
        node = resolve_region_query("RA", nodes, acr, name)
        assert node["id"] == 10

    def test_resolve_by_name(self, tiny_region_csv):
        nodes, acr, name = self._tree(tiny_region_csv)
        node = resolve_region_query("RegionB", nodes, acr, name)
        assert node["id"] == 20

    def test_resolve_by_integer_id_string(self, tiny_region_csv):
        nodes, acr, name = self._tree(tiny_region_csv)
        node = resolve_region_query("10", nodes, acr, name)
        assert node["id"] == 10

    def test_resolve_unknown_raises(self, tiny_region_csv):
        nodes, acr, name = self._tree(tiny_region_csv)
        with pytest.raises((KeyError, ValueError)):
            resolve_region_query("BOGUS_XYZ", nodes, acr, name)


# ---------------------------------------------------------------------------
# analyze_regions_from_skeleton  (integration smoke test)
# ---------------------------------------------------------------------------


class TestAnalyzeRegionsFromSkeleton:
    def test_returns_summary_table(
        self, tiny_skeleton_csvs, tiny_annotation_zarr, tiny_mask_zarr, tiny_region_csv
    ):
        vertex_csv, edge_csv = tiny_skeleton_csvs
        result = analyze_regions_from_skeleton(
            vertex_csv_path=vertex_csv,
            edge_csv_path=edge_csv,
            mask_zarr_path=tiny_mask_zarr,
            annotation_zarr_path=tiny_annotation_zarr,
            region_cfg_csv=tiny_region_csv,
            regions="RA,RB",
            output_dir=None,
            annotation_resolution_xyz=(25.0, 25.0, 25.0),
        )
        assert "summary_table" in result
        df = result["summary_table"]
        assert set(df["query"]) == {"RA", "RB"}
        assert "vessel_volume_um3" in df.columns
        assert "branch_point_path_length_sd_um" in df.columns
        assert "num_edges" not in df.columns

    def test_writes_csv_and_json(
        self, tmp_path, tiny_skeleton_csvs, tiny_annotation_zarr, tiny_mask_zarr, tiny_region_csv
    ):
        vertex_csv, edge_csv = tiny_skeleton_csvs
        out_dir = tmp_path / "region_out"
        result = analyze_regions_from_skeleton(
            vertex_csv_path=vertex_csv,
            edge_csv_path=edge_csv,
            mask_zarr_path=tiny_mask_zarr,
            annotation_zarr_path=tiny_annotation_zarr,
            region_cfg_csv=tiny_region_csv,
            regions="RA",
            output_dir=out_dir,
            annotation_resolution_xyz=(25.0, 25.0, 25.0),
        )
        assert result["summary_csv_path"].exists()
        assert result["summary_json_path"].exists()
        assert result["manifest_path"].exists()

    def test_defaults_to_sample_label_zarr_and_config_resolution(
        self, tmp_path, tiny_skeleton_csvs, tiny_annotation_zarr, tiny_mask_zarr, tiny_region_csv
    ):
        import shutil

        sample_dir = tmp_path / "sample01"
        tubule_dir = sample_dir / "tubule_reconstruction"
        tubule_dir.mkdir(parents=True)
        default_label_zarr = sample_dir / "upsampled_atlas_label.zarr"
        default_mask_zarr = sample_dir / "ch1_mask.zarr"
        shutil.copytree(tiny_annotation_zarr, default_label_zarr)
        shutil.copytree(tiny_mask_zarr, default_mask_zarr)

        vertex_csv, edge_csv = tiny_skeleton_csvs
        default_vertex_csv = tubule_dir / "skeleton_vertices.csv"
        default_edge_csv = tubule_dir / "skeleton_edges.csv"
        shutil.copy2(vertex_csv, default_vertex_csv)
        shutil.copy2(edge_csv, default_edge_csv)

        config_path = tmp_path / "config.json"
        config_path.write_text(
            '{"input": {"resolution_xyz": [25.0, 25.0, 25.0]}}',
            encoding="utf-8",
        )

        result = analyze_regions_from_skeleton(
            vertex_csv_path=default_vertex_csv,
            edge_csv_path=default_edge_csv,
            region_cfg_csv=tiny_region_csv,
            regions="RA",
            output_dir=None,
            config_path=config_path,
        )
        assert "summary_table" in result

    def test_empty_regions_raises(
        self, tiny_skeleton_csvs, tiny_annotation_zarr, tiny_mask_zarr, tiny_region_csv
    ):
        vertex_csv, edge_csv = tiny_skeleton_csvs
        with pytest.raises(ValueError):
            analyze_regions_from_skeleton(
                vertex_csv_path=vertex_csv,
                edge_csv_path=edge_csv,
                mask_zarr_path=tiny_mask_zarr,
                annotation_zarr_path=tiny_annotation_zarr,
                region_cfg_csv=tiny_region_csv,
                regions="",
                output_dir=None,
            )

    def test_all_regions_exports_every_csv_id(
        self, tiny_skeleton_csvs, tiny_annotation_zarr, tiny_mask_zarr, tiny_region_csv
    ):
        vertex_csv, edge_csv = tiny_skeleton_csvs
        result = analyze_regions_from_skeleton(
            vertex_csv_path=vertex_csv,
            edge_csv_path=edge_csv,
            mask_zarr_path=tiny_mask_zarr,
            annotation_zarr_path=tiny_annotation_zarr,
            region_cfg_csv=tiny_region_csv,
            all_regions=True,
            output_dir=None,
            annotation_resolution_xyz=(25.0, 25.0, 25.0),
        )
        df = result["summary_table"]
        assert "num_branch_points" in df.columns
        assert len(df) >= 3
        assert set(df["query"]).issuperset({"1", "10", "20"})

    def test_missing_vertex_csv_raises(
        self, tmp_path, tiny_annotation_zarr, tiny_mask_zarr, tiny_region_csv
    ):
        with pytest.raises(Exception):
            analyze_regions_from_skeleton(
                vertex_csv_path=tmp_path / "ghost_vertices.csv",
                edge_csv_path=tmp_path / "ghost_edges.csv",
                mask_zarr_path=tiny_mask_zarr,
                annotation_zarr_path=tiny_annotation_zarr,
                region_cfg_csv=tiny_region_csv,
                regions="RA",
                output_dir=None,
            )
