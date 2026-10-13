from __future__ import annotations

import pandas as pd
import pytest

from pipeline_modules.visualization.top_level7_density_ratio import (
    DEFAULT_PARCELLATION_CSV,
    build_auto_ratio_table,
    build_default_parcellation,
    build_level_ratio_table,
    generate_top_level_ratio_plot,
    is_layer_child_name,
    is_part_of_parent,
    load_all_level_counts,
    resolve_parcellation_counts,
    split_raw_name,
)


def write_structures_csv(path):
    rows = [
        (997, "root,root", "[997]"),
        (8, "Basic cell groups and regions,grey", "[997, 8]"),
        (567, "Cerebrum,CH", "[997, 8, 567]"),
        (688, "Cortical plate,CTXpl", "[997, 8, 567, 688]"),
        (695, "Isocortex,Isocortex", "[997, 8, 567, 688, 695]"),
        (31, "Anterior cingulate area,ACA", "[997, 8, 567, 688, 695, 31]"),
        (39, "Anterior cingulate area, dorsal part,ACAd", "[997, 8, 567, 688, 695, 31, 39]"),
        (48, "Anterior cingulate area, ventral part,ACAv", "[997, 8, 567, 688, 695, 31, 48]"),
        (49, "Anterior cingulate area, layer 1,ACA1", "[997, 8, 567, 688, 695, 31, 49]"),
        (703, "Cortical subplate,CTXsp", "[997, 8, 567, 688, 703]"),
        (9991, "Layer 6b, isocortex,6b", "[997, 8, 567, 688, 703, 9991]"),
        (131, "Lateral amygdalar nucleus,LA", "[997, 8, 567, 688, 703, 131]"),
        (295, "Basolateral amygdalar nucleus,BLA", "[997, 8, 567, 688, 703, 295]"),
        (303, "Basolateral amygdalar nucleus, anterior part,BLAa", "[997, 8, 567, 688, 703, 295, 303]"),
        (311, "Basolateral amygdalar nucleus, posterior part,BLAp", "[997, 8, 567, 688, 703, 295, 311]"),
        (1089, "Hippocampal formation,HPF", "[997, 8, 567, 688, 1089]"),
        (822, "Retrohippocampal region,RHP", "[997, 8, 567, 688, 1089, 822]"),
        (909, "Entorhinal area,ENT", "[997, 8, 567, 688, 1089, 822, 909]"),
        (918, "Entorhinal area, lateral part,ENTl", "[997, 8, 567, 688, 1089, 822, 909, 918]"),
        (926, "Entorhinal area, medial part, dorsal zone,ENTm", "[997, 8, 567, 688, 1089, 822, 909, 926]"),
        (1000, "fiber tracts,fiber tracts", "[997, 1000]"),
        (1001, "olfactory nerve,In", "[997, 1000, 1001]"),
        (1002, "anterior commissure, olfactory limb,aco", "[997, 1000, 1001, 1002]"),
        (1003, "olfactory nerve layer of main olfactory bulb,onl", "[997, 1000, 1001, 1003]"),
        (1004, "ventricular systems,VS", "[997, 1004]"),
        (1005, "3rd ventricle,V3", "[997, 1004, 1005]"),
    ]
    frame = pd.DataFrame(rows, columns=["id", "name", "structure_id_path"])
    csv_path = path / "structures.csv"
    frame.to_csv(csv_path, index=False)
    return csv_path


def region_map(regions):
    return {region["name"]: region for region in regions}


def test_split_raw_name():
    assert split_raw_name("Anterior cingulate area,ACA") == ("Anterior cingulate area", "ACA")
    assert split_raw_name("no comma") == ("no comma", "")


def test_layer_and_part_name_rules():
    assert is_layer_child_name("Anterior cingulate area, layer 1")
    assert is_layer_child_name("Primary visual area, 6a")
    assert is_layer_child_name("Field CA1, stratum oriens")
    assert not is_layer_child_name("Anterior cingulate area, dorsal part")
    assert is_part_of_parent("Basolateral amygdalar nucleus", "Basolateral amygdalar nucleus, anterior part")
    assert is_part_of_parent("Dentate gyrus", "Dentate gyrus lateral blade")
    assert not is_part_of_parent("Auditory areas", "Dorsal auditory area")
    assert not is_part_of_parent("Basolateral amygdalar nucleus", "Basolateral amygdalar nucleus")


def test_build_default_parcellation_rules(tmp_path):
    csv_path = write_structures_csv(tmp_path)
    regions = build_default_parcellation(csv_path)
    names = {region["name"] for region in regions}
    by_name = region_map(regions)

    # isocortex part children still split
    assert "Anterior cingulate area, dorsal part,ACAd" in names
    assert "Anterior cingulate area, ventral part,ACAv" in names
    assert "Anterior cingulate area, layer 1,ACA1" not in names
    # ACA divided -> remainder row subtracting its two areas
    aca = by_name["Anterior cingulate area,ACA"]
    assert aca["remainder_children"] == [
        "Anterior cingulate area, dorsal part,ACAd",
        "Anterior cingulate area, ventral part,ACAv",
    ]

    # ENT is the agreed non-isocortex part-split exception
    assert "Entorhinal area, lateral part,ENTl" in names
    assert "Entorhinal area, medial part, dorsal zone,ENTm" in names
    assert by_name["Entorhinal area,ENT"]["remainder_children"] is not None

    # nuclei stay whole: BLA region present, its parts absent
    assert "Basolateral amygdalar nucleus,BLA" in names
    assert "Basolateral amygdalar nucleus, anterior part,BLAa" not in names
    assert "Basolateral amygdalar nucleus, posterior part,BLAp" not in names

    # CTXsp divided into nuclei; stray Layer 6b child absorbed into remainder
    ctxsp = by_name["Cortical subplate,CTXsp"]
    assert ctxsp["remainder_children"] == [
        "Lateral amygdalar nucleus,LA",
        "Basolateral amygdalar nucleus,BLA",
    ]
    assert "Lateral amygdalar nucleus,LA" in names
    assert "Layer 6b, isocortex,6b" not in names

    # non-grey divided to leaves even when layer-named; no root region/remainder
    assert "anterior commissure, olfactory limb,aco" in names
    assert "olfactory nerve layer of main olfactory bulb,onl" in names
    assert by_name["olfactory nerve,In"]["remainder_children"] is not None
    assert "3rd ventricle,V3" in names
    assert "root,root" not in names


def test_resolve_parcellation_counts_remainder_math(tmp_path):
    csv_path = write_structures_csv(tmp_path)
    regions = build_default_parcellation(csv_path)
    counts = {
        "Anterior cingulate area,ACA": 40.0,
        "Anterior cingulate area, dorsal part,ACAd": 10.0,
        "Anterior cingulate area, ventral part,ACAv": 20.0,
        "Anterior cingulate area, layer 1,ACA1": 5.0,
        "Cortical subplate,CTXsp": 17.0,
        "Basolateral amygdalar nucleus,BLA": 9.0,
        "Lateral amygdalar nucleus,LA": 7.0,
        "3rd ventricle,V3": 3.0,
    }
    table = resolve_parcellation_counts(regions, counts)
    values = dict(zip(table["Name"], table["count"]))
    # remainder = parent row - included children (layer child + parent direct stay inside)
    assert values["Anterior cingulate area,ACA"] == 40.0 - 10.0 - 20.0
    assert values["Cortical subplate,CTXsp"] == 17.0 - 9.0 - 7.0
    assert values["Basolateral amygdalar nucleus,BLA"] == 9.0
    assert values["3rd ventricle,V3"] == 3.0
    missing = [name for name in values if name not in counts and values[name] != 0.0]
    assert missing == []


def write_sample_workbook(path, name, rows):
    frame = pd.DataFrame(rows, columns=["Name", "Signal Count"])
    path.mkdir(parents=True, exist_ok=True)
    frame.to_excel(path / f"{name}_result.xlsx", sheet_name="Level_6", index=False)
    return path / f"{name}_result.xlsx"


def test_build_auto_ratio_table_and_level_regression(tmp_path):
    csv_path = write_structures_csv(tmp_path)
    excel_a = write_sample_workbook(
        tmp_path / "sampleA", "sampleA",
        [
            ("Anterior cingulate area, dorsal part,ACAd", 200.0),
            ("Anterior cingulate area,ACA", 40.0),
            ("Basolateral amygdalar nucleus,BLA", 50.0),
            ("3rd ventricle,V3", 5.0),
        ],
    )
    excel_b = write_sample_workbook(
        tmp_path / "sampleB", "sampleB",
        [
            ("Anterior cingulate area, dorsal part,ACAd", 10.0),
            ("Basolateral amygdalar nucleus,BLA", 500.0),
        ],
    )

    table = build_auto_ratio_table(excel_a, excel_b, metric="Signal Count", min_count=0, top_n=5, csv_path=csv_path)
    values = dict(zip(table["Name"], zip(table["count_a"], table["count_b"])))
    # ACAd = 200 vs 10; ACA remainder = 40 - 200 = 0 vs 0; BLA = 50 vs 500
    assert values["Anterior cingulate area, dorsal part,ACAd"] == (200.0, 10.0)
    assert values["Basolateral amygdalar nucleus,BLA"] == (50.0, 500.0)
    assert values["Anterior cingulate area,ACA"] == (0.0, 0.0)
    assert set(table["sample_a"]) == {"sampleA"} and set(table["sample_b"]) == {"sampleB"}

    # numeric level path still works on the same workbooks
    level_table = build_level_ratio_table(excel_a, excel_b, metric="Signal Count", level=6, min_count=0, top_n=5)
    assert "Anterior cingulate area, dorsal part,ACAd" in set(level_table["Name"])


def test_generate_plot_auto_and_numeric_outputs(tmp_path):
    csv_path = write_structures_csv(tmp_path)
    write_sample_workbook(
        tmp_path / "duo" / "sampleA", "sampleA",
        [("Anterior cingulate area, dorsal part,ACAd", 200.0), ("Basolateral amygdalar nucleus,BLA", 50.0)],
    )
    write_sample_workbook(
        tmp_path / "duo" / "sampleB", "sampleB",
        [("Anterior cingulate area, dorsal part,ACAd", 10.0), ("Basolateral amygdalar nucleus,BLA", 500.0)],
    )

    outputs_auto = generate_top_level_ratio_plot(
        root_dir=tmp_path / "duo", min_count=0, top_n=5, csv_path=csv_path, dpi=80
    )
    assert outputs_auto["csv_path"].name == "duo_auto_ratio_top.csv"
    assert outputs_auto["plot_path"].name == "duo_auto_ratio_top5.png"
    assert outputs_auto["plot_path"].exists()

    outputs_level = generate_top_level_ratio_plot(
        root_dir=tmp_path / "duo", level=6, min_count=0, top_n=5, dpi=80
    )
    assert outputs_level["csv_path"].name == "duo_level6_ratio_top.csv"
    assert outputs_level["plot_path"].exists()


def test_load_all_level_counts_sums_sheets(tmp_path):
    workbook = tmp_path / "wb_result.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame({"Name": ["A,A", "B,B"], "Signal Count": [1, 2]}).to_excel(writer, sheet_name="Level_1", index=False)
        pd.DataFrame({"Name": ["C,C"], "Signal Count": [4]}).to_excel(writer, sheet_name="Level_2", index=False)
        pd.DataFrame({"Name": ["ignored"]}).to_excel(writer, sheet_name="Notes", index=False)
    counts = load_all_level_counts(workbook, "Signal Count")
    assert counts == {"A,A": 1.0, "B,B": 2.0, "C,C": 4.0}


@pytest.mark.skipif(not DEFAULT_PARCELLATION_CSV.exists(), reason="production structure CSV not available")
def test_production_parcellation_snapshot():
    regions = build_default_parcellation()
    names = {region["name"] for region in regions}
    by_name = region_map(regions)

    # layers collapse to whole regions
    for expected in [
        "Field CA1,CA1",
        "Dentate gyrus,DG",
        "Main olfactory bulb,MOB",
        "Subiculum,SUB",
        "Superior colliculus, motor related,SCm",
    ]:
        assert expected in names
    assert not any(name.endswith("CA1sp") or name.endswith("MOBgl") for name in names)

    # nuclei whole instead of parts
    for expected in ["Basolateral amygdalar nucleus,BLA", "Basomedial amygdalar nucleus,BMA", "Endopiriform nucleus,EP",
                     "Medial amygdalar nucleus,MEA", "Dorsal part of the lateral geniculate complex,LGd"]:
        assert expected in names
    assert "Basolateral amygdalar nucleus, anterior part,BLAa" not in names

    # cortical subareas split
    for expected in ["Anterior cingulate area, dorsal part,ACAd", "Retrosplenial area, dorsal part,RSPd",
                     "Orbital area, lateral part,ORBl", "Agranular insular area, dorsal part,AId",
                     "Entorhinal area, lateral part,ENTl"]:
        assert expected in names

    # CTXsp remainder subtracts exactly the amygdala nuclei
    ctxsp = by_name["Cortical subplate,CTXsp"]
    assert set(ctxsp["remainder_children"]) == {
        "Basolateral amygdalar nucleus,BLA",
        "Basomedial amygdalar nucleus,BMA",
        "Claustrum,CLA",
        "Endopiriform nucleus,EP",
        "Lateral amygdalar nucleus,LA",
        "Posterior amygdalar nucleus,PA",
    }

    # comma-less part names ("Dentate gyrus lateral blade") keep DG whole
    assert by_name["Dentate gyrus,DG"]["remainder_children"] is None
    assert "Dentate gyrus lateral blade,DGlb" not in names

    # no region may be labelled root
    assert "root,root" not in names
