# -*- coding: utf-8 -*-
"""Top-30 region selection + chord diagram for YF2026082501 (d22 vs SY1).

Step 1: per group, from the whole-brain region stats xlsx:
    density = count / volume_mm3 (volume==0 -> NaN)
    merge hemispheres by SUM of count and volume, then divide
    group mean over samples, drop regions with no valid density in the group
    tree from Region_Csv structure_id_path (validated)
    top-30 depth-4 parents by mean density
    top-30 depth-6 descendants of those parents
    collapse: any depth-5 parent contributing >2 of the 30 children is replaced
    by one row of its own (its own count/volume), all its depth-6 children are
    blacklisted, backfill priority depth6 -> depth5 -> depth4, max 10 rounds
    -> top30_<group>.csv

Step 2: chord diagram (SVG + 300dpi PNG, white background).
"""
import ast
import colorsys
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, patheffects
from matplotlib.patches import Patch, Polygon, Wedge

PROJ = Path(r"H:/arivis-analysis/_active/YF2026082501")
REPO = Path(r"H:/yifu0428/Yifu")
REGION_CSV = REPO / "pipeline_modules/registration/Region_Csv_Rev1_updated.CSV"
OUT_DIR = PROJ / "_figure_materials" / "chord_top30"

# group -> list of sample stats xlsx (one entry per sample; a group may hold
# any number of samples). Group density = mean over samples that pass the
# quorum (more than half the group's samples must have a valid density).
GROUP_SAMPLES = {
    "d22": [PROJ / "YF2026082501_AYD_nao_d22/results/YF2026082501_AYD_nao_d22_ch1_brain_distribution_stats_v2.xlsx"],
    "SY1": [PROJ / "YF2026082501_AYD_nao_SY1/results/YF2026082501_AYD_nao_SY1_ch1_brain_distribution_stats_v2.xlsx"],
}

DEPTH_START = 4
TARGET_DEPTH = 6
TOP_N_PARENTS = 30
TOP_N_FINAL = 30
COLLAPSE_THRESHOLD = 2  # collapse when a depth-5 parent contributes > 2 children
MAX_ROUNDS = 10


# ---------------------------------------------------------------- hierarchy --
def load_tree(csv_path):
    df = pd.read_csv(csv_path)
    nodes = {}
    for _, row in df.iterrows():
        rid = int(row["id"])
        path = ast.literal_eval(row["structure_id_path"])
        nodes[rid] = {
            "id": rid,
            "name": str(row["name"]),
            "depth": len(path) - 1,
            "path": [int(v) for v in path],
        }
    ids = set(nodes)
    missing_parent, depth_bad, cycle_bad = [], [], []
    for rid, node in nodes.items():
        p = node["path"]
        if len(p) >= 2:
            par = p[-2]
            if par not in ids:
                missing_parent.append((rid, par))
                continue
            node["parent"] = par
            if nodes[par]["depth"] + 1 != node["depth"]:
                depth_bad.append((rid, node["depth"], par, nodes[par]["depth"]))
            if nodes[par]["path"] != p[:-1]:
                depth_bad.append((rid, "prefix-mismatch", par))
    roots = [rid for rid, n in nodes.items() if "parent" not in n]
    if len(roots) != 1:
        print(f"ERROR: expected exactly one root, got {roots}")
        sys.exit(1)
    for rid in nodes:
        seen, cur = {rid}, nodes[rid].get("parent")
        while cur is not None:
            if cur in seen:
                cycle_bad.append(rid)
                break
            seen.add(cur)
            cur = nodes[cur].get("parent")
    if missing_parent or depth_bad or cycle_bad:
        print("ERROR: invalid region tree")
        print("  missing parents:", missing_parent)
        print("  depth/prefix violations:", depth_bad)
        print("  cycles:", cycle_bad)
        sys.exit(1)
    by_name = {}
    for node in nodes.values():
        if node["name"] in by_name:
            print(f"ERROR: duplicate region name in CSV: {node['name']}")
            sys.exit(1)
        by_name[node["name"]] = node
    children = {rid: [] for rid in nodes}
    for rid, node in nodes.items():
        if "parent" in node:
            children[node["parent"]].append(rid)
    return nodes, by_name, children


def descendants(children_map, rid):
    out, stack = [], [rid]
    while stack:
        cur = stack.pop()
        for ch in children_map[cur]:
            out.append(ch)
            stack.append(ch)
    return out


def split_name(full):
    if "," in full:
        head, acr = full.rsplit(",", 1)
        return head, acr
    return full, ""


# ------------------------------------------------------------------- step 1 --
def sample_density_table(xlsx_path, by_name, voxel_mm3, group):
    xl = pd.ExcelFile(xlsx_path)
    rows = {}
    for sheet in xl.sheet_names:
        frame = xl.parse(sheet)
        for _, r in frame.iterrows():
            full = str(r["Name"])
            if full in rows:
                print(f"ERROR: duplicate Name {full!r} in {xlsx_path.name}")
                sys.exit(1)
            node = by_name.get(full)
            if node is None:
                print(f"ERROR: xlsx names not found in region CSV (cannot guess hierarchy):")
                print(f"  sheet {sheet}: {full!r}")
                sys.exit(1)
            if "Left Signal Count" in frame.columns:
                count = float(r["Left Signal Count"]) + float(r["Right Signal Count"])
                voxels = float(r["Left Total Voxels"]) + float(r["Right Total Voxels"])
            else:
                # table produced without hemisphere split: whole-brain columns
                # already cover both hemispheres (region-pair collapsed counts)
                count = float(r["Signal Count"])
                voxels = float(r["Total Voxels"])
            vol = voxels * voxel_mm3
            rows[full] = {
                "node": node,
                "density": (count / vol) if vol > 0 else np.nan,
                "count": count,
                "volume_mm3": vol,
            }
    print(f"[{group}] {Path(xlsx_path).name}: {len(rows)} regions, "
          f"{sum(1 for v in rows.values() if not np.isfinite(v['density']))} with zero volume -> NaN")
    return rows


def group_mean_density(samples, by_name):
    """Mean density per region over the group's samples. Quorum: a region is
    kept only if MORE THAN HALF the samples have a valid density; the mean
    runs over those valid samples only."""
    n = len(samples)
    mean = {}
    for full in by_name:
        vals = [s[full]["density"] for s in samples if full in s and np.isfinite(s[full]["density"])]
        mean[full] = float(np.mean(vals)) if len(vals) * 2 > n else np.nan
    return mean


def select_top30(mean_density, by_name, nodes, children_map, group):
    depth4 = [n["name"] for n in nodes.values() if n["depth"] == DEPTH_START]
    depth4 = [n for n in depth4 if np.isfinite(mean_density.get(n, np.nan))]
    parents = sorted(depth4, key=lambda n: mean_density[n], reverse=True)[:TOP_N_PARENTS]
    print(f"[{group}] depth-4 candidates: {len(depth4)}, selected {len(parents)} parents")

    cand6 = []
    for p in parents:
        pid = by_name[p]["id"]
        for d in descendants(children_map, pid):
            if nodes[d]["depth"] == TARGET_DEPTH:
                cand6.append(nodes[d]["name"])
    cand6 = [n for n in cand6 if np.isfinite(mean_density.get(n, np.nan))]
    cand6 = sorted(set(cand6), key=lambda n: mean_density[n], reverse=True)
    chosen = list(cand6[:TOP_N_FINAL])
    print(f"[{group}] initial depth-6 set: {len(chosen)}")

    blacklist = set()
    verdict = "incomplete"
    for round_i in range(1, MAX_ROUNDS + 1):
        contrib = {}
        for n in chosen:
            node = by_name[n]
            if node["depth"] == TARGET_DEPTH and "parent" in node:
                par_node = nodes[node["parent"]]
                if par_node["depth"] == DEPTH_START + 1:
                    contrib.setdefault(par_node["name"], 0)
                    contrib[par_node["name"]] += 1
        violators = [p for p, c in contrib.items() if c > COLLAPSE_THRESHOLD]
        if not violators:
            verdict = f"clean after round {round_i - 1}"
            break
        print(f"[{group}] round {round_i}: collapsing depth-5 parents "
              f"{[(p, contrib[p]) for p in violators]}")
        for p in violators:
            pid = by_name[p]["id"]
            kids = [nodes[i]["name"] for i in descendants(children_map, pid)
                    if nodes[i]["depth"] == TARGET_DEPTH]
            blacklist.update(kids)
            chosen = [n for n in chosen if n not in kids]
            if p not in chosen:
                chosen.append(p)
                if not np.isfinite(mean_density.get(p, np.nan)):
                    print(f"  WARNING: collapsed parent {p!r} has no valid density (NaN)")
        pool6 = [n for n in cand6 if n not in blacklist and n not in chosen]
        depth5_all = sorted(
            {nodes[d]["name"] for p in parents
             for d in descendants(children_map, by_name[p]["id"])
             if nodes[d]["depth"] == DEPTH_START + 1
             and np.isfinite(mean_density.get(nodes[d]["name"], np.nan))},
            key=lambda n: mean_density[n], reverse=True)
        depth4_rest = sorted((p for p in parents if p not in chosen),
                             key=lambda n: mean_density[n], reverse=True)
        def pool_tier(cands):
            return [n for n in cands if n not in chosen]
        while len(chosen) < TOP_N_FINAL:
            for tier in (pool6, depth5_all, depth4_rest):
                nxt = pool_tier(tier)
                if nxt:
                    chosen.append(nxt[0])
                    break
            else:
                print(f"  WARNING: candidate pool exhausted at {len(chosen)} rows")
                break
        chosen = sorted(chosen, key=lambda n: (not np.isfinite(mean_density.get(n, np.nan)),
                                               -mean_density.get(n, 0.0)))[:TOP_N_FINAL]
    else:
        print(f"  ERROR: collapse did not converge in {MAX_ROUNDS} rounds")
    print(f"[{group}] final set: {len(chosen)} ({verdict}); "
          f"depths: {sorted(by_name[n]['depth'] for n in chosen)}")

    rows = []
    ranked = sorted(chosen, key=lambda n: (not np.isfinite(mean_density.get(n, np.nan)),
                                           -mean_density.get(n, 0.0)))
    for rank, full in enumerate(ranked, 1):
        node = by_name[full]
        head, acr = split_name(full)
        par = nodes[node["parent"]]["name"] if "parent" in node else ""
        md = mean_density[full]
        rows.append({
            "rank": rank,
            "region_id": node["id"],
            "region_name": head,
            "acronym": acr,
            "depth": node["depth"],
            "parent_region": par,
            "mean_density": round(md, 4) if np.isfinite(md) else np.nan,
        })
    return pd.DataFrame(rows), {r["region_name"] + "," + r["acronym"] for r in rows}


def main_step1(nodes, by_name, children_map):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables, final_sets = {}, {}
    for group, sample_xlsx in GROUP_SAMPLES.items():
        samples = []
        for xlsx in sample_xlsx:
            sample_dir = Path(xlsx).parent.parent  # <sample>/results/<xlsx>
            cfg = json.load(open(sample_dir / "config.json", encoding="utf-8"))
            rx, ry, rz = cfg["input"]["resolution_xyz"]
            voxel_mm3 = rx * ry * rz * 1e-9
            label = f"{group}/{sample_dir.name}"
            samples.append(sample_density_table(xlsx, by_name, voxel_mm3, label))
        if len(samples) > 1:
            print(f"[{group}] group mean over {len(samples)} samples, "
                  f"quorum: > {len(samples) / 2:.0f} valid")
        mean_density = group_mean_density(samples, by_name)
        tables[group], final_sets[group] = select_top30(mean_density, by_name, nodes, children_map, group)
        out = OUT_DIR / f"top30_{group}.csv"
        tables[group].to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[{group}] wrote {out}")
    return tables, final_sets


# ------------------------------------------------------------------- step 2 --
def polar(theta, r=1.0):
    return r * math.sin(theta), r * math.cos(theta)


def quad_bezier(p0, pc, p1, n=60):
    t = np.linspace(0.0, 1.0, n)[:, None]
    p0 = np.asarray(p0); pc = np.asarray(pc); p1 = np.asarray(p1)
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * pc + t ** 2 * p1


def ribbon_points(a0, a1, rc):
    """rim-hugging band between rim angles a0 and a1 (radians, rim radius 0.995);
    control point sits on the bisector at radius rc (adaptive, never the center)."""
    am = (a0 + a1) / 2.0
    if abs(a1 - a0) > math.pi:  # go the short way for the control point
        am += math.pi
    c = polar(am, rc)
    return (quad_bezier(polar(a0, 0.995), c, polar(a1, 0.995)), am)


def group_colors(groups):
    """One color per group. The first groups get stable hand-picked colors
    (d22 blue, SY1 orange, then green/purple/...); any further groups take
    hues evenly spread over the wheel with the red sector reserved for the
    shared regions."""
    fixed = ["#0072B2", "#D55E00", "#009E73", "#6A51A3", "#CC79A7", "#56B4E9"]
    out = {g: fixed[i] for i, g in enumerate(groups[:len(fixed)])}
    rest = groups[len(fixed):]
    for i, g in enumerate(rest):
        hue = ((30.0 + 300.0 * (i + 0.5) / len(rest)) % 360.0) / 360.0
        r, gc, b = colorsys.hsv_to_rgb(hue, 0.72, 0.92)
        out[g] = (r, gc, b, 1.0)
    return out


def draw_chord(tables, final_sets, by_name, nodes, children_map):
    union = sorted(set().union(*final_sets.values()),
                   key=lambda n: by_name[n]["id"])
    groups = list(GROUP_SAMPLES)
    shared = {n for n in union if sum(n in final_sets[g] for g in groups) > 1}

    # anchor = depth-4 ancestor; sub = depth-5 slot (own id for depth-5 rows)
    def anchor(full):
        node = by_name[full]
        while node["depth"] > DEPTH_START and "parent" in node:
            node = nodes[node["parent"]]
        return node["name"]

    def sub(full):
        node = by_name[full]
        if node["depth"] <= DEPTH_START + 1:
            return node["name"]
        return nodes[node["parent"]]["name"]

    # density for ordering: mean across groups (NaN -> 0)
    dens = {}
    for full in union:
        vals = []
        for group in groups:
            row = tables[group].loc[tables[group]["region_name"] + "," + tables[group]["acronym"] == full,
                                    "mean_density"]
            if len(row) and np.isfinite(row.iloc[0]):
                vals.append(float(row.iloc[0]))
        dens[full] = float(np.mean(vals)) if vals else 0.0

    clusters = {}
    for full in union:
        clusters.setdefault(anchor(full), []).append(full)
    cluster_order = sorted(clusters, key=lambda a: -max(dens[f] for f in clusters[a]))
    ordered = []
    for a in cluster_order:
        subs = {}
        for f in clusters[a]:
            subs.setdefault(sub(f), []).append(f)
        for s in sorted(subs, key=lambda s: -max(dens[f] for f in subs[s])):
            ordered += sorted(subs[s], key=lambda f: -dens[f])

    # ---- angular layout: proportional chord diagram --------------------------
    # every ribbon occupies arc length proportional to sqrt(density) at BOTH
    # ends and completely fills it; a region's total arc = sum of its ribbons
    # (a shared region shows two separate segments, one per ribbon)
    circle_order = ordered + groups
    # parent clusters are marked by adjacency only: small gap inside a cluster,
    # a much larger one wherever the parent-cluster key changes
    INTRA_GAP, BOUNDARY_GAP, SPLIT_GAP = 0.8, 3.8, 0.4
    keys = [anchor(f) for f in ordered] + groups
    n_nodes = len(circle_order)
    boundary = [keys[i] != keys[(i + 1) % n_nodes] for i in range(n_nodes)]

    def density_of(g, f):
        row = tables[g].loc[tables[g]["region_name"] + "," + tables[g]["acronym"] == f,
                            "mean_density"]
        return float(row.iloc[0]) if len(row) and np.isfinite(row.iloc[0]) else 0.0
    all_vals = [density_of(g, f) for g in groups for f in final_sets[g]]
    glo, ghi = min(all_vals), max(all_vals)
    if ghi <= glo:
        glo, ghi = 0.0, 1.0

    def width_frac(g, f):  # relative ribbon weight: sqrt(density), 0.15 .. 1.0
        v = density_of(g, f)
        t = (math.sqrt(v) - math.sqrt(glo)) / (math.sqrt(ghi) - math.sqrt(glo))
        return 0.15 + 0.85 * min(max(t, 0.0), 1.0)

    region_members = {f: [(g, width_frac(g, f)) for g in groups if f in final_sets[g]]
                      for f in ordered}
    hub_members = {g: [(f, width_frac(g, f)) for f in ordered if f in final_sets[g]]
                   for g in groups}
    weight_sum = sum(w for f in ordered for _, w in region_members[f])
    n_shared = sum(1 for f in ordered if len(region_members[f]) > 1)
    gap_total = (INTRA_GAP * n_nodes
                 + sum(BOUNDARY_GAP - INTRA_GAP for b in boundary if b)
                 + SPLIT_GAP * n_shared)
    base_deg = (360.0 - gap_total) / (2.0 * weight_sum)

    theta = 0.0  # degrees, clockwise from 12 o'clock
    region_slots, group_slots, node_mid = {}, {}, {}
    for i, node in enumerate(circle_order):
        theta += BOUNDARY_GAP if boundary[i - 1] else INTRA_GAP
        if node in groups:
            slots = []
            for f, w in hub_members[node]:
                slots.append((f, math.radians(theta), math.radians(theta + w * base_deg)))
                theta += w * base_deg
            group_slots[node] = slots
            node_mid[node] = (slots[0][1] + slots[-1][2]) / 2.0
        else:
            mem = region_members[node]
            slots = []
            for j, (g, w) in enumerate(mem):
                slots.append((g, math.radians(theta), math.radians(theta + w * base_deg)))
                theta += w * base_deg
                if j < len(mem) - 1:
                    theta += SPLIT_GAP
            region_slots[node] = slots
            node_mid[node] = (slots[0][1] + slots[-1][2]) / 2.0

    GROUP_COLORS = group_colors(groups)
    RED = "#D62728"  # regions present in >=2 groups' top-30: one red ribbon per group

    fig = plt.figure(figsize=(19, 19), facecolor="white")
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.55, 1.55)

    # ribbons first (under arcs): shared (red) drawn last so it stays readable
    # over the group fans; ribbons span their arc segments exactly
    ribbons = []
    for g in groups:
        for f, t0, t1 in group_slots[g]:
            _, reg_t0, reg_t1 = [s for s in region_slots[f] if s[0] == g][0]
            ribbons.append({
                "group": g, "region": f,
                "shared": f in shared,
                "e0": t0, "e1": t1, "r0": reg_t0, "r1": reg_t1,
            })
    def ctrl_radius(th_a, th_b):
        d = abs((th_b - th_a + math.pi) % (2 * math.pi) - math.pi)  # 0..pi
        return 0.62 - 0.34 * (d / math.pi)  # close pairs hug the rim, far pairs dip deeper
    ribbons.sort(key=lambda rb: rb["shared"])  # unique first, shared on top
    for rb in ribbons:
        col = RED if rb["shared"] else GROUP_COLORS[rb["group"]]
        ca = ctrl_radius(rb["e0"], rb["r0"])
        cb = ctrl_radius(rb["e1"], rb["r1"])
        up_a, _ = ribbon_points(rb["e0"], rb["r0"], ca)
        up_b, _ = ribbon_points(rb["e1"], rb["r1"], cb)
        side = np.vstack([up_a, up_b[::-1]])
        ax.add_patch(Polygon(side, closed=True, facecolor=col,
                             alpha=0.45, edgecolor="none", zorder=1))

    # node arcs (Wedge uses math CCW angles: polar() theta -> math angle 90-theta)
    # one color per region: owner group, red for regions in both top-30s;
    # a shared region draws one Wedge PER ribbon (two red segments)
    R_OUT = 1.0
    for f in ordered:
        gs = [g for g in groups if f in final_sets[g]]
        fill = RED if len(gs) > 1 else GROUP_COLORS[gs[0]]
        for _, t0, t1 in region_slots[f]:
            ax.add_patch(Wedge((0, 0), R_OUT, 90 - math.degrees(t1), 90 - math.degrees(t0),
                               width=0.04, facecolor=fill,
                               edgecolor="#bbbbbb", linewidth=0.5, zorder=3))
    for g in groups:
        t0, t1 = group_slots[g][0][1], group_slots[g][-1][2]
        ax.add_patch(Wedge((0, 0), R_OUT, 90 - math.degrees(t1), 90 - math.degrees(t0),
                           width=0.04, facecolor=GROUP_COLORS[g],
                           edgecolor="#1a1a1a", linewidth=1.2, zorder=3))

    # one label per arc, anchored at its footprint midpoint, always along the
    # radius; labels on the left half flip 180 (ha="right") so they never read
    # upside-down while staying radially aligned
    halo = [patheffects.withStroke(linewidth=2.5, foreground="white")]
    def radial_label(text, thm, color, size, bold=False):
        deg = math.degrees(thm) % 360.0
        rot = 90.0 - deg
        if rot > 90.0 or rot < -90.0:
            rot += 180.0 if rot > 0 else -180.0
            ha = "right"
        else:
            ha = "left"
        x, y = polar(thm, R_OUT + 0.03)
        ax.text(x, y, text, rotation=rot, ha=ha, va="center", fontsize=size,
                path_effects=halo, fontweight=("bold" if bold else "normal"),
                color=color, zorder=4)
    for f in ordered:
        _, acr = split_name(f)
        radial_label(acr, node_mid[f], "#222222", 8.5)
    for g in groups:
        radial_label(g, node_mid[g], GROUP_COLORS[g], 13, bold=True)

    handles = [Patch(facecolor=GROUP_COLORS[g], label=f"{g} only") for g in groups]
    handles.append(Patch(facecolor=RED,
                         label=("in both groups' top-30" if len(groups) == 2
                                else f"in >=2 of {len(groups)} groups' top-30")))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0),
              frameon=False, fontsize=10)
    ax.set_title(f"cFos top-30 regions: {' vs '.join(groups)} (chord diagram)",
                 fontsize=17, fontweight="bold", pad=28)
    fig.text(0.5, 0.028,
             f"Group colors per legend; red = region present in >=2 groups' top-30 "
             f"(one red ribbon per group).\n"
             f"Same-parent regions sit adjacent on the circle (large gaps separate parents). "
             f"Ribbon width \u221d \u221a(mean density, count/mm\u00b3); arc length = total width "
             f"of the ribbons it carries (ribbons fill their arcs).",
             ha="center", fontsize=9.5, color="#444444")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg = OUT_DIR / f"chord_top30_{'_'.join(groups)}.svg"
    png = OUT_DIR / f"chord_top30_{'_'.join(groups)}.png"
    fig.savefig(svg, facecolor="white", bbox_inches="tight")
    fig.savefig(png, facecolor="white", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {svg}")
    print(f"wrote {png}")
    print(f"union={len(union)} regions, shared={len(shared)}, unique={len(union) - len(shared)}")


def main():
    font_manager.fontManager  # init
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["svg.fonttype"] = "none"
    nodes, by_name, children_map = load_tree(REGION_CSV)
    print(f"region tree ok: {len(nodes)} nodes, root depth counts "
          f"{pd.Series([n['depth'] for n in nodes.values()]).value_counts().sort_index().to_dict()}")
    tables, final_sets = main_step1(nodes, by_name, children_map)
    for group in GROUP_SAMPLES:
        print(f"--- top30 {group} ---")
        print(tables[group].to_string(index=False))
    draw_chord(tables, final_sets, by_name, nodes, children_map)


if __name__ == "__main__":
    main()
