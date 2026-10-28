"""neurofly 全脑神经元分割推理（atlas 掩码限制 + 断点续跑）。

复用 neurofly 官方推理链的全部契约，保证与预训练一致：
- **权重**：neurofly 自带 universal_tiny.pth（VISoR/fMOST 训练）；
- **预处理**：与 ``neurofly.seger.Seger.get_large_mask`` 完全同路（逐 100³ 块
  min-max 归一化到 [0,1]、4 体素 border、300³ chunk、bg_thres 体素门控）——
  本模块直接调用它们的函数，不做任何改写；
- **输入块尺寸**：100³ 块 + 2×4 border（pad 后 108³）与官方 seg CLI 一致，
  等于参考推理/训练块尺寸；
- **后端**：tinygrad 0.14 与 neurofly 0.1.4 源码不兼容（CreationMixin API 变更），
  使用 neurofly 自带的 torch 后端 ``models/unet_torch.SegNet``（同一权重文件、
  同一预处理语义），走本机 torch cu128 GPU。

在此之上增加：
1. **atlas 掩码限制**：只在 ``upsampled_atlas_label.zarr``（信号网格）有标注的
   区域推理——每个 chunk 先查掩码覆盖率，全覆盖为 0 直接跳过（脑外块零成本），
   有覆盖则推理后把分割 mask 与掩码相与再骨架化；
2. **断点续跑**：逐 chunk 结果落 sidecar jsonl，重跑自动跳过已完成块；
3. **PM 进度**：stdout 打 ``PROGRESS done/total [segment]``，wrap 包装即有进度条。

用法（yifu env；经 PM wrap 提交见下）::

    python -m pipeline_modules.segmentation.neurofly_brain_seg \
        --image "//tnas-ad13/MegaSpim/<brain>.ims" \
        --label-zarr S:/Arivis_Analysis/_active/<sample>/upsampled_atlas_label.zarr \
        --db S:/.../neurofly_segments.db \
        [--level 2] [--channel 0] [--chunk-size 300] [--bg-thres 150] \
        [--min-coverage 0.0] [--device cuda]

--level 选择要点：块尺寸契约不变的前提下，应选体素尺寸与训练数据相当的层级
（VISoR/fMOST ≈ 每轴 1.6–4 µm 量级）；启动时会打印该层各轴 spacing 供核对。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

PROGRESS_EVERY = 1


# --------------------------------------------------------------------------
# 纯函数：chunk(ims level, XYZ) ↔ atlas label zarr(ZYX) 的网格映射与门控
# --------------------------------------------------------------------------

def axis_ratios(image_info: list[dict], level: int) -> tuple[float, float, float]:
    """ims level 体素 → L0 信号网格体素的每轴倍率 (x, y, z)。

    info[level]['spacing'] 是 **µm/体素**（neurofly 口径，随层变）；L0 体素比
    = spacing[level] / spacing[0]。label zarr L0 即 L0 信号网格。
    """
    spacing = image_info[level]["spacing"]
    spacing0 = image_info[0]["spacing"]
    return tuple(float(s) / float(s0) for s, s0 in zip(spacing[:3], spacing0[:3]))


def label_window_for_chunk(
    roi_xyz: list[int],
    ratios_xyz: tuple[float, float, float],
    label_shape_zyx: tuple[int, int, int],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """chunk roi [x,y,z,sx,sy,sz]（level 坐标）→ label L0 的 zyx 半开窗口。

    label 网格 = ims L0 真实 extent（无 padding、原点 0），level 坐标与 L0 坐标
    是逐轴线性关系，直接按倍率缩放并夹边。
    """
    x, y, z, sx, sy, sz = roi_xyz
    lz, ly, lx = label_shape_zyx
    rx, ry, rz = ratios_xyz
    x0 = int(np.clip(round(x * rx), 0, lx))
    x1 = int(np.clip(round((x + sx) * rx), 0, lx))
    y0 = int(np.clip(round(y * ry), 0, ly))
    y1 = int(np.clip(round((y + sy) * ry), 0, ly))
    z0 = int(np.clip(round(z * rz), 0, lz))
    z1 = int(np.clip(round((z + sz) * rz), 0, lz))
    return (z0, z1), (y0, y1), (x0, x1)



def _reduce_any(block: np.ndarray, edges: np.ndarray, axis: int) -> np.ndarray:
    """reduceat 版 any-pool：按 edges 左端点分箱，任一 True 则 True。"""
    moved = np.moveaxis(block, axis, 0)
    if len(edges) <= 1:
        reduced = moved.astype(np.uint8)[None]
    else:
        reduced = np.add.reduceat(moved.astype(np.uint8), edges.tolist(), axis=0)
    return np.moveaxis(reduced > 0, 0, axis)


def coarse_atlas_mask(
    label_block_zyx: np.ndarray,
    chunk_shape_xyz: tuple[int, int, int],
) -> np.ndarray:
    """label 块 (zyx) → chunk 网格 (xyz) 的布尔掩码（精确 any-pool）。

    cell (i,j,k) 为 True ⟺ 对应 label 窗口内存在任一非零标注体素——cell 足迹
    完整覆盖，细薄脑区边界不会因采样而漏掉。
    """
    sx, sy, sz = chunk_shape_xyz
    block = label_block_zyx > 0
    if block.size == 0:
        return np.zeros((sx, sy, sz), dtype=bool)
    # label 轴 (z,y,x) -> chunk 轴 (x,y,z)：逐轴 any-pool 后转置
    block = _reduce_any(block, _bin_edges(block.shape[0], sz), 0)
    block = _reduce_any(block, _bin_edges(block.shape[1], sy), 1)
    block = _reduce_any(block, _bin_edges(block.shape[2], sx), 2)
    return np.ascontiguousarray(np.transpose(block, (2, 1, 0)))  # zyx -> xyz


def _bin_edges(length: int, n_out: int) -> np.ndarray:
    """把 [0,length) 均分成 n_out 个 bin 的左端点（reduceat 用）。

    n_out > length 时后段 bin 左端点重复（reduceat 对相邻相等索引给出单元素
    bin），保证输出恒为 n_out 个 bin。
    """
    if length <= 0:
        return np.array([0])
    n_out = max(n_out, 1)
    return np.minimum(np.arange(n_out) * length // n_out, length - 1)


def pick_label_level(
    label_levels_available: list[int],
    ratios_xyz: tuple[float, float, float],
) -> int:
    """选体素不粗于图像最粗轴的最粗 label 金字塔层（读取量最小）。

    label 层 k 体素 = 2^k 个 L0 体素；门控采样只要求 label 不比图像最粗轴更粗，
    即 2^k <= max(ratios)。不满足则退最细层。
    """
    if not label_levels_available:
        return 0
    rmax = max(ratios_xyz) if ratios_xyz else 1.0
    candidates = [k for k in label_levels_available if 2.0**k <= max(rmax, 1.0)]
    return max(candidates) if candidates else min(label_levels_available)


# --------------------------------------------------------------------------
# 序列化 / 断点续跑
# --------------------------------------------------------------------------

def seg_to_jsonable(seg: dict) -> dict:
    out = {}
    for key, value in seg.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, np.generic):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def seg_from_jsonable(seg: dict) -> dict:
    """jsonl 往返还原。points/sampled_points 保持 python list——segs2db 对
    坐标元素直接 str()，np 数组行会变成 '[ 47 190  69]'（无逗号）导致
    read_nodes 的 eval 崩溃；python list 才是 neurofly 原生口径。"""
    out = {}
    for key, value in seg.items():
        out[key] = value
    return out


def load_sidecar(path: Path) -> dict[str, list[dict]]:
    """{roi_key: segs}，roi_key = ','.join(roi)。坏行跳过。"""
    done: dict[str, list[dict]] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            done[record["key"]] = [seg_from_jsonable(s) for s in record["segs"]]
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def append_sidecar(path: Path, roi_key: str, segs: list[dict]) -> None:
    record = {"key": roi_key, "segs": [seg_to_jsonable(s) for s in segs], "ts": time.time()}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=_json_default) + "\n")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def make_seger_torch(ckpt_path: str, bg_thres: int):
    """构造行为与 neurofly.seger.Seger 相同的对象，但 seg_net 用 torch 后端。

    tinygrad 0.14 与 neurofly 0.1.4 源码不兼容（CreationMixin.full 不再收
    requires_grad）；torch 后端同一权重、同一预处理语义。get_large_mask /
    mask_to_segs / postprocess 全部原样复用 Seger 类方法。
    """
    from neurofly.models.unet_torch import SegNet as SegNetTorch
    from neurofly.seger import Seger

    seger = Seger.__new__(Seger)
    seger.seg_net = SegNetTorch(ckpt_path, bg_thres=bg_thres)
    seger.bw = 4
    return seger


def run(
    image_path: str,
    label_zarr: str,
    db_path: str,
    *,
    ckpt_path: str | None = None,
    level: int = 0,
    channel: int = 0,
    chunk_size: int = 300,
    bg_thres: int = 150,
    splice: int = 300,
    min_coverage: float = 0.0,
    sidecar: str | None = None,
    max_chunks: int | None = None,
    skip_max_below: int = 0,
) -> dict:
    from neurofly.common import default_seger_weight_path
    from neurofly.dbio import segs2db
    from neurofly.image_reader import wrap_image
    from neurofly.patch import patchify_without_splices

    ckpt_path = ckpt_path or default_seger_weight_path
    image = wrap_image(image_path)
    info = image.info
    spacing = info[level].get("spacing")
    print(f"image level {level}: true size {info[level].get('image_size')}, "
          f"spacing(L0vox/levelvox) {spacing}", flush=True)

    # ---- atlas 掩码源 ----
    import zarr

    from pipeline_modules.visualization.napari_atlas_overlay import label_levels

    label_store = Path(label_zarr)
    levels_avail = label_levels(label_store)
    ratios = axis_ratios(info, level)  # level 体素 -> L0 体素
    label_k = pick_label_level(levels_avail, ratios)
    group = zarr.open_group(store=str(label_store), mode="r")
    label_arr = group[str(label_k)]
    label_shape_k = tuple(int(v) for v in label_arr.shape)  # zyx
    k_factor = 2.0**label_k  # label_k 体素 = k_factor 个 L0 体素
    ratios_k = tuple(r / k_factor for r in ratios)  # level 体素 -> label_k 体素
    print(f"atlas label level L{label_k} shape(zyx) {label_shape_k}", flush=True)

    # ---- chunk 划分（与 process_whole 一致）----
    rois = patchify_without_splices(image.rois[level], chunk_size, splices=splice)
    print(f"total chunks: {len(rois)}", flush=True)

    sidecar_path = Path(sidecar) if sidecar else Path(db_path).with_suffix(".chunks.jsonl")
    done = load_sidecar(sidecar_path)
    seger = make_seger_torch(ckpt_path, bg_thres)

    skipped = 0
    inferred = 0
    t_start = time.time()
    for idx, roi in enumerate(rois):
        roi = [int(v) for v in roi]
        key = ",".join(str(v) for v in roi)
        if key in done:
            continue

        # ---- atlas 门控：先算覆盖率，全覆盖为 0 跳过 ----
        (z0, z1), (y0, y1), (x0, x1) = label_window_for_chunk(roi, ratios_k, label_shape_k)
        if z1 <= z0 or y1 <= y0 or x1 <= x0:
            coverage = 0.0
            gate = None
        else:
            label_block = np.asarray(label_arr[z0:z1, y0:y1, x0:x1])
            gate = coarse_atlas_mask(label_block, tuple(roi[3:]))
            coverage = float(gate.mean())
        if coverage <= min_coverage:
            skipped += 1
            append_sidecar(sidecar_path, key, [])
            done[key] = []
            print(f"PROGRESS {idx + 1}/{len(rois)} [segment] skip coverage={coverage:.3f}", flush=True)
            continue

        # ---- 读图像块（与 process_whole 同路：小直通 / 大块走 border 分块）----
        if (np.array(roi[3:]) <= np.array([128, 128, 128])).all():
            img = image.from_roi(roi, level, channel, padding="reflect")
            offset = roi[:3]
            gate_i = gate
        else:
            padded_roi = list(roi)
            padded_roi[:3] = [i - seger.bw for i in padded_roi[:3]]
            padded_roi[3:] = [i + seger.bw * 2 for i in padded_roi[3:]]
            img = image.from_roi(padded_roi, level, channel, padding="reflect")
            offset = [i + seger.bw for i in padded_roi[:3]]
            # get_large_mask 返回前自裁 border，mask 与未 pad 的 gate 同形
            gate_i = gate

        # ---- 块强度门控：max ≤ skip_max_below 直接跳过（省推理与骨架化）----
        if skip_max_below and int(np.max(img)) <= skip_max_below:
            skipped += 1
            append_sidecar(sidecar_path, key, [])
            done[key] = []
            print(f"PROGRESS {idx + 1}/{len(rois)} [segment] skip max<={skip_max_below}", flush=True)
            continue

        mask = seger.seg_net.get_mask(img) if (np.array(roi[3:]) <= np.array([128, 128, 128])).all() \
            else seger.get_large_mask(img)

        # ---- atlas 掩码相与后骨架化 ----
        mask = np.asarray(mask) > 0
        if gate_i is not None:
            mask &= gate_i
        _, segs_in_block = seger.mask_to_segs(mask.astype(np.uint8), offset=offset)
        append_sidecar(sidecar_path, key, segs_in_block)
        done[key] = segs_in_block
        elapsed = time.time() - t_start
        done_n = idx + 1
        rate = done_n / max(elapsed, 1e-6)
        eta = (len(rois) - done_n) / max(rate, 1e-6)
        print(f"PROGRESS {done_n}/{len(rois)} [segment] cov={coverage:.2f} "
              f"segs={len(segs_in_block)} eta={eta / 60:.0f}min", flush=True)
        inferred += 1
        if max_chunks is not None and inferred >= max_chunks:
            print(f"max_chunks={max_chunks} reached, stopping early", flush=True)
            break

    # ---- 汇总出库 ----
    all_segs = []
    for roi in rois:
        key = ",".join(str(int(v)) for v in roi)
        for seg in done.get(key, []):
            all_segs.append(seg)
    for i, seg in enumerate(all_segs):
        seg["sid"] = i
    db_out = Path(db_path)
    if db_out.exists():
        db_out.unlink()
    segs2db(all_segs, str(db_out))
    print(f"DB DONE: {len(all_segs)} segments -> {db_out} "
          f"(skipped {skipped} zero-coverage chunks)", flush=True)
    return {"segments": len(all_segs), "skipped_chunks": skipped, "chunks": len(rois)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", required=True, help="ims / zarr 图像路径")
    parser.add_argument("--label-zarr", required=True, help="upsampled_atlas_label.zarr（信号网格）")
    parser.add_argument("--db", required=True, help="输出 SQLite db（Segs Annotator 可直接开）")
    parser.add_argument("--weight", default=None, help="分割权重（默认 neurofly 自带 universal_tiny）")
    parser.add_argument("--level", type=int, default=0, help="ims resolution level（选体素尺寸≈训练数据的层）")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=300, help="与官方 CLI 一致的立方块边长")
    parser.add_argument("--bg-thres", type=int, default=150, help="背景体素门控（官方 CLI 默认）")
    parser.add_argument("--splice", type=int, default=300)
    parser.add_argument("--min-coverage", type=float, default=0.0,
                        help="chunk 的 atlas 覆盖率低于此值即跳过（0=只要有覆盖就推理）")
    parser.add_argument("--sidecar", default=None, help="断点续跑 sidecar（默认 <db>.chunks.jsonl）")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="最多处理 N 个非跳过 chunk 后停止（冒烟验证用；已处理块仍会入库）")
    parser.add_argument("--skip-max-below", type=int, default=0,
                        help="块最大强度 ≤ 此值即跳过（如 100；0=不启用）")
    args = parser.parse_args(argv)

    stats = run(
        args.image, args.label_zarr, args.db,
        ckpt_path=args.weight, level=args.level, channel=args.channel,
        chunk_size=args.chunk_size, bg_thres=args.bg_thres, splice=args.splice,
        min_coverage=args.min_coverage, sidecar=args.sidecar, max_chunks=args.max_chunks,
        skip_max_below=args.skip_max_below,
    )
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
