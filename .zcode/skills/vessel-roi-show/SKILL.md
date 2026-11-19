---
name: vessel-roi-show
description: 血管样本局部区域（ROI）可视化三联图与旋转视频。对已分割+重建的血管样本，给定中心和边长生成同一方块、同一视角的三张图（原始信号角视角 MIP、mask 等值面、直径上色骨架+分支点）以及三段渐变切换的 360° 旋转视频。用户说"局部区域/典型区域出图、ROI 三联图、MIP+mask+骨架对照、旋转视频、方块展示"等时触发。
---

# vessel-roi-show：血管 ROI 三联图 + 旋转视频

## 前置条件（缺一不可，先向用户确认/检查）

1. 原始信号 zarr（dataset `0`，z,y,x 序）——如 `ch1.zarr`
2. 二值血管 mask zarr（dataset `0`）——如 `ch1_signal_thr1500_filtered.zarr`
3. 全脑 SWC 目录（`vessel_swc_export` 的输出，含 `<prefix>.swc`）——骨架从全脑裁剪，勿用局部重建
4. 体素大小（z,y,x µm）——zarr 不存体素大小，从 ims `DataSetInfo/Image` 的 ExtMax÷尺寸读
5. ROI 中心（z,y,x 体素，napari 里看点坐标时按此顺序）与边长（µm）

若 SWC 尚未导出：`python -m pipeline_modules.tubule_reconstruction.vessel_swc_export --run_dir <recon目录> --output_dir <recon目录>/swc --output_prefix <样本名>`。

## 使用（yifu env，仓库根目录为 cwd）

三联图（`roi3d_cube_{raw,mask,vessels}.png` 三个文件到 --out_dir）：

```bash
python .zcode/skills/vessel-roi-show/scripts/roi_trio.py \
  --image_zarr <样本zarr根>/ch1.zarr \
  --mask_zarr <样本zarr根>/ch1_signal_thr1500_filtered.zarr \
  --swc_dir <样本zarr根>/vessel_recon_filtered/swc \
  --center 234,1867,2087 --side_um 1000 \
  --resolution 3.5006,2.0,2.0 \
  --out_dir <结果文件夹>/figures
```

旋转视频（32s 默认：raw MIP→mask→vessels 各 10s 转 360°+2s 渐变）：

```bash
python .zcode/skills/vessel-roi-show/scripts/roi_video.py \
  --image_zarr ... --mask_zarr ... --swc_dir ... \
  --center 234,1867,2087 --side_um 1000 --resolution 3.5006,2.0,2.0 \
  --out <结果文件夹>/roi3d_cube_rotation.mp4
```

常用可调：`--tube_width 9`（骨架管粗，px）、`--branch_sphere_um 7`（分支点球半径；7µm≈12px，约为管宽 1.3 倍）、`--window`（画布边长）、视频的 `--rot_s/--fade_s/--fps`。先出三联图给人眼确认中心与比例，再渲染视频（约 4-6 分钟）。

## 定稿视觉参数（2026-11 用户验收过，勿随意改）

纯黑背景；mask 亮红 `#ff4646`；骨架 bright 色图按直径上色（clip 2-98 百分位）带 Diameter(µm) 色标；分支点红球；白色线框盒；相机=方块左上角顶点方向 (−1,−1,+1)、平行投影、parallel_scale=0.95×边长；三张同画布同相机；视频侧视图绕竖轴旋转。

## 脚本内已固化的坑（改脚本前必读，均曾被用户肉眼抓到）

- 相机焦点/位置必须是 (x,y,z) 序；center_um 是 (z,y,x) 序，`[::-1]` 转换
- marching_cubes 顶点是 (z,y,x)，喂 pyvista 前 `[:, ::-1]`
- pyvista ImageData 体渲染点数据必须 C 序 `raw.ravel()`（order="F" 会搅成灰团）
- 角视角投影轮廓是**高个六边形**（半宽=边长/√2、半高=2×边长/√6），MIP 采样与贴图都按此包围盒
- MIP 贴图行序必须 `np.flipud`（贴图第 0 行画在屏幕上方，采样 iy=0 在下方）
- 贴图 alpha = 射线-方块求交的六边形内掩码，方形贴图会"顶断"线框盒
- 线框盒每帧沿视线朝相机平移（平行投影下轮廓不变、永不被内容遮挡）
- pyvista `screenshot()` 在改相机后返回**缓存帧**，必须先显式 `p.render()`
- 分支点球 glyph 的 scale 数组会**再乘一次**几何半径，固定 1.0、半径只由 Sphere(radius=) 控制
- 视频原始图用 vtk MIP blend：`actor.mapper.SetBlendModeToMaximumIntensity()`（`mapper="maximum_intensity"` 旗标不存在）
- 视频写 mp4 用 cv2.VideoWriter（mp4v）+ RGB2BGR（imageio 无 ffmpeg 会落 tiff 报错）
- 骨架分支只保留**完全在方块内**的（`.all()`），出界 edge 剔除

## 验收

渲染结果一律程序化像素检查（角落背景色、nonbg 占比、抽帧差分确认旋转），人眼判断交用户；判断翻转/镜像类问题用不对称探针图案，不要用跨模态相关（无分辨力）。
