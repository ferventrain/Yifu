---
name: yifu-develop
description: 开发新程序、新工具、新模块、napari/Qt 小部件、管线新步骤或任何"给我做一个新东西"，以及排查"结果不对 / 图不对 / 跑不上"类症状之前必读的开发守则。包含症状定位（先列假设再实验、数字定位 + 人眼裁决）、程序化复现、插桩定位、量化验收、GUI/线程/生命周期防御、宿主程序源码对齐的方法论。
---

# Yifu 开发守则

写新程序（管线模块、可视化工具、napari/Qt 小部件、组合分析脚本）时的方法论。
不是流程清单，是"怎么想、怎么做才算确定"的判断力。

## 背景：3D 光片图像分析（新工程师快速入门）

- **成像**：SmartSpim 光片显微镜，全脑 3D 成像（NAS 目录名沿用旧称 `MegaSpim`，是同一台）。
- **体量常识**：单通道全脑 TB 级（典型 (4475, 32752, 18317) @ 0.71/0.71/2.0 µm）。
  任何新代码默认按"金字塔懒加载 / 分块 / zarr"设计，全量载入不是优化项而是设计错误。
- **我们的栈**：`pipeline_modules/` 六模块——preprocessing（ims↔zarr、降采样 NIfTI）、
  registration（ANTs + Allen 图谱）、segmentation（cfos UNet / spotiflow）、
  tubule_reconstruction（血管骨架）、visualization（napari 小部件、报告出图）、
  harness（PM 看板、wrap/进度协议）。可用程序与 CLI 契约：根 `capabilities.json`
  → 各模块 `capability_manifest.json`，开工先读，不凭记忆猜。
- **环境**：运行时一律 micromamba **yifu env** 的 python；napari env 仅作查看器；
  任务经 PM（http://127.0.0.1:8766）提交跟踪；Windows + GPU（torch cu128）。
- **存储**：repo `S:/Yifu`；分析根 `S:/Arivis_Analysis/_active`（批=父目录）；
  原始 ims 在 NAS `//tnas-ad13/MegaSpim/`（UNC 路径两端引号、映射盘符等坑见各工具文档）。
- **常用格式一句话**：zarr v2 / NGFF（multiscales + image-label 键=napari Labels 层）、
  NIfTI（配准空间）、TIFF 栈（中间产物）、SWC（神经元形态）、
  ims/arivis 细则见同目录 `ims-arivis-integration.md`。

## 1. 症状排查：先定位域，再动手

用户说"不对 / 再查一遍"，正确动作是**出定位实验，不是重读代码**；定位实验没做就改代码
= 在错误假设下烧 token。顺序：先列假设 → 便宜实验 → 人眼裁决 → 有思路才改。

1. **先列假设**：把症状的可能原因列成 2-4 个候选——新输入/新配置违反隐含假设、计算错、
   渲染错、查看工具自身错、不是 bug 是数据本性——每个标一个最便宜的判别办法。
   理论先行，再进数据。
2. **时间二分（谁最新谁最可疑）**：程序近期在别的样本上验收过 → 代码降为低嫌疑，
   先 diff 这次输入/config 与上次成功 run 的差异（配准通道换语义这类真凶，diff 一眼可见）。
3. **空间二分（数字和图切开）**：图像类症状先 dump 进入绘图函数前的数组（npy/CSV +
   min/max/分位数），再用最笨的默认画法重画同一份数字：数字对图不对 → 渲染；数字不对 → 往上游切。
   这一刀同时自证查看工具本身没坏（QC 渲染器三连 bug 曾把判断连续带偏三轮）。
4. **新方法首跑交生效证据**：新旧方法同输入 A/B，报告实际改变了什么（剔了多少体素 + diff 图）；
   "新旧结果很像"必须能区分"没生效"和"本来就没多少可改"。
5. **数字定位、人眼裁决**：数字回答"哪一段坏了"；图像"对不对"只有用户能判。
   数字全绿 ≠ 问题不存在——用户报告的视觉症状是待解释的输入，不是要被"指标正常"推翻的对象；
   数字与眼睛矛盾说明查的维度不对，换维度，不要和用户争论谁对。
6. **需要人眼确认直接派单**：图像问题多数实验代替不了人看。给用户单步操作清单
   （打开哪个文件、看哪里、把什么现象带回来），等结果再走下一步；不连环自主实验烧半天 token。

案例锚点：配准反复失败的真凶 = 首次用非自发荧光通道做配准（隐含假设被违反，不是代码 bug）；
用户对可视化异常的肉眼直觉多次命中真 bug——症状本身是最高优先级的输入。

## 2. 复现优先于修复

定位到代码或交互层之后，先把"不行"变成**可程序化驱动的复现**，再动手改。

- GUI/交互问题：搭 headless 驱动（napari `Viewer(show=False)` + `QApplication.processEvents()` 泵），像操作真用户一样设置值、点按钮、切层级。
- 没有 100% 复现路径就改代码 = 盲修。这次 napari+neurofly 的全部根因（choices 被清空、scale 公式、L0 物化）都是靠驱动脚本逐层定位的，不是靠看代码看出来的。
- 复现脚本放临时目录，用完删，不进仓库。

## 3. 插桩二分，找到"哪一步"而不是"哪个文件"

状态被破坏类的问题（面板失灵、参数变 None），在每个可疑步骤后打印关键状态
（如 `len(choices)`），把破坏点夹逼到单步。修完一步立刻复跑验收，不攒着。

## 4. 验收必须量化

对齐/匹配类需求用数字验收：bbox 对比、中心偏移、比例，明确"多少以内算过"。
"看起来差不多"不算验收。验收数字写进汇报，让用户能复核。

## 5. 集成宿主前先读它的源码

要和 neurofly / arivis / napari 插件等宿主共存时，先读 `site-packages` 里它的
实现：轴序约定（xyz vs zyx）、scale/translate 单位、数据是否 padding、事件流
（谁 reset 谁）。**按它真实的约定对齐，不按文档印象猜。**
本次教训：neurofly 的 spacing 是 µm/体素且各层非严格 2 倍——不读源码永远猜不到。

## 6. 跨 GUI 集成三防

给宿主程序做伴生面板/插件时，防三件事：

- **宿主全局副作用**：宿主框架可能 reset 你依赖的组件（napari 图层事件清空
  magicgui choices）→ 装守卫 + 周期巡检 + 恢复时先断信号再重连；
- **生命周期**：面板销毁后回调必须静默退出（`_alive()` 守卫、QTimer 挂在
  面板下、destroyed 时断开 viewer 级连接）；
- **线程**：zarr3 async × blosc × Qt 事件循环会非确定性 MemoryError →
  blosc 单线程 + zarr 并发 1 + 短重试；大数据宁可同步物化小结果。

## 7. 交互工具的形态与顺序

napari 内的工具做成 **npe2 插件包**（`pip install -e`，参照
`pipeline_modules/visualization/napari_point_cropper`），用户从 Plugins 菜单
打开。集成顺序：宿主面板（neurofly）先开、我们的后挂——避免被宿主初始化时
的 `layers.clear()` / `remove_dock_widget('all')` 清掉。

## 8. 环境噪音 ≠ 代码 bug

偶发 MemoryError / 卡死先查环境：邻居进程（渲染类可瞬间提交几十 GB）、
Windows commit limit、机器是否在跑别的任务。邻居噪音的特征是"同代码时好时坏
+ 系统空闲时单测通过"。识别后等待或让用户协调，不要当 bug 修。

## 9. 节奏

- 一次只修**已证实**的一个原因，修完复跑验收再动下一个；
- 用户说"先别 commit/push"就攒着本地，验收全绿后一次提交；
- 改动最小化：能加守卫不改架构，能后挂不侵入宿主。

## 10. 接线与测试纪律

- 改 main.py 流程、加新步骤或新模块接线后，必须跑一遍 **stub 接线测试**（不跑算法、几秒完成，
  验证 6 步路径 / 键名 / 分支不断链），并让 `tests/test_main_stub_run.py` 的 EXPECTED_COMMANDS
  覆盖新步骤的命令描述。手动跑法：空样本目录放 config 后
  `python main.py --config <cfg> --sample_dir <dir> --stub`，末行 `STUB_SUMMARY` 应列出全部命令。
- 测试方针：秒级可跑、防"要跑真样本才能发现"的错 → 常驻 `tests/`；一次性输出验证 → 用完即删。
  新模块自身的算法测试照常进 `tests/`；失败测试不修就删或改，不留长期红测试。

## 相关仓库约定

- 新代码归属先问用户（AGENTS.md 宗旨 4）；仓库 vs 项目文件夹用完即删。
- 新程序让 PM 可读（wrap / report / attach 三选一）；GUI 交互工具除外
  （无任务语义，如 napari_point_cropper 也不登记）。
