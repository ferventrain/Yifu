---
name: yifu-work
description: 用户要求跑分析、处理样本、出报告、查结果、操作 PM 看板，或做项目级组合分析（组合指标/出图/交付）等"用现有程序干活"的工作模式细则。与 yifu-develop（写新代码）相对：本 skill 是调用方视角——先读契约再动手，不猜不改仓库程序。
---

# Yifu 工作模式（调用现有程序干活）

大部分时间是这个模式：用户给样本/项目，用仓库现成程序跑出结果。
与开发模式（skill `yifu-develop`）相对。行为准则与意图路由见根 `AGENTS.md`；
本 skill 持有调用姿势与项目分析工作模式的细则。

## 进入姿势（按序）

1. **读契约**：根 `capabilities.json` → 涉及模块的 `capability_manifest.json`。
   可用程序、CLI 参数、输入输出、图谱体系（registration manifest 的 `atlas_reference`）都在那里。
2. **跑流程**：标准入口三件套——`preflight`（只读校验）→ `harness run`（提交，返回 run_id）
   → PM 看板（http://127.0.0.1:8766）跟踪。状态与成败只认结构化错误
   （`error.code / step_id / retryable / suggestion`），不认日志文字。
3. **config 规则**：template 已定的参数不重复写；run config 只写覆盖项；
   继承链 `config_1 ← config.json ← cfos_template`。首建必查三项：
   `project_name`、`input.resolution_xyz`、`input.channels`。
4. **环境**：一切命令用 micromamba **yifu env** 的 python；fresh shell 带
   `YIFU_ANALYSIS_ROOT=S:/Arivis_Analysis YIFU_ACTIVE_DIR=S:/Arivis_Analysis/_active`。

## 边界（什么时候切换到别的 skill）

- 要**写新代码**（新工具、新模块）→ 先读 `yifu-develop`，
  并按 AGENTS.md 宗旨 4 确认代码归属（repo / 项目文件夹用完即删）。
  项目分析的组合脚本是例外：固定放项目文件夹，见下节第 2 条。
- 要动 `pipeline_modules/` 里的程序 → 默认禁止；明确 bug 才最小修复+跑测试+报告。
- 涉及 ims/arivis 读写细节 → `.zcode/skills/yifu-develop/ims-arivis-integration.md`。

## 项目分析工作模式

1. **repo 只调用、不改动**：跑项目分析时不修改 `pipeline_modules/` 里的程序。唯一例外是发现明确 bug：
   先向用户报告，做最小修复，并跑相关测试（例：slice_heatmap_render 的 colorbar 标签被画布裁掉）。
2. **组合分析程序放项目文件夹**：为特定分析组合新写的程序（多次调用 repo 程序、拼不同指标）写到
   `<analysis_root>/_active/<项目>/` 下，不进 repo；repo 程序按 CLI / capability_manifest 契约调用。
3. **结果归一**：每个项目跑完，所有分析结果（Excel/图表/热图等）集合到该项目的单一结果文件夹
   （如 `<项目>/_reports/`），不散落在各样本目录。
4. **说明文档（PDF）**：结果文件夹必须附一份面向用户的 PDF 说明，逐项解释图表和表格——每个指标的
   含义、口径（如圆柱法表面积、血管环数定义）与使用注意；交付前用视觉验收过一遍。
5. **项目状态卡**：每个项目维护 `_active/<项目>/PROJECT.md` 一页纸——目标、样本清单与状态、
   当前卡点、已知坑、交付物指针、下一步。接手项目先读它；出现新结论、新坑或阶段完成时随手更新。
6. **项目收尾 debrief（固定流程）**：项目完成"分析 + 出图"交付后做一次（单个任务不做，免得浪费）：
   ① 把 `PROJECT.md` 收成终态（结果结论 + 遗留事项）；② 列一份"本次新经验 → 建议写入哪个文档"
   清单问用户（对应 AGENTS.md 宗旨 6）；③ 给出本项目临时文件（项目内 / 样本 Temp / 仓库 `_tmp/`）
   的清理清单，确认后清掉。
