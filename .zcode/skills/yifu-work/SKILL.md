---
name: yifu-work
description: 用户要求跑分析、处理样本、出报告、查结果、操作 PM 看板等"用现有程序干活"的工作模式入门。与 yifu-develop（写新代码）相对：本 skill 是调用方视角——先读契约再动手，不猜不改仓库程序。
---

# Yifu 工作模式（调用现有程序干活）

大部分时间是这个模式：用户给样本/项目，用仓库现成程序跑出结果。
与开发模式（skill `yifu-develop`）相对。细则以 `AGENTS.md` 为准，本 skill 不重复维护，
只给新会话一个最快的进入姿势。

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

- 要**写新代码**（组合分析、新工具、新模块）→ 先读 `yifu-develop`，
  并按 AGENTS.md 宗旨 4 确认代码归属（repo / 项目文件夹用完即删）。
- 要动 `pipeline_modules/` 里的程序 → 默认禁止；明确 bug 才最小修复+跑测试+报告。
- 涉及 ims/arivis 读写细节 → `.zcode/skills/yifu-develop/ims-arivis-integration.md`。

## 交付纪律

结果归一到项目单一结果文件夹（如 `_reports/`），附面向用户的 PDF 说明
（逐项解释图表指标口径），交付前视觉验收。原始 ims/zarr 中间产物按 PM"归档"清理。
