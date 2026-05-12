---
name: Pipeline skill 通过实际运行持续迭代
description: 用户希望每次pipeline脚本经过调整后，将最终版本回写到 .claude/skills/generate-pipeline.md 中
type: feedback
---

每次为用户生成 pipeline shell 脚本后，如果用户对脚本参数、模块选择或配置做了调整，将**最终使用的版本**回写到 `.claude/skills/generate-pipeline.md` 的 "Script generation rules" 或 "Template structure" 部分。

**How to apply:** 生成脚本 → 用户调整 → 确认最终版本 → 更新 skill 文件中的对应段落（参数默认值、命令行模板、场景配置）。不写中间试验版本，只保留最终确认的。
