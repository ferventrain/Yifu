# Yifu PM → DeepSeek Harness 插件迁移方案

状态：Phase 0 完成，Phase 1 工具完成并已安装验证，Phase 2 面板待页面确认（2026-11-19）。

## 0. 已确认的决定

| 项 | 决定 |
|---|---|
| 范围 | 工具 + 极简面板都做 |
| 状态归属 | 暂沿用 `H:\arivis-analysis\_active`；后续一项目一文件夹（项目下多样本），支持“暂不排期/待回款”标记（操作者自行设置） |
| 顺序 | Phase 0 收敛 → Phase 1 插件 v1 → Phase 2 面板 → Phase 3 下线旧 PM |
| 旧 PM（8766） | 迁移期保留并行 |
| 插件源码 | 放仓库内 |

## 1. 架构定位

- **Python 引擎**（`pipeline_modules/harness`）= 执行者与状态所有者：文件队列、单串行 detached worker、结构化错误、输出验证判定成功、progress.json 心跳。
- **DSH 插件**（仓库内 `dsh-plugin/yifu-pm`）= 遥控器 + 仪表盘：把 preflight/submit/jobs/job/cancel 变成 agent 工具，把队列变成一块极简面板。

不把队列搬进 DSH，理由（DSH 0.2.2-rc.5 实测文档）：

1. `ctx.jobs` / `ctx.subprocess` 是进程内托管，插件 dispose 会终止子进程，承载不了小时级 detached run；
2. 第三方 bundle 不能新增 Remote namespace，面板数据只能走 Host `webServer` 路由或会话投影；
3. `verify.py` 的“四条腿”成功判定与结构化错误是这套系统最值钱的部分，重写等于重做测试。

## 2. 不可破的不变量

1. 同一时刻最多一个 `kind=main` 的运行；状态只由 worker 推进。
2. 成功 = 退出码 0 + 必需产物存在 + 可读 + artifact manifest 写入；永不靠日志文本判成功。
3. `_active` 下的 JSON（jobs/runs/queue/worker/timing_history）是唯一真相；DSH 侧不另存第二份状态。
4. 读路径不写盘（GET/查询工具不改任务记录）；写只发生在 submit/cancel/worker 推进。
5. 失败必须是结构化错误：`code / message / step_id / retryable / suggestion`。

## 2.5 进度（2026-11-19）

| 阶段 | 状态 | 证据 |
|---|---|---|
| Phase 0 收敛 | 完成 | harness 测试 67 passed / 1 个既有时间窗失败（analytics，与本次无关）；`harness list` 读真实根：57 任务、历史 `done` 归一为 `succeeded`、被读任务文件 hash 前后一致；`list`/`run` 输出重定向不再因控制台编码崩溃 |
| Phase 1 工具 | 完成并安装 | 插件行 `include:yifu-pm` active；独立子会话调用 `yifu_jobs` / `yifu_job` / `yifu_preflight` 成功；`GET /yifu-pm/jobs` 返回真实队列（2026-11-19 现场：SHAM_1 running 66.7%） |
| Phase 2 面板 | 完成（操作者已确认可见） | 左侧“Yifu 队列”图标 + 主面板：项目分组、状态灯、步骤/百分比/耗时、结构化错误 + 日志尾、默认只看未结束（可切换含已结束）；数据走 Host 只读路由 `/yifu-pm/*` |
| 项目模型 | 已实现读侧 | `<项目>/project.json` 的 `state: active|paused|delivered` 决定分组变暗/标签；缺省 active，操作者自己维护 |
| 文档收敛 | 完成 | AGENTS/README/capabilities.json/.zcode/.cursor/.claude 统一到 `run` 入口、`runs/<id>/progress.json`、"无开发/测试 tab（= agent 对话）"、新产物命名与 9 模块清单；过时说法已清零 |
| Phase 3 下线旧 PM | 未开始 | 旧 PM 8766 保留并行 |

实现要点：插件是**零依赖 Host 包**（不 import dsh 内部包，避免 profile 安装后解析不到）；
`ctx.tools.register()` 的 `parameters` 必须是标准 JSON Schema（扁平的 defineTool 语法会被服务端拒收，
报 `tools.NN.function.parameters must be a JSON Schema of type object`，整个会话模型请求会失败）。

## 3. Phase 0 收敛（先做，做不完不封装）

| # | 项 | 改什么 | 验收 |
|---|---|---|---|
| 1 | 入口单一 | 对外只保留 `run` / `preflight` / `cancel` / `list` / `wrap`；`enqueue`/`attach` 降级为内部兼容（帮助文本标 LEGACY）；`run` 去重分支也确保 worker 在线；PM 的 `POST /api/jobs` 与 CLI 一样按样本去重 | 新增/修改测试覆盖“去重且 worker 被拉起” |
| 2 | 状态词表 | 存储状态只允许 `queued/running/verifying/succeeded/failed/cancelled/stalled`；读入时把历史 `done` 归一为 `succeeded`（不批量改历史文件）；显示状态 `stalled` 只在读取端算一次 | 测试覆盖 done→succeeded 归一；面板/CLI 输出不再出现 done |
| 3 | 进度协议 | schema1 `progress.json` 为唯一契约；`pm_progress.json`(schema2)、wrap 的 `module_progress.json`、两行 status.txt 标注为兼容输入；百分比/步骤状态由读取端统一计算（不再只在 JS 里） | 契约写进 capability_manifest；读取端函数有测试 |
| 4 | 只读承诺 | GET / 查询不再 `save_job`、不再 `ensure_layout`；external 同步由 worker tick / wrap 收尾负责 | 测试：读一次列表后任务记录文件 mtime/hash 不变 |
| 5 | 死代码 | 删 `RunContext`（零调用）；`RunSpec.command()` 成为 worker 唯一拼命令处；修 `worker.py` 未定义的 `logger`；清渲染端 done 别名 | 全量 harness 测试绿 |
| 6 | 常量单一 | 步骤表只留 `progress.planned_step_names`；删 `projects._CANONICAL_STEP_NAMES`、`VESSEL_STEP_TOTAL=7` | 两处引用改为同一来源；测试覆盖 6/3 步 |
| 7 | 文档单一真相 | `capability_manifest.json` 当唯一契约；修 `capabilities.json` 的 enqueue/MANDATORY、死引用；README §4 重写成单入口；AGENTS 修正默认 root 与入口口径；`.claude`/`.cursor` 旧说明降级为指针 | 文档里不再出现互相矛盾的入口/端口/路径 |

## 4. Phase 1 插件 v1

包布局（仓库内）：

```
dsh-plugin/yifu-pm/
  package.json          # name: @local/yifu-pm, dsh.bundle.patch, dsh.client
  cordis.patch.yml      # insert 一行
  index.js              # Host: service + tools + config
  client.js             # Phase 2
  locale/en.json zh.json
  docs/DETAIL.md  docs/preview.svg
  icon.svg
```

工具（Host，调 `python -m pipeline_modules.harness ...` + 读 `_active` JSON）：

| 工具 | 参数 | 返回（精简 JSON） |
|---|---|---|
| `yifu_preflight` | `sample_dir`, `config_path?`, `extra_args?` | `{ok, pipeline, required_outputs, active_run, worker}` |
| `yifu_submit` | `sample_dir`, `config_path?`, `extra_args?` | `{run_id, status, already_active, log_path}` |
| `yifu_jobs` | `limit?`, `sample_dir?`, `project?` | `[{run_id, sample, project, status, display, step, percent?, eta_s?, started_at, error?}]` |
| `yifu_job` | `run_id`, `tail?` | `{record, progress, artifacts, log_tail}` |
| `yifu_cancel` | `run_id` | `{run_id, status}` |

配置（`Config` schema，插件设置页可改）：

| 键 | 默认 | 说明 |
|---|---|---|
| `repoRoot` | 仓库路径 | 调 CLI 的 cwd |
| `pythonExe` | `micromamba` 方案 | 形式如 `["micromamba","run","-n","yifu","python"]`，避免系统 python 拉起无依赖 worker |
| `analysisRoot` | `H:\arivis-analysis` | 读 `_active` 的根 |
| `staleHeartbeatS` | 120 | 超时判定 stalled |
| `jobsLimit` | 20 | 列表默认返回条数 |
| `logTailLines` | 100 | 日志尾默认行数 |

## 5. Phase 2 极简面板

- 位置：`main` 面板（`sidebar.panellist` 图标 + `main` keyed 注册，参考 `client-ui-plugin-manager`）。
- 数据：Host `ctx.webServer.register({kind:'prefix', path:'/yifu-pm/'})` 提供只读 JSON；客户端 2 s 轮询。
- 内容：队列列表（项目 → 样本 → 状态灯 / 当前步骤 / 百分比 / 已耗时 / 结构化错误 / 日志尾），一个“提交”入口调 `yifu_submit` 同一服务。无 ECharts、无看板分析。
- 项目模型：项目 = `_active/<项目>/` 或未来独立项目根；可选 `<项目>/project.json`：
  `{ "display_name": "...", "state": "active|paused|delivered" }`，`paused/delivered` 不排期、列表折叠。
  未提供该文件时一律视为 `active`。

## 6. Phase 3 下线旧 PM

插件稳定后停用 `apps/pipeline_harness`（保留代码与只读路由一个版本周期），文档只留一份；`harness` CLI 永久保留（插件也依赖它）。

## 7. 风险与已知坑

- 现场 `_active` 有运行中的任务：Phase 0 的验证一律用临时 analysis root，不改真实记录。
- DSH 版本 0.2.2-rc.5；第三方 client bundle 需手写 lazy-CJS（模板可循）；不能加 Remote namespace。
- Windows：插件调 Python 必须走 micromamba yifu 解释器；DSH 沙箱会拦工作区 ACL（本会话已遇到并修复过一次）。
- 历史记录里混有 `done` 状态（现场 57 条里 25 条）：只做读取归一，不改历史文件。
