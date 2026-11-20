# yifu-pm（仓库内 DSH 插件）

把 Yifu 流程的提交/监视接进 DeepSeek Harness：5 个工具，Python harness 仍是执行引擎与状态所有者。

## 安装

DSH → **插件** → **添加插件** → 选本目录；或让 agent 用 `plugin_manager install_bundle`
指向本目录。装完重启 DSH（替换已装包需重启才加载新代码）。

## 配置

改 `cordis.patch.yml` 里 `yifu-pm` 行的 `config` 后重启：

- `repoRoot`：仓库路径（CLI 的 cwd）
- `pythonExe`：必须是 micromamba `yifu` 环境的 python，否则拉起的 worker 会 `ModuleNotFoundError`
- `analysisRoot`：分析根，队列在其 `_active`
- `activeDir`：可选，显式覆盖 `YIFU_ACTIVE_DIR`
- `cliTimeoutMs` / `jobsLimit` / `logTailLines` / `staleHeartbeatS`

## 工具

| 工具 | 读写 | 用途 |
|---|---|---|
| `yifu_preflight` | 只读 | 提交前校验样本/config/产物/worker |
| `yifu_submit` | 写 | 提交样本，去重，返回 run_id |
| `yifu_jobs` | 只读 | 队列列表：状态/步骤/百分比/ETA/结构化错误 |
| `yifu_job` | 只读 | 单任务详情 + artifact + 日志尾 |
| `yifu_cancel` | 写 | 协作式取消 |

## 项目模型（一项目一文件夹）

项目 = 样本目录的父目录：现在的 `_active/<项目>/<样本>`，或以后独立的 `<项目>/<样本>`。
项目文件夹里可选放一个 `project.json`（操作者自己维护，插件只读不写）：

```json
{
  "display_name": "SQ_25_2610_0413 血管批",
  "state": "paused",
  "note": "等客户回款，暂不排期"
}
```

- `state`: `active`（缺省）/ `paused`（暂不排期）/ `delivered`（已交付）
- `paused` / `delivered` 的项目在面板里整组变暗并带标签，不进“运行/排队/失败”的紧迫感排序
- 每个任务行都带 `project_state`，工具里可直接按项目过滤

## 与旧 PM 的关系

旧 PM（`http://127.0.0.1:8766`）迁移期保留；两者读写同一份 `_active` 状态，可并行使用。
队列、worker、输出验证都以 Python harness 为准；插件不复制状态、不自行判成功。

## 插件注册注意（踩过的坑）

`ctx.tools.register()` **不做任何 schema 转换**，`parameters` 会原样发给模型 API，必须是标准 JSON Schema：

```js
parameters: {
  type: 'object',
  properties: {
    sample_dir: { type: 'string', description: '...' },
  },
  required: ['sample_dir'],        // required 只在顶层数组里
  additionalProperties: false,
}
```

不要写 dsh 内部 `defineTool()` 的扁平属性表（`{ 参数名: { type, required } }`，required 在属性里）：
手写注册用那套会被服务端拒收，报
`tools.NN.function.parameters must be a JSON Schema of type object`，整个会话的模型请求直接失败。
`output.schema` 同样是原始 JSON Schema（本插件用 `{ type: 'object', additionalProperties: true }`）。

改完 `index.js` 后要禁用/启用一次插件行（或重启 DSH）才会加载新代码；
`package.json` 的 `dsh.client` 改动建议 `remove_bundle` + `install_bundle` 重新读取清单，再刷新页面。
