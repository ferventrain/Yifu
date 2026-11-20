/**
 * yifu-pm — DeepSeek Harness plugin that exposes the Yifu Pipeline Monitor (PM)
 * as agent tools plus a read-only queue panel.
 *
 * The Python harness (`python -m pipeline_modules.harness`) stays the engine and
 * the owner of all state: run queue, the single detached worker, progress.json,
 * output verification and artifact manifests. This plugin is a thin adapter:
 *
 *   - it never writes job records and never starts the pipeline itself;
 *   - every read goes through `harness list`/`preflight` (read-only) or through
 *     the JSON files under YIFU_ACTIVE_DIR;
 *   - submissions and cancellations go through the documented CLI, so the same
 *     validation, dedupe and structured errors apply to humans and agents.
 *
 * No runtime imports of harness packages: a profile-installed bundle can only
 * import what it ships, so the tool definitions below are plain objects (the
 * registry accepts a raw ToolDefinition; `defineTool` is only a typing helper).
 */
import { execFile } from 'node:child_process'
import { existsSync } from 'node:fs'
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)

export const name = 'yifu-pm'
export const inject = ['tools']

const DEFAULTS = {
  repoRoot: 'H:\\yifu0428\\Yifu',
  pythonExe: 'C:\\Users\\admin\\micromamba\\envs\\yifu\\python.exe',
  analysisRoot: 'H:\\arivis-analysis',
  activeDir: '',
  cliTimeoutMs: 120000,
  jobsLimit: 20,
  logTailLines: 100,
  staleHeartbeatS: 120,
  legacyPmUrl: 'http://127.0.0.1:8766',
}

const OUTPUT_SCHEMA = { type: 'object', additionalProperties: true }
const ACTIVE_DISPLAY = new Set(['queued', 'running', 'verifying', 'stalled'])

// ----------------------------------------------------------------- helpers

function resolveConfig(raw) {
  const cfg = { ...DEFAULTS }
  if (raw && typeof raw === 'object') {
    for (const key of Object.keys(DEFAULTS)) {
      const value = raw[key]
      if (value !== undefined && value !== null && value !== '') cfg[key] = value
    }
  }
  cfg.repoRoot = path.resolve(String(cfg.repoRoot))
  cfg.analysisRoot = path.resolve(String(cfg.analysisRoot))
  cfg.activeDir = cfg.activeDir
    ? path.resolve(String(cfg.activeDir))
    : path.join(cfg.analysisRoot, '_active')
  cfg.cliTimeoutMs = Math.max(1000, Number(cfg.cliTimeoutMs) || DEFAULTS.cliTimeoutMs)
  cfg.jobsLimit = Math.max(1, Number(cfg.jobsLimit) || DEFAULTS.jobsLimit)
  cfg.logTailLines = Math.max(1, Number(cfg.logTailLines) || DEFAULTS.logTailLines)
  cfg.staleHeartbeatS = Math.max(10, Number(cfg.staleHeartbeatS) || DEFAULTS.staleHeartbeatS)
  cfg.legacyPmUrl = String(cfg.legacyPmUrl || DEFAULTS.legacyPmUrl).replace(/\/+$/, '')
  return cfg
}

function tailText(text, maxChars) {
  const value = String(text ?? '')
  return value.length <= maxChars ? value : value.slice(value.length - maxChars)
}

/** Last ``count`` log lines, with carriage-return progress overwrites split into
 * lines and the whole tail capped, so a tqdm-heavy log never floods a reader. */
function tailLines(text, count, maxChars = 8000) {
  const cleaned = String(text ?? '').replace(/\r/g, '\n')
  const lines = cleaned.split('\n')
  const tail = lines.slice(Math.max(0, lines.length - count)).join('\n').trimEnd()
  return tail.length > maxChars ? `…${tail.slice(tail.length - maxChars)}` : tail
}

function isoAgeSeconds(iso, nowMs = Date.now()) {
  if (!iso) return null
  const parsed = Date.parse(String(iso))
  if (!Number.isFinite(parsed)) return null
  return Math.max(0, Math.round((nowMs - parsed) / 100) / 10)
}

function numOrNull(value) {
  if (value === null || value === undefined || value === '') return null
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

/** Run one harness CLI command and parse its JSON stdout. */
async function runCli(cfg, args, exec) {
  const env = {
    ...process.env,
    YIFU_ANALYSIS_ROOT: cfg.analysisRoot,
    YIFU_ACTIVE_DIR: cfg.activeDir,
    PYTHONUTF8: '1',
  }
  const options = {
    cwd: cfg.repoRoot,
    env,
    timeout: cfg.cliTimeoutMs,
    maxBuffer: 32 * 1024 * 1024,
    windowsHide: true,
    encoding: 'utf8',
  }
  if (exec && exec.signal) options.signal = exec.signal
  try {
    const { stdout } = await execFileAsync(
      cfg.pythonExe,
      ['-m', 'pipeline_modules.harness', ...args],
      options,
    )
    return JSON.parse(stdout)
  } catch (err) {
    if (err && (err.name === 'AbortError' || err.code === 'ABORT_ERR')) throw err
    const stderr = String((err && err.stderr) || '')
    let structured = null
    try {
      structured = JSON.parse(stderr.trim()).error || null
    } catch {
      structured = null
    }
    const failure = new Error(
      (structured && structured.message) || `yifu harness ${args[0]} failed: ${err && err.message}`,
    )
    failure.code = (structured && structured.code) || 'YIFU_CLI_FAILED'
    failure.detail = {
      args,
      exit_code: typeof (err && err.code) === 'number' ? err.code : null,
      structured_error: structured,
      stderr_tail: tailText(stderr, 4000),
    }
    throw failure
  }
}

/**
 * Project of one job: the batch folder that owns the sample.
 *
 * Under YIFU_ACTIVE_DIR the first path segment is the project; outside it the
 * sample's parent directory is (the future "one project, one folder" layout).
 * Returns ``{ name, dir }`` so the project's own `project.json` can be read.
 */
function projectOf(cfg, view) {
  const sampleDir = typeof view.sample_dir === 'string' ? view.sample_dir : ''
  if (!sampleDir) return { name: null, dir: null }
  const resolved = path.resolve(sampleDir)
  const active = cfg.activeDir.toLowerCase()
  const lower = resolved.toLowerCase()
  if (lower === active) return { name: null, dir: null }
  if (lower.startsWith(active + path.sep)) {
    const rest = resolved.slice(cfg.activeDir.length + 1)
    const parts = rest.split(/[\\/]/)
    if (parts.length > 1) {
      return { name: parts[0], dir: path.join(cfg.activeDir, parts[0]) }
    }
    return { name: null, dir: null }
  }
  const dir = path.dirname(resolved)
  return { name: path.basename(dir), dir }
}

/**
 * Project state from `<project dir>/project.json`:
 * `{ "state": "active" | "paused" | "delivered", "display_name"?, "note"? }`.
 * Missing or unreadable file = active, so existing batches need no setup.
 */
async function projectState(cfg, projectDir) {
  if (!projectDir) return 'active'
  const file = path.join(projectDir, 'project.json')
  if (!existsSync(file)) return 'active'
  try {
    const payload = JSON.parse(await readFile(file, 'utf8'))
    const state = payload && typeof payload.state === 'string' ? payload.state.trim().toLowerCase() : ''
    return ['active', 'paused', 'delivered'].includes(state) ? state : 'active'
  } catch {
    return 'active'
  }
}

function percentOf(view) {
  const progress = (view && view.progress) || {}
  const done = Number(progress.unit_done)
  const total = Number(progress.unit_total)
  if (Number.isFinite(done) && Number.isFinite(total) && total > 0) {
    return Math.min(100, Math.round((done / total) * 1000) / 10)
  }
  const index = Number(progress.step_index)
  const stepTotal = Number(progress.step_total)
  if (Number.isFinite(index) && Number.isFinite(stepTotal) && stepTotal > 0 && index > 0) {
    return Math.min(99, Math.round((index / stepTotal) * 1000) / 10)
  }
  return null
}

function compactJob(cfg, view, project, projectStateValue) {
  const progress = (view && view.progress) || {}
  const eta = (view && view.eta) || {}
  return {
    run_id: view.id,
    kind: view.kind || 'main',
    title: view.title || null,
    sample: view.sample_name || (view.sample_dir ? path.basename(String(view.sample_dir)) : null),
    sample_dir: view.sample_dir || null,
    project: project,
    project_state: projectStateValue || 'active',
    config: view.config_path ? path.basename(String(view.config_path)) : null,
    status: view.status || null,
    display: view.display_status || view.status || null,
    step: progress.step_name || null,
    step_index: numOrNull(progress.step_index),
    step_total: numOrNull(progress.step_total),
    percent: percentOf(view),
    unit_done: numOrNull(progress.unit_done),
    unit_total: numOrNull(progress.unit_total),
    eta_total_s: numOrNull(eta.total_remaining_s),
    started_at: view.started_at || null,
    ended_at: view.ended_at || null,
    duration_s: numOrNull(view.duration_s),
    heartbeat_age_s: numOrNull(view.heartbeat_age_s),
    worker_alive: view.worker_alive === undefined ? null : Boolean(view.worker_alive),
    error: view.error || null,
  }
}

async function readWorker(cfg) {
  const file = path.join(cfg.activeDir, 'worker.json')
  if (!existsSync(file)) {
    return { alive: false, pid: null, current_job_id: null, heartbeat_age_s: null }
  }
  try {
    const state = JSON.parse(await readFile(file, 'utf8'))
    const heartbeatAge = isoAgeSeconds(state.heartbeat_at)
    const alive = !state.stopped_at && heartbeatAge !== null && heartbeatAge <= cfg.staleHeartbeatS
    return {
      alive,
      pid: state.pid ?? null,
      current_job_id: alive ? state.current_job_id ?? null : null,
      heartbeat_age_s: heartbeatAge,
      started_at: state.started_at ?? null,
      note: 'liveness is inferred from worker.json freshness; the harness remains the authority',
    }
  } catch {
    return { alive: false, pid: null, current_job_id: null, heartbeat_age_s: null }
  }
}

/** Queue rows for tools and panel. Read-only. */
async function collectJobs(cfg, options, exec) {
  const views = await runCli(cfg, ['list'], exec)
  const requestedLimit = numOrNull(options && options.limit) || cfg.jobsLimit
  const limit = Math.max(1, Math.min(500, requestedLimit))
  const includeFinished = !(options && options.include_finished === false)
  const target = options && options.sample_dir
    ? path.resolve(String(options.sample_dir)).toLowerCase()
    : null
  const wantedProject = options && options.project ? String(options.project) : null

  const stateCache = new Map()
  const rows = []
  for (const view of views) {
    const project = projectOf(cfg, view)
    if (wantedProject && project.name !== wantedProject) continue
    if (target && String(view.sample_dir || '').toLowerCase() !== target) continue
    const cacheKey = project.dir || '(root)'
    let state = stateCache.get(cacheKey)
    if (state === undefined) {
      state = await projectState(cfg, project.dir)
      stateCache.set(cacheKey, state)
    }
    const row = compactJob(cfg, view, project.name, state)
    if (!includeFinished && !ACTIVE_DISPLAY.has(String(row.display))) continue
    rows.push(row)
    if (rows.length >= limit) break
  }

  const counts = {}
  for (const row of rows) counts[row.display] = (counts[row.display] || 0) + 1
  const projects = {}
  for (const row of rows) {
    const key = row.project || '(root)'
    if (!projects[key]) projects[key] = { name: row.project, state: row.project_state, total: 0, running: 0, queued: 0, failed: 0 }
    const entry = projects[key]
    entry.total += 1
    if (entry.state !== row.project_state && row.project_state !== 'active') entry.state = row.project_state
    if (row.display === 'running') entry.running += 1
    if (row.display === 'queued') entry.queued += 1
    if (row.display === 'failed' || row.display === 'stalled') entry.failed += 1
  }
  return {
    ok: true,
    count: rows.length,
    counts,
    jobs: rows,
    projects: Object.values(projects),
    worker: await readWorker(cfg),
    legacy_pm_url: cfg.legacyPmUrl,
    generated_at: new Date().toISOString(),
  }
}

/** One run's detail: record, progress, artifacts, log tail. Read-only. */
async function jobDetail(cfg, runId, tailCount, exec) {
  const views = await runCli(cfg, ['list'], exec)
  const view = views.find((item) => String(item.id) === String(runId))
  if (!view) {
    const failure = new Error(`run_id 不存在: ${runId}`)
    failure.code = 'YIFU_JOB_NOT_FOUND'
    throw failure
  }
  const project = projectOf(cfg, view)
  const record = compactJob(cfg, view, project.name, await projectState(cfg, project.dir))

  let artifacts = null
  const manifestPath = path.join(cfg.activeDir, 'runs', String(view.id), 'artifacts.json')
  if (existsSync(manifestPath)) {
    try {
      artifacts = JSON.parse(await readFile(manifestPath, 'utf8'))
    } catch {
      artifacts = { error: 'artifacts.json 不可解析' }
    }
  }

  const logPath = view.log_path ? String(view.log_path) : null
  let logTail = null
  if (logPath && existsSync(logPath)) {
    try {
      logTail = tailLines(await readFile(logPath, 'utf8'), Math.max(1, tailCount || cfg.logTailLines))
    } catch {
      logTail = null
    }
  }
  return {
    ok: true,
    run_id: view.id,
    record,
    progress: view.progress || null,
    eta: view.eta || null,
    verification: artifacts ? artifacts.verification || null : null,
    artifacts: artifacts ? artifacts.artifacts || null : null,
    log_path: logPath,
    log_tail: logTail,
  }
}

/** Build one raw ToolDefinition.
 *
 * ``parameters`` is already a standard JSON Schema object
 * (``{ type: 'object', properties, required }``) and is passed to
 * ``ctx.tools.register`` untouched: the registry performs no conversion, so
 * anything else reaches the model API verbatim and is rejected there.
 */
function tool(definition) {
  const { run, ...rest } = definition
  return {
    ...rest,
    output: {
      schema: OUTPUT_SCHEMA,
      render(_args, value) {
        return [{ type: 'text', text: JSON.stringify(value, null, 2) }]
      },
    },
    async execute(args, exec) {
      // `this` is the registered definition, which carries the resolved config.
      return run.call(this, args || {}, exec)
    },
  }
}

const TOOLS = [
  tool({
    name: 'yifu_preflight',
    description:
      'Yifu 流程只读预检：检查样本目录、config.json、pipeline 模式、必需产物与当前 worker 状态，不创建任务。' +
      '在提交前用它确认输入和配置；返回 {status, pipeline, required_outputs, active_run, worker}。',
    parameters: {
      type: 'object',
      properties: {
        sample_dir: { type: 'string', description: '样本目录（必须在 analysis root 下）' },
        config_path: { type: 'string', description: 'run config；缺省 <sample_dir>/config.json' },
      },
      required: ['sample_dir'],
      additionalProperties: false,
    },
    async run(args, exec) {
      const cfg = this.config
      const cliArgs = ['preflight', '--sample-dir', String(args.sample_dir)]
      if (args.config_path) cliArgs.push('--config', String(args.config_path))
      const payload = await runCli(cfg, cliArgs, exec)
      return { ok: payload.status === 'ok', ...payload }
    },
  }),
  tool({
    name: 'yifu_submit',
    description:
      'Yifu 流程提交（唯一标准入口）：校验后入队并启动单串行 detached worker，立即返回 run_id。' +
      '同一样本已有 queued/running/verifying 任务时会去重并返回原 run_id（already_active=true），仍会确保 worker 在线。' +
      '只接受预声明附加旗标 --skip_registration / --only_tubule / --only_region_analysis。',
    parameters: {
      type: 'object',
      properties: {
        sample_dir: { type: 'string', description: '样本目录（必须在 analysis root 下）' },
        config_path: { type: 'string', description: 'run config；缺省 <sample_dir>/config.json' },
        extra_args: {
          type: 'array',
          items: { type: 'string', enum: ['--skip_registration', '--only_tubule', '--only_region_analysis'] },
          description: '额外主流程旗标（可空）',
        },
      },
      required: ['sample_dir'],
      additionalProperties: false,
    },
    async run(args, exec) {
      const cfg = this.config
      const cliArgs = ['run', '--sample-dir', String(args.sample_dir)]
      if (args.config_path) cliArgs.push('--config', String(args.config_path))
      const extras = Array.isArray(args.extra_args) ? args.extra_args : []
      for (const item of extras) cliArgs.push('--extra-arg', String(item))
      const payload = await runCli(cfg, cliArgs, exec)
      return {
        ok: true,
        run_id: payload.run_id,
        status: payload.status,
        already_active: Boolean(payload.already_active),
        sample_dir: payload.sample_dir,
        config_path: payload.config_path,
        log_path: payload.log_path,
        worker: payload.worker || null,
      }
    },
  }),
  tool({
    name: 'yifu_jobs',
    description:
      'Yifu 任务队列只读列表（按状态排序）：主流程任务与 wrap 挂载的外部任务，含进度、步骤、百分比、ETA、结构化错误。' +
      '可按样本目录、项目或是否含已结束任务过滤；默认返回最近 N 条。数据来自 harness list，不写任务记录。',
    parameters: {
      type: 'object',
      properties: {
        sample_dir: { type: 'string', description: '只看这个样本目录的任务' },
        project: { type: 'string', description: '只看这个项目（_active/<项目>）的任务' },
        limit: { type: 'number', description: '返回条数，缺省取插件配置 jobsLimit' },
        include_finished: { type: 'boolean', description: '是否包含已结束任务（默认 true）' },
      },
      required: [],
      additionalProperties: false,
    },
    async run(args, exec) {
      const cfg = this.config
      return collectJobs(cfg, args, exec)
    },
  }),
  tool({
    name: 'yifu_job',
    description:
      'Yifu 单个任务详情（只读）：任务记录、progress.json、artifact manifest 摘要与日志尾部。' +
      '失败时直接给结构化错误 code/message/step_id/retryable/suggestion，不要靠读日志猜结论。',
    parameters: {
      type: 'object',
      properties: {
        run_id: { type: 'string', description: '提交时返回的 run_id' },
        tail: { type: 'number', description: '日志尾行数，缺省取插件配置 logTailLines' },
      },
      required: ['run_id'],
      additionalProperties: false,
    },
    async run(args, exec) {
      const cfg = this.config
      return jobDetail(cfg, String(args.run_id), numOrNull(args.tail), exec)
    },
  }),
  tool({
    name: 'yifu_cancel',
    description:
      '取消 Yifu 任务（协作式）：queued 直接取消；running/verifying 写 cancel.flag 由 worker 终止整棵进程树。' +
      '返回 {run_id, status: cancelled|cancel_requested}。',
    parameters: {
      type: 'object',
      properties: {
        run_id: { type: 'string', description: '提交时返回的 run_id' },
      },
      required: ['run_id'],
      additionalProperties: false,
    },
    async run(args, exec) {
      const cfg = this.config
      const payload = await runCli(cfg, ['cancel', String(args.run_id)], exec)
      return { ok: true, run_id: payload.job_id, status: payload.status }
    },
  }),
]

// ------------------------------------------------------------ HTTP routes
// Read-only JSON for the panel half. The panel runs in the Web GUI page, so a
// same-origin fetch of /yifu-pm/* is enough; nothing here mutates state.

function sendJson(res, status, value) {
  const body = JSON.stringify(value)
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'no-store',
    'content-length': Buffer.byteLength(body),
  })
  res.end(body)
}

function registerRoutes(ctx, cfg) {
  const webServer = typeof ctx.get === 'function' ? ctx.get('webServer') : undefined
  if (!webServer || typeof webServer.register !== 'function') return null

  const handler = async (req, res) => {
    let url
    try {
      url = new URL(req.url || '/', 'http://127.0.0.1')
    } catch {
      sendJson(res, 400, { error: 'bad request url' })
      return
    }
    try {
      if (url.pathname === '/yifu-pm/jobs' || url.pathname === '/yifu-pm/jobs/') {
        const limit = numOrNull(url.searchParams.get('limit')) || cfg.jobsLimit
        const includeFinished = url.searchParams.get('finished') !== '0'
        const sampleDir = url.searchParams.get('sample') || undefined
        const project = url.searchParams.get('project') || undefined
        sendJson(res, 200, await collectJobs(cfg, { limit, include_finished: includeFinished, sample_dir: sampleDir, project }))
        return
      }
      if (url.pathname === '/yifu-pm/job' || url.pathname === '/yifu-pm/job/') {
        const runId = url.searchParams.get('run_id')
        if (!runId) {
          sendJson(res, 400, { error: 'run_id is required' })
          return
        }
        const tail = numOrNull(url.searchParams.get('tail')) || cfg.logTailLines
        sendJson(res, 200, await jobDetail(cfg, runId, tail))
        return
      }
      sendJson(res, 404, { error: 'not found' })
    } catch (err) {
      sendJson(res, 500, {
        error: (err && err.message) || String(err),
        code: (err && err.code) || null,
        detail: (err && err.detail) || null,
      })
    }
  }

  return webServer.register({ kind: 'prefix', path: '/yifu-pm', handler })
}

export function apply(ctx, config) {
  const cfg = resolveConfig(config)
  for (const definition of TOOLS) {
    ctx.tools.register({ ...definition, config: cfg })
  }
  if (typeof ctx.effect === 'function') {
    ctx.effect(() => {
      const dispose = registerRoutes(ctx, cfg)
      return () => {
        if (typeof dispose === 'function') dispose()
      }
    })
  } else {
    registerRoutes(ctx, cfg)
  }
}
