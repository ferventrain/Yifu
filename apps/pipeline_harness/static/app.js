const POLL_MS = 1500;
const STATUS_LABELS = {
  idle: "未运行",
  queued: "等待",
  running: "运行中",
  verifying: "验证输出",
  succeeded: "完成",
  done: "完成",
  failed: "失败",
  cancelled: "已取消",
  stalled: "失联",
};

const els = {
  paths: document.getElementById("header-paths"),
  benchStats: document.getElementById("bench-stats"),
  runner: document.getElementById("runner-status"),
  resources: document.getElementById("resource-panel"),
  form: document.getElementById("add-form"),
  sampleDir: document.getElementById("sample-dir"),
  configPath: document.getElementById("config-path"),
  formError: document.getElementById("form-error"),
  board: document.getElementById("job-list"),
  logDialog: document.getElementById("log-dialog"),
  logTitle: document.getElementById("log-title"),
  logBody: document.getElementById("log-body"),
  logClose: document.getElementById("log-close"),
  stageDialog: document.getElementById("stage-dialog"),
  stageBody: document.getElementById("stage-body"),
  stageClose: document.getElementById("stage-close"),
  stageBtn: document.getElementById("stage-times-btn"),
};

let openLogJobId = null;
const expandedBatches = new Set();
const runOpenState = new Map(); // "<scope>::<id>" -> pinned open/closed
const nodeCache = new Map(); // sample key -> live card element (focus protection)
let currentSampleOrder = []; // flat display order of sample_dirs (for drag priority)
let dragSampleKey = null;
let errorHoldUntil = 0;

function sampleKey(sampleDir) {
  return String(sampleDir || "").replace(/[\\/]+$/, "").toLowerCase();
}

function showError(message, holdMs = 0, kind = "error") {
  els.formError.textContent = message || "";
  els.formError.classList.toggle("hidden", !message);
  els.formError.classList.toggle("info-banner", Boolean(message) && kind === "info");
  errorHoldUntil = message && holdMs ? Date.now() + holdMs : 0;
}

function formatDuration(seconds) {
  if (seconds == null || Number.isNaN(Number(seconds))) return null;
  const total = Math.max(0, Math.round(Number(seconds)));
  if (total < 60) return `${total} 秒`;
  const minutes = Math.floor(total / 60);
  if (minutes < 60) return `${minutes} 分`;
  const hours = Math.floor(minutes / 60);
  return `${hours} 时 ${minutes % 60} 分`;
}

function formatEta(seconds) {
  if (seconds == null || Number.isNaN(Number(seconds))) return null;
  const total = Math.max(0, Math.round(Number(seconds)));
  if (total < 90) return `约 ${total} 秒`;
  if (total < 5400) return `约 ${(total / 60).toFixed(total < 600 ? 1 : 0)} 分`;
  return `约 ${(total / 3600).toFixed(1)} 时`;
}

// Honest progress: completed steps + real unit fraction only.
function progressPercent(view) {
  const status = view.display_status || view.status;
  if (status === "succeeded" || status === "done") return 100;
  const progress = view.progress || {};
  const unitTotal = Number(progress.unit_total || 0);
  const unitDone = Number(progress.unit_done || 0);
  const total = Number(progress.step_total || 0);
  const index = Number(progress.step_index || 0);
  if (total <= 0 || index <= 0) return 0;
  const base = ((index - 1) / total) * 100;
  const stepSpan = 100 / total;
  if (unitTotal > 0) {
    const frac = Math.max(0, Math.min(1, unitDone / unitTotal));
    return Math.min(99, Math.round(base + stepSpan * frac));
  }
  return Math.min(99, Math.round(base));
}

function apiError(payload) {
  if (payload && payload.error && payload.error.message) return payload.error.message;
  if (typeof payload?.detail === "string") return payload.detail;
  if (Array.isArray(payload?.detail)) {
    return payload.detail.map((item) => item.msg || item.message || JSON.stringify(item)).join("; ");
  }
  return "请求失败";
}

async function api(path, options) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(apiError(payload) || `HTTP ${response.status}`);
  return payload;
}

// ------------------------------------------------------------------ board

function renderBenchStats(running, queued, samples, failed) {
  if (!els.benchStats) return;
  els.benchStats.replaceChildren();
  for (const [value, label, cls] of [
    [running, "运行", "is-run"],
    [queued, "排队", ""],
    [samples, "样本", ""],
    [failed, "失败", failed ? "is-bad" : ""],
  ]) {
    const stat = document.createElement("span");
    stat.className = `bench-stat ${cls}`.trim();
    const num = document.createElement("strong");
    num.textContent = value;
    const lab = document.createElement("span");
    lab.textContent = label;
    stat.append(num, lab);
    els.benchStats.appendChild(stat);
  }
}

function renderBoard(data) {
  const batches = data.batches || [];
  const runs = [];
  for (const batch of batches) for (const project of batch.projects) runs.push(...project.runs);

  els.paths.textContent = `${data.analysis_root}  /  ${data.active_dir}`;
  const worker = data.worker || {};
  const running = runs.filter((run) => (run.display_status || run.status) === "running");
  const queued = runs.filter((run) => run.status === "queued");
  const sampleCount = batches.reduce((acc, b) => acc + b.projects.length, 0);
  const failedCount = batches.reduce(
    (acc, b) => acc + b.projects.filter((p) => ["failed", "stalled"].includes(p.status)).length,
    0
  );
  renderBenchStats(running.length, queued.length, sampleCount, failedCount);
  if (worker.alive) {
    els.runner.textContent = `Worker pid ${worker.pid} 在线` + (running.length ? `，运行中 ${running.length}` : "") + (queued.length ? `，排队 ${queued.length}` : "");
    els.runner.classList.toggle("busy", running.length > 0);
  } else {
    els.runner.textContent = running.length ? "有任务标记运行中但 Worker 不在线（失联）" : "Worker 离线（下次提交任务时自动启动）";
    els.runner.classList.remove("busy");
  }

  // prune stale cache entries
  const liveKeys = new Set();
  for (const batch of batches) for (const project of batch.projects) liveKeys.add(sampleKey(project.sample_dir));
  for (const key of nodeCache.keys()) if (!liveKeys.has(key)) nodeCache.delete(key);

  const activeBatches = batches.filter((b) => b.under_active);
  const otherBatches = batches.filter((b) => !b.under_active);

  els.board.replaceChildren();
  if (!batches.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "还没有项目。提交任务或用 wrap 挂载程序后，样本目录会自动出现在这里。";
    els.board.appendChild(empty);
    return;
  }
  const order = [];
  for (const batch of activeBatches) {
    els.board.appendChild(renderBatch(batch));
    for (const project of batch.projects) order.push(project.sample_dir);
  }
  if (otherBatches.length) {
    els.board.appendChild(renderOtherSection(otherBatches, order));
  }
  currentSampleOrder = order;
  renderResourcePanel(data.resources || null, batches);
}

async function setPriority(sampleDir, level) {
  try {
    await api("/api/projects/priority", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sample_dir: sampleDir, level }),
    });
  } catch (error) {
    showError(error.message, 6000);
  }
  await refresh();
}

function renderResourcePanel(resources, batches) {
  if (!els.resources) return;
  els.resources.replaceChildren();
  if (!resources) return;

  const title = document.createElement("h3");
  title.textContent = "资源与决策";
  els.resources.appendChild(title);

  // capacity bars
  for (const [label, used, total, unit] of [
    ["CPU", resources.used.cpu, resources.capacity.cpu, "核"],
    ["GPU", resources.used.gpu, resources.capacity.gpu, "卡"],
  ]) {
    const row = document.createElement("div");
    row.className = "res-row";
    const head = document.createElement("div");
    head.className = "res-head";
    const name = document.createElement("span");
    name.textContent = label;
    const num = document.createElement("strong");
    num.textContent = `${used}${total ? ` / ${total}` : ""}${unit ? ` ${unit}` : ""}`;
    head.append(name, num);
    const track = document.createElement("div");
    track.className = "res-track";
    const bar = document.createElement("div");
    bar.className = `res-bar${total && used / total > 0.85 ? " hot" : ""}`;
    bar.style.width = total ? `${Math.min(100, (used / total) * 100)}%` : "0%";
    track.appendChild(bar);
    row.append(head, track);
    els.resources.appendChild(row);
  }

  // active tasks draw
  if ((resources.active_tasks || []).length) {
    const head = document.createElement("h4");
    head.textContent = "运行中占用";
    els.resources.appendChild(head);
    for (const task of resources.active_tasks) {
      const line = document.createElement("div");
      line.className = "res-task";
      const name = document.createElement("span");
      name.className = "res-task-name";
      name.textContent = task.sample;
      const draw = document.createElement("span");
      draw.className = "res-task-draw";
      const parts = [`CPU ${task.cpu}`];
      if (task.gpu) parts.push(`GPU ${task.gpu}`);
      const rem = formatEta(task.remaining_s);
      if (rem) parts.push(`剩 ${rem}`);
      draw.textContent = parts.join("，");
      line.append(name, draw);
      els.resources.appendChild(line);
    }
  } else {
    const idle = document.createElement("p");
    idle.className = "res-note";
    idle.textContent = "当前没有运行中的任务";
    els.resources.appendChild(idle);
  }

  // reference costs + verdict
  const head2 = document.createElement("h4");
  head2.textContent = "可否加跑一个";
  els.resources.appendChild(head2);
  for (const cost of resources.profile_costs || []) {
    const line = document.createElement("div");
    line.className = "res-verdict";
    const name = document.createElement("span");
    name.textContent = cost.label;
    const draw = document.createElement("span");
    draw.textContent = `CPU ${cost.cpu}${cost.gpu ? ` + GPU ${cost.gpu}` : ""}`;
    const verdict = document.createElement("span");
    const cpuOk = (resources.free.cpu ?? 0) >= cost.cpu;
    const gpuOk = (resources.free.gpu ?? 0) >= cost.gpu;
    verdict.className = cpuOk && gpuOk ? "ok" : "bad";
    verdict.textContent = cpuOk && gpuOk ? "可跑" : gpuOk ? "CPU 不足" : "GPU 占用";
    line.append(name, draw, verdict);
    els.resources.appendChild(line);
  }
}

function batchHeader(batch) {
  const heading = document.createElement("summary");
  const name = document.createElement("strong");
  name.className = "batch-name";
  name.textContent = batch.batch;
  const meta = document.createElement("span");
  meta.className = "batch-meta";
  const active = batch.projects.filter((p) => ["running", "verifying", "queued"].includes(p.status)).length;
  const failed = batch.projects.reduce((acc, p) => acc + (p.failed_count || 0), 0);
  const anomaly = batch.projects.filter((p) => p.anomaly).length;
  meta.textContent = `${batch.projects.length} 样本` + (active ? `，${active} 进行中` : "") + (failed ? `，${failed} 失败` : "") + (anomaly ? `，${anomaly} 异常` : "");
  heading.append(name, meta);
  return heading;
}

function renderBatch(batch) {
  const box = document.createElement("details");
  box.className = "batch";
  const hasActive = batch.projects.some((p) => ["running", "verifying", "queued", "stalled"].includes(p.status));
  box.open = expandedBatches.has(batch.batch) ? true : expandedBatches.size === 0 ? hasActive : expandedBatches.has(batch.batch);
  if (box.open) expandedBatches.add(batch.batch);
  box.addEventListener("toggle", () => {
    if (box.open) expandedBatches.add(batch.batch);
    else expandedBatches.delete(batch.batch);
  });
  box.appendChild(batchHeader(batch));
  const list = document.createElement("div");
  list.className = "batch-projects";
  for (const project of batch.projects) list.appendChild(renderProject(project));
  box.appendChild(list);
  return box;
}

function renderOtherSection(batches, orderSink) {
  const box = document.createElement("details");
  box.className = "batch other-root";
  box.open = expandedBatches.has("__other_root__");
  box.addEventListener("toggle", () => {
    if (box.open) expandedBatches.add("__other_root__");
    else expandedBatches.delete("__other_root__");
  });
  const heading = document.createElement("summary");
  const name = document.createElement("strong");
  name.className = "batch-name";
  name.textContent = "其他目录";
  const meta = document.createElement("span");
  meta.className = "batch-meta";
  meta.textContent = `${batches.length} 个批次（分析根目录下、非 _active）`;
  heading.append(name, meta);
  box.appendChild(heading);
  const wrap = document.createElement("div");
  wrap.className = "batch-projects";
  for (const batch of batches) {
    const inner = renderBatch(batch);
    inner.classList.add("nested");
    wrap.appendChild(inner);
    for (const project of batch.projects) orderSink.push(project.sample_dir);
  }
  box.appendChild(wrap);
  return box;
}

// ------------------------------------------------------------------ project

function renderProject(project) {
  const key = sampleKey(project.sample_dir);
  const cached = nodeCache.get(key);
  // Never swap a card the user is interacting with (typing notes, open
  // select...) — that is what made the notes box collapse every 1.5s.
  if (cached && cached.isConnected && cached.contains(document.activeElement)) {
    return cached;
  }

  const card = document.createElement("article");
  const pcls = project.priority != null ? ` has-p${project.priority}` : "";
  card.className = `project-card status-${project.status}${project.anomaly ? " anomalous" : ""}${pcls}`;
  card.dataset.sampleKey = key;

  // drag & drop (priority)
  const handle = document.createElement("span");
  handle.className = "drag-handle";
  handle.title = "拖动调整优先级";
  handle.textContent = "⠿";
  handle.draggable = true;
  handle.addEventListener("dragstart", (event) => {
    dragSampleKey = key;
    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData("text/plain", key);
  });
  handle.addEventListener("dragend", () => {
    dragSampleKey = null;
    document.querySelectorAll(".drag-over").forEach((el) => el.classList.remove("drag-over"));
    refresh();
  });
  card.appendChild(handle);

  // ---- header
  const head = document.createElement("div");
  head.className = "job-head";
  const titleWrap = document.createElement("div");
  const title = document.createElement("strong");
  title.className = "job-title";
  title.textContent = project.sample_name;
  const badge = document.createElement("span");
  badge.className = "job-id";
  const parts = [];
  if (project.runs.length) parts.push(`${project.runs.length} 次 run`);
  if (project.failed_count) parts.push(`失败 ${project.failed_count}`);
  badge.textContent = parts.join(" · ");
  titleWrap.append(title, document.createElement("br"), badge);

  const statusCluster = document.createElement("span");
  statusCluster.className = "status-cluster";
  const pBadge = document.createElement("button");
  pBadge.type = "button";
  pBadge.className = `pbadge${project.priority != null ? ` p${project.priority}` : ""}`;
  pBadge.title = "优先级：点击循环 P0–P3";
  pBadge.textContent = project.priority != null ? `P${project.priority}` : "—";
  pBadge.addEventListener("click", () => {
    const next = project.priority == null ? 0 : (project.priority + 1) % 4;
    setPriority(project.sample_dir, next);
  });
  const lamp = document.createElement("i");
  lamp.className = `lamp ${project.status}`;
  const label = document.createElement("span");
  label.textContent = STATUS_LABELS[project.status] || project.status;
  statusCluster.append(pBadge, lamp, label);
  head.append(titleWrap, statusCluster);
  card.appendChild(head);

  // ---- chips: anomaly / profile / estimate
  const chips = document.createElement("div");
  chips.className = "chip-row";
  if (project.anomaly) {
    const chip = document.createElement("span");
    chip.className = "chip danger";
    chip.textContent = `异常：${project.anomaly}`;
    chips.appendChild(chip);
  }
  if (project.profile_label) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = project.profile_label;
    chips.appendChild(chip);
  }
  if (project.estimate && project.estimate.seconds) {
    const chip = document.createElement("span");
    chip.className = "chip estimate";
    chip.textContent = `预计 ${formatEta(project.estimate.seconds)}（${project.estimate.basis}）`;
    chips.appendChild(chip);
  }
  if (chips.childNodes.length) card.appendChild(chips);

  // ---- meta: sample dir + manifest summary
  const meta = document.createElement("p");
  meta.className = "card-meta";
  meta.textContent = project.sample_dir;
  card.appendChild(meta);
  const manifest = project.last_manifest;
  if (manifest && manifest.module) {
    const manLine = document.createElement("p");
    manLine.className = "card-meta manifest";
    const commit = manifest.git_commit ? ` @ ${manifest.git_commit}` : "";
    const dur = manifest.duration_seconds != null ? ` · ${formatDuration(manifest.duration_seconds) || "?"}` : "";
    const incomplete = manifest.status === "incomplete" ? " · ⚠️ 未跑完" : "";
    manLine.textContent = `最近产物：${manifest.module}${commit}${dur} · 输出 ${manifest.outputs.length} 项${incomplete}`;
    card.appendChild(manLine);
  }

  // ---- start row (one-click start / resume)
  const startRow = document.createElement("div");
  startRow.className = "start-row";
  const select = document.createElement("select");
  if (!project.configs.length) {
    const opt = document.createElement("option");
    opt.textContent = "(无 config*.json)";
    opt.value = "";
    select.appendChild(opt);
  } else {
    for (const name of project.configs) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      select.appendChild(opt);
    }
  }
  const startBtn = document.createElement("button");
  startBtn.type = "button";
  startBtn.className = "btn primary";
  const busy = Boolean(project.active_run_id);
  startBtn.textContent = busy ? "运行中（点击重开）" : "开始 / 继续";
  startBtn.disabled = !project.configs.length;
  startBtn.addEventListener("click", async () => {
    startBtn.disabled = true;
    startBtn.textContent = "提交中…";
    try {
      await api("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sample_dir: project.sample_dir,
          config_path: `${project.sample_dir.replace(/[\\/]+$/, "")}/${select.value}`,
        }),
      });
      showError(`已提交：${project.sample_name} / ${select.value}`, 4000, "info");
    } catch (error) {
      showError(error.message, 6000);
    }
    await refresh();
  });
  const openBtn = document.createElement("button");
  openBtn.type = "button";
  openBtn.className = "btn";
  openBtn.textContent = "打开目录";
  openBtn.addEventListener("click", () => openPath(project.sample_dir, openBtn));
  startRow.append(select, startBtn, openBtn);
  card.appendChild(startRow);

  // ---- notes (requirements / retro), state survives polling
  const notes = project.notes || {};
  const notesKey = `notes::${key}`;
  const notesBox = document.createElement("details");
  notesBox.className = "notes";
  notesBox.open = runOpenState.get(notesKey) === true;
  notesBox.addEventListener("toggle", () => runOpenState.set(notesKey, notesBox.open));
  const notesSummary = document.createElement("summary");
  const reqPreview = (notes.requirements || "").trim().split("\n")[0] || "点击填写分析需求";
  notesSummary.textContent = `需求：${reqPreview.slice(0, 80)}`;
  notesBox.appendChild(notesSummary);

  const reqArea = document.createElement("textarea");
  reqArea.rows = 2;
  reqArea.placeholder = "分析需求（这个项目要做什么、对照组、交付物…）";
  reqArea.value = notes.requirements || "";
  const retroArea = document.createElement("textarea");
  retroArea.rows = 3;
  retroArea.placeholder = "复盘（踩过的坑、失误、下次注意…）";
  retroArea.value = notes.retro || "";
  const saveBtn = document.createElement("button");
  saveBtn.type = "button";
  saveBtn.className = "btn";
  saveBtn.textContent = "保存笔记";
  saveBtn.addEventListener("click", async () => {
    saveBtn.disabled = true;
    saveBtn.textContent = "保存中…";
    try {
      await api("/api/projects/notes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sample_dir: project.sample_dir,
          requirements: reqArea.value,
          retro: retroArea.value,
        }),
      });
      showError("笔记已保存", 3000, "info");
    } catch (error) {
      showError(error.message, 6000);
    }
    saveBtn.disabled = false;
    saveBtn.textContent = "保存笔记";
  });
  const noteLog = document.createElement("div");
  noteLog.className = "note-log";
  for (const entry of notes.log || []) {
    const line = document.createElement("p");
    line.textContent = `${(entry.at || "").replace("T", " ").slice(0, 16)}  ${entry.text}`;
    noteLog.appendChild(line);
  }
  const quickRow = document.createElement("div");
  quickRow.className = "quick-note";
  const quickInput = document.createElement("input");
  quickInput.type = "text";
  quickInput.placeholder = "记一笔（时间戳笔记），回车提交";
  const quickBtn = document.createElement("button");
  quickBtn.type = "button";
  quickBtn.className = "btn";
  quickBtn.textContent = "记一笔";
  const submitQuick = async () => {
    const text = quickInput.value.trim();
    if (!text) return;
    try {
      await api("/api/projects/notes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample_dir: project.sample_dir, append_note: text }),
      });
      quickInput.value = "";
      nodeCache.delete(key); // allow re-render with the new log line
      await refresh();
    } catch (error) {
      showError(error.message, 6000);
    }
  };
  quickBtn.addEventListener("click", submitQuick);
  quickInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") submitQuick();
  });
  quickRow.append(quickInput, quickBtn);
  notesBox.append(reqArea, retroArea, saveBtn, quickRow, noteLog);
  card.appendChild(notesBox);

  // ---- error history (auto)
  if (project.error_history && project.error_history.length) {
    const errBox = document.createElement("details");
    errBox.className = "results error-history";
    const errSummary = document.createElement("summary");
    errSummary.textContent = `报错历史（${project.error_history.length}）`;
    errBox.appendChild(errSummary);
    for (const item of project.error_history) {
      const line = document.createElement("p");
      line.textContent = `${(item.ended_at || item.run_id || "").toString().replace("T", " ").slice(0, 16)}  ${item.error || ""}`;
      errBox.appendChild(line);
    }
    card.appendChild(errBox);
  }

  // ---- runs list
  const runsBox = document.createElement("div");
  runsBox.className = "runs";
  for (const run of project.runs) runsBox.appendChild(renderRunRow(project, run));
  card.appendChild(runsBox);

  // ---- drag & drop targets
  card.addEventListener("dragover", (event) => {
    if (dragSampleKey && dragSampleKey !== key) {
      event.preventDefault();
      event.dataTransfer.dropEffect = "move";
      card.classList.add("drag-over");
    }
  });
  card.addEventListener("dragleave", () => card.classList.remove("drag-over"));
  card.addEventListener("drop", async (event) => {
    event.preventDefault();
    card.classList.remove("drag-over");
    if (!dragSampleKey || dragSampleKey === key) return;
    const draggedDir = currentSampleOrder.find((dir) => sampleKey(dir) === dragSampleKey);
    dragSampleKey = null;
    if (!draggedDir) return;
    // drop A onto B: A takes B's priority level (drag assigns priority,
    // dropping onto an unprioritized card clears it)
    await setPriority(draggedDir, project.priority != null ? project.priority : -1);
  });

  nodeCache.set(key, card);
  return card;
}

function renderRunRow(project, run) {
  const wrap = document.createElement("div");
  wrap.className = "run";
  const row = document.createElement("button");
  row.type = "button";
  row.className = `run-row status-${run.display_status || run.status}`;
  const lamp = document.createElement("i");
  lamp.className = `lamp ${run.display_status || run.status}`;
  const name = document.createElement("span");
  name.className = "run-name";
  name.textContent = run.config_name || (run.kind === "external" ? "外部程序" : run.title || run.id);
  const statusLabel = document.createElement("span");
  statusLabel.className = "run-status";
  statusLabel.textContent = STATUS_LABELS[run.display_status || run.status] || run.status;
  const dur = document.createElement("span");
  dur.className = "run-dur";
  dur.textContent = formatDuration(run.duration_s) || (run.started_at || run.created_at || "").toString().slice(5, 16);
  row.append(lamp, name, statusLabel, dur);
  wrap.appendChild(row);

  const key = `${sampleKey(project.sample_dir)}::${run.id}`;
  const defaultOpen =
    project.runs[0] === run &&
    ["running", "verifying", "queued", "stalled", "failed"].includes(project.status);
  const expanded = runOpenState.has(key) ? runOpenState.get(key) : defaultOpen;
  const details = document.createElement("div");
  details.className = "run-details";
  if (expanded && project.latest_full_view && run.id === project.latest_full_view.id) {
    details.appendChild(renderRunCard(project.latest_full_view));
  } else if (expanded) {
    const line = document.createElement("p");
    line.className = "card-meta";
    line.textContent = run.error ? `错误：${run.error}` : `run ${run.id}（${run.kind}）· ${run.started_at || ""}`;
    details.appendChild(line);
  }
  if (expanded) wrap.appendChild(details);
  row.addEventListener("click", () => {
    runOpenState.set(key, !expanded);
    refresh();
  });
  return wrap;
}

// ------------------------------------------------------ rich run card

function renderRunCard(job) {
  const rawStatus = job.status || "queued";
  const status = job.display_status || rawStatus;
  const card = document.createElement("article");
  card.className = `job-card status-${status}`;
  const progress = job.progress || {};

  const stepLine = document.createElement("p");
  stepLine.className = "step-line";
  if (status === "queued") {
    stepLine.textContent = "等待 Worker 执行";
  } else if (progress.step_name) {
    const total = progress.step_total || "?";
    const units = Number(progress.unit_total || 0) > 0 ? ` · ${progress.unit_done}/${progress.unit_total}` : "";
    stepLine.textContent = `步骤 ${progress.step_index || "?"}/${total} ${progress.step_name}${units}`;
  } else if (status === "verifying") {
    stepLine.textContent = "正在验证输出文件";
  } else if ((status === "running" || status === "stalled") && !progress.step_name) {
    stepLine.textContent = "进度未知（程序未上报单位进度）";
  }
  card.appendChild(stepLine);

  const etaLine = document.createElement("p");
  etaLine.className = "eta-line";
  if (status === "running" || status === "verifying" || status === "stalled") {
    const chunks = [];
    const dur = formatDuration(job.duration_s);
    if (dur) chunks.push(`总耗时 ${dur}`);
    const stepDur = formatDuration(job.current_step_duration_s);
    if (stepDur) chunks.push(`当前步骤 ${stepDur}`);
    if (job.heartbeat_age_s != null) chunks.push(`心跳 ${job.heartbeat_age_s}s 前`);
    const eta = job.eta || {};
    if (!eta.collecting && (eta.current_step_remaining_s != null || eta.total_remaining_s != null)) {
      const step = formatEta(eta.current_step_remaining_s);
      const total = formatEta(eta.total_remaining_s);
      if (step && total && step !== total) chunks.push(`本步剩余 ${step}，全部剩余 ${total}`);
      else if (step || total) chunks.push(`预计剩余 ${step || total}`);
    }
    etaLine.textContent = chunks.join("，");
  } else if (status === "succeeded" || status === "done") {
    etaLine.textContent = formatDuration(job.duration_s) ? `耗时 ${formatDuration(job.duration_s)}` : "分析完成";
  }
  if (etaLine.textContent) card.appendChild(etaLine);

  const track = document.createElement("div");
  track.className = "progress-track";
  const fill = document.createElement("div");
  fill.className = `progress-fill status-${status}`;
  fill.style.width = `${progressPercent(job)}%`;
  track.appendChild(fill);
  card.appendChild(track);

  const error = job.error;
  if (error) {
    const box = document.createElement("div");
    box.className = "error-detail";
    const msgLine = document.createElement("p");
    if (typeof error === "object") {
      const head = `[${error.code || "?"}]` + (error.step_id != null ? ` 步骤 ${error.step_id}` : "") + (error.retryable ? " · 可重试" : " · 不可自动重试");
      const headLine = document.createElement("strong");
      headLine.textContent = head;
      msgLine.textContent = error.message || "";
      box.append(headLine, msgLine);
      if (error.suggestion) {
        const sug = document.createElement("p");
        sug.textContent = `建议：${error.suggestion}`;
        box.appendChild(sug);
      }
    } else {
      msgLine.textContent = String(error);
      box.appendChild(msgLine);
    }
    card.appendChild(box);
  }

  const qcImages = job.qc_images || [];
  if (qcImages.length) {
    const box = document.createElement("details");
    box.className = "results qc-images";
    box.open = runOpenState.get(`qc::${job.id}`) === true;
    box.addEventListener("toggle", () => runOpenState.set(`qc::${job.id}`, box.open));
    const heading = document.createElement("summary");
    heading.textContent = `QC 图（${qcImages.length}）`;
    const list = document.createElement("ul");
    for (const item of qcImages) {
      const li = document.createElement("li");
      const link = document.createElement("a");
      link.href = item.url;
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = item.name;
      li.append(link, document.createTextNode(`  ${item.path} `));
      const actions = document.createElement("span");
      actions.className = "result-actions";
      const copyBtn = document.createElement("button");
      copyBtn.type = "button";
      copyBtn.className = "linkish";
      copyBtn.textContent = "复制";
      copyBtn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        navigator.clipboard.writeText(item.path);
      });
      actions.append(copyBtn);
      li.appendChild(actions);
      list.appendChild(li);
    }
    box.append(heading, list);
    card.appendChild(box);
  }

  const results = job.results || [];
  if ((status === "succeeded" || status === "done") && results.length) {
    const box = document.createElement("details");
    box.className = "results";
    box.open = runOpenState.get(`res::${job.id}`) === true;
    box.addEventListener("toggle", () => runOpenState.set(`res::${job.id}`, box.open));
    const heading = document.createElement("summary");
    heading.textContent = `结果位置（${results.length}）`;
    const list = document.createElement("ul");
    for (const item of results) {
      const li = document.createElement("li");
      const name = document.createElement("strong");
      name.textContent = item.name;
      li.append(name, document.createTextNode(`  ${item.path} `));
      const actions = document.createElement("span");
      actions.className = "result-actions";
      const copyBtn = document.createElement("button");
      copyBtn.type = "button";
      copyBtn.className = "linkish";
      copyBtn.textContent = "复制";
      copyBtn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        navigator.clipboard.writeText(item.path);
      });
      const openBtn = document.createElement("button");
      openBtn.type = "button";
      openBtn.className = "linkish";
      openBtn.textContent = "打开目录";
      openBtn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        openPath(item.path, openBtn);
      });
      actions.append(copyBtn, openBtn);
      li.appendChild(actions);
      list.appendChild(li);
    }
    box.append(heading, list);
    card.appendChild(box);
  }

  const actions = document.createElement("div");
  actions.className = "card-actions";
  const logBtn = document.createElement("button");
  logBtn.type = "button";
  logBtn.className = status === "failed" || status === "stalled" ? "btn danger" : "btn";
  logBtn.textContent = "查看日志";
  logBtn.addEventListener("click", () => openLog(job));
  actions.appendChild(logBtn);
  if (status === "running" || status === "verifying" || status === "stalled" || status === "queued") {
    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.className = "btn danger";
    cancel.textContent = "取消";
    cancel.addEventListener("click", () => cancelJob(job.id));
    actions.appendChild(cancel);
  } else {
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "btn";
    remove.textContent = "移除";
    remove.addEventListener("click", () => removeJob(job.id));
    actions.appendChild(remove);
  }
  card.appendChild(actions);
  return card;
}

// ------------------------------------------------------------ stage-times view

async function openStageTimes() {
  els.stageDialog.showModal();
  els.stageBody.textContent = "统计中…";
  try {
    const data = await api("/api/stage-times");
    renderStageTimes(data);
  } catch (error) {
    els.stageBody.textContent = `加载失败：${error.message}`;
  }
}

function renderStageTimes(data) {
  const stages = data.stages || [];
  els.stageBody.replaceChildren();
  if (!stages.length) {
    els.stageBody.textContent = `没有找到 run manifest（扫描 ${data.manifests_scanned} 个）。`;
    return;
  }
  const meta = document.createElement("p");
  meta.className = "stage-meta";
  meta.textContent = `扫描 ${data.manifests_scanned} 个 run manifest · 按中位耗时排序 · ${String(data.generated_at).replace("T", " ").slice(0, 16)}`;
  els.stageBody.appendChild(meta);
  const maxMedian = Math.max(...stages.map((s) => s.median_s), 1);
  for (const stage of stages) {
    const row = document.createElement("div");
    row.className = "stage-row";
    const head = document.createElement("div");
    head.className = "stage-head";
    const name = document.createElement("strong");
    name.textContent = stage.module;
    const stat = document.createElement("span");
    stat.className = "stage-stat";
    stat.textContent = `中位 ${formatDuration(stage.median_s) || "?"} · 均值 ${formatDuration(stage.mean_s) || "?"} · ${stage.count} 次`;
    head.append(name, stat);
    const track = document.createElement("div");
    track.className = "stage-track";
    const bar = document.createElement("div");
    bar.className = "stage-bar";
    bar.style.width = `${Math.max(2, (stage.median_s / maxMedian) * 100)}%`;
    track.appendChild(bar);
    row.append(head, track);
    els.stageBody.appendChild(row);
  }
}

// ------------------------------------------------------------------ actions

async function refresh() {
  try {
    const data = await api("/api/projects");
    // freeze the board while a drag is in flight: the rebuild would remove
    // the drag source from the DOM and cancel the drop
    if (dragSampleKey == null) {
      renderBoard(data);
    }
    if (Date.now() >= errorHoldUntil) showError("");
    if (openLogJobId && els.logDialog.open) await loadLog(openLogJobId, false);
  } catch (error) {
    showError(error.message);
  }
}

async function removeJob(jobId) {
  try {
    await api(`/api/jobs/${encodeURIComponent(jobId)}`, { method: "DELETE" });
    await refresh();
  } catch (error) {
    showError(error.message);
  }
}

async function cancelJob(jobId) {
  try {
    await api(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
    await refresh();
  } catch (error) {
    showError(error.message);
  }
}

async function loadLog(jobId, resetScroll) {
  const payload = await api(`/api/jobs/${encodeURIComponent(jobId)}/log?tail=250`);
  els.logBody.textContent = payload.text || "(日志还是空的)";
  els.logTitle.textContent = payload.path || "日志";
  if (resetScroll) els.logBody.scrollTop = els.logBody.scrollHeight;
}

async function openLog(job) {
  openLogJobId = job.id;
  els.logDialog.showModal();
  els.logBody.textContent = "加载中…";
  await loadLog(job.id, true);
}

async function openPath(path, button) {
  const original = button ? button.textContent : "";
  if (button) button.textContent = "正在打开…";
  try {
    await api("/api/open", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    if (button) button.textContent = "已打开";
    showError(`已在资源管理器打开：${path}`, 4000, "info");
  } catch (error) {
    if (button) button.textContent = original || "打开目录";
    showError(error.message, 8000);
  } finally {
    if (button) setTimeout(() => { button.textContent = original || "打开目录"; }, 2500);
  }
}

els.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const sampleDir = els.sampleDir.value.trim();
  if (!sampleDir) return;
  try {
    await api("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sample_dir: sampleDir, config_path: els.configPath.value.trim() }),
    });
    els.sampleDir.value = "";
    els.configPath.value = "";
    showError("");
    await refresh();
  } catch (error) {
    showError(error.message, 8000);
  }
});

els.logClose.addEventListener("click", () => {
  openLogJobId = null;
  els.logDialog.close();
});

if (els.stageBtn) els.stageBtn.addEventListener("click", openStageTimes);
if (els.stageClose) {
  els.stageClose.addEventListener("click", () => els.stageDialog.close());
}

refresh();
setInterval(refresh, POLL_MS);
