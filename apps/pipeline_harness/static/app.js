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
const batchOpenState = new Map(); // batch name -> user-pinned open/closed
const runOpenState = new Map(); // "<scope>::<id>" -> pinned open/closed
const nodeCache = new Map(); // sample key -> live card element (focus protection)
const batchNodes = new Map(); // batch key -> {node, fingerprint} (DOM reconciliation)
const dirtyNotes = new Set(); // sample keys with unsaved note edits
const batchNotesCache = new Map(); // batch name -> live notes node (edit pinning)
let currentSampleOrder = []; // flat display order of sample_dirs
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

  if (!batches.length) {
    els.board.replaceChildren();
    batchNodes.clear();
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "还没有项目。提交任务或用 wrap 挂载程序后，样本目录会自动出现在这里。";
    els.board.appendChild(empty);
    return;
  }

  // DOM reconciliation: an unchanged batch keeps its node untouched; a changed
  // batch swaps only its non-interacted project cards. The node a user is
  // typing notes in (or has unsaved note edits in) is never detached —
  // detaching resets focus, IME composition, and <details> open state.
  // Only TEXT-INPUT widgets pin their card: a button click (run row, config
  // select's row...) must NOT pin, or the very click that toggles a row open
  // would freeze the card and the toggle would never render.
  const focusedKey = isTextInput(document.activeElement)
    ? sampleKeyOfNode(document.activeElement)
    : null;
  const order = [];
  const wantKeys = [];
  const applyBatch = (batch, dim) => {
    for (const project of batch.projects) order.push(project.sample_dir);
    const key = sampleKey(batch.batch_dir);
    wantKeys.push(key);
    reconcileBatch(batch, dim, focusedKey);
  };
  for (const batch of activeBatches) applyBatch(batch, false);
  for (const batch of otherBatches) applyBatch(batch, true); // 展开显示，仅样式略暗

  for (const [key, entry] of [...batchNodes]) {
    if (!wantKeys.includes(key)) {
      entry.node.remove();
      batchNodes.delete(key);
    }
  }
  // fix display order only when it actually changed (moving a node blurs focus)
  const domKeys = [...els.board.children]
    .map((n) => n.dataset.batchKey)
    .join("\u0001");
  if (domKeys !== wantKeys.join("\u0001")) {
    for (const key of wantKeys) {
      const entry = batchNodes.get(key);
      if (entry) els.board.appendChild(entry.node);
    }
  }
  currentSampleOrder = order;
  renderResourcePanel(data.resources || null, batches);
}

function sampleKeyOfNode(node) {
  while (node && node !== document.body) {
    if (node.dataset && node.dataset.sampleKey) return node.dataset.sampleKey;
    node = node.parentElement;
  }
  return null;
}

// text-input widgets pin their card; buttons/details do not (their clicks
// change UI state that the next render must show)
function isTextInput(node) {
  if (!node || !node.tagName) return false;
  const tag = node.tagName.toUpperCase();
  return tag === "TEXTAREA" || tag === "INPUT" || tag === "SELECT";
}

function batchFingerprint(batch, dim) {
  // Everything the batch DOM renders, minus the heavy embedded full views.
  return (
    JSON.stringify(batch, (key, value) =>
      key === "full_view" || key === "latest_full_view" ? undefined : value
    ) + (dim ? "|dim" : "")
  );
}

function reconcileBatch(batch, dim, focusedKey) {
  const key = sampleKey(batch.batch_dir);
  const fingerprint = batchFingerprint(batch, dim);
  const entry = batchNodes.get(key);
  if (entry && entry.fingerprint === fingerprint && entry.node.isConnected) return;

  const holdsUserState = batch.projects.some(
    (p) => sampleKey(p.sample_dir) === focusedKey || dirtyNotes.has(sampleKey(p.sample_dir))
  );
  if (entry && entry.node.isConnected && holdsUserState) {
    updateBatchInPlace(entry.node, batch, dim, focusedKey);
    entry.fingerprint = fingerprint;
    return;
  }
  const node = renderBatch(batch);
  if (dim) node.classList.add("other-root");
  node.dataset.batchKey = key;
  if (entry && entry.node.isConnected) entry.node.replaceWith(node);
  batchNodes.set(key, { node, fingerprint });
}

function updateBatchInPlace(node, batch, dim, focusedKey) {
  node.className = `batch${batch.priority != null ? ` has-p${batch.priority}` : ""}${dim ? " other-root" : ""}`;
  const oldSummary = node.querySelector("summary");
  if (oldSummary) oldSummary.replaceWith(batchHeader(batch));
  const list = node.querySelector(".batch-projects");
  if (!list) return;
  const oldCards = new Map();
  for (const card of [...list.querySelectorAll(".project-card")]) {
    oldCards.set(card.dataset.sampleKey, card);
  }
  const kept = new Set();
  for (const project of batch.projects) {
    const key = sampleKey(project.sample_dir);
    kept.add(key);
    const old = oldCards.get(key);
    // the card being typed in (or holding unsaved notes) stays untouched
    if (old && (key === focusedKey || dirtyNotes.has(key))) continue;
    const fresh = renderProject(project);
    if (old) old.replaceWith(fresh);
    else list.appendChild(fresh);
  }
  for (const [key, card] of oldCards) if (!kept.has(key)) card.remove();
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
  const progress = batch.progress || {};
  const chunks = [];
  if (progress.done) chunks.push(`${progress.done} 完成`);
  if (progress.running) chunks.push(`${progress.running} 运行`);
  if (progress.failed) chunks.push(`${progress.failed} 失败`);
  if (progress.waiting) chunks.push(`${progress.waiting} 未跑`);
  meta.textContent = chunks.join("，") || `${batch.projects.length} 样本`;

  const wrap = document.createElement("div");
  wrap.className = "batch-summary";
  wrap.append(name, meta);

  const right = document.createElement("div");
  right.className = "batch-right";

  if (batch.remaining_s != null) {
    const rem = document.createElement("span");
    rem.className = "batch-remaining";
    rem.textContent = `还需 ≈ ${formatEta(batch.remaining_s)}`;
    right.appendChild(rem);
  }
  if (progress.total) {
    const bar = document.createElement("div");
    bar.className = "batch-progress";
    for (const [kind, cls] of [["done", "ok"], ["running", "run"], ["failed", "bad"], ["waiting", "idle"]]) {
      if (!progress[kind]) continue;
      const seg = document.createElement("i");
      seg.className = `bp-seg ${cls}`;
      seg.style.width = `${(progress[kind] / progress.total) * 100}%`;
      bar.appendChild(seg);
    }
    right.appendChild(bar);
  }

  // priority selector lives on the PROJECT (batch): P-high moves to the top;
  // rendered last so it sits at the far right of the batch header
  const prioSeg = document.createElement("span");
  prioSeg.className = "prio-seg";
  prioSeg.title = "项目紧急度（P0 最高，自动置顶；再点一次取消）";
  for (const level of [0, 1, 2, 3]) {
    const btn = document.createElement("button");
    btn.type = "button";
    const active = batch.priority === level;
    btn.className = `prio-btn p${level}${active ? " active" : ""}`;
    btn.textContent = `P${level}`;
    btn.setAttribute("aria-pressed", active ? "true" : "false");
    btn.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      setPriority(batch.batch_dir, batch.priority === level ? -1 : level);
    });
    prioSeg.appendChild(btn);
  }
  right.appendChild(prioSeg);

  heading.append(wrap, right);
  return heading;
}

function renderBatchNotes(batch) {
  // Project-level notes: 需求 / 复盘 as two separate entries + 记一笔
  const notes = batch.notes || {};
  const key = `batch::${batch.batch}`;
  const holder = document.createElement("div");
  holder.className = "batch-notes";
  holder.dataset.batchKey = key;

  const dirty = new Set(); // scopes with unsaved edits -> pin this node

  const makeEntry = (scope, label, placeholder, rows, value) => {
    const entryKey = `notes::${key}::${scope}`;
    const box = document.createElement("details");
    box.className = "notes";
    box.open = runOpenState.get(entryKey) === true;
    box.addEventListener("toggle", () => runOpenState.set(entryKey, box.open));
    const summary = document.createElement("summary");
    const renderPreview = () => {
      const preview = (area.value || "").trim().split("\n")[0] || "点击填写";
      summary.textContent = `${label}：${preview.slice(0, 80)}`;
    };
    const area = document.createElement("textarea");
    area.rows = rows;
    area.placeholder = placeholder;
    area.value = value || "";
    renderPreview();
    area.addEventListener("input", () => dirty.add(scope));
    const saveBtn = document.createElement("button");
    saveBtn.type = "button";
    saveBtn.className = "btn";
    saveBtn.textContent = "保存";
    saveBtn.addEventListener("click", async () => {
      saveBtn.disabled = true;
      saveBtn.textContent = "保存中…";
      try {
        const body = { sample_dir: batch.batch_dir };
        body[scope === "requirements" ? "requirements" : "retro"] = area.value;
        await api("/api/projects/notes", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        dirty.delete(scope);
        renderPreview();
        showError(`${label}已保存`, 3000, "info");
      } catch (error) {
        showError(error.message, 6000);
      }
      saveBtn.disabled = false;
      saveBtn.textContent = "保存";
    });
    box.append(summary, area, saveBtn);
    return box;
  };

  holder.appendChild(
    makeEntry("requirements", "需求", "分析需求（这个项目要做什么、对照组、交付物…）", 2, notes.requirements)
  );
  holder.appendChild(
    makeEntry("retro", "复盘", "复盘（踩过的坑、失误、下次注意…）", 3, notes.retro)
  );

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
        body: JSON.stringify({ sample_dir: batch.batch_dir, append_note: text }),
      });
      quickInput.value = "";
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
  holder.appendChild(quickRow);

  const noteLog = document.createElement("div");
  noteLog.className = "note-log";
  for (const entry of (notes.log || []).slice(-3)) {
    const line = document.createElement("p");
    line.textContent = `${(entry.at || "").replace("T", " ").slice(0, 16)}  ${entry.text}`;
    noteLog.appendChild(line);
  }
  holder.appendChild(noteLog);

  // pin while editing: never rebuild a notes node with unsaved input
  holder._isPinned = () => dirty.size > 0 || (isTextInput(document.activeElement) && holder.contains(document.activeElement));
  return holder;
}

function renderBatch(batch) {
  const box = document.createElement("details");
  box.className = "batch" + (batch.priority != null ? ` has-p${batch.priority}` : "");
  const hasActive = batch.projects.some((p) => ["running", "verifying", "queued", "stalled"].includes(p.status));
  // three-state memory: once the user toggles a batch, their choice sticks
  // across polls; the auto-open default only applies before any interaction
  const userChoice = batchOpenState.get(batch.batch);
  box.open = userChoice != null ? userChoice : hasActive;
  box.addEventListener("toggle", () => batchOpenState.set(batch.batch, box.open));
  box.appendChild(batchHeader(batch));

  // project-level notes (需求/复盘) sit above the sample list
  const cachedNotes = batchNotesCache.get(batch.batch);
  if (cachedNotes && cachedNotes._isPinned && cachedNotes._isPinned() && cachedNotes.isConnected) {
    box.appendChild(cachedNotes);
  } else {
    const node = renderBatchNotes(batch);
    batchNotesCache.set(batch.batch, node);
    box.appendChild(node);
  }

  const list = document.createElement("div");
  list.className = "batch-projects";
  for (const project of batch.projects) list.appendChild(renderProject(project));
  box.appendChild(list);
  return box;
}

// ------------------------------------------------------------------ project

function renderProject(project) {
  const key = sampleKey(project.sample_dir);
  const cached = nodeCache.get(key);
  // Never swap a card holding TEXT-INPUT focus or unsaved note edits —
  // that is what made notes lose focus/text every 1.5s poll. Button clicks
  // (run rows, toggles) must NOT pin: the state they change needs re-render.
  const inputFocused = cached && cached.isConnected && isTextInput(document.activeElement) &&
    cached.contains(document.activeElement);
  if (inputFocused || (cached && cached.isConnected && dirtyNotes.has(key))) {
    return cached;
  }

  const card = document.createElement("article");
  card.className = `project-card status-${project.status}${project.anomaly ? " anomalous" : ""}`;
  card.dataset.sampleKey = key;

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
  const lamp = document.createElement("i");
  lamp.className = `lamp ${project.status}`;
  const label = document.createElement("span");
  label.textContent = STATUS_LABELS[project.status] || project.status;
  statusCluster.append(lamp, label);
  head.append(titleWrap, statusCluster);
  card.appendChild(head);

  // ---- chips: anomaly / profile / estimate (short — details on hover)
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
    chip.textContent = `≈ ${formatEta(project.estimate.seconds)}`;
    chip.title = project.estimate.basis || "";
    chips.appendChild(chip);
  }
  if (chips.childNodes.length) card.appendChild(chips);

  // ---- meta: sample dir only (manifest summary dropped: info overload)
  const meta = document.createElement("p");
  meta.className = "card-meta";
  meta.textContent = project.sample_dir;
  card.appendChild(meta);

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
  const archiveBtn = document.createElement("button");
  archiveBtn.type = "button";
  archiveBtn.className = "btn danger";
  archiveBtn.textContent = "归档";
  archiveBtn.addEventListener("click", () => archiveSample(project, archiveBtn));
  startRow.append(select, startBtn, openBtn, archiveBtn);
  card.appendChild(startRow);

  // ---- notes moved UP to the project (batch) level; sample rows carry no notes

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

const STEP_STATE_LABELS = { done: "完成", active: "进行中", pending: "未开始", failed: "失败" };

function renderStepper(job, status) {
  // Coarse step dots: from wrap --steps (steps/step_states) or, for main
  // pipeline runs, the canonical steps[] in progress.json (list of dicts).
  // Falls back to the plain bar when no step list is known.
  const progress = job.progress || {};

  let names = null;
  let states = null;
  if (Array.isArray(progress.steps) && progress.steps.length) {
    if (typeof progress.steps[0] === "object" && progress.steps[0] !== null) {
      // main pipeline: steps[] is [{name, ended_at, skipped}]
      names = progress.steps.map((s) => s.name || "步骤");
      states = progress.steps.map((s) => (s.ended_at ? "done" : "pending"));
      const cur = Number(progress.step_index || 0);
      if (cur > 0 && (status === "failed" || status === "running" || status === "stalled")) {
        const activeState = status === "failed" ? "failed" : "active";
        states = progress.steps.map((s, i) =>
          i + 1 < cur ? "done" : i + 1 === cur ? activeState : "pending"
        );
      }
    } else {
      // wrap --steps: steps[] is a list of names + step_states
      names = progress.steps;
      states = Array.isArray(progress.step_states) && progress.step_states.length === names.length
        ? progress.step_states
        : names.map((_, i) => {
            const cur = progress.current_step;
            if (status === "succeeded" || status === "done") return "done";
            if (status === "failed") return i + 1 === cur ? "failed" : i + 1 < cur ? "done" : "pending";
            if (cur == null) return "pending";
            return i + 1 < cur ? "done" : i + 1 === cur ? "active" : "pending";
          });
    }
  }

  if (!names) {
    // fallback: the plain percentage bar
    const track = document.createElement("div");
    track.className = "progress-track";
    const fill = document.createElement("div");
    fill.className = `progress-fill status-${status}`;
    fill.style.width = `${progressPercent(job)}%`;
    track.appendChild(fill);
    return track;
  }

  const stepper = document.createElement("div");
  stepper.className = "stepper";
  names.forEach((name, i) => {
    const state = states[i] || "pending";
    const dot = document.createElement("span");
    dot.className = `step-dot ${state}`;
    dot.title = `${i + 1}. ${name} — ${STEP_STATE_LABELS[state] || state}`;
    if (i > 0) stepper.appendChild(document.createElement("i")).className = "step-link " + state;
    stepper.appendChild(dot);
  });
  return stepper;
}

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

  const track = renderStepper(job, status);
  if (track) card.appendChild(track);

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

function formatMemMB(mb) {
  if (mb == null) return "?";
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`;
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
    const mem = stage.peak_mem_mb_median
      ? ` · 峰值内存 ${formatMemMB(stage.peak_mem_mb_median)}${stage.peak_mem_count ? ` (${stage.peak_mem_count} 次)` : ""}`
      : " · 峰值内存 无记录";
    stat.textContent = `中位 ${formatDuration(stage.median_s) || "?"} · 均值 ${formatDuration(stage.mean_s) || "?"} · ${stage.count} 次${mem}`;
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
    renderBoard(data);
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

async function archiveSample(project, button) {
  // Dry-run first: the confirm dialog lists exactly what will be deleted,
  // so the plan is reviewed against real names, not a generic description.
  let preview;
  try {
    preview = await api("/api/projects/archive", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sample_dir: project.sample_dir, dry_run: true }),
    });
  } catch (error) {
    showError(error.message, 8000);
    return;
  }
  const names = preview.would_delete || [];
  const shown = names.slice(0, 30).join("\n");
  const more = names.length > 30 ? `\n… 共 ${names.length} 项` : "";
  const message =
    `归档 ${project.sample_name}，将删除以下 ${names.length} 项（zarr / 原始 ims / transforms / 完成标记）：\n` +
    `${shown || "（无）"}${more}\n` +
    `保留 result*、visualization*、qc 及小文件（config、笔记、manifest、日志），目录移入 _archived。\n` +
    `此操作不可撤销，确定继续？`;
  if (!window.confirm(message)) return;
  button.disabled = true;
  button.textContent = "归档中…";
  try {
    const result = await api("/api/projects/archive", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sample_dir: project.sample_dir }),
    });
    const key = sampleKey(project.sample_dir);
    nodeCache.delete(key);
    dirtyNotes.delete(key);
    showError(`已归档：清理 ${result.deleted_count} 项 → ${result.archived_to}`, 10000, "info");
  } catch (error) {
    button.disabled = false;
    button.textContent = "归档";
    showError(error.message, 8000);
  }
  await refresh();
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
