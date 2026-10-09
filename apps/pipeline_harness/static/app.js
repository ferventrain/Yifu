const POLL_MS = 1500;
const STATUS_LABELS = {
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
  runner: document.getElementById("runner-status"),
  form: document.getElementById("add-form"),
  sampleDir: document.getElementById("sample-dir"),
  configPath: document.getElementById("config-path"),
  formError: document.getElementById("form-error"),
  jobList: document.getElementById("job-list"),
  logDialog: document.getElementById("log-dialog"),
  logTitle: document.getElementById("log-title"),
  logBody: document.getElementById("log-body"),
  logClose: document.getElementById("log-close"),
};

let openLogJobId = null;
const expandedResultJobs = new Set();
let errorHoldUntil = 0;

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

// Honest progress: completed steps + real unit fraction only. Steps without
// unit data contribute nothing beyond their boundary — never a guessed %.
function progressPercent(job) {
  const status = job.display_status || job.status;
  if (status === "succeeded" || status === "done") return 100;
  const progress = job.progress || {};
  const unitTotal = Number(progress.unit_total || 0);
  const unitDone = Number(progress.unit_done || 0);
  const total = Number(progress.step_total || 0);
  const index = Number(progress.step_index || 0);
  if (total <= 0 || index <= 0) return status === "running" ? 0 : 0;
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

function renderJobs(data) {
  const jobs = data.jobs || [];
  els.paths.textContent = `样本根目录 ${data.analysis_root}  ·  Active ${data.active_dir}`;
  const worker = data.worker || {};
  const running = jobs.filter((job) => (job.display_status || job.status) === "running");
  const queued = jobs.filter((job) => job.status === "queued");
  if (worker.alive) {
    els.runner.textContent = `Worker pid ${worker.pid} 在线` + (running.length ? `，运行中 ${running.length}` : "") + (queued.length ? `，排队 ${queued.length}` : "");
    els.runner.classList.toggle("busy", running.length > 0);
  } else {
    els.runner.textContent = running.length ? "有任务标记运行中但 Worker 不在线（失联）" : "Worker 离线（下次提交任务时自动启动）";
    els.runner.classList.remove("busy");
  }

  els.jobList.replaceChildren();
  if (!jobs.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "还没有任务。用标准命令提交：python -m pipeline_modules.harness run --sample-dir <样本目录>";
    els.jobList.appendChild(empty);
    return;
  }
  for (const job of jobs) els.jobList.appendChild(renderCard(job));
}

function renderCard(job) {
  const rawStatus = job.status || "queued";
  const status = job.display_status || rawStatus;
  const card = document.createElement("article");
  card.className = `job-card status-${status}`;

  const head = document.createElement("div");
  head.className = "job-head";
  const titleWrap = document.createElement("div");
  const title = document.createElement("div");
  title.className = "job-title";
  title.textContent = job.title || job.sample_name || job.sample_dir;
  const idLine = document.createElement("div");
  idLine.className = "job-id";
  idLine.textContent = job.pid ? `${job.id}  ·  pid ${job.pid}` : job.id;
  titleWrap.append(title, idLine);
  const cluster = document.createElement("div");
  cluster.className = "status-cluster";
  const lamp = document.createElement("i");
  lamp.className = `lamp ${status}`;
  const label = document.createElement("span");
  label.className = `status-label ${status}`;
  label.textContent = STATUS_LABELS[status] || status;
  cluster.append(lamp, label);
  head.append(titleWrap, cluster);

  const meta = document.createElement("p");
  meta.className = "job-meta";
  meta.textContent = job.config_path ? `${job.sample_dir}  ·  ${job.config_path}` : job.sample_dir;
  card.append(head, meta);

  // --- step / progress line (unknown stays unknown) ---
  const progress = job.progress || {};
  const step = document.createElement("p");
  step.className = "job-step";
  const unitTotal = Number(progress.unit_total || 0);
  const unitDone = Number(progress.unit_done || 0);
  if (status === "queued") {
    step.textContent = "等待 Worker 执行";
  } else if (progress.step_name) {
    const total = progress.step_total || 0;
    const index = progress.step_index || 0;
    const stepLabel = index && total ? `步骤 ${index}/${total}  ${progress.step_name}` : progress.step_name;
    if (unitTotal > 0) {
      step.textContent = `${stepLabel}  ·  ${unitDone} / ${unitTotal}`;
    } else if (status === "running" || status === "verifying") {
      step.textContent = `${stepLabel}  ·  进度未知`;
    } else {
      step.textContent = stepLabel;
    }
  } else if (status === "verifying") {
    step.textContent = "正在验证输出文件";
  }

  // --- timing / heartbeat line ---
  const etaLine = document.createElement("p");
  etaLine.className = "eta-line";
  const parts = [];
  if (status === "running" || status === "verifying" || status === "stalled") {
    const totalDur = formatDuration(job.duration_s);
    const stepDur = formatDuration(job.current_step_duration_s);
    if (totalDur) parts.push(`总耗时 ${totalDur}`);
    if (stepDur) parts.push(`当前步骤 ${stepDur}`);
    if (job.heartbeat_age_s != null) parts.push(`最后心跳 ${formatDuration(job.heartbeat_age_s)}前`);
    else parts.push("最后心跳 无");
    const eta = job.eta || {};
    if (eta.collecting || (eta.total_remaining_s == null && eta.current_step_remaining_s == null)) {
      parts.push("预计剩余：数据收集中");
    } else {
      const current = formatEta(eta.current_step_remaining_s);
      const total = formatEta(eta.total_remaining_s);
      if (current) parts.push(`本步剩余 ${current}`);
      if (total && total !== current) parts.push(`全部剩余 ${total}`);
    }
  } else if (status === "succeeded" || status === "done") {
    parts.push(job.duration_s != null ? `耗时 ${formatDuration(job.duration_s)}` : "分析完成");
  }
  etaLine.textContent = parts.join("  ·  ");
  card.append(step, etaLine);

  const track = document.createElement("div");
  track.className = "progress-track";
  const fill = document.createElement("div");
  fill.className = `progress-fill ${status}`;
  fill.style.width = `${progressPercent(job)}%`;
  track.appendChild(fill);
  card.appendChild(track);

  // --- structured error panel ---
  const error = job.error;
  if (error && typeof error === "object" && error.message) {
    const box = document.createElement("div");
    box.className = "error-detail";
    const codeLine = document.createElement("p");
    codeLine.className = "error-code";
    codeLine.textContent = `[${error.code}]${error.step_id != null ? ` 步骤 ${error.step_id}` : ""}${error.retryable ? " · 可重试" : " · 不可自动重试"}`;
    const msgLine = document.createElement("p");
    msgLine.textContent = error.message;
    box.append(codeLine, msgLine);
    if (error.suggestion) {
      const sug = document.createElement("p");
      sug.className = "error-suggestion";
      sug.textContent = `建议：${error.suggestion}`;
      box.appendChild(sug);
    }
    card.appendChild(box);
  } else if (typeof error === "string" && error) {
    const box = document.createElement("div");
    box.className = "error-detail";
    const msgLine = document.createElement("p");
    msgLine.textContent = error;
    box.appendChild(msgLine);
    card.appendChild(box);
  }

  // --- QC images (registration rigid/final check etc.), clickable ---
  const qcImages = job.qc_images || [];
  if (qcImages.length) {
    const box = document.createElement("details");
    box.className = "results qc-images";
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

  // --- artifact / result paths ---
  const results = job.results || progress.results || [];
  if ((status === "succeeded" || status === "done") && results.length) {
    const box = document.createElement("details");
    box.className = "results";
    box.open = expandedResultJobs.has(job.id);
    box.addEventListener("toggle", () => {
      if (box.open) expandedResultJobs.add(job.id);
      else expandedResultJobs.delete(job.id);
    });
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

async function refresh() {
  try {
    const data = await api("/api/jobs");
    renderJobs(data);
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
  try {
    await loadLog(job.id, true);
  } catch (error) {
    els.logBody.textContent = error.message;
  }
}

async function openPath(path, button) {
  const original = button ? button.textContent : "";
  if (button) {
    button.disabled = true;
    button.textContent = "正在打开…";
  }
  try {
    const payload = await api("/api/open", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    if (button) button.textContent = "已打开";
    showError(payload.opened ? `已打开 ${payload.opened}` : "", 4000, "info");
    window.setTimeout(() => {
      if (button && button.textContent === "已打开") button.textContent = original || "打开目录";
      if (button) button.disabled = false;
    }, 1500);
  } catch (error) {
    if (button) {
      button.disabled = false;
      button.textContent = original || "打开目录";
    }
    showError(error.message, 8000, "error");
  }
}

els.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await api("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sample_dir: els.sampleDir.value.trim(),
        config_path: els.configPath.value.trim(),
      }),
    });
    els.configPath.value = "";
    await refresh();
  } catch (error) {
    showError(error.message);
  }
});

els.logClose.addEventListener("click", () => {
  openLogJobId = null;
  els.logDialog.close();
});

refresh();
setInterval(refresh, POLL_MS);
