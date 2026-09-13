const POLL_MS = 1500;
const STATUS_LABELS = {
  queued: "等待",
  running: "运行中",
  failed: "报错停止",
  done: "完成",
  cancelled: "已取消",
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

function formatSeconds(value) {
  if (value == null || Number.isNaN(Number(value))) return null;
  const seconds = Math.max(0, Number(value));
  if (seconds < 90) return `约 ${Math.round(seconds)} 秒`;
  const minutes = seconds / 60;
  if (minutes < 90) return `约 ${minutes < 10 ? minutes.toFixed(1) : Math.round(minutes)} 分钟`;
  return `约 ${(minutes / 60).toFixed(1)} 小时`;
}

function progressPercent(job) {
  if (job.status === "done") return 100;
  const progress = job.progress || {};
  const unitTotal = Number(progress.unit_total || 0);
  const unitDone = Number(progress.unit_done || 0);
  const total = Number(progress.step_total || 6);
  const index = Number(progress.step_index || 0);
  if (total <= 0 || index <= 0) return job.status === "running" ? 8 : 0;
  const base = ((index - 1) / total) * 100;
  const stepSpan = 100 / total;
  if (unitTotal > 0) {
    const frac = Math.max(0, Math.min(1, unitDone / unitTotal));
    return Math.min(99, Math.round(base + stepSpan * frac));
  }
  return Math.min(99, Math.round(base + stepSpan * 0.45));
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
  const running = jobs.filter((job) => job.status === "running");
  const queued = jobs.filter((job) => job.status === "queued");
  if (running.length) {
    els.runner.textContent = `运行中 ${running.length} 个` + (queued.length ? `，排队 ${queued.length} 个` : "");
    els.runner.classList.add("busy");
  } else if (queued.length) {
    els.runner.textContent = `等待自动开始（${queued.length}）`;
    els.runner.classList.remove("busy");
  } else {
    els.runner.textContent = "队列空闲";
    els.runner.classList.remove("busy");
  }

  els.jobList.replaceChildren();
  if (!jobs.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "还没有任务。加入样本后会在当前任务结束后自动开始。";
    els.jobList.appendChild(empty);
    return;
  }
  for (const job of jobs) els.jobList.appendChild(renderCard(job));
}

function renderCard(job) {
  const status = job.status || "queued";
  const card = document.createElement("article");
  card.className = "job-card";

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

  const progress = job.progress || {};
  const step = document.createElement("p");
  step.className = "job-step";
  if (progress.step_name) {
    const total = progress.step_total || 0;
    const index = progress.step_index || 0;
    step.textContent = index && total ? `步骤 ${index}/${total}  ${progress.step_name}` : progress.step_name;
  } else if (status === "queued") {
    step.textContent = "等待当前任务结束后自动开始";
  } else {
    step.textContent = progress.error || job.error || "";
  }

  const eta = job.eta || {};
  const etaLine = document.createElement("p");
  etaLine.className = "eta-line";
  if (status === "running") {
    const current = formatSeconds(eta.current_step_remaining_s);
    const total = formatSeconds(eta.total_remaining_s);
    const unitTotal = Number(eta.unit_total || progress.unit_total || 0);
    const unitDone = Number(eta.unit_done || progress.unit_done || 0);
    if (eta.collecting && !current && !total) {
      etaLine.textContent = unitTotal
        ? `正在收集耗时  ·  ${unitDone}/${unitTotal}`
        : "正在收集耗时";
    } else {
      const parts = [];
      if (unitTotal) parts.push(`${unitDone}/${unitTotal}`);
      if (current) parts.push(`本步剩余 ${current}`);
      if (total && total !== current) parts.push(`全部剩余 ${total}`);
      else if (total && !current) parts.push(`全部剩余 ${total}`);
      etaLine.textContent = parts.join("  ·  ") || "正在收集耗时";
    }
  } else if (status === "done") {
    etaLine.textContent = "分析完成";
  } else if (job.error) {
    etaLine.textContent = job.error;
  }

  const track = document.createElement("div");
  track.className = "progress-track";
  const fill = document.createElement("div");
  fill.className = `progress-fill ${status}`;
  fill.style.width = `${progressPercent(job)}%`;
  track.appendChild(fill);
  card.append(head, meta, step, etaLine, track);

  const results = job.results || progress.results || [];
  if (status === "done" && results.length) {
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
  logBtn.className = status === "failed" ? "btn danger" : "btn";
  logBtn.textContent = "查看日志";
  logBtn.addEventListener("click", () => openLog(job));
  actions.appendChild(logBtn);
  if (status === "running") {
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
