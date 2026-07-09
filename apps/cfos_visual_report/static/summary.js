/** Render Summary tab from bundle.summary or GET /api/summary payload. */

const SYSTEM_METRIC_OPTIONS = [
  { id: "system_cfos_count", label: "cFos count" },
  { id: "activation_load", label: "Activation load" },
  { id: "enrichment_score", label: "Enrichment" },
  { id: "system_voxel_density", label: "Voxel density" },
];

const systemChartAnimators = new WeakMap();

function easeOutCubic(t) {
  return 1 - (1 - t) ** 3;
}

export function resolveSummaryPayload(bundle) {
  return bundle?.summary?.schema_version ? bundle.summary : null;
}

function escapeHtml(text) {
  return String(text ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function sortedSystemRows(systems, metricKey) {
  return [...(systems || [])].sort((a, b) => Number(b[metricKey] || 0) - Number(a[metricKey] || 0));
}

export function drawSystemBarChart(
  canvas,
  systems,
  metricKey,
  formatNumber,
  { progress = 1, hoverIndex = null } = {}
) {
  if (!canvas) return null;
  const sorted = sortedSystemRows(systems, metricKey);
  const width = Math.max(canvas.parentElement?.clientWidth || 520, 320);
  const height = Math.max(220, sorted.length * 28 + 70);
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  if (!sorted.length) {
    ctx.fillStyle = "#64748b";
    ctx.font = "13px 'Segoe UI', sans-serif";
    ctx.fillText("No system-level metrics for this sample.", 16, 36);
    canvas._systemChartLayout = null;
    return null;
  }

  const maxValue = Math.max(...sorted.map((row) => Math.abs(Number(row[metricKey] || 0))), 1e-9);
  const leftPad = 108;
  const rightPad = 24;
  const topPad = 28;
  const rowHeight = Math.min(30, Math.max(22, (height - topPad - 24) / sorted.length));
  const barMaxW = width - leftPad - rightPad;
  const animProgress = Math.max(0, Math.min(1, Number(progress) || 0));

  ctx.fillStyle = "#1a2332";
  ctx.font = "600 14px 'Segoe UI', sans-serif";
  const metricLabel = SYSTEM_METRIC_OPTIONS.find((item) => item.id === metricKey)?.label || metricKey;
  ctx.fillText(`System summary · ${metricLabel}`, leftPad, 18);

  const rowLayouts = [];
  sorted.forEach((row, index) => {
    const y = topPad + index * rowHeight;
    const value = Number(row[metricKey] || 0);
    const targetBarW = (Math.abs(value) / maxValue) * barMaxW;
    const barW = targetBarW * animProgress;
    const isHover = hoverIndex === index;

    ctx.fillStyle = "#475569";
    ctx.font = isHover ? "600 12px 'Segoe UI', sans-serif" : "12px 'Segoe UI', sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(row.system_acronym || row.system_name || "—", leftPad - 10, y + rowHeight * 0.62);
    ctx.textAlign = "left";
    ctx.fillStyle = "#dbe6f3";
    ctx.fillRect(leftPad, y + 4, barMaxW, rowHeight - 10);
    ctx.fillStyle = isHover ? "#1d4ed8" : "#2b6cb0";
    ctx.fillRect(leftPad, y + 4, barW, rowHeight - 10);
    if (isHover) {
      ctx.strokeStyle = "#1e40af";
      ctx.lineWidth = 1;
      ctx.strokeRect(leftPad, y + 4, barMaxW, rowHeight - 10);
    }

    rowLayouts.push({
      index,
      row,
      value,
      y,
      rowHeight,
      leftPad,
      barMaxW,
      targetBarW,
    });
  });

  const layout = {
    sorted,
    metricKey,
    leftPad,
    topPad,
    rowHeight,
    barMaxW,
    rowLayouts,
  };
  canvas._systemChartLayout = layout;
  return layout;
}

function showSystemChartTooltip(event, layout, rowLayout, formatNumber) {
  const tooltip = document.getElementById("chart-tooltip");
  if (!tooltip || !rowLayout) return;
  const row = rowLayout.row;
  const metricLabel = SYSTEM_METRIC_OPTIONS.find((item) => item.id === layout.metricKey)?.label || layout.metricKey;
  tooltip.innerHTML = `<strong>${escapeHtml(row.system_name || row.system_acronym || "System")}</strong><div>${escapeHtml(metricLabel)}: ${escapeHtml(formatNumber(rowLayout.value))}</div>`;
  tooltip.classList.remove("hidden");
  tooltip.style.left = `${event.clientX + 12}px`;
  tooltip.style.top = `${event.clientY + 12}px`;
}

function hideSystemChartTooltip() {
  document.getElementById("chart-tooltip")?.classList.add("hidden");
}

function findSystemChartRow(layout, clientX, clientY, canvas) {
  if (!layout?.rowLayouts?.length) return null;
  const rect = canvas.getBoundingClientRect();
  const y = ((clientY - rect.top) / rect.height) * canvas.height;
  for (const rowLayout of layout.rowLayouts) {
    const rowTop = rowLayout.y + 2;
    const rowBottom = rowLayout.y + rowLayout.rowHeight - 2;
    if (y >= rowTop && y <= rowBottom) return rowLayout;
  }
  return null;
}

function animateSystemBarChart(canvas, systems, metricKey, formatNumber, { hoverIndex = null } = {}) {
  const prior = systemChartAnimators.get(canvas);
  if (prior?.frameId) cancelAnimationFrame(prior.frameId);

  const durationMs = 520;
  const start = performance.now();
  let frameId = 0;

  const step = (now) => {
    const t = Math.min(1, (now - start) / durationMs);
    const layout = drawSystemBarChart(canvas, systems, metricKey, formatNumber, {
      progress: easeOutCubic(t),
      hoverIndex,
    });
    if (t < 1) {
      frameId = requestAnimationFrame(step);
      systemChartAnimators.set(canvas, { frameId, hoverIndex });
    } else {
      systemChartAnimators.set(canvas, { frameId: 0, hoverIndex, layout });
    }
  };

  frameId = requestAnimationFrame(step);
  systemChartAnimators.set(canvas, { frameId, hoverIndex });
}

function bindSystemChartInteractions(canvas, systems, metricSelect, formatNumber) {
  if (!canvas || canvas._systemChartBound) return;
  canvas._systemChartBound = true;
  let hoverIndex = null;

  const rerender = (animate = false) => {
    const metricKey = metricSelect?.value || "system_cfos_count";
    if (animate) {
      animateSystemBarChart(canvas, systems, metricKey, formatNumber, { hoverIndex });
      return;
    }
    drawSystemBarChart(canvas, systems, metricKey, formatNumber, { progress: 1, hoverIndex });
  };

  canvas.addEventListener("mousemove", (event) => {
    const metricKey = metricSelect?.value || "system_cfos_count";
    const layout =
      canvas._systemChartLayout ||
      drawSystemBarChart(canvas, systems, metricKey, formatNumber, { progress: 1, hoverIndex });
    const rowLayout = findSystemChartRow(layout, event.clientX, event.clientY, canvas);
    const nextHover = rowLayout?.index ?? null;
    if (nextHover !== hoverIndex) {
      hoverIndex = nextHover;
      rerender(false);
    }
    if (rowLayout) {
      showSystemChartTooltip(event, layout, rowLayout, formatNumber);
      canvas.style.cursor = "default";
    } else {
      hideSystemChartTooltip();
      canvas.style.cursor = "default";
    }
  });

  canvas.addEventListener("mouseleave", () => {
    hoverIndex = null;
    hideSystemChartTooltip();
    rerender(false);
  });
}

export function renderSummaryPanel(container, summary, { formatNumber, onRegionClick } = {}) {
  if (!container || !summary) return;

  const sample = summary.sample || {};
  const atlas = summary.atlas || {};
  const stats = summary.headline_stats || {};
  const topRegions = summary.top_regions_by_count || [];
  const systems = summary.systems || [];
  const findings = summary.findings || [];
  const lateralityIndex = summary.laterality?.whole_brain_count_laterality_index;
  const activated = stats.activated_region_count ?? 0;
  const totalRegions = stats.total_region_count ?? 0;
  const activatedLabel =
    totalRegions > 0 ? `${activated}/${totalRegions}` : String(activated ?? "—");

  container.innerHTML = `
    <div class="summary-header">
      <h3>${escapeHtml(sample.sample_id || "Sample")}</h3>
    </div>
    <div class="stat-grid summary-stat-grid">
      <div class="stat-card"><div class="label">Total cFos count</div><div class="value">${formatNumber(stats.total_cfos_count)}</div></div>
      <div class="stat-card"><div class="label">Signal volume (µm³)</div><div class="value">${formatNumber(stats.signal_volume_um3)}</div></div>
      <div class="stat-card"><div class="label">Brain volume (µm³)</div><div class="value">${formatNumber(stats.brain_volume_um3)}</div></div>
      <div class="stat-card"><div class="label">Activated regions</div><div class="value">${escapeHtml(activatedLabel)}</div></div>
      <div class="stat-card"><div class="label">Count laterality index</div><div class="value">${
        lateralityIndex !== undefined && lateralityIndex !== null ? formatNumber(lateralityIndex) : "—"
      }</div></div>
    </div>
    <div class="summary-columns">
      <section class="summary-block">
        <h4>Top activated regions</h4>
        <table class="top-table">
          <thead><tr><th>Region</th><th>Count</th><th>Density</th></tr></thead>
          <tbody>
            ${topRegions
              .map(
                (row) =>
                  `<tr data-region-id="${row.region_id}" title="${escapeHtml(row.region_name || row.region_acronym)}"><td>${escapeHtml(row.region_acronym)}</td><td>${formatNumber(row.cfos_count)}</td><td>${formatNumber(row.voxel_density)}</td></tr>`
              )
              .join("")}
          </tbody>
        </table>
      </section>
      <section class="summary-block summary-system-block">
        <div class="summary-system-toolbar">
          <h4>System summary</h4>
          <label>
            Metric
            <select id="summary-system-metric">
              ${SYSTEM_METRIC_OPTIONS.map(
                (opt) => `<option value="${opt.id}">${escapeHtml(opt.label)}</option>`
              ).join("")}
            </select>
          </label>
        </div>
        <canvas id="summary-system-chart" class="summary-system-chart" height="280"></canvas>
      </section>
    </div>
    ${
      findings.length
        ? `<section class="summary-block"><h4>Notable findings</h4><ul class="summary-findings">${findings
            .map(
              (f) =>
                `<li${f.region_id ? ` data-region-id="${f.region_id}"` : ""}>${escapeHtml(f.message)}</li>`
            )
            .join("")}</ul></section>`
        : ""
    }
  `;

  const metricSelect = container.querySelector("#summary-system-metric");
  const chartCanvas = container.querySelector("#summary-system-chart");
  const renderChart = (animate = true) => {
    const metricKey = metricSelect?.value || "system_cfos_count";
    if (animate) {
      animateSystemBarChart(chartCanvas, systems, metricKey, formatNumber);
    } else {
      drawSystemBarChart(chartCanvas, systems, metricKey, formatNumber, { progress: 1 });
    }
  };
  metricSelect?.addEventListener("change", () => renderChart(true));
  bindSystemChartInteractions(chartCanvas, systems, metricSelect, formatNumber);
  renderChart(true);

  if (typeof onRegionClick === "function") {
    container.querySelectorAll("[data-region-id]").forEach((row) => {
      row.addEventListener("click", () => onRegionClick(Number(row.dataset.regionId)));
      row.style.cursor = "pointer";
    });
  }
}

export async function fetchSummary(apiBase, { sampleDir, signalCh, level, refresh = false } = {}) {
  const params = new URLSearchParams({
    sample_dir: sampleDir,
    signal_ch: signalCh || "ch1",
  });
  if (level) params.set("level", level);
  if (refresh) params.set("refresh", "true");
  const response = await fetch(`${apiBase}/api/summary?${params.toString()}`);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Summary request failed (${response.status})`);
  }
  return response.json();
}
