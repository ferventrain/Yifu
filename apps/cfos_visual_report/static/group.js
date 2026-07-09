/** Group comparison charts and API wiring. */

const GROUP_COLORS = ["#3182bd", "#e6550d", "#31a354", "#756bb1", "#636363", "#969696"];

const CHART = {
  bg: "#ffffff",
  text: "#1a2332",
  muted: "#475569",
  axis: "#64748b",
  grid: "rgba(100, 116, 139, 0.28)",
  point: "#2563eb",
  pointHover: "#1d4ed8",
  pointActive: "#c44e52",
  rowHover: "rgba(37, 99, 235, 0.18)",
  rowActive: "rgba(196, 78, 82, 0.22)",
  titleFont: "600 15px 'Segoe UI', 'Helvetica Neue', Arial, sans-serif",
  bodyFont: "13px 'Segoe UI', 'Helvetica Neue', Arial, sans-serif",
  smallFont: "12px 'Segoe UI', 'Helvetica Neue', Arial, sans-serif",
};

let compareInteraction = {
  payload: null,
  hoverRegionId: null,
  activeRegionId: null,
  onRegionClick: null,
};

export function bindGroupElements(root = document) {
  return {
    sampleA: root.getElementById("group-sample-a"),
    sampleB: root.getElementById("group-sample-b"),
    labelA: root.getElementById("group-a-label"),
    labelB: root.getElementById("group-b-label"),
    manifestPath: root.getElementById("group-manifest-path"),
    groupASelect: root.getElementById("group-a-select"),
    groupBSelect: root.getElementById("group-b-select"),
    levelSelect: root.getElementById("group-level-select"),
    metricSelect: root.getElementById("group-metric-select"),
    regionScopeSelect: root.getElementById("group-region-scope"),
    heatmapModeSelect: root.getElementById("group-heatmap-mode"),
    samplesBody: root.getElementById("analysis-samples-body"),
    addSamplePath: root.getElementById("analysis-add-sample-path"),
    addSampleBtn: root.getElementById("analysis-add-sample-btn"),
    runButton: root.getElementById("run-group-analysis"),
    status: root.getElementById("group-status"),
    scatterCanvas: root.getElementById("group-scatter"),
    heatmapCanvas: root.getElementById("group-heatmap"),
    correlationCanvas: root.getElementById("group-correlation"),
    correlationWrap: root.getElementById("group-correlation-wrap"),
    diffTable: root.getElementById("group-diff-table"),
    exportButton: root.getElementById("export-group-diff"),
  };
}

export function buildRegionMetaMap(payload) {
  const map = new Map();
  for (const row of payload?.differential_regions || []) {
    map.set(Number(row.region_id), row);
  }
  for (const point of payload?.pairwise_scatter?.points || []) {
    if (!map.has(Number(point.region_id))) {
      map.set(Number(point.region_id), point);
    }
  }
  for (let index = 0; index < (payload?.heatmap?.region_ids?.length || 0); index += 1) {
    const regionId = Number(payload.heatmap.region_ids[index]);
    if (!map.has(regionId)) {
      map.set(regionId, {
        region_id: regionId,
        region_acronym: payload.heatmap.region_labels[index],
        region_name: payload.heatmap.region_labels[index],
      });
    }
  }
  return map;
}

function groupColor(groupName, groupNames) {
  const index = groupNames.indexOf(groupName);
  return GROUP_COLORS[(index >= 0 ? index : 0) % GROUP_COLORS.length];
}

function formatChartNumber(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "—";
  if (Math.abs(num) >= 1000) return num.toLocaleString(undefined, { maximumFractionDigits: 1 });
  if (Math.abs(num) >= 10) return num.toLocaleString(undefined, { maximumFractionDigits: 2 });
  return num.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function metricLabel(metric) {
  return (
    {
      cfos_count: "cFos count",
      voxel_density: "Cell / voxel density",
      signal_voxels: "Signal voxels",
      mean_cfos_intensity: "Mean cFos intensity",
    }[metric] || metric
  );
}

function heatmapCorrelationColor(t) {
  const clamped = Math.max(-1, Math.min(1, t));
  if (clamped >= 0) return heatmapBlue(clamped);
  const mix = Math.abs(clamped);
  const r = Math.round(255 * mix + 247 * (1 - mix));
  const g = Math.round(255 * (1 - mix) + 251 * (1 - mix));
  const b = Math.round(255 * (1 - mix) + 255 * (1 - mix));
  return `rgb(${r}, ${g}, ${b})`;
}

function drawColorbar(ctx, { left, top, width, height, minValue, maxValue, colorFn, label }) {
  for (let x = 0; x < width; x += 1) {
    ctx.fillStyle = colorFn(x / Math.max(width - 1, 1));
    ctx.fillRect(left + x, top, 1, height);
  }
  ctx.strokeStyle = CHART.axis;
  ctx.lineWidth = 1;
  ctx.strokeRect(left, top, width, height);
  ctx.fillStyle = CHART.muted;
  ctx.font = CHART.smallFont;
  ctx.textAlign = "left";
  ctx.fillText(formatChartNumber(minValue), left, top - 4);
  ctx.textAlign = "right";
  ctx.fillText(formatChartNumber(maxValue), left + width, top - 4);
  ctx.textAlign = "center";
  if (label) ctx.fillText(label, left + width / 2, top + height + 14);
  ctx.textAlign = "left";
}

function heatmapBlue(t) {
  const clamped = Math.max(0, Math.min(1, t));
  const stops = [
    [0, [247, 251, 255]],
    [0.35, [198, 219, 239]],
    [0.65, [107, 174, 214]],
    [1, [8, 81, 156]],
  ];
  let lower = stops[0];
  let upper = stops[stops.length - 1];
  for (let index = 0; index < stops.length - 1; index += 1) {
    if (clamped >= stops[index][0] && clamped <= stops[index + 1][0]) {
      lower = stops[index];
      upper = stops[index + 1];
      break;
    }
  }
  const span = Math.max(upper[0] - lower[0], 1e-6);
  const mix = (clamped - lower[0]) / span;
  const rgb = lower[1].map((value, channel) =>
    Math.round(value + (upper[1][channel] - value) * mix)
  );
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

function truncateLabel(text, maxWidth, ctx) {
  const value = String(text || "");
  if (ctx.measureText(value).width <= maxWidth) return value;
  let trimmed = value;
  while (trimmed.length > 1 && ctx.measureText(`${trimmed}…`).width > maxWidth) {
    trimmed = trimmed.slice(0, -1);
  }
  return `${trimmed}…`;
}

function drawScatter(canvas, scatter, comparison, options = {}) {
  if (!canvas || !scatter?.available || !scatter.points?.length) {
    canvas._layout = null;
    return null;
  }
  const { hoverRegionId = null, activeRegionId = null } = options;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = CHART.bg;
  ctx.fillRect(0, 0, width, height);

  const points = scatter.points;
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const minVal = Math.min(...xs, ...ys, 0);
  const maxVal = Math.max(...xs, ...ys, 1);
  const leftPad = 54;
  const rightPad = 18;
  const topPad = 44;
  const bottomPad = 42;
  const plotW = width - leftPad - rightPad;
  const plotH = height - topPad - bottomPad;
  const toX = (value) => leftPad + ((value - minVal) / Math.max(maxVal - minVal, 1e-6)) * plotW;
  const toY = (value) => topPad + plotH - ((value - minVal) / Math.max(maxVal - minVal, 1e-6)) * plotH;

  ctx.strokeStyle = CHART.grid;
  ctx.lineWidth = 1;
  for (let tick = 0; tick <= 4; tick += 1) {
    const value = minVal + ((maxVal - minVal) * tick) / 4;
    ctx.beginPath();
    ctx.moveTo(toX(value), topPad);
    ctx.lineTo(toX(value), topPad + plotH);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(leftPad, toY(value));
    ctx.lineTo(leftPad + plotW, toY(value));
    ctx.stroke();
  }

  ctx.strokeStyle = CHART.axis;
  ctx.beginPath();
  ctx.moveTo(leftPad, topPad + plotH);
  ctx.lineTo(leftPad + plotW, topPad + plotH);
  ctx.moveTo(leftPad, topPad);
  ctx.lineTo(leftPad, topPad + plotH);
  ctx.stroke();

  ctx.strokeStyle = "#94a3b8";
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(toX(minVal), toY(minVal));
  ctx.lineTo(toX(maxVal), toY(maxVal));
  ctx.stroke();
  ctx.setLineDash([]);

  const hitPoints = [];
  for (const point of points) {
    const px = toX(point.x);
    const py = toY(point.y);
    const regionId = Number(point.region_id);
    const isActive = activeRegionId === regionId;
    const isHover = hoverRegionId === regionId;
    const radius = isActive ? 6 : isHover ? 5 : 3.5;
    ctx.fillStyle = isActive ? CHART.pointActive : isHover ? CHART.pointHover : CHART.point;
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = isActive || isHover ? 2 : 1;
    ctx.beginPath();
    ctx.arc(px, py, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    hitPoints.push({ regionId, px, py, radius: radius + 4, point });
  }

  ctx.fillStyle = CHART.text;
  ctx.font = CHART.titleFont;
  ctx.fillText(`${comparison.group_a} vs ${comparison.group_b}`, 14, 24);
  ctx.font = CHART.bodyFont;
  if (Number.isFinite(scatter.pearson_r)) {
    ctx.fillStyle = CHART.muted;
    const modeHint =
      scatter.mode === "group_mean"
        ? "Group means across samples · hover/click a point"
        : "Pairwise sample · hover/click a point";
    ctx.fillText(`Pearson r = ${Number(scatter.pearson_r).toFixed(3)} · ${modeHint}`, 14, 42);
  }

  ctx.fillStyle = CHART.text;
  ctx.font = CHART.smallFont;
  ctx.textAlign = "center";
  ctx.fillText(comparison.group_a || "Sample A", leftPad + plotW / 2, height - 12);
  ctx.save();
  ctx.translate(16, topPad + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(comparison.group_b || "Sample B", 0, 0);
  ctx.restore();
  ctx.textAlign = "left";

  canvas._layout = { type: "scatter", hitPoints };
  return canvas._layout;
}

function drawHeatmap(canvas, heatmap, groups, regionMetaById, options = {}) {
  if (!canvas || !heatmap?.matrix?.length) {
    canvas._layout = null;
    return null;
  }
  const { hoverRegionId = null, activeRegionId = null, metric = "cfos_count", focusRegion = null } = options;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = CHART.bg;
  ctx.fillRect(0, 0, width, height);

  const matrix = heatmap.matrix;
  const samples = heatmap.samples;
  const flat = matrix.flat();
  const minValue = Number.isFinite(heatmap.value_min) ? Number(heatmap.value_min) : Math.min(...flat, 0);
  const maxValue = Math.max(Number.isFinite(heatmap.value_max) ? Number(heatmap.value_max) : Math.max(...flat), 1e-6);
  const span = Math.max(maxValue - minValue, 1e-9);
  const leftPad = 92;
  const titleY = 24;
  const legendY = 42;
  const legendH = 14;
  const colorbarH = 16;
  const colorbarGap = 10;
  const topPad = 66;
  const bottomPad = colorbarH + colorbarGap + 22;
  const cellW = (width - leftPad - 10) / Math.max(samples.length, 1);
  const cellH = (height - topPad - bottomPad) / Math.max(matrix.length, 1);
  const hitRows = [];

  ctx.fillStyle = CHART.text;
  ctx.font = CHART.titleFont;
  const scopeLabel = focusRegion?.region_acronym
    ? `${focusRegion.region_acronym} + subregions`
    : "top regions";
  const modeLabel = heatmap.mode === "absolute" ? "raw values" : "differential";
  ctx.fillText(`Analysis heatmap (${scopeLabel}, ${modeLabel})`, 14, titleY);

  samples.forEach((sampleId, col) => {
    const group = heatmap.sample_groups[sampleId] || "";
    const x = leftPad + col * cellW;
    ctx.fillStyle = groupColor(group, groups);
    ctx.fillRect(x + 1, legendY, Math.max(cellW - 2, 4), legendH);
    ctx.strokeStyle = CHART.axis;
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 1, legendY, Math.max(cellW - 2, 4), legendH);
    ctx.fillStyle = CHART.text;
    ctx.font = CHART.smallFont;
    ctx.textAlign = "center";
    ctx.fillText(truncateLabel(sampleId, Math.max(cellW - 4, 24), ctx), x + cellW / 2, legendY - 5);
    ctx.textAlign = "left";
  });

  matrix.forEach((row, rowIndex) => {
    const regionId = Number(heatmap.region_ids?.[rowIndex] || 0);
    const rowTop = topPad + rowIndex * cellH;
    const isActive = activeRegionId === regionId;
    const isHover = hoverRegionId === regionId;
    const isParent = focusRegion && Number(focusRegion.region_id) === regionId;
    if (isActive) {
      ctx.fillStyle = CHART.rowActive;
      ctx.fillRect(0, rowTop, width, Math.max(cellH, 1));
    } else if (isHover) {
      ctx.fillStyle = CHART.rowHover;
      ctx.fillRect(0, rowTop, width, Math.max(cellH, 1));
    } else if (isParent) {
      ctx.fillStyle = "rgba(148, 163, 184, 0.12)";
      ctx.fillRect(0, rowTop, width, Math.max(cellH, 1));
    }

    row.forEach((value, col) => {
      const t = (Number(value) - minValue) / span;
      ctx.fillStyle = heatmapBlue(Math.max(0, Math.min(1, t)));
      ctx.fillRect(leftPad + col * cellW + 1, rowTop + 1, Math.max(cellW - 2, 1), Math.max(cellH - 2, 1));
    });

    ctx.fillStyle = CHART.text;
    ctx.font = isActive || isHover || isParent ? "600 12px 'Segoe UI', sans-serif" : CHART.smallFont;
    ctx.textAlign = "right";
    const meta = regionMetaById.get(regionId);
    const label = meta?.region_acronym || heatmap.region_labels[rowIndex] || "";
    ctx.fillText(truncateLabel(label, leftPad - 12, ctx), leftPad - 8, rowTop + cellH / 2 + 4);
    ctx.textAlign = "left";
    hitRows.push({ regionId, rowTop, rowBottom: rowTop + cellH, left: leftPad, right: width, meta });
  });

  const barTop = height - colorbarH - 8;
  drawColorbar(ctx, {
    left: leftPad,
    top: barTop,
    width: width - leftPad - 10,
    height: colorbarH,
    minValue,
    maxValue,
    colorFn: heatmapBlue,
    label: metricLabel(metric),
  });

  canvas._layout = { type: "heatmap", hitRows, leftPad, topPad };
  return canvas._layout;
}

function drawSampleCorrelation(canvas, correlation, groups) {
  if (!canvas || !correlation?.available || !correlation.matrix?.length) {
    if (canvas) canvas._layout = null;
    return null;
  }
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = CHART.bg;
  ctx.fillRect(0, 0, width, height);

  const samples = correlation.samples;
  const matrix = correlation.matrix;
  const n = samples.length;
  const leftPad = 108;
  const topPad = 44;
  const colorbarH = 16;
  const bottomPad = colorbarH + 28;
  const size = Math.min((width - leftPad - 20) / n, (height - topPad - bottomPad) / n);
  const gridW = size * n;

  ctx.fillStyle = CHART.text;
  ctx.font = CHART.titleFont;
  ctx.fillText("Sample correlation (region vectors)", 14, 24);
  ctx.font = CHART.smallFont;
  ctx.fillStyle = CHART.muted;
  ctx.fillText(`${correlation.n_regions || 0} regions · Pearson r`, 14, 40);

  for (let row = 0; row < n; row += 1) {
    for (let col = 0; col < n; col += 1) {
      const value = Number(matrix[row]?.[col] ?? 0);
      ctx.fillStyle = heatmapCorrelationColor(value);
      ctx.fillRect(leftPad + col * size + 1, topPad + row * size + 1, Math.max(size - 2, 1), Math.max(size - 2, 1));
    }
    ctx.fillStyle = CHART.text;
    ctx.font = CHART.smallFont;
    ctx.textAlign = "right";
    ctx.fillText(truncateLabel(samples[row], leftPad - 8, ctx), leftPad - 8, topPad + row * size + size / 2 + 4);
    ctx.textAlign = "center";
    ctx.fillText(
      truncateLabel(samples[col], Math.max(size - 4, 20), ctx),
      leftPad + col * size + size / 2,
      topPad - 6
    );
    ctx.textAlign = "left";
  }

  const barTop = topPad + gridW + 12;
  drawColorbar(ctx, {
    left: leftPad,
    top: barTop,
    width: gridW,
    height: colorbarH,
    minValue: -1,
    maxValue: 1,
    colorFn: (t) => heatmapCorrelationColor(t * 2 - 1),
    label: "Pearson r",
  });

  canvas._layout = { type: "correlation" };
  return canvas._layout;
}

function showChartTooltip(text, x, y, { html = false } = {}) {
  const tooltip = document.getElementById("chart-tooltip");
  if (!tooltip) return;
  if (!text) {
    tooltip.classList.add("hidden");
    return;
  }
  if (html) {
    tooltip.innerHTML = text;
  } else {
    tooltip.textContent = text;
  }
  tooltip.classList.remove("hidden");
  tooltip.style.left = `${x + 14}px`;
  tooltip.style.top = `${y + 14}px`;
}

function escapeTooltipText(text) {
  return String(text ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function regionTooltipText(regionId, regionMetaById) {
  const id = Number(regionId);
  const meta = regionMetaById.get(id);
  const name = meta?.region_name || "—";
  const acronym = meta?.region_acronym || "—";
  return `<strong>${escapeTooltipText(name)}</strong><div>${escapeTooltipText(acronym)} · ID ${id}</div>`;
}

function getCompareRegionMeta() {
  return buildRegionMetaMap(compareInteraction.payload);
}

function redrawCompareCharts(els, payload, options = {}) {
  const regionMetaById = buildRegionMetaMap(payload);
  const drawOpts = {
    hoverRegionId: options.hoverRegionId ?? compareInteraction.hoverRegionId,
    activeRegionId: options.activeRegionId ?? compareInteraction.activeRegionId,
    metric: payload.metric,
    focusRegion: payload.focus_region,
  };
  drawScatter(els.scatterCanvas, payload.pairwise_scatter, payload.comparison, drawOpts);
  drawHeatmap(els.heatmapCanvas, payload.heatmap, payload.groups, regionMetaById, drawOpts);
  drawSampleCorrelation(els.correlationCanvas, payload.sample_correlation, payload.groups);
  if (els.correlationWrap) {
    els.correlationWrap.classList.toggle("hidden", !payload.sample_correlation?.available);
  }
}

function findScatterHit(canvas, clientX, clientY) {
  const layout = canvas?._layout;
  if (!layout || layout.type !== "scatter") return null;
  const rect = canvas.getBoundingClientRect();
  const x = ((clientX - rect.left) / rect.width) * canvas.width;
  const y = ((clientY - rect.top) / rect.height) * canvas.height;
  let best = null;
  let bestDist = Infinity;
  for (const hit of layout.hitPoints) {
    const dist = Math.hypot(x - hit.px, y - hit.py);
    if (dist <= hit.radius && dist < bestDist) {
      best = hit;
      bestDist = dist;
    }
  }
  return best?.regionId ?? null;
}

function findHeatmapHit(canvas, clientX, clientY) {
  const layout = canvas?._layout;
  if (!layout || layout.type !== "heatmap") return null;
  const rect = canvas.getBoundingClientRect();
  const x = ((clientX - rect.left) / rect.width) * canvas.width;
  const y = ((clientY - rect.top) / rect.height) * canvas.height;
  for (const row of layout.hitRows) {
    if (x >= row.left && x <= row.right && y >= row.rowTop && y <= row.rowBottom) {
      return row.regionId;
    }
  }
  return null;
}

function attachCompareChartInteractions(els, payload, { onRegionClick, activeRegionId = null }) {
  compareInteraction.payload = payload;
  compareInteraction.onRegionClick = onRegionClick;
  compareInteraction.activeRegionId = activeRegionId;
  compareInteraction.hoverRegionId = null;

  for (const canvas of [els.scatterCanvas, els.heatmapCanvas]) {
    if (!canvas || canvas._compareBound) continue;
    canvas._compareBound = true;
    canvas.addEventListener("mousemove", (event) => {
      if (!compareInteraction.payload) return;
      const regionId =
        canvas === els.scatterCanvas
          ? findScatterHit(canvas, event.clientX, event.clientY)
          : findHeatmapHit(canvas, event.clientX, event.clientY);
      const metaMap = getCompareRegionMeta();
      if (regionId === compareInteraction.hoverRegionId) {
        if (regionId) {
          showChartTooltip(regionTooltipText(regionId, metaMap), event.clientX, event.clientY, { html: true });
        }
        return;
      }
      compareInteraction.hoverRegionId = regionId;
      redrawCompareCharts(els, compareInteraction.payload);
      if (regionId) {
        showChartTooltip(regionTooltipText(regionId, metaMap), event.clientX, event.clientY, { html: true });
        canvas.style.cursor = "pointer";
      } else {
        showChartTooltip("", 0, 0);
        canvas.style.cursor = "default";
      }
    });
    canvas.addEventListener("mouseleave", () => {
      compareInteraction.hoverRegionId = null;
      showChartTooltip("", 0, 0);
      canvas.style.cursor = "default";
      if (compareInteraction.payload) redrawCompareCharts(els, compareInteraction.payload);
    });
    canvas.addEventListener("click", (event) => {
      if (!compareInteraction.payload) return;
      const regionId =
        canvas === els.scatterCanvas
          ? findScatterHit(canvas, event.clientX, event.clientY)
          : findHeatmapHit(canvas, event.clientX, event.clientY);
      if (regionId && compareInteraction.onRegionClick) {
        compareInteraction.activeRegionId = regionId;
        compareInteraction.onRegionClick(regionId);
        redrawCompareCharts(els, compareInteraction.payload, { activeRegionId: regionId });
      }
    });
  }
}

export function syncGroupLevelOptions(els, levels, defaultLevel) {
  if (!els.levelSelect) return;
  els.levelSelect.replaceChildren();
  for (const level of levels) {
    const option = document.createElement("option");
    option.value = level;
    option.textContent = level;
    if (level === defaultLevel) option.selected = true;
    els.levelSelect.appendChild(option);
  }
}

export function renderGroupDiffTable(container, rows, onRegionClick, activeRegionId = null) {
  if (!container) return;
  if (!rows?.length) {
    container.innerHTML = "<p class='muted'>No regions to compare.</p>";
    return;
  }
  container.innerHTML = `
    <h4>Region comparison</h4>
    <table class="group-diff-table">
      <thead>
        <tr><th>Region</th><th>mean A</th><th>mean B</th><th>Δ (B−A)</th><th>log2FC</th></tr>
      </thead>
      <tbody>
        ${rows
          .map(
            (row) =>
              `<tr data-region-id="${row.region_id}" class="${Number(row.region_id) === Number(activeRegionId) ? "selected" : ""}"><td title="${row.region_name || row.region_acronym}">${row.region_acronym}</td><td>${Number(row.mean_a).toFixed(1)}</td><td>${Number(row.mean_b).toFixed(1)}</td><td>${Number(row.delta ?? row.mean_b - row.mean_a).toFixed(1)}</td><td>${Number(row.log2_fold_change).toFixed(2)}</td></tr>`
          )
          .join("")}
      </tbody>
    </table>
  `;
  container.querySelectorAll("tr[data-region-id]").forEach((row) => {
    row.addEventListener("click", () => onRegionClick(Number(row.dataset.regionId)));
  });
}

export function populateGroupSelects(els, groups, comparison) {
  if (!els.groupASelect || !els.groupBSelect) return;
  for (const select of [els.groupASelect, els.groupBSelect]) {
    select.replaceChildren();
    for (const group of groups) {
      const option = document.createElement("option");
      option.value = group;
      option.textContent = group;
      select.appendChild(option);
    }
  }
  if (comparison?.group_a) els.groupASelect.value = comparison.group_a;
  if (comparison?.group_b) els.groupBSelect.value = comparison.group_b;
  if (els.labelA && comparison?.group_a && !els.labelA.value) els.labelA.value = comparison.group_a;
  if (els.labelB && comparison?.group_b && !els.labelB.value) els.labelB.value = comparison.group_b;
}

export function resizeCompareCanvases(root = document) {
  const grid = root.querySelector(".compare-charts");
  const wraps = root.querySelectorAll(".compare-charts .chart-wrap");
  const cellWidth = Math.max(Math.floor(((grid?.clientWidth || 1040) - 12) / 2), 320);
  for (const wrap of wraps) {
    const canvas = wrap.querySelector("canvas");
    if (!canvas) continue;
    canvas.width = cellWidth;
    canvas.height = canvas.id === "group-heatmap" ? Math.max(Math.round(cellWidth * 0.82), 340) : Math.max(Math.round(cellWidth * 0.72), 320);
  }
  const corrWrap = root.getElementById("group-correlation-wrap");
  const corrCanvas = root.getElementById("group-correlation");
  if (corrCanvas && corrWrap) {
    corrCanvas.width = Math.max(corrWrap.clientWidth || grid?.clientWidth || 1040, 640);
    corrCanvas.height = 280;
  }
}

function sampleDirName(sampleDir) {
  return String(sampleDir || "")
    .split(/[/\\]/)
    .filter(Boolean)
    .pop();
}

export function renderAnalysisSamplesTable(els, samples, { onChange } = {}) {
  if (!els.samplesBody) return;
  els.samplesBody.replaceChildren();
  for (const entry of samples) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td title="${entry.sample_dir}">${sampleDirName(entry.sample_dir)}</td>
      <td><input type="text" class="analysis-group-input" value="${entry.group || ""}" placeholder="Group label" /></td>
      <td><button type="button" class="analysis-remove-sample" ${samples.length <= 2 ? "disabled" : ""}>Remove</button></td>
    `;
    const groupInput = row.querySelector(".analysis-group-input");
    groupInput?.addEventListener("change", () => {
      entry.group = groupInput.value.trim() || sampleDirName(entry.sample_dir);
      onChange?.();
    });
    row.querySelector(".analysis-remove-sample")?.addEventListener("click", () => {
      if (samples.length <= 2) return;
      const index = samples.indexOf(entry);
      if (index >= 0) samples.splice(index, 1);
      renderAnalysisSamplesTable(els, samples, { onChange });
      onChange?.();
    });
    els.samplesBody.appendChild(row);
  }
}

export function readAnalysisSamplesFromTable(els, samples) {
  const rows = els.samplesBody?.querySelectorAll("tr") || [];
  rows.forEach((row, index) => {
    const input = row.querySelector(".analysis-group-input");
    if (!samples[index]) return;
    samples[index].group = input?.value.trim() || sampleDirName(samples[index].sample_dir);
  });
  return samples;
}

export function buildAnalysisManifestFromSamples(samples, signalCh = "ch1") {
  return samples.map((entry) => ({
    sample_dir: entry.sample_dir,
    group: entry.group || sampleDirName(entry.sample_dir),
    signal_ch: entry.signal_ch || signalCh,
    sample_id: entry.sample_id || sampleDirName(entry.sample_dir),
  }));
}

export function syncGroupSampleFields(els, sampleADir, sampleBDir = "") {
  if (els.sampleA) els.sampleA.value = sampleADir || "";
  if (els.sampleB && sampleBDir) els.sampleB.value = sampleBDir;
}

export async function runGroupAnalysis(els, requestBody) {
  const response = await fetch("/api/group/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
    throw new Error(typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail));
  }
  return response.json();
}

export function renderGroupAnalysis(els, payload, { onRegionClick, activeRegionId = null }) {
  const placeholder = document.getElementById("compare-placeholder");
  const chartWrap = document.querySelector(".compare-charts");
  if (placeholder) placeholder.classList.add("hidden");
  if (chartWrap) chartWrap.classList.remove("hidden");
  resizeCompareCanvases(document);
  populateGroupSelects(els, payload.groups, payload.comparison);
  compareInteraction.activeRegionId = activeRegionId;
  redrawCompareCharts(els, payload, { activeRegionId });
  attachCompareChartInteractions(els, payload, { onRegionClick, activeRegionId });
  const tableRows = payload.region_scope_ids?.length
    ? payload.differential_regions.filter((row) =>
        payload.region_scope_ids.includes(Number(row.region_id))
      )
    : payload.top_differential_regions || payload.differential_regions;
  renderGroupDiffTable(els.diffTable, tableRows, onRegionClick, activeRegionId);
  if (els.exportButton) {
    els.exportButton.disabled = !payload.differential_regions?.length;
  }
  if (els.status) {
    els.status.classList.add("hidden");
  }
  return payload;
}

export function updateCompareChartSelection(els, activeRegionId) {
  if (!compareInteraction.payload) return;
  compareInteraction.activeRegionId = activeRegionId;
  redrawCompareCharts(els, compareInteraction.payload, { activeRegionId });
}
