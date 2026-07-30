const ALLEN_ROOT_REGION_ID = 997;

import {
  bindGroupElements,
  renderGroupAnalysis,
  buildAnalysisManifestFromSamples,
  readAnalysisSamplesFromTable,
  renderAnalysisSamplesTable,
  runGroupAnalysis,
  syncGroupLevelOptions,
  syncGroupSampleFields,
  updateCompareChartSelection,
} from "./group.js?v=34";
import { fetchSummary, renderSummaryPanel, resolveSummaryPayload } from "./summary.js?v=33";

const state = {
  bundle: null,
  sampleDir: "",
  signalCh: "ch1",
  group: "",
  selectedRegionIds: new Set(),
  activeRegionId: null,
  defaultLevel: "",
  metricsById: new Map(),
  spatialAxes: null,
  regionCentroidStats: null,
  regionDescendantsIndex: null,
  activeView: "summary",
  axisProfile: "AP",
  viewer3d: null,
  atlasShape: [456, 528, 320],
  bregmaIndex: [18, 216, 228],
  sliceRefreshTimer: null,
  sliceRefreshSeq: 0,
  highlightMemberIds: null,
  collapsedRegionIds: new Set(),
  showBrainOutline: false,
  showRegionOutlines: true,
  brainOutlinePayload: null,
  cachedRegionSurface: null,
  cachedRegionSurfaceId: null,
  regionSurfaceCache: new Map(),
  regionSubtreeCache: new Map(),
  regionSelectionSeq: 0,
  regionTreeBound: false,
  groupAnalysis: null,
  sliceBookmarks: [],
  sliceColorMode: "region",
  compareSampleDir: "",
  sliceLayout: null,
  sliceImageObjectUrl: "",
  filterInBrainOnly: true,
  loadedSamples: [],
  analysisSamples: [],
  pairSampleA: "",
  pairSampleB: "",
  summary: null,
};

const groupEls = bindGroupElements(document);

const PLANE_AXIS = {
  coronal: "AP",
  sagittal: "ML",
  horizontal: "DV",
};

const SLICE_EXPORT_MODES = [
  { id: "region", label: "Region" },
  { id: "signal", label: "Signal" },
];

function currentLevel() {
  return state.defaultLevel || state.bundle?.parameters?.default_level || "";
}

function defaultExportModes() {
  return Object.fromEntries(SLICE_EXPORT_MODES.map((mode) => [mode.id, true]));
}

const els = {
  appLayout: document.getElementById("app-layout"),
  panelLeft: document.getElementById("panel-left"),
  panelRight: document.getElementById("panel-right"),
  togglePanelLeft: document.getElementById("toggle-panel-left"),
  togglePanelRight: document.getElementById("toggle-panel-right"),
  headerSampleId: document.getElementById("header-sample-id"),
  regionSearch: document.getElementById("region-search"),
  regionTree: document.getElementById("region-tree"),
  exportSelected: document.getElementById("export-selected"),
  metricSelect: document.getElementById("metric-select"),
  planeSelect: document.getElementById("plane-select"),
  sliceIndex: document.getElementById("slice-index"),
  sliceIndexLabel: document.getElementById("slice-index-label"),
  sliceImage: document.getElementById("slice-image"),
  slicePlaceholder: document.getElementById("slice-placeholder"),
  summaryPanel: document.getElementById("summary-panel"),
  summaryView: document.getElementById("summary-view"),
  regionDetailPanel: document.getElementById("region-detail-panel"),
  regionDetail: document.getElementById("region-detail"),
  viewTabs: document.querySelectorAll(".view-tab"),
  sliceControls: document.getElementById("slice-controls"),
  sliceColorMode: document.getElementById("slice-color-mode"),
  compareSampleDir: document.getElementById("compare-sample-dir"),
  bregmaLabel: document.getElementById("bregma-label"),
  addSliceBookmark: document.getElementById("add-slice-bookmark"),
  exportSliceBookmarks: document.getElementById("export-slice-bookmarks"),
  sliceExportTableWrap: document.getElementById("slice-export-table-wrap"),
  viewer3dControls: document.getElementById("viewer3d-controls"),
  sliceView: document.getElementById("slice-view"),
  points3dView: document.getElementById("points3d-view"),
  compareView: document.getElementById("compare-view"),
  compareControls: document.getElementById("compare-controls"),
  viewer3dHost: document.getElementById("viewer3d"),
  viewer3dCanvas: document.getElementById("viewer3d-canvas"),
  viewer3dPlaceholder: document.getElementById("viewer3d-placeholder"),
  sliceNavigator: document.getElementById("slice-navigator"),
  histAxisCanvas: document.getElementById("hist-axis"),
  filterInBrainPoints: document.getElementById("filter-in-brain-points"),
  resetCamera: document.getElementById("reset-camera"),
  saveCamera: document.getElementById("save-camera"),
  savedViewsList: document.getElementById("saved-views-list"),
  screenshot3d: document.getElementById("screenshot-3d"),
  toggleBrainOutline: document.getElementById("toggle-brain-outline"),
  toggleRegionOutlines: document.getElementById("toggle-region-outlines"),
  regionColorLegend: document.getElementById("region-color-legend"),
};

function currentAxisFromPlane() {
  return PLANE_AXIS[els.planeSelect?.value || "coronal"] || "AP";
}

function parseSliceLayoutHeaders(headers) {
  const num = (key) => Number(headers.get(key) || 0);
  const layout = {
    atlas_width: num("X-Slice-Atlas-Width"),
    atlas_height: num("X-Slice-Atlas-Height"),
    image_width: num("X-Slice-Image-Width"),
    image_height: num("X-Slice-Image-Height"),
    slice_left: num("X-Slice-Left"),
    slice_top: num("X-Slice-Top"),
    slice_width: num("X-Slice-Width"),
    slice_height: num("X-Slice-Height"),
  };
  if (!layout.image_width || !layout.image_height) return null;
  return layout;
}

function imageDisplayRect(img, rect) {
  const naturalW = img.naturalWidth || rect.width;
  const naturalH = img.naturalHeight || rect.height;
  const scale = Math.min(rect.width / naturalW, rect.height / naturalH);
  const drawnW = naturalW * scale;
  const drawnH = naturalH * scale;
  return {
    offsetX: (rect.width - drawnW) / 2,
    offsetY: (rect.height - drawnH) / 2,
    scale,
    drawnW,
    drawnH,
    naturalW,
    naturalH,
  };
}

function mapSliceClickToAtlasPixels(event) {
  const layout = state.sliceLayout;
  const rect = els.sliceImage.getBoundingClientRect();
  if (!layout) {
    const display = imageDisplayRect(els.sliceImage, rect);
    const pixelX = (event.clientX - rect.left - display.offsetX) / display.scale;
    const pixelY = (event.clientY - rect.top - display.offsetY) / display.scale;
    return {
      pixel_x: pixelX,
      pixel_y: pixelY,
      image_width: display.naturalW,
      image_height: display.naturalH,
    };
  }
  const display = imageDisplayRect(els.sliceImage, rect);
  const imgX = (event.clientX - rect.left - display.offsetX) / display.scale;
  const imgY = (event.clientY - rect.top - display.offsetY) / display.scale;
  const relX = imgX - layout.slice_left;
  const relY = imgY - layout.slice_top;
  if (relX < 0 || relY < 0 || relX > layout.slice_width || relY > layout.slice_height) {
    return null;
  }
  const atlasX = (relX / Math.max(layout.slice_width, 1)) * layout.atlas_width;
  const atlasY = (relY / Math.max(layout.slice_height, 1)) * layout.atlas_height;
  return {
    pixel_x: atlasX,
    pixel_y: atlasY,
    image_width: layout.atlas_width,
    image_height: layout.atlas_height,
  };
}

function findRegionMetric(regionId, preferredLevel) {
  if (!state.bundle || !regionId) return null;
  const levels = [
    preferredLevel,
    state.defaultLevel,
  ].filter(Boolean);
  for (const level of state.bundle.levels || []) {
    if (!levels.includes(level)) levels.push(level);
  }
  for (const level of levels) {
    const metric = state.bundle.region_metrics.find(
      (row) => row.region_id === regionId && row.level === level
    );
    if (metric) return metric;
  }
  return state.bundle.region_metrics.find((row) => row.region_id === regionId) || null;
}

function findRegionTreeNode(nodes, regionId) {
  for (const node of nodes || []) {
    if (Number(node.region_id) === Number(regionId)) return node;
    const child = findRegionTreeNode(node.children, regionId);
    if (child) return child;
  }
  return null;
}

function findRegionPath(regionId) {
  if (!state.bundle) return null;
  const levels = [...(state.bundle.levels || [])].reverse();
  for (const level of levels) {
    const metric = findRegionMetric(regionId, level);
    if (metric?.structure_id_path?.length) return metric.structure_id_path;
  }
  const node = findRegionTreeNode(state.bundle.region_tree, regionId);
  return node?.structure_id_path || null;
}

function resolveDisplayRegionIdLocal(regionId) {
  const path = findRegionPath(regionId);
  if (!path?.length) return Number(regionId);
  const level = currentLevel();
  const levelNum = Number.parseInt(String(level).split("_", 2)[1], 10);
  if (!Number.isFinite(levelNum)) return Number(regionId);
  const nativeLevel = path.length - 1;
  if (nativeLevel < levelNum) return Number(path[path.length - 1]);
  if (levelNum < 0 || levelNum >= path.length) return Number(path[path.length - 1]);
  return Number(path[levelNum]);
}

function rebuildRegionDescendantsIndex() {
  state.regionDescendantsIndex = new Map();
  if (!state.bundle?.region_tree?.length) return;

  function walk(node) {
    let ids = [Number(node.region_id)];
    for (const child of node.children || []) {
      ids = ids.concat(walk(child));
    }
    state.regionDescendantsIndex.set(Number(node.region_id), ids);
    return ids;
  }

  for (const root of state.bundle.region_tree) {
    walk(root);
  }
}

function collectSubtreeRegionIds(regionId) {
  const cached = state.regionDescendantsIndex?.get(Number(regionId));
  if (cached?.length) return cached;
  return [Number(regionId)];
}

function computeSubtreeCentroid(regionIds) {
  const regions = state.regionCentroidStats?.regions;
  if (!regions) return null;
  let totalCount = 0;
  let sumDv = 0;
  let sumAp = 0;
  let sumMl = 0;
  for (const regionId of regionIds) {
    const row = regions[String(regionId)];
    if (!row?.count) continue;
    totalCount += Number(row.count);
    sumDv += Number(row.sum_dv);
    sumAp += Number(row.sum_ap);
    sumMl += Number(row.sum_ml);
  }
  if (totalCount <= 0) return null;
  return {
    index_dv: sumDv / totalCount,
    index_ap: sumAp / totalCount,
    index_ml: sumMl / totalCount,
  };
}

function applyRegionSliceFocus(focus) {
  if (!focus) return false;
  const apIndex = Math.round(Number(focus.recommended_index_ap ?? focus.coordinate));
  if (!Number.isFinite(apIndex)) return false;
  els.planeSelect.value = focus.recommended_plane || focus.plane || "coronal";
  updateSliceSliderForPlane();
  els.sliceIndex.value = String(apIndex);
  els.sliceIndexLabel.textContent = String(apIndex);
  syncSliceBregmaLabel();
  scheduleSliceRefresh(true);
  if (state.activeView === "slice2d") {
    renderAxisHistograms({ axis: "AP", coordinate: apIndex });
  }
  return true;
}

function jumpToRegionSliceLocal(regionId) {
  if (!state.regionCentroidStats?.available) return false;
  const rid = Number(regionId);
  // Root's ontology subtree is the whole brain; jump using root's own voxels only.
  const memberIds =
    rid === ALLEN_ROOT_REGION_ID ? [ALLEN_ROOT_REGION_ID] : collectSubtreeRegionIds(rid);
  const centroid = computeSubtreeCentroid(memberIds);
  if (!centroid) return false;
  return applyRegionSliceFocus({
    recommended_plane: "coronal",
    recommended_index_ap: centroid.index_ap,
    coordinate: centroid.index_ap,
  });
}

async function resolveDisplayRegionId(regionId) {
  if (!state.sampleDir || !regionId) return regionId;
  if (state.bundle) return resolveDisplayRegionIdLocal(regionId);
  const level = currentLevel();
  const params = new URLSearchParams({
    sample_dir: state.sampleDir,
    signal_ch: state.signalCh,
    region_id: String(regionId),
  });
  if (level) params.set("level", level);
  try {
    const response = await fetch(`/api/region/resolve-display?${params.toString()}`);
    if (!response.ok) return regionId;
    const payload = await response.json();
    return Number(payload.display_region_id || regionId);
  } catch {
    return regionId;
  }
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (Math.abs(value) >= 1000) return Number(value).toLocaleString(undefined, { maximumFractionDigits: 1 });
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function metricsForLevel(level) {
  if (!state.bundle) return [];
  return state.bundle.region_metrics.filter((row) => row.level === level);
}

function rebuildMetricsIndex(level) {
  state.metricsById = new Map();
  for (const row of metricsForLevel(level)) {
    state.metricsById.set(row.region_id, row);
  }
}

function nodeMatchesSearch(node, query) {
  if (!query) return true;
  const q = query.toLowerCase();
  return (
    String(node.region_id).includes(q) ||
    node.region_name.toLowerCase().includes(q) ||
    node.region_acronym.toLowerCase().includes(q)
  );
}

function filterTreeNodes(nodes, query) {
  const filtered = [];
  for (const node of nodes) {
    const children = filterTreeNodes(node.children || [], query);
    const selfMatch = nodeMatchesSearch(node, query);
    if (selfMatch || children.length) {
      filtered.push({ ...node, children });
    }
  }
  return filtered;
}

function collectParentRegionIds(nodes) {
  const ids = new Set();
  for (const node of nodes) {
    if (node.children?.length) {
      ids.add(node.region_id);
      for (const id of collectParentRegionIds(node.children)) {
        ids.add(id);
      }
    }
  }
  return ids;
}

function clearRegionSelection() {
  state.activeRegionId = null;
  state.selectedRegionIds.clear();
  state.highlightMemberIds = null;
  state.showBrainOutline = false;
  state.cachedRegionSurface = null;
  state.cachedRegionSurfaceId = null;
  els.regionDetail?.classList.add("hidden");
  els.regionDetailPanel?.querySelector(".muted")?.classList.remove("hidden");
  syncBrainOutlineButton();
  syncRegionOutlinesButton();
  renderRegionColorLegend([]);
  if (state.viewer3d) {
    state.viewer3d.clearRegionFocus();
  }
  renderRegionTree();
  updateExportButton();
  if (state.bundle && state.activeView === "slice2d") {
    scheduleSliceRefresh(true);
  }
}

function initRegionTreeInteractions() {
  if (state.regionTreeBound || !els.regionTree) return;
  state.regionTreeBound = true;

  els.regionTree.addEventListener("click", (event) => {
    const toggle = event.target.closest(".tree-toggle[data-region-id]");
    if (toggle) {
      event.preventDefault();
      event.stopPropagation();
      const regionId = Number(toggle.dataset.regionId);
      if (state.collapsedRegionIds.has(regionId)) {
        state.collapsedRegionIds.delete(regionId);
      } else {
        state.collapsedRegionIds.add(regionId);
      }
      renderRegionTree();
      return;
    }

    const item = event.target.closest(".tree-item[data-region-id]");
    if (!item) return;
    event.stopPropagation();
    const regionId = Number(item.dataset.regionId);
    if (event.shiftKey || event.ctrlKey || event.metaKey) {
      void toggleMultiRegionSelection(regionId);
      return;
    }
    toggleRegionSelection(regionId);
  });
}

function renderTreeNode(node) {
  const metric = state.metricsById.get(node.region_id);
  const children = node.children || [];
  const hasChildren = children.length > 0;
  const isSearching = Boolean(els.regionSearch.value.trim());
  const isCollapsed = !isSearching && state.collapsedRegionIds.has(node.region_id);

  const row = document.createElement("div");
  row.className = "tree-row";

  if (hasChildren) {
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "tree-toggle";
    toggle.dataset.regionId = String(node.region_id);
    toggle.setAttribute("aria-label", isCollapsed ? "Expand region branch" : "Collapse region branch");
    toggle.textContent = isCollapsed ? "▸" : "▾";
    row.appendChild(toggle);
  } else {
    const spacer = document.createElement("span");
    spacer.className = "tree-toggle-spacer";
    spacer.setAttribute("aria-hidden", "true");
    row.appendChild(spacer);
  }

  const item = document.createElement("div");
  item.className = "tree-item";
  item.dataset.regionId = String(node.region_id);
  if (state.selectedRegionIds.has(node.region_id)) item.classList.add("selected");
  if (state.activeRegionId === node.region_id) item.classList.add("active");

  const selectedIndex = [...state.selectedRegionIds].indexOf(node.region_id);
  if (selectedIndex >= 0) {
    const swatch = document.createElement("span");
    swatch.className = "tree-color-swatch";
    const palette = [
      "#c44e52",
      "#3182bd",
      "#31a354",
      "#e6550d",
      "#756bb1",
      "#17becf",
      "#d6616b",
      "#8c6d31",
      "#7b4173",
      "#637939",
    ];
    swatch.style.background = palette[selectedIndex % palette.length];
    item.appendChild(swatch);
  }

  const acronym = document.createElement("span");
  acronym.className = "acronym";
  acronym.textContent = node.region_acronym || `#${node.region_id}`;

  const name = document.createElement("span");
  name.className = "tree-name";
  name.textContent = node.region_name;

  const count = document.createElement("span");
  count.className = "count";
  count.textContent = metric ? formatNumber(metric.cfos_count) : "—";

  item.append(acronym, name, count);
  row.appendChild(item);

  const wrapper = document.createElement("div");
  wrapper.className = "tree-node";
  wrapper.appendChild(row);

  if (hasChildren && !isCollapsed) {
    const childWrap = document.createElement("div");
    childWrap.className = "tree-children";
    for (const child of children) {
      childWrap.appendChild(renderTreeNode(child));
    }
    wrapper.appendChild(childWrap);
  }
  return wrapper;
}

function syncBrainOutlineButton() {
  const btn = els.toggleBrainOutline;
  if (!btn) return;
  const enabled = state.selectedRegionIds.size > 0;
  btn.disabled = !enabled;
  btn.classList.toggle("active", enabled && state.showBrainOutline);
  btn.textContent = enabled
    ? state.showBrainOutline
      ? "Brain outline: ON"
      : "Brain outline: OFF"
    : "Brain outline";
  btn.setAttribute("aria-pressed", enabled && state.showBrainOutline ? "true" : "false");
}

function syncRegionOutlinesButton() {
  const btn = els.toggleRegionOutlines;
  if (!btn) return;
  const enabled = state.selectedRegionIds.size > 0;
  btn.disabled = !enabled;
  btn.classList.toggle("active", enabled && state.showRegionOutlines);
  btn.textContent = enabled
    ? state.showRegionOutlines
      ? "Region outlines: ON"
      : "Region outlines: OFF"
    : "Region outlines";
  btn.setAttribute("aria-pressed", enabled && state.showRegionOutlines ? "true" : "false");
}

function hexCssColor(hex) {
  return `#${Number(hex).toString(16).padStart(6, "0")}`;
}

function renderRegionColorLegend(selections) {
  const legend = els.regionColorLegend;
  if (!legend) return;
  if (!selections?.length) {
    legend.classList.add("hidden");
    legend.innerHTML = "";
    return;
  }
  legend.classList.remove("hidden");
  legend.innerHTML = selections
    .map((entry) => {
      const node = findRegionTreeNode(state.bundle?.region_tree, entry.regionId);
      const label = node?.region_acronym || `#${entry.regionId}`;
      return `<span class="region-legend-item"><i style="background:${hexCssColor(entry.color)}"></i>${label}</span>`;
    })
    .join("");
}

async function toggleBrainOutline() {
  if (!state.selectedRegionIds.size || !state.viewer3d) return;
  state.showBrainOutline = !state.showBrainOutline;
  syncBrainOutlineButton();
  if (state.showBrainOutline) {
    if (!state.brainOutlinePayload) {
      state.brainOutlinePayload = await fetchBrainOutlineSurface();
    }
    state.viewer3d.updateBrainOutline(state.brainOutlinePayload);
  } else {
    state.viewer3d.clearBrainOutlineSurface();
  }
}

async function toggleRegionOutlines() {
  if (!state.selectedRegionIds.size || !state.viewer3d) return;
  state.showRegionOutlines = !state.showRegionOutlines;
  syncRegionOutlinesButton();
  state.viewer3d.setRegionOutlinesVisible(state.showRegionOutlines);
}

function selectRegion(regionId) {
  void selectRegionCore(regionId);
}

async function toggleMultiRegionSelection(regionId) {
  if (!regionId) return;
  const seq = ++state.regionSelectionSeq;
  const displayRegionId = state.bundle ? resolveDisplayRegionIdLocal(regionId) : regionId;
  if (seq !== state.regionSelectionSeq) return;

  if (state.selectedRegionIds.has(displayRegionId)) {
    state.selectedRegionIds.delete(displayRegionId);
  } else {
    state.selectedRegionIds.add(displayRegionId);
  }

  if (state.selectedRegionIds.size === 0) {
    clearRegionSelection();
    return;
  }

  state.activeRegionId = displayRegionId;
  if (!state.selectedRegionIds.has(state.activeRegionId)) {
    state.activeRegionId = [...state.selectedRegionIds][state.selectedRegionIds.size - 1];
  }
  syncBrainOutlineButton();
  syncRegionOutlinesButton();
  renderRegionDetail(state.activeRegionId);
  renderRegionTree();
  updateExportButton();
  void applySelectedRegionsFocus3D(seq);
  showRegionOnSlice(state.activeRegionId);
}

async function selectRegionCore(regionId) {
  if (!regionId) return;
  const seq = ++state.regionSelectionSeq;
  const displayRegionId = state.bundle ? resolveDisplayRegionIdLocal(regionId) : regionId;
  if (seq !== state.regionSelectionSeq) return;

  state.selectedRegionIds = new Set([displayRegionId]);
  state.activeRegionId = displayRegionId;
  state.showBrainOutline = true;
  syncBrainOutlineButton();
  syncRegionOutlinesButton();
  renderRegionDetail(displayRegionId);
  void applySelectedRegionsFocus3D(seq);
  showRegionOnSlice(displayRegionId);
  renderRegionTree();
  updateExportButton();
  if (state.groupAnalysis) {
    updateCompareChartSelection(groupEls, displayRegionId);
    renderGroupAnalysis(groupEls, state.groupAnalysis, {
      onRegionClick: (id) => toggleRegionSelection(id),
      activeRegionId: displayRegionId,
    });
  }
}

async function fetchRegionSliceFocus(regionId) {
  const params = new URLSearchParams({
    sample_dir: state.sampleDir,
    signal_ch: state.signalCh,
    region_id: String(regionId),
    plane: "coronal",
  });
  const response = await fetch(`/api/region/slice-focus?${params.toString()}`);
  if (!response.ok) return null;
  return response.json();
}

async function jumpToRegionSlice(regionId) {
  if (!state.bundle || !regionId) return false;
  if (jumpToRegionSliceLocal(regionId)) return true;
  const focus = await fetchRegionSliceFocus(regionId);
  if (!focus?.available) return false;
  return applyRegionSliceFocus(focus);
}

function showRegionOnSlice(regionId) {
  // Highlight must not depend on jump succeeding: paint outline on the current
  // slice immediately, then optionally move the camera/slice index.
  setActiveView("slice2d");
  scheduleSliceRefresh(true);
  void jumpToRegionSlice(regionId).then((moved) => {
    if (!moved) {
      // Jump failed; keep the immediate highlight refresh already issued.
      scheduleSliceRefresh(true);
    }
  });
}

function toggleRegionSelection(regionId) {
  if (state.activeRegionId === regionId) {
    clearRegionSelection();
    return;
  }
  selectRegion(regionId);
}

function renderRegionTree() {
  if (!state.bundle) {
    els.regionTree.innerHTML = "<p class='muted'>No sample loaded.</p>";
    return;
  }

  const level = currentLevel();
  rebuildMetricsIndex(level);

  const filtered = filterTreeNodes(state.bundle.region_tree, els.regionSearch.value.trim());

  els.regionTree.replaceChildren();
  if (!filtered.length) {
    els.regionTree.innerHTML = "<p class='muted'>No regions match the current filters.</p>";
    return;
  }
  for (const node of filtered) {
    els.regionTree.appendChild(renderTreeNode(node));
  }
  initRegionTreeInteractions();
}

function renderSummary() {
  if (!els.summaryPanel) return;
  const summary = state.summary || (state.bundle ? resolveSummaryPayload(state.bundle) : null);
  if (!summary) return;
  renderSummaryPanel(els.summaryPanel, summary, {
    formatNumber,
    onRegionClick: toggleRegionSelection,
  });
}

async function refreshSummary({ refresh = false } = {}) {
  if (!state.sampleDir) return;
  try {
    state.summary = await fetchSummary("", {
      sampleDir: state.sampleDir,
      signalCh: state.signalCh,
      level: currentLevel(),
      refresh,
    });
  } catch {
    state.summary = resolveSummaryPayload(state.bundle);
  }
  renderSummary();
}

function renderRegionDetail(regionId) {
  if (!state.bundle || !regionId) {
    els.regionDetail?.classList.add("hidden");
    els.regionDetailPanel?.querySelector(".muted")?.classList.remove("hidden");
    return;
  }
  const level = currentLevel();
  const metric = findRegionMetric(regionId, level);
  const treeNode = !metric ? findRegionTreeNode(state.bundle.region_tree, regionId) : null;
  if (!metric && !treeNode) {
    els.regionDetail?.classList.add("hidden");
    els.regionDetailPanel?.querySelector(".muted")?.classList.remove("hidden");
    return;
  }

  els.regionDetail?.classList.remove("hidden");
  els.regionDetailPanel?.querySelector(".muted")?.classList.add("hidden");
  const systems = state.bundle.system_metrics.filter((item) =>
    item.member_region_ids.includes(regionId)
  );

  const rows = metric
    ? [
        ["Region", `${metric.region_name} (${metric.region_acronym})`],
        ["Allen ID", metric.region_id],
        ["cFos count", formatNumber(metric.cfos_count)],
        ["Signal voxels", formatNumber(metric.signal_voxels)],
        ["Voxel density", formatNumber(metric.voxel_density)],
        ["Region volume (voxels)", formatNumber(metric.region_volume_voxels)],
        ["Mean cFos intensity", formatNumber(metric.mean_cfos_intensity)],
        ["Rank by count", metric.rank_by_count],
        ["Rank by density", metric.rank_by_density],
      ]
    : [
        ["Region", `${treeNode.region_name} (${treeNode.region_acronym})`],
        ["Allen ID", treeNode.region_id],
        ["cFos count", "—"],
        ["Note", "No metrics at the current analysis level for this structure."],
      ];

  if (metric?.has_hemisphere) {
    rows.push(
      ["Left cFos count", formatNumber(metric.left_cfos_count)],
      ["Right cFos count", formatNumber(metric.right_cfos_count)],
      ["Count laterality index", formatNumber(metric.count_laterality_index)],
      ["Density laterality index", formatNumber(metric.density_laterality_index)]
    );
  }

  if (systems.length) {
    rows.push(["System membership", systems.map((item) => item.system_name).join(", ")]);
  }

  els.regionDetail.replaceChildren();
  for (const [label, value] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value;
    els.regionDetail.append(dt, dd);
  }
}

function atlasResolutionUm() {
  const values = state.bundle?.parameters?.atlas_resolution_um_dv_ap_ml;
  return Array.isArray(values) && values.length === 3 ? values : [25, 25, 25];
}

function indexToBregmaMm(axisName, index) {
  const resolution = atlasResolutionUm();
  const axisIdx = axisName === "DV" ? 0 : axisName === "AP" ? 1 : 2;
  const bregma = state.bregmaIndex[axisIdx];
  const resUm = Number(resolution[axisIdx]) || 25;
  if (axisName === "AP") {
    return ((bregma - Number(index)) * resUm) / 1000;
  }
  return ((Number(index) - bregma) * resUm) / 1000;
}

function bregmaMmForCurrentSlice() {
  const plane = els.planeSelect?.value || "coronal";
  const axis = PLANE_AXIS[plane];
  const index = Number(els.sliceIndex?.value);
  return indexToBregmaMm(axis, index);
}

function syncSliceBregmaLabel() {
  if (!els.bregmaLabel) return;
  const plane = els.planeSelect?.value || "coronal";
  const axis = PLANE_AXIS[plane];
  const index = Number(els.sliceIndex?.value);
  const bregmaMm = bregmaMmForCurrentSlice();
  const sign = bregmaMm >= 0 ? "+" : "";
  els.bregmaLabel.textContent = `${axis} index ${Math.round(index)} · Bregma ${sign}${bregmaMm.toFixed(2)} mm`;
}

function updateSliceSliderForPlane() {
  const plane = els.planeSelect.value;
  const axis = PLANE_AXIS[plane];
  const ranges = state.bundle?.parameters?.slice_ranges?.[axis];
  const bregma = state.bregmaIndex[axis === "DV" ? 0 : axis === "AP" ? 1 : 2];
  const min = ranges ? ranges[0] : 0;
  const max = ranges ? ranges[1] : 455;
  const current = Number(els.sliceIndex.value);
  const next = Number.isFinite(current) ? Math.min(Math.max(current, min), max) : bregma;
  els.sliceIndex.min = String(min);
  els.sliceIndex.max = String(max);
  els.sliceIndex.value = String(next);
  els.sliceIndexLabel.textContent = String(next);
  syncSliceBregmaLabel();
}

function goToSliceFromAxis(axis, binCenterIndex) {
  const planeMap = { AP: "coronal", ML: "sagittal", DV: "horizontal" };
  els.planeSelect.value = planeMap[axis];
  updateSliceSliderForPlane();
  els.sliceIndex.value = String(Math.round(binCenterIndex));
  els.sliceIndexLabel.textContent = String(Math.round(binCenterIndex));
  syncSliceBregmaLabel();
  setActiveView("slice2d");
  scheduleSliceRefresh(true);
}

async function fetchRegionAtSliceClick(event) {
  if (!state.bundle || !els.sliceImage?.src) return;
  const colorMode = els.sliceColorMode?.value || state.sliceColorMode || "region";
  if (compareColorModes().has(colorMode)) return;
  const mapped = mapSliceClickToAtlasPixels(event);
  if (!mapped) return;
  const level = currentLevel();
  const params = new URLSearchParams({
    sample_dir: state.sampleDir,
    signal_ch: state.signalCh,
    plane: els.planeSelect.value,
    coordinate_system: "index",
    coordinate: els.sliceIndex.value,
    pixel_x: String(mapped.pixel_x),
    pixel_y: String(mapped.pixel_y),
    image_width: String(mapped.image_width),
    image_height: String(mapped.image_height),
  });
  if (level) params.set("level", level);
  const response = await fetch(`/api/region/at-slice?${params.toString()}`);
  if (!response.ok) return;
  const payload = await response.json();
  const regionId = Number(payload.display_region_id || payload.region_id || 0);
  if (regionId > 0) selectRegion(regionId);
}

function attachSliceImageInteractions() {
  if (!els.sliceImage || els.sliceImage._regionPickBound) return;
  els.sliceImage._regionPickBound = true;
  els.sliceImage.style.cursor = "crosshair";
  els.sliceImage.addEventListener("click", (event) => {
    void fetchRegionAtSliceClick(event);
  });
}

function drawAxisLineChart(axis, histogram, selectedIndex = null, hoverIndex = null) {
  const canvas = els.histAxisCanvas;
  if (!canvas || !histogram?.counts?.length) return;
  const wrap = canvas.parentElement;
  const width = Math.max(wrap?.clientWidth || 960, 480);
  canvas.width = width;
  canvas.height = 88;
  const ctx = canvas.getContext("2d");
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(0, 0, width, height);

  const counts = histogram.counts;
  const maxCount = Math.max(...counts, 1);
  const leftPad = 40;
  const rightPad = 12;
  const topPad = 24;
  const bottomPad = 18;
  const plotW = width - leftPad - rightPad;
  const plotH = height - topPad - bottomPad;
  const stepX = plotW / Math.max(counts.length - 1, 1);

  ctx.strokeStyle = "rgba(100, 116, 139, 0.25)";
  for (let tick = 0; tick <= 4; tick += 1) {
    const y = topPad + plotH - (plotH * tick) / 4;
    ctx.beginPath();
    ctx.moveTo(leftPad, y);
    ctx.lineTo(leftPad + plotW, y);
    ctx.stroke();
  }

  ctx.strokeStyle = "#64748b";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  counts.forEach((count, index) => {
    const x = leftPad + index * stepX;
    const y = topPad + plotH - (count / maxCount) * plotH;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = "rgba(37, 99, 235, 0.12)";
  ctx.beginPath();
  counts.forEach((count, index) => {
    const x = leftPad + index * stepX;
    const y = topPad + plotH - (count / maxCount) * plotH;
    if (index === 0) ctx.moveTo(x, topPad + plotH);
    ctx.lineTo(x, y);
  });
  ctx.lineTo(leftPad + (counts.length - 1) * stepX, topPad + plotH);
  ctx.closePath();
  ctx.fill();

  if (selectedIndex !== null) {
    const selIdx = histogram.bin_centers_index.findIndex(
      (value, index) =>
        Math.abs(value - selectedIndex) <
        (histogram.bin_edges[1] - histogram.bin_edges[0]) * 0.6
    );
    if (selIdx >= 0) {
      const x = leftPad + selIdx * stepX;
      ctx.strokeStyle = "#2563eb";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, topPad);
      ctx.lineTo(x, topPad + plotH);
      ctx.stroke();
    }
  }

  if (hoverIndex !== null && hoverIndex >= 0 && hoverIndex < counts.length) {
    const x = leftPad + hoverIndex * stepX;
    const y = topPad + plotH - (counts[hoverIndex] / maxCount) * plotH;
    ctx.fillStyle = "#2563eb";
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#1e4d8c";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, topPad);
    ctx.lineTo(x, topPad + plotH);
    ctx.stroke();
  }

  ctx.fillStyle = "#1a2332";
  ctx.font = "600 12px 'Segoe UI', sans-serif";
  const sourceLabel =
    histogram.source === "volume" ? "atlas volume" : histogram.source === "points" ? "sampled points" : histogram.source;
  ctx.fillText(`${axis} cFos profile (${sourceLabel})`, leftPad, 20);
  ctx.font = "12px 'Segoe UI', sans-serif";
  ctx.fillStyle = "#475569";
  ctx.fillText(`total ${Math.round(histogram.total || 0).toLocaleString()}`, leftPad, height - 8);
  if (axis === "AP" && histogram.bin_centers_bregma_mm?.length) {
    const mid = Math.floor(histogram.bin_centers_bregma_mm.length / 2);
    const bregmaMid = histogram.bin_centers_bregma_mm[mid];
    if (Number.isFinite(bregmaMid)) {
      ctx.fillText(`~${bregmaMid >= 0 ? "+" : ""}${bregmaMid.toFixed(1)} mm bregma`, width - 120, height - 8);
    }
  }

  canvas._axisLayout = { axis, histogram, leftPad, topPad, plotW, plotH, stepX, countsLength: counts.length };
}

function showAxisTooltip(event, layout, binIndex) {
  const tooltip = document.getElementById("chart-tooltip");
  if (!tooltip || !layout?.histogram) return;
  const histogram = layout.histogram;
  const count = histogram.counts[binIndex] ?? 0;
  const indexValue = histogram.bin_centers_index?.[binIndex];
  const bregma =
    layout.axis === "AP" && histogram.bin_centers_bregma_mm?.[binIndex] !== undefined
      ? histogram.bin_centers_bregma_mm[binIndex]
      : null;
  const bregmaLine =
    bregma !== null && Number.isFinite(bregma)
      ? `<div>Bregma ${bregma >= 0 ? "+" : ""}${Number(bregma).toFixed(2)} mm</div>`
      : "";
  tooltip.innerHTML = `<strong>${layout.axis} index ${Math.round(indexValue)}</strong><div>cFos count ${Math.round(count).toLocaleString()}</div>${bregmaLine}`;
  tooltip.classList.remove("hidden");
  tooltip.style.left = `${event.clientX + 12}px`;
  tooltip.style.top = `${event.clientY + 12}px`;
}

function hideAxisTooltip() {
  document.getElementById("chart-tooltip")?.classList.add("hidden");
}

function renderAxisHistograms(selected = null) {
  if (!state.spatialAxes?.available || state.activeView !== "slice2d") {
    els.sliceNavigator?.classList.add("hidden");
    return;
  }
  els.sliceNavigator?.classList.remove("hidden");
  const axis = currentAxisFromPlane();
  const histogram = state.spatialAxes.axes[axis];
  const selectedIndex =
    selected?.axis === axis
      ? selected.coordinate
      : Number(els.sliceIndex?.value || state.bregmaIndex?.[axis === "DV" ? 0 : axis === "AP" ? 1 : 2] || 0);
  drawAxisLineChart(axis, histogram, selectedIndex);

  const canvas = els.histAxisCanvas;
  if (!canvas) return;
  let hoverIndex = null;
  canvas.onmousemove = (event) => {
    const layout = canvas._axisLayout;
    if (!layout) return;
    const rect = canvas.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * canvas.width;
    const binIndex = Math.min(
      layout.countsLength - 1,
      Math.max(0, Math.round((x - layout.leftPad) / Math.max(layout.stepX, 1e-6)))
    );
    if (binIndex !== hoverIndex) {
      hoverIndex = binIndex;
      drawAxisLineChart(
        layout.axis,
        layout.histogram,
        selected?.axis === layout.axis ? selected.coordinate : null,
        hoverIndex
      );
    }
    showAxisTooltip(event, layout, binIndex);
  };
  canvas.onmouseleave = () => {
    hoverIndex = null;
    hideAxisTooltip();
    const layout = canvas._axisLayout;
    if (layout) {
      drawAxisLineChart(
        layout.axis,
        layout.histogram,
        selected?.axis === layout.axis ? selected.coordinate : null,
        null
      );
    }
  };
  canvas.onclick = (event) => {
    const layout = canvas._axisLayout;
    if (!layout) return;
    const rect = canvas.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * canvas.width;
    const binIndex = Math.min(
      layout.countsLength - 1,
      Math.max(0, Math.round((x - layout.leftPad) / Math.max(layout.stepX, 1e-6)))
    );
    const coordinate = layout.histogram.bin_centers_index[binIndex];
    goToSliceFromAxis(layout.axis, coordinate);
    renderAxisHistograms({ axis: layout.axis, coordinate });
  };
}

async function loadRegionCentroids() {
  if (state.regionCentroidStats?.available || !state.sampleDir) return;
  if (state.bundle?.parameters?.atlas_label_available === false) return;
  const params = new URLSearchParams({
    sample_dir: state.sampleDir,
    signal_ch: state.signalCh,
  });
  try {
    const response = await fetch(`/api/spatial/region-centroids?${params.toString()}`);
    if (!response.ok) return;
    state.regionCentroidStats = await response.json();
  } catch {
    state.regionCentroidStats = null;
  }
}

async function loadSpatialData() {
  if (!state.bundle?.spatial?.available) {
    state.spatialAxes = null;
    renderAxisHistograms();
    return;
  }
  const params = new URLSearchParams({
    sample_dir: state.sampleDir,
    signal_ch: state.signalCh,
  });
  const response = await fetch(`/api/spatial/axes?${params.toString()}`);
  if (!response.ok) {
    state.spatialAxes = null;
    renderAxisHistograms();
    return;
  }
  state.spatialAxes = await response.json();
  renderAxisHistograms();
}

async function ensureViewer3d() {
  if (!state.viewer3d) {
    try {
      const module = await import("./view3d.js?v=33");
      state.viewer3d = new module.PointsViewer3D(els.viewer3dCanvas);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      els.viewer3dPlaceholder.style.display = "block";
      els.viewer3dPlaceholder.textContent = `3D viewer unavailable (${message}). 2D report still works.`;
      return null;
    }
  }
  state.viewer3d.onRegionPick = (regionId) => selectRegion(regionId);
  return state.viewer3d;
}

async function load3DPoints() {
  const viewer = await ensureViewer3d();
  if (!viewer) return;
  await new Promise((resolve) => requestAnimationFrame(resolve));
  viewer.resize();
  els.viewer3dPlaceholder.classList.add("is-blocking");
  els.viewer3dPlaceholder.style.display = "flex";
  els.viewer3dPlaceholder.textContent = "Loading 3D points…";

  const params = new URLSearchParams({
    sample_dir: state.sampleDir,
    signal_ch: state.signalCh,
    max_points: "25000",
    in_brain_only: state.filterInBrainOnly ? "true" : "false",
  });
  try {
    const [pointsResponse, rootSurfaceResponse] = await Promise.all([
      fetch(`/api/spatial/points?${params.toString()}`),
      fetch(
        `/api/spatial/region-surface?${new URLSearchParams({
          sample_dir: state.sampleDir,
          signal_ch: state.signalCh,
          region_id: String(ALLEN_ROOT_REGION_ID),
        }).toString()}`
      ),
    ]);
    if (!pointsResponse.ok) {
      els.viewer3dPlaceholder.textContent = `Failed to load 3D points (${pointsResponse.status}).`;
      return;
    }
    const payload = await pointsResponse.json();
    const rootSurface = rootSurfaceResponse.ok ? await rootSurfaceResponse.json() : null;
    const loaded = viewer.loadPayload(payload, state.bundle.sample.sample_id, rootSurface);
    if (!loaded) {
      els.viewer3dPlaceholder.textContent = "3D point view requires points.csv or atlas volume TIFF.";
      return;
    }
    els.viewer3dPlaceholder.style.display = "none";
    els.viewer3dPlaceholder.classList.remove("is-blocking");
    renderSavedViews();
    if (state.selectedRegionIds.size) {
      await applySelectedRegionsFocus3D();
    }
    await new Promise((resolve) => requestAnimationFrame(resolve));
    viewer.resize();
  } catch (error) {
    els.viewer3dPlaceholder.style.display = "flex";
    els.viewer3dPlaceholder.classList.add("is-blocking");
    els.viewer3dPlaceholder.textContent =
      error instanceof Error ? error.message : "Failed to load 3D viewer.";
  }
}

async function fetchRegionSubtree(regionId) {
  const cached = state.regionSubtreeCache.get(Number(regionId));
  if (cached) return cached;
  const params = new URLSearchParams({
    sample_dir: state.sampleDir,
    signal_ch: state.signalCh,
    region_id: String(regionId),
  });
  const response = await fetch(`/api/region/subtree?${params.toString()}`);
  if (!response.ok) {
    const fallback = [regionId];
    state.regionSubtreeCache.set(Number(regionId), fallback);
    return fallback;
  }
  const payload = await response.json();
  const members = payload.member_region_ids || [regionId];
  state.regionSubtreeCache.set(Number(regionId), members);
  return members;
}

async function fetchRegionSurface(regionId) {
  const cached = state.regionSurfaceCache.get(Number(regionId));
  if (cached) return cached;
  const params = new URLSearchParams({
    sample_dir: state.sampleDir,
    signal_ch: state.signalCh,
    region_id: String(regionId),
  });
  const response = await fetch(`/api/spatial/region-surface?${params.toString()}`);
  if (!response.ok) return null;
  const payload = await response.json();
  state.regionSurfaceCache.set(Number(regionId), payload);
  state.cachedRegionSurface = payload;
  state.cachedRegionSurfaceId = Number(regionId);
  return payload;
}

async function fetchBrainOutlineSurface() {
  const params = new URLSearchParams({
    sample_dir: state.sampleDir,
    signal_ch: state.signalCh,
  });
  const response = await fetch(`/api/spatial/brain-outline-surface?${params.toString()}`);
  if (!response.ok) return null;
  return response.json();
}

async function applySelectedRegionsFocus3D(seq = state.regionSelectionSeq) {
  const selectedIds = [...state.selectedRegionIds];
  if (!state.bundle || !selectedIds.length) return;

  if (!state.viewer3d?.fullPayload) {
    if (state.activeView === "points3d") {
      await load3DPoints();
    }
    return;
  }

  const { regionFocusColor } = await import("./view3d.js?v=33");
  if (seq !== state.regionSelectionSeq) return;

  const selections = await Promise.all(
    selectedIds.map(async (regionId, index) => {
      const [memberIds, surface] = await Promise.all([
        fetchRegionSubtree(regionId),
        fetchRegionSurface(regionId),
      ]);
      return {
        regionId: Number(regionId),
        memberIds,
        color: regionFocusColor(index),
        surface,
      };
    })
  );
  if (seq !== state.regionSelectionSeq) return;

  state.highlightMemberIds = selections.flatMap((entry) => entry.memberIds);

  let brainOutline = null;
  if (state.showBrainOutline) {
    if (!state.brainOutlinePayload) {
      state.brainOutlinePayload = await fetchBrainOutlineSurface();
      if (seq !== state.regionSelectionSeq) return;
    }
    brainOutline = state.brainOutlinePayload;
  }

  if (seq !== state.regionSelectionSeq) return;
  state.viewer3d.setRegionFocus({
    selections,
    showOutlines: state.showRegionOutlines,
    brainOutlinePayload: state.showBrainOutline ? brainOutline : null,
  });
  renderRegionColorLegend(selections);
}

async function applyRegionFocus3D(regionId, seq = state.regionSelectionSeq) {
  if (!regionId) return;
  if (!state.selectedRegionIds.has(Number(regionId))) {
    state.selectedRegionIds = new Set([Number(regionId)]);
    state.activeRegionId = Number(regionId);
  }
  await applySelectedRegionsFocus3D(seq);
}

function scheduleSliceRefresh(immediate = false) {
  if (state.sliceRefreshTimer) {
    clearTimeout(state.sliceRefreshTimer);
    state.sliceRefreshTimer = null;
  }
  if (immediate) {
    void refreshSliceImage();
    return;
  }
  state.sliceRefreshTimer = setTimeout(() => {
    state.sliceRefreshTimer = null;
    void refreshSliceImage();
  }, 250);
}

function syncPairSampleFields() {
  if (state.pairSampleA) {
    syncGroupSampleFields(groupEls, state.pairSampleA, state.pairSampleB || "");
    if (els.compareSampleDir && state.pairSampleB) {
      els.compareSampleDir.value = state.pairSampleB;
      state.compareSampleDir = state.pairSampleB;
    }
  } else {
    syncGroupSampleFields(groupEls, state.sampleDir, state.compareSampleDir);
  }
}

function defaultGroupLabel(sampleDir) {
  return String(sampleDir || "")
    .split(/[/\\]/)
    .filter(Boolean)
    .pop();
}

function syncAnalysisSamplesFromState() {
  if (!state.analysisSamples.length && state.sampleDir) {
    state.analysisSamples = [
      {
        sample_dir: state.sampleDir,
        group: defaultGroupLabel(state.sampleDir),
        signal_ch: state.signalCh,
        sample_id: defaultGroupLabel(state.sampleDir),
      },
    ];
  }
  renderAnalysisSamplesTable(groupEls, state.analysisSamples, {
    onChange: () => updateAnalysisStatus(),
  });
}

function updateAnalysisStatus() {
  if (!groupEls.status) return;
  const count = state.analysisSamples.length;
  if (count >= 2) {
    groupEls.status.classList.remove("hidden");
    groupEls.status.textContent = `${count} samples loaded. Set group labels and run Analysis.`;
  } else if (count === 1) {
    groupEls.status.classList.remove("hidden");
    groupEls.status.textContent = "Load at least one more sample for Analysis.";
  }
}

function registerLoadedSample(sampleDir) {
  const dir = String(sampleDir || "").trim();
  if (!dir) return false;
  if (!state.loadedSamples.includes(dir)) {
    state.loadedSamples.push(dir);
  }
  if (!state.analysisSamples.some((entry) => entry.sample_dir === dir)) {
    state.analysisSamples.push({
      sample_dir: dir,
      group: defaultGroupLabel(dir),
      signal_ch: state.signalCh,
      sample_id: defaultGroupLabel(dir),
    });
  }
  state.pairSampleA = state.analysisSamples[0]?.sample_dir || "";
  state.pairSampleB = state.analysisSamples[1]?.sample_dir || "";
  if (state.pairSampleB) {
    state.compareSampleDir = state.pairSampleB;
  }
  syncAnalysisSamplesFromState();
  updateAnalysisStatus();
  if (state.bundle) renderSummary();
  return state.analysisSamples.length >= 2 && state.analysisSamples.at(-1)?.sample_dir === dir;
}

async function runAnalysis(options = {}) {
  const manifestPath = groupEls.manifestPath?.value.trim();
  readAnalysisSamplesFromTable(groupEls, state.analysisSamples);
  const sampleManifest = buildAnalysisManifestFromSamples(state.analysisSamples, state.signalCh);
  if (!manifestPath && sampleManifest.length < 2) {
    if (groupEls.status) {
      groupEls.status.classList.remove("hidden");
      groupEls.status.textContent = "Load at least two samples or provide a manifest CSV.";
    }
    return null;
  }
  const regionScope = groupEls.regionScopeSelect?.value || "all";
  if (regionScope === "subtree" && !state.activeRegionId) {
    if (groupEls.status) {
      groupEls.status.classList.remove("hidden");
      groupEls.status.textContent = "Select a parent region in the browser for subtree scope.";
    }
    return null;
  }
  if (groupEls.runButton) groupEls.runButton.disabled = true;
  if (groupEls.status) groupEls.status.classList.add("hidden");
  try {
    const requestBody = {
      level: groupEls.levelSelect?.value || state.defaultLevel,
      metric: groupEls.metricSelect?.value || "cfos_count",
      heatmap_mode: groupEls.heatmapModeSelect?.value || "differential",
      top_n: 36,
      group_a: groupEls.groupASelect?.value || null,
      group_b: groupEls.groupBSelect?.value || null,
    };
    if (regionScope === "subtree" && state.activeRegionId) {
      requestBody.focus_region_id = Number(state.activeRegionId);
    }
    if (manifestPath) {
      requestBody.manifest_path = manifestPath;
    } else {
      requestBody.manifest_json = JSON.stringify(sampleManifest);
    }
    const payload = await runGroupAnalysis(groupEls, requestBody);
    state.groupAnalysis = payload;
    renderGroupAnalysis(groupEls, payload, {
      onRegionClick: (regionId) => toggleRegionSelection(regionId),
      activeRegionId: state.activeRegionId,
    });
    if (options.switchToAnalysis !== false) {
      setActiveView("compare");
    }
    return payload;
  } catch (error) {
    if (groupEls.status) {
      groupEls.status.classList.remove("hidden");
      groupEls.status.textContent = error instanceof Error ? error.message : "Analysis failed.";
    }
    return null;
  } finally {
    if (groupEls.runButton) groupEls.runButton.disabled = false;
  }
}

async function runPairwiseAnalysis(options = {}) {
  return runAnalysis({ switchToAnalysis: options.switchToCompare !== false });
}

function setActiveView(viewName) {
  state.activeView = viewName;
  els.viewTabs.forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.view === viewName);
  });
  els.summaryView?.classList.toggle("active", viewName === "summary");
  els.sliceView.classList.toggle("active", viewName === "slice2d");
  els.points3dView.classList.toggle("active", viewName === "points3d");
  els.compareView?.classList.toggle("active", viewName === "compare");
  els.sliceControls.classList.toggle("hidden", viewName !== "slice2d");
  els.viewer3dControls.classList.toggle("hidden", viewName !== "points3d");
  els.compareControls?.classList.toggle("hidden", viewName !== "compare");
  els.sliceNavigator?.classList.toggle("hidden", viewName !== "slice2d" || !state.spatialAxes?.available);
  if (viewName === "summary" && (state.summary || state.bundle)) {
    renderSummary();
  }
  if (viewName === "points3d" && state.bundle) {
    requestAnimationFrame(() => {
      load3DPoints();
      renderSavedViews();
    });
  }
  if (viewName === "slice2d" && state.bundle) {
    requestAnimationFrame(() => {
      state.viewer3d?.resize();
      renderAxisHistograms();
      // Always re-render so an active selection outline appears immediately on
      // entering the 2D view (do not wait for a second tab switch).
      refreshSliceImage();
      if (state.bundle?.spatial?.available && !state.spatialAxes) {
        loadSpatialData();
      }
      if (!state.regionCentroidStats?.available) {
        void loadRegionCentroids();
      }
    });
  }
  if (viewName === "compare" && state.groupAnalysis) {
    requestAnimationFrame(() => {
      renderGroupAnalysis(groupEls, state.groupAnalysis, {
        onRegionClick: (regionId) => toggleRegionSelection(regionId),
        activeRegionId: state.activeRegionId,
      });
    });
  }
}

async function refreshSliceImage() {
  if (!state.bundle) return;
  const seq = ++state.sliceRefreshSeq;
  if (state.bundle.parameters?.atlas_label_available === false) {
    els.sliceImage.classList.add("is-hidden");
    els.slicePlaceholder.style.display = "block";
    els.slicePlaceholder.textContent =
      state.bundle.parameters.atlas_label_error ||
      "Atlas label TIFF not found. Set YIFU_DATA_DIR or place atlas_label.tiff under data/reference/.";
    return;
  }

  const level = currentLevel();
  const colorMode = els.sliceColorMode?.value || state.sliceColorMode || "region";
  state.sliceColorMode = colorMode;
  const params = new URLSearchParams({
    sample_dir: state.sampleDir,
    signal_ch: state.signalCh,
    plane: els.planeSelect.value,
    coordinate_system: "index",
    coordinate: els.sliceIndex.value,
    metric: els.metricSelect.value,
    color_mode: colorMode,
  });
  if (level) params.set("level", level);
  const compareDir = els.compareSampleDir?.value.trim() || state.compareSampleDir;
  if (compareDir && compareColorModes().has(colorMode)) {
    params.set("compare_sample_dir", compareDir);
    params.set("compare_signal_ch", state.signalCh);
  }
  if (state.activeRegionId && colorMode === "region") {
    params.set("highlight_region_id", String(state.activeRegionId));
  }
  const url = `/api/slice.png?${params.toString()}`;

  els.sliceImage.classList.add("is-hidden");
  els.slicePlaceholder.style.display = "block";
  els.slicePlaceholder.textContent = "Rendering 2D slice… (first load may take several seconds)";

  try {
    const response = await fetch(url);
    if (seq !== state.sliceRefreshSeq) return;
    if (!response.ok) {
      const payload = await response.json().catch(() => ({ detail: `Slice render failed (${response.status})` }));
      if (seq !== state.sliceRefreshSeq) return;
      const detail = payload.detail || `Slice render failed (${response.status})`;
      els.slicePlaceholder.textContent = typeof detail === "string" ? detail : JSON.stringify(detail);
      return;
    }
    state.sliceLayout = parseSliceLayoutHeaders(response.headers);
    const blob = await response.blob();
    if (seq !== state.sliceRefreshSeq) return;
    if (state.sliceImageObjectUrl) {
      URL.revokeObjectURL(state.sliceImageObjectUrl);
    }
    state.sliceImageObjectUrl = URL.createObjectURL(blob);
    await new Promise((resolve, reject) => {
      els.sliceImage.onload = () => resolve();
      els.sliceImage.onerror = () => reject(new Error("Failed to display slice image."));
      els.sliceImage.src = state.sliceImageObjectUrl;
    });
    if (seq !== state.sliceRefreshSeq) return;
    els.sliceImage.classList.remove("is-hidden");
    els.slicePlaceholder.style.display = "none";
    attachSliceImageInteractions();
  } catch (error) {
    if (seq !== state.sliceRefreshSeq) return;
    els.slicePlaceholder.textContent =
      error instanceof Error ? error.message : "Failed to load 2D slice image.";
  }
}

function compareColorModes() {
  return new Set(["dual", "split_lr", "diff", "fold"]);
}

function syncMetricControlForColorMode() {
  const colorMode = els.sliceColorMode?.value || "region";
  if (els.metricSelect) {
    els.metricSelect.disabled = colorMode === "signal";
  }
  if (els.compareSampleDir) {
    els.compareSampleDir.disabled = !compareColorModes().has(colorMode);
  }
}

function bookmarkPositionLabel(bookmark) {
  const axis = PLANE_AXIS[bookmark.plane] || "AP";
  const bregma = bookmark.bregma_mm ?? indexToBregmaMm(axis, bookmark.coordinate);
  const sign = bregma >= 0 ? "+" : "";
  return `${bookmark.label || "Slice"} · ${bookmark.plane} idx ${bookmark.coordinate} · Bregma ${sign}${Number(bregma).toFixed(2)} mm`;
}

function selectedSliceExportCount() {
  return state.sliceBookmarks.reduce((count, bookmark) => {
    const modes = bookmark.exportModes || defaultExportModes();
    return count + SLICE_EXPORT_MODES.filter((mode) => modes[mode.id]).length;
  }, 0);
}

function renderSliceExportTable() {
  const wrap = els.sliceExportTableWrap;
  if (!wrap) return;

  if (!state.sliceBookmarks.length) {
    wrap.innerHTML = `<p class="muted slice-export-empty">Add slices to export region and signal heatmaps.</p>`;
    if (els.exportSliceBookmarks) els.exportSliceBookmarks.disabled = true;
    return;
  }

  const headerCells = SLICE_EXPORT_MODES.map((mode) => {
    const allSelected = state.sliceBookmarks.every((bookmark) => bookmark.exportModes?.[mode.id]);
    const someSelected = state.sliceBookmarks.some((bookmark) => bookmark.exportModes?.[mode.id]);
    const mark = allSelected ? "☑" : someSelected ? "◫" : "☐";
    return `<th class="mode-header" data-mode="${mode.id}" title="Toggle all ${mode.label}">${mark} ${mode.label}</th>`;
  }).join("");

  const bodyRows = state.sliceBookmarks
    .map((bookmark, index) => {
      if (!bookmark.exportModes) bookmark.exportModes = defaultExportModes();
      const modeCells = SLICE_EXPORT_MODES.map((mode) => {
        const checked = bookmark.exportModes[mode.id] ? "checked" : "";
        return `<td class="mode-cell"><input type="checkbox" data-bookmark-index="${index}" data-mode="${mode.id}" ${checked} /></td>`;
      }).join("");
      return `<tr>
        <td class="position-cell" title="${bookmarkPositionLabel(bookmark)}">${bookmarkPositionLabel(bookmark)}</td>
        ${modeCells}
        <td><button type="button" class="row-remove" data-remove-index="${index}">Remove</button></td>
      </tr>`;
    })
    .join("");

  wrap.innerHTML = `<table class="slice-export-table">
    <thead><tr><th>Position</th>${headerCells}<th></th></tr></thead>
    <tbody>${bodyRows}</tbody>
  </table>`;

  wrap.querySelectorAll(".mode-header").forEach((header) => {
    header.addEventListener("click", () => {
      const modeId = header.dataset.mode;
      const allOn = state.sliceBookmarks.every((bookmark) => bookmark.exportModes?.[modeId]);
      for (const bookmark of state.sliceBookmarks) {
        if (!bookmark.exportModes) bookmark.exportModes = defaultExportModes();
        bookmark.exportModes[modeId] = !allOn;
      }
      renderSliceExportTable();
    });
  });

  wrap.querySelectorAll('input[type="checkbox"][data-bookmark-index]').forEach((input) => {
    input.addEventListener("change", () => {
      const bookmark = state.sliceBookmarks[Number(input.dataset.bookmarkIndex)];
      if (!bookmark) return;
      if (!bookmark.exportModes) bookmark.exportModes = defaultExportModes();
      bookmark.exportModes[input.dataset.mode] = input.checked;
      renderSliceExportTable();
    });
  });

  wrap.querySelectorAll("[data-remove-index]").forEach((button) => {
    button.addEventListener("click", () => {
      const index = Number(button.dataset.removeIndex);
      state.sliceBookmarks = state.sliceBookmarks.filter((_, idx) => idx !== index);
      renderSliceExportTable();
    });
  });

  if (els.exportSliceBookmarks) {
    els.exportSliceBookmarks.disabled = selectedSliceExportCount() === 0;
  }
}

function addSliceBookmark(override = {}) {
  const plane = override.plane || els.planeSelect.value;
  const coordinate = Math.round(Number(override.coordinate ?? els.sliceIndex.value));
  const bregma_mm = override.bregma_mm ?? bregmaMmForCurrentSlice();
  let label = override.label;
  if (!label && state.activeRegionId) {
    const metric = metricsForLevel(currentLevel()).find(
      (row) => Number(row.region_id) === Number(state.activeRegionId)
    );
    label = metric?.region_acronym || metric?.region_name || `region_${state.activeRegionId}`;
  }
  if (!label) {
    label = `${plane}_${coordinate}`;
  }

  const bookmark = {
    plane,
    coordinate_system: "index",
    coordinate,
    label,
    bregma_mm,
    region_id: override.region_id ?? state.activeRegionId ?? null,
    exportModes: defaultExportModes(),
  };
  const duplicate = state.sliceBookmarks.some(
    (entry) => entry.plane === bookmark.plane && entry.coordinate === bookmark.coordinate
  );
  if (!duplicate) {
    state.sliceBookmarks.push(bookmark);
  }
  renderSliceExportTable();
}

async function exportSliceBookmarksZip() {
  if (!state.bundle || state.sliceBookmarks.length === 0) return;
  const bookmarks = state.sliceBookmarks
    .map(({ plane, coordinate_system, coordinate, label, bregma_mm, region_id, exportModes }) => {
      const color_modes = SLICE_EXPORT_MODES.map((mode) => mode.id).filter((modeId) => exportModes?.[modeId]);
      if (!color_modes.length) return null;
      return {
        plane,
        coordinate_system,
        coordinate,
        label,
        bregma_mm,
        region_id,
        color_modes,
      };
    })
    .filter(Boolean);
  if (!bookmarks.length) {
    window.alert("Select at least one heatmap to export.");
    return;
  }

  const body = {
    sample_dir: state.sampleDir,
    signal_ch: state.signalCh,
    metric: els.metricSelect.value,
    level: currentLevel(),
    color_modes: SLICE_EXPORT_MODES.map((mode) => mode.id),
    bookmarks,
    focus_region_id: state.activeRegionId,
  };

  const response = await fetch("/api/export/slice-bookmarks.zip", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: "Export failed" }));
    window.alert(typeof payload.detail === "string" ? payload.detail : "Export failed.");
    return;
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${state.bundle.sample.sample_id}_slice_bookmarks.zip`;
  anchor.click();
  URL.revokeObjectURL(url);
}

function updateExportButton() {
  els.exportSelected.disabled = !state.bundle || state.selectedRegionIds.size === 0;
}

function setHeaderSample(sampleId) {
  if (!els.headerSampleId) return;
  els.headerSampleId.textContent = sampleId || "No sample";
}

function toggleSidePanel(side) {
  const isLeft = side === "left";
  const panel = isLeft ? els.panelLeft : els.panelRight;
  const toggle = isLeft ? els.togglePanelLeft : els.togglePanelRight;
  if (!panel || !els.appLayout) return;
  const collapsed = panel.classList.toggle("collapsed");
  els.appLayout.classList.toggle(isLeft ? "left-collapsed" : "right-collapsed", collapsed);
  if (toggle) {
    toggle.textContent = collapsed ? (isLeft ? "›" : "‹") : isLeft ? "‹" : "›";
    toggle.title = collapsed
      ? `Expand ${isLeft ? "region browser" : "region detail"}`
      : `Collapse ${isLeft ? "region browser" : "region detail"}`;
    toggle.setAttribute("aria-expanded", collapsed ? "false" : "true");
  }
  if (state.activeView === "points3d") {
    requestAnimationFrame(() => state.viewer3d?.resize());
  }
}

function renderSavedViews() {
  const list = els.savedViewsList;
  const viewer = state.viewer3d;
  const sampleId = state.bundle?.sample?.sample_id;
  if (!list || !viewer || !sampleId) {
    if (list) list.replaceChildren();
    return;
  }

  list.replaceChildren();
  for (const view of viewer.listSavedViews(sampleId)) {
    const item = document.createElement("li");
    item.className = "saved-view-item";

    const nameBtn = document.createElement("button");
    nameBtn.type = "button";
    nameBtn.className = "view-name";
    nameBtn.textContent = view.name;
    nameBtn.title = "Double-click to restore this view";
    nameBtn.addEventListener("dblclick", () => {
      viewer.loadNamedView(view.name, sampleId);
    });

    const deleteBtn = document.createElement("button");
    deleteBtn.type = "button";
    deleteBtn.textContent = "×";
    deleteBtn.title = `Delete ${view.name}`;
    deleteBtn.addEventListener("click", () => {
      viewer.deleteNamedView(view.name, sampleId);
      renderSavedViews();
    });

    item.append(nameBtn, deleteBtn);
    list.appendChild(item);
  }
}

function saveCurrentView() {
  const viewer = state.viewer3d;
  const sampleId = state.bundle?.sample?.sample_id;
  if (!viewer || !sampleId) return;
  viewer.saveCamera(sampleId);
  renderSavedViews();
}

function parseLaunchParams() {
  const params = new URLSearchParams(window.location.search);
  const sampleDir = (params.get("sample_dir") || params.get("sampleDir") || "").trim();
  if (!sampleDir) return null;
  return {
    sampleDir,
    signalCh: (params.get("signal_ch") || params.get("signalCh") || "ch1").trim() || "ch1",
    group: (params.get("group") || "").trim(),
    refresh: params.get("refresh") === "1" || params.get("refresh") === "true",
  };
}

function showSummaryLoading(message) {
  if (!els.summaryPanel) return;
  els.summaryPanel.innerHTML = `<p class="summary-note muted">${message}</p>`;
}

async function resolveStartupSample() {
  const launch = parseLaunchParams();
  if (launch?.sampleDir) return launch;
  const response = await fetch("/api/bootstrap");
  if (!response.ok) return null;
  const payload = await response.json();
  if (!payload.sample_dir) return null;
  return {
    sampleDir: payload.sample_dir,
    signalCh: payload.signal_ch || "ch1",
    group: payload.group || "",
    refresh: false,
    sampleId: payload.sample_id || "",
  };
}

function applyLoadedBundle() {
  if (!state.bundle) return;

  state.defaultLevel = state.bundle.parameters.default_level;
  state.atlasShape = state.bundle.parameters.atlas_shape_dv_ap_ml || state.atlasShape;
  state.bregmaIndex = state.bundle.parameters.bregma_index || state.bregmaIndex;
  state.selectedRegionIds.clear();
  state.activeRegionId = null;
  state.highlightMemberIds = null;
  state.brainOutlinePayload = null;
  state.cachedRegionSurface = null;
  state.cachedRegionSurfaceId = null;
  state.showBrainOutline = false;
  state.showRegionOutlines = true;
  state.regionSurfaceCache = new Map();
  state.regionSubtreeCache = new Map();
  state.sliceBookmarks = [];
  state.collapsedRegionIds = collectParentRegionIds(state.bundle.region_tree);
  rebuildRegionDescendantsIndex();
  syncBrainOutlineButton();
  syncRegionOutlinesButton();
  renderRegionColorLegend([]);
  renderSliceExportTable();

  syncGroupLevelOptions(groupEls, state.bundle.levels, state.defaultLevel);
  if (!state.analysisSamples.some((entry) => entry.sample_dir === state.sampleDir)) {
    state.analysisSamples.unshift({
      sample_dir: state.sampleDir,
      group: defaultGroupLabel(state.sampleDir),
      signal_ch: state.signalCh,
      sample_id: defaultGroupLabel(state.sampleDir),
    });
  }
  syncAnalysisSamplesFromState();
  updateAnalysisStatus();
  updateSliceSliderForPlane();
  renderSummary();
  renderRegionTree();
  updateExportButton();
}

async function initReport({ sampleDir, signalCh, group = "", refresh = false } = {}) {
  state.sampleDir = sampleDir;
  state.signalCh = signalCh || "ch1";
  state.group = group || "";

  setHeaderSample("Loading…");
  showSummaryLoading("Building report from pipeline Excel…");

  const params = new URLSearchParams({
    sample_dir: state.sampleDir,
    signal_ch: state.signalCh,
  });
  if (refresh) params.set("refresh", "true");
  if (state.group) params.set("group", state.group);

  try {
    const response = await fetch(`/api/report?${params.toString()}`);
    if (!response.ok) {
      const payload = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
      const detail = typeof payload.detail === "string" ? payload.detail : "Failed to load report";
      showSummaryLoading(detail);
      setHeaderSample("Error");
      return;
    }
    state.bundle = await response.json();
    state.summary = state.bundle.summary || resolveSummaryPayload(state.bundle);
    if (!state.summary) {
      state.summary = await fetchSummary("", {
        sampleDir: state.sampleDir,
        signalCh: state.signalCh,
        refresh,
      });
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : "Network error";
    showSummaryLoading(message);
    setHeaderSample("Error");
    return;
  }

  const sampleId = state.bundle.sample?.sample_id || state.summary?.sample?.sample_id || "sample";
  setHeaderSample(sampleId);

  applyLoadedBundle();
  void loadRegionCentroids();
  const becamePair = registerLoadedSample(state.sampleDir);

  if (becamePair) {
    await runPairwiseAnalysis({ switchToCompare: true });
  } else if (state.activeView === "compare" && state.groupAnalysis) {
    setActiveView("compare");
  } else if (state.activeView !== "points3d" && state.activeView !== "compare") {
    setActiveView("summary");
  }

  if (state.activeView === "slice2d") {
    await refreshSliceImage();
    loadSpatialData();
  }
  if (state.activeView === "points3d") {
    await load3DPoints();
  }
}

async function initApp() {
  showSummaryLoading("Resolving sample…");
  let startup;
  try {
    startup = await resolveStartupSample();
  } catch {
    startup = null;
  }
  if (!startup?.sampleDir) {
    showSummaryLoading("No sample configured. Start the server with --sample-dir or set CFOS_DEFAULT_SAMPLE_DIR.");
    setHeaderSample("No sample");
    return;
  }
  if (startup.sampleId) setHeaderSample(startup.sampleId);
  await initReport({
    sampleDir: startup.sampleDir,
    signalCh: startup.signalCh,
    group: startup.group,
    refresh: startup.refresh,
  });
}

function exportSelectedRegions() {
  const params = new URLSearchParams({
    sample_dir: state.sampleDir,
    signal_ch: state.signalCh,
    region_ids: [...state.selectedRegionIds].join(","),
    level: currentLevel(),
  });
  window.location.href = `/api/export/regions.csv?${params.toString()}`;
}

function downloadScreenshot3D() {
  if (!state.viewer3d || !state.bundle) return;
  const dataUrl = state.viewer3d.screenshot();
  const link = document.createElement("a");
  link.href = dataUrl;
  link.download = `${state.bundle.sample.sample_id}_3d_points.png`;
  link.click();
}

async function handleRunGroupAnalysis() {
  await runPairwiseAnalysis({ switchToCompare: true });
}

function exportGroupDifferentialCsv() {
  if (!state.groupAnalysis) return;
  const params = new URLSearchParams({
    level: groupEls.levelSelect?.value || state.defaultLevel,
    metric: groupEls.metricSelect?.value || "cfos_count",
    group_a: state.groupAnalysis.comparison.group_a,
    group_b: state.groupAnalysis.comparison.group_b,
  });
  const manifestPath = groupEls.manifestPath?.value.trim();
  if (manifestPath) {
    params.set("manifest_path", manifestPath);
  } else if (state.pairSampleA && state.pairSampleB) {
    params.set("sample_a_dir", state.pairSampleA);
    params.set("sample_b_dir", state.pairSampleB);
  } else if (state.sampleDir && state.compareSampleDir) {
    params.set("sample_a_dir", state.sampleDir);
    params.set("sample_b_dir", state.compareSampleDir);
  } else {
    return;
  }
  window.location.href = `/api/group/export/differential-regions.csv?${params.toString()}`;
}

els.toggleBrainOutline?.addEventListener("click", toggleBrainOutline);
els.toggleRegionOutlines?.addEventListener("click", toggleRegionOutlines);
els.togglePanelLeft?.addEventListener("click", () => toggleSidePanel("left"));
els.togglePanelRight?.addEventListener("click", () => toggleSidePanel("right"));
els.regionSearch?.addEventListener("input", renderRegionTree);
els.metricSelect?.addEventListener("change", () => scheduleSliceRefresh(true));
els.planeSelect?.addEventListener("change", () => {
  updateSliceSliderForPlane();
  renderAxisHistograms();
  scheduleSliceRefresh(true);
});
els.filterInBrainPoints?.addEventListener("change", () => {
  state.filterInBrainOnly = Boolean(els.filterInBrainPoints.checked);
  if (state.activeView === "points3d") {
    void load3DPoints();
  }
});
els.sliceIndex?.addEventListener("input", () => {
  els.sliceIndexLabel.textContent = els.sliceIndex.value;
  syncSliceBregmaLabel();
  renderAxisHistograms({ axis: currentAxisFromPlane(), coordinate: Number(els.sliceIndex.value) });
  scheduleSliceRefresh(false);
});
els.sliceColorMode?.addEventListener("change", () => {
  syncMetricControlForColorMode();
  scheduleSliceRefresh(true);
});
els.addSliceBookmark?.addEventListener("click", () => addSliceBookmark());
els.exportSliceBookmarks?.addEventListener("click", exportSliceBookmarksZip);
els.exportSelected?.addEventListener("click", exportSelectedRegions);
els.viewTabs?.forEach((tab) => {
  tab.addEventListener("click", () => setActiveView(tab.dataset.view));
});
els.resetCamera?.addEventListener("click", () => state.viewer3d?.resetCamera());
els.saveCamera?.addEventListener("click", saveCurrentView);
els.screenshot3d?.addEventListener("click", downloadScreenshot3D);
groupEls.sampleB?.addEventListener("input", () => {
  state.compareSampleDir = groupEls.sampleB.value.trim();
  state.pairSampleB = state.compareSampleDir;
  if (els.compareSampleDir) els.compareSampleDir.value = state.compareSampleDir;
  syncMetricControlForColorMode();
});
groupEls.runButton?.addEventListener("click", () => handleRunGroupAnalysis());
groupEls.metricSelect?.addEventListener("change", () => {
  if (state.groupAnalysis) handleRunGroupAnalysis();
});
groupEls.levelSelect?.addEventListener("change", () => {
  if (state.groupAnalysis) handleRunGroupAnalysis();
});
groupEls.regionScopeSelect?.addEventListener("change", () => {
  if (state.groupAnalysis) handleRunGroupAnalysis();
});
groupEls.heatmapModeSelect?.addEventListener("change", () => {
  if (state.groupAnalysis) handleRunGroupAnalysis();
});
groupEls.addSampleBtn?.addEventListener("click", () => {
  const path = groupEls.addSamplePath?.value.trim();
  if (!path) return;
  registerLoadedSample(path);
  if (groupEls.addSamplePath) groupEls.addSamplePath.value = "";
});
groupEls.addSamplePath?.addEventListener("keydown", (event) => {
  if (event.key === "Enter") groupEls.addSampleBtn?.click();
});
groupEls.exportButton?.addEventListener("click", exportGroupDifferentialCsv);

setActiveView("summary");
syncMetricControlForColorMode();
syncSliceBregmaLabel();
syncBrainOutlineButton();
attachSliceImageInteractions();
void initApp().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  showSummaryLoading(`Startup failed: ${message}`);
  setHeaderSample("Error");
});
