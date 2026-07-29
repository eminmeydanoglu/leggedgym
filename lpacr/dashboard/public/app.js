import {
  CELL_STATES, cellInk, color, extent, formatValue, groupLabels, healthSpec,
  metricKind, paletteFor, parseGroupedLabel, pickDefaultMetric, pickDefaultSelection,
  pickSignals, pretty, shortAxisLabel,
} from "./atlas_core.js";

const $ = (selector) => document.querySelector(selector);
const els = {
  atlas: $("#atlas"), triptych: $("#triptych"), timelines: $("#timelines"), health: $("#health"),
  run: $("#runSelect"), x: $("#xDimension"), y: $("#yDimension"),
  facet: $("#facetDimension"), filter: $("#filterValue"), filterWrap: $("#filterWrap"),
  metric: $("#metricSelect"), live: $("#liveButton"), history: $("#historyButton"),
  colorLegend: $("#colorLegend"),
  slider: $("#timeSlider"), play: $("#playButton"), count: $("#frameCount"),
  step: $("#stepValue"), stepLabel: $("#stepLabel"), detail: $("#detail"), empty: $("#empty"),
};

const state = {
  runs: [], frames: [], index: 0, mode: "live", stream: null, timer: null,
  selection: { x: null, y: null, facet: null, filterDim: null, filterIndex: 0 },
  /** True after the user changes X/Y/Panels/Filter; protects defaults from stream re-init. */
  selectionLocked: false,
};
/** Names the publisher gave the categorical cell states, when it supplied any. */
function stateNames(frame) {
  const names = frame?.metadata?.frame?.cell_state_names;
  return Array.isArray(names) && names.length ? names : CELL_STATES.map((entry) => entry.name);
}

function sanitizeSelection(dims) {
  if (!dims.includes(state.selection.x)) {
    state.selection.x = pickDefaultSelection(dims).x;
  }
  if (!dims.includes(state.selection.y) || state.selection.y === state.selection.x) {
    state.selection.y = dims.find((d) => d !== state.selection.x) || state.selection.x;
  }
  const facet = state.selection.facet;
  if (facet && facet !== "__none__" && !dims.includes(facet)) {
    state.selection.facet = "__none__";
  }
  if (state.selection.facet === state.selection.x || state.selection.facet === state.selection.y) {
    state.selection.facet = "__none__";
  }
}

async function json(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error((await response.json()).error || response.statusText);
  return response.json();
}

function setOptions(select, values, selected, labelFn = pretty) {
  select.replaceChildren(...values.map((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value === "__none__" ? "none" : labelFn(value);
    option.selected = value === selected;
    return option;
  }));
}

async function loadRuns(preferred) {
  state.runs = (await json("/api/runs")).runs;
  setOptions(els.run, state.runs.map((run) => run.run_id), preferred || state.runs[0]?.run_id, (v) => v);
  if (els.run.value) await selectRun(els.run.value);
  else showEmpty(true);
}

async function selectRun(runId) {
  closeStream();
  const payload = await json(`/api/runs/${encodeURIComponent(runId)}/frames`);
  state.frames = payload.frames.sort((a, b) => a.step - b.step);
  state.index = Math.max(0, state.frames.length - 1);
  state.selection = { x: null, y: null, facet: null, filterDim: null, filterIndex: 0 };
  state.selectionLocked = false;
  initializeControls();
  openStream(runId);
  render();
}

function initializeControls() {
  const frame = state.frames[0];
  if (!frame) return;
  const dims = frame.task_space.dimensions;
  if (state.selection.x == null || !state.selectionLocked && state.selection.facet == null) {
    // First paint (or run change): apply smart defaults.
    if (state.selection.x == null) {
      Object.assign(state.selection, pickDefaultSelection(dims));
    }
  }
  sanitizeSelection(dims);
  updateDimensionControls();
  if (!els.metric.value || !(els.metric.value in frame.metrics)) {
    setOptions(
      els.metric,
      Object.keys(frame.metrics),
      pickDefaultMetric(Object.keys(frame.metrics)),
    );
  } else {
    setOptions(els.metric, Object.keys(frame.metrics), els.metric.value);
  }
}

function updateDimensionControls() {
  const frame = state.frames[0];
  if (!frame) return;
  const dims = frame.task_space.dimensions;
  sanitizeSelection(dims);
  setOptions(els.x, dims, state.selection.x);
  const yChoices = dims.filter((d) => d !== state.selection.x);
  if (!yChoices.includes(state.selection.y)) {
    state.selection.y = yChoices[0] || state.selection.x;
  }
  setOptions(els.y, yChoices, state.selection.y);
  const facetValues = ["__none__", ...dims.filter((d) => ![state.selection.x, state.selection.y].includes(d))];
  if (!facetValues.includes(state.selection.facet)) state.selection.facet = "__none__";
  setOptions(els.facet, facetValues, state.selection.facet || "__none__");
  const excluded = [state.selection.x, state.selection.y, state.selection.facet];
  state.selection.filterDim = dims.find((d) => !excluded.includes(d)) || null;
  if (state.selection.filterDim) {
    els.filterWrap.firstChild.textContent = `${pretty(state.selection.filterDim)} `;
    const values = frame.task_space.coordinates[state.selection.filterDim];
    setOptions(
      els.filter,
      values.map((_, i) => String(i)),
      String(Math.min(state.selection.filterIndex, values.length - 1)),
      (i) => values[Number(i)],
    );
    els.filterWrap.hidden = false;
  } else {
    els.filterWrap.hidden = true;
  }
}

function openStream(runId) {
  state.stream = new EventSource(`/api/runs/${encodeURIComponent(runId)}/stream`);
  state.stream.addEventListener("frame", (event) => {
    const frame = JSON.parse(event.data);
    const last = state.frames.at(-1);
    if (!last || frame.step > last.step) state.frames.push(frame);
    else {
      const index = state.frames.findIndex((item) => item.step === frame.step);
      if (index >= 0) state.frames[index] = frame;
    }
    if (state.mode === "live") state.index = state.frames.length - 1;
    // Keep axes as the user left them; only refresh metric list if needed.
    const keys = Object.keys(frame.metrics || {});
    if (keys.length && !keys.includes(els.metric.value)) {
      setOptions(els.metric, keys, keys[0]);
    }
    render();
  });
}

function closeStream() {
  state.stream?.close();
  state.stream = null;
}

function strides(frame) {
  const dims = frame.task_space.dimensions;
  const lengths = dims.map((d) => frame.task_space.coordinates[d].length);
  const result = new Array(lengths.length).fill(1);
  for (let i = lengths.length - 2; i >= 0; i--) result[i] = result[i + 1] * lengths[i + 1];
  return { dims, lengths, strides: result };
}

function flatIndex(frame, indices) {
  const shape = strides(frame);
  return shape.dims.reduce((sum, dim, i) => sum + (indices[dim] || 0) * shape.strides[i], 0);
}

function sliceMatrix(frame, metric, facetIndex = null) {
  const { x, y, facet, filterDim, filterIndex } = state.selection;
  const coords = frame.task_space.coordinates;
  const width = coords[x].length;
  const height = coords[y].length;
  const matrix = [];
  for (let yi = 0; yi < height; yi++) {
    const row = [];
    for (let xi = 0; xi < width; xi++) {
      const indices = { [x]: xi, [y]: yi };
      if (facet && facet !== "__none__") indices[facet] = facetIndex;
      if (filterDim) indices[filterDim] = Number(filterIndex);
      row.push(frame.metrics[metric]?.[flatIndex(frame, indices)] ?? null);
    }
    matrix.push(row);
  }
  return matrix;
}

function renderColorLegend(metric, range, frame) {
  const palette = paletteFor(metric);
  // Categorical states get one named swatch each; a gradient would imply that
  // "mastered" sits halfway between "frontier" and "unstable".
  if (metricKind(metric) === "categorical") {
    const names = stateNames(frame);
    els.colorLegend.replaceChildren(...palette.flatMap((value, index) => {
      const swatch = document.createElement("i");
      swatch.className = "legend-swatch";
      swatch.style.background = value;
      const label = document.createElement("span");
      label.className = "legend-label";
      label.textContent = names[index] ?? String(index);
      return [swatch, label];
    }));
    return;
  }
  const start = document.createElement("span");
  start.className = "legend-label";
  start.textContent = range ? format(range[0], metric) : "Low";
  const end = document.createElement("span");
  end.className = "legend-label";
  end.textContent = range ? format(range[1], metric) : "High";
  const swatches = palette.map((value) => {
    const swatch = document.createElement("i");
    swatch.className = "legend-swatch";
    swatch.style.background = value;
    return swatch;
  });
  if (metricKind(metric) === "diverging" && range) {
    const zero = document.createElement("span");
    zero.className = "legend-label";
    zero.textContent = "0";
    els.colorLegend.replaceChildren(start, ...swatches.slice(0, 2), zero, ...swatches.slice(2), end);
  } else {
    els.colorLegend.replaceChildren(start, ...swatches, end);
  }
}

/**
 * Crisp HiDPI canvas. Always pass explicit CSS pixel size — never trust a
 * post-layout clientWidth that may be half-resolved (that was the blur bug).
 */
function setupCanvas(canvas, cssWidth, cssHeight) {
  const dpr = Math.max(1, Math.min(3, window.devicePixelRatio || 1));
  const width = Math.max(1, Math.round(cssWidth));
  const height = Math.max(1, Math.round(cssHeight));
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  canvas.style.maxHeight = "none";
  canvas.style.aspectRatio = "auto";
  const bw = Math.round(width * dpr);
  const bh = Math.round(height * dpr);
  if (canvas.width !== bw) canvas.width = bw;
  if (canvas.height !== bh) canvas.height = bh;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  return { ctx, width, height, dpr };
}

function prettyTypeName(name) {
  return String(name).replaceAll("_", " ");
}

/** Compute pixel layout for any X/Y pairing (terrain×vx or swapped). */
function computeHeatmapLayout(frame, mode = "full") {
  const xLabels = frame.task_space.coordinates[state.selection.x] || [];
  const yLabels = frame.task_space.coordinates[state.selection.y] || [];
  const xGroups = groupLabels(xLabels);
  const yGroups = groupLabels(yLabels);
  const nX = Math.max(1, xLabels.length);
  const nY = Math.max(1, yLabels.length);

  // Bigger cells when the grid is small; still near-square.
  let cell = mode === "full" ? 44 : 28;
  if (mode === "full") {
    if (nX * nY <= 24) cell = 52;
    else if (nX * nY <= 48) cell = 46;
    else if (nX >= 16 || nY >= 16) cell = 40;
  }

  const margin = { l: 16, r: 14, t: 12, b: 16 };
  if (mode === "bare") return { xLabels, yLabels, xGroups, yGroups, cell, margin: { l: 8, r: 8, t: 8, b: 8 }, width: nX * cell + 16, height: nY * cell + 16 };

  if (yGroups) margin.l = mode === "full" ? 112 : 96;
  else if (yLabels.some((l) => /m\/s/i.test(l))) margin.l = mode === "full" ? 78 : 64;
  else margin.l = mode === "full" ? 72 : 56;

  if (xGroups) margin.b = mode === "full" ? 92 : 76;
  else if (xLabels.some((l) => /m\/s/i.test(l))) margin.b = mode === "full" ? 42 : 36;
  else margin.b = mode === "full" ? 48 : 40;

  margin.t = mode === "full" ? 12 : 10;
  margin.r = mode === "full" ? 16 : 12;

  const width = margin.l + margin.r + nX * cell;
  const height = margin.t + margin.b + nY * cell;
  return { xLabels, yLabels, xGroups, yGroups, cell, margin, width, height };
}

function drawGroupedAxis(ctx, groups, {
  orientation, // "y" | "x"
  margin, width, height, cellW, cellH, nLabels, fontSize,
}) {
  const small = `${Math.max(10, fontSize - 1)}px "Noto Sans", "Liberation Sans", system-ui, sans-serif`;
  const groupFont = `500 ${Math.max(11, fontSize)}px "Noto Sans", "Liberation Sans", system-ui, sans-serif`;

  groups.forEach((group) => {
    if (orientation === "y") {
      const top = margin.t + (nLabels - 1 - group.end) * cellH;
      const bottom = margin.t + (nLabels - group.start) * cellH;
      const mid = (top + bottom) / 2;
      ctx.save();
      ctx.translate(Math.max(18, margin.l * 0.30), mid);
      ctx.rotate(-Math.PI / 5);
      ctx.fillStyle = "#c0cbc5";
      ctx.font = groupFont;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(prettyTypeName(group.name), 0, 0);
      ctx.restore();

      ctx.fillStyle = "#8a9690";
      ctx.font = small;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let yi = group.start; yi <= group.end; yi++) {
        const y = margin.t + (nLabels - yi - .5) * cellH;
        ctx.fillText(group.leaves[yi - group.start], margin.l - 8, y);
      }
      if (group.end < nLabels - 1) {
        const yLine = margin.t + (nLabels - 1 - group.end) * cellH;
        ctx.strokeStyle = "rgba(42, 53, 49, 0.95)";
        ctx.beginPath();
        ctx.moveTo(margin.l, yLine);
        ctx.lineTo(width - margin.r, yLine);
        ctx.stroke();
      }
    } else {
      // X: levels under each column + diagonal type under the group.
      const left = margin.l + group.start * cellW;
      const right = margin.l + (group.end + 1) * cellW;
      const mid = (left + right) / 2;
      ctx.fillStyle = "#8a9690";
      ctx.font = small;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (let xi = group.start; xi <= group.end; xi++) {
        const x = margin.l + (xi + .5) * cellW;
        ctx.fillText(group.leaves[xi - group.start], x, height - margin.b + 6);
      }
      ctx.save();
      ctx.translate(mid, height - Math.max(18, margin.b * 0.38));
      ctx.rotate(-Math.PI / 6);
      ctx.fillStyle = "#c0cbc5";
      ctx.font = groupFont;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(prettyTypeName(group.name), 0, 0);
      ctx.restore();
      if (group.end < nLabels - 1) {
        const xLine = margin.l + (group.end + 1) * cellW;
        ctx.strokeStyle = "rgba(42, 53, 49, 0.95)";
        ctx.beginPath();
        ctx.moveTo(xLine, margin.t);
        ctx.lineTo(xLine, height - margin.b);
        ctx.stroke();
      }
    }
  });
}

function drawHeatmapAxes(ctx, layout) {
  const { xLabels, yLabels, xGroups, yGroups, margin, width, height } = layout;
  const cellW = (width - margin.l - margin.r) / xLabels.length;
  const cellH = (height - margin.t - margin.b) / yLabels.length;
  const fontSize = 12;
  const font = `${fontSize}px "Noto Sans", "Liberation Sans", system-ui, sans-serif`;
  const tickColor = "#8d9894";

  if (xGroups) {
    drawGroupedAxis(ctx, xGroups, {
      orientation: "x", margin, width, height, cellW, cellH,
      nLabels: xLabels.length, fontSize,
    });
  } else {
    ctx.fillStyle = tickColor;
    ctx.font = font;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    xLabels.forEach((label, i) => {
      ctx.fillText(
        shortAxisLabel(label, cellW > 48 ? 14 : 10),
        margin.l + (i + .5) * cellW,
        height - margin.b + 8,
      );
    });
  }

  if (yGroups) {
    drawGroupedAxis(ctx, yGroups, {
      orientation: "y", margin, width, height, cellW, cellH,
      nLabels: yLabels.length, fontSize,
    });
  } else {
    ctx.fillStyle = tickColor;
    ctx.font = font;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    yLabels.forEach((label, i) => {
      ctx.fillText(
        shortAxisLabel(label, 16),
        margin.l - 8,
        margin.t + (yLabels.length - i - .5) * cellH,
      );
    });
  }
}

const formatCellValue = (value, metric, frame) =>
  formatValue(value, metric, { compact: true, stateNames: stateNames(frame) });

/**
 * @param {number[][]|null} frontierMask  matrix of cell states, when the run
 *   publishes them; frontier cells get an outline so the active edge of the
 *   curriculum is visible no matter which metric is being coloured.
 */
function drawHeatmap(canvas, matrix, range, metric, frame, facetIndex, mode = "full", frontierMask = null) {
  const layout = computeHeatmapLayout(frame, mode);
  const { ctx, width, height } = setupCanvas(canvas, layout.width, layout.height);
  const { xLabels, yLabels, margin } = layout;
  const cellW = (width - margin.l - margin.r) / xLabels.length;
  const cellH = (height - margin.t - margin.b) / yLabels.length;
  const gap = mode === "full" ? 2 : 1.25;
  ctx.clearRect(0, 0, width, height);

  if (mode === "full") {
    ctx.strokeStyle = "#24302c";
    ctx.lineWidth = 1;
    ctx.strokeRect(
      margin.l - 0.5,
      margin.t - 0.5,
      width - margin.l - margin.r + 1,
      height - margin.t - margin.b + 1,
    );
  }

  const showValues = mode === "full" && cellW >= 32 && cellH >= 18;
  const valueFontSize = Math.max(10, Math.min(13, Math.floor(Math.min(cellW, cellH) * 0.34)));
  const valueFont = `${valueFontSize}px "Noto Mono", "DejaVu Sans Mono", ui-monospace, monospace`;

  matrix.forEach((row, yi) => row.forEach((value, xi) => {
    const x = margin.l + xi * cellW + gap;
    const y = margin.t + (yLabels.length - 1 - yi) * cellH + gap;
    const w = Math.max(0, cellW - gap * 2);
    const h = Math.max(0, cellH - gap * 2);
    ctx.fillStyle = color(value, range, metric);
    ctx.fillRect(x, y, w, h);
    // 1 === CELL_STATE_FRONTIER: unlocked but not yet mastered.
    if (frontierMask && Math.round(frontierMask[yi]?.[xi]) === 1) {
      ctx.strokeStyle = "#d9a24c";
      ctx.lineWidth = mode === "full" ? 1.5 : 1;
      ctx.strokeRect(x + 0.5, y + 0.5, Math.max(0, w - 1), Math.max(0, h - 1));
    }
    if (showValues) {
      const text = formatCellValue(value, metric, frame);
      if (text) {
        ctx.fillStyle = cellInk(value, range, metric);
        ctx.font = valueFont;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        // Half-pixel nudge for sharper glyph centering on integer DPR
        ctx.fillText(text, Math.round(x + w / 2) + 0.5, Math.round(y + h / 2) + 0.5);
      }
    }
  }));

  if (mode !== "bare") drawHeatmapAxes(ctx, layout);

  canvas.onmousemove = (event) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = width / rect.width;
    const scaleY = height / rect.height;
    const px = (event.clientX - rect.left) * scaleX;
    const py = (event.clientY - rect.top) * scaleY;
    const xi = Math.floor((px - margin.l) / cellW);
    const displayY = Math.floor((py - margin.t) / cellH);
    const yi = yLabels.length - 1 - displayY;
    if (xi < 0 || yi < 0 || xi >= xLabels.length || yi >= yLabels.length) return hideDetail();
    const indices = { [state.selection.x]: xi, [state.selection.y]: yi };
    if (state.selection.facet && state.selection.facet !== "__none__") indices[state.selection.facet] = facetIndex;
    if (state.selection.filterDim) indices[state.selection.filterDim] = Number(state.selection.filterIndex);
    showDetail(event, frame, indices);
  };
  canvas.onmouseleave = hideDetail;
}

function showDetail(event, frame, indices) {
  const index = flatIndex(frame, indices);
  const rows = frame.task_space.dimensions.map((dim) => {
    const label = frame.task_space.coordinates[dim][indices[dim] || 0];
    const grouped = parseGroupedLabel(label);
    if (grouped && dim === "terrain_cell") {
      return `<dt>Terrain</dt><dd>${grouped.group}</dd><dt>Level</dt><dd>${grouped.leaf}</dd>`;
    }
    return `<dt>${pretty(dim)}</dt><dd>${label}</dd>`;
  });
  // Lead with the decision-bearing metrics, then everything else the run
  // publishes -- a V6 cell carries ~30 of them and all of them are inspectable.
  const lead = [
    "state", "sampling_probability", "success_probability", "success_probability_delta",
    "learning_progress", "performance", "success_rate",
  ];
  const names = Object.keys(frame.metrics);
  const ordered = [
    ...lead.filter((name) => names.includes(name)),
    ...names.filter((name) => !lead.includes(name)),
  ];
  for (const metric of ordered) {
    const value = frame.metrics[metric][index];
    rows.push(`<dt>${pretty(metric)}</dt><dd>${format(value, metric, frame)}</dd>`);
  }
  els.detail.innerHTML = `<dl>${rows.join("")}</dl>`;
  els.detail.classList.add("visible");
  els.detail.style.left = `${Math.min(event.clientX + 15, innerWidth - 235)}px`;
  // Measure after painting: the readout is as tall as the run has metrics.
  const height = els.detail.offsetHeight || 230;
  els.detail.style.top = `${Math.max(8, Math.min(event.clientY + 15, innerHeight - height - 12))}px`;
}
function hideDetail() { els.detail.classList.remove("visible"); }
const format = (value, metric, frame) =>
  formatValue(value, metric, { stateNames: stateNames(frame) });

function renderAtlas(frame) {
  const facet = state.selection.facet;
  const faceted = facet && facet !== "__none__";
  const facetValues = faceted ? frame.task_space.coordinates[facet] : ["All tasks"];
  const metric = els.metric.value;
  const matrices = facetValues.map((_, i) => sliceMatrix(frame, metric, i));
  const stateMatrices = frame.metrics.state && metric !== "state"
    ? facetValues.map((_, i) => sliceMatrix(frame, "state", i))
    : null;
  const range = extent(matrices, metric);
  renderColorLegend(metric, range, frame);
  els.atlas.classList.toggle("single", !faceted);
  const figures = facetValues.map((label, i) => {
    const figure = document.createElement("article");
    figure.className = "heatmap-figure";
    if (!faceted) figure.classList.add("single-map");
    const title = faceted ? `<h3><strong>${label}</strong></h3>` : "";
    figure.innerHTML = `${title}<canvas aria-label="${pretty(metric)} heatmap"></canvas>`;
    return figure;
  });
  els.atlas.replaceChildren(...figures);
  // Draw immediately with explicit pixel sizes (no rAF layout race → no blur).
  figures.forEach((figure, i) => {
    drawHeatmap(
      figure.querySelector("canvas"), matrices[i], range, metric, frame, i, "full",
      stateMatrices ? stateMatrices[i] : null,
    );
  });
}

function renderTriptych(frame) {
  const signals = pickSignals(Object.keys(frame.metrics));
  const facetIndex = state.selection.facet && state.selection.facet !== "__none__" ? 0 : null;
  const figures = signals.map((metric) => {
    const figure = document.createElement("article");
    figure.className = "signal-figure";
    figure.innerHTML = `<header><strong>${pretty(metric)}</strong></header><canvas aria-label="${pretty(metric)} heatmap"></canvas>`;
    return { figure, metric };
  });
  els.triptych.replaceChildren(...figures.map((item) => item.figure));
  figures.forEach(({ figure, metric }) => {
    const matrix = sliceMatrix(frame, metric, facetIndex);
    drawHeatmap(figure.querySelector("canvas"), matrix, extent(matrix, metric), metric, frame, facetIndex ?? 0, "axes");
  });
}

function marginal(frame, dim, metric = "sampling_probability") {
  const values = new Array(frame.task_space.coordinates[dim].length).fill(0);
  const shape = strides(frame);
  const series = frame.metrics[metric];
  if (!series) return values;
  series.forEach((value, index) => {
    const dimPosition = shape.dims.indexOf(dim);
    const coordinate = Math.floor(index / shape.strides[dimPosition]) % shape.lengths[dimPosition];
    if (Number.isFinite(value)) values[coordinate] += value;
  });
  return values;
}

const TIMELINE_COLORS = [
  "#5cb89a", "#c9a05a", "#6a96c4", "#a8b86a", "#a87cb8",
  "#c97878", "#6aada0", "#c48a52", "#7a9ab8", "#a09088",
  "#4a9fd4", "#d4a04a", "#8bc46a", "#c46a9a", "#6ac4b8",
];

/** Round up rawMax to a clean tick ceiling (1 / 2 / 5 × 10^n). */
function niceCeil(rawMax) {
  if (!Number.isFinite(rawMax) || rawMax <= 0) return 1;
  const exp = Math.floor(Math.log10(rawMax));
  const base = 10 ** exp;
  const n = rawMax / base;
  const nice = n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10;
  return nice * base;
}

function formatTimelineValue(value, yKind = "auto") {
  if (!Number.isFinite(value)) return "—";
  if (yKind === "probability") {
    const s = value.toFixed(3);
    return s.startsWith("0") && value < 1 ? s.slice(1) : s;
  }
  if (yKind === "count") {
    if (Math.abs(value) >= 100) return Math.round(value).toLocaleString();
    if (Math.abs(value) >= 10) return value.toFixed(1);
    return value.toFixed(2);
  }
  if (value >= 0 && value <= 1.001) {
    const s = value.toFixed(3);
    return s.startsWith("0") && value < 1 ? s.slice(1) : s;
  }
  if (Math.abs(value) >= 100) return Math.round(value).toLocaleString();
  return value.toFixed(2);
}

function formatStepLabel(step) {
  if (!Number.isFinite(step)) return "—";
  if (Math.abs(step) >= 1e6) return `${(step / 1e6).toFixed(1)}M`;
  if (Math.abs(step) >= 1e4) return `${Math.round(step / 1e3)}k`;
  if (Math.abs(step) >= 1e3) return `${(step / 1e3).toFixed(1)}k`;
  return String(Math.round(step));
}

/**
 * Interactive multi-series line chart with axes, grid, and hover readout.
 * @param {HTMLCanvasElement} canvas
 * @param {number[][]} series  frame → values[line]
 * @param {string[]} labels    one per series line
 * @param {number} active      current playback index in the sampled series
 * @param {{ steps?: number[], yKind?: "probability"|"count"|"auto" }} [options]
 */
function drawTimeline(canvas, series, labels, active, options = {}) {
  const steps = options.steps || series.map((_, i) => i);
  const yKind = options.yKind || "auto";
  const n = series.length;
  const nLines = labels.length;
  const rect = canvas.parentElement?.getBoundingClientRect();
  const cssW = Math.max(280, rect?.width || canvas.clientWidth || 320);
  const cssH = Math.max(160, Math.round(cssW * 0.48));
  const { ctx, width, height } = setupCanvas(canvas, cssW, cssH);

  // Room for Y ticks (left), X ticks + legend (bottom).
  const legendRows = Math.ceil(Math.min(nLines, 12) / Math.max(1, Math.floor((width - 16) / 88)));
  const margin = {
    l: 52,
    r: 14,
    t: 12,
    b: 28 + legendRows * 16,
  };
  const chartW = Math.max(1, width - margin.l - margin.r);
  const chartH = Math.max(1, height - margin.t - margin.b);

  const rawMax = Math.max(...series.flatMap((row) => row.slice(0, nLines)), 0);
  let yMax;
  if (yKind === "probability" && rawMax <= 1.05) yMax = Math.max(rawMax * 1.05, 0.1);
  else yMax = niceCeil(rawMax * 1.08);
  if (yMax <= 0) yMax = 1;

  const xAt = (i) => margin.l + (n <= 1 ? chartW / 2 : (i / (n - 1)) * chartW);
  const yAt = (v) => margin.t + chartH - (Math.max(0, v) / yMax) * chartH;

  const yTicks = (() => {
    const count = 4;
    const ticks = [];
    for (let i = 0; i <= count; i++) ticks.push((yMax * i) / count);
    return ticks;
  })();

  const xTickIndices = (() => {
    if (n <= 1) return [0];
    const count = Math.min(5, n);
    const out = [];
    for (let i = 0; i < count; i++) {
      out.push(Math.round((i / (count - 1)) * (n - 1)));
    }
    return [...new Set(out)];
  })();

  let hoverIndex = null;

  function paint() {
    ctx.clearRect(0, 0, width, height);

    // Plot background
    ctx.fillStyle = "rgba(18, 24, 22, 0.55)";
    ctx.fillRect(margin.l, margin.t, chartW, chartH);

    // Horizontal grid + Y labels
    ctx.font = '11px "Noto Sans", "Liberation Sans", system-ui, sans-serif';
    yTicks.forEach((tick, i) => {
      const y = yAt(tick);
      ctx.strokeStyle = i === 0 ? "#2e3c37" : "#1e2925";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(margin.l, y);
      ctx.lineTo(margin.l + chartW, y);
      ctx.stroke();
      ctx.fillStyle = "#8d9894";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(formatTimelineValue(tick, yKind), margin.l - 8, y);
    });

    // Left / bottom axis spines
    ctx.strokeStyle = "#3a4a44";
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    ctx.moveTo(margin.l, margin.t);
    ctx.lineTo(margin.l, margin.t + chartH);
    ctx.lineTo(margin.l + chartW, margin.t + chartH);
    ctx.stroke();

    // X tick marks + labels (control step)
    ctx.fillStyle = "#8d9894";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    xTickIndices.forEach((i) => {
      const x = xAt(i);
      ctx.strokeStyle = "#2e3c37";
      ctx.beginPath();
      ctx.moveTo(x, margin.t + chartH);
      ctx.lineTo(x, margin.t + chartH + 4);
      ctx.stroke();
      ctx.fillStyle = "#8d9894";
      ctx.fillText(formatStepLabel(steps[i]), x, margin.t + chartH + 6);
    });

    // Series lines
    for (let line = 0; line < nLines; line++) {
      const color = TIMELINE_COLORS[line % TIMELINE_COLORS.length];
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.75;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < n; i++) {
        const v = series[i][line];
        if (!Number.isFinite(v)) {
          started = false;
          continue;
        }
        const x = xAt(i);
        const y = yAt(v);
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    }

    // Active playback cursor (solid)
    if (n > 0 && Number.isFinite(active)) {
      const ai = Math.max(0, Math.min(n - 1, active));
      const ax = xAt(ai);
      ctx.strokeStyle = "rgba(197, 205, 201, 0.85)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(ax, margin.t);
      ctx.lineTo(ax, margin.t + chartH);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Hover cursor + dots
    if (hoverIndex != null && n > 0) {
      const hi = Math.max(0, Math.min(n - 1, hoverIndex));
      const hx = xAt(hi);
      ctx.strokeStyle = "rgba(77, 176, 140, 0.95)";
      ctx.lineWidth = 1.25;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(hx, margin.t);
      ctx.lineTo(hx, margin.t + chartH);
      ctx.stroke();

      for (let line = 0; line < nLines; line++) {
        const v = series[hi][line];
        if (!Number.isFinite(v)) continue;
        const color = TIMELINE_COLORS[line % TIMELINE_COLORS.length];
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(hx, yAt(v), 3.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#0c0f0e";
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }

    // Legend under the plot
    const legendY0 = margin.t + chartH + 22;
    const colW = Math.max(72, Math.min(110, (width - 16) / Math.min(nLines, 6)));
    ctx.font = '11px "Noto Sans", "Liberation Sans", system-ui, sans-serif';
    ctx.textBaseline = "middle";
    labels.slice(0, 12).forEach((label, i) => {
      const col = i % Math.max(1, Math.floor((width - 16) / colW));
      const row = Math.floor(i / Math.max(1, Math.floor((width - 16) / colW)));
      const lx = 8 + col * colW;
      const ly = legendY0 + row * 16;
      const color = TIMELINE_COLORS[i % TIMELINE_COLORS.length];
      ctx.fillStyle = color;
      ctx.fillRect(lx, ly - 4, 10, 8);
      ctx.fillStyle = "#c0cbc5";
      ctx.textAlign = "left";
      ctx.fillText(shortAxisLabel(label, 14), lx + 14, ly);
    });
    if (nLines > 12) {
      ctx.fillStyle = "#8d9894";
      ctx.fillText(`+${nLines - 12} more`, width - 60, legendY0);
    }
  }

  function indexFromEvent(event) {
    if (n <= 0) return null;
    const bounds = canvas.getBoundingClientRect();
    const scaleX = width / bounds.width;
    const px = (event.clientX - bounds.left) * scaleX;
    if (px < margin.l - 4 || px > margin.l + chartW + 4) return null;
    if (n === 1) return 0;
    const t = (px - margin.l) / chartW;
    return Math.max(0, Math.min(n - 1, Math.round(t * (n - 1))));
  }

  function showTimelineDetail(event, index) {
    const values = series[index] || [];
    // Sort by magnitude so the dominant series is readable first.
    const ranked = labels
      .map((label, line) => ({
        label,
        value: values[line],
        color: TIMELINE_COLORS[line % TIMELINE_COLORS.length],
      }))
      .filter((row) => Number.isFinite(row.value))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

    const rows = [
      `<dt>Step</dt><dd>${Number.isFinite(steps[index]) ? Math.round(steps[index]).toLocaleString() : "—"}</dd>`,
      ...ranked.slice(0, 10).map((row) =>
        `<dt><i class="swatch" style="background:${row.color}"></i>${pretty(row.label)}</dt>` +
        `<dd>${formatTimelineValue(row.value, yKind)}</dd>`
      ),
    ];
    if (ranked.length > 10) {
      rows.push(`<dt></dt><dd>+${ranked.length - 10} more</dd>`);
    }
    els.detail.innerHTML = `<dl>${rows.join("")}</dl>`;
    els.detail.classList.add("visible");
    els.detail.style.left = `${Math.min(event.clientX + 15, innerWidth - 260)}px`;
    els.detail.style.top = `${Math.min(event.clientY + 15, innerHeight - 280)}px`;
  }

  paint();
  canvas.style.cursor = "crosshair";
  canvas.onmousemove = (event) => {
    const index = indexFromEvent(event);
    if (index == null) {
      if (hoverIndex != null) {
        hoverIndex = null;
        paint();
      }
      hideDetail();
      return;
    }
    if (hoverIndex !== index) {
      hoverIndex = index;
      paint();
    }
    showTimelineDetail(event, index);
  };
  canvas.onmouseleave = () => {
    hoverIndex = null;
    paint();
    hideDetail();
  };
}

function renderTimelines(frame) {
  const dims = frame.task_space.dimensions;
  const maxTimelineFrames = 360;
  const stride = Math.max(1, Math.ceil(state.frames.length / maxTimelineFrames));
  const sampledFrames = state.frames.filter((_, index) => index % stride === 0 || index === state.frames.length - 1);
  const sampledActive = Math.min(sampledFrames.length - 1, Math.round(state.index / stride));
  const steps = sampledFrames.map((item) => item.step);
  els.timelines.replaceChildren(...dims.map((dim) => {
    const figure = document.createElement("article");
    figure.className = "timeline-figure";
    const labels = frame.task_space.coordinates[dim];
    figure.innerHTML = `<header><strong>${pretty(dim)}</strong><span>probability mass · hover for values</span></header><canvas></canvas>`;
    const series = sampledFrames.map((item) => marginal(item, dim));
    requestAnimationFrame(() =>
      drawTimeline(figure.querySelector("canvas"), series, labels, sampledActive, {
        steps,
        yKind: "probability",
      })
    );
    return figure;
  }));
}

/** Per-stage scalars from frame metadata: LP-ACRL's ESS, or V6's frontier. They
 * aren't per-cell metrics, but drawTimeline's (series, labels) contract is
 * generic enough to reuse directly. */
function renderHealth() {
  const maxTimelineFrames = 360;
  const stride = Math.max(1, Math.ceil(state.frames.length / maxTimelineFrames));
  const sampledFrames = state.frames.filter((_, index) => index % stride === 0 || index === state.frames.length - 1);
  const sampledActive = Math.min(sampledFrames.length - 1, Math.round(state.index / stride));
  const steps = sampledFrames.map((item) => item.step);
  const spec = healthSpec(sampledFrames);
  const figure = document.createElement("article");
  figure.className = "timeline-figure";
  figure.innerHTML = `<header><strong>${spec.title}</strong><span>${spec.subtitle}</span></header><canvas></canvas>`;
  requestAnimationFrame(() =>
    drawTimeline(
      figure.querySelector("canvas"),
      spec.series,
      spec.labels,
      sampledActive,
      { steps, yKind: spec.yKind },
    )
  );
  els.health.replaceChildren(figure);
}

function render() {
  const frame = state.frames[state.index];
  showEmpty(!frame);
  if (!frame) return;
  els.slider.max = Math.max(0, state.frames.length - 1);
  els.slider.value = state.index;
  els.count.textContent = `${state.index + 1} / ${state.frames.length}`;
  // frame.step is the publisher's step key. V5 UED publishes global_control_steps
  // here (not PPO iteration). Prefer that label; append stage when present.
  els.step.textContent = frame.step.toLocaleString();
  const stage = frame?.metadata?.frame?.stage_index;
  els.stepLabel.textContent = Number.isFinite(stage)
    ? `control step · stage ${stage}`
    : "control step";
  renderAtlas(frame);
  renderTriptych(frame);
  renderTimelines(frame);
  renderHealth();
}

function setMode(mode) {
  state.mode = mode;
  els.live.classList.toggle("active", mode === "live");
  els.history.classList.toggle("active", mode === "history");
  if (mode === "live") { stopPlayback(); state.index = Math.max(0, state.frames.length - 1); render(); }
}
function showEmpty(value) { els.empty.classList.toggle("visible", value); }
function stopPlayback() { clearInterval(state.timer); state.timer = null; els.play.textContent = "▶"; }
function togglePlayback() {
  if (state.timer) return stopPlayback();
  setMode("history"); els.play.textContent = "Ⅱ";
  state.timer = setInterval(() => {
    if (state.index >= state.frames.length - 1) state.index = 0;
    else state.index++;
    render();
  }, 220);
}

function lockSelection() {
  state.selectionLocked = true;
}

els.run.addEventListener("change", () => selectRun(els.run.value));
els.metric.addEventListener("change", render);
els.x.addEventListener("change", () => {
  lockSelection();
  state.selection.x = els.x.value;
  if (state.selection.y === state.selection.x) {
    state.selection.y = state.frames[0].task_space.dimensions.find((d) => d !== state.selection.x);
  }
  if (state.selection.facet === state.selection.x) state.selection.facet = "__none__";
  updateDimensionControls();
  render();
});
els.y.addEventListener("change", () => {
  lockSelection();
  state.selection.y = els.y.value;
  if (state.selection.facet === state.selection.y) state.selection.facet = "__none__";
  updateDimensionControls();
  render();
});
els.facet.addEventListener("change", () => {
  lockSelection();
  state.selection.facet = els.facet.value;
  updateDimensionControls();
  render();
});
els.filter.addEventListener("change", () => {
  lockSelection();
  state.selection.filterIndex = Number(els.filter.value);
  render();
});
els.live.addEventListener("click", () => setMode("live"));
els.history.addEventListener("click", () => setMode("history"));
els.slider.addEventListener("input", () => { setMode("history"); state.index = Number(els.slider.value); render(); });
els.play.addEventListener("click", togglePlayback);
window.addEventListener("resize", () => requestAnimationFrame(render));
document.addEventListener("visibilitychange", () => { if (document.hidden) stopPlayback(); });

loadRuns().catch((error) => {
  showEmpty(true);
  els.empty.querySelector("strong").textContent = "Dashboard could not load";
  els.empty.querySelector("span").textContent = error.message;
});

setInterval(async () => {
  if (els.run.value) return;
  try { await loadRuns(); } catch { /* The empty state remains informative. */ }
}, 2000);
