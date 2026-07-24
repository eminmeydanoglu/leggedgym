const $ = (selector) => document.querySelector(selector);
const els = {
  atlas: $("#atlas"), triptych: $("#triptych"), timelines: $("#timelines"),
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
// Sequential / probability: low = light, high = dark (readable on dark UI).
const PALETTES = {
  sequential: ["#dce8e2", "#a8c9b8", "#6fa08a", "#3f7a64", "#245445", "#132c24"],
  probability: ["#e4eee8", "#b7d4c4", "#7fad95", "#4a8268", "#2b5a48", "#16352b"],
  diverging: ["#3d6eb0", "#6a9fc4", "#1a2220", "#c98a52", "#e0b86a"],
  count: ["#e2e8e4", "#b0bbb4", "#7a8a82", "#4d5a54", "#2c342f", "#161c19"],
};
const LABELS = {
  terrain_type: "Terrain", terrain_level: "Level", terrain_cell: "Terrain",
  vx_bin: "|vx|", yaw_bin: "|ωz|",
  performance: "Performance", learning_progress: "Learning progress",
  sampling_probability: "Sampling probability", success_rate: "Success rate",
  sample_count: "Sample count", stage_episode_count: "Stage episodes",
  task_assignment_count: "Assignments", task_completion_count: "Completions",
};
const pretty = (value) => LABELS[value] || String(value).replaceAll("_", " ");

/** Prefer single map: terrain across X (many cells), velocity on Y (few bins). */
function pickDefaultSelection(dims) {
  const has = (name) => dims.includes(name);
  let x = null;
  if (has("terrain_cell")) x = "terrain_cell";
  else if (has("terrain_type")) x = "terrain_type";
  else if (has("terrain_level")) x = "terrain_level";
  else x = dims[0];
  let y = null;
  if (has("vx_bin") && "vx_bin" !== x) y = "vx_bin";
  else y = dims.find((d) => d !== x) || x;
  return {
    x,
    y,
    facet: "__none__",
    filterDim: null,
    filterIndex: 0,
  };
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

/** Parse "stairs_up · L2" style labels into hierarchical type/level. */
function parseGroupedLabel(label) {
  const text = String(label);
  const match = text.match(/^(.*?)\s*[·•|]\s*(L?\d+)\s*$/i);
  if (!match) return null;
  const level = match[2].toUpperCase().startsWith("L") ? match[2].toUpperCase() : `L${match[2]}`;
  return { group: match[1].trim(), leaf: level, raw: text };
}

function groupLabels(labels) {
  const parsed = labels.map(parseGroupedLabel);
  if (!parsed.length || !parsed.every(Boolean)) return null;
  const groups = [];
  let current = null;
  parsed.forEach((item, index) => {
    if (!current || current.name !== item.group) {
      current = { name: item.group, start: index, end: index, leaves: [] };
      groups.push(current);
    }
    current.end = index;
    current.leaves.push(item.leaf);
  });
  return groups;
}

function shortAxisLabel(label, maxLen = 14) {
  const text = String(label);
  if (text.length <= maxLen) return text;
  // Prefer dropping units for velocity labels.
  const noUnit = text.replace(/\s*m\/s\s*$/i, "").trim();
  if (noUnit.length <= maxLen) return noUnit;
  return `${text.slice(0, maxLen - 1)}…`;
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
  const preferredMetric = ["sampling_probability", "learning_progress", "performance"]
    .find((name) => name in frame.metrics) || Object.keys(frame.metrics)[0];
  if (!els.metric.value || !(els.metric.value in frame.metrics)) {
    setOptions(els.metric, Object.keys(frame.metrics), preferredMetric);
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

function extent(values, metric) {
  const clean = values.flat(Infinity).filter(Number.isFinite);
  if (!clean.length) return [0, 1];
  if (metric === "learning_progress") {
    const bound = Math.max(...clean.map(Math.abs), 1e-9);
    return [-bound, bound];
  }
  if (metric === "success_rate") return [0, 1];
  if (metric === "sampling_probability") return [0, Math.max(...clean, 1e-9)];
  return [Math.min(...clean), Math.max(...clean)];
}

function paletteFor(metric) {
  if (metric === "learning_progress") return PALETTES.diverging;
  if (metric === "sampling_probability") return PALETTES.probability;
  if (
    metric === "sample_count" || metric === "stage_episode_count" ||
    metric === "task_assignment_count" || metric === "task_completion_count"
  ) return PALETTES.count;
  return PALETTES.sequential;
}

function color(value, range, metric) {
  if (!Number.isFinite(value)) return "#1a2220";
  const palette = paletteFor(metric);
  const t = Math.max(0, Math.min(.9999, (value - range[0]) / (range[1] - range[0] || 1)));
  const scaled = t * (palette.length - 1);
  const a = palette[Math.floor(scaled)];
  const b = palette[Math.ceil(scaled)];
  const mix = scaled - Math.floor(scaled);
  const channels = [1, 3, 5].map((offset) => Math.round(
    parseInt(a.slice(offset, offset + 2), 16) * (1 - mix) + parseInt(b.slice(offset, offset + 2), 16) * mix
  ));
  return `rgb(${channels.join(",")})`;
}

/** Lightness 0–1 of a filled cell color; used to pick ink for in-cell labels. */
function cellInk(value, range, metric) {
  if (!Number.isFinite(value)) return "#8d9894";
  const t = Math.max(0, Math.min(1, (value - range[0]) / (range[1] - range[0] || 1)));
  // Probability/sequential: light cells → dark text; dark cells → light text.
  if (metric === "learning_progress") return Math.abs(t - 0.5) < 0.12 ? "#c5cdc9" : (t > 0.5 ? "#1a1410" : "#e8ecea");
  return t > 0.55 ? "#e8ecea" : "#152019";
}

function renderColorLegend(metric, range) {
  const palette = paletteFor(metric);
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
  if (metric === "learning_progress" && range) {
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

function formatCellValue(value, metric) {
  if (!Number.isFinite(value)) return "";
  if (metric === "sampling_probability" || metric === "success_rate") {
    const s = value.toFixed(3);
    return s.startsWith("0") ? s.slice(1) : s;
  }
  if (
    metric === "sample_count" || metric === "stage_episode_count" ||
    metric === "task_assignment_count" || metric === "task_completion_count"
  ) {
    return String(Math.round(value));
  }
  if (metric === "learning_progress") {
    const sign = value > 0 ? "+" : "";
    return `${sign}${value.toFixed(1)}`;
  }
  return value.toFixed(1);
}

function drawHeatmap(canvas, matrix, range, metric, frame, facetIndex, mode = "full") {
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
    if (showValues) {
      const text = formatCellValue(value, metric);
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
  for (const metric of [
    "sampling_probability", "learning_progress", "performance", "success_rate",
    "sample_count", "stage_episode_count", "task_assignment_count", "task_completion_count",
  ]) {
    if (frame.metrics[metric]) {
      const value = frame.metrics[metric][index];
      rows.push(`<dt>${pretty(metric)}</dt><dd>${format(value, metric)}</dd>`);
    }
  }
  els.detail.innerHTML = `<dl>${rows.join("")}</dl>`;
  els.detail.classList.add("visible");
  els.detail.style.left = `${Math.min(event.clientX + 15, innerWidth - 235)}px`;
  els.detail.style.top = `${Math.min(event.clientY + 15, innerHeight - 230)}px`;
}
function hideDetail() { els.detail.classList.remove("visible"); }
function format(value, metric) {
  if (!Number.isFinite(value)) return "—";
  if (metric === "sampling_probability" || metric === "success_rate") {
    const s = value.toFixed(3);
    return s.startsWith("0") ? s.slice(1) : s;
  }
  if (
    metric === "sample_count" || metric === "stage_episode_count" ||
    metric === "task_assignment_count" || metric === "task_completion_count"
  ) {
    return Math.round(value).toLocaleString();
  }
  return value.toFixed(3);
}

function renderAtlas(frame) {
  const facet = state.selection.facet;
  const faceted = facet && facet !== "__none__";
  const facetValues = faceted ? frame.task_space.coordinates[facet] : ["All tasks"];
  const metric = els.metric.value;
  const matrices = facetValues.map((_, i) => sliceMatrix(frame, metric, i));
  const range = extent(matrices, metric);
  renderColorLegend(metric, range);
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
    drawHeatmap(figure.querySelector("canvas"), matrices[i], range, metric, frame, i, "full");
  });
}

function renderTriptych(frame) {
  const signals = ["performance", "learning_progress", "sampling_probability"].filter((m) => frame.metrics[m]);
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

function drawTimeline(canvas, series, labels, active) {
  // Timelines use container width; measure after layout.
  const rect = canvas.parentElement?.getBoundingClientRect();
  const cssW = Math.max(240, rect?.width || canvas.clientWidth || 320);
  const cssH = Math.max(110, Math.round(cssW * 0.42));
  const { ctx, width, height } = setupCanvas(canvas, cssW, cssH);
  const margin = { l: 4, r: 5, t: 7, b: 22 };
  const chartW = width - margin.l - margin.r;
  const chartH = height - margin.t - margin.b;
  const max = Math.max(...series.flat(), 1e-9);
  const colors = ["#5cb89a", "#c9a05a", "#6a96c4", "#a8b86a", "#a87cb8", "#c97878", "#6aada0", "#c48a52", "#7a9ab8", "#a09088"];
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "#24302c";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, margin.t + chartH);
  ctx.lineTo(width, margin.t + chartH);
  ctx.stroke();
  labels.forEach((label, line) => {
    ctx.strokeStyle = colors[line % colors.length];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    series.forEach((values, i) => {
      const x = margin.l + (series.length === 1 ? 0 : i / (series.length - 1)) * chartW;
      const y = margin.t + chartH - (values[line] / max) * chartH;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  });
  const x = margin.l + (series.length === 1 ? 0 : active / (series.length - 1)) * chartW;
  ctx.strokeStyle = "#c5cdc9";
  ctx.lineWidth = 1;
  ctx.setLineDash([2, 3]);
  ctx.beginPath();
  ctx.moveTo(x, margin.t);
  ctx.lineTo(x, margin.t + chartH);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.font = '12px "Noto Sans", "Liberation Sans", system-ui, sans-serif';
  ctx.textAlign = "left";
  labels.slice(0, 6).forEach((label, i) => {
    ctx.fillStyle = colors[i];
    ctx.fillText(shortAxisLabel(label, 12), 4 + i * Math.max(60, width / 7), height - 6);
  });
}

function renderTimelines(frame) {
  const dims = frame.task_space.dimensions;
  const maxTimelineFrames = 360;
  const stride = Math.max(1, Math.ceil(state.frames.length / maxTimelineFrames));
  const sampledFrames = state.frames.filter((_, index) => index % stride === 0 || index === state.frames.length - 1);
  const sampledActive = Math.min(sampledFrames.length - 1, Math.round(state.index / stride));
  els.timelines.replaceChildren(...dims.map((dim) => {
    const figure = document.createElement("article");
    figure.className = "timeline-figure";
    const labels = frame.task_space.coordinates[dim];
    figure.innerHTML = `<header><strong>${pretty(dim)}</strong><span>probability mass</span></header><canvas></canvas>`;
    const series = sampledFrames.map((item) => marginal(item, dim));
    requestAnimationFrame(() => drawTimeline(figure.querySelector("canvas"), series, labels, sampledActive));
    return figure;
  }));
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
