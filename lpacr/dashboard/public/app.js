const $ = (selector) => document.querySelector(selector);
const els = {
  atlas: $("#atlas"), triptych: $("#triptych"), timelines: $("#timelines"),
  run: $("#runSelect"), x: $("#xDimension"), y: $("#yDimension"),
  facet: $("#facetDimension"), filter: $("#filterValue"), filterWrap: $("#filterWrap"),
  metric: $("#metricSelect"), live: $("#liveButton"), history: $("#historyButton"),
  colorLegend: $("#colorLegend"),
  slider: $("#timeSlider"), play: $("#playButton"), count: $("#frameCount"),
  step: $("#stepValue"), detail: $("#detail"), empty: $("#empty"),
  panelDimension: $("#panelDimension"),
};

const state = {
  runs: [], frames: [], index: 0, mode: "live", stream: null, timer: null,
  selection: { x: null, y: null, facet: null, filterDim: null, filterIndex: 0 },
};
const PALETTES = {
  sequential: ["#141c1a", "#1a332e", "#245a4d", "#2f8a72", "#5cb89a", "#b8e0d0"],
  probability: ["#141c1a", "#1c312c", "#256655", "#348a6f", "#5aad8e", "#c5dfc0"],
  diverging: ["#3d6eb0", "#6a9fc4", "#1a2220", "#c98a52", "#e0b86a"],
  count: ["#141c1a", "#262e2b", "#4d5a54", "#7a8a82", "#b5c4ba"],
};
const LABELS = {
  terrain_type: "Terrain", terrain_level: "Level", vx_bin: "|vx|", yaw_bin: "|ωz|",
  performance: "Performance", learning_progress: "Learning progress",
  sampling_probability: "Sampling probability", success_rate: "Success rate",
  sample_count: "Sample count",
};
const pretty = (value) => LABELS[value] || value.replaceAll("_", " ");

async function json(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error((await response.json()).error || response.statusText);
  return response.json();
}

function setOptions(select, values, selected) {
  select.replaceChildren(...values.map((value) => {
    const option = document.createElement("option");
    option.value = value; option.textContent = pretty(value);
    option.selected = value === selected;
    return option;
  }));
}

async function loadRuns(preferred) {
  state.runs = (await json("/api/runs")).runs;
  setOptions(els.run, state.runs.map((run) => run.run_id), preferred || state.runs[0]?.run_id);
  if (els.run.value) await selectRun(els.run.value);
  else showEmpty(true);
}

async function selectRun(runId) {
  closeStream();
  const payload = await json(`/api/runs/${encodeURIComponent(runId)}/frames`);
  state.frames = payload.frames.sort((a, b) => a.step - b.step);
  state.index = Math.max(0, state.frames.length - 1);
  state.selection = { x: null, y: null, facet: null, filterDim: null, filterIndex: 0 };
  initializeControls();
  openStream(runId);
  render();
}

function initializeControls() {
  const frame = state.frames[0];
  if (!frame) return;
  const dims = frame.task_space.dimensions;
  state.selection.x ||= dims.includes("vx_bin") ? "vx_bin" : dims.at(-1);
  state.selection.y ||= dims.includes("terrain_level") ? "terrain_level" : dims.at(-2);
  state.selection.facet ||= dims.find((d) => d === "terrain_type") || dims.find((d) => ![state.selection.x, state.selection.y].includes(d));
  updateDimensionControls();
  setOptions(els.metric, Object.keys(frame.metrics), els.metric.value || "sampling_probability");
}

function updateDimensionControls() {
  const frame = state.frames[0];
  if (!frame) return;
  const dims = frame.task_space.dimensions;
  setOptions(els.x, dims, state.selection.x);
  setOptions(els.y, dims.filter((d) => d !== state.selection.x), state.selection.y);
  const facetValues = ["__none__", ...dims.filter((d) => ![state.selection.x, state.selection.y].includes(d))];
  setOptions(els.facet, facetValues, state.selection.facet || "__none__");
  const excluded = [state.selection.x, state.selection.y, state.selection.facet];
  state.selection.filterDim = dims.find((d) => !excluded.includes(d)) || null;
  if (state.selection.filterDim) {
    els.filterWrap.firstChild.textContent = `${pretty(state.selection.filterDim)} `;
    const values = frame.task_space.coordinates[state.selection.filterDim];
    setOptions(els.filter, values.map((_, i) => String(i)), String(Math.min(state.selection.filterIndex, values.length - 1)));
    [...els.filter.options].forEach((option, i) => option.textContent = values[i]);
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
    initializeControls();
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
  const width = coords[x].length, height = coords[y].length;
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

function color(value, range, metric) {
  if (!Number.isFinite(value)) return "#1a2220";
  const palette = metric === "learning_progress" ? PALETTES.diverging :
    metric === "sampling_probability" ? PALETTES.probability :
    metric === "sample_count" ? PALETTES.count : PALETTES.sequential;
  const t = Math.max(0, Math.min(.9999, (value - range[0]) / (range[1] - range[0] || 1)));
  const scaled = t * (palette.length - 1);
  const a = palette[Math.floor(scaled)], b = palette[Math.ceil(scaled)];
  const mix = scaled - Math.floor(scaled);
  const channels = [1, 3, 5].map((offset) => Math.round(
    parseInt(a.slice(offset, offset + 2), 16) * (1 - mix) + parseInt(b.slice(offset, offset + 2), 16) * mix
  ));
  return `rgb(${channels.join(",")})`;
}

function renderColorLegend(metric) {
  const palette = metric === "learning_progress" ? PALETTES.diverging :
    metric === "sampling_probability" ? PALETTES.probability :
    metric === "sample_count" ? PALETTES.count : PALETTES.sequential;
  const labels = metric === "learning_progress"
    ? ["Negative", "Zero", "Positive"]
    : metric === "sample_count"
      ? ["Few", "Many"]
      : ["Low", "High"];
  const start = document.createElement("span");
  start.className = "legend-label";
  start.textContent = labels[0];
  const end = document.createElement("span");
  end.className = "legend-label";
  end.textContent = labels.at(-1);
  const swatches = palette.map((value) => {
    const swatch = document.createElement("i");
    swatch.className = "legend-swatch";
    swatch.style.background = value;
    return swatch;
  });
  if (labels.length === 3) {
    const zero = document.createElement("span");
    zero.className = "legend-label";
    zero.textContent = labels[1];
    els.colorLegend.replaceChildren(start, ...swatches.slice(0, 2), zero, ...swatches.slice(2), end);
  } else {
    els.colorLegend.replaceChildren(start, ...swatches, end);
  }
}

function setupCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  return { ctx, width: rect.width, height: rect.height };
}

function drawHeatmapAxes(ctx, width, height, margin, xLabels, yLabels, { fontSize = 12, showTicks = true } = {}) {
  const font = `${fontSize}px "Noto Sans", "Liberation Sans", system-ui, sans-serif`;
  const axisColor = "#6e7d78";
  const tickColor = "#8d9894";
  const cellW = (width - margin.l - margin.r) / xLabels.length;
  const cellH = (height - margin.t - margin.b) / yLabels.length;

  if (showTicks) {
    ctx.fillStyle = tickColor;
    ctx.font = font;
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    xLabels.forEach((label, i) => {
      ctx.fillText(label, margin.l + (i + .5) * cellW, height - Math.max(8, fontSize - 2));
    });
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    yLabels.forEach((label, i) => {
      ctx.fillText(label, margin.l - 8, margin.t + (yLabels.length - i - .5) * cellH);
    });
  }

  ctx.strokeStyle = axisColor;
  ctx.fillStyle = axisColor;
  ctx.lineWidth = 1;
  const xY = height - Math.round(margin.b * 0.52);
  ctx.beginPath();
  ctx.moveTo(margin.l, xY);
  ctx.lineTo(width - margin.r - 2, xY);
  ctx.lineTo(width - margin.r - 9, xY - 3.5);
  ctx.moveTo(width - margin.r - 2, xY);
  ctx.lineTo(width - margin.r - 9, xY + 3.5);
  ctx.stroke();
  const yX = Math.max(12, Math.round(margin.l * 0.28));
  ctx.beginPath();
  ctx.moveTo(yX, height - margin.b);
  ctx.lineTo(yX, margin.t + 2);
  ctx.lineTo(yX - 3.5, margin.t + 9);
  ctx.moveTo(yX, margin.t + 2);
  ctx.lineTo(yX + 3.5, margin.t + 9);
  ctx.stroke();

  ctx.font = font;
  ctx.textAlign = "right";
  ctx.textBaseline = "alphabetic";
  ctx.fillText(pretty(state.selection.x), width - margin.r, xY - 6);
  ctx.save();
  ctx.translate(yX + 10, height - margin.b);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "left";
  ctx.fillText(pretty(state.selection.y), 0, 0);
  ctx.restore();
}

function drawHeatmap(canvas, matrix, range, metric, frame, facetIndex, mode = "full") {
  const { ctx, width, height } = setupCanvas(canvas);
  const xLabels = frame.task_space.coordinates[state.selection.x];
  const yLabels = frame.task_space.coordinates[state.selection.y];
  // bare: color cells only; axes: dim names + arrows; full: ticks + arrows (atlas)
  const margin = mode === "bare"
    ? { l: 8, r: 8, t: 8, b: 8 }
    : mode === "axes"
      ? { l: 48, r: 12, t: 10, b: 44 }
      : { l: 54, r: 10, t: 10, b: 50 };
  const cellW = (width - margin.l - margin.r) / xLabels.length;
  const cellH = (height - margin.t - margin.b) / yLabels.length;
  ctx.clearRect(0, 0, width, height);
  matrix.forEach((row, yi) => row.forEach((value, xi) => {
    ctx.fillStyle = color(value, range, metric);
    ctx.fillRect(margin.l + xi * cellW + .5, margin.t + (yLabels.length - 1 - yi) * cellH + .5, Math.max(0, cellW - 1), Math.max(0, cellH - 1));
  }));
  if (mode !== "bare") {
    drawHeatmapAxes(ctx, width, height, margin, xLabels, yLabels, {
      fontSize: mode === "axes" ? 11 : 12,
      showTicks: true,
    });
  }
  canvas.onmousemove = (event) => {
    const rect = canvas.getBoundingClientRect();
    const xi = Math.floor((event.clientX - rect.left - margin.l) / cellW);
    const displayY = Math.floor((event.clientY - rect.top - margin.t) / cellH);
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
  const rows = frame.task_space.dimensions.map((dim) =>
    `<dt>${pretty(dim)}</dt><dd>${frame.task_space.coordinates[dim][indices[dim] || 0]}</dd>`
  );
  for (const metric of ["sampling_probability", "learning_progress", "performance", "success_rate", "sample_count"]) {
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
  if (metric === "sampling_probability" || metric === "success_rate") return `${(value * 100).toFixed(2)}%`;
  if (metric === "sample_count") return Math.round(value).toLocaleString();
  return value.toFixed(3);
}

function renderAtlas(frame) {
  const facet = state.selection.facet;
  const facetValues = facet && facet !== "__none__" ? frame.task_space.coordinates[facet] : ["All tasks"];
  const metric = els.metric.value;
  renderColorLegend(metric);
  const matrices = facetValues.map((_, i) => sliceMatrix(frame, metric, i));
  const range = extent(matrices, metric);
  els.panelDimension.textContent = facet && facet !== "__none__" ? pretty(facet) : "Single view";
  els.atlas.replaceChildren(...facetValues.map((label, i) => {
    const figure = document.createElement("article");
    figure.className = "heatmap-figure";
    const kind = facet && facet !== "__none__" ? pretty(facet) : "Panel";
    figure.innerHTML = `<h3><span>${kind}</span><strong>${label}</strong></h3><canvas aria-label="${label} ${pretty(metric)} heatmap"></canvas>`;
    requestAnimationFrame(() => drawHeatmap(figure.querySelector("canvas"), matrices[i], range, metric, frame, i));
    return figure;
  }));
}

function renderTriptych(frame) {
  const signals = ["performance", "learning_progress", "sampling_probability"].filter((m) => frame.metrics[m]);
  const facetIndex = 0;
  const facetLabel = state.selection.facet && state.selection.facet !== "__none__"
    ? frame.task_space.coordinates[state.selection.facet][facetIndex]
    : null;
  const axesHint = `${pretty(state.selection.y)} × ${pretty(state.selection.x)}`;
  const context = facetLabel ? `${facetLabel} · ${axesHint}` : axesHint;
  els.triptych.replaceChildren(...signals.map((metric) => {
    const figure = document.createElement("article");
    figure.className = "signal-figure";
    figure.innerHTML = `<header><strong>${pretty(metric)}</strong><span title="Y × X axes">${context}</span></header><canvas aria-label="${pretty(metric)} heatmap, ${context}"></canvas>`;
    const matrix = sliceMatrix(frame, metric, facetIndex);
    // Axes + ticks: bare color grids are not readable without X/Y encoding
    requestAnimationFrame(() => drawHeatmap(figure.querySelector("canvas"), matrix, extent(matrix, metric), metric, frame, facetIndex, "axes"));
    return figure;
  }));
}

function marginal(frame, dim, metric = "sampling_probability") {
  const values = new Array(frame.task_space.coordinates[dim].length).fill(0);
  const counts = new Array(values.length).fill(0);
  const shape = strides(frame);
  frame.metrics[metric].forEach((value, index) => {
    const dimPosition = shape.dims.indexOf(dim);
    const coordinate = Math.floor(index / shape.strides[dimPosition]) % shape.lengths[dimPosition];
    if (Number.isFinite(value)) { values[coordinate] += value; counts[coordinate]++; }
  });
  return values;
}

function drawTimeline(canvas, series, labels, active) {
  const { ctx, width, height } = setupCanvas(canvas);
  const margin = { l: 4, r: 5, t: 7, b: 22 };
  const chartW = width - margin.l - margin.r, chartH = height - margin.t - margin.b;
  const max = Math.max(...series.flat(), 1e-9);
  // Distinct but non-neon series colors for dark panels
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
    ctx.fillText(label, 4 + i * Math.max(60, width / 7), height - 6);
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
  els.step.textContent = frame.step.toLocaleString();
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

els.run.addEventListener("change", () => selectRun(els.run.value));
els.metric.addEventListener("change", render);
els.x.addEventListener("change", () => {
  state.selection.x = els.x.value;
  if (state.selection.y === state.selection.x) state.selection.y = state.frames[0].task_space.dimensions.find((d) => d !== state.selection.x);
  updateDimensionControls(); render();
});
els.y.addEventListener("change", () => { state.selection.y = els.y.value; updateDimensionControls(); render(); });
els.facet.addEventListener("change", () => { state.selection.facet = els.facet.value; updateDimensionControls(); render(); });
els.filter.addEventListener("change", () => { state.selection.filterIndex = Number(els.filter.value); render(); });
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
