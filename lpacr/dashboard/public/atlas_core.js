/**
 * Pure presentation rules shared by the atlas UI and its tests.
 *
 * Nothing here touches the DOM: metric semantics (is this a probability, a
 * signed delta, a categorical state?) are decided from the metric *name* so a
 * publisher can add a metric without the dashboard learning about it first.
 */

// Sequential / probability: low = light, high = dark (readable on dark UI).
export const PALETTES = {
  sequential: ["#dce8e2", "#a8c9b8", "#6fa08a", "#3f7a64", "#245445", "#132c24"],
  probability: ["#e4eee8", "#b7d4c4", "#7fad95", "#4a8268", "#2b5a48", "#16352b"],
  diverging: ["#3d6eb0", "#6a9fc4", "#1a2220", "#c98a52", "#e0b86a"],
  count: ["#e2e8e4", "#b0bbb4", "#7a8a82", "#4d5a54", "#2c342f", "#161c19"],
};

/**
 * V6 cell lifecycle. The order mirrors CELL_STATE_* in the frontier curriculum;
 * a frame may override the names through metadata.frame.cell_state_names.
 */
export const CELL_STATES = [
  { name: "locked", color: "#26302d" },
  { name: "frontier", color: "#d9a24c" },
  { name: "mastered", color: "#4d9f7d" },
  { name: "unstable", color: "#c4696b" },
];

export const LABELS = {
  terrain_type: "Terrain", terrain_level: "Level", terrain_cell: "Terrain",
  terrain_family: "Terrain family", vx_bin: "|vx|", yaw_bin: "|ωz|",
  performance: "Performance", learning_progress: "Learning progress",
  sampling_probability: "Sampling probability", success_rate: "Success rate",
  sample_count: "Sample count", stage_episode_count: "Stage episodes",
  task_assignment_count: "Assignments", task_completion_count: "Completions",
  state: "Cell state", success_probability: "Success probability",
  success_probability_delta: "Success Δ / stage",
  window_episode_count: "Window episodes", window_success_count: "Window successes",
  window_fill_fraction: "Window fill", episodes_until_eligible: "Episodes to eligible",
  consecutive_mastery: "Consecutive mastery", assignment_count: "Assignments",
  completion_count: "Completions", completion_count_last_stage: "Episodes last stage",
  standstill_placement_count: "Standstill placements",
  unlocked_at_stage: "Unlocked at stage", mastered_at_stage: "Mastered at stage",
  stages_since_unlock: "Stages since unlock", stages_since_mastery: "Stages since mastery",
  mean_linear_error: "Linear tracking error", mean_yaw_error: "Yaw tracking error",
  mean_episode_length: "Episode length", mean_episodic_return: "Episodic return",
  timeout_fraction: "Timeout fraction",
  source_frontier_count: "Frontier draws", source_replay_count: "Replay draws",
  source_uniform_count: "Uniform draws", source_frontier_share: "Frontier share",
  source_replay_share: "Replay share", source_uniform_share: "Uniform share",
};

export const pretty = (value) => LABELS[value] || String(value).replaceAll("_", " ");

const BINARY_METRICS = new Set([
  "unlocked", "mastered", "unstable", "eligible", "eligible_for_lp",
]);

/**
 * Classify a metric from its name.
 * @returns {"categorical"|"binary"|"diverging"|"probability"|"unit"|"count"|"sequential"}
 */
export function metricKind(metric) {
  const name = String(metric);
  if (name === "state") return "categorical";
  if (BINARY_METRICS.has(name)) return "binary";
  if (name === "learning_progress" || name === "effective_learning_progress") return "diverging";
  if (name.endsWith("_delta")) return "diverging";
  if (name.endsWith("_share") || name.endsWith("_fraction") || name.endsWith("_rate")) return "unit";
  if (name.includes("probability")) return "probability";
  if (name.endsWith("_count") || name === "sample_count") return "count";
  return "sequential";
}

export function paletteFor(metric) {
  switch (metricKind(metric)) {
    case "categorical": return CELL_STATES.map((entry) => entry.color);
    case "diverging": return PALETTES.diverging;
    case "probability":
    case "unit": return PALETTES.probability;
    case "count": return PALETTES.count;
    default: return PALETTES.sequential;
  }
}

/** Colour scale bounds for one metric over the values currently on screen. */
export function extent(values, metric) {
  const kind = metricKind(metric);
  if (kind === "categorical") return [0, CELL_STATES.length - 1];
  if (kind === "binary" || kind === "unit") return [0, 1];
  const clean = [values].flat(Infinity).filter(Number.isFinite);
  if (!clean.length) return [0, 1];
  if (kind === "diverging") {
    const bound = Math.max(...clean.map(Math.abs), 1e-9);
    return [-bound, bound];
  }
  if (kind === "probability") return [0, Math.max(...clean, 1e-9)];
  if (metric === "success_rate") return [0, 1];
  return [Math.min(...clean), Math.max(...clean)];
}

/** Colour for one cell. Categorical metrics snap to a bucket, never blend. */
export function color(value, range, metric) {
  if (!Number.isFinite(value)) return "#1a2220";
  const palette = paletteFor(metric);
  if (metricKind(metric) === "categorical") {
    return palette[Math.max(0, Math.min(palette.length - 1, Math.round(value)))];
  }
  const t = Math.max(0, Math.min(0.9999, (value - range[0]) / (range[1] - range[0] || 1)));
  const scaled = t * (palette.length - 1);
  const a = palette[Math.floor(scaled)];
  const b = palette[Math.ceil(scaled)];
  const mix = scaled - Math.floor(scaled);
  const channels = [1, 3, 5].map((offset) => Math.round(
    parseInt(a.slice(offset, offset + 2), 16) * (1 - mix)
    + parseInt(b.slice(offset, offset + 2), 16) * mix
  ));
  return `rgb(${channels.join(",")})`;
}

/** Ink for an in-cell label, chosen against the fill lightness. */
export function cellInk(value, range, metric) {
  if (!Number.isFinite(value)) return "#8d9894";
  const kind = metricKind(metric);
  if (kind === "categorical") return Math.round(value) === 0 ? "#8d9894" : "#121a17";
  const t = Math.max(0, Math.min(1, (value - range[0]) / (range[1] - range[0] || 1)));
  if (kind === "diverging") {
    return Math.abs(t - 0.5) < 0.12 ? "#c5cdc9" : (t > 0.5 ? "#1a1410" : "#e8ecea");
  }
  return t > 0.55 ? "#e8ecea" : "#152019";
}

const trimZero = (text) => (text.startsWith("0.") ? text.slice(1) : text);

/**
 * Format a metric value.
 * @param {number} value
 * @param {string} metric
 * @param {{ compact?: boolean, stateNames?: string[] }} [options]
 *   compact: in-cell label (terser, blank for unknown); otherwise tooltip text.
 */
export function formatValue(value, metric, options = {}) {
  const { compact = false, stateNames } = options;
  const kind = metricKind(metric);
  if (!Number.isFinite(value)) return compact ? "" : "—";
  if (kind === "categorical") {
    const names = stateNames || CELL_STATES.map((entry) => entry.name);
    const name = names[Math.round(value)] ?? String(Math.round(value));
    // In-cell: one letter. Truncating to "mast"/"fron" reads as noise, and the
    // legend right above the map already carries the full names.
    return compact ? name.slice(0, 1).toUpperCase() : name;
  }
  if (kind === "binary") {
    const yes = value >= 0.5;
    return compact ? (yes ? "✓" : "") : (yes ? "yes" : "no");
  }
  if (kind === "count") {
    return compact ? String(Math.round(value)) : Math.round(value).toLocaleString();
  }
  if (kind === "diverging") {
    const digits = Math.abs(value) >= 10 ? 1 : 3;
    return `${value > 0 ? "+" : ""}${value.toFixed(digits)}`;
  }
  if (kind === "probability" || kind === "unit" || metric === "success_rate") {
    return trimZero(value.toFixed(3));
  }
  if (compact) return Math.abs(value) >= 100 ? String(Math.round(value)) : value.toFixed(1);
  return value.toFixed(3);
}

/** Parse "stairs_up · L2" style labels into hierarchical type/level. */
export function parseGroupedLabel(label) {
  const text = String(label);
  const match = text.match(/^(.*?)\s*[·•|]\s*(L?\d+)\s*$/i);
  if (!match) return null;
  const level = match[2].toUpperCase().startsWith("L") ? match[2].toUpperCase() : `L${match[2]}`;
  return { group: match[1].trim(), leaf: level, raw: text };
}

export function groupLabels(labels) {
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

export function shortAxisLabel(label, maxLen = 14) {
  const text = String(label);
  if (text.length <= maxLen) return text;
  const noUnit = text.replace(/\s*m\/s\s*$/i, "").trim();
  if (noUnit.length <= maxLen) return noUnit;
  return `${text.slice(0, maxLen - 1)}…`;
}

/**
 * Default axes per curriculum shape.
 *
 * V6 advances the frontier along *both* level and speed, so level×|vx| per
 * terrain family is the picture that shows where the curriculum actually is.
 * V5 has one composite terrain axis and gets the single wide map it was tuned
 * for.
 */
export function pickDefaultSelection(dims) {
  const has = (name) => dims.includes(name);
  if (has("terrain_family") && has("terrain_level") && has("vx_bin")) {
    return { x: "terrain_level", y: "vx_bin", facet: "terrain_family", filterDim: null, filterIndex: 0 };
  }
  let x;
  if (has("terrain_cell")) x = "terrain_cell";
  else if (has("terrain_type")) x = "terrain_type";
  else if (has("terrain_level")) x = "terrain_level";
  else x = dims[0];
  const y = has("vx_bin") && x !== "vx_bin" ? "vx_bin" : (dims.find((d) => d !== x) || x);
  return { x, y, facet: "__none__", filterDim: null, filterIndex: 0 };
}

/** Metric shown first when a run is opened: the most decision-bearing one. */
export function pickDefaultMetric(metricNames) {
  const preferred = ["state", "sampling_probability", "learning_progress", "performance"];
  return preferred.find((name) => metricNames.includes(name)) || metricNames[0];
}

const SIGNAL_CHAINS = [
  ["performance", "learning_progress", "sampling_probability"],
  ["success_probability", "success_probability_delta", "sampling_probability"],
];

/**
 * The three-panel signal chain: what the learner scores, how that is moving,
 * and where the sampler puts its mass. V6 has no learning-progress signal, so
 * the per-stage success delta stands in for it.
 */
export function pickSignals(metricNames) {
  const available = new Set(metricNames);
  const chain = SIGNAL_CHAINS.find((names) => names.every((name) => available.has(name)));
  if (chain) return chain;
  return SIGNAL_CHAINS.flat().filter((name, index, all) =>
    available.has(name) && all.indexOf(name) === index).slice(0, 3);
}

/**
 * Sampler-health series: LP-ACRL reports effective sample size, the frontier
 * reports how the 240 cells are distributed over the lifecycle. Both are
 * per-stage scalars carried in frame metadata, not per-cell metrics.
 */
export function healthSpec(frames) {
  const diagnostics = frames.at(-1)?.metadata?.frame?.diagnostics || {};
  if ("unlocked_cell_count" in diagnostics || "frontier_cell_count" in diagnostics) {
    const keys = ["unlocked_cell_count", "frontier_cell_count", "mastered_cell_count", "unstable_cell_count"];
    return {
      title: "Curriculum frontier",
      subtitle: "cells by lifecycle state · hover for values",
      labels: ["Unlocked", "Frontier", "Mastered", "Unstable"],
      yKind: "count",
      series: frames.map((frame) => keys.map((key) => {
        const value = Number(frame?.metadata?.frame?.diagnostics?.[key]);
        return Number.isFinite(value) ? value : 0;
      })),
    };
  }
  return {
    title: "Effective sample size",
    subtitle: "uniform-equivalent cell count · hover for values",
    labels: ["Effective sample size", "Target ESS"],
    yKind: "count",
    series: frames.map((frame) => {
      const frameDiagnostics = frame?.metadata?.frame?.diagnostics || {};
      const ess = Number(frameDiagnostics.effective_sample_size);
      const target = Number(frameDiagnostics.target_ess);
      return [Number.isFinite(ess) ? ess : 0, Number.isFinite(target) ? target : 0];
    }),
  };
}
