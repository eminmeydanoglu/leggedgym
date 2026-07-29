import test from "node:test";
import assert from "node:assert/strict";

import {
  CELL_STATES, color, extent, formatValue, healthSpec, metricKind,
  paletteFor, pickDefaultMetric, pickDefaultSelection, pickSignals, pretty,
} from "../public/atlas_core.js";

const V5_DIMS = ["vx_bin", "terrain_cell"];
const V6_DIMS = ["vx_bin", "terrain_family", "terrain_level"];

test("metric semantics come from the name, so new metrics need no UI change", () => {
  assert.equal(metricKind("state"), "categorical");
  assert.equal(metricKind("unlocked"), "binary");
  assert.equal(metricKind("learning_progress"), "diverging");
  assert.equal(metricKind("success_probability_delta"), "diverging");
  assert.equal(metricKind("sampling_probability"), "probability");
  assert.equal(metricKind("timeout_fraction"), "unit");
  assert.equal(metricKind("source_replay_share"), "unit");
  assert.equal(metricKind("window_episode_count"), "count");
  assert.equal(metricKind("mean_linear_error"), "sequential");
});

test("cell state renders as discrete buckets, never as a gradient", () => {
  const range = extent([[0, 1, 2, 3]], "state");
  assert.deepEqual(range, [0, CELL_STATES.length - 1]);
  const colors = [0, 1, 2, 3].map((value) => color(value, range, "state"));
  assert.deepEqual(colors, CELL_STATES.map((entry) => entry.color));
  // A blended value would mean "half mastered", which is not a state.
  assert.equal(color(1.4, range, "state"), CELL_STATES[1].color);
  assert.equal(paletteFor("state").length, CELL_STATES.length);
});

test("state values format as their published names", () => {
  assert.equal(formatValue(2, "state"), "mastered");
  assert.equal(formatValue(1, "state", { stateNames: ["kilitli", "cephe", "usta", "kararsız"] }), "cephe");
  assert.equal(formatValue(null, "state"), "—");
  assert.equal(formatValue(null, "state", { compact: true }), "");
  assert.equal(formatValue(2, "state", { compact: true }), "M");
});

test("signed deltas keep a symmetric scale around zero", () => {
  const range = extent([[-0.02, 0.11]], "success_probability_delta");
  assert.deepEqual(range, [-0.11, 0.11]);
  assert.equal(formatValue(0.031, "success_probability_delta"), "+0.031");
  assert.equal(formatValue(-0.031, "success_probability_delta"), "-0.031");
});

test("counts and probabilities keep their V5 formatting", () => {
  assert.equal(formatValue(1234, "task_assignment_count"), "1,234");
  assert.equal(formatValue(1234, "task_assignment_count", { compact: true }), "1234");
  assert.equal(formatValue(0.125, "sampling_probability"), ".125");
  assert.deepEqual(extent([[0.1, 0.4]], "sampling_probability"), [0, 0.4]);
  assert.deepEqual(extent([[0.1, 0.4]], "timeout_fraction"), [0, 1]);
});

test("V6 defaults to level x speed per family; V5 keeps its single wide map", () => {
  assert.deepEqual(pickDefaultSelection(V6_DIMS), {
    x: "terrain_level", y: "vx_bin", facet: "terrain_family", filterDim: null, filterIndex: 0,
  });
  const v5 = pickDefaultSelection(V5_DIMS);
  assert.equal(v5.x, "terrain_cell");
  assert.equal(v5.y, "vx_bin");
  assert.equal(v5.facet, "__none__");
});

test("the opening metric is the most decision-bearing one available", () => {
  assert.equal(pickDefaultMetric(["success_probability", "state", "sampling_probability"]), "state");
  assert.equal(pickDefaultMetric(["performance", "sampling_probability"]), "sampling_probability");
  assert.equal(pickDefaultMetric(["mean_yaw_error"]), "mean_yaw_error");
});

test("the signal chain follows whichever curriculum published the frame", () => {
  assert.deepEqual(
    pickSignals(["performance", "learning_progress", "sampling_probability", "state"]),
    ["performance", "learning_progress", "sampling_probability"],
  );
  assert.deepEqual(
    pickSignals(["state", "success_probability", "success_probability_delta", "sampling_probability"]),
    ["success_probability", "success_probability_delta", "sampling_probability"],
  );
  // A partial run still gets whatever it does publish, without empty panels.
  assert.deepEqual(pickSignals(["sampling_probability", "unlocked"]), ["sampling_probability"]);
});

test("sampler health switches between ESS and the frontier lifecycle", () => {
  const v5 = healthSpec([{ metadata: { frame: { diagnostics: { effective_sample_size: 12, target_ess: 20 } } } }]);
  assert.equal(v5.title, "Effective sample size");
  assert.deepEqual(v5.series, [[12, 20]]);

  const v6 = healthSpec([{
    metadata: { frame: { diagnostics: {
      unlocked_cell_count: 8, frontier_cell_count: 7, mastered_cell_count: 1, unstable_cell_count: 0,
    } } },
  }]);
  assert.equal(v6.title, "Curriculum frontier");
  assert.deepEqual(v6.series, [[8, 7, 1, 0]]);
  assert.equal(v6.labels.length, 4);
});

test("metric and dimension labels stay human-readable without a lookup entry", () => {
  assert.equal(pretty("terrain_family"), "Terrain family");
  assert.equal(pretty("state"), "Cell state");
  assert.equal(pretty("some_new_metric"), "some new metric");
});
