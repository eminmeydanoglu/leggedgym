import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const dataDir = await mkdtemp(join(tmpdir(), "curriculum-atlas-"));
process.env.LPACRL_DATA_DIR = dataDir;
const { createDashboardServer, validateFrame } = await import("../server.mjs");
const server = createDashboardServer();
let origin;

const frame = {
  step: 10,
  wall_time: 1234,
  task_space: {
    dimensions: ["terrain", "level"],
    coordinates: { terrain: ["flat", "rough"], level: ["1", "2"] },
  },
  metrics: {
    sampling_probability: [0.4, 0.3, 0.2, 0.1],
    learning_progress: [1, 0.5, -0.2, 0],
  },
};

before(async () => {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  origin = `http://127.0.0.1:${server.address().port}`;
});
after(async () => {
  await new Promise((resolve) => server.close(resolve));
  await rm(dataDir, { recursive: true, force: true });
});

test("validates tensor length", () => {
  assert.equal(validateFrame(structuredClone(frame)).step, 10);
  const invalid = structuredClone(frame);
  invalid.metrics.learning_progress.pop();
  assert.throws(() => validateFrame(invalid), /expected 4/);
});

test("stores and returns an immutable history", async () => {
  const post = await fetch(`${origin}/api/runs/test-run/frames`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(frame),
  });
  assert.equal(post.status, 202);
  const history = await (await fetch(`${origin}/api/runs/test-run/frames`)).json();
  assert.deepEqual(history.frames, [frame]);
  const line = await readFile(join(dataDir, "test-run", "frames.ndjson"), "utf8");
  assert.equal(JSON.parse(line).metrics.learning_progress[2], -0.2);
});

test("lists recorded runs", async () => {
  const payload = await (await fetch(`${origin}/api/runs`)).json();
  assert.equal(payload.runs[0].run_id, "test-run");
  assert.equal(payload.runs[0].frame_count, 1);
  assert.equal(payload.runs[0].last_step, 10);
});

test("rejects malformed frames", async () => {
  const bad = structuredClone(frame);
  bad.step = -1;
  const response = await fetch(`${origin}/api/runs/test-run/frames`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(bad),
  });
  assert.equal(response.status, 400);
});
