import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { createReadStream, existsSync } from "node:fs";
import { appendFile, mkdir, readFile, readdir, stat, writeFile } from "node:fs/promises";
import { extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL(".", import.meta.url));
const PUBLIC_DIR = join(ROOT, "public");
const DATA_DIR = resolve(process.env.LPACRL_DATA_DIR || join(ROOT, "data"));
const LOGS_DIR = resolve(
  process.env.LPACRL_LOGS_DIR || join(ROOT, "../../logs/go2_v5_ued"),
);
const PYTHON = process.env.LPACRL_PYTHON || "python3";
const BENCHMARK_SCRIPT = join(ROOT, "tb_benchmark.py");
const BENCHMARK_TTL_MS = Number(process.env.LPACRL_BENCHMARK_TTL_MS || 20000);
const HOST = process.env.LPACRL_HOST || "127.0.0.1";
const PORT = Number(process.env.LPACRL_PORT || 8765);
const clients = new Map();
const writeQueues = new Map();
let benchmarkCache = { key: "", at: 0, payload: null };

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
};

export function validateFrame(frame) {
  if (!frame || typeof frame !== "object") throw new Error("Body must be a JSON object.");
  if (!Number.isFinite(frame.step) || frame.step < 0) throw new Error("step must be a non-negative number.");
  if (!Number.isFinite(frame.wall_time)) frame.wall_time = Date.now() / 1000;
  const space = frame.task_space;
  if (!space || !Array.isArray(space.dimensions) || !space.coordinates) {
    throw new Error("task_space.dimensions and task_space.coordinates are required.");
  }
  const shape = space.dimensions.map((name) => {
    const values = space.coordinates[name];
    if (!Array.isArray(values) || values.length === 0) throw new Error(`Missing coordinates for ${name}.`);
    return values.length;
  });
  const size = shape.reduce((a, b) => a * b, 1);
  const metrics = frame.metrics;
  if (!metrics || typeof metrics !== "object") throw new Error("metrics are required.");
  for (const [name, values] of Object.entries(metrics)) {
    if (!Array.isArray(values) || values.length !== size) {
      throw new Error(`Metric ${name} has ${values?.length ?? 0} values; expected ${size}.`);
    }
    if (values.some((value) => value !== null && !Number.isFinite(value))) {
      throw new Error(`Metric ${name} contains a non-finite value.`);
    }
  }
  if (frame.metadata !== undefined && (
    !frame.metadata || typeof frame.metadata !== "object" || Array.isArray(frame.metadata)
  )) {
    throw new Error("metadata must be an object when supplied.");
  }
  return frame;
}

function safeRunId(value) {
  if (!/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}$/.test(value || "")) throw new Error("Invalid run id.");
  return value;
}

function sendJson(res, status, value) {
  const body = JSON.stringify(value);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(body),
    "cache-control": "no-store",
  });
  res.end(body);
}

async function readBody(req, limit = 16 * 1024 * 1024) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > limit) throw new Error("Request body is too large.");
    chunks.push(chunk);
  }
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

function enqueue(runId, operation) {
  const previous = writeQueues.get(runId) || Promise.resolve();
  const next = previous.then(operation, operation);
  const tracked = next.finally(() => {
    if (writeQueues.get(runId) === tracked) writeQueues.delete(runId);
  });
  writeQueues.set(runId, tracked);
  return tracked;
}

async function storeFrame(runId, frame) {
  const runDir = join(DATA_DIR, runId);
  await mkdir(runDir, { recursive: true });
  const metadataPath = join(runDir, "metadata.json");
  if (!existsSync(metadataPath)) {
    const runMetadata = frame.metadata?.run;
    await writeFile(metadataPath, JSON.stringify({
      run_id: runId,
      created_at: frame.wall_time,
      task_space: frame.task_space,
      metric_names: Object.keys(frame.metrics),
      ...(runMetadata ? { run_metadata: runMetadata } : {}),
    }, null, 2));
  }
  await appendFile(join(runDir, "frames.ndjson"), `${JSON.stringify(frame)}\n`);
}

async function listRuns() {
  await mkdir(DATA_DIR, { recursive: true });
  const names = await readdir(DATA_DIR);
  const runs = [];
  for (const runId of names) {
    try {
      const metadata = JSON.parse(await readFile(join(DATA_DIR, runId, "metadata.json"), "utf8"));
      const info = await stat(join(DATA_DIR, runId, "frames.ndjson"));
      const lines = (await readFile(join(DATA_DIR, runId, "frames.ndjson"), "utf8")).trim().split("\n");
      const last = lines.at(-1) ? JSON.parse(lines.at(-1)) : null;
      runs.push({ ...metadata, updated_at: info.mtimeMs / 1000, last_step: last?.step ?? 0, frame_count: lines[0] ? lines.length : 0 });
    } catch {
      // Ignore incomplete run directories.
    }
  }
  return runs.sort((a, b) => b.updated_at - a.updated_at);
}

async function getFrames(runId, from = 0) {
  const path = join(DATA_DIR, runId, "frames.ndjson");
  if (!existsSync(path)) return [];
  const lines = (await readFile(path, "utf8")).split("\n");
  const frames = [];
  for (const line of lines) {
    if (!line) continue;
    const frame = JSON.parse(line);
    if (frame.step >= from) frames.push(frame);
  }
  return frames;
}

function broadcast(runId, frame) {
  for (const res of clients.get(runId) || []) {
    res.write(`event: frame\ndata: ${JSON.stringify(frame)}\n\n`);
  }
}

function runBenchmark(query) {
  const window = Math.max(1, Math.min(500, Number(query.get("window") || 50) || 50));
  const seed = query.get("seed");
  const leftPath = query.get("left_path") || process.env.LPACRL_BENCHMARK_LEFT_PATH || "";
  const rightPath = query.get("right_path") || process.env.LPACRL_BENCHMARK_RIGHT_PATH || "";
  const key = JSON.stringify({
    logs: LOGS_DIR,
    window,
    seed,
    leftPath,
    rightPath,
    python: PYTHON,
  });
  const now = Date.now();
  if (benchmarkCache.payload && benchmarkCache.key === key && now - benchmarkCache.at < BENCHMARK_TTL_MS) {
    return Promise.resolve(benchmarkCache.payload);
  }
  const args = [BENCHMARK_SCRIPT, "--logs-dir", LOGS_DIR, "--window", String(window)];
  if (seed != null && seed !== "") args.push("--seed", String(seed));
  if (leftPath) args.push("--left-path", leftPath);
  if (rightPath) args.push("--right-path", rightPath);

  return new Promise((resolvePromise) => {
    const child = spawn(PYTHON, args, {
      cwd: ROOT,
      env: process.env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
    }, 30000);
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", (error) => {
      clearTimeout(timer);
      resolvePromise({
        ok: false,
        error: `Failed to spawn benchmark helper: ${error.message}`,
        logs_dir: LOGS_DIR,
        rows: [],
      });
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      let payload;
      try {
        payload = JSON.parse(stdout.trim() || "{}");
      } catch {
        payload = {
          ok: false,
          error: `Benchmark helper returned non-JSON (exit ${code}): ${stderr || stdout}`.slice(0, 500),
          logs_dir: LOGS_DIR,
          rows: [],
        };
      }
      if (!payload.ok && stderr && !payload.error) {
        payload.error = stderr.slice(0, 500);
      }
      benchmarkCache = { key, at: Date.now(), payload };
      resolvePromise(payload);
    });
  });
}

async function api(req, res, url) {
  if (req.method === "GET" && url.pathname === "/api/runs") {
    sendJson(res, 200, { runs: await listRuns() });
    return true;
  }
  if (req.method === "GET" && url.pathname === "/api/benchmark") {
    const payload = await runBenchmark(url.searchParams);
    sendJson(res, 200, payload);
    return true;
  }
  const match = url.pathname.match(/^\/api\/runs\/([^/]+)\/(frames|stream)$/);
  if (!match) return false;
  const runId = safeRunId(decodeURIComponent(match[1]));
  if (req.method === "POST" && match[2] === "frames") {
    const frame = validateFrame(await readBody(req));
    await enqueue(runId, () => storeFrame(runId, frame));
    broadcast(runId, frame);
    sendJson(res, 202, { stored: true, step: frame.step });
    return true;
  }
  if (req.method === "GET" && match[2] === "frames") {
    const frames = await getFrames(runId, Number(url.searchParams.get("from") || 0));
    sendJson(res, 200, { run_id: runId, frames });
    return true;
  }
  if (req.method === "GET" && match[2] === "stream") {
    res.writeHead(200, {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      connection: "keep-alive",
      "x-accel-buffering": "no",
    });
    res.write(": connected\n\n");
    const set = clients.get(runId) || new Set();
    set.add(res);
    clients.set(runId, set);
    const keepAlive = setInterval(() => res.write(": keepalive\n\n"), 15000);
    req.on("close", () => {
      clearInterval(keepAlive);
      set.delete(res);
      if (!set.size) clients.delete(runId);
    });
    return true;
  }
  return false;
}

async function serveStatic(req, res, url) {
  const requested = url.pathname === "/" ? "index.html" : decodeURIComponent(url.pathname.slice(1));
  const path = normalize(join(PUBLIC_DIR, requested));
  if (!path.startsWith(`${PUBLIC_DIR}${sep}`) && path !== join(PUBLIC_DIR, "index.html")) {
    return sendJson(res, 403, { error: "Forbidden." });
  }
  const target = existsSync(path) ? path : join(PUBLIC_DIR, "index.html");
  const info = await stat(target);
  res.writeHead(200, {
    "content-type": MIME[extname(target)] || "application/octet-stream",
    "content-length": info.size,
    "cache-control": "no-cache",
  });
  createReadStream(target).pipe(res);
}

export function createDashboardServer() {
  return createServer(async (req, res) => {
    try {
      const url = new URL(req.url, `http://${req.headers.host || "localhost"}`);
      if (url.pathname.startsWith("/api/")) {
        const handled = await api(req, res, url);
        if (!handled) sendJson(res, 404, { error: "Not found." });
      } else {
        await serveStatic(req, res, url);
      }
    } catch (error) {
      sendJson(res, 400, { error: error.message });
    }
  });
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  await mkdir(DATA_DIR, { recursive: true });
  createDashboardServer().listen(PORT, HOST, () => {
    console.log(`Curriculum Atlas: http://${HOST}:${PORT}`);
    console.log(`Data: ${DATA_DIR}`);
    console.log(`Logs (benchmark): ${LOGS_DIR}`);
    console.log(`Python (benchmark): ${PYTHON}`);
  });
}
