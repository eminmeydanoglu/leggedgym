"""Non-blocking LP-ACRL dashboard plugger with no third-party dependencies."""

from __future__ import annotations

import atexit
import json
import math
import queue
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class TaskSpace:
    """Ordered tensor dimensions and their human-readable coordinates."""

    dimensions: Sequence[str]
    coordinates: Mapping[str, Sequence[object]]

    @property
    def size(self) -> int:
        result = 1
        for name in self.dimensions:
            values = self.coordinates.get(name)
            if not values:
                raise ValueError(f"Missing coordinates for {name}")
            result *= len(values)
        return result

    def as_dict(self) -> dict:
        self.size  # validate
        return {
            "dimensions": list(self.dimensions),
            "coordinates": {key: list(values) for key, values in self.coordinates.items()},
        }


def _json_safe(value: object) -> object:
    """Convert values to strict-JSON forms Node can parse.

    Python's ``json.dumps`` emits non-standard ``NaN``/``Infinity`` tokens by
    default.  The Curriculum Atlas server uses ``JSON.parse``, which rejects
    those tokens with HTTP 400 — and a plugger worker that retries 4xx forever
    silently drops every subsequent stage frame.  Metrics already went through
    this conversion; nested ``metadata`` (diagnostics with optional NaNs) must
    too.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    # bool is a subclass of int; keep it above any numeric coercion.
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    # NumPy scalars: prefer ``item()`` so np.float64(nan) → null, not [null].
    numpy_item = getattr(value, "item", None)
    if callable(numpy_item) and type(value).__module__.startswith("numpy"):
        try:
            return _json_safe(numpy_item())
        except (ValueError, OverflowError, AttributeError):
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist) and not isinstance(value, (list, tuple, dict)):
        try:
            return _json_safe(tolist())
        except (TypeError, ValueError, AttributeError):
            pass
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    return str(value)


class CurriculumDashboardPlugger:
    """Queues metric snapshots and uploads them without blocking training.

    Flatten each metric in C order following ``task_space.dimensions``.
    Failed uploads are retried; if the queue fills, the oldest unsent frame is
    dropped so dashboard I/O can never stall the training loop.

    When ``local_dir`` is set, every accepted frame is also appended to
    ``local_dir/frames.ndjson`` (and a one-shot ``metadata.json``) *before*
    the HTTP attempt.  That local write is the durable record for sample
    probabilities / LP / episode counts even if the dashboard server is down
    or never started -- critical for headless UHeM jobs.
    """

    def __init__(
        self,
        run_id: str,
        task_space: TaskSpace,
        *,
        metadata: Mapping[str, object] | None = None,
        server_url: str | None = "http://127.0.0.1:8765",
        local_dir: str | None = None,
        queue_size: int = 32,
        timeout_seconds: float = 2.0,
        retry_seconds: float = 1.0,
    ) -> None:
        if not server_url and not local_dir:
            raise ValueError("CurriculumDashboardPlugger needs server_url and/or local_dir")
        self.run_id = run_id
        self.task_space = task_space
        # Run metadata is deliberately optional.  Existing generic users keep
        # precisely the old frame shape, while a training adapter can attach
        # provenance without teaching this framework-agnostic transport about
        # PPO, Genesis, or a particular curriculum implementation.
        self.metadata = dict(metadata) if metadata is not None else None
        self.server_url = server_url.rstrip("/") if server_url else None
        self.endpoint = (
            f"{self.server_url}/api/runs/{run_id}/frames" if self.server_url else None
        )
        self.local_dir = local_dir
        self.timeout_seconds = timeout_seconds
        self.retry_seconds = retry_seconds
        self._local_lock = threading.Lock()
        self._local_metadata_written = False
        self._queue: queue.Queue[dict | None] = queue.Queue(maxsize=queue_size)
        self._closed = False
        # Local-only mode still uses the worker so log() stays non-blocking;
        # the worker just writes disk and skips HTTP when endpoint is None.
        self._thread = threading.Thread(target=self._worker, name="curriculum-dashboard", daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def log(
        self,
        step: int,
        metrics: Mapping[str, object],
        *,
        wall_time: float | None = None,
        frame_metadata: Mapping[str, object] | None = None,
    ) -> bool:
        """Queue one snapshot. Returns False only when an older frame was dropped."""
        if self._closed:
            raise RuntimeError("Plugger is closed.")
        expected = self.task_space.size
        flattened = {}
        for name, values in metrics.items():
            if hasattr(values, "detach"):
                values = values.detach().cpu()
            if hasattr(values, "numpy"):
                values = values.numpy()
            if hasattr(values, "reshape"):
                values = values.reshape(-1)
            data = values.tolist() if hasattr(values, "tolist") else list(values)
            if len(data) != expected:
                raise ValueError(f"{name} has {len(data)} values; expected {expected}")
            # V5 uses NaN to mean "this cell has not been observed".  JSON and
            # the dashboard schema represent that truthfully as null instead of
            # emitting non-standard NaN tokens that Node must reject.
            flattened[name] = [
                None if value is None or not math.isfinite(float(value)) else float(value)
                for value in data
            ]
        frame = {
            "step": int(step),
            "wall_time": float(wall_time if wall_time is not None else time.time()),
            "task_space": self.task_space.as_dict(),
            "metrics": flattened,
        }
        if self.metadata is not None or frame_metadata is not None:
            # Sanitize nested metadata the same way metrics are sanitized so
            # optional diagnostic NaNs become JSON null, not illegal tokens.
            frame["metadata"] = _json_safe({
                **({"run": self.metadata} if self.metadata is not None else {}),
                **({"frame": dict(frame_metadata)} if frame_metadata is not None else {}),
            })
        # Durable local write is synchronous and independent of the HTTP
        # queue: a backed-up dashboard must never drop sample-probability
        # history that analysis needs after the job ends.
        if self.local_dir is not None:
            try:
                self._write_local(frame)
            except Exception as exc:  # noqa: BLE001
                print(f"[ued-dashboard] local frame write failed step={frame.get('step')}: {exc!r}")
        if self.endpoint is None:
            return True
        dropped = False
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            self._queue.get_nowait()
            self._queue.task_done()
            self._queue.put_nowait(frame)
            dropped = True
        return not dropped

    def flush(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.02)
        return self._queue.unfinished_tasks == 0

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2)

    def _write_local(self, frame: dict) -> None:
        """Append one frame under local_dir; create metadata.json once.

        Mirrors lpacr/dashboard/server.mjs storeFrame so analysis tools can
        read either the live Atlas dir or the run-local copy interchangeably.
        """
        if not self.local_dir:
            return
        from pathlib import Path

        run_dir = Path(self.local_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(frame, separators=(",", ":"), allow_nan=False) + "\n"
        with self._local_lock:
            meta_path = run_dir / "metadata.json"
            if not self._local_metadata_written and not meta_path.is_file():
                run_meta = None
                if isinstance(frame.get("metadata"), dict):
                    run_meta = frame["metadata"].get("run")
                payload = {
                    "run_id": self.run_id,
                    "created_at": frame.get("wall_time"),
                    "task_space": frame.get("task_space"),
                    "metric_names": list(frame.get("metrics", {}).keys()),
                }
                if run_meta is not None:
                    payload["run_metadata"] = run_meta
                meta_path.write_text(
                    json.dumps(payload, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                self._local_metadata_written = True
            with (run_dir / "frames.ndjson").open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()

    def _worker(self) -> None:
        while True:
            frame = self._queue.get()
            if frame is None:
                self._queue.task_done()
                return
            try:
                # allow_nan=False turns a missed sanitization into a hard error
                # instead of a Node 400 that we used to retry forever.
                payload = json.dumps(frame, separators=(",", ":"), allow_nan=False).encode()
            except (TypeError, ValueError) as exc:
                print(f"[ued-dashboard] drop non-JSON frame step={frame.get('step')}: {exc}")
                self._queue.task_done()
                continue
            if self.endpoint is None:
                self._queue.task_done()
                continue
            request = urllib.request.Request(
                self.endpoint,
                data=payload,
                headers={"content-type": "application/json"},
                method="POST",
            )
            while not self._closed:
                try:
                    with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                        if response.status < 300:
                            break
                        # Unexpected 2xx-with-large-status shouldn't spin hot.
                        time.sleep(self.retry_seconds)
                except urllib.error.HTTPError as exc:
                    # 4xx is a permanent payload/route problem; retrying only
                    # wedges the queue (this is how NaN metadata went silent).
                    if 400 <= int(exc.code) < 500:
                        detail = ""
                        try:
                            detail = exc.read().decode("utf-8", errors="replace")[:300]
                        except Exception:  # noqa: BLE001
                            pass
                        print(
                            f"[ued-dashboard] drop HTTP {exc.code} for step="
                            f"{frame.get('step')}: {detail}"
                        )
                        break
                    time.sleep(self.retry_seconds)
                except (urllib.error.URLError, TimeoutError):
                    # Local frames already landed; keep retrying HTTP until
                    # close() so a late-started Atlas still fills in.
                    time.sleep(self.retry_seconds)
            self._queue.task_done()
