"""Non-blocking LP-ACRL dashboard plugger with no third-party dependencies."""

from __future__ import annotations

import atexit
import json
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


class CurriculumDashboardPlugger:
    """Queues metric snapshots and uploads them without blocking training.

    Flatten each metric in C order following ``task_space.dimensions``.
    Failed uploads are retried; if the queue fills, the oldest unsent frame is
    dropped so dashboard I/O can never stall the training loop.
    """

    def __init__(
        self,
        run_id: str,
        task_space: TaskSpace,
        *,
        server_url: str = "http://127.0.0.1:8765",
        queue_size: int = 32,
        timeout_seconds: float = 2.0,
        retry_seconds: float = 1.0,
    ) -> None:
        self.run_id = run_id
        self.task_space = task_space
        self.endpoint = f"{server_url.rstrip('/')}/api/runs/{run_id}/frames"
        self.timeout_seconds = timeout_seconds
        self.retry_seconds = retry_seconds
        self._queue: queue.Queue[dict | None] = queue.Queue(maxsize=queue_size)
        self._closed = False
        self._thread = threading.Thread(target=self._worker, name="curriculum-dashboard", daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def log(self, step: int, metrics: Mapping[str, object], *, wall_time: float | None = None) -> bool:
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
            flattened[name] = [None if value is None else float(value) for value in data]
        frame = {
            "step": int(step),
            "wall_time": float(wall_time if wall_time is not None else time.time()),
            "task_space": self.task_space.as_dict(),
            "metrics": flattened,
        }
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

    def _worker(self) -> None:
        while True:
            frame = self._queue.get()
            if frame is None:
                self._queue.task_done()
                return
            payload = json.dumps(frame, separators=(",", ":")).encode()
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
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(self.retry_seconds)
            self._queue.task_done()
