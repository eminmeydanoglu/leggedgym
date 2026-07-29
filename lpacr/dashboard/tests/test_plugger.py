import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plugger import CurriculumDashboardPlugger, TaskSpace


class TaskSpaceTest(unittest.TestCase):
    def test_size_and_serialization(self):
        space = TaskSpace(("a", "b"), {"a": [1, 2], "b": ["x", "y", "z"]})
        self.assertEqual(space.size, 6)
        self.assertEqual(space.as_dict()["dimensions"], ["a", "b"])

    def test_rejects_wrong_metric_shape_before_upload(self):
        space = TaskSpace(("a",), {"a": [1, 2]})
        plugger = CurriculumDashboardPlugger("unit", space, server_url="http://127.0.0.1:1")
        with self.assertRaisesRegex(ValueError, "expected 2"):
            plugger.log(1, {"performance": [0.1]})
        plugger.close()

    def test_nan_and_optional_metadata_use_valid_dashboard_json(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["content-length"])
                body = self.rfile.read(length)
                # Mirror Node: reject non-standard NaN tokens.
                text = body.decode("utf-8")
                if "NaN" in text or "Infinity" in text:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b'{"error":"invalid JSON number"}')
                    return
                received.append(json.loads(body))
                self.send_response(202)
                self.end_headers()

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            plugger = CurriculumDashboardPlugger(
                "unit-metadata", TaskSpace(("a",), {"a": [1, 2]}),
                server_url=f"http://127.0.0.1:{server.server_port}",
                metadata={"source": "v5"},
            )
            plugger.log(
                10,
                {"performance": [float("nan"), 3.0]},
                frame_metadata={
                    "stage_index": 1,
                    "diagnostics": {
                        "top10_overlap_prev": float("nan"),
                        "lp_reliability_median": float("nan"),
                        "entropy": 1.5,
                    },
                },
            )
            self.assertTrue(plugger.flush())
            plugger.close()
        finally:
            server.shutdown()
            server.server_close()

        self.assertEqual(received[0]["metrics"]["performance"], [None, 3.0])
        self.assertEqual(received[0]["metadata"]["run"]["source"], "v5")
        self.assertEqual(received[0]["metadata"]["frame"]["stage_index"], 1)
        self.assertEqual(
            received[0]["metadata"]["frame"]["diagnostics"],
            {
                "top10_overlap_prev": None,
                "lp_reliability_median": None,
                "entropy": 1.5,
            },
        )

    def test_local_dir_records_frames_without_server(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            space = TaskSpace(("vx_bin",), {"vx_bin": ["a", "b", "c", "d"]})
            plugger = CurriculumDashboardPlugger(
                "flat-local",
                space,
                server_url=None,
                local_dir=tmp,
                metadata={"source": "v7_flat", "task": "go2_v7_flat_lpacrl_him"},
            )
            self.assertTrue(
                plugger.log(
                    2000,
                    {
                        "sampling_probability": [0.1, 0.2, 0.3, 0.4],
                        "learning_progress": [0.0, 1.0, 2.0, 0.5],
                    },
                    frame_metadata={"stage_index": 1},
                )
            )
            plugger.close()
            frames_path = Path(tmp) / "frames.ndjson"
            meta_path = Path(tmp) / "metadata.json"
            self.assertTrue(frames_path.is_file())
            self.assertTrue(meta_path.is_file())
            frame = json.loads(frames_path.read_text(encoding="utf-8").strip())
            self.assertEqual(frame["step"], 2000)
            self.assertEqual(frame["metrics"]["sampling_probability"], [0.1, 0.2, 0.3, 0.4])
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["run_id"], "flat-local")
            self.assertEqual(meta["run_metadata"]["task"], "go2_v7_flat_lpacrl_him")


    def test_http_queue_drop_does_not_lose_local_history(self):
        """Even when the HTTP queue overflows, local frames stay complete."""
        import tempfile
        import time

        class SlowHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["content-length"])
                self.rfile.read(length)
                time.sleep(0.2)
                self.send_response(202)
                self.end_headers()

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), SlowHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                space = TaskSpace(("a",), {"a": [1]})
                plugger = CurriculumDashboardPlugger(
                    "local-durable",
                    space,
                    server_url=f"http://127.0.0.1:{server.server_port}",
                    local_dir=tmp,
                    queue_size=2,
                    timeout_seconds=0.5,
                    retry_seconds=0.05,
                )
                for step in range(8):
                    plugger.log(step, {"sampling_probability": [float(step)]})
                plugger.close()
                lines = (Path(tmp) / "frames.ndjson").read_text(encoding="utf-8").strip().splitlines()
                self.assertEqual(len(lines), 8)
                steps = [json.loads(line)["step"] for line in lines]
                self.assertEqual(steps, list(range(8)))
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
