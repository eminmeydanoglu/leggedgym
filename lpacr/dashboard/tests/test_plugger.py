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


if __name__ == "__main__":
    unittest.main()
