import json
import sys
import unittest
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


if __name__ == "__main__":
    unittest.main()
