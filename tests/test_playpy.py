"""Unit tests for the playpy discovery / command builder."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import playpy as pp  # noqa: E402


class TestTaskInference(unittest.TestCase):
    def test_manifest_and_name(self):
        self.assertEqual(pp.infer_task_from_run_name("Jul20_14-38-45_v4_dreamwaq_genesis_seed1"),
                         "go2_v4_dreamwaq")
        self.assertEqual(pp.infer_task_from_run_name("Jul20_12-02-36_v4_mlp_genesis_seed1"),
                         "go2_v4_mlp")
        self.assertEqual(pp.infer_task_from_run_name("Jul14_13-26-14_bench_rma_genesis_seed1"),
                         "go2_bench_rma")
        self.assertEqual(pp.infer_task_from_run_name("Jul20_15-00-57_v4_him_fixed_genesis_seed1"),
                         "go2_v4_him_fixed")
        self.assertEqual(
            pp.infer_task_from_run_name(
                "Jul28_14-12-45_v6_v4_frontier_oracle_genesis"
            ),
            "go2_v6_frontier_oracle",
        )
        self.assertEqual(
            pp.infer_task_from_run_name(
                "Jul28_12-00-00_v6_v4_frontier_dreamwaq_genesis"
            ),
            "go2_v6_frontier_dreamwaq",
        )

    def test_ckpt_cli_mapping(self):
        self.assertEqual(pp.ckpt_filename_to_cli("best_tracking.pt"), "best_tracking")
        self.assertEqual(pp.ckpt_filename_to_cli("best.pt"), "best")
        self.assertEqual(pp.ckpt_filename_to_cli("model_3000.pt"), "3000")
        self.assertEqual(pp.prefer_checkpoint(["model_100.pt", "best_tracking.pt", "model_200.pt"]),
                         "best_tracking.pt")

    def test_terrain_choices_include_v6(self):
        self.assertIn("v6", pp.TERRAIN_CHOICES)
        self.assertIn("v6_full", pp.TERRAIN_CHOICES)
        self.assertIn("v6", pp.SHOWCASE_TERRAINS)
        self.assertEqual(pp.default_num_envs_for_terrain("v6"), 1)
        self.assertEqual(pp.default_num_envs_for_terrain("v6_full"), 1)
        self.assertEqual(pp.default_num_envs_for_terrain("flat"), 2)


class TestDiscovery(unittest.TestCase):
    def test_fake_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp)
            exp = logs / "go2_v4_terrain_curriculum"
            run = exp / "Jul20_01-00-00_v4_mlp_genesis_seed1"
            run.mkdir(parents=True)
            (run / "best_tracking.pt").write_bytes(b"x")
            (run / "events.out.tfevents.fake").write_bytes(b"x")
            (run / "go2_v4_config.py").write_text("# cfg\n", encoding="utf-8")
            (run / "run_manifest.json").write_text(
                json.dumps({"task": "go2_v4_mlp"}), encoding="utf-8"
            )
            # junk
            (logs / "eval").mkdir()
            (logs / "play").mkdir()
            empty = exp / "empty_run"
            empty.mkdir()

            exps = pp.list_experiments(logs)
            self.assertEqual(exps, ["go2_v4_terrain_curriculum"])
            runs = pp.discover_runs(logs)
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0].task, "go2_v4_mlp")
            self.assertEqual(runs[0].preferred_ckpt, "best_tracking.pt")
            self.assertEqual(runs[0].method_tag, "mlp")

    def test_name_inference_without_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp)
            run = logs / "go2_benchmark" / "Jul14_bench_dreamwaq_genesis_seed1"
            run.mkdir(parents=True)
            (run / "model_2000.pt").write_bytes(b"x")
            (run / "events.out.tfevents.fake").write_bytes(b"x")
            runs = pp.discover_runs(logs)
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0].task, "go2_bench_dreamwaq")
            self.assertEqual(runs[0].preferred_ckpt, "model_2000.pt")


class TestBuildCommand(unittest.TestCase):
    def test_argv_defaults(self):
        cfg = pp.PlayConfig(
            task="go2_v4_dreamwaq",
            load_run="/abs/logs/go2_v4_terrain_curriculum/runA",
            experiment="go2_v4_terrain_curriculum",
            ckpt="best_tracking.pt",
            terrain="taxonomy",
            viewer="viser",
            viser_port=8080,
            num_envs=1,
        )
        argv = pp.build_play_argv(cfg)
        self.assertIn("--task", argv)
        self.assertIn("go2_v4_dreamwaq", argv)
        self.assertIn("--terrain", argv)
        self.assertIn("taxonomy", argv)
        self.assertIn("--load_run", argv)
        self.assertIn("/abs/logs/go2_v4_terrain_curriculum/runA", argv)
        # ckpt stripped to CLI token
        i = argv.index("--ckpt")
        self.assertEqual(argv[i + 1], "best_tracking")
        self.assertIn("--viewer", argv)
        self.assertIn("viser", argv)

    def test_real_logs_if_present(self):
        """Smoke: discover at least one real run when logs exist in repo."""
        logs = _ROOT / "logs"
        if not logs.is_dir():
            self.skipTest("no logs/")
        runs = pp.discover_runs(logs)
        if not runs:
            self.skipTest("no checkpoints under logs/")
        # All preferred ckpts should exist on disk
        for r in runs[:5]:
            self.assertTrue((r.path / r.preferred_ckpt).is_file(), r.rel)
        # Command for first dreamwaq-ish run if any
        dream = next((r for r in runs if "dreamwaq" in r.task), runs[0])
        cfg = pp.PlayConfig(
            task=dream.task,
            load_run=str(dream.path.resolve()),
            experiment=dream.experiment,
            ckpt=dream.preferred_ckpt,
            terrain="taxonomy",
            run_name=dream.run,
        )
        cmd = pp.format_command(cfg)
        self.assertIn("play.py", cmd)
        self.assertIn("taxonomy", cmd)
        self.assertIn(dream.task, cmd)


if __name__ == "__main__":
    unittest.main(verbosity=2)
