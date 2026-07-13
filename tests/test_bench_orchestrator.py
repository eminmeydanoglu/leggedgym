"""Static / dry-run tests for the benchmark_tripler orchestrator.

No GPU. Drives the shipped planner + dry-run path.
"""

from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("SIMULATOR", "genesis")

from scripts.benchmark_tripler_orchestrator import (  # noqa: E402
    ARTIFACT_ROOT_NAME,
    METHODS,
    RUN_MAP,
    SEEDS,
    TRAINING_COMMIT,
    build_cmd,
    dry_map_summary,
    plan_cells,
    plan_phase_a1,
    plan_phase_a2,
    plan_smoke_cells,
)


# Nine folders from benchmark_tripler.md
EXPECTED_RUNS = {
    "Jul13_09-47-39_bench_mlp_genesis_seed1",
    "Jul13_09-48-03_bench_mlp_genesis_seed2",
    "Jul13_09-48-28_bench_mlp_genesis_seed3",
    "Jul13_09-48-26_bench_oracle_id_genesis_seed1",
    "Jul13_09-48-51_bench_oracle_id_genesis_seed2",
    "Jul13_09-49-17_bench_oracle_id_genesis_seed3",
    "Jul13_10-34-00_bench_oracle_id_vel_genesis_seed1",
    "Jul13_10-34-29_bench_oracle_id_vel_genesis_seed2",
    "Jul13_10-35-00_bench_oracle_id_vel_genesis_seed3",
}


class TestRunMap(unittest.TestCase):
    def test_nine_fixed_runs(self):
        self.assertEqual(len(RUN_MAP), 9)
        self.assertEqual(set(RUN_MAP.values()), EXPECTED_RUNS)
        self.assertEqual(set(METHODS.keys()), {"mlp", "p5", "p5v"})
        self.assertEqual(tuple(SEEDS), (1, 2, 3))
        self.assertTrue(TRAINING_COMMIT.startswith("d138a2d"))

    def test_tasks(self):
        self.assertEqual(METHODS["mlp"][0], "go2_bench_mlp")
        self.assertEqual(METHODS["p5"][0], "go2_bench_oracle_id")
        self.assertEqual(METHODS["p5v"][0], "go2_bench_oracle_id_vel")


class TestPlanCells(unittest.TestCase):
    def test_a1_count(self):
        cells = plan_phase_a1("/tmp/x")
        self.assertEqual(len(cells), 9)  # 3 methods × 3 seeds
        self.assertTrue(all(c.ckpt == "best" for c in cells))
        self.assertTrue(all(c.module == "indist" for c in cells))

    def test_a2_a3_counts(self):
        a2 = plan_phase_a2("/tmp/x", "best")
        a3 = plan_phase_a2("/tmp/x", "3000")
        self.assertEqual(len(a2), 18)  # 3×3×2 vx
        self.assertEqual(len(a3), 18)
        self.assertTrue(all(c.ckpt == "best" for c in a2))
        self.assertTrue(all(c.ckpt == "3000" for c in a3))
        self.assertTrue(all("primary" in c.out_rel for c in a2))
        self.assertTrue(all(c.out_rel.startswith("model_3000/") for c in a3))

    def test_phase_a_total(self):
        cells = plan_cells("A", "/tmp/x")
        self.assertEqual(len(cells), 9 + 18 + 18)

    def test_smoke_three_tasks(self):
        cells = plan_smoke_cells("/tmp/x")
        self.assertEqual(len(cells), 3)
        tasks = {c.task for c in cells}
        self.assertEqual(tasks, {
            "go2_bench_mlp", "go2_bench_oracle_id", "go2_bench_oracle_id_vel",
        })

    def test_artifact_root_name(self):
        self.assertEqual(ARTIFACT_ROOT_NAME, "benchmark_tripler_2026-07-13")

    def test_build_cmd_serial_shape(self):
        cells = plan_phase_a1("/tmp/x")
        cmd = build_cmd(cells[0], "/tmp/out.npz")
        self.assertIn("legged_gym.scripts.eval.indist", " ".join(cmd))
        self.assertIn("--ckpt", cmd)
        self.assertIn("best", cmd)
        self.assertIn("--out", cmd)

    def test_dry_map_mentions_runs(self):
        cells = plan_cells("A1", "/tmp/x")
        text = dry_map_summary(cells)
        self.assertIn("n_cells=9", text)
        self.assertIn("Jul13_09-47-39_bench_mlp_genesis_seed1", text)
        self.assertIn(ARTIFACT_ROOT_NAME, text)


class TestDryRunPath(unittest.TestCase):
    def test_dry_run_writes_ledger_no_npz(self):
        """Shipped main() dry-run path: ledger/commands without spawning eval."""
        import scripts.benchmark_tripler_orchestrator as orch

        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, ARTIFACT_ROOT_NAME)
            rc = orch.main(["--phase", "smoke", "--dry-run", "--artifact-root", root])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.isfile(os.path.join(root, "ledger.tsv")))
            self.assertTrue(os.path.isfile(os.path.join(root, "commands.log")))
            self.assertTrue(os.path.isfile(os.path.join(root, "run_map.tsv")))
            self.assertTrue(os.path.isfile(os.path.join(root, "manifest.json")))
            # no real eval npz under smoke/
            smoke_dir = os.path.join(root, "smoke")
            if os.path.isdir(smoke_dir):
                npzs = [f for f in os.listdir(smoke_dir) if f.endswith(".npz")]
                self.assertEqual(npzs, [])
            with open(os.path.join(root, "commands.log")) as f:
                log = f.read()
            self.assertIn("DRY", log)
            # all nine run folders appear in run_map
            with open(os.path.join(root, "run_map.tsv")) as f:
                rm = f.read()
            for name in EXPECTED_RUNS:
                self.assertIn(name, rm)


class TestPreflightRunMap(unittest.TestCase):
    def test_preflight_requires_manifest_and_commit(self):
        import json
        from scripts.benchmark_tripler_orchestrator import (
            RUN_MAP, METHODS, TRAINING_COMMIT, preflight_run_map,
        )

        with tempfile.TemporaryDirectory() as tmp:
            # incomplete tree → fail
            with self.assertRaises(FileNotFoundError):
                preflight_run_map(tmp)

            for (mk, seed), run in RUN_MAP.items():
                d = os.path.join(tmp, run)
                os.makedirs(d)
                with open(os.path.join(d, "best.pt"), "wb") as f:
                    f.write(b"x")
                with open(os.path.join(d, "model_3000.pt"), "wb") as f:
                    f.write(b"y")
                # missing manifest
            with self.assertRaises(FileNotFoundError):
                preflight_run_map(tmp)

            for (mk, seed), run in RUN_MAP.items():
                task = METHODS[mk][0]
                d = os.path.join(tmp, run)
                with open(os.path.join(d, "run_manifest.json"), "w") as f:
                    json.dump({
                        "task": task,
                        "training_seed": seed,
                        "git_commit": TRAINING_COMMIT,
                    }, f)
            lines = preflight_run_map(tmp)
            self.assertEqual(len(lines), 9)
            self.assertTrue(all(l.startswith("OK") for l in lines))

            # wrong commit
            bad = list(RUN_MAP.values())[0]
            with open(os.path.join(tmp, bad, "run_manifest.json"), "w") as f:
                json.dump({
                    "task": METHODS["mlp"][0],
                    "training_seed": 1,
                    "git_commit": "deadbeef00000000",
                }, f)
            with self.assertRaises(ValueError):
                preflight_run_map(tmp)


if __name__ == "__main__":
    unittest.main()
