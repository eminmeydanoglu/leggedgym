"""CPU-only contracts for the frozen V5 UED validation and selector path."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("SIMULATOR", "genesis")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from legged_gym.scripts.eval.select_checkpoint import (  # noqa: E402
    load_candidates,
    scheduled_iterations,
    select_best,
    select_run,
)
from legged_gym.scripts.eval.ued_validation import (  # noqa: E402
    FROZEN_NUM_CELLS,
    FROZEN_REPLICAS_PER_CELL,
    aggregate_measurements,
    bank_fingerprint,
    build_validation_bank,
    evaluate_with_rollout,
    load_config,
    make_validation_artifact,
)


FINGERPRINT_R12 = "3e08ac398ef5f9166cd29c14af44868b0916991b736737712cf8388ebb607a9d"
BANK_SIZE = FROZEN_NUM_CELLS * FROZEN_REPLICAS_PER_CELL  # 1008


class TestUEDCheckpointSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.config = load_config(root / "configs/eval/v5_ued.yaml")
        cls.bank = build_validation_bank(cls.config)

    def measurements(self, *, spnte_by_cell=None, fall=0.0, failed_cells=()):
        values = spnte_by_cell or [0.2] * FROZEN_NUM_CELLS
        failed = set(failed_cells)
        return [
            {
                "cell_id": row.cell_id,
                "replica_id": row.replica_id,
                "spnte_lin": float(values[row.cell_id]),
                "fall_rate": float(fall),
                "survival_steps": 800 if row.cell_id in failed else 1000,
                "command_vx": row.command_vx,
                "geometry_hash": row.geometry_hash,
            }
            for row in self.bank
        ]

    def write_candidate(self, run, iteration, *, values=None, fall=0.0, failed_cells=(), extra_checkpoint=None):
        checkpoint = {
            "iter": iteration,
            "model_state_dict": {"actor.weight": torch.tensor([float(iteration)])},
            "episode_curriculum": {
                "task_assignment_counts": [7, 11],
                "transition_occupancy": [3, 19],
            },
        }
        checkpoint.update(extra_checkpoint or {})
        checkpoint_path = run / f"model_{iteration}.pt"
        torch.save(checkpoint, checkpoint_path)
        checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        artifact = make_validation_artifact(
            checkpoint_iteration=iteration,
            measurements=self.measurements(spnte_by_cell=values, fall=fall, failed_cells=failed_cells),
            config=self.config,
            checkpoint_sha256=checkpoint_sha256,
            assignment_distribution={"task_0": 0.25, "task_1": 0.75},
            ppo_transition_occupancy={"task_0": 0.10, "task_1": 0.90},
        )
        output = run / self.config["selection"]["artifact_dir"]
        output.mkdir(exist_ok=True)
        path = output / self.config["selection"]["artifact_pattern"].format(iteration=iteration)
        path.write_text(json.dumps(artifact), encoding="utf-8")
        return path

    def write_checkpoint_only(self, run, iteration):
        torch.save({"iter": iteration, "model_state_dict": {}}, run / f"model_{iteration}.pt")

    def test_frozen_bank_has_seeds_geometry_hashes_and_84_by_12_replicas(self):
        self.assertEqual(FROZEN_REPLICAS_PER_CELL, 12)
        self.assertEqual(len(self.bank), BANK_SIZE)
        self.assertEqual(self.config["validation_seed"], 31001)
        self.assertEqual(self.config["eval_seed"], 41001)
        self.assertEqual(self.config["validation_bank"]["replicas_per_cell"], 12)
        self.assertEqual(bank_fingerprint(self.config), FINGERPRINT_R12)
        self.assertEqual(len({(row.cell_id, row.replica_id) for row in self.bank}), BANK_SIZE)
        self.assertTrue(all(0 <= row.command_vx <= 2.0 for row in self.bank))
        self.assertEqual(
            set(self.config["validation_bank"]["geometry_hashes"]),
            {"stairs_up", "stairs_down", "slope_up", "slope_down", "rough", "flat"},
        )

    def test_selection_schedule_fields_are_frozen(self):
        selection = self.config["selection"]
        self.assertEqual(selection["min_iteration"], 1000)
        self.assertEqual(selection["iteration_stride"], 500)
        self.assertTrue(selection["always_include_final"])

    def test_scheduled_iterations_floor_and_stride_from_pick(self):
        # Saves every 200: floor missing exact targets (1500→1400, 1900→1800, …).
        available = list(range(200, 3001, 200))
        self.assertEqual(
            scheduled_iterations(available, min_iteration=1000, iteration_stride=500),
            [1000, 1400, 1800, 2200, 2600, 3000],
        )
        # Explicit 1900 present after a 1400 floor; next targets floor on the
        # 200-grid (2400, then 2800) and final is always appended.
        available_with_1900 = sorted(set(available) | {1900})
        self.assertEqual(
            scheduled_iterations(available_with_1900, min_iteration=1000, iteration_stride=500),
            [1000, 1400, 1900, 2400, 2800, 3000],
        )
        # Final always included even when below min_iteration-only run ends early.
        self.assertEqual(
            scheduled_iterations([200, 400, 800], min_iteration=1000, iteration_stride=500),
            [800],
        )

    def test_macro_aggregation_weights_cells_not_replica_order(self):
        values = [0.1] * (FROZEN_NUM_CELLS - 1) + [0.9]
        scores = aggregate_measurements(self.measurements(spnte_by_cell=values), self.config)
        self.assertAlmostEqual(scores["macro_mean_spnte_lin"], sum(values) / float(FROZEN_NUM_CELLS))
        self.assertAlmostEqual(scores["worst_10pct_cvar_spnte_lin"], (0.9 + 8 * 0.1) / 9.0)
        self.assertEqual(len(scores["cells"]), FROZEN_NUM_CELLS)

    def test_selector_refuses_incomplete_candidate_bank(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for iteration in range(200, 3001, 200):
                self.write_checkpoint_only(run, iteration)
            artifact_path = self.write_candidate(run, 1000)
            # Complete schedule also needs later artifacts; incomplete first bank is enough to fail.
            payload = json.loads(artifact_path.read_text())
            payload["measurements"].pop()
            artifact_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete validation bank"):
                load_candidates(run, self.config)

    def test_selector_uses_schedule_not_every_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            # Full 200-grid through 3000; only schedule a subset fully.
            for iteration in range(200, 3001, 200):
                self.write_checkpoint_only(run, iteration)
            # Better SPNTE at 800 (before min) and at 1200 (between schedule picks)
            # must not win if we only complete scheduled artifacts with worse mid scores.
            self.write_candidate(run, 800, values=[0.01] * FROZEN_NUM_CELLS)
            self.write_candidate(run, 1200, values=[0.01] * FROZEN_NUM_CELLS)
            for iteration in (1000, 1400, 1800, 2200, 2600, 3000):
                self.write_candidate(run, iteration, values=[0.25] * FROZEN_NUM_CELLS)
            # Make 2200 clearly best among scheduled.
            self.write_candidate(run, 2200, values=[0.05] * FROZEN_NUM_CELLS)
            candidates = load_candidates(run, self.config)
            self.assertEqual([c.iteration for c in candidates], [1000, 1400, 1800, 2200, 2600, 3000])
            self.assertEqual(select_best(candidates, self.config).iteration, 2200)

    def test_selector_refuses_stale_checkpoint_and_geometry_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for iteration in range(200, 3001, 200):
                self.write_checkpoint_only(run, iteration)
            for iteration in (1000, 1400, 1800, 2200, 2600, 3000):
                self.write_candidate(run, iteration)
            artifact_path = run / "ued_validation" / "model_1000.json"
            payload = json.loads(artifact_path.read_text())
            payload["checkpoint_sha256"] = "0" * 64
            artifact_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_candidates(run, self.config)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for iteration in range(200, 3001, 200):
                self.write_checkpoint_only(run, iteration)
            for iteration in (1000, 1400, 1800, 2200, 2600, 3000):
                self.write_candidate(run, iteration)
            artifact_path = run / "ued_validation" / "model_1000.json"
            payload = json.loads(artifact_path.read_text())
            payload["geometry_hashes"]["flat"] = "0" * 64
            artifact_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "geometry hash pins mismatch"):
                load_candidates(run, self.config)

    def test_primary_score_wins_when_difference_exceeds_tolerance(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for iteration in range(200, 3001, 200):
                self.write_checkpoint_only(run, iteration)
            for iteration in (1000, 1400, 1800, 2200, 2600, 3000):
                self.write_candidate(run, iteration, values=[0.200002] * FROZEN_NUM_CELLS, fall=0.0)
            # 1000 is slightly better primary even with terrible secondary metrics.
            self.write_candidate(
                run, 1000,
                values=[0.200000] * FROZEN_NUM_CELLS,
                fall=0.8,
                failed_cells=range(FROZEN_NUM_CELLS),
            )
            self.assertEqual(select_best(load_candidates(run, self.config), self.config).iteration, 1000)

    def test_tie_break_order_applies_only_inside_tolerance(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for iteration in range(200, 3001, 200):
                self.write_checkpoint_only(run, iteration)
            # Equal macro; better CVaR at 1400.
            cvar_worse = [0.176] * 75 + [0.4] * 9
            cvar_better = [0.188] * 75 + [0.3] * 9
            for iteration in (1000, 1800, 2200, 2600, 3000):
                self.write_candidate(run, iteration, values=cvar_worse)
            self.write_candidate(run, 1400, values=cvar_better)
            self.assertEqual(select_best(load_candidates(run, self.config), self.config).iteration, 1400)

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for iteration in range(200, 3001, 200):
                self.write_checkpoint_only(run, iteration)
            tie_base = [0.188] * 75 + [0.3] * 9
            for iteration in (1000, 1400, 1800, 2200, 2600, 3000):
                self.write_candidate(run, iteration, values=tie_base, fall=0.2)
            self.write_candidate(run, 1800, values=tie_base, fall=0.1)
            self.assertEqual(select_best(load_candidates(run, self.config), self.config).iteration, 1800)

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for iteration in range(200, 3001, 200):
                self.write_checkpoint_only(run, iteration)
            tie_base = [0.188] * 75 + [0.3] * 9
            for iteration in (1000, 1400, 1800, 2200, 2600, 3000):
                self.write_candidate(run, iteration, values=tie_base, failed_cells=range(20))
            self.write_candidate(run, 2200, values=tie_base, failed_cells=range(10))
            self.assertEqual(select_best(load_candidates(run, self.config), self.config).iteration, 2200)

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for iteration in range(200, 3001, 200):
                self.write_checkpoint_only(run, iteration)
            for iteration in (1000, 1400, 1800, 2200, 2600, 3000):
                self.write_candidate(run, iteration, values=[0.2] * FROZEN_NUM_CELLS)
            # Complete equality chooses the earlier scheduled iteration.
            self.assertEqual(select_best(load_candidates(run, self.config), self.config).iteration, 1000)

    def test_artifact_metadata_and_resume_preservation_keep_distributions_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for iteration in range(200, 3001, 200):
                self.write_checkpoint_only(run, iteration)
            for iteration in (1000, 1400, 1800, 2200, 2600):
                self.write_candidate(run, iteration)
            self.write_candidate(
                run, 3000,
                extra_checkpoint={
                    "best_spnte": {"legacy_selection_note": "preserved"},
                    "resume_metadata": {"best_spnte": {"resume_marker": 1}, "other_resume_key": "kept"},
                },
            )
            winner, target = select_run(run, self.config)
            self.assertEqual(winner.iteration, 1000)  # equal scores → earliest scheduled
            self.assertTrue(target.is_file())
            self.assertFalse((run / "best_tracking.pt").exists())
            # Re-select with only 3000 better so metadata path still exercises 3000.
            self.write_candidate(
                run, 3000,
                values=[0.01] * FROZEN_NUM_CELLS,
                extra_checkpoint={
                    "best_spnte": {"legacy_selection_note": "preserved"},
                    "resume_metadata": {"best_spnte": {"resume_marker": 1}, "other_resume_key": "kept"},
                },
            )
            winner, target = select_run(run, self.config)
            self.assertEqual(winner.iteration, 3000)
            selected = torch.load(target, map_location="cpu", weights_only=False)
            metadata = selected["best_spnte"]
            self.assertEqual(metadata["selection_metric"], "spnte_v1")
            self.assertEqual(metadata["selected_iteration"], 3000)
            self.assertEqual(metadata["validation_bank_fingerprint"], bank_fingerprint(self.config))
            self.assertEqual(metadata["spnte_v_scale"], 2.0)
            self.assertEqual(metadata["legacy_selection_note"], "preserved")
            self.assertEqual(metadata["selection_min_iteration"], 1000)
            self.assertEqual(metadata["selection_iteration_stride"], 500)
            self.assertEqual(selected["resume_metadata"]["best_spnte"]["resume_marker"], 1)
            self.assertEqual(selected["resume_metadata"]["other_resume_key"], "kept")
            self.assertEqual(metadata["assignment_distribution"], {"task_0": 0.25, "task_1": 0.75})
            self.assertEqual(metadata["ppo_transition_occupancy"], {"task_0": 0.10, "task_1": 0.90})
            self.assertNotEqual(metadata["assignment_distribution"], metadata["ppo_transition_occupancy"])
            self.assertEqual(selected["episode_curriculum"]["task_assignment_counts"], [7, 11])
            self.assertEqual(selected["episode_curriculum"]["transition_occupancy"], [3, 19])

    def test_rollout_boundary_is_mockable(self):
        def mocked_rollout(rows):
            self.assertEqual(len(rows), BANK_SIZE)
            return self.measurements()

        artifact = evaluate_with_rollout(1000, self.config, mocked_rollout)
        self.assertEqual(artifact["checkpoint_iteration"], 1000)
        self.assertEqual(len(artifact["measurements"]), BANK_SIZE)


if __name__ == "__main__":
    unittest.main()
