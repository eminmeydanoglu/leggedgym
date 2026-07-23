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
    select_best,
    select_run,
)
from legged_gym.scripts.eval.ued_validation import (  # noqa: E402
    aggregate_measurements,
    bank_fingerprint,
    build_validation_bank,
    evaluate_with_rollout,
    load_config,
    make_validation_artifact,
)


class TestUEDCheckpointSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.config = load_config(root / "configs/eval/v5_ued.yaml")
        cls.bank = build_validation_bank(cls.config)

    def measurements(self, *, spnte_by_cell=None, fall=0.0, failed_cells=()):
        values = spnte_by_cell or [0.2] * 84
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

    def winner_for(self, first, second):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            self.write_candidate(run, 200, **first)
            self.write_candidate(run, 400, **second)
            return select_best(load_candidates(run, self.config), self.config).iteration

    def test_frozen_bank_has_seeds_geometry_hashes_and_84_by_48_replicas(self):
        self.assertEqual(len(self.bank), 84 * 48)
        self.assertEqual(self.config["validation_seed"], 31001)
        self.assertEqual(self.config["eval_seed"], 41001)
        self.assertEqual(bank_fingerprint(self.config), "141e9401f8e93817e3fe2d619d13694a048653d59ac5bae78bab465d2a798275")
        self.assertEqual(len({(row.cell_id, row.replica_id) for row in self.bank}), 4032)
        self.assertTrue(all(0 <= row.command_vx <= 2.0 for row in self.bank))
        self.assertEqual(set(self.config["validation_bank"]["geometry_hashes"]), {"stairs_up", "stairs_down", "slope_up", "slope_down", "rough", "flat"})

    def test_macro_aggregation_weights_cells_not_replica_order(self):
        values = [0.1] * 83 + [0.9]
        scores = aggregate_measurements(self.measurements(spnte_by_cell=values), self.config)
        self.assertAlmostEqual(scores["macro_mean_spnte_lin"], sum(values) / 84.0)
        self.assertAlmostEqual(scores["worst_10pct_cvar_spnte_lin"], (0.9 + 8 * 0.1) / 9.0)
        self.assertEqual(len(scores["cells"]), 84)

    def test_selector_refuses_incomplete_candidate_bank(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            artifact_path = self.write_candidate(run, 200)
            payload = json.loads(artifact_path.read_text())
            payload["measurements"].pop()
            artifact_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete validation bank"):
                load_candidates(run, self.config)

    def test_selector_refuses_stale_checkpoint_and_geometry_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            artifact_path = self.write_candidate(run, 200)
            payload = json.loads(artifact_path.read_text())
            payload["checkpoint_sha256"] = "0" * 64
            artifact_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_candidates(run, self.config)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            artifact_path = self.write_candidate(run, 200)
            payload = json.loads(artifact_path.read_text())
            payload["geometry_hashes"]["flat"] = "0" * 64
            artifact_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "geometry hash pins mismatch"):
                load_candidates(run, self.config)

    def test_primary_score_wins_when_difference_exceeds_tolerance(self):
        # Candidate 400 has vastly better secondary scores, but its primary
        # SPNTE is 2e-6 worse, outside the frozen 1e-6 tie window.
        self.assertEqual(
            self.winner_for(
                {"values": [0.200000] * 84, "fall": 0.8, "failed_cells": range(84)},
                {"values": [0.200002] * 84, "fall": 0.0},
            ),
            200,
        )

    def test_tie_break_order_applies_only_inside_tolerance(self):
        # Lower worst-10% task CVaR wins while macro means are exactly equal.
        cvar_worse = [0.176] * 75 + [0.4] * 9
        cvar_better = [0.188] * 75 + [0.3] * 9
        self.assertEqual(self.winner_for({"values": cvar_worse}, {"values": cvar_better}), 400)
        # With equal primary/CVaR, lower fall rate wins.
        tie_base = [0.188] * 75 + [0.3] * 9
        tie_near = [0.18800056] * 75 + [0.3] * 9
        self.assertEqual(self.winner_for({"values": tie_base, "fall": 0.2}, {"values": tie_near, "fall": 0.1}), 400)
        # With equal preceding components, higher replica success rate wins.
        self.assertEqual(self.winner_for({"values": tie_base, "failed_cells": range(20)}, {"values": tie_near, "failed_cells": range(10)}), 400)
        # Complete equality chooses the earlier periodic iteration.
        self.assertEqual(self.winner_for({"values": [0.2] * 84}, {"values": [0.2] * 84}), 200)

    def test_artifact_metadata_and_resume_preservation_keep_distributions_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            self.write_candidate(
                run, 400,
                extra_checkpoint={
                    "best_spnte": {"legacy_selection_note": "preserved"},
                    "resume_metadata": {"best_spnte": {"resume_marker": 1}, "other_resume_key": "kept"},
                },
            )
            winner, target = select_run(run, self.config)
            self.assertEqual(winner.iteration, 400)
            self.assertTrue(target.is_file())
            self.assertFalse((run / "best_tracking.pt").exists())
            selected = torch.load(target, map_location="cpu", weights_only=False)
            metadata = selected["best_spnte"]
            self.assertEqual(metadata["selection_metric"], "spnte_v1")
            self.assertEqual(metadata["selected_iteration"], 400)
            self.assertEqual(metadata["validation_bank_fingerprint"], bank_fingerprint(self.config))
            self.assertEqual(metadata["spnte_v_scale"], 2.0)
            self.assertEqual(metadata["legacy_selection_note"], "preserved")
            self.assertEqual(selected["resume_metadata"]["best_spnte"]["resume_marker"], 1)
            self.assertEqual(selected["resume_metadata"]["other_resume_key"], "kept")
            self.assertEqual(metadata["assignment_distribution"], {"task_0": 0.25, "task_1": 0.75})
            self.assertEqual(metadata["ppo_transition_occupancy"], {"task_0": 0.10, "task_1": 0.90})
            self.assertNotEqual(metadata["assignment_distribution"], metadata["ppo_transition_occupancy"])
            self.assertEqual(selected["episode_curriculum"]["task_assignment_counts"], [7, 11])
            self.assertEqual(selected["episode_curriculum"]["transition_occupancy"], [3, 19])

    def test_rollout_boundary_is_mockable(self):
        def mocked_rollout(rows):
            self.assertEqual(len(rows), 4032)
            return self.measurements()

        artifact = evaluate_with_rollout(200, self.config, mocked_rollout)
        self.assertEqual(artifact["checkpoint_iteration"], 200)
        self.assertEqual(len(artifact["measurements"]), 4032)


if __name__ == "__main__":
    unittest.main()
