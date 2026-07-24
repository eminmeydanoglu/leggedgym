"""CPU-only contracts for the frozen V5 UED validation and selector path."""
from __future__ import annotations

import copy
import hashlib
import itertools
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
    GEOMETRY_HASH_VERSION,
    aggregate_measurements,
    bank_fingerprint,
    build_holdout_bank,
    build_validation_bank,
    evaluate_with_rollout,
    geometry_pins,
    load_config,
    make_shard_payload,
    make_validation_artifact,
    merge_shard_payloads,
    protocol_fingerprint,
)


# v3 (3-axis command, per-(type, level) heightfield-byte geometry pins).
FINGERPRINT_R12 = "dd1a2cd006a3774fd5f58ffe573e40a3fb63c2d3d917a5427bfa064abab09bcc"
HOLDOUT_FINGERPRINT = "a00aa2ed52fd7a975c74004c75f83905f4dc5bfb7749a4b47ffe52c5faebebd5"
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
                "command_vy": row.command_vy,
                "command_yaw": row.command_yaw,
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
        self.assertEqual(bank_fingerprint(self.config, kind="validation"), FINGERPRINT_R12)
        self.assertEqual(bank_fingerprint(self.config, kind="holdout"), HOLDOUT_FINGERPRINT)
        self.assertEqual(len({(row.cell_id, row.replica_id) for row in self.bank}), BANK_SIZE)
        self.assertTrue(all(0 <= row.command_vx <= 2.0 for row in self.bank))
        # 3-axis bank: vy / omega_z frozen over the shared [-1, 1] nuisance support.
        self.assertTrue(all(-1.0 <= row.command_vy <= 1.0 for row in self.bank))
        self.assertTrue(all(-1.0 <= row.command_yaw <= 1.0 for row in self.bank))
        self.assertTrue(any(row.command_vy != 0.0 for row in self.bank))
        self.assertTrue(any(row.command_yaw != 0.0 for row in self.bank))
        pins = geometry_pins(self.config)
        self.assertEqual(set(pins), {"stairs_up", "stairs_down", "slope_up", "slope_down", "rough", "flat"})
        # Per-(type, level): four levels for every moving type, one flat cell.
        for moving in ("stairs_up", "stairs_down", "slope_up", "slope_down", "rough"):
            self.assertEqual(set(pins[moving]), {"0", "1", "2", "3"})
        self.assertEqual(set(pins["flat"]), {"0"})
        self.assertEqual(self.config["validation_bank"]["geometry_hash_version"], GEOMETRY_HASH_VERSION)

    def test_holdout_bank_is_disjoint_and_never_named_validation(self):
        holdout = build_holdout_bank(self.config)
        self.assertEqual(len(holdout), BANK_SIZE)
        # Same terrain grid + geometry pins, INDEPENDENT commands (eval_seed).
        self.assertTrue(any(a.command_vx != b.command_vx for a, b in zip(self.bank, holdout)))
        self.assertTrue(all(a.geometry_hash == b.geometry_hash for a, b in zip(self.bank, holdout)))
        self.assertNotEqual(bank_fingerprint(self.config, kind="validation"),
                            bank_fingerprint(self.config, kind="holdout"))

    def test_geometry_pins_match_headless_heightfield_rebuild(self):
        # The reviewer item-7 anchor: the frozen pins are the ACTUAL heightfield
        # bytes, regenerable headless, not a self-referential builder description.
        from legged_gym.utils.terrain import build_taxonomy_geometry_hashes
        rebuilt = build_taxonomy_geometry_hashes()
        pins = geometry_pins(self.config)
        for terrain_type, levels in rebuilt.items():
            for level, digest in levels.items():
                self.assertEqual(pins[terrain_type][str(level)], digest)

    def test_selector_refuses_holdout_bank_artifact(self):
        # A held-out artifact (eval_seed) must never be accepted for selection.
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for iteration in range(200, 3001, 200):
                self.write_checkpoint_only(run, iteration)
            for iteration in (1000, 1400, 1800, 2200, 2600, 3000):
                self.write_candidate(run, iteration)
            artifact_path = run / "ued_validation" / "model_1000.json"
            payload = json.loads(artifact_path.read_text())
            payload["bank_kind"] = "holdout"
            artifact_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "held-out bank must not feed selection"):
                load_candidates(run, self.config)

    def test_wrong_runtime_geometry_hash_is_rejected(self):
        # With the tautology fixed, a measurement whose reported (runtime) hash
        # disagrees with the frozen pin must fail closed, not pass silently.
        from legged_gym.scripts.eval.ued_validation import validate_measurements
        measurements = self.measurements()
        measurements[0]["geometry_hash"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "geometry hash diverges"):
            validate_measurements(measurements, self.config, self.bank)

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

    # --- reviewer 9: selection must be independent of candidate ordering -------

    @staticmethod
    def _band_values(mean, tail):
        """84-cell SPNTE array with a fixed worst-9 tail (CVaR == tail)."""
        body = (84.0 * mean - 9.0 * tail) / 75.0
        return [body] * 75 + [tail] * 9

    def _write_ordering_trap(self, run):
        """A, B, C chained within pairwise tolerance but spanning > tolerance.

        Global-best primary is A (1000); B (1400) is within tolerance of A with a
        better CVaR; C (1800) is > tolerance worse than A (out of band) yet has
        the best CVaR -- the exact case the old pairwise reduce could mis-select.
        Winner must be B for every candidate ordering.
        """
        for iteration in range(200, 3001, 200):
            self.write_checkpoint_only(run, iteration)
        self.write_candidate(run, 1000, values=self._band_values(0.2000000, 0.30))
        self.write_candidate(run, 1400, values=self._band_values(0.2000008, 0.28))
        self.write_candidate(run, 1800, values=self._band_values(0.2000016, 0.26))
        # Fillers are clearly worse on both primary and CVaR so C keeps the best
        # CVaR among all candidates while still losing on the band rule.
        for iteration in (2200, 2600, 3000):
            self.write_candidate(run, iteration, values=[0.5] * FROZEN_NUM_CELLS)

    def test_selection_is_permutation_invariant_inside_tolerance_band(self):
        self.assertEqual(self.config["selection"]["primary_tolerance"], 0.000001)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            self._write_ordering_trap(run)
            candidates = load_candidates(run, self.config)
            # Sanity: the out-of-band C has the best CVaR but must still lose.
            best_cvar = min(candidates, key=lambda c: c.scores["worst_10pct_cvar_spnte_lin"])
            self.assertEqual(best_cvar.iteration, 1800)
            winners = {
                select_best(list(order), self.config).iteration
                for order in itertools.permutations(candidates)
            }
            self.assertEqual(winners, {1400})

    def test_out_of_band_candidate_never_wins_on_secondary(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            self._write_ordering_trap(run)
            winner = select_best(load_candidates(run, self.config), self.config)
            self.assertEqual(winner.iteration, 1400)
            self.assertNotEqual(winner.iteration, 1800)

    # --- reviewer 8b: protocol fingerprint binds the frozen scoring knobs ------

    def test_selector_refuses_protocol_fingerprint_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for iteration in range(200, 3001, 200):
                self.write_checkpoint_only(run, iteration)
            for iteration in (1000, 1400, 1800, 2200, 2600, 3000):
                self.write_candidate(run, iteration)
            artifact_path = run / "ued_validation" / "model_1000.json"
            payload = json.loads(artifact_path.read_text())
            payload["protocol_fingerprint"] = "0" * 64
            artifact_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "protocol fingerprint mismatch"):
                load_candidates(run, self.config)

    def test_smoke_horizon_override_cannot_pose_as_headline(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for iteration in range(200, 3001, 200):
                self.write_checkpoint_only(run, iteration)
            for iteration in (1000, 1400, 1800, 2200, 2600, 3000):
                self.write_candidate(run, iteration)
            # Re-emit the 1000 artifact as if it were rolled out for only 50 steps.
            checkpoint_sha = hashlib.sha256((run / "model_1000.pt").read_bytes()).hexdigest()
            smoke = make_validation_artifact(
                checkpoint_iteration=1000,
                measurements=self.measurements(),
                config=self.config,
                checkpoint_sha256=checkpoint_sha,
                provenance={"rollout_steps": 50, "warmup_steps": 100},
            )
            self.assertNotEqual(smoke["protocol_fingerprint"], protocol_fingerprint(self.config))
            (run / "ued_validation" / "model_1000.json").write_text(json.dumps(smoke), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "protocol fingerprint mismatch"):
                load_candidates(run, self.config)

    # --- reviewer 8c: best_spnte.pt is published atomically --------------------

    def test_materialize_best_spnte_leaves_no_temp_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for iteration in range(200, 3001, 200):
                self.write_checkpoint_only(run, iteration)
            for iteration in (1000, 1400, 1800, 2200, 2600, 3000):
                self.write_candidate(run, iteration)
            _, target = select_run(run, self.config)
            self.assertTrue(target.is_file())
            self.assertFalse((run / "best_spnte.pt.tmp").exists())

    # --- reviewer 8a: shard partials are provenance-bound before merge ---------

    def _shard(self, index, num_shards, measurements, *, sha="sha0", rollout_steps=None):
        return make_shard_payload(
            checkpoint_iteration=1000,
            checkpoint_sha256=sha,
            bank_kind="validation",
            shard_index=index,
            num_shards=num_shards,
            config=self.config,
            measurements=measurements,
            rollout_steps=rollout_steps,
        )

    def _merge(self, payloads, *, sha="sha0", num_shards=2):
        return merge_shard_payloads(
            payloads,
            checkpoint_iteration=1000,
            checkpoint_sha256=sha,
            bank_kind="validation",
            num_shards=num_shards,
            config=self.config,
        )

    def test_shard_round_trip_reconstructs_full_bank(self):
        meas = self.measurements()
        shards = [self._shard(0, 2, meas[0::2]), self._shard(1, 2, meas[1::2])]
        merged = self._merge(shards)
        self.assertEqual(len(merged), BANK_SIZE)
        artifact = make_validation_artifact(
            checkpoint_iteration=1000, measurements=merged, config=self.config, checkpoint_sha256="sha0",
        )
        self.assertEqual(len(artifact["measurements"]), BANK_SIZE)

    def test_merge_rejects_bare_headerless_shard(self):
        meas = self.measurements()
        with self.assertRaisesRegex(ValueError, "provenance-bound envelope"):
            self._merge([meas[0::2], meas[1::2]])

    def test_merge_rejects_foreign_checkpoint_sha(self):
        meas = self.measurements()
        shards = [self._shard(0, 2, meas[0::2], sha="other"), self._shard(1, 2, meas[1::2], sha="other")]
        with self.assertRaisesRegex(ValueError, "SHA-256 does not match"):
            self._merge(shards, sha="sha0")

    def test_merge_accepts_consistent_smoke_horizon_shards(self):
        """Short-horizon smoke shards may merge when they agree; selection still rejects."""
        meas = self.measurements()
        shards = [
            self._shard(0, 2, meas[0::2], rollout_steps=50),
            self._shard(1, 2, meas[1::2], rollout_steps=50),
        ]
        # Consensus-only merge (no explicit horizon): smoke path is alive.
        merged = self._merge(shards)
        self.assertEqual(len(merged), BANK_SIZE)
        # Explicit effective horizon also merges when it matches the shards.
        merged_explicit = merge_shard_payloads(
            shards,
            checkpoint_iteration=1000,
            checkpoint_sha256="sha0",
            bank_kind="validation",
            num_shards=2,
            config=self.config,
            rollout_steps=50,
        )
        self.assertEqual(len(merged_explicit), BANK_SIZE)
        smoke_artifact = make_validation_artifact(
            checkpoint_iteration=1000,
            measurements=merged,
            config=self.config,
            checkpoint_sha256="sha0",
            provenance={"rollout_steps": 50, "warmup_steps": 100},
        )
        self.assertNotEqual(
            smoke_artifact["protocol_fingerprint"],
            protocol_fingerprint(self.config),
        )

    def test_merge_rejects_disagreeing_horizons(self):
        meas = self.measurements()
        shards = [
            self._shard(0, 2, meas[0::2], rollout_steps=50),
            self._shard(1, 2, meas[1::2], rollout_steps=100),
        ]
        with self.assertRaisesRegex(ValueError, "protocol fingerprint mismatch"):
            self._merge(shards)

    def test_merge_rejects_smoke_shards_when_headline_horizon_forced(self):
        """Forcing the frozen headline horizon must not legitimise smoke shards."""
        meas = self.measurements()
        shards = [
            self._shard(0, 2, meas[0::2], rollout_steps=50),
            self._shard(1, 2, meas[1::2], rollout_steps=50),
        ]
        with self.assertRaisesRegex(ValueError, "protocol fingerprint mismatch"):
            merge_shard_payloads(
                shards,
                checkpoint_iteration=1000,
                checkpoint_sha256="sha0",
                bank_kind="validation",
                num_shards=2,
                config=self.config,
                rollout_steps=int(self.config["rollout"]["steps"]),
            )

    def test_merge_rejects_wrong_count_duplicate_and_missing_indices(self):
        meas = self.measurements()
        with self.assertRaisesRegex(ValueError, "num_shards mismatch"):
            self._merge([self._shard(0, 3, meas[0::2]), self._shard(1, 3, meas[1::2])], num_shards=2)
        with self.assertRaisesRegex(ValueError, "duplicate shard_index"):
            self._merge([self._shard(0, 2, meas[0::2]), self._shard(0, 2, meas[1::2])])
        with self.assertRaisesRegex(ValueError, "missing shard indices"):
            self._merge([self._shard(0, 2, meas[0::2])])

    def test_rollout_boundary_is_mockable(self):
        def mocked_rollout(rows):
            self.assertEqual(len(rows), BANK_SIZE)
            return self.measurements()

        artifact = evaluate_with_rollout(1000, self.config, mocked_rollout)
        self.assertEqual(artifact["checkpoint_iteration"], 1000)
        self.assertEqual(len(artifact["measurements"]), BANK_SIZE)

    # --- mid-window command resampling must not fire during bank scoring -------

    def test_bank_eval_env_disables_in_window_command_resampling(self):
        """Training default resampling_time lands inside the measure window.

        ``legged_robot._post_physics_step_callback`` resamples when
        ``episode_length % int(resampling_time / dt) == 0``.  At the V5
        training default (10 s / 0.02 s = 500 steps) that is one random
        3-axis command mid-window while SPNTE is scored against the pinned
        bank setpoint.  ``apply_bank_eval_env_overrides`` must push the
        period strictly past warmup+steps (same contract as campaign/play).
        """
        import legged_gym.envs  # noqa: F401  -- register go2_v5_* tasks
        from legged_gym.scripts.eval.ued_rollout import (
            _BANK_EVAL_RESAMPLING_TIME_S,
            apply_bank_eval_env_overrides,
        )
        from legged_gym.utils import task_registry

        registered_env_cfg, _ = task_registry.get_cfgs(name="go2_v5_lpacrl")
        env_cfg = copy.deepcopy(registered_env_cfg)
        dt = float(env_cfg.control.dt)
        steps = int(self.config["rollout"]["steps"])
        warmup = int(self.config["rollout"]["warmup_steps"])
        horizon = warmup + steps

        default_period = int(env_cfg.commands.resampling_time / dt)
        self.assertGreater(dt, 0.0)
        # Document the bug condition this override closes: default period is
        # inside the measure window (and therefore also inside warmup+steps).
        self.assertLessEqual(default_period, steps)
        self.assertLessEqual(default_period, horizon)

        apply_bank_eval_env_overrides(env_cfg)
        period = int(env_cfg.commands.resampling_time / dt)
        self.assertEqual(env_cfg.commands.resampling_time, _BANK_EVAL_RESAMPLING_TIME_S)
        self.assertGreater(period, steps)
        self.assertGreater(period, horizon)
        self.assertFalse(env_cfg.commands.heading_command)
        self.assertEqual(env_cfg.commands.zero_cmd_prob, 0.0)
        self.assertFalse(env_cfg.commands.per_env_standstill)
        self.assertFalse(env_cfg.env.ued_enabled)


if __name__ == "__main__":
    unittest.main()
