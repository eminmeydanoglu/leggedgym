"""CPU-only contract tests for the incremental V3 scorecard runner."""

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

os.environ.setdefault("SIMULATOR", "genesis")

from legged_gym.scripts.eval.v3_eval import (  # noqa: E402
    _planned_cells,
    _discover_run,
    _run_cell,
    _v4_artifact_valid,
    _v4_campaign_fingerprint,
    _v4_normalized_headroom_payload,
    _v4_protocol_fingerprint,
    _scope,
    _sysid_trace_summary,
    _terrain_raw_rows,
    classify_v4_tracking_world,
    cmd_aggregate,
    load_config,
    payload_com_x,
    score_cell,
)
from legged_gym.scripts.eval.headroom_report import normalise_json  # noqa: E402


class TestV3EvalPlan(unittest.TestCase):
    @staticmethod
    def payload_cfg():
        return {"com_x_per_kg": 0.01, "com_x_limit_m": 0.08}

    def test_payload_mapping_is_signed_and_clamped(self):
        cfg = self.payload_cfg()
        self.assertAlmostEqual(payload_com_x(-2.0, cfg), -0.02)
        self.assertAlmostEqual(payload_com_x(5.0, cfg), 0.05)
        self.assertAlmostEqual(payload_com_x(12.0, cfg), 0.08)

    def test_frozen_config_plans_384_env_cells(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(str(root / "configs/eval/v3_ilk_deneme.yaml"))
        self.assertEqual(cfg["protocol"]["num_envs"], 384)
        self.assertEqual(len(_planned_cells(cfg, "s0")), 24)  # 6 models x 2 seeds x static/dynamic
        self.assertEqual(len(_planned_cells(cfg, "s1")), 432)  # 6 x 2 x 6 payload x 6 headline commands
        self.assertEqual(len(_planned_cells(cfg, "s2")), 108)  # 6 x 2 x 3 switch x 3 headline commands
        self.assertEqual(len(_planned_cells(cfg, "s1", include_diagnostics=True)), 720)
        self.assertEqual(len(_planned_cells(cfg, "s2", include_diagnostics=True)), 180)

    def test_v4_headroom_matrix_smoke_expands_isolated_and_secondary_cells(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(str(root / "configs/eval/v4_headroom_smoke.yaml"))
        cells = _planned_cells(cfg, "all")
        # Four adapters x two training seeds x (two primary + one secondary).
        self.assertEqual(len(cells), 24)
        primary = [cell for cell in cells if cell.scenario["tier"] == "primary_nominal_headroom"]
        secondary = [cell for cell in cells if cell.scenario["tier"] == "secondary_combined_stress_payload"]
        self.assertEqual(len(primary), 16)
        self.assertEqual(len(secondary), 8)
        self.assertEqual({cell.seed for cell in cells}, {1, 2})
        self.assertEqual({cell.model for cell in cells}, {"MLP", "Superset-Oracle", "DreamWaQ", "HIM-fixed"})
        self.assertTrue(all(cell.scenario["terrain_type"] == "stairs_up" for cell in cells))
        self.assertTrue(all(cell.scenario["terrain_level"] == 5 for cell in cells))
        self.assertTrue(all(cell.scenario["command_vx"] == 0.8 for cell in cells))
        self.assertTrue(all(cell.scenario["physics"]["com_x_m"] == 0.0 and cell.scenario["physics"]["friction"] == 1.0
                            for cell in primary))
        self.assertEqual({cell.scenario["physics"]["mass_kg"] for cell in primary}, {0.0, 6.0})
        self.assertTrue(all(cell.scenario["physics_axis"] == "combined" for cell in secondary))

    def test_v4_run_cell_does_not_require_legacy_num_envs(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(str(root / "configs/eval/v4_headroom_smoke.yaml"))
        cell = _planned_cells(cfg, "all")[0]
        models = {cell.model: {"label": cell.model}}
        with tempfile.TemporaryDirectory() as tmp, \
                patch("legged_gym.scripts.eval.v3_eval._run_v4_headroom_cell") as run_v4:
            _run_cell(Path(tmp), cfg, models, cell, resume=False)
        run_v4.assert_called_once()

    def test_v4_resume_uses_strict_matrix_artifact_validation(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(str(root / "configs/eval/v4_headroom_smoke.yaml"))
        cell = _planned_cells(cfg, "all")[0]
        models = {cell.model: {"label": cell.model}}
        with tempfile.TemporaryDirectory() as tmp, \
                patch("legged_gym.scripts.eval.v3_eval._artifact_valid", return_value=True), \
                patch("legged_gym.scripts.eval.v3_eval._run_v4_headroom_cell") as run_v4:
            _run_cell(Path(tmp), cfg, models, cell, resume=True)
        run_v4.assert_called_once_with(Path(tmp), cfg, models[cell.model], cell, resume=True)

    def test_v4_headroom_full_plan_matches_declared_budget(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(str(root / "configs/eval/v4_headroom.yaml"))
        cells = _planned_cells(cfg, "all")
        self.assertEqual(len(cells), cfg["protocol"]["planned_cell_budget"]["total"])
        with self.assertRaisesRegex(ValueError, "--suite all"):
            _planned_cells(cfg, "t1")

    def test_v4_resume_rejects_a_finite_artifact_with_wrong_protocol_identity(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(str(root / "configs/eval/v4_headroom_smoke.yaml"))
        cell = _planned_cells(cfg, "all")[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cell.npz"
            scenario = cell.scenario
            np.savez(path, tracking_lin=np.array([0.1]), tracking_yaw=np.array([0.0]), fall_rate=np.array([0.0]),
                     return_per_step=np.array([1.0]), protocol_kind="v4_headroom_matrix_v1",
                     protocol_fingerprint=_v4_protocol_fingerprint(cfg), campaign_fingerprint=_v4_campaign_fingerprint(cfg),
                     training_seed=cell.seed, model=cell.model,
                     tier=scenario["tier"], terrain_type_name=scenario["terrain_type"],
                     terrain_level=scenario["terrain_level"], command_vx=scenario["command_vx"],
                     physics_axis=scenario["physics_axis"], physics_band=scenario["physics_band"],
                     physics_signature=json.dumps(scenario["physics"], sort_keys=True, separators=(",", ":")))
            self.assertTrue(_v4_artifact_valid(path, cfg, cell))
            other = json.loads(json.dumps(cfg)); other["protocol"]["measured_steps"] += 1
            self.assertFalse(_v4_artifact_valid(path, other, cell))

    def test_ambiguous_run_wildcard_requires_an_explicit_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("run_a_seed1", "run_b_seed1"):
                run = root / name; run.mkdir()
                (run / "best_spnte.pt").write_bytes(b"checkpoint")
            model = {"label": "MLP", "task": "go2_v4_mlp", "log_root": str(root),
                     "run_pattern": "*_seed{seed}", "checkpoint": "best_spnte"}
            with patch("legged_gym.scripts.eval.v3_eval.resolve_checkpoint_path", side_effect=lambda run, _kind: str(Path(run) / "best_spnte.pt")), \
                    patch("legged_gym.scripts.eval.v3_eval.verify_run_identity"):
                with self.assertRaisesRegex(RuntimeError, "multiple verified runs"):
                    _discover_run(model, 1, require_final=False)
                model["run_paths"] = {1: "run_a_seed1"}
                self.assertEqual(_discover_run(model, 1, require_final=False), root / "run_a_seed1")


class TestV3Scorecard(unittest.TestCase):
    score_cfg = {
        "relative_headroom": 0.10,
        "absolute_headroom": 0.05,
        "fall_gate_pp": 0.05,
        "achieved_speed_ratio": 0.90,
    }

    @staticmethod
    def row(seed, error, fall, speed=1.0):
        return {"training_seed": seed, "tracking_error": error, "fall_rate": fall, "achieved_speed_ratio": speed}

    def test_best_stable_oracle_and_positive_gap(self):
        baseline = self.row(0, 1.0, 0.10)
        oracle = [self.row(1, 0.7, 0.10, 0.95), self.row(2, 0.5, 0.20, 0.99)]
        method = self.row(1, 0.8, 0.12)
        out = score_cell(baseline, oracle, method, self.score_cfg)
        self.assertEqual(out["headline_status"], "eligible")
        self.assertEqual(out["oracle_seed"], 1)  # seed 2 is not a stable ceiling
        self.assertAlmostEqual(out["raw_gap_closed"], 2.0 / 3.0)

    def test_invalid_oracle_is_not_scored_as_zero(self):
        out = score_cell(self.row(0, 1.0, 0.0), [self.row(1, 0.6, 0.0, 0.7)], self.row(1, 0.8, 0.0), self.score_cfg)
        self.assertEqual(out["headline_status"], "oracle_speed_saturated")
        self.assertFalse(out["headline_include"])
        self.assertNotIn("headline_gap_closed", out)

    def test_method_fall_gate_retains_raw_score_but_zeroes_headline(self):
        out = score_cell(self.row(0, 1.0, 0.0), [self.row(1, 0.5, 0.0)], self.row(1, 0.6, 0.10), self.score_cfg)
        self.assertEqual(out["headline_status"], "method_fall_gated")
        self.assertTrue(out["headline_include"])
        self.assertGreater(out["raw_gap_closed"], 0.0)
        self.assertEqual(out["headline_gap_closed"], 0.0)


class TestV3SysIDTrace(unittest.TestCase):
    def test_mae_and_rmse_do_not_cancel_opposite_errors(self):
        # Genesis initialization in an earlier test may set Torch's global
        # default device to CUDA; this unit contract intentionally uses NumPy.
        pred = torch.tensor([[2.0, -2.0], [-2.0, 2.0]], device="cpu")
        truth = torch.zeros_like(pred)
        summary = _sysid_trace_summary(pred, truth)
        np.testing.assert_allclose(summary["error_mean"].numpy(), [0.0, 0.0])
        np.testing.assert_allclose(summary["mae"].numpy(), [2.0, 2.0])
        np.testing.assert_allclose(summary["rmse"].numpy(), [2.0, 2.0])


class TestV3IncrementalAggregation(unittest.TestCase):
    def test_legacy_scorecard_forms_cells_within_the_same_training_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed, baseline_error, oracle_error, method_error in (
                (1, 1.0, 0.5, 0.75),
                (2, 3.0, 2.0, 2.5),
            ):
                for model, error in (("MLP", baseline_error), ("Superset-Oracle", oracle_error), ("SysID", method_error)):
                    raw = root / "raw" / "s1" / model / f"seed_{seed}"
                    raw.mkdir(parents=True, exist_ok=True)
                    np.savez(raw / "payload_nominal.npz", suite="s1", scenario="payload_nominal", model=model,
                             training_seed=seed, tracking_lin=np.array([error]), tracking_yaw=np.array([0.0]),
                             fall_rate=np.array([0.0]), return_per_step=np.array([1.0]),
                             achieved_speed=np.array([1.0]), achieved_speed_ratio=np.array([1.0]))
            cfg = {"campaign": "legacy_seed_contract", "artifact_root": str(root),
                   "scorecard": {"baseline_label": "MLP", "oracle_label": "Superset-Oracle", "method_labels": ["SysID"],
                                 "relative_headroom": 0.1, "absolute_headroom": 0.05, "fall_gate_pp": 0.05,
                                 "achieved_speed_ratio": 0.9}}
            self.assertEqual(cmd_aggregate(cfg), 0)
            with (root / "tables" / "scorecard_cells.csv").open() as f:
                cells = list(csv.DictReader(f))
            self.assertEqual({row["training_seed"] for row in cells}, {"1", "2"})
            self.assertTrue(all(abs(float(row["raw_gap_closed"]) - 0.5) < 1e-9 for row in cells))

    def test_partial_campaign_keeps_raw_data_without_requiring_oracle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw" / "s1" / "MLP" / "seed_1"
            raw.mkdir(parents=True)
            np.savez(
                raw / "payload_nominal__lateral_0_5.npz",
                suite="s1", scenario="payload_nominal__lateral_0_5", model="MLP", training_seed=1,
                tracking_lin=np.array([0.4, 0.5]), tracking_yaw=np.array([0.0, 0.0]),
                fall_rate=np.array([0.0, 0.0]), return_per_step=np.array([1.0, 1.0]),
                achieved_speed=np.array([0.5, 0.5]), achieved_speed_ratio=np.array([1.0, 1.0]),
            )
            cfg = {
                "campaign": "incremental_test", "artifact_root": str(root),
                "scorecard": {"baseline_label": "MLP", "oracle_label": "Superset-Oracle", "method_labels": ["SysID"],
                              "relative_headroom": 0.1, "absolute_headroom": 0.05, "fall_gate_pp": 0.05,
                              "achieved_speed_ratio": 0.9},
            }
            self.assertEqual(cmd_aggregate(cfg), 0)
            self.assertTrue((root / "tables" / "raw_cells.csv").is_file())
            summary = json.loads((root / "tables" / "headline.json").read_text())
            self.assertEqual(summary["headline"], [])

    def test_command_ood_rows_remain_raw_but_never_enter_scorecard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for model, error in (("MLP", 1.0), ("Superset-Oracle", 0.5), ("SysID", 0.7)):
                raw = root / "raw" / "s1" / model / "seed_1"
                raw.mkdir(parents=True)
                np.savez(
                    raw / "lateral_ood.npz", suite="s1", scenario="lateral_ood", model=model, training_seed=1,
                    tracking_lin=np.array([error]), tracking_yaw=np.array([0.0]), fall_rate=np.array([0.0]),
                    return_per_step=np.array([1.0]), achieved_speed=np.array([1.5]), achieved_speed_ratio=np.array([1.0]),
                    command_speed=1.5, headline_command=False, diagnostic_kind="command_ood_diagnostic",
                )
            cfg = {
                "campaign": "diagnostic_test", "artifact_root": str(root),
                "scorecard": {"baseline_label": "MLP", "oracle_label": "Superset-Oracle", "method_labels": ["SysID"],
                              "relative_headroom": 0.1, "absolute_headroom": 0.05, "fall_gate_pp": 0.05,
                              "achieved_speed_ratio": 0.9},
            }
            self.assertEqual(cmd_aggregate(cfg), 0)
            summary = json.loads((root / "tables" / "headline.json").read_text())
            self.assertEqual(summary["scorecard_cells"], 0)
            self.assertEqual(summary["command_ood_diagnostic_raw_cells"], 3)


class TestV3EvalTerrainPlan(unittest.TestCase):
    @staticmethod
    def cfg():
        root = Path(__file__).resolve().parents[1]
        return load_config(str(root / "configs/eval/v4_terrain.yaml"))

    def test_terrain_suites_plan_expected_counts(self):
        cfg = self.cfg()
        self.assertEqual(len(_planned_cells(cfg, "t0")), 6)  # 6 models x 1 seed
        self.assertEqual(len(_planned_cells(cfg, "t1")), 30)  # x 5 severity levels
        self.assertEqual(len(_planned_cells(cfg, "t2")), 18)  # x 3 payloads
        # This campaign configures no flat s-suites, so 'all' is terrain-only.
        self.assertEqual(len(_planned_cells(cfg, "all")), 6 + 30 + 18)

    def test_seed_count_is_configurable_and_append_only(self):
        cfg = self.cfg()
        base = len(_planned_cells(cfg, "t1"))
        wider = json.loads(json.dumps(cfg))
        wider["training_seeds"] = [1, 2, 3]
        self.assertEqual(len(_planned_cells(wider, "t1")), base * 3)
        # A per-model override wins over the campaign-wide seed list.
        mixed = json.loads(json.dumps(cfg))
        mixed["training_seeds"] = [1, 2]
        mixed["models"][0]["training_seeds"] = [1]  # MLP stays single-seed
        self.assertEqual(len(_planned_cells(mixed, "t1")), 5 * 2 * 5 + 1 * 1 * 5)


class TestV3EvalTerrainAggregation(unittest.TestCase):
    TYPE_NAMES = ["slope", "random_uniform", "stairs_down", "stairs_up", "discrete"]

    @staticmethod
    def _save_terrain_cell(root, suite, model, seed, name, level, error, *,
                           payload_tier="", payload_name="", num_cols=5, fall=0.0, ratio=1.0,
                           terrain_hash="deadbeef"):
        raw = root / "raw" / suite / model / f"seed_{seed}"
        raw.mkdir(parents=True, exist_ok=True)
        n = num_cols * 2
        terrain_type = np.repeat(np.arange(num_cols), 2).astype(np.int64)
        np.savez(
            raw / f"{name}.npz",
            suite=suite, scenario=name, model=model, training_seed=seed,
            terrain_level=level, terrain_num_cols=num_cols, command_speed=1.0,
            headline_command=True, payload_tier=payload_tier, payload_name=payload_name,
            terrain_hash=terrain_hash, terrain_type=terrain_type,
            tracking_lin=np.full(n, float(error)), tracking_yaw=np.zeros(n),
            fall_rate=np.full(n, float(fall)), return_per_step=np.ones(n),
            achieved_speed=np.full(n, float(ratio)), achieved_speed_ratio=np.full(n, float(ratio)),
        )

    @staticmethod
    def terrain_cfg(root):
        return {
            "campaign": "terrain_test", "artifact_root": str(root),
            "scorecard": {"baseline_label": "MLP", "oracle_label": "Superset-Oracle",
                          "method_labels": ["SysID"], "relative_headroom": 0.1, "absolute_headroom": 0.05,
                          "fall_gate_pp": 0.05, "achieved_speed_ratio": 0.9},
        }

    def test_terrain_rows_expand_one_row_per_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._save_terrain_cell(root, "t1", "MLP", 1, "terrain_severity_level_5", 5, 1.0)
            rows = _terrain_raw_rows(root)
            self.assertEqual(len(rows), 5)
            self.assertEqual({row["scenario"] for row in rows},
                             {f"terrain_type_{name}_level_5" for name in self.TYPE_NAMES})
            self.assertTrue(all(row["num_replicas"] == 2 for row in rows))
            self.assertTrue(all(row["tracking_error"] == 1.0 for row in rows))

    def test_scope_maps_terrain_suites(self):
        self.assertEqual(_scope({"suite": "t1", "terrain_type_name": "stairs_up", "payload_tier": ""}),
                         "GapClosed_terrain_stairs_up")
        self.assertEqual(_scope({"suite": "t2", "terrain_type_name": "slope", "payload_tier": "near_ood"}),
                         "GapClosed_terrain_payload_near_ood")

    def test_terrain_scorecard_scores_gap_closed_per_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._save_terrain_cell(root, "t1", "MLP", 1, "terrain_severity_level_5", 5, 1.0)
            self._save_terrain_cell(root, "t1", "Superset-Oracle", 1, "terrain_severity_level_5", 5, 0.5)
            self._save_terrain_cell(root, "t1", "SysID", 1, "terrain_severity_level_5", 5, 0.8)
            self.assertEqual(cmd_aggregate(self.terrain_cfg(root)), 0)
            summary = json.loads((root / "tables" / "headline.json").read_text())
            self.assertEqual(summary["scorecard_cells"], 5)  # 5 types x 1 method
            with (root / "tables" / "scorecard_cells.csv").open() as _f:
                cells = list(csv.DictReader(_f))
            self.assertTrue(all(cell["scope"].startswith("GapClosed_terrain_") for cell in cells))
            self.assertTrue(all(cell["headline_status"] == "eligible" for cell in cells))
            # gap_closed = (1.0 - 0.8) / (1.0 - 0.5) = 0.4
            self.assertTrue(all(abs(float(cell["headline_gap_closed"]) - 0.4) < 1e-9 for cell in cells))

    def test_oracle_ceiling_gate_applies_on_terrain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Oracle is unstable (falls far more than the MLP baseline) at this
            # difficulty: every type-cell must drop out as a physical ceiling.
            self._save_terrain_cell(root, "t1", "MLP", 1, "terrain_severity_level_9", 9, 1.0, fall=0.0)
            self._save_terrain_cell(root, "t1", "Superset-Oracle", 1, "terrain_severity_level_9", 9, 0.5, fall=0.9)
            self._save_terrain_cell(root, "t1", "SysID", 1, "terrain_severity_level_9", 9, 0.8, fall=0.0)
            self.assertEqual(cmd_aggregate(self.terrain_cfg(root)), 0)
            with (root / "tables" / "scorecard_cells.csv").open() as _f:
                cells = list(csv.DictReader(_f))
            self.assertEqual(len(cells), 5)
            self.assertTrue(all(cell["headline_status"] == "oracle_unstable" for cell in cells))
            self.assertTrue(all(cell["headline_include"] in ("False", "") for cell in cells))

    def test_diverged_terrain_geometry_fails_loud(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._save_terrain_cell(root, "t1", "MLP", 1, "terrain_severity_level_5", 5, 1.0,
                                    terrain_hash="aaaa")
            # Oracle silently saw a different heightfield: the scorecard must
            # refuse to compare rather than emit an invalid gap-closed number.
            self._save_terrain_cell(root, "t1", "Superset-Oracle", 1, "terrain_severity_level_5", 5, 0.5,
                                    terrain_hash="bbbb")
            self._save_terrain_cell(root, "t1", "SysID", 1, "terrain_severity_level_5", 5, 0.8,
                                    terrain_hash="aaaa")
            with self.assertRaisesRegex(RuntimeError, "terrain geometry mismatch"):
                cmd_aggregate(self.terrain_cfg(root))


class TestV4WorldHeadroomAggregation(unittest.TestCase):
    @staticmethod
    def cfg(root):
        return {
            "campaign": "v4_headroom_contract_test", "artifact_root": str(root),
            "scorecard": {"baseline_label": "MLP", "oracle_label": "Superset-Oracle",
                          "method_labels": ["SysID", "DreamWaQ", "HIM-fixed"],
                          "relative_headroom": 0.10, "absolute_headroom": 0.10, "fall_gate_pp": 0.05,
                          "achieved_speed_ratio": 0.90,
                          "headline_tier": "primary_nominal_headroom"},
        }

    @staticmethod
    def save(root, model, seed, *, error, fall=0.0, ratio=1.0, vx=1.0,
             physics_tier="primary_nominal_headroom", scenario="mass_0", level=5, terrain_hash=None):
        raw = root / "raw" / "t1" / model / f"seed_{seed}"
        raw.mkdir(parents=True, exist_ok=True)
        np.savez(
            raw / f"{scenario}_{vx}_{physics_tier}.npz",
            suite="t1", scenario=scenario, model=model, training_seed=seed,
            terrain_level=level, terrain_num_cols=1, command_speed=vx, command_vx=vx,
            physics_tier=physics_tier, physics_signature=scenario,
            headline_command=True, terrain_hash=terrain_hash or f"geometry-L{level}", terrain_type=np.array([0]),
            tracking_lin=np.array([error]), tracking_yaw=np.array([0.0]), fall_rate=np.array([fall]),
            return_per_step=np.array([1.0]), achieved_speed=np.array([ratio]),
            achieved_speed_ratio=np.array([ratio]),
        )

    def test_references_define_tracking_world_before_adaptation_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.save(root, "MLP", 1, error=1.0)
            self.save(root, "Superset-Oracle", 1, error=0.8)
            # These extreme method rows must not select the world or alter its
            # headroom; they only receive the already selected comparison.
            self.save(root, "SysID", 1, error=0.7, fall=0.9)
            self.save(root, "DreamWaQ", 1, error=0.0, fall=1.0)
            self.assertEqual(cmd_aggregate(self.cfg(root)), 0)
            with (root / "tables" / "scorecard_cells.csv").open() as f:
                rows = list(csv.DictReader(f))
            sysid = next(row for row in rows if row["model"] == "SysID")
            self.assertEqual(sysid["world_classification"], "tracking")
            self.assertEqual(sysid["method_survival_status"], "fall_rate_gt_0.05")
            # (1.0 - 0.7) / (1.0 - 0.8) = 1.5: V4 must not clip it.
            self.assertAlmostEqual(float(sysid["raw_gap_closed"]), 1.5)
            self.assertEqual(sysid["headline_status"], "method_fall_gated")
            self.assertAlmostEqual(float(sysid["headline_gap_closed"]), 0.0)

    def test_survival_world_has_reason_but_no_tracking_gap_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.save(root, "MLP", 1, error=1.0, fall=0.06)
            self.save(root, "Superset-Oracle", 1, error=0.5)
            self.save(root, "SysID", 1, error=0.7)
            self.assertEqual(cmd_aggregate(self.cfg(root)), 0)
            with (root / "tables" / "scorecard_cells.csv").open() as f:
                cell = next(csv.DictReader(f))
            self.assertEqual(cell["world_classification"], "survival")
            self.assertEqual(cell["tracking_status"], "excluded_survival_world")
            self.assertIn("mlp_fall_rate_gt_0.05", cell["tracking_exclusion_reasons"])
            self.assertNotIn("raw_gap_closed", cell)
            summary = json.loads((root / "tables" / "headline.json").read_text())
            self.assertEqual(summary["headline"], [])

    def test_world_identity_and_seed_aggregation_never_pool_replicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed, method_error in ((1, 0.9), (2, 0.6)):
                self.save(root, "MLP", seed, error=1.0, vx=0.6)
                self.save(root, "Superset-Oracle", seed, error=0.8, vx=0.6)
                self.save(root, "SysID", seed, error=method_error, vx=0.6)
            # Same scenario but a distinct physical tier/vx is a distinct world
            # and must not enter the primary scope's seed values above.
            for model, error in (("MLP", 1.0), ("Superset-Oracle", 0.8), ("SysID", 0.8)):
                self.save(root, model, 1, error=error, vx=1.0,
                          physics_tier="secondary_combined_stress_payload", scenario="payload_plus4")
            self.assertEqual(cmd_aggregate(self.cfg(root)), 0)
            with (root / "tables" / "scorecard_worlds.csv").open() as f:
                worlds = list(csv.DictReader(f))
            self.assertEqual(len(worlds), 3)
            self.assertEqual({row["training_seed"] for row in worlds}, {"1", "2"})
            self.assertEqual({row["command_vx"] for row in worlds}, {"0.6", "1.0"})
            self.assertIn("secondary_combined_stress_payload", {row["physics_tier"] for row in worlds})
            with (root / "tables" / "scorecard_seed_scores.csv").open() as f:
                seeds = [row for row in csv.DictReader(f) if row["scope"] == "GapClosed_terrain_type0"]
            self.assertEqual({row["training_seed"] for row in seeds}, {"1", "2"})
            by_seed = {row["training_seed"]: float(row["gap_closed_median"]) for row in seeds}
            self.assertAlmostEqual(by_seed["1"], 0.5)
            self.assertAlmostEqual(by_seed["2"], 2.0)

    def test_v4_tracking_gate_uses_declared_scorecard_thresholds(self):
        gates = {"fall_gate_pp": 0.05, "achieved_speed_ratio": 0.90, "absolute_headroom": 0.10, "relative_headroom": 0.10}
        self.assertEqual(
            classify_v4_tracking_world(
                {"tracking_error": 1.0, "fall_rate": 0.0},
                {"tracking_error": 0.95, "fall_rate": 0.0, "achieved_speed_ratio": 0.89},
                gates,
            )["tracking_exclusion_reasons"],
            "oracle_achieved_speed_ratio_lt_0.90;absolute_tracking_headroom_lt_0.10;relative_tracking_headroom_lt_0.10",
        )

    def test_v4_rejects_mismatched_geometry_inside_one_world_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.save(root, "MLP", 1, error=1.0, terrain_hash="aaa")
            self.save(root, "Superset-Oracle", 1, error=0.8, terrain_hash="bbb")
            self.save(root, "SysID", 1, error=0.9, terrain_hash="aaa")
            with self.assertRaisesRegex(RuntimeError, "terrain geometry mismatch in V4 world"):
                cmd_aggregate(self.cfg(root))

    def test_v4_aggregate_writes_headroom_report_normalized_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.save(root, "MLP", 1, error=1.0)
            self.save(root, "Superset-Oracle", 1, error=0.8)
            self.assertEqual(cmd_aggregate(self.cfg(root)), 0)
            report_input = root / "tables" / "headroom_report_normalized.json"
            experiment, worlds = normalise_json(json.loads(report_input.read_text()))
            self.assertEqual(experiment.seed_count, 1)
            self.assertEqual(len(worlds), 1)
            self.assertTrue(worlds[0].include)

    def test_v4_normalized_report_deduplicates_seed_exclusion_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed in (1, 2):
                self.save(root, "MLP", seed, error=1.0)
                self.save(root, "Superset-Oracle", seed, error=0.95, ratio=0.89)
            self.assertEqual(cmd_aggregate(self.cfg(root)), 0)
            payload = json.loads((root / "tables" / "headroom_report_normalized.json").read_text())
            reason = payload["worlds"][0]["exclusion_reason"]
            self.assertEqual(reason.count("oracle_achieved_speed_ratio_lt_0.90"), 1)
            self.assertEqual(reason.count("absolute_tracking_headroom_lt_0.10"), 1)

    def test_normalized_html_world_names_include_the_physical_point(self):
        cfg = self.cfg(Path("/tmp/v4-name-contract"))
        base = {
            "suite": "h4", "terrain_type": "stairs_up", "terrain_level": 5, "command_vx": 0.8,
            "physics_tier": "primary_nominal_headroom", "eval_seed": 1,
            "physics_axis": "mass_kg", "physics_band": "id", "training_seed": 1,
            "tracking_include": True, "tracking_exclusion_reasons": "",
            "mlp_tracking_error": 1.0, "oracle_tracking_error": 0.5,
            "mlp_fall_rate": 0.0, "oracle_fall_rate": 0.0, "oracle_achieved_speed_ratio": 1.0,
        }
        worlds = [
            {**base, "physics_signature": json.dumps({"mass_kg": mass, "com_x_m": 0.0, "friction": 1.0}, sort_keys=True)}
            for mass in (-2.0, 0.0, 5.0)
        ]
        payload = _v4_normalized_headroom_payload(cfg, worlds, [])
        experiment, parsed = normalise_json(payload)
        self.assertEqual(experiment.seed_count, 1)
        self.assertEqual(len({world.name for world in parsed}), 3)
        self.assertTrue(any("kütle = +5 kg" in world.name for world in parsed))


if __name__ == "__main__":
    unittest.main()
