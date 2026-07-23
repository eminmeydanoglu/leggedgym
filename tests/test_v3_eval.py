"""CPU-only contract tests for the incremental V3 scorecard runner."""

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("SIMULATOR", "genesis")

from legged_gym.scripts.eval.v3_eval import (  # noqa: E402
    _planned_cells,
    _scope,
    _sysid_trace_summary,
    _terrain_raw_rows,
    cmd_aggregate,
    load_config,
    payload_com_x,
    score_cell,
)


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
        pred = torch.tensor([[2.0, -2.0], [-2.0, 2.0]])
        truth = torch.zeros_like(pred)
        summary = _sysid_trace_summary(pred, truth)
        np.testing.assert_allclose(summary["error_mean"].numpy(), [0.0, 0.0])
        np.testing.assert_allclose(summary["mae"].numpy(), [2.0, 2.0])
        np.testing.assert_allclose(summary["rmse"].numpy(), [2.0, 2.0])


class TestV3IncrementalAggregation(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
