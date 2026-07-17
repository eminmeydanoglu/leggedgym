"""CPU-only contract tests for the incremental V3 scorecard runner."""

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
    _sysid_trace_summary,
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


if __name__ == "__main__":
    unittest.main()
