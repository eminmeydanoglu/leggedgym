"""Focused first-fall SPNTE regression tests."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from legged_gym.scripts.eval.indist import _command_support_scale, run_indist_eval
from legged_gym.scripts.eval.metrics import MetricAccumulator
from legged_gym.scripts.eval.v3_eval import (
    _raw_rows,
    _rollout_switch,
    cmd_aggregate,
    cmd_report,
    score_cell,
    score_cell_on,
    spnte_scales_from_command_bank,
)


def _update(acc, *, lin_x, lin=None, yaw=0.0, fall=False, timeout=False, reward=1.0):
    lin = lin_x if lin is None else lin
    acc.update(
        torch.tensor([reward]),
        torch.tensor([fall or timeout]),
        torch.tensor([timeout]),
        torch.tensor([lin]),
        torch.tensor([yaw]),
        lin_x_err=torch.tensor([lin_x]),
    )


class TestSPNTE(unittest.TestCase):
    def test_payload_switch_artifact_uses_first_fall_spnte_and_bank_scale(self):
        class Adapter:
            def reset(self, env):
                self.calls = 0
                return torch.zeros((1, 1))

            def act(self, state):
                return torch.zeros((1, 1))

            def step(self, env, action):
                self.calls += 1
                # The warmup is call 1.  The first two measured steps have
                # 0.2 x-error; the second one falls, after which auto-reset
                # would otherwise supply perfect tracking.
                x = 0.8 if self.calls in (2, 3) else 1.0
                env.simulator.base_lin_vel[:] = torch.tensor([[x, 0.0, 0.0]])
                env.simulator.base_ang_vel.zero_()
                return action, torch.ones(1), torch.tensor([self.calls == 3])

        env = SimpleNamespace(
            num_envs=1,
            device="cpu",
            commands=torch.zeros((1, 3)),
            episode_length_buf=torch.zeros(1, dtype=torch.long),
            time_out_buf=torch.zeros(1, dtype=torch.bool),
            simulator=SimpleNamespace(base_lin_vel=torch.zeros((1, 3)), base_ang_vel=torch.zeros((1, 3))),
        )
        session = SimpleNamespace(env=env, adapter=Adapter(), runner=SimpleNamespace())
        bank = {"lin_vel_x": [0.0, 2.0], "ang_vel_yaw": [-2.0, 2.0]}
        with patch("legged_gym.scripts.eval.v3_eval._apply_payload", return_value={"switch_applied": np.asarray(1)}):
            got = _rollout_switch(
                session, np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), warmup=1,
                pre_steps=1, post_steps=4, target_mass=1.0, payload_cfg={}, command_bank=bank,
            )
        self.assertAlmostEqual(float(got["spnte_lin"][0]), 0.64, places=6)
        self.assertAlmostEqual(float(got["spnte_yaw"][0]), 0.6, places=6)
        self.assertEqual(int(got["first_fall_step"][0]), 2)
        self.assertEqual(float(got["spnte_v_scale"]), 2.0)
        self.assertEqual(float(got["spnte_yaw_scale"]), 2.0)
        # Existing dynamic-payload observables remain present and are still
        # derived from every stream step, including the reset episode.
        self.assertAlmostEqual(float(got["tracking_lin"][0]), 0.08, places=6)
        self.assertAlmostEqual(float(got["fall_rate"][0]), 1.0)

    def test_no_fall_constant_error_equals_normalized_tracking_error(self):
        acc = MetricAccumulator(1, "cpu", v_scale=2.0, v_scale_yaw=4.0)
        for _ in range(4):
            _update(acc, lin_x=1.0, yaw=1.0)
        got = acc.compute()
        self.assertAlmostEqual(got["spnte_lin"].item(), 0.5)
        self.assertAlmostEqual(got["spnte_yaw"].item(), 0.25)
        self.assertEqual(got["first_fall_step"].item(), 4)

    def test_first_fall_prevents_auto_reset_from_improving_score(self):
        acc = MetricAccumulator(1, "cpu", v_scale=1.0)
        _update(acc, lin_x=0.2)
        _update(acc, lin_x=0.2, fall=True)
        for _ in range(3):  # excellent post-reset rollout must not count
            _update(acc, lin_x=0.0)
        got = acc.compute()
        self.assertEqual(got["first_fall_step"].item(), 2)
        self.assertAlmostEqual(got["spnte_lin"].item(), (0.2 + 0.2 + 3.0) / 5.0)

    def test_zero_command_error_is_finite_and_bounded(self):
        acc = MetricAccumulator(1, "cpu", v_scale=2.0, v_scale_yaw=1.0)
        _update(acc, lin_x=0.0, yaw=0.0)
        _update(acc, lin_x=99.0, yaw=99.0)
        got = acc.compute()
        for key in ("spnte_lin", "spnte_yaw"):
            self.assertTrue(torch.isfinite(got[key]).all())
            self.assertGreaterEqual(got[key].item(), 0.0)
            self.assertLessEqual(got[key].item(), 1.0)

    def test_dynamic_support_scale_is_persisted_in_v3_artifact_rows(self):
        self.assertEqual(spnte_scales_from_command_bank({"lin_vel_x": [-1, 1], "ang_vel_yaw": [-1, 1]})[0], 1.0)
        self.assertEqual(spnte_scales_from_command_bank({"lin_vel_x": [0, 2], "ang_vel_yaw": [-1, 1]})[0], 2.0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            np.savez(
                raw / "fixed.npz", suite="s0", scenario="static_id", model="MLP", training_seed=1,
                tracking_lin=np.array([0.3]), fall_rate=np.array([0.0]), return_per_step=np.array([1.0]),
                spnte_lin=np.array([0.15]), spnte_yaw=np.array([0.2]), spnte_v_scale=np.asarray(2.0),
            )
            row = _raw_rows(root)[0]
            self.assertEqual(row["spnte_v_scale"], 2.0)
            self.assertEqual(row["spnte_lin"], 0.15)

    def test_v3_report_shows_append_only_spnte_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            np.savez(
                raw / "fixed.npz", suite="s0", scenario="static_id", model="MLP", training_seed=1,
                tracking_lin=np.array([0.3]), fall_rate=np.array([0.0]), return_per_step=np.array([1.0]),
                spnte_lin=np.array([0.15]), spnte_yaw=np.array([0.2]), spnte_v_scale=np.asarray(2.0),
            )
            cfg = {
                "campaign": "spnte_report", "artifact_root": str(root),
                "scorecard": {"baseline_label": "MLP", "oracle_label": "Oracle", "method_labels": [],
                              "relative_headroom": 0.1, "absolute_headroom": 0.05,
                              "fall_gate_pp": 0.05, "achieved_speed_ratio": 0.9},
            }
            self.assertEqual(cmd_aggregate(cfg), 0)
            self.assertEqual(cmd_report(cfg), 0)
            report = (root / "report" / "scorecard.md").read_text()
            self.assertIn("SPNTE (append-only raw cross-check)", report)
            self.assertIn("| s0 | MLP | 0.1500 | 0.2000 | 2 | 1 |", report)

    def test_legacy_metrics_keep_their_all_stream_semantics(self):
        acc = MetricAccumulator(1, "cpu", v_scale=1.0)
        _update(acc, lin_x=1.0, lin=3.0, fall=True, reward=2.0)
        _update(acc, lin_x=0.0, lin=0.0, reward=4.0)
        got = acc.compute()
        # These are the pre-SPNTE definitions: all stream steps remain visible
        # after the auto-reset, and fall_rate is per completed episode.
        self.assertAlmostEqual(got["tracking_lin_err"].item(), 1.5)
        self.assertAlmostEqual(got["fall_rate"].item(), 1.0)
        self.assertAlmostEqual(got["return_per_step"].item(), 3.0)


class TestCommandSupportScale(unittest.TestCase):
    """`_command_support_scale` derives the in-training-eval SPNTE v_scale."""

    def test_symmetric_range_matches_max_abs_endpoint(self):
        self.assertEqual(_command_support_scale({"lin_vel_x": [-1, 1]}, "lin_vel_x"), 1.0)

    def test_asymmetric_range_takes_larger_abs_endpoint(self):
        self.assertEqual(_command_support_scale({"lin_vel_x": [-2, 3]}, "lin_vel_x"), 3.0)

    def test_missing_key_falls_back_to_default(self):
        self.assertEqual(_command_support_scale({}, "lin_vel_x"), 1.0)
        self.assertEqual(_command_support_scale({}, "lin_vel_x", default=2.5), 2.5)

    def test_degenerate_zero_range_floors_at_epsilon_not_zero(self):
        scale = _command_support_scale({"lin_vel_x": [0.0, 0.0]}, "lin_vel_x")
        self.assertGreater(scale, 0.0)
        self.assertLess(scale, 1e-3)


class _FakeSimulator(SimpleNamespace):
    pass


class _FakeIndistEnv:
    """Minimal fake env driving `run_indist_eval` without a real simulator.

    Mirrors the fake-env pattern already used in the v3_eval SPNTE test above:
    a scripted adapter drives known tracking errors and a single fall so the
    resulting spnte_lin/spnte_yaw are hand-computable.
    """

    def __init__(self):
        self.num_envs = 1
        self.device = "cpu"
        self.command_ranges = {
            "lin_vel_x": [-1.0, 1.0], "lin_vel_y": [-1.0, 1.0], "ang_vel_yaw": [-1.0, 1.0],
        }
        self.cfg = SimpleNamespace(commands=SimpleNamespace(curriculum=True))
        self.commands = torch.tensor([[1.0, 0.0, 0.0]])
        self.episode_length_buf = torch.zeros(1, dtype=torch.long)
        self.time_out_buf = torch.zeros(1, dtype=torch.bool)
        self.actions = torch.zeros((1, 1))
        self.last_actions = torch.zeros((1, 1))
        self.simulator = _FakeSimulator(
            base_lin_vel=torch.zeros((1, 3)), base_ang_vel=torch.zeros((1, 3)),
            projected_gravity=torch.zeros((1, 3)), torques=torch.zeros((1, 1)),
            dof_vel=torch.zeros((1, 1)), feet_vel=torch.zeros((1, 1, 3)),
        )
        self.feet_max_force_z = torch.zeros((1, 1))

    def reset(self):
        pass


class _FakeIndistAdapter:
    """Scripted rollout: 0.2 x-error on the first two measured steps, a fall
    on the second, then perfect tracking that must NOT un-freeze SPNTE."""

    def __init__(self):
        self.calls = 0

    def reset(self, env):
        return torch.zeros((1, 1))

    def act(self, state):
        return torch.zeros((1, 1))

    def step(self, env, action):
        self.calls += 1
        x_vel = 0.8 if self.calls in (1, 2) else 1.0
        env.simulator.base_lin_vel[:] = torch.tensor([[x_vel, 0.0, 0.0]])
        env.simulator.base_ang_vel.zero_()
        fall = torch.tensor([self.calls == 2])
        return action, torch.ones(1), fall


class TestRunIndistEvalSPNTE(unittest.TestCase):
    """`run_indist_eval` surfaces spnte_lin/spnte_yaw with a dynamic v_scale."""

    def test_returns_spnte_metrics_with_v_scale_one_for_symmetric_unit_range(self):
        env = _FakeIndistEnv()
        adapter = _FakeIndistAdapter()
        metrics = run_indist_eval(
            env, adapter=adapter, steps=4, warmup=0,
            command_ranges={"lin_vel_x": [-1.0, 1.0], "ang_vel_yaw": [-1.0, 1.0]},
        )
        self.assertIn("spnte_lin", metrics)
        self.assertIn("spnte_yaw", metrics)
        # v_scale=1.0 (from lin_vel_x=[-1,1]): errors 0.2, 0.2(fall), then two
        # steps charged at the 1.0 tail penalty -> (0.2+0.2+1.0+1.0)/4.
        self.assertAlmostEqual(metrics["spnte_lin"], (0.2 + 0.2 + 1.0 + 1.0) / 4.0, places=6)
        # yaw error is 0 throughout, so only the tail penalty contributes.
        self.assertAlmostEqual(metrics["spnte_yaw"], (0.0 + 0.0 + 1.0 + 1.0) / 4.0, places=6)


class TestScoreCellOn(unittest.TestCase):
    """``score_cell_on`` reuses ``score_cell``'s gate rule for any error field."""

    score_cfg = {
        "relative_headroom": 0.10,
        "absolute_headroom": 0.05,
        "fall_gate_pp": 0.05,
        "achieved_speed_ratio": 0.90,
    }

    @staticmethod
    def row(seed, tracking_error, spnte_lin, fall, speed=1.0):
        return {"training_seed": seed, "tracking_error": tracking_error, "spnte_lin": spnte_lin,
                "fall_rate": fall, "achieved_speed_ratio": speed}

    def test_tracking_error_field_is_a_no_op_identical_to_score_cell(self):
        baseline = self.row(0, 1.0, 1.0, 0.0)
        oracle = [self.row(1, 0.5, 0.5, 0.0, 0.95)]
        method = self.row(1, 0.9, 1.2, 0.0, 0.95)
        direct = score_cell(baseline, oracle, method, self.score_cfg)
        via_wrapper = score_cell_on("tracking_error", baseline, oracle, method, self.score_cfg)
        self.assertEqual(direct, via_wrapper)

    def test_spnte_lin_gap_closed_can_diverge_from_tracking_gap_closed(self):
        # Method is better than baseline on tracking_error but *worse* than
        # baseline on spnte_lin: the two scored views must disagree, using
        # the identical gate rule (oracle stability/speed/headroom/fall-gate)
        # for both -- only the closed error field differs.
        baseline = self.row(0, 1.0, 1.0, 0.0)
        oracle = [self.row(1, 0.5, 0.5, 0.0, 0.95)]
        method = self.row(1, 0.9, 1.2, 0.0, 0.95)
        tracking_score = score_cell_on("tracking_error", baseline, oracle, method, self.score_cfg)
        spnte_score = score_cell_on("spnte_lin", baseline, oracle, method, self.score_cfg)
        self.assertEqual(tracking_score["headline_status"], "eligible")
        self.assertEqual(spnte_score["headline_status"], "eligible")
        self.assertAlmostEqual(tracking_score["raw_gap_closed"], 0.2)  # (1.0-0.9)/(1.0-0.5)
        self.assertAlmostEqual(spnte_score["raw_gap_closed"], -0.4)  # (1.0-1.2)/(1.0-0.5)
        self.assertNotAlmostEqual(tracking_score["headline_gap_closed"], spnte_score["headline_gap_closed"])

    def test_same_gates_apply_regardless_of_error_field(self):
        # An oracle that falls too much is gated identically whether the
        # score is being closed on tracking_error or on spnte_lin: the
        # oracle-stability check keys only on fall_rate, never on the error
        # field being scored.
        baseline = self.row(0, 1.0, 1.0, 0.0)
        unstable_oracle = [self.row(1, 0.5, 0.5, 0.9, 0.95)]
        method = self.row(1, 0.9, 1.2, 0.0, 0.95)
        for field in ("tracking_error", "spnte_lin"):
            out = score_cell_on(field, baseline, unstable_oracle, method, self.score_cfg)
            self.assertEqual(out["headline_status"], "oracle_unstable")
            self.assertFalse(out["headline_include"])


class TestSPNTEScoredAggregation(unittest.TestCase):
    """cmd_aggregate/cmd_report expose SPNTE gap-closed as a parallel scored view."""

    score_cfg = {
        "baseline_label": "MLP", "oracle_label": "Superset-Oracle", "method_labels": ["SysID"],
        "relative_headroom": 0.10, "absolute_headroom": 0.05, "fall_gate_pp": 0.05,
        "achieved_speed_ratio": 0.90,
    }
    SCENARIO = "payload_nominal__forward_1_0"

    @classmethod
    def _cfg(cls, root):
        return {"campaign": "spnte_scored_test", "artifact_root": str(root), "scorecard": cls.score_cfg}

    @classmethod
    def _save_s1_cell(cls, root, model, seed, tracking_lin, spnte_lin, *, fall=0.0, speed=1.0, with_spnte=True):
        raw = root / "raw" / "s1" / model / f"seed_{seed}"
        raw.mkdir(parents=True, exist_ok=True)
        fields = dict(
            suite="s1", scenario=cls.SCENARIO, model=model, training_seed=seed,
            tracking_lin=np.array([tracking_lin]), tracking_yaw=np.array([0.0]),
            fall_rate=np.array([fall]), return_per_step=np.array([1.0]),
            achieved_speed=np.array([speed]), achieved_speed_ratio=np.array([speed]),
        )
        if with_spnte:
            fields.update(spnte_lin=np.array([spnte_lin]), spnte_yaw=np.array([0.0]),
                          spnte_v_scale=np.asarray(1.0))
        np.savez(raw / f"{cls.SCENARIO}.npz", **fields)

    def test_scorecard_cells_gain_a_parallel_spnte_metric_kind_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # MLP=baseline, Superset-Oracle=oracle, SysID=method. Tracking
            # favors SysID (0.9 vs baseline 1.0); SPNTE disfavors it
            # (1.2 vs baseline 1.0), so the two scored views diverge.
            self._save_s1_cell(root, "MLP", 1, 1.0, 1.0)
            self._save_s1_cell(root, "Superset-Oracle", 1, 0.5, 0.5, speed=0.95)
            self._save_s1_cell(root, "SysID", 1, 0.9, 1.2, speed=0.95)
            self.assertEqual(cmd_aggregate(self._cfg(root)), 0)
            with (root / "tables" / "scorecard_cells.csv").open() as f:
                cells = list(csv.DictReader(f))
            self.assertEqual(len(cells), 2)  # 1 tracking + 1 spnte row for this single cell
            by_kind = {cell["metric_kind"]: cell for cell in cells}
            self.assertEqual(set(by_kind), {"tracking", "spnte"})
            self.assertAlmostEqual(float(by_kind["tracking"]["headline_gap_closed"]), 0.2, places=9)
            self.assertAlmostEqual(float(by_kind["spnte"]["headline_gap_closed"]), -0.4, places=9)
            self.assertEqual(by_kind["tracking"]["headline_status"], "eligible")
            self.assertEqual(by_kind["spnte"]["headline_status"], "eligible")
            # Metric-specific error columns are populated only on their own row.
            self.assertEqual(by_kind["tracking"]["method_tracking_error"], "0.9")
            self.assertEqual(by_kind["spnte"]["method_spnte_error"], "1.2")
            self.assertIn(by_kind["tracking"]["method_spnte_error"], ("", None))
            self.assertIn(by_kind["spnte"]["method_tracking_error"], ("", None))

            with (root / "tables" / "scorecard_seed_scores.csv").open() as f:
                seed_rows = list(csv.DictReader(f))
            self.assertEqual(len(seed_rows), 2)
            seed_by_kind = {row["metric_kind"]: row for row in seed_rows}
            self.assertAlmostEqual(float(seed_by_kind["tracking"]["gap_closed_median"]), 0.2, places=9)
            self.assertAlmostEqual(float(seed_by_kind["spnte"]["gap_closed_median"]), -0.4, places=9)

            summary = json.loads((root / "tables" / "headline.json").read_text())
            headline_by_kind = {row["metric_kind"]: row for row in summary["headline"]}
            self.assertEqual(set(headline_by_kind), {"tracking", "spnte"})
            self.assertAlmostEqual(headline_by_kind["tracking"]["value_median"], 0.2, places=9)
            self.assertAlmostEqual(headline_by_kind["spnte"]["value_median"], -0.4, places=9)
            # Both headline rows share the historical scope name ("metric" key
            # holds the scope, unchanged); metric_kind is the new dimension.
            self.assertEqual(headline_by_kind["tracking"]["metric"], "GapClosed_static")
            self.assertEqual(headline_by_kind["spnte"]["metric"], "GapClosed_static")
            # status_counts stays scoped to tracking only (backward compatible);
            # the new spnte_status_counts key carries the SPNTE-scored counts
            # separately, proving the two views were never merged.
            self.assertEqual(summary["status_counts"], {"eligible": 1})
            self.assertEqual(summary["spnte_status_counts"], {"eligible": 1})

    def test_report_renders_spnte_gap_closed_next_to_tracking_headline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._save_s1_cell(root, "MLP", 1, 1.0, 1.0)
            self._save_s1_cell(root, "Superset-Oracle", 1, 0.5, 0.5, speed=0.95)
            self._save_s1_cell(root, "SysID", 1, 0.9, 1.2, speed=0.95)
            cfg = self._cfg(root)
            self.assertEqual(cmd_aggregate(cfg), 0)
            self.assertEqual(cmd_report(cfg), 0)
            report = (root / "report" / "scorecard.md").read_text()
            self.assertIn("## Headline", report)
            self.assertIn("## SPNTE Gap-Closed (scored view, parallel to Headline)", report)
            self.assertIn("## SPNTE Hücre durumları (scored view)", report)
            headline_section = report.split("## SPNTE Gap-Closed")[0]
            spnte_section = report.split("## SPNTE Gap-Closed")[1]
            self.assertIn("SysID", headline_section)
            self.assertIn("SysID", spnte_section)

    def test_artifacts_without_spnte_lin_score_exactly_as_before(self):
        # Historical artifacts (no spnte_lin field at all) must keep producing
        # only the tracking scorecard row -- no phantom SPNTE row, and
        # spnte_status_counts stays empty. This is the byte-for-byte
        # backward-compatibility guarantee for the pre-existing scorecard.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._save_s1_cell(root, "MLP", 1, 1.0, 1.0, with_spnte=False)
            self._save_s1_cell(root, "Superset-Oracle", 1, 0.5, 0.5, speed=0.95, with_spnte=False)
            self._save_s1_cell(root, "SysID", 1, 0.9, 1.2, speed=0.95, with_spnte=False)
            self.assertEqual(cmd_aggregate(self._cfg(root)), 0)
            with (root / "tables" / "scorecard_cells.csv").open() as f:
                cells = list(csv.DictReader(f))
            self.assertEqual(len(cells), 1)
            self.assertEqual(cells[0]["metric_kind"], "tracking")
            self.assertAlmostEqual(float(cells[0]["headline_gap_closed"]), 0.2, places=9)
            summary = json.loads((root / "tables" / "headline.json").read_text())
            self.assertEqual(len(summary["headline"]), 1)
            self.assertEqual(summary["headline"][0]["metric_kind"], "tracking")
            self.assertEqual(summary["spnte_status_counts"], {})


if __name__ == "__main__":
    unittest.main()
