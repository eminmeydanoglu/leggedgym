"""Focused first-fall SPNTE regression tests."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from legged_gym.scripts.eval.metrics import MetricAccumulator
from legged_gym.scripts.eval.v3_eval import (
    _raw_rows,
    _rollout_switch,
    cmd_aggregate,
    cmd_report,
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


if __name__ == "__main__":
    unittest.main()
