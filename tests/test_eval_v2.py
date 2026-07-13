"""CPU-only contract tests for Eval V2's non-simulator semantics."""

import unittest

import torch

from legged_gym.scripts.eval.campaign import _terrain_raw, delay_ms, selection_order
from legged_gym.scripts.eval.indist import tracking_score, tracking_selection_key
from legged_gym.scripts.eval.metrics import MetricAccumulator


class TestCheckpointSelection(unittest.TestCase):
    def test_safe_checkpoint_beats_unsafe(self):
        chosen = selection_order([
            {"iteration": 100, "fall_rate": 0.01, "tracking_score": 0.9},
            {"iteration": 200, "fall_rate": 0.10, "tracking_score": 0.01},
        ])
        self.assertEqual(chosen["iteration"], 100)

    def test_tracking_then_early_iteration_breaks_safe_ties(self):
        chosen = selection_order([
            {"iteration": 200, "fall_rate": 0.01, "tracking_score": 0.4},
            {"iteration": 100, "fall_rate": 0.02, "tracking_score": 0.4},
            {"iteration": 300, "fall_rate": 0.01, "tracking_score": 0.5},
        ])
        self.assertEqual(chosen["iteration"], 100)

    def test_all_unsafe_uses_fall_then_tracking(self):
        chosen = selection_order([
            {"iteration": 100, "fall_rate": 0.3, "tracking_score": 0.1},
            {"iteration": 200, "fall_rate": 0.2, "tracking_score": 0.9},
            {"iteration": 300, "fall_rate": 0.2, "tracking_score": 0.8},
        ])
        self.assertEqual(chosen["iteration"], 300)

    def test_training_selection_key_has_same_lexicographic_rule(self):
        safe = {"fall_rate": 0.01, "tracking_lin_err": 1.0, "tracking_ang_err": 1.0}
        unsafe = {"fall_rate": 0.20, "tracking_lin_err": 0.0, "tracking_ang_err": 0.0}
        self.assertLess(tracking_selection_key(safe, 100, .05),
                        tracking_selection_key(unsafe, 200, .05))
        self.assertAlmostEqual(tracking_score(safe), .5 * (1.0 / 2 ** .5 + 1.0))

    def test_training_selection_key_breaks_ties_by_iteration(self):
        m = {"fall_rate": 0.01, "tracking_lin_err": 0.2, "tracking_ang_err": 0.3}
        self.assertLess(tracking_selection_key(m, 100, .05), tracking_selection_key(m, 200, .05))


class TestV2MetricContract(unittest.TestCase):
    def test_window_fall_and_censored_return(self):
        acc = MetricAccumulator(2, "cpu")
        # env 0 falls once then is auto-reset; env 1 never terminates.  All four
        # rewards must count in return_per_step, regardless of completed episodes.
        for t in range(2):
            acc.update(torch.tensor([2.0, 4.0]), torch.tensor([t == 0, False]),
                       torch.tensor([False, False]), torch.zeros(2), torch.zeros(2))
        got = acc.compute()
        self.assertTrue(torch.equal(got["ever_fell"], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.equal(got["return_per_step"], torch.tensor([2.0, 4.0])))

    def test_delay_step_to_ms(self):
        self.assertEqual(delay_ms(0), 0.0)
        self.assertEqual(delay_ms(6), 120.0)

    def test_deterministic_terrain_map_and_scale(self):
        raw_a = _terrain_raw(0.05, 41001, 0.2)
        raw_b = _terrain_raw(0.05, 41001, 0.2)
        self.assertEqual(raw_a.shape, (60, 60))
        self.assertTrue((raw_a == raw_b).all())
        self.assertEqual(int(abs(raw_a).max()), 20)  # 5cm / 2.5mm


if __name__ == "__main__":
    unittest.main()
