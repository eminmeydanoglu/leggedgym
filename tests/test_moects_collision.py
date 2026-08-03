"""Collision-reward threshold tests for go2_moects / go2_moects_him.

CPU-only (no simulator env is built): they pin the measurement-driven
``cfg.rewards.collision_force_threshold`` (0.1 N, reference parity -- see
tests/_measure_collision_force_noise.py + tmp/collision_force_stats.json),
verify the WtyCurriculumMixin override is the one both task classes resolve
through their MRO, and exercise the reward tensor logic on fabricated force
norms with a mock env.

Run:  .venv/bin/python -m unittest tests.test_moects_collision -v
(or:  .venv/bin/python -m pytest tests/test_moects_collision.py -q)
"""

import os
os.environ.setdefault("SIMULATOR", "genesis")

import unittest
from types import SimpleNamespace

import torch

import legged_gym.envs  # noqa: F401  (registers go2_moects / go2_moects_him)
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.go2.go2_moects.go2_moects import Go2MoECTS
from legged_gym.envs.go2.go2_moects.go2_moects_him import Go2MoECTSHIM
from legged_gym.envs.go2.go2_moects.wty_curriculum_mixin import WtyCurriculumMixin
from legged_gym.utils import task_registry

TASKS = ("go2_moects", "go2_moects_him")

# Measurement-driven value (Genesis noise floor on non-contacting links is
# exactly 0.0 N -> reference parity). Host LeggedRobot hardcodes 10.0.
EXPECTED_THRESHOLD = 0.1


def _fake_env(force_norms, threshold=EXPECTED_THRESHOLD):
    """Mock env carrying only what the mixin's _reward_collision reads."""
    return SimpleNamespace(
        penalized_bodies_force_norm=torch.as_tensor(force_norms, dtype=torch.float),
        cfg=SimpleNamespace(
            rewards=SimpleNamespace(collision_force_threshold=threshold)))


class TestCollisionThresholdConfig(unittest.TestCase):

    def test_threshold_field_present_and_pinned(self):
        for task in TASKS:
            with self.subTest(task=task):
                env_cfg, _ = task_registry.get_cfgs(task)
                self.assertTrue(
                    hasattr(env_cfg.rewards, "collision_force_threshold"),
                    f"{task}: cfg.rewards.collision_force_threshold missing")
                self.assertAlmostEqual(
                    env_cfg.rewards.collision_force_threshold,
                    EXPECTED_THRESHOLD,
                    msg=f"{task}: threshold drifted from the "
                        f"measurement-justified {EXPECTED_THRESHOLD} N")
                # guard against silently falling back to the host default
                self.assertNotEqual(
                    env_cfg.rewards.collision_force_threshold, 10.0)

    def test_collision_scale_still_enabled(self):
        for task in TASKS:
            with self.subTest(task=task):
                env_cfg, _ = task_registry.get_cfgs(task)
                self.assertEqual(env_cfg.rewards.scales.collision, -1.0)


class TestCollisionRewardMRO(unittest.TestCase):

    def test_mixin_override_wins_for_both_arms(self):
        for cls in (Go2MoECTS, Go2MoECTSHIM):
            with self.subTest(cls=cls.__name__):
                self.assertIs(cls._reward_collision,
                              WtyCurriculumMixin._reward_collision)

    def test_mixin_override_is_not_host_default(self):
        self.assertIsNot(WtyCurriculumMixin._reward_collision,
                         LeggedRobot._reward_collision)


class TestCollisionRewardValues(unittest.TestCase):
    """Tensor-level behavior of WtyCurriculumMixin._reward_collision."""

    def _reward(self, force_norms, threshold=EXPECTED_THRESHOLD):
        return WtyCurriculumMixin._reward_collision(
            _fake_env(force_norms, threshold))

    def test_zero_forces_give_zero_penalty(self):
        norms = torch.zeros(4, 8)
        self.assertTrue(torch.equal(self._reward(norms), torch.zeros(4)))

    def test_below_threshold_gives_zero_penalty(self):
        norms = torch.full((3, 8), 0.5 * EXPECTED_THRESHOLD)
        self.assertTrue(torch.equal(self._reward(norms), torch.zeros(3)))

    def test_exactly_at_threshold_gives_zero_penalty(self):
        # strict '>' (reference formula): norm == threshold must NOT fire
        norms = torch.full((2, 8), EXPECTED_THRESHOLD)
        self.assertTrue(torch.equal(self._reward(norms), torch.zeros(2)))

    def test_above_threshold_counts_links_per_env(self):
        norms = torch.zeros(3, 8)
        norms[0, 6] = EXPECTED_THRESHOLD + 1e-3          # one link just above
        norms[1, [6, 7]] = 5.0                           # both rear calves
        norms[2, :] = 50.0                               # all 8 links
        rew = self._reward(norms)
        self.assertTrue(torch.equal(rew, torch.tensor([1., 2., 8.])))

    def test_threshold_actually_comes_from_config(self):
        # same norms, different configured thresholds -> different counts
        norms = torch.tensor([[0.05, 0.5, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertTrue(torch.equal(
            self._reward(norms, threshold=0.1), torch.tensor([2.])))
        self.assertTrue(torch.equal(
            self._reward(norms, threshold=1.0), torch.tensor([1.])))
        self.assertTrue(torch.equal(
            self._reward(norms, threshold=10.0), torch.tensor([0.])))

    def test_output_shape_and_dtype(self):
        rew = self._reward(torch.rand(17, 8) * 100.0)
        self.assertEqual(tuple(rew.shape), (17,))
        self.assertEqual(rew.dtype, torch.float)


if __name__ == "__main__":
    unittest.main()
