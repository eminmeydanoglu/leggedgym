"""go2_moects resume must restore the curriculum, not restart it.

The wty curriculum is driven off ``common_step_counter // num_steps_per_iter``
(terrain levels being the one piece of state that is NOT a function of it).
``_init_buffers`` zeroes that counter unconditionally, and the runner used to
persist it only for ``cfg.env.ued_enabled`` runs -- go2_moects does not set
that flag. A resumed Slurm job therefore continued with 15k-iteration weights
on an iteration-0 curriculum: terrain levels back to round-robin, command
ranges back to the narrowest band, reward ramps back to start_value.

Covered here:
1. runner.save persists the clock and the env curriculum state for a non-UED env
2. runner.load restores both, and legacy checkpoints degrade gracefully
3. the mixin's terrain-level state dict round-trips and fails closed on mismatch
4. set_wty_total_iterations resyncs command ranges / reward ramps to progress
"""
from __future__ import annotations

import os
os.environ.setdefault("SIMULATOR", "genesis")

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from legged_gym.envs.go2.go2_moects.wty_curriculum_mixin import WtyCurriculumMixin
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


NUM_ENVS = 8
NUM_COLS = 4
NUM_ROWS = 6


def _make_simulator(levels=None, types=None):
    """Minimal stand-in for the terrain bookkeeping the mixin touches."""
    if types is None:
        types = torch.div(torch.arange(NUM_ENVS), NUM_ENVS / NUM_COLS,
                          rounding_mode="floor").to(torch.long)
    if levels is None:
        levels = torch.fmod(torch.arange(NUM_ENVS), 2)
    sim = SimpleNamespace(
        _terrain_levels=levels.clone(),
        _terrain_types=types.clone(),
        _terrain_origins=torch.arange(NUM_ROWS * NUM_COLS * 3, dtype=torch.float).reshape(
            NUM_ROWS, NUM_COLS, 3),
        _env_origins=torch.zeros(NUM_ENVS, 3),
    )
    # the mixin reads the public properties, writes the private buffers
    sim.terrain_levels = sim._terrain_levels
    sim.terrain_types = sim._terrain_types
    return sim


class _FakeMoectsEnv:
    """A mixin-only env double: no simulator, no physics, just the state the
    resume protocol touches."""

    curriculum_state_dict = WtyCurriculumMixin.curriculum_state_dict
    load_curriculum_state_dict = WtyCurriculumMixin.load_curriculum_state_dict
    WTY_CURRICULUM_STATE_VERSION = WtyCurriculumMixin.WTY_CURRICULUM_STATE_VERSION
    _wty_progress = WtyCurriculumMixin._wty_progress
    _get_current_scale = WtyCurriculumMixin._get_current_scale
    _update_reward_curriculum = WtyCurriculumMixin._update_reward_curriculum
    _apply_command_range_curriculum = WtyCurriculumMixin._apply_command_range_curriculum
    _wty_resync_curriculum = WtyCurriculumMixin._wty_resync_curriculum
    set_wty_total_iterations = WtyCurriculumMixin.set_wty_total_iterations

    def __init__(self, common_step_counter=0, levels=None, active=True):
        self.num_envs = NUM_ENVS
        self.num_steps_per_iter = 24
        self._wty_total_iterations = 30000
        self.common_step_counter = common_step_counter
        self.simulator = _make_simulator(levels=levels)
        self._wty_curriculum_active = active
        self.wty_terrain_ids = None
        self.command_ranges = {
            "lin_vel_x": [-0.5, 0.5], "lin_vel_y": [-0.5, 0.5],
            "ang_vel_yaw": [-1.0, 1.0], "heading": [-1.57, 1.57],
        }
        self.max_lin_vel = 0.5
        self.zero_command_proba = 0.0
        self.device = "cpu"
        self.reward_curriculum_configs = [
            {"reward_name": "dof_acc", "start_ratio": 0.0, "end_ratio": 0.2,
             "start_value": 1.0, "end_value": 0.0},
        ]
        self.reward_curriculum_scales = {"dof_acc": 1.0}
        self.cfg = SimpleNamespace(
            terrain=SimpleNamespace(num_cols=NUM_COLS),
            env=SimpleNamespace(ued_enabled=False),
            commands=SimpleNamespace(
                heading_command=False,
                zero_command_curriculum={
                    "start_ratio": 0.0, "end_ratio": 0.01,
                    "start_value": 0.0, "end_value": 0.1},
                command_range_curriculum=[
                    {"ratio": 20000 / 150000, "lin_vel_x": [-1.0, 1.0],
                     "lin_vel_y": [-1.0, 1.0], "ang_vel_yaw": [-1.5, 1.5],
                     "heading": [-1.57, 1.57]},
                    {"ratio": 50000 / 150000, "lin_vel_x": [-2.0, 2.0],
                     "lin_vel_y": [-1.0, 1.0], "ang_vel_yaw": [-2.0, 2.0],
                     "heading": [-1.57, 1.57]},
                ],
                terrain_max_command_ranges=[],
            ),
        )

    def _update_env_command_ranges(self):
        pass  # per-terrain clamping is exercised by the mixin's own tests


def _make_runner(env):
    obj = OnPolicyRunner.__new__(OnPolicyRunner)
    actor_critic = torch.nn.Module()
    actor_critic.actor = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=1e-3)
    obj.alg = mock.Mock(actor_critic=actor_critic, optimizer=optimizer)
    obj.device = torch.device("cpu")
    obj.current_learning_iteration = 0
    obj.best_eval_score = float("inf")
    obj.best_tracking_key = None
    obj.training_seed = 0
    obj._active_schedule_start = None
    obj._active_schedule_range = None
    obj._aux_optimizers = lambda: {}
    obj.log_dir = None
    obj.env = env
    return obj


class TestMoectsResumePersistence(unittest.TestCase):
    def test_save_persists_clock_and_curriculum_for_non_ued_env(self):
        env = _FakeMoectsEnv(common_step_counter=15_000 * 24)
        env.simulator._terrain_levels[:] = torch.tensor([3, 4, 2, 5, 1, 0, 3, 4])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            _make_runner(env).save(str(path), iteration=15_000)
            payload = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(payload["common_step_counter"], 15_000 * 24)
        state = payload["env_curriculum_state"]
        torch.testing.assert_close(
            state["terrain_levels"], torch.tensor([3, 4, 2, 5, 1, 0, 3, 4]))

    def test_resume_restores_clock_and_terrain_levels(self):
        source = _FakeMoectsEnv(common_step_counter=15_000 * 24)
        source.simulator._terrain_levels[:] = torch.tensor([3, 4, 2, 5, 1, 0, 3, 4])
        dest = _FakeMoectsEnv(common_step_counter=0)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            _make_runner(source).save(str(path), iteration=15_000)
            _make_runner(dest).load(str(path))

        self.assertEqual(dest.common_step_counter, 15_000 * 24)
        torch.testing.assert_close(
            dest.simulator._terrain_levels, torch.tensor([3, 4, 2, 5, 1, 0, 3, 4]))
        # env origins follow the restored levels, not the round-robin ones
        expected = source.simulator._terrain_origins[
            torch.tensor([3, 4, 2, 5, 1, 0, 3, 4]), dest.simulator.terrain_types]
        torch.testing.assert_close(dest.simulator._env_origins, expected)

    def test_legacy_checkpoint_without_state_loads_with_warning(self):
        env = _FakeMoectsEnv(common_step_counter=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            _make_runner(env).save(str(path), iteration=1)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            del payload["common_step_counter"]
            del payload["env_curriculum_state"]
            torch.save(payload, path)
            _make_runner(env).load(str(path))  # must not raise
        self.assertEqual(env.common_step_counter, 0)

    def test_inactive_curriculum_saves_nothing(self):
        env = _FakeMoectsEnv(common_step_counter=100, active=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            _make_runner(env).save(str(path), iteration=1)
            payload = torch.load(path, map_location="cpu", weights_only=False)
        self.assertNotIn("env_curriculum_state", payload)
        self.assertEqual(payload["common_step_counter"], 100)


class TestCurriculumStateDictGuards(unittest.TestCase):
    def test_geometry_mismatch_fails_closed(self):
        env = _FakeMoectsEnv()
        state = env.curriculum_state_dict()
        state["num_envs"] = NUM_ENVS + 1
        with self.assertRaisesRegex(ValueError, "geometry mismatch"):
            env.load_curriculum_state_dict(state)

    def test_version_mismatch_fails_closed(self):
        env = _FakeMoectsEnv()
        state = env.curriculum_state_dict()
        state["version"] = 999
        with self.assertRaisesRegex(ValueError, "version"):
            env.load_curriculum_state_dict(state)

    def test_terrain_type_mismatch_fails_closed(self):
        env = _FakeMoectsEnv()
        state = env.curriculum_state_dict()
        state["terrain_types"] = torch.flip(state["terrain_types"], dims=(0,))
        with self.assertRaisesRegex(ValueError, "terrain_types"):
            env.load_curriculum_state_dict(state)


class TestResyncAfterResume(unittest.TestCase):
    def test_resync_lands_on_the_latest_crossed_command_band(self):
        """Multi-stage jump: progress past BOTH thresholds must land on the
        widest band. The vendored backwards pop loop landed on the earliest."""
        env = _FakeMoectsEnv(common_step_counter=15_000 * 24)  # progress 0.5
        env.set_wty_total_iterations(30_000)
        self.assertEqual(env.command_ranges["lin_vel_x"], [-2.0, 2.0])
        self.assertEqual(env.command_ranges["ang_vel_yaw"], [-2.0, 2.0])
        self.assertEqual(env.max_lin_vel, 2.0)
        self.assertEqual(env.cfg.commands.command_range_curriculum, [])

    def test_resync_applies_single_crossed_band(self):
        # progress 0.2: past 20000/150000 (0.133), short of 50000/150000 (0.333)
        env = _FakeMoectsEnv(common_step_counter=6_000 * 24)
        env.set_wty_total_iterations(30_000)
        self.assertEqual(env.command_ranges["lin_vel_x"], [-1.0, 1.0])
        self.assertEqual(len(env.cfg.commands.command_range_curriculum), 1)

    def test_resync_rebuilds_reward_ramp_and_zero_command_proba(self):
        env = _FakeMoectsEnv(common_step_counter=3_000 * 24)  # progress 0.1
        env.set_wty_total_iterations(30_000)
        # dof_acc ramps 1.0 -> 0.0 over [0.0, 0.2] => halfway at progress 0.1
        self.assertAlmostEqual(env.reward_curriculum_scales["dof_acc"], 0.5)
        # zero-command curriculum saturates at ratio 0.01
        self.assertAlmostEqual(env.zero_command_proba, 0.1)

    def test_fresh_run_resync_is_a_noop(self):
        env = _FakeMoectsEnv(common_step_counter=0)
        env.set_wty_total_iterations(30_000)
        self.assertEqual(env.command_ranges["lin_vel_x"], [-0.5, 0.5])
        self.assertAlmostEqual(env.reward_curriculum_scales["dof_acc"], 1.0)
        self.assertAlmostEqual(env.zero_command_proba, 0.0)
        self.assertEqual(len(env.cfg.commands.command_range_curriculum), 2)


if __name__ == "__main__":
    unittest.main()
