"""BUG 2: UED resume must restore env.common_step_counter with the teacher.

The teacher's stage boundary is
  global_control_steps - stage_start_global_steps >= stage_length
where global_control_steps is env.common_step_counter. Persisting only
episode_curriculum.state_dict() (which holds stage_start_global_steps) leaves a
fresh env at counter=0 and either crashes advance() or silently re-opens stage 0.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from legged_gym.envs.go2.go2_v5_config import V5_STAGE_LENGTH_CONTROL_STEPS, build_ued_teacher
from legged_gym.envs.go2.go2_v5_config import Go2V5LPACRLCfg
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


def _make_curriculum():
    env_cfg = Go2V5LPACRLCfg()
    curriculum, _ = build_ued_teacher(env_cfg)
    return curriculum


def _make_ued_env(curriculum, common_step_counter: int = 0):
    return SimpleNamespace(
        cfg=SimpleNamespace(env=SimpleNamespace(ued_enabled=True)),
        episode_curriculum=curriculum,
        common_step_counter=common_step_counter,
    )


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


class TestUedResumeClock(unittest.TestCase):
    def test_save_persists_common_step_counter_with_teacher(self):
        curriculum = _make_curriculum()
        curriculum.stage_start_global_steps = 40_000
        curriculum.stage_index = 20
        curriculum.sampler_revision = 20
        env = _make_ued_env(curriculum, common_step_counter=40_500)
        runner = _make_runner(env)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            runner.save(str(path), iteration=100)
            payload = torch.load(path, map_location="cpu", weights_only=False)

        self.assertIn("episode_curriculum", payload)
        self.assertEqual(payload["common_step_counter"], 40_500)
        self.assertEqual(payload["episode_curriculum"]["stage_start_global_steps"], 40_000)

    def test_load_restores_counter_and_advance_does_not_crash(self):
        source_curriculum = _make_curriculum()
        source_curriculum.stage_start_global_steps = 40_000
        source_curriculum.stage_index = 20
        source_curriculum.sampler_revision = 20
        source_env = _make_ued_env(source_curriculum, common_step_counter=40_500)
        source = _make_runner(source_env)

        # Fresh process: counter starts at 0, teacher empty until load.
        dest_curriculum = _make_curriculum()
        dest_env = _make_ued_env(dest_curriculum, common_step_counter=0)
        dest = _make_runner(dest_env)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            source.save(str(path), iteration=100)
            dest.load(str(path))

        self.assertEqual(dest_env.common_step_counter, 40_500)
        self.assertEqual(dest_curriculum.stage_start_global_steps, 40_000)
        # Pre-fix BUG 2: advance(0) raised here.
        self.assertIsNone(dest_curriculum.advance(dest_env.common_step_counter))
        # Stage still open until 40000 + 2000.
        remaining = V5_STAGE_LENGTH_CONTROL_STEPS - (
            dest_env.common_step_counter - dest_curriculum.stage_start_global_steps
        )
        self.assertEqual(remaining, 1_500)
        self.assertIsNone(
            dest_curriculum.advance(dest_env.common_step_counter + remaining - 1)
        )
        snap = dest_curriculum.advance(dest_env.common_step_counter + remaining)
        self.assertIsNotNone(snap)

    def test_load_without_common_step_counter_fails_closed(self):
        curriculum = _make_curriculum()
        env = _make_ued_env(curriculum, common_step_counter=0)
        runner = _make_runner(env)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            runner.save(str(path), iteration=1)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            del payload["common_step_counter"]
            torch.save(payload, path)

            with self.assertRaisesRegex(ValueError, "common_step_counter"):
                runner.load(str(path))

        # Env clock must not have been partially mutated before the raise.
        self.assertEqual(env.common_step_counter, 0)

    def test_load_without_episode_curriculum_still_fails_closed(self):
        curriculum = _make_curriculum()
        env = _make_ued_env(curriculum, common_step_counter=0)
        runner = _make_runner(env)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            runner.save(str(path), iteration=1)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            del payload["episode_curriculum"]
            torch.save(payload, path)

            with self.assertRaisesRegex(ValueError, "episode_curriculum"):
                runner.load(str(path))

    def test_load_rejects_non_integer_counter(self):
        curriculum = _make_curriculum()
        env = _make_ued_env(curriculum, common_step_counter=10)
        runner = _make_runner(env)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            runner.save(str(path), iteration=1)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload["common_step_counter"] = 1.5
            torch.save(payload, path)

            with self.assertRaisesRegex(ValueError, "non-negative integer"):
                runner.load(str(path))

            payload["common_step_counter"] = -1
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "non-negative integer"):
                runner.load(str(path))

    def test_non_ued_save_omits_clock_keys(self):
        env = SimpleNamespace(
            cfg=SimpleNamespace(env=SimpleNamespace(ued_enabled=False)),
            episode_curriculum=None,
            common_step_counter=999,
        )
        runner = _make_runner(env)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            runner.save(str(path), iteration=1)
            payload = torch.load(path, map_location="cpu", weights_only=False)
        self.assertNotIn("episode_curriculum", payload)
        self.assertNotIn("common_step_counter", payload)

    def test_mid_stage0_resume_keeps_remaining_stage_length(self):
        """Silent BUG 2 case: partial stage sums + restored clock, not a full re-count."""
        stage_len = V5_STAGE_LENGTH_CONTROL_STEPS
        mid = 1_500
        source_curriculum = _make_curriculum()
        source_curriculum._stage_return_sums[:] = 3.0
        source_curriculum._stage_episode_counts[:] = 2
        source_curriculum.stage_start_global_steps = 0
        source = _make_runner(_make_ued_env(source_curriculum, common_step_counter=mid))

        dest_curriculum = _make_curriculum()
        dest_env = _make_ued_env(dest_curriculum, common_step_counter=0)
        dest = _make_runner(dest_env)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            source.save(str(path), iteration=50)
            dest.load(str(path))

        self.assertEqual(dest_env.common_step_counter, mid)
        np.testing.assert_array_equal(dest_curriculum._stage_episode_counts, 2)
        # Without counter restore, advance would only fire at 2000 from 0 and
        # would have re-accumulated on top of the restored partial sums.
        self.assertIsNone(dest_curriculum.advance(mid + (stage_len - mid) - 1))
        snap = dest_curriculum.advance(mid + (stage_len - mid))
        self.assertIsNotNone(snap)
        self.assertEqual(snap.global_control_steps, stage_len)


if __name__ == "__main__":
    unittest.main()
