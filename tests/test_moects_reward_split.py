"""CPU-only tests for the MoE-CTS per-role reward/episode-length telemetry.

Reference: go2_rl_gym on_policy_runner_cts.py:138-174 (rollout bookkeeping)
and 245-254 (TensorBoard writes). Covers, without a simulator/env or GPU:

1. Log mapping: known teacher/student reward/length buffers produce the exact
   reference TensorBoard names (iteration step + /time variants) with the
   correct means, and the aggregate Train/mean_reward channels still fire.
2. Rollout bookkeeping: a scripted fake env + fake alg drive
   MoECTSRunner.learn for one iteration; done envs with known reward sums and
   the interleaved role idxs (compute_role_env_idxs, NOT contiguous blocks)
   must land in the right per-role buffers.

Run: .venv/bin/python -m unittest tests.test_moects_reward_split -v
"""

import contextlib
import io
import os
import unittest
from collections import deque
from unittest import mock

# legged_gym's package __init__ gates on this; the runner import chain only
# needs the genesis *import* (works on CPU), never builds a simulator.
os.environ.setdefault("SIMULATOR", "genesis")

import torch  # noqa: E402

from rsl_rl.algorithms.ppo_moe_cts import compute_role_env_idxs  # noqa: E402
from rsl_rl.runners import MoECTSRunner  # noqa: E402


def _make_log_runner():
    """MoECTSRunner shell with only the attributes the log path touches."""
    runner = MoECTSRunner.__new__(MoECTSRunner)
    runner.writer = mock.MagicMock()
    runner.device = torch.device("cpu")
    runner.tot_timesteps = 0
    runner.tot_time = 0.0
    runner.num_steps_per_env = 24
    runner.current_learning_iteration = 0
    runner.env = mock.MagicMock()
    runner.env.num_envs = 8
    runner.alg = mock.MagicMock()
    runner.alg.actor_critic.std = torch.ones(12)
    runner.alg.learning_rate = 1e-3
    runner.alg.encoder_lr = 1e-3
    return runner


def _base_locs(**overrides):
    locs = dict(
        it=0,
        num_learning_iterations=1,
        collection_time=0.5,
        learn_time=0.25,
        ep_infos=[],
        rewbuffer=deque(maxlen=100),
        lenbuffer=deque(maxlen=100),
        mean_value_loss=1.0,
        mean_teacher_surrogate_loss=2.0,
        mean_student_surrogate_loss=3.0,
        mean_reconstruction_loss=4.0,
        moe_stats={},
    )
    locs.update(overrides)
    return locs


def _scalars(writer):
    """name -> (value, step) of every add_scalar call; last write wins."""
    out = {}
    for call in writer.add_scalar.call_args_list:
        out[call.args[0]] = (call.args[1], call.args[2])
    return out


class RoleSplitLogMappingTest(unittest.TestCase):
    def test_per_role_tensorboard_names_and_values(self):
        runner = _make_log_runner()
        locs = _base_locs(
            it=3,
            rewbuffer=deque([4.0, 12.0, 2.0, 10.0], maxlen=100),
            lenbuffer=deque([2.0, 2.0, 2.0, 2.0], maxlen=100),
            teacher_rewbuffer=deque([4.0, 12.0], maxlen=100),
            teacher_lenbuffer=deque([20.0, 40.0], maxlen=100),
            student_rewbuffer=deque([2.0, 10.0], maxlen=100),
            student_lenbuffer=deque([10.0, 30.0], maxlen=100),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            runner.log(locs)
        scalars = _scalars(runner.writer)

        # Reference set (on_policy_runner_cts.py:245-254): iteration + /time.
        self.assertEqual(scalars["Train/mean_teacher_reward"], (8.0, 3))
        self.assertEqual(scalars["Train/mean_student_reward"], (6.0, 3))
        self.assertEqual(scalars["Train/mean_teacher_episode_length"], (30.0, 3))
        self.assertEqual(scalars["Train/mean_student_episode_length"], (20.0, 3))
        tot_time = runner.tot_time
        self.assertEqual(scalars["Train/mean_teacher_reward/time"], (8.0, tot_time))
        self.assertEqual(scalars["Train/mean_student_reward/time"], (6.0, tot_time))
        self.assertEqual(scalars["Train/mean_teacher_episode_length/time"], (30.0, tot_time))
        self.assertEqual(scalars["Train/mean_student_episode_length/time"], (20.0, tot_time))

        # Aggregate channels (CTSRunner.log) keep working unchanged.
        self.assertEqual(scalars["Train/mean_reward"], (7.0, 3))
        self.assertEqual(scalars["Train/mean_episode_length"], (2.0, 3))

    def test_per_role_terminal_output(self):
        runner = _make_log_runner()
        locs = _base_locs(
            rewbuffer=deque([1.0], maxlen=100),
            lenbuffer=deque([1.0], maxlen=100),
            teacher_rewbuffer=deque([1.0], maxlen=100),
            teacher_lenbuffer=deque([2.0], maxlen=100),
            student_rewbuffer=deque([3.0], maxlen=100),
            student_lenbuffer=deque([4.0], maxlen=100),
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            runner.log(locs)
        out = buf.getvalue()
        self.assertIn("Mean teacher reward:", out)
        self.assertIn("Mean teacher episode length:", out)
        self.assertIn("Mean student reward:", out)
        self.assertIn("Mean student episode length:", out)

    def test_missing_role_buffers_skip_channels(self):
        # Hand-built locs without the split buffers (e.g. the pre-existing
        # runner-contract tests) must neither fail nor write the new channels.
        runner = _make_log_runner()
        with contextlib.redirect_stdout(io.StringIO()):
            runner.log(_base_locs())
        scalars = _scalars(runner.writer)
        self.assertNotIn("Train/mean_teacher_reward", scalars)
        self.assertNotIn("Train/mean_student_reward", scalars)
        self.assertNotIn("Train/mean_teacher_episode_length", scalars)
        self.assertNotIn("Train/mean_student_episode_length", scalars)


class _ScriptedEnv:
    """Minimal VecEnv stand-in: fixed obs, scripted rewards/dones per step."""

    def __init__(self, num_envs, rewards_script, dones_script):
        self.num_envs = num_envs
        self._rewards_script = list(rewards_script)
        self._dones_script = list(dones_script)
        self._obs = torch.zeros(num_envs, 1)

    def get_observations(self):
        return (self._obs.clone(), self._obs.clone(),
                self._obs.clone(), self._obs.clone())

    def step(self, actions):
        rewards = self._rewards_script.pop(0)
        dones = self._dones_script.pop(0)
        return (self._obs.clone(), self._obs.clone(), self._obs.clone(),
                self._obs.clone(), rewards, dones, {})


class _FakeAlg:
    """Algorithm stand-in carrying the real interleaved role idxs."""

    def __init__(self, num_envs, teacher_env_ratio):
        self.teacher_env_idxs, self.student_env_idxs = compute_role_env_idxs(
            num_envs, teacher_env_ratio, "cpu")
        self.actor_critic = mock.MagicMock()
        self.actor_critic.std = torch.ones(12)
        self.learning_rate = 1e-3
        self.encoder_lr = 1e-3

    def act(self, obs, privileged_obs, obs_history, critic_obs):
        return torch.zeros(obs.shape[0], 1)

    def process_env_step(self, rewards, dones, infos):
        pass

    def compute_returns(self, critic_obs, obs_history):
        pass

    def update(self):
        return 0.0, 0.0, 0.0, 0.0, {}


class RoleSplitRolloutTest(unittest.TestCase):
    def test_done_episodes_split_by_interleaved_role_idxs(self):
        num_envs = 8
        # Reference interleave for ratio 0.75 (moe_cts.py:96-102): students
        # {0, 4}, teachers the rest -- deliberately NOT contiguous blocks.
        alg = _FakeAlg(num_envs, teacher_env_ratio=0.75)
        self.assertEqual(alg.student_env_idxs.tolist(), [0, 4])
        self.assertEqual(alg.teacher_env_idxs.tolist(), [1, 2, 3, 5, 6, 7])

        rewards = torch.arange(1, num_envs + 1, dtype=torch.float)
        dones_0 = torch.zeros(num_envs)
        dones_1 = torch.zeros(num_envs)
        # Two student (0, 4) and two teacher (1, 5) episodes end on step 2.
        dones_1[[0, 1, 4, 5]] = 1.0
        env = _ScriptedEnv(num_envs,
                           rewards_script=[rewards, rewards],
                           dones_script=[dones_0, dones_1])

        runner = MoECTSRunner.__new__(MoECTSRunner)
        runner.writer = mock.MagicMock()
        runner.device = torch.device("cpu")
        runner.tot_timesteps = 0
        runner.tot_time = 0.0
        runner.num_steps_per_env = 2
        runner.current_learning_iteration = 0
        runner.save_interval = 500
        # Non-None log_dir enables the bookkeeping; writer is pre-set so
        # _pre_learn creates nothing, and save is mocked out entirely.
        runner.log_dir = "/tmp/moects_reward_split_test"
        runner.env = env
        runner.alg = alg
        runner.save = mock.MagicMock()

        with contextlib.redirect_stdout(io.StringIO()):
            runner.learn(num_learning_iterations=1)

        scalars = _scalars(runner.writer)
        # Every env received arange(1..8) twice, so done sums are 2*(i+1):
        # teachers {1, 5} -> 4.0, 12.0 (mean 8.0);
        # students {0, 4} -> 2.0, 10.0 (mean 6.0); all lengths are 2 steps.
        self.assertEqual(scalars["Train/mean_teacher_reward"][0], 8.0)
        self.assertEqual(scalars["Train/mean_student_reward"][0], 6.0)
        self.assertEqual(scalars["Train/mean_reward"][0], 7.0)
        self.assertEqual(scalars["Train/mean_teacher_episode_length"][0], 2.0)
        self.assertEqual(scalars["Train/mean_student_episode_length"][0], 2.0)
        self.assertIn("Train/mean_teacher_reward/time", scalars)
        self.assertIn("Train/mean_student_reward/time", scalars)
        self.assertIn("Train/mean_teacher_episode_length/time", scalars)
        self.assertIn("Train/mean_student_episode_length/time", scalars)


if __name__ == "__main__":
    unittest.main()
