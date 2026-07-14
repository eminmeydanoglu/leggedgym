"""Unit tests for the HIM benchmark method (go2_bench_him).

CPU-only: they exercise the estimator/actor-critic shapes and math plus the
config/registry wiring WITHOUT building a simulator env (no GPU / genesis
runtime). Runtime smoke tests (training / eval) live elsewhere.

Run:  .venv/bin/python -m unittest tests/test_bench_him.py -v
(or:  .venv/bin/python -m pytest tests/test_bench_him.py -q)
"""

import os
os.environ.setdefault("SIMULATOR", "genesis")

import unittest

import torch

from rsl_rl.modules.him_estimator import HIMEstimator, sinkhorn
from rsl_rl.modules.him_actor_critic import HIMActorCritic


TEMPORAL_STEPS = 6
NUM_ONE_STEP = 45
NUM_ACTOR_OBS = TEMPORAL_STEPS * NUM_ONE_STEP   # 270
NUM_SINGLE_CRITIC_OBS = 3 + NUM_ONE_STEP + 5    # 53
CRITIC_FRAME_STACK = 5
NUM_CRITIC_OBS = CRITIC_FRAME_STACK * NUM_SINGLE_CRITIC_OBS  # 265
NUM_ACTIONS = 12
NUM_LATENT = 16


class TestHIMEstimator(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.est = HIMEstimator(temporal_steps=TEMPORAL_STEPS, num_one_step_obs=NUM_ONE_STEP)

    def test_num_latent(self):
        self.assertEqual(self.est.num_latent, NUM_LATENT)

    def test_encode_shapes_and_unit_norm(self):
        B = 8
        obs_history = torch.randn(B, NUM_ACTOR_OBS)
        vel, z = self.est.encode(obs_history)
        self.assertEqual(tuple(vel.shape), (B, 3))
        self.assertEqual(tuple(z.shape), (B, NUM_LATENT))
        norms = z.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_get_latent_detached(self):
        obs_history = torch.randn(4, NUM_ACTOR_OBS)
        vel, z = self.est.get_latent(obs_history)
        self.assertFalse(vel.requires_grad)
        self.assertFalse(z.requires_grad)

    def test_sinkhorn_row_stochastic(self):
        B, K = 16, 32
        scores = torch.randn(B, K)
        Q = sinkhorn(scores)
        self.assertEqual(tuple(Q.shape), (B, K))
        row_sums = Q.sum(dim=1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))
        # non-negative assignment
        self.assertTrue((Q >= 0).all())

    def test_update_finite_and_decreases_estimation(self):
        B = 64
        torch.manual_seed(1)
        obs_history = torch.randn(B, NUM_ACTOR_OBS)
        # fixed critic obs -> constant velocity target
        next_critic_obs = torch.randn(B, NUM_CRITIC_OBS)
        terminated = torch.ones(B, 1)

        first_est, first_swap = self.est.update(obs_history, next_critic_obs, terminated)
        self.assertTrue(torch.isfinite(torch.tensor(first_est)))
        self.assertTrue(torch.isfinite(torch.tensor(first_swap)))

        last_est = first_est
        for _ in range(50):
            last_est, _ = self.est.update(obs_history, next_critic_obs, terminated)
        # the explicit head should fit the fixed velocity target better over time
        self.assertLess(last_est, first_est)


class TestHIMActorCritic(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.ac = HIMActorCritic(
            NUM_ACTOR_OBS, NUM_CRITIC_OBS, NUM_ONE_STEP, NUM_ACTIONS,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            enc_hidden_dims=[128, 64, NUM_LATENT],
            tar_hidden_dims=[128, 64],
        )

    def test_actor_input_dim(self):
        # actor input = current one-step frame(45) + vel(3) + latent(16) = 64
        self.assertEqual(self.ac.actor[0].in_features, NUM_ONE_STEP + 3 + NUM_LATENT)

    def test_critic_input_dim(self):
        self.assertEqual(self.ac.critic[0].in_features, NUM_CRITIC_OBS)

    def test_act_and_inference_shapes(self):
        B = 10
        obs_history = torch.randn(B, NUM_ACTOR_OBS)
        actions = self.ac.act(obs_history)
        self.assertEqual(tuple(actions.shape), (B, NUM_ACTIONS))
        mean = self.ac.act_inference(obs_history)
        self.assertEqual(tuple(mean.shape), (B, NUM_ACTIONS))

    def test_evaluate_shape(self):
        B = 10
        critic_obs = torch.randn(B, NUM_CRITIC_OBS)
        value = self.ac.evaluate(critic_obs)
        self.assertEqual(tuple(value.shape), (B, 1))


class TestHIMRegistryAndConfig(unittest.TestCase):
    def test_registered_and_dims_consistent(self):
        import legged_gym.envs  # noqa: F401  (side effect: registers tasks)
        from legged_gym.utils import task_registry

        self.assertIn("go2_bench_him", task_registry.task_classes)

        env_cfg, train_cfg = task_registry.get_cfgs("go2_bench_him")
        env = env_cfg.env
        self.assertEqual(env.num_observations, env.frame_stack * env.num_one_step_obs)
        self.assertEqual(env.c_frame_stack, 5)
        self.assertEqual(env.num_single_critic_obs, 3 + env.num_one_step_obs + 5)
        self.assertEqual(env.num_privileged_obs,
                         env.c_frame_stack * env.num_single_critic_obs)
        self.assertEqual(train_cfg.runner.critic_contract, "stacked_5x53_265d")
        self.assertEqual(train_cfg.runner.policy_class_name, "HIMActorCritic")
        self.assertEqual(train_cfg.runner.algorithm_class_name, "PPO_HIM")
        self.assertEqual(train_cfg.runner_class_name, "HIMRunner")
        # HIM keeps the fair command_schedule + eval machinery
        self.assertTrue(hasattr(train_cfg.runner, "command_schedule"))
        self.assertEqual(train_cfg.runner.eval_interval, 200)


if __name__ == "__main__":
    unittest.main()
