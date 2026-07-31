"""Opt-in Genesis lifecycle integration test for the ``go2_moects`` task.

Builds the REAL ``go2_moects`` environment (Go2MoECTS = WtyCurriculumMixin +
LeggedRobotCTS) under the Genesis simulator with a small ``num_envs`` and a
test-only short ``episode_length_s`` override, then drives it with zero /
small-random actions through several full episodes.  No PPO, no benchmark --
this validates the environment lifecycle only:

1. moe_grid terrain metadata is generated and the WtyCurriculumMixin game
   curriculum is active on it.
2. Terrain levels / origins stay consistent after resets (shape, range,
   env -> origin mapping).
3. At least one completed / reset episode is observed.
4. Command-resampling state is valid (commands buffer shape / range).
5. Episode extras carry the terrain-curriculum metrics.
6. obs / rewards stay NaN- and inf-free.

GATING: the whole module is skipped unless BOTH env vars are set:
    MOECTS_GENESIS_INTEGRATION=1   (opt-in; keeps the default CPU suite light)
    SIMULATOR=genesis              (the test is Genesis-only; needs a GPU)

Skip-path / collect check (no GPU work, must report "skipped"):
    .venv/bin/python -m unittest tests.test_moects_genesis_lifecycle -v

Real run (UHeM A100 compute node, post-integration):
    SIMULATOR=genesis MOECTS_GENESIS_INTEGRATION=1 \
        python -m unittest tests.test_moects_genesis_lifecycle -v
"""

import os
import unittest
from types import SimpleNamespace

import torch

_INTEGRATION = os.environ.get("MOECTS_GENESIS_INTEGRATION") == "1"
_GENESIS = os.environ.get("SIMULATOR") == "genesis"
_SKIP_REASON = (
    "Genesis lifecycle test is opt-in: set MOECTS_GENESIS_INTEGRATION=1 and "
    "SIMULATOR=genesis (requires a GPU node; see module docstring)")

# Expected moe_grid metadata for the go2_moects cfg (10 levels x 20 type
# columns; mirrors tests/test_moects_contract.py::TestMoEGRidTerrain).
_EXPECTED_COLS2ID = [0, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 8, 8, 8]
_EXPECTED_TERRAIN_NAMES = {"wave", "slope", "rough_slope", "stairs_up",
                           "stairs_down", "obstacles", "flat"}


@unittest.skipUnless(_INTEGRATION and _GENESIS, _SKIP_REASON)
class TestMoECTSGenesisLifecycle(unittest.TestCase):
    NUM_ENVS = 16
    EPISODE_LENGTH_S = 0.5   # test-only override -> 25 control steps @ dt 0.02
    ROLLOUT_EPISODES = 3     # rollout spans 3 episodes: every env resets >= 2x

    @classmethod
    def setUpClass(cls):
        import genesis as gs
        import legged_gym.envs  # noqa: F401  (registers go2_moects)
        from legged_gym.utils import task_registry

        gs.init(backend=gs.gpu, logging_level="warning")
        cfg, _ = task_registry.get_cfgs("go2_moects")
        cfg.env.num_envs = cls.NUM_ENVS
        cfg.env.episode_length_s = cls.EPISODE_LENGTH_S
        args = SimpleNamespace(
            task="go2_moects", seed=7, debug=False, headless=True, cpu=False,
            num_envs=cls.NUM_ENVS, max_iterations=None, resume=False,
            sync_wandb=False, ckpt=None, load_run=None, export_onnx=False,
            motion_file=None, num_student=None)
        cls.env, _ = task_registry.make_env("go2_moects", args=args, env_cfg=cfg)

        sim = cls.env.simulator
        # Pre-rollout curriculum assignment snapshot (set by the mixin's
        # _wty_setup_terrain_curriculum during env construction).
        cls.types_at_build = sim.terrain_types.clone()
        cls.levels_at_build = sim.terrain_levels.clone()

        # Rollout: zero actions, small random actions every 13th step.
        cls.total_resets = 0
        cls.last_extras = {}
        cls.nonfinite_events = []  # (step, buffer_name)
        cls.env.reset()
        n_steps = int(cls.ROLLOUT_EPISODES * cls.env.max_episode_length)
        for step in range(n_steps):
            if step % 13 == 0:
                actions = torch.rand(
                    cls.NUM_ENVS, cls.env.num_actions,
                    device=cls.env.device) * 0.4 - 0.2
            else:
                actions = torch.zeros(
                    cls.NUM_ENVS, cls.env.num_actions, device=cls.env.device)
            obs, priv, hist, critic, rew, resets, extras = cls.env.step(actions)
            for name, buf in (("obs", obs), ("privileged_obs", priv),
                              ("obs_history", hist), ("critic_obs", critic),
                              ("rew", rew)):
                if buf is not None and not bool(torch.isfinite(buf).all()):
                    cls.nonfinite_events.append((step, name))
            cls.total_resets += int(resets.sum().item())
            if extras.get("episode"):
                cls.last_extras = extras

    @classmethod
    def tearDownClass(cls):
        env = getattr(cls, "env", None)
        if env is not None and hasattr(env, "destroy"):
            env.destroy()

    # ------------------------------------------------------------------
    # 1. moe_grid metadata + mixin curriculum active
    # ------------------------------------------------------------------

    def test_moe_grid_metadata_and_curriculum_active(self):
        env, sim = self.env, self.env.simulator
        self.assertTrue(env.cfg.terrain.moe_grid)
        # the mixin owns the game curriculum; the host flag stays off
        self.assertFalse(env.cfg.terrain.curriculum)
        terrain = sim._terrain
        self.assertEqual(list(terrain.cols2id), _EXPECTED_COLS2ID)
        self.assertEqual(set(terrain.name2cols), _EXPECTED_TERRAIN_NAMES)
        self.assertEqual(tuple(sim._terrain_origins.shape[:2]),
                         (env.cfg.terrain.num_rows, env.cfg.terrain.num_cols))
        self.assertTrue(env._wty_curriculum_active)
        self.assertIsNotNone(env.wty_terrain_ids)
        self.assertEqual(tuple(env.wty_terrain_ids.shape), (self.NUM_ENVS,))
        semantic_ids = set(int(v) for v in env.wty_terrain_ids.unique().tolist())
        self.assertTrue(semantic_ids.issubset(set(_EXPECTED_COLS2ID)))

    # ------------------------------------------------------------------
    # 2. terrain levels / origins consistency after resets
    # ------------------------------------------------------------------

    def test_terrain_levels_origins_consistent(self):
        env, sim = self.env, self.env.simulator
        num_rows, num_cols = env.cfg.terrain.num_rows, env.cfg.terrain.num_cols
        levels, types = sim.terrain_levels, sim.terrain_types
        self.assertEqual(tuple(levels.shape), (self.NUM_ENVS,))
        self.assertEqual(tuple(types.shape), (self.NUM_ENVS,))
        self.assertGreaterEqual(int(levels.min()), 0)
        self.assertLess(int(levels.max()), num_rows)
        self.assertGreaterEqual(int(types.min()), 0)
        self.assertLess(int(types.max()), num_cols)
        # env -> origin mapping holds after all resets / curriculum moves
        self.assertTrue(torch.allclose(
            sim.env_origins, sim._terrain_origins[levels, types]))
        # the game curriculum moves levels only; the column assignment is fixed
        self.assertTrue(torch.equal(types, self.types_at_build))
        # build-time round-robin start within [0, max_init_terrain_level]
        self.assertLessEqual(
            int(self.levels_at_build.max()), env.cfg.terrain.max_init_terrain_level)

    # ------------------------------------------------------------------
    # 3. at least one completed / reset episode observed
    # ------------------------------------------------------------------

    def test_completed_episode_observed(self):
        # every env times out at least twice over a 3-episode rollout
        self.assertGreaterEqual(self.total_resets, self.NUM_ENVS)
        self.assertTrue(self.last_extras.get("episode"))

    # ------------------------------------------------------------------
    # 4. command-resampling state valid
    # ------------------------------------------------------------------

    def test_command_resampling_state(self):
        env = self.env
        self.assertEqual(tuple(env.commands.shape),
                         (self.NUM_ENVS, env.cfg.commands.num_commands))
        self.assertTrue(bool(torch.isfinite(env.commands).all()))
        eps = 1e-4
        for col, key in ((0, "lin_vel_x"), (1, "lin_vel_y"), (2, "ang_vel_yaw")):
            lo, hi = env.command_ranges[key]
            self.assertGreaterEqual(float(env.commands[:, col].min()), lo - eps, key)
            self.assertLessEqual(float(env.commands[:, col].max()), hi + eps, key)
        # vendored per-env resample countdown state
        self.assertEqual(tuple(env.commands_resampling_step.shape), (self.NUM_ENVS,))
        self.assertTrue(bool(torch.isfinite(env.commands_resampling_step).all()))
        max_countdown = env.cfg.commands.resampling_time / env.dt
        self.assertLessEqual(
            float(env.commands_resampling_step.max()), max_countdown + 1.0)
        self.assertEqual(tuple(env.commands_xy_accumulation.shape), (self.NUM_ENVS, 2))
        self.assertTrue(bool(torch.isfinite(env.commands_xy_accumulation).all()))

    # ------------------------------------------------------------------
    # 5. episode extras carry terrain-curriculum metrics
    # ------------------------------------------------------------------

    def test_episode_extras_curriculum_metrics(self):
        episode = self.last_extras.get("episode", {})
        self.assertIn("terrain_level_all", episode)
        self.assertIn("max_command_x", episode)
        per_type = [k for k in episode
                    if k.startswith("terrain_level_") and k != "terrain_level_all"]
        self.assertTrue(per_type, "expected per-terrain-type curriculum metrics")
        self.assertTrue(any(k.startswith("rew_") for k in episode),
                        "expected reward episode sums in extras")
        level_all = float(episode["terrain_level_all"])
        self.assertGreaterEqual(level_all, 0.0)
        self.assertLess(level_all, float(self.env.cfg.terrain.num_rows))

    # ------------------------------------------------------------------
    # 6. NaN / inf free
    # ------------------------------------------------------------------

    def test_obs_rewards_finite(self):
        self.assertEqual(self.nonfinite_events, [],
                         f"non-finite values at (step, buffer): "
                         f"{self.nonfinite_events[:5]}")
        # final buffers as well (post-rollout state)
        self.assertTrue(bool(torch.isfinite(self.env.obs_buf).all()))
        self.assertTrue(bool(torch.isfinite(self.env.rew_buf).all()))


if __name__ == "__main__":
    unittest.main()
