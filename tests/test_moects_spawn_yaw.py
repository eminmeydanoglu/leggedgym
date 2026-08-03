"""Spawn-yaw randomization tests for ``go2_moects`` / ``go2_moects_him``.

Reference parity: go2_rl_gym's ``legged_robot._reset_root_states`` draws the
initial base yaw from U(-pi, pi) at EVERY reset (roll/pitch exactly 0 --
``random_yaw = torch_rand_float(-np.pi, np.pi, ...)``; ``get_quat(random_yaw,
0.0)`` with pitch pinned to zero). The host consumes
``cfg.init_state.yaw_random_scale`` in ``LeggedRobot._reset_root_states``
(legged_gym/envs/base/legged_robot.py:664-671) via independent per-axis
uniform draws + ``quat_from_euler_xyz``, so setting ``yaw_random_scale =
3.14`` on the moects cfg reproduces the reference distribution while
roll/pitch stay unrandomized (host scales remain 0.0).

Two layers of checks:

1. ``TestMoECTSSpawnYawConfig`` (CPU-only, always runs): both moects task
   cfgs carry the yaw contract and the host default / other tasks are
   untouched (the change is config-scoped, hence inert elsewhere).
2. ``TestMoECTSSpawnYawGenesis`` (opt-in, GPU): builds the real
   ``go2_moects`` env (16 envs, headless) and reads the spawn orientation
   back through ``simulator.base_quat`` (gym xyzw format) immediately after
   several explicit ``reset_idx`` rounds, converting with the repo's own
   ``get_euler_xyz``. It deliberately does not call ``env.reset()``: that
   convenience method takes one zero-action physics step after the reset, so
   its sampled base angular velocity can legitimately change roll/pitch. The
   HIM arm shares the same cfg init_state block and
   the same host reset path, so the mechanism is validated once here.

GATING for the GPU layer: both env vars must be set:
    MOECTS_GENESIS_INTEGRATION=1   (opt-in; keeps the default CPU suite light)
    SIMULATOR=genesis              (Genesis-only; needs a GPU)

Skip-path / collect check (GPU class reports "skipped"):
    .venv/bin/python -m pytest tests/test_moects_spawn_yaw.py -q

Real run:
    SIMULATOR=genesis MOECTS_GENESIS_INTEGRATION=1 \
        .venv/bin/python -m pytest tests/test_moects_spawn_yaw.py -q
"""

import math
import os
os.environ.setdefault("SIMULATOR", "genesis")

import unittest
from types import SimpleNamespace

import torch

_INTEGRATION = os.environ.get("MOECTS_GENESIS_INTEGRATION") == "1"
_GENESIS = os.environ.get("SIMULATOR") == "genesis"
_SKIP_REASON = (
    "Genesis spawn-yaw test is opt-in: set MOECTS_GENESIS_INTEGRATION=1 and "
    "SIMULATOR=genesis (requires a GPU node; see module docstring)")

_YAW_SCALE = 3.14   # expected cfg.init_state.yaw_random_scale (~pi)


def _ensure_genesis():
    """Init Genesis once per process (test classes may share a pytest run)."""
    import genesis as gs
    if not gs._initialized:
        gs.init(backend=gs.gpu, logging_level="warning")
    return gs


# ----------------------------------------------------------------------
# 1. config contract (CPU-only)
# ----------------------------------------------------------------------

class TestMoECTSSpawnYawConfig(unittest.TestCase):
    """Both moects arms inherit the spawn-yaw contract; everything else is
    untouched (config-scoped change, inert for non-moects tasks)."""

    @classmethod
    def setUpClass(cls):
        import legged_gym.envs  # noqa: F401  (registers the tasks)
        from legged_gym.utils import task_registry
        cls.get_cfgs = staticmethod(task_registry.get_cfgs)

    def test_moects_cfg_yaw_contract(self):
        for task in ("go2_moects", "go2_moects_him"):
            cfg, _ = self.get_cfgs(task)
            # draw range is U(-scale, scale) with scale ~= pi, never beyond pi
            self.assertAlmostEqual(cfg.init_state.yaw_random_scale, _YAW_SCALE,
                                   places=6, msg=task)
            self.assertLessEqual(cfg.init_state.yaw_random_scale,
                                 math.pi + 1e-6, task)
            # reference randomizes yaw only: roll/pitch stay unrandomized
            self.assertEqual(cfg.init_state.roll_random_scale, 0.0, task)
            self.assertEqual(cfg.init_state.pitch_random_scale, 0.0, task)

    def test_other_tasks_untouched(self):
        # host default unchanged
        from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
        self.assertEqual(LeggedRobotCfg.init_state.yaw_random_scale, 0.0)
        self.assertEqual(LeggedRobotCfg.init_state.roll_random_scale, 0.0)
        self.assertEqual(LeggedRobotCfg.init_state.pitch_random_scale, 0.0)
        # a sampling of non-moects go2 tasks keeps the host default
        for task in ("go2", "go2_ts", "go2_dreamwaq"):
            cfg, _ = self.get_cfgs(task)
            self.assertEqual(cfg.init_state.yaw_random_scale, 0.0, task)


# ----------------------------------------------------------------------
# 2. real-env spawn-yaw distribution (opt-in, GPU)
# ----------------------------------------------------------------------

@unittest.skipUnless(_INTEGRATION and _GENESIS, _SKIP_REASON)
class TestMoECTSSpawnYawGenesis(unittest.TestCase):
    NUM_ENVS = 16
    RESET_ROUNDS = 12   # +1 initial reset -> 13 * 16 = 208 yaw samples
    MIN_STD = 1.0       # U(-3.14, 3.14) has std 3.14/sqrt(3) ~= 1.81
    RP_ATOL = 1e-4      # roll/pitch pinned to 0 (unrandomized)

    @classmethod
    def setUpClass(cls):
        _ensure_genesis()
        import legged_gym.envs  # noqa: F401  (registers go2_moects)
        from legged_gym.utils import task_registry
        from legged_gym.utils.math_utils import get_euler_xyz

        cfg, _ = task_registry.get_cfgs("go2_moects")
        cfg.env.num_envs = cls.NUM_ENVS
        # Spawn orientation is independent of the MoE terrain curriculum.
        # Use the mixin's supported plane fallback so this focused Genesis
        # integration test does not build the 10 x 20 training heightfield.
        cfg.terrain.mesh_type = "plane"
        args = SimpleNamespace(
            task="go2_moects", seed=7, debug=False, headless=True, cpu=False,
            num_envs=cls.NUM_ENVS, max_iterations=None, resume=False,
            sync_wandb=False, ckpt=None, load_run=None, export_onnx=False,
            motion_file=None, num_student=None)
        cls.env, _ = task_registry.make_env("go2_moects", args=args, env_cfg=cfg)

        sim = cls.env.simulator
        all_ids = torch.arange(cls.NUM_ENVS, device=cls.env.device)
        rolls, pitchs, yaws = [], [], []
        # Read each spawn quaternion immediately after reset_idx.  Do NOT use
        # env.reset(): BaseTask.reset calls reset_idx and then step(zeros), and
        # _reset_root_states deliberately randomizes base_ang_vel in
        # U(-0.5, 0.5) rad/s.  One 0.02 s control step can therefore create
        # ~1e-2 rad physical roll/pitch despite a pure-yaw spawn quaternion.
        # reset_root_states writes simulator.base_quat synchronously in gym
        # xyzw format, before any physics step or scene refresh.
        for _ in range(cls.RESET_ROUNDS + 1):
            cls.env.reset_idx(all_ids)
            rpy = get_euler_xyz(sim.base_quat)
            rolls.append(rpy[:, 0])
            pitchs.append(rpy[:, 1])
            yaws.append(rpy[:, 2])
        cls.roll = torch.cat(rolls).float().cpu()
        cls.pitch = torch.cat(pitchs).float().cpu()
        cls.yaw = torch.cat(yaws).float().cpu()
        print(f"\n[spawn-yaw] n={cls.yaw.numel()} "
              f"yaw: min={cls.yaw.min():.3f} max={cls.yaw.max():.3f} "
              f"mean={cls.yaw.mean():.3f} std={cls.yaw.std():.3f} | "
              f"|roll|max={cls.roll.abs().max():.2e} "
              f"|pitch|max={cls.pitch.abs().max():.2e}")

    @classmethod
    def tearDownClass(cls):
        env = getattr(cls, "env", None)
        if env is not None and hasattr(env, "destroy"):
            env.destroy()

    def test_cfg_scale_reached_the_env(self):
        self.assertAlmostEqual(
            self.env.cfg.init_state.yaw_random_scale, _YAW_SCALE, places=6)

    def test_yaw_within_pi_bounds(self):
        self.assertGreaterEqual(float(self.yaw.min()), -math.pi - 1e-5)
        self.assertLessEqual(float(self.yaw.max()), math.pi + 1e-5)

    def test_yaw_has_real_spread(self):
        std = float(self.yaw.std())
        self.assertGreater(std, self.MIN_STD,
                           f"spawn yaw collapsed (std={std:.3f}); "
                           f"yaw_random_scale not consumed?")
        # both signs present (not all facing one hemisphere)
        self.assertLess(float(self.yaw.min()), -1.0,
                        f"no negative spawn yaw (min={float(self.yaw.min()):.3f})")
        self.assertGreater(float(self.yaw.max()), 1.0,
                           f"no positive spawn yaw (max={float(self.yaw.max()):.3f})")

    def test_roll_pitch_unrandomized(self):
        self.assertLess(float(self.roll.abs().max()), self.RP_ATOL,
                        f"roll randomized: max |roll|={float(self.roll.abs().max()):.2e}")
        self.assertLess(float(self.pitch.abs().max()), self.RP_ATOL,
                        f"pitch randomized: max |pitch|={float(self.pitch.abs().max()):.2e}")


if __name__ == "__main__":
    unittest.main()
