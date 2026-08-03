"""dof_acc timing-parity tests for ``go2_moects`` / ``go2_moects_him``.

Reference semantics (go2_rl_gym): ``last_dof_vel`` is refreshed ONCE PER
CONTROL STEP, at the very end of ``post_physics_step``
(go2_rl_gym/legged_gym/envs/base/legged_robot.py:143-145), so both the
dof_acc reward (ref :1257-1259) and the privileged-obs dof_acc feature
(go2_env.py:45) span a full 20 ms control window.

Host default: ``genesis_simulator.py:40`` refreshes its ``_last_dof_vel``
every SIM SUBSTEP (5 ms), which made the same formulas ~4x smaller (feature)
and ~16x weaker (reward). The simulator buffer also feeds PD velocity
damping and is deliberately untouched.

Fix: ``WtyCurriculumMixin`` keeps its own control-rate tracker
``_wty_last_dof_vel``: zeroed on reset, read by the ``_reward_dof_acc``
override and by the MoE arm's privileged-obs feature, refreshed AFTER
observations each control step (inline in ``Go2MoECTS.compute_observations``
-- its own method shadows the mixin wrap in the MRO -- and via the mixin's
``compute_observations`` wrap on the HIM arm).

Layers:

1. ``TestMoECTSDofAccResolution`` (CPU-only): override resolution / MRO
   facts and host-base untouchedness.
2. ``TestMoECTSDofAccFunctional`` (CPU-only): the reward formula on a mock
   self.
3. ``TestMoECTSDofAccGenesis`` (opt-in, GPU): real env, 16 envs -- reset
   zeroing, post-step refresh equality, and the privileged-obs dof_acc slice
   matching (prev_dof_vel - dof_vel)/dt over 20 ms.

GATING for the GPU layer: both env vars must be set:
    MOECTS_GENESIS_INTEGRATION=1   (opt-in; keeps the default CPU suite light)
    SIMULATOR=genesis              (Genesis-only; needs a GPU)

Real run:
    SIMULATOR=genesis MOECTS_GENESIS_INTEGRATION=1 \
        .venv/bin/python -m pytest tests/test_moects_dof_acc.py -q
"""

import os
os.environ.setdefault("SIMULATOR", "genesis")

import unittest
from types import SimpleNamespace

import torch

_INTEGRATION = os.environ.get("MOECTS_GENESIS_INTEGRATION") == "1"
_GENESIS = os.environ.get("SIMULATOR") == "genesis"
_SKIP_REASON = (
    "Genesis dof_acc test is opt-in: set MOECTS_GENESIS_INTEGRATION=1 and "
    "SIMULATOR=genesis (requires a GPU node; see module docstring)")

# privileged obs layout (go2_moects.py:49-56):
# lin_vel(3) + clean 45 + feet_force(4) + torques(12) + dof_acc(12) + heights
_DOF_ACC_SLICE = slice(3 + 45 + 4 + 12, 3 + 45 + 4 + 12 + 12)


def _ensure_genesis():
    import genesis as gs
    if not gs._initialized:
        gs.init(backend=gs.gpu, logging_level="warning")
    return gs


# ----------------------------------------------------------------------
# 1. resolution / MRO (CPU-only)
# ----------------------------------------------------------------------

class TestMoECTSDofAccResolution(unittest.TestCase):

    def test_reward_override_wins_both_arms(self):
        from legged_gym.envs.go2.go2_moects.wty_curriculum_mixin import (
            WtyCurriculumMixin)
        from legged_gym.envs.go2.go2_moects.go2_moects import Go2MoECTS
        from legged_gym.envs.go2.go2_moects.go2_moects_him import Go2MoECTSHIM
        for cls in (Go2MoECTS, Go2MoECTSHIM):
            self.assertIs(cls._reward_dof_acc,
                          WtyCurriculumMixin._reward_dof_acc, cls.__name__)

    def test_observations_refresh_paths(self):
        from legged_gym.envs.go2.go2_moects.wty_curriculum_mixin import (
            WtyCurriculumMixin)
        from legged_gym.envs.go2.go2_moects.go2_moects import Go2MoECTS
        from legged_gym.envs.go2.go2_moects.go2_moects_him import Go2MoECTSHIM
        # HIM arm: the mixin wrap wins the MRO
        self.assertIs(Go2MoECTSHIM.compute_observations,
                      WtyCurriculumMixin.compute_observations)
        # MoE arm: its own method shadows the wrap and must therefore contain
        # the inline refresh (behavior verified in the GPU layer)
        self.assertIsNot(Go2MoECTS.compute_observations,
                         WtyCurriculumMixin.compute_observations)

    def test_host_base_untouched(self):
        from legged_gym.envs.base.legged_robot import LeggedRobot
        from legged_gym.envs.go2.go2_moects.wty_curriculum_mixin import (
            WtyCurriculumMixin)
        self.assertIsNot(LeggedRobot._reward_dof_acc,
                         WtyCurriculumMixin._reward_dof_acc)
        # base compute_observations has no _wty tracking
        import inspect
        self.assertNotIn("_wty_last_dof_vel",
                         inspect.getsource(LeggedRobot.compute_observations))


# ----------------------------------------------------------------------
# 2. reward formula on a mock self (CPU-only)
# ----------------------------------------------------------------------

class TestMoECTSDofAccFunctional(unittest.TestCase):

    def test_reward_uses_control_rate_tracker(self):
        from legged_gym.envs.go2.go2_moects.wty_curriculum_mixin import (
            WtyCurriculumMixin)
        num_envs, num_actions = 4, 12
        last = torch.full((num_envs, num_actions), 2.0)
        cur = torch.full((num_envs, num_actions), 0.5)
        fake_self = SimpleNamespace(
            _wty_last_dof_vel=last,
            simulator=SimpleNamespace(dof_vel=cur),
            dt=0.02)
        out = WtyCurriculumMixin._reward_dof_acc(fake_self)
        expected = torch.sum(((last - cur) / 0.02) ** 2, dim=1)
        self.assertTrue(torch.allclose(out, expected))
        # sanity: the value is the 20 ms-window one, i.e. 16x the value a
        # 5 ms-window dt-normalization would give for the same velocity jump
        # (same formula, but the POINT of the tracker is the window length --
        # verified behaviorally in the GPU layer).

    def test_zero_when_velocity_constant(self):
        from legged_gym.envs.go2.go2_moects.wty_curriculum_mixin import (
            WtyCurriculumMixin)
        vel = torch.randn(3, 12)
        fake_self = SimpleNamespace(
            _wty_last_dof_vel=vel.clone(),
            simulator=SimpleNamespace(dof_vel=vel.clone()),
            dt=0.02)
        out = WtyCurriculumMixin._reward_dof_acc(fake_self)
        self.assertTrue(torch.all(out == 0.0))


# ----------------------------------------------------------------------
# 3. real-env behavior (opt-in, GPU)
# ----------------------------------------------------------------------

@unittest.skipUnless(_INTEGRATION and _GENESIS, _SKIP_REASON)
class TestMoECTSDofAccGenesis(unittest.TestCase):
    NUM_ENVS = 16

    @classmethod
    def setUpClass(cls):
        _ensure_genesis()
        import legged_gym.envs  # noqa: F401  (registers go2_moects)
        from legged_gym.utils import task_registry

        cfg, _ = task_registry.get_cfgs("go2_moects")
        cfg.env.num_envs = cls.NUM_ENVS
        args = SimpleNamespace(
            task="go2_moects", seed=7, debug=False, headless=True, cpu=False,
            num_envs=cls.NUM_ENVS, max_iterations=None, resume=False,
            sync_wandb=False, ckpt=None, load_run=None, export_onnx=False,
            motion_file=None, num_student=None)
        cls.env, _ = task_registry.make_env("go2_moects", args=args, env_cfg=cfg)

    def test_reset_zeroes_tracker(self):
        env = self.env
        all_ids = torch.arange(self.NUM_ENVS, device=env.device)
        env.reset_idx(all_ids)
        self.assertTrue(torch.all(env._wty_last_dof_vel == 0.0))

    def test_post_step_refresh_and_feature_window(self):
        env = self.env
        actions = 0.1 * torch.randn(
            self.NUM_ENVS, env.num_actions, device=env.device)
        # step 1: tracker must equal the post-step dof_vel
        env.step(actions)
        self.assertTrue(torch.allclose(
            env._wty_last_dof_vel, env.simulator.dof_vel, atol=1e-6))
        prev_vel = env.simulator.dof_vel.clone()
        # step 2: the privileged-obs dof_acc slice must be computed against
        # the PREVIOUS control step's velocity (20 ms window), and the
        # tracker afterwards equals the new velocity.
        env.step(actions)
        feature = env.privileged_obs_buf[:, _DOF_ACC_SLICE]
        expected = (prev_vel - env.simulator.dof_vel) / env.dt * 1e-4
        self.assertTrue(torch.allclose(feature, expected, atol=1e-5),
                        msg="dof_acc obs feature is not using the 20 ms "
                            "control-rate tracker")
        self.assertTrue(torch.allclose(
            env._wty_last_dof_vel, env.simulator.dof_vel, atol=1e-6))
        # and the 20 ms window really differs from the simulator's 5 ms
        # per-substep buffer for this step (guards the point of the fix)
        if torch.allclose(env.simulator.last_dof_vel, prev_vel, atol=1e-6):
            self.fail("simulator per-substep buffer unexpectedly equals the "
                      "previous control-step velocity -- the test cannot "
                      "distinguish the two windows")


if __name__ == "__main__":
    unittest.main()
