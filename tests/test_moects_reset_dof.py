"""Reset-DOF randomization tests for ``go2_moects`` / ``go2_moects_him``.

Reference parity: go2_rl_gym's ``legged_robot._reset_dofs``
(go2_rl_gym/legged_gym/envs/base/legged_robot.py:629) samples the reset joint
positions MULTIPLICATIVELY::

    dof_pos = default_dof_pos * U(0.5, 1.5);  dof_vel = 0

The host base class (``LeggedRobot._reset_dofs``,
legged_gym/envs/base/legged_robot.py:645-652) uses the additive
``default + U(-0.2, 0.2)`` instead. ``WtyCurriculumMixin._reset_dofs`` is a
REPLACEMENT override restoring the vendored multiplicative draw, scoped to
the two moects tasks (the mixin is only used by them).

Layers of checks:

1. ``TestMoECTSResetDofResolution`` (CPU-only, always runs): the override
   wins the MRO for both arms, and the base class / other tasks are
   untouched.
2. ``TestMoECTSResetDofFunctional`` (CPU-only, always runs): calls the
   override on a mock self and validates the produced tensors (ratio to
   default inside [0.5, 1.5] with real spread, dof_vel zeroed).
3. ``TestMoECTSResetDofGenesis`` (opt-in, GPU): builds the real
   ``go2_moects`` env (16 envs, headless) and reads ``simulator.dof_pos``
   back after several full ``reset_idx`` rounds.

GATING for the GPU layer: both env vars must be set:
    MOECTS_GENESIS_INTEGRATION=1   (opt-in; keeps the default CPU suite light)
    SIMULATOR=genesis              (Genesis-only; needs a GPU)

Skip-path / collect check (GPU class reports "skipped"):
    .venv/bin/python -m pytest tests/test_moects_reset_dof.py -q

Real run:
    SIMULATOR=genesis MOECTS_GENESIS_INTEGRATION=1 \
        .venv/bin/python -m pytest tests/test_moects_reset_dof.py -q
"""

import os
os.environ.setdefault("SIMULATOR", "genesis")

import unittest
from types import SimpleNamespace

import torch

_INTEGRATION = os.environ.get("MOECTS_GENESIS_INTEGRATION") == "1"
_GENESIS = os.environ.get("SIMULATOR") == "genesis"
_SKIP_REASON = (
    "Genesis reset-dof test is opt-in: set MOECTS_GENESIS_INTEGRATION=1 and "
    "SIMULATOR=genesis (requires a GPU node; see module docstring)")


def _ensure_genesis():
    """Init Genesis once per process (test classes may share a pytest run)."""
    import genesis as gs
    if not gs._initialized:
        gs.init(backend=gs.gpu, logging_level="warning")
    return gs


# ----------------------------------------------------------------------
# 1. method resolution / scoping (CPU-only)
# ----------------------------------------------------------------------

class TestMoECTSResetDofResolution(unittest.TestCase):
    """The mixin override wins for both moects arms; the host base class and
    non-moects tasks keep their additive reset."""

    def test_mixin_override_wins_mro(self):
        from legged_gym.envs.go2.go2_moects.wty_curriculum_mixin import (
            WtyCurriculumMixin)
        from legged_gym.envs.go2.go2_moects.go2_moects import Go2MoECTS
        from legged_gym.envs.go2.go2_moects.go2_moects_him import Go2MoECTSHIM
        for cls in (Go2MoECTS, Go2MoECTSHIM):
            self.assertIs(cls._reset_dofs, WtyCurriculumMixin._reset_dofs,
                          cls.__name__)

    def test_host_base_and_other_tasks_untouched(self):
        from legged_gym.envs.base.legged_robot import LeggedRobot
        from legged_gym.envs.go2.go2_moects.wty_curriculum_mixin import (
            WtyCurriculumMixin)
        self.assertIsNot(LeggedRobot._reset_dofs,
                         WtyCurriculumMixin._reset_dofs)
        # a sampling of non-moects tasks still resolves to their own/base impl
        from legged_gym.envs.go2.go2 import GO2
        from legged_gym.envs.go2.go2_cts.go2_cts import Go2CTS
        for cls in (GO2, Go2CTS):
            self.assertIsNot(cls._reset_dofs,
                             WtyCurriculumMixin._reset_dofs, cls.__name__)


# ----------------------------------------------------------------------
# 2. functional check on a mock self (CPU-only)
# ----------------------------------------------------------------------

class TestMoECTSResetDofFunctional(unittest.TestCase):
    NUM_ENVS = 64
    NUM_ACTIONS = 12

    def _run_override(self, default_dof_pos):
        from legged_gym.envs.go2.go2_moects.wty_curriculum_mixin import (
            WtyCurriculumMixin)
        captured = {}

        def fake_reset_dofs(env_ids, dof_pos, dof_vel):
            captured["dof_pos"] = dof_pos.clone()
            captured["dof_vel"] = dof_vel.clone()

        fake_self = SimpleNamespace(
            simulator=SimpleNamespace(
                default_dof_pos=default_dof_pos,
                reset_dofs=fake_reset_dofs),
            num_actions=self.NUM_ACTIONS,
            device="cpu")
        env_ids = torch.arange(self.NUM_ENVS)
        WtyCurriculumMixin._reset_dofs(fake_self, env_ids)
        return captured

    def test_multiplicative_draw(self):
        torch.manual_seed(0)
        default = torch.tensor(
            [0.1, 0.8, -1.5, 0.1, 0.8, -1.5, 0.1, 0.8, -1.5, 0.1, 0.8, -1.5])
        out = self._run_override(default)
        self.assertEqual(out["dof_pos"].shape, (self.NUM_ENVS, self.NUM_ACTIONS))
        # dof_vel zeroed (both reference and host agree on this)
        self.assertTrue(torch.all(out["dof_vel"] == 0.0))
        # ratio to default must land in [0.5, 1.5] with real spread
        ratio = out["dof_pos"] / default
        self.assertGreaterEqual(ratio.min().item(), 0.5)
        self.assertLessEqual(ratio.max().item(), 1.5)
        self.assertGreater(ratio.std().item(), 0.1)  # U(0.5,1.5): std ~= 0.29

    def test_not_additive_host_semantics(self):
        # additive U(-0.2,0.2) would let a default of 3.0 produce ratios in
        # [0.933, 1.067] only; the multiplicative draw must exceed that band.
        torch.manual_seed(0)
        default = torch.full((self.NUM_ACTIONS,), 3.0)
        out = self._run_override(default)
        ratio = out["dof_pos"] / default
        self.assertLess(ratio.min().item(), 0.9)
        self.assertGreater(ratio.max().item(), 1.1)


# ----------------------------------------------------------------------
# 3. real-env reset distribution (opt-in, GPU)
# ----------------------------------------------------------------------

@unittest.skipUnless(_INTEGRATION and _GENESIS, _SKIP_REASON)
class TestMoECTSResetDofGenesis(unittest.TestCase):
    NUM_ENVS = 16
    RESET_ROUNDS = 8    # +1 initial reset -> 9 * 16 * 12 = 1728 samples/joint

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

        sim = cls.env.simulator
        all_ids = torch.arange(cls.NUM_ENVS, device=cls.env.device)
        samples = []
        cls.env.reset()
        for _ in range(cls.RESET_ROUNDS + 1):
            samples.append(sim.dof_pos.clone())
            cls.env.reset_idx(all_ids)
        cls.samples = torch.stack(samples[:-1])  # last reset has no read-back

    def test_ratio_in_range_with_spread(self):
        sim = self.env.simulator
        default = sim.default_dof_pos.reshape(1, 1, -1)
        nonzero = default.abs() > 1e-6  # 0-default joints stay 0 under any draw
        ratio = self.samples / default
        ratio = ratio[nonzero.expand_as(ratio)]
        self.assertGreaterEqual(ratio.min().item(), 0.5 - 1e-5)
        self.assertLessEqual(ratio.max().item(), 1.5 + 1e-5)
        self.assertGreater(ratio.std().item(), 0.1)  # U(0.5,1.5): std ~= 0.29

    def test_dof_vel_zeroed(self):
        self.assertTrue(torch.all(self.env.simulator.dof_vel == 0.0))


if __name__ == "__main__":
    unittest.main()
