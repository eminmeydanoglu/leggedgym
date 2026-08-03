"""Friction domain-randomization tests for ``go2_moects`` / ``go2_moects_him``.

The moects tasks target an EFFECTIVE (post-combine) robot<->terrain friction of
U[0.5, 1.5] per env, matching the reference go2_rl_gym (PhysX: absolute link
friction U[0, 2], average-combined with the ground at 1.0 -> ~[0.5, 1.5]).

Genesis instead MAX-combines link and ground friction
(genesis/engine/solvers/rigid/collider/contact.py:
``contact_friction = max(link_base * link_ratio, ground_base * ground_ratio, 1e-2)``),
so the moects cfg sets the ground to 0.5 (``terrain.static_friction``) and draws
the per-env link ratio from U[0.5, 1.5] (``domain_rand.friction_range``; the
go2 MJCF base friction is 1.0, so the ratio IS the absolute link friction).
Effective = max(link, 0.5) = link in [0.5, 1.5].

Three layers of checks:

1. ``TestMoECTSFrictionConfig`` (CPU-only, always runs): both moects task cfgs
   carry the friction contract and the host defaults / other tasks are
   untouched (the change is config-scoped, hence inert elsewhere).
2. ``TestGenesisMaxCombineProbe`` (opt-in, GPU): a tiny standalone Genesis
   scene that PHYSICALLY verifies the max-combine rule this design relies on
   (tilted-gravity slip onset of two boxes with known frictions).
3. ``TestMoECTSFrictionDistribution`` (opt-in, GPU): builds the real
   ``go2_moects`` env (16 envs, headless) and reads the friction state back
   through the simulator's own getters: per-env link friction in [0.5, 1.5]
   with nontrivial spread, ground at 0.5, effective max(link, ground) in
   [0.5, 1.5] and not collapsed.

GATING for the GPU layers: both env vars must be set:
    MOECTS_GENESIS_INTEGRATION=1   (opt-in; keeps the default CPU suite light)
    SIMULATOR=genesis              (Genesis-only; needs a GPU)

Skip-path / collect check (GPU classes report "skipped"):
    .venv/bin/python -m pytest tests/test_moects_friction.py -q

Real run:
    SIMULATOR=genesis MOECTS_GENESIS_INTEGRATION=1 \
        .venv/bin/python -m pytest tests/test_moects_friction.py -q
"""

import os
os.environ.setdefault("SIMULATOR", "genesis")

import unittest
from types import SimpleNamespace

import torch

_INTEGRATION = os.environ.get("MOECTS_GENESIS_INTEGRATION") == "1"
_GENESIS = os.environ.get("SIMULATOR") == "genesis"
_SKIP_REASON = (
    "Genesis friction test is opt-in: set MOECTS_GENESIS_INTEGRATION=1 and "
    "SIMULATOR=genesis (requires a GPU node; see module docstring)")

_FRICTION_RANGE = [0.5, 1.5]   # expected cfg.domain_rand.friction_range
_GROUND_FRICTION = 0.5         # expected cfg.terrain.static_friction


def _ensure_genesis():
    """Init Genesis once per process (test classes may share a pytest run)."""
    import genesis as gs
    if not gs._initialized:
        gs.init(backend=gs.gpu, logging_level="warning")
    return gs


# ----------------------------------------------------------------------
# 1. config contract (CPU-only)
# ----------------------------------------------------------------------

class TestMoECTSFrictionConfig(unittest.TestCase):
    """Both moects arms inherit the friction contract; everything else is
    untouched (config-scoped change, inert for non-moects tasks)."""

    @classmethod
    def setUpClass(cls):
        import legged_gym.envs  # noqa: F401  (registers the tasks)
        from legged_gym.utils import task_registry
        cls.get_cfgs = staticmethod(task_registry.get_cfgs)

    def test_moects_cfg_friction_contract(self):
        for task in ("go2_moects", "go2_moects_him"):
            cfg, _ = self.get_cfgs(task)
            self.assertEqual(list(cfg.domain_rand.friction_range), _FRICTION_RANGE, task)
            self.assertTrue(cfg.domain_rand.randomize_friction, task)
            self.assertAlmostEqual(cfg.terrain.static_friction, _GROUND_FRICTION, msg=task)

    def test_other_tasks_untouched(self):
        # host defaults unchanged
        from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
        self.assertAlmostEqual(LeggedRobotCfg.terrain.static_friction, 1.0)
        self.assertEqual(list(LeggedRobotCfg.domain_rand.friction_range), [0.5, 1.25])
        # a sampling of non-moects go2 tasks keeps the host defaults
        for task in ("go2", "go2_ts", "go2_dreamwaq"):
            cfg, _ = self.get_cfgs(task)
            self.assertAlmostEqual(cfg.terrain.static_friction, 1.0, task)
            self.assertNotEqual(list(cfg.domain_rand.friction_range), _FRICTION_RANGE
                                if task != "go2" else [0.0, 2.0], task)


# ----------------------------------------------------------------------
# 2. physical probe of the Genesis max-combine rule (opt-in, GPU)
# ----------------------------------------------------------------------

@unittest.skipUnless(_INTEGRATION and _GENESIS, _SKIP_REASON)
class TestGenesisMaxCombineProbe(unittest.TestCase):
    """Two boxes under tilted gravity (tan(theta) = 0.65) on a mu=0.5 plane:

    - box_lo (link mu 0.3): max(0.3, 0.5) = 0.50 < 0.85  -> must SLIDE
    - box_hi (link mu 1.0): max(1.0, 0.5) = 1.00 > 0.85  -> must STAY

    Only the max-combine rule predicts (slide, stay): min-combine predicts
    (slide, slide) [min(1.0, 0.5) = 0.5 < 0.85] and PhysX-style average-combine
    predicts (slide, slide) [(1.0 + 0.5) / 2 = 0.75 < 0.85].

    Assertions use the terminal sliding SPEED (a stuck box has ~zero velocity
    after the initial contact transient; a sliding box keeps accelerating) plus
    the net displacement, so solver micro-creep cannot flip the verdict.
    """

    TAN_THETA = 0.85

    @classmethod
    def setUpClass(cls):
        gs = _ensure_genesis()
        sin_t = cls.TAN_THETA / (1.0 + cls.TAN_THETA ** 2) ** 0.5
        cos_t = 1.0 / (1.0 + cls.TAN_THETA ** 2) ** 0.5
        g = 9.81
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=0.005, gravity=(0.0, -g * sin_t, -g * cos_t)),
            show_viewer=False,
        )
        plane = scene.add_entity(gs.morphs.Plane())
        box_lo = scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.0, 0.0, 0.1)))
        box_hi = scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(1.0, 0.0, 0.1)))
        # same call order as the host simulator (set_friction before build)
        plane.set_friction(_GROUND_FRICTION)
        box_lo.set_friction(0.3)
        box_hi.set_friction(1.0)
        scene.build()
        for _ in range(300):  # 1.5 s: a sliding box reaches ~3 m/s, moves ~1 m
            scene.step()
        cls.y_lo = float(box_lo.get_pos()[1])
        cls.y_hi = float(box_hi.get_pos()[1])
        cls.v_lo = float(box_lo.get_vel()[1])
        cls.v_hi = float(box_hi.get_vel()[1])
        print(f"\n[probe] tan(theta)={cls.TAN_THETA}: "
              f"box_lo(mu=0.3) y={cls.y_lo:.4f} vy={cls.v_lo:.4f} | "
              f"box_hi(mu=1.0) y={cls.y_hi:.4f} vy={cls.v_hi:.4f}", flush=True)

    def test_low_friction_box_slides(self):
        # effective mu = max(0.3, 0.5) = 0.5 < tan(theta) -> accelerates in -y
        self.assertLess(self.y_lo, -0.1,
                        f"box_lo should have slid (y={self.y_lo:.4f})")
        self.assertLess(self.v_lo, -0.5,
                        f"box_lo should still be sliding (vy={self.v_lo:.4f})")

    def test_high_friction_box_stays(self):
        # effective mu = max(1.0, 0.5) = 1.0 > tan(theta) -> static
        self.assertAlmostEqual(self.v_hi, 0.0, delta=0.05,
                               msg=f"box_hi should be at rest under max-combine "
                                   f"(vy={self.v_hi:.4f}, y={self.y_hi:.4f})")
        self.assertAlmostEqual(self.y_hi, 0.0, delta=0.02,
                               msg=f"box_hi should stick under max-combine (y={self.y_hi:.4f})")


# ----------------------------------------------------------------------
# 3. real-env friction distribution (opt-in, GPU)
# ----------------------------------------------------------------------

@unittest.skipUnless(_INTEGRATION and _GENESIS, _SKIP_REASON)
class TestMoECTSFrictionDistribution(unittest.TestCase):
    NUM_ENVS = 16
    MIN_SPREAD = 0.3   # P(spread < 0.3 | 16 iid U[0.5,1.5] draws) ~ 1e-11

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
        solver = sim._robot._solver
        # --- robot links: base friction (global) x per-env ratio ---
        robot_geoms = list(range(sim._robot.geom_start, sim._robot.geom_end))
        cls.link_base = solver.get_geoms_friction(robot_geoms).float()          # (n_geoms,)
        cls.link_ratio = solver.get_geoms_friction_ratio(robot_geoms).float()   # (n_envs, n_geoms)
        cls.link_friction = cls.link_base.unsqueeze(0) * cls.link_ratio
        # --- terrain: global friction + its ratio (default 1.0) ---
        terr_geoms = list(range(sim._gs_terrain.geom_start, sim._gs_terrain.geom_end))
        cls.ground_base = solver.get_geoms_friction(terr_geoms).float()
        cls.ground_ratio = solver.get_geoms_friction_ratio(terr_geoms).float()
        cls.ground_friction = cls.ground_base * cls.ground_ratio[0]  # global, same for all envs
        # --- effective contact friction under the max-combine rule ---
        cls.effective = torch.maximum(
            cls.link_friction, cls.ground_friction.max().expand_as(cls.link_friction))
        cls.env_friction = cls.effective[:, 0]  # one draw per env (see test below)

    @classmethod
    def tearDownClass(cls):
        env = getattr(cls, "env", None)
        if env is not None and hasattr(env, "destroy"):
            env.destroy()

    def test_mjcf_base_friction_is_one(self):
        # premise of the design: ratio == absolute link friction (go2.xml = 1.0)
        self.assertTrue(torch.allclose(
            self.link_base, torch.ones_like(self.link_base), atol=1e-6),
            f"robot geom base frictions: {self.link_base.unique().tolist()}")

    def test_link_friction_in_range_per_env(self):
        lo, hi = _FRICTION_RANGE
        self.assertGreaterEqual(float(self.link_friction.min()), lo - 1e-5)
        self.assertLessEqual(float(self.link_friction.max()), hi + 1e-5)
        # the host draws ONE ratio per env (same value on every link/geom)
        per_env_span = (self.link_friction.max(dim=1).values
                        - self.link_friction.min(dim=1).values)
        self.assertLess(float(per_env_span.max()), 1e-6)
        # matches the simulator's own bookkeeping buffer
        self.assertTrue(torch.allclose(
            self.env.simulator._friction_values.squeeze(1).float(),
            self.link_ratio[:, 0], atol=1e-6))

    def test_link_friction_has_spread(self):
        per_env = self.link_friction[:, 0]
        spread = float(per_env.max() - per_env.min())
        self.assertGreater(spread, self.MIN_SPREAD,
                           f"per-env friction collapsed: {per_env.tolist()}")
        self.assertGreater(float(per_env.std()), 0.05)

    def test_ground_friction_is_half(self):
        self.assertTrue(torch.allclose(
            self.ground_friction,
            torch.full_like(self.ground_friction, _GROUND_FRICTION), atol=1e-6),
            f"terrain geoms friction: {self.ground_friction.tolist()}")

    def test_effective_friction_in_range_with_spread(self):
        lo, hi = _FRICTION_RANGE
        # effective = max(link, 0.5) = link since link >= 0.5
        self.assertTrue(torch.allclose(self.effective, self.link_friction, atol=1e-6))
        self.assertGreaterEqual(float(self.effective.min()), lo - 1e-5)
        self.assertLessEqual(float(self.effective.max()), hi + 1e-5)
        spread = float(self.env_friction.max() - self.env_friction.min())
        self.assertGreater(spread, self.MIN_SPREAD,
                           f"effective friction collapsed: {self.env_friction.tolist()}")
        # not degenerate: ~all envs got distinct draws
        self.assertGreater(len(torch.unique(self.env_friction)), self.NUM_ENVS // 2)

    def test_friction_values_summary(self):
        # not an assertion: leaves the observed distribution in the test log
        per_env = self.env_friction
        print(f"\n[friction] per-env effective over {self.NUM_ENVS} envs: "
              f"min={float(per_env.min()):.4f} max={float(per_env.max()):.4f} "
              f"mean={float(per_env.mean()):.4f} std={float(per_env.std()):.4f} "
              f"ground={float(self.ground_friction.max()):.4f}")
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
