"""Unit/smoke tests for the added-mass learn+use probe (no Genesis required).

Drives the shipped modules:
  - probe_physics_logic
  - probe_adapters.{rma,dreamwaq,him}
  - probe_physics_use.analyze_samples / CLI --help
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "legged_gym" / "scripts" / "eval"

# Adapters / runner import through legged_gym package (needs SIMULATOR).
os.environ.setdefault("SIMULATOR", "genesis")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load pure logic without pulling legged_gym package init
logic = _load(
    "probe_physics_logic_under_test",
    EVAL / "probe_physics_logic.py",
)


def _env_with_sim():
    env = os.environ.copy()
    env["SIMULATOR"] = "genesis"
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    return env


# ---------------------------------------------------------------------------
# Mass map + invariant
# ---------------------------------------------------------------------------


class TestOppositeMassMap(unittest.TestCase):
    def test_canonical_pairs(self):
        self.assertEqual(logic.opposite_mass_map(-2.0), 5.0)
        self.assertEqual(logic.opposite_mass_map(0.0), 3.0)
        self.assertEqual(logic.opposite_mass_map(3.0), 0.0)
        self.assertEqual(logic.opposite_mass_map(5.0), -2.0)

    def test_vectorized(self):
        m = np.array([-2.0, 0.0, 3.0, 5.0])
        out = logic.opposite_mass_map(m)
        np.testing.assert_allclose(out, [5.0, 3.0, 0.0, -2.0])

    def test_index_pairs(self):
        pairs = logic.opposite_mass_index_pairs()
        grid = list(logic.MASS_GRID_KG)
        for i, j in pairs:
            self.assertEqual(
                float(logic.opposite_mass_map(grid[i])), float(grid[j])
            )


class TestMassInvariant(unittest.TestCase):
    def test_pass(self):
        logic.check_mass_invariant(np.array([0.0, 3.0, 3.0]), np.array([0.0, 3.0, 3.0]))

    def test_fail_on_drift(self):
        with self.assertRaises(ValueError) as ctx:
            logic.check_mass_invariant(
                np.array([0.0, 3.5, 3.0]),
                np.array([0.0, 3.0, 3.0]),
                atol=1e-3,
            )
        self.assertIn("mass invariant violated", str(ctx.exception))

    def test_groups_constant(self):
        logic.assert_mass_groups_constant({-2: [-2, -2], 5: [5.0, 5.0]})
        with self.assertRaises(ValueError):
            logic.assert_mass_groups_constant({0: [0.0, 1.0]})


# ---------------------------------------------------------------------------
# Trajectory split
# ---------------------------------------------------------------------------


class TestTrajectorySplit(unittest.TestCase):
    def _synth(self, n_traj_per_mass=8, steps=5):
        masses = np.array([-2.0, 0.0, 3.0, 5.0])
        traj_ids = []
        mass = []
        tid = 0
        for m in masses:
            for _ in range(n_traj_per_mass):
                traj_ids.extend([tid] * steps)
                mass.extend([m] * steps)
                tid += 1
        return np.array(traj_ids), np.array(mass)

    def test_no_traj_leakage(self):
        traj, mass = self._synth()
        split = logic.trajectory_train_test_split(traj, mass, seed=0)
        tr = set(traj[split.train_idx].tolist())
        te = set(traj[split.test_idx].tolist())
        self.assertEqual(tr & te, set())

    def test_all_masses_in_both(self):
        traj, mass = self._synth()
        split = logic.trajectory_train_test_split(traj, mass, seed=1)
        for m in np.unique(mass):
            self.assertTrue(np.any(mass[split.train_idx] == m), f"train missing {m}")
            self.assertTrue(np.any(mass[split.test_idx] == m), f"test missing {m}")


# ---------------------------------------------------------------------------
# Decoder R²
# ---------------------------------------------------------------------------


class TestMassDecoder(unittest.TestCase):
    def test_planted_linear_high_r2_shuffle_low(self):
        rng = np.random.default_rng(0)
        n_traj = 40
        steps = 10
        d = 8
        traj = np.repeat(np.arange(n_traj), steps)
        # 4 mass levels assigned to trajs
        mass_levels = np.array([-2.0, 0.0, 3.0, 5.0])
        traj_mass = mass_levels[np.arange(n_traj) % 4]
        mass = traj_mass[traj]
        # plant mass in z[:,0]
        z = rng.normal(size=(len(traj), d)).astype(np.float64)
        z[:, 0] = mass / 5.0 + 0.01 * rng.normal(size=len(traj))

        out = logic.mass_decode_with_shuffle_control(z, mass, traj, seed=0, epochs=200)
        self.assertGreater(out["r2"], 0.85, f"planted R²={out['r2']}")
        self.assertLess(out["shuffled_r2"], 0.3, f"shuffle R²={out['shuffled_r2']}")


# ---------------------------------------------------------------------------
# Paired Δuse + classify
# ---------------------------------------------------------------------------


class TestUseMetrics(unittest.TestCase):
    def test_positive_delta_use_ci(self):
        rng = np.random.default_rng(0)
        control = rng.normal(0.3, 0.05, size=64)
        wrong = control + 0.15 + rng.normal(0, 0.02, size=64)
        normal = control + rng.normal(0, 0.01, size=64)
        res = logic.aggregate_use_test(normal, control, wrong, seed=0, n_bootstrap=500)
        self.assertGreater(res.delta_use, 0.1)
        self.assertGreater(res.delta_use_ci_lo, 0.0)

    def test_classify(self):
        self.assertEqual(
            logic.classify_result(0.8, 0.0, 0.1, 0.05),
            "öğrendi ve kullandı",
        )
        self.assertEqual(
            logic.classify_result(0.8, 0.0, 0.01, -0.05),
            "öğrendi ama kullanmadı",
        )
        self.assertEqual(
            logic.classify_result(0.2, 0.0, 0.5, 0.4),
            "öğrenildiği gösterilemedi",
        )
        self.assertEqual(
            logic.classify_result(0.9, 0.5, 0.5, 0.4),
            "öğrenildiği gösterilemedi",  # shuffle high → probe broken
        )
        # NaN tracking CI must NOT pass used_track
        self.assertEqual(
            logic.classify_result(
                0.8, 0.0, 0.1, float("nan"),
                delta_fall=0.0, delta_fall_ci_lo=float("nan"),
            ),
            "öğrendi ama kullanmadı",
        )

    def test_used_via_channels(self):
        flags = logic.use_evidence_flags(0.1, 0.05, delta_fall=0.2, delta_fall_ci_lo=0.1)
        self.assertEqual(logic.used_via_label(flags), "both")
        flags = logic.use_evidence_flags(0.1, 0.05, delta_fall=0.0, delta_fall_ci_lo=-1.0)
        self.assertEqual(logic.used_via_label(flags), "tracking")
        flags = logic.use_evidence_flags(0.0, float("nan"), delta_fall=0.2, delta_fall_ci_lo=0.1)
        self.assertEqual(logic.used_via_label(flags), "fall")

    def test_frozen_latent_bank_helper(self):
        import torch
        frozen = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        live = frozen + 100.0
        ever_fell = np.array([False, True, False, True])
        out = logic.update_frozen_latent_bank(frozen, live, ever_fell)
        # alive slots refresh
        torch.testing.assert_close(out[0], live[0])
        torch.testing.assert_close(out[2], live[2])
        # fallen slots keep pre-fall
        torch.testing.assert_close(out[1], frozen[1])
        torch.testing.assert_close(out[3], frozen[3])

    def test_cross_donors_one_to_one_when_equal_groups(self):
        mass = np.repeat(np.array([-2.0, 0.0, 3.0, 5.0]), 4)
        within, cross = logic.make_within_and_cross_donors(mass, seed=0)
        stats = logic.donor_map_stats(cross)
        # equal per-mass groups → bijection → max receivers per donor == 1
        self.assertEqual(stats["max_receivers_per_donor"], 1.0)
        self.assertEqual(stats["unique_donor_count"], float(len(mass)))

    def test_within_donors_are_bijective_derangement(self):
        mass = np.repeat(np.array([-2.0, 0.0, 3.0, 5.0]), 8)
        for seed in range(20):
            within, cross = logic.make_within_and_cross_donors(mass, seed=seed)
            n = len(mass)
            self.assertTrue(np.all(within != np.arange(n)), f"seed={seed} self-match")
            self.assertTrue(np.allclose(mass[within], mass), f"seed={seed} mass")
            wstats = logic.donor_map_stats(within)
            self.assertEqual(wstats["unique_donor_count"], float(n), f"seed={seed}")
            self.assertEqual(wstats["max_receivers_per_donor"], 1.0, f"seed={seed}")
            cstats = logic.donor_map_stats(cross)
            self.assertEqual(cstats["unique_donor_count"], float(n), f"seed={seed}")
            self.assertEqual(cstats["max_receivers_per_donor"], 1.0, f"seed={seed}")


# ---------------------------------------------------------------------------
# Physics contract
# ---------------------------------------------------------------------------


class TestPhysicsContract(unittest.TestCase):
    def test_disables_v3_switch_and_push(self):
        dr = SimpleNamespace(
            resample_physics_within_episode=True,
            push_robots=True,
            randomize_friction=True,
            randomize_base_mass=True,
            randomize_com_displacement=True,
        )
        ranges = SimpleNamespace(
            lin_vel_x=[-1, 1], lin_vel_y=[-1, 1], ang_vel_yaw=[-1, 1]
        )
        cmds = SimpleNamespace(
            curriculum=True, heading_command=True, zero_cmd_prob=0.1, ranges=ranges
        )
        cfg = SimpleNamespace(domain_rand=dr, commands=cmds)
        applied = logic.apply_probe_physics_contract(cfg)
        self.assertFalse(dr.resample_physics_within_episode)
        self.assertFalse(dr.push_robots)
        self.assertFalse(dr.randomize_friction)
        self.assertEqual(applied["command_default"], [0.0, 1.0, 0.0])
        self.assertEqual(list(ranges.lin_vel_y), [1.0, 1.0])


# ---------------------------------------------------------------------------
# RMA wrong privilege construction
# ---------------------------------------------------------------------------


class TestRMAWrongPriv(unittest.TestCase):
    def test_mass_slot_wrong_vel_intact(self):
        n = 4
        # priv: [fr, mass, cx, cy, cz, vx, vy, vz]
        priv = torch.zeros(n, 8)
        priv[:, 0] = 0.1  # friction norm
        priv[:, 1] = torch.tensor([-1.0, -0.4, 0.4, 1.0])  # placeholder mass norms
        priv[:, 5] = torch.tensor([0.5, 0.6, 0.7, 0.8])  # vx
        priv[:, 6] = 1.0  # vy
        real_mass = torch.tensor([-2.0, 0.0, 3.0, 5.0])
        wrong = logic.build_rma_wrong_privilege(priv, real_mass)
        # velocity slots unchanged
        torch.testing.assert_close(wrong[:, 5:], priv[:, 5:])
        # friction/com unchanged
        torch.testing.assert_close(wrong[:, 0], priv[:, 0])
        torch.testing.assert_close(wrong[:, 2:5], priv[:, 2:5])
        # mass slot equals opposite-end normalized
        expected_raw = np.array([5.0, 3.0, 0.0, -2.0])
        expected_norm = logic.dr_normalize(expected_raw, -2.0, 5.0)
        np.testing.assert_allclose(
            wrong[:, 1].numpy(), expected_norm, atol=1e-5
        )


# ---------------------------------------------------------------------------
# Adapter latent swap (synthetic modules)
# ---------------------------------------------------------------------------


class _FakeVAE(nn.Module):
    def __init__(self, hist_dim=45, z=16, v=3):
        super().__init__()
        self.enc = nn.Linear(hist_dim, z * 2 + v * 2)
        self.z = z
        self.v = v

    def encode(self, h):
        o = self.enc(h)
        return o[:, : self.z], o[:, self.z : self.z * 2], o[:, self.z * 2 : self.z * 2 + self.v], o[:, -self.v :]


class _FakeDreamWaQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.vae = _FakeVAE()
        self.actor = nn.Sequential(nn.Linear(45 + 16 + 3, 12))

    def act_inference(self, obs, history):
        from legged_gym.scripts.eval.probe_adapters.dreamwaq import encode_posterior_mean, actor_from_parts
        lm, vm = encode_posterior_mean(self, history)
        return actor_from_parts(self, obs, lm, vm)


class _FakeHIMEst(nn.Module):
    def __init__(self, hist=45 * 5, z=16):
        super().__init__()
        self.net = nn.Linear(hist, 3 + z)
        self.z = z

    def encode(self, h):
        o = self.net(h)
        vel, z = o[..., :3], o[..., 3:]
        z = torch.nn.functional.normalize(z, dim=-1, p=2)
        return vel, z

    def get_latent(self, h):
        v, z = self.encode(h)
        return v.detach(), z.detach()


class _FakeHIM(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_one_step_obs = 45
        self.estimator = _FakeHIMEst()
        self.actor = nn.Sequential(nn.Linear(45 + 3 + 16, 12))

    def act_inference(self, obs_history):
        from legged_gym.scripts.eval.probe_adapters.him import encode_him, actor_from_parts
        vel, z = encode_him(self, obs_history)
        return actor_from_parts(self, obs_history, vel, z)


class _FakeRMA(nn.Module):
    def __init__(self):
        super().__init__()
        self.history_encoder_type = "MLP"
        self.privilege_encoder = nn.Sequential(nn.Linear(8, 8))
        self.history_encoder = nn.Sequential(nn.Linear(45 * 2, 8))
        self.actor = nn.Sequential(nn.Linear(45 + 8, 12))

    def act_student(self, obs, history):
        z = self.history_encoder(history)
        return self.actor(torch.cat([obs, z], dim=-1))

    def act_teacher(self, obs, priv):
        z = self.privilege_encoder(priv)
        return self.actor(torch.cat([obs, z], dim=-1))


class TestAdapters(unittest.TestCase):
    def test_swap_only_replaces_latent(self):
        n, d = 8, 16
        own = torch.randn(n, d)
        bank = torch.randn(n, d)
        idx = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6])
        out = logic.swap_implicit_latent(own, bank, idx)
        torch.testing.assert_close(out, bank[idx])

    def test_within_cross_donors(self):
        mass = np.repeat(np.array([-2.0, 0.0, 3.0, 5.0]), 4)
        within, cross = logic.make_within_and_cross_donors(mass, seed=0)
        for i in range(len(mass)):
            self.assertTrue(np.isclose(mass[within[i]], mass[i]))
            # cross should be opposite-end mass when the group exists
            opp = logic.opposite_mass_map(float(mass[i]))
            self.assertTrue(
                np.isclose(mass[cross[i]], opp),
                f"i={i} got {mass[cross[i]]} want {opp}",
            )
            if len(np.where(np.isclose(mass, mass[i]))[0]) > 1:
                self.assertNotEqual(int(within[i]), i)

    def test_donors_respect_custom_grid_and_valid_mask(self):
        grid = [-1.0, 2.0]
        mass = np.array([-1.0, -1.0, 2.0, 2.0])
        within, cross = logic.make_within_and_cross_donors(
            mass, seed=0, mass_grid=grid
        )
        for i in range(len(mass)):
            self.assertTrue(np.isclose(mass[within[i]], mass[i]))
            self.assertTrue(np.isclose(mass[cross[i]], float(logic.opposite_mass_map(mass[i], grid))))
        # mark env 0 invalid → within/cross should avoid it when alternatives exist
        valid = np.array([False, True, True, True])
        within2, cross2 = logic.make_within_and_cross_donors(
            mass, seed=1, mass_grid=grid, valid_mask=valid
        )
        # receivers that can pick another -1 donor should not pick 0
        self.assertNotEqual(int(within2[1]), 0)
        # mass=2 receivers should only cross-swap to valid -1 donor (env 1)
        self.assertEqual(int(cross2[2]), 1)
        self.assertEqual(int(cross2[3]), 1)

    def test_dreamwaq_swap_keeps_vel_path(self):
        # Import via package path — may need PYTHONPATH
        sys.path.insert(0, str(REPO))
        from legged_gym.scripts.eval.probe_adapters.dreamwaq import (
            DreamWaQAdapter, encode_posterior_mean, actor_from_parts,
        )
        ac = _FakeDreamWaQ()
        ac.eval()
        n = 8
        obs = torch.randn(n, 45)
        hist = torch.randn(n, 45)
        mass = np.repeat([-2.0, 0.0, 3.0, 5.0], 2)
        within, cross = logic.make_within_and_cross_donors(mass, seed=1)
        adapter = DreamWaQAdapter()
        state = {"obs": obs, "history": hist}
        lat = adapter.extract_latent(ac, state)
        # decode features must be latent_mu only (no vel)
        feat = adapter.decode_features(lat)
        self.assertEqual(tuple(feat.shape), (n, 16))
        donors = {
            "within_idx": torch.as_tensor(within),
            "cross_idx": torch.as_tensor(cross),
            "latent_bank": lat["latent_mu"],
        }
        a_n = adapter.act_normal(ac, state)
        a_c = adapter.act_control(ac, state, donors)
        a_w = adapter.act_wrong(ac, state, donors)
        self.assertEqual(a_n.shape, (n, 12))
        self.assertEqual(a_c.shape, (n, 12))
        self.assertEqual(a_w.shape, (n, 12))
        # within/cross should differ from normal when donors differ (not always if latent similar)
        # stronger check: manually swapped latent with fixed own vel equals adapter output
        lm, vm = encode_posterior_mean(ac, hist)
        swapped = logic.swap_implicit_latent(lm, lm, torch.as_tensor(cross))
        manual = actor_from_parts(ac, obs, swapped, vm)
        torch.testing.assert_close(a_w, manual)

    def test_him_swap_keeps_vel(self):
        sys.path.insert(0, str(REPO))
        from legged_gym.scripts.eval.probe_adapters.him import HIMAdapter, actor_from_parts, encode_him
        ac = _FakeHIM()
        ac.eval()
        n = 8
        hist = torch.randn(n, 45 * 5)
        mass = np.repeat([-2.0, 0.0, 3.0, 5.0], 2)
        within, cross = logic.make_within_and_cross_donors(mass, seed=2)
        adapter = HIMAdapter()
        state = {"obs": hist, "obs_history": hist}
        lat = adapter.extract_latent(ac, state)
        self.assertEqual(tuple(adapter.decode_features(lat).shape), (n, 16))
        donors = {
            "within_idx": torch.as_tensor(within),
            "cross_idx": torch.as_tensor(cross),
            "latent_bank": lat["z"],
        }
        a_w = adapter.act_wrong(ac, state, donors)
        vel, z = encode_him(ac, hist)
        swapped = logic.swap_implicit_latent(z, z, torch.as_tensor(cross))
        manual = actor_from_parts(ac, hist, vel, swapped)
        torch.testing.assert_close(a_w, manual)

    def test_rma_student_z_swap_is_primary_use_path(self):
        """Primary control/wrong must swap student z_s, not teacher privilege."""
        sys.path.insert(0, str(REPO))
        from legged_gym.scripts.eval.probe_adapters.rma import (
            RMAAdapter, actor_from_student_z, student_latent,
        )
        ac = _FakeRMA()
        ac.eval()
        n = 4
        obs = torch.randn(n, 45)
        hist = torch.randn(n, 90)
        priv = torch.randn(n, 8)
        real_mass = torch.tensor([-2.0, 0.0, 3.0, 5.0])
        adapter = RMAAdapter()
        self.assertEqual(adapter.use_test_kind, "student_latent_swap")
        state = {
            "obs": obs, "history": hist, "priv_obs": priv,
            "real_mass_raw": real_mass,
        }
        lat = adapter.extract_latent(ac, state)
        self.assertEqual(tuple(lat["z_s"].shape), (n, 8))
        mass = real_mass.numpy()
        within, cross = logic.make_within_and_cross_donors(mass, seed=0)
        donors = {
            "within_idx": torch.as_tensor(within),
            "cross_idx": torch.as_tensor(cross),
            "latent_bank": lat["z_s"],
        }
        a_s = adapter.act_normal(ac, state)
        a_c = adapter.act_control(ac, state, donors)
        a_w = adapter.act_wrong(ac, state, donors)
        self.assertEqual(a_s.shape, (n, 12))
        # control/wrong must match manual student z swap
        z = student_latent(ac, hist)
        manual_w = actor_from_student_z(
            ac, obs, logic.swap_implicit_latent(z, z, torch.as_tensor(cross))
        )
        torch.testing.assert_close(a_w, manual_w)
        # teacher diagnostic remains available but is NOT act_control
        a_tc = adapter.act_teacher_correct(ac, state)
        a_tw = adapter.act_teacher_wrong(ac, state)
        self.assertEqual(a_tc.shape, (n, 12))
        self.assertFalse(torch.allclose(a_tc, a_tw, atol=1e-6))

    def test_itt_fall_penalty_keeps_fallen_envs(self):
        # env0: never falls, mean err 0.2 over 10 steps
        # env1: falls after 2 steps of 0.5 err → penalty 2.0 for remaining 8
        err_sum = np.array([2.0, 1.0])  # 10*0.2, 2*0.5
        alive = np.array([10.0, 2.0])
        fell = np.array([False, True])
        out = logic.apply_fall_penalty_itt(err_sum, alive, fell, n_steps=10, penalty=2.0)
        self.assertAlmostEqual(out[0], 0.2, places=5)
        # (1.0 + 8*2.0) / 10 = 1.7
        self.assertAlmostEqual(out[1], 1.7, places=5)

    def test_traj_level_mass_shuffle(self):
        traj = np.array([0, 0, 1, 1, 2, 2])
        mass = np.array([0.0, 0.0, 3.0, 3.0, 5.0, 5.0])
        shuf = logic.shuffle_labels_by_trajectory(mass, traj, seed=0)
        # constant within traj
        self.assertEqual(shuf[0], shuf[1])
        self.assertEqual(shuf[2], shuf[3])
        # not identity with high probability
        self.assertFalse(np.allclose(shuf, mass) and True)
        # still uses original mass levels
        self.assertTrue(set(shuf.tolist()).issubset(set(mass.tolist())))

    def test_classify_used_via_fall_requires_ci(self):
        # fall alone without CI_lo > 0 is NOT enough
        self.assertEqual(
            logic.classify_result(
                0.8, 0.0, 0.0, -0.1, delta_fall=0.2, delta_fall_ci_lo=-0.01
            ),
            "öğrendi ama kullanmadı",
        )
        # with positive fall CI and effect size → used
        self.assertEqual(
            logic.classify_result(
                0.8, 0.0, 0.0, -0.1, delta_fall=0.2, delta_fall_ci_lo=0.05
            ),
            "öğrendi ve kullandı",
        )

    def test_fall_paired_bootstrap_ci(self):
        # wrong falls more often → positive Δfall with CI > 0
        rng = np.random.default_rng(0)
        control_f = rng.binomial(1, 0.1, size=200).astype(np.float64)
        wrong_f = np.clip(control_f + rng.binomial(1, 0.4, size=200), 0, 1).astype(
            np.float64
        )
        err = rng.normal(0.3, 0.05, size=200)
        res = logic.aggregate_use_test(
            err, err, err + 0.01,
            control_fall=control_f, wrong_fall=wrong_f,
            normal_fall=control_f, seed=0, n_bootstrap=500,
        )
        self.assertGreater(res.delta_fall, 0.05)
        self.assertGreater(res.delta_fall_ci_lo, 0.0)


# ---------------------------------------------------------------------------
# Offline analyze path + CLI help
# ---------------------------------------------------------------------------


class TestAnalyzeAndCLI(unittest.TestCase):
    def test_analyze_samples_end_to_end(self):
        sys.path.insert(0, str(REPO))
        from legged_gym.scripts.eval.probe_physics_use import analyze_samples

        rng = np.random.default_rng(0)
        n_traj = 32
        steps = 8
        d = 8
        traj = np.repeat(np.arange(n_traj), steps)
        mass_levels = np.array([-2.0, 0.0, 3.0, 5.0])
        traj_mass = mass_levels[np.arange(n_traj) % 4]
        mass = traj_mass[traj]
        z = rng.normal(size=(len(traj), d))
        z[:, 0] = mass / 5.0
        z_t = z + 0.01 * rng.normal(size=z.shape)
        samples = {
            "z": z.astype(np.float32),
            "z_t": z_t.astype(np.float32),
            "mass": mass.astype(np.float32),
            "traj_id": traj.astype(np.int64),
            # mass_targets_per_env indexed by traj_id 0..n_traj-1
            "mass_targets_per_env": traj_mass.astype(np.float32),
        }

        use = {
            "normal_err": 0.25,
            "control_err": 0.24,
            "wrong_err": 0.40,
            "delta_use": 0.16,
            "delta_use_ci_lo": 0.10,
            "delta_fall": 0.0,
            "delta_fall_ci_lo": -0.01,
            "control_fall": 0.0,
            "wrong_fall": 0.0,
        }
        result = analyze_samples(samples, method="rma", seed_label="1", seed=0, use_result=use)
        self.assertGreater(result["decode"]["r2"], 0.8)
        self.assertLess(result["decode"]["shuffled_r2"], 0.35)
        self.assertEqual(result["row"]["result"], "öğrendi ve kullandı")
        self.assertIn("Mass R²", result["table_md"])
        self.assertEqual(result["row"]["use_test_kind"], "student_latent_swap")

    def test_cli_help(self):
        script = EVAL / "probe_physics_use.py"
        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True, text=True, env=_env_with_sim(), cwd=str(REPO),
            timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Added-mass", proc.stdout)
        self.assertIn("--method", proc.stdout)
        self.assertIn("--mass_grid", proc.stdout)
        self.assertIn("vy", proc.stdout.lower() + proc.stdout)  # docs mention lateral

    def test_default_command_is_lateral(self):
        self.assertEqual(logic.LATERAL_CMD, (0.0, 1.0, 0.0))
        self.assertEqual(logic.MASS_GRID_KG, (-2.0, 0.0, 3.0, 5.0))

    def test_offline_cli_analyze(self):
        rng = np.random.default_rng(1)
        n_traj = 24
        steps = 6
        traj = np.repeat(np.arange(n_traj), steps)
        mass_levels = np.array([-2.0, 0.0, 3.0, 5.0])
        traj_mass = mass_levels[np.arange(n_traj) % 4]
        mass = traj_mass[traj]
        z = rng.normal(size=(len(traj), 8))
        z[:, 0] = mass / 5.0
        with tempfile.TemporaryDirectory() as td:
            sp = os.path.join(td, "samples.npz")
            np.savez(
                sp,
                z=z.astype(np.float32),
                mass=mass.astype(np.float32),
                traj_id=traj.astype(np.int64),
                mass_targets_per_env=traj_mass.astype(np.float32),
            )
            up = os.path.join(td, "use.npz")
            np.savez(
                up,
                normal_err=0.2,
                control_err=0.2,
                wrong_err=0.35,
                delta_use=0.15,
                delta_use_ci_lo=0.05,
                delta_fall=0.0,
                control_fall=0.0,
                wrong_fall=0.0,
            )
            out = os.path.join(td, "out")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(EVAL / "probe_physics_use.py"),
                    "--analyze_only",
                    "--samples", sp,
                    "--use_npz", up,
                    "--method", "rma",
                    "--seed_label", "1",
                    "--out_dir", out,
                ],
                capture_output=True, text=True, env=_env_with_sim(), cwd=str(REPO),
                timeout=120,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            result_path = os.path.join(out, "probe_result.json")
            self.assertTrue(os.path.isfile(result_path))
            with open(result_path) as f:
                data = json.load(f)
            self.assertIn("row", data)
            self.assertGreater(data["row"]["mass_r2"], 0.5)


class TestStaticContractInSource(unittest.TestCase):
    """Structural checks on shipped collection path (no sim)."""

    def test_runner_calls_physics_contract_and_live_mass(self):
        src = (EVAL / "probe_physics_use.py").read_text()
        self.assertIn("apply_probe_physics_contract", src)
        self.assertIn("_read_live_mass", src)
        self.assertIn("check_mass_invariant", src)
        self.assertIn("command_vy", src)
        self.assertIn("resample_physics", (EVAL / "probe_physics_logic.py").read_text())
        # must not rely solely on cached P5_raw key as mass label in new path
        self.assertNotIn("p5_raw_active", src)
        self.assertIn("mask_valid_measurement", src)
        self.assertIn("make_within_and_cross_donors", src)

    def test_him_step_contract_documented(self):
        """HIM uses LeggedRobot 5-tuple; runner must take rew/dones from [2]/[3]."""
        src = (EVAL / "probe_physics_use.py").read_text()
        self.assertIn('adapter_name == "HIM"', src)
        # wrong indexing would use out[1] as rew (that's priv)
        compact = "".join(src.split())
        self.assertIn("out[0],out[2],out[3],out[4]", compact)


if __name__ == "__main__":
    unittest.main()
