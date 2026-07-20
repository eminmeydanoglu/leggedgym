"""Unit/smoke tests for the base-velocity estimation probe (no full sim)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "legged_gym" / "scripts" / "eval"
os.environ.setdefault("SIMULATOR", "genesis")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


logic = _load("probe_velocity_logic_ut", EVAL / "probe_velocity_logic.py")


def _env():
    e = os.environ.copy()
    e["SIMULATOR"] = "genesis"
    e["PYTHONPATH"] = str(REPO) + os.pathsep + e.get("PYTHONPATH", "")
    return e


# ---------------------------------------------------------------------------
# Traj split
# ---------------------------------------------------------------------------


class TestTrajSplit(unittest.TestCase):
    def test_no_leakage(self):
        traj = np.repeat(np.arange(20), 5)
        split = logic.trajectory_split(traj, seed=0)
        tr = set(traj[split.train_idx].tolist())
        te = set(traj[split.test_idx].tolist())
        self.assertEqual(tr & te, set())
        self.assertGreater(len(tr), 0)
        self.assertGreater(len(te), 0)


# ---------------------------------------------------------------------------
# RMA-style decoder
# ---------------------------------------------------------------------------


class TestVelocityDecoder(unittest.TestCase):
    def test_planted_latent_high_r2_shuffle_low(self):
        rng = np.random.default_rng(0)
        n_traj, steps, d = 32, 8, 8
        traj = np.repeat(np.arange(n_traj), steps)
        # true velocity varies by traj (and a bit within)
        base = rng.normal(size=(n_traj, 3))
        vel = base[traj] + 0.02 * rng.normal(size=(len(traj), 3))
        z = rng.normal(size=(len(traj), d))
        # plant vel into first 3 latent dims
        z[:, :3] = vel * 0.5 + 0.01 * rng.normal(size=(len(traj), 3))

        out = logic.decode_velocity_with_shuffle(z, vel, traj, seed=0, epochs=250)
        self.assertGreater(out["r2_mean"], 0.85, f"R²={out['r2_mean']}")
        self.assertLess(out["shuffled_r2"], 0.35, f"shuffle={out['shuffled_r2']}")
        self.assertIn("vx", out["r2_per_dim"])
        self.assertGreater(out["r2_per_dim"]["vx"], 0.7)


# ---------------------------------------------------------------------------
# Explicit head metrics
# ---------------------------------------------------------------------------


class TestExplicitVelocity(unittest.TestCase):
    def test_perfect_head_high_r2(self):
        rng = np.random.default_rng(1)
        n_traj, steps = 24, 6
        traj = np.repeat(np.arange(n_traj), steps)
        vel = rng.normal(size=(len(traj), 3))
        hat = vel + 0.01 * rng.normal(size=vel.shape)
        m = logic.explicit_velocity_metrics(hat, vel, traj, seed=0)
        self.assertGreater(m["r2_mean"], 0.95)
        self.assertLess(m["shuffled_r2"], 0.2)
        self.assertEqual(m["kind"], "explicit")

    def test_noise_head_low_r2(self):
        rng = np.random.default_rng(2)
        n_traj, steps = 24, 6
        traj = np.repeat(np.arange(n_traj), steps)
        vel = rng.normal(size=(len(traj), 3))
        hat = rng.normal(size=vel.shape)  # unrelated
        m = logic.explicit_velocity_metrics(hat, vel, traj, seed=0)
        self.assertLess(m["r2_mean"], 0.3)


# ---------------------------------------------------------------------------
# Analyze + classify + table
# ---------------------------------------------------------------------------


class TestAnalyze(unittest.TestCase):
    def test_rma_analyze_two_rows(self):
        rng = np.random.default_rng(0)
        n_traj, steps, d = 28, 6, 8
        traj = np.repeat(np.arange(n_traj), steps)
        vel = rng.normal(size=(len(traj), 3))
        z_s = rng.normal(size=(len(traj), d))
        z_s[:, :3] = vel * 0.4
        z_t = z_s + 0.02 * rng.normal(size=z_s.shape)
        samples = {
            "vel_true": vel.astype(np.float32),
            "traj_id": traj.astype(np.int64),
            "z_s": z_s.astype(np.float32),
            "z_t": z_t.astype(np.float32),
        }
        res = logic.analyze_velocity_samples(samples, method="rma", seed_label="1", seed=0)
        self.assertEqual(len(res["rows"]), 2)
        sources = {r["source"] for r in res["rows"]}
        self.assertEqual(sources, {"z_s→v", "z_t→v"})
        self.assertIn("Vel R²", res["table_md"])
        self.assertEqual(res["rows"][0]["result"], "velocity tahmin ediyor")

    def test_dreamwaq_analyze_explicit(self):
        rng = np.random.default_rng(3)
        n_traj, steps = 20, 5
        traj = np.repeat(np.arange(n_traj), steps)
        vel = rng.normal(size=(len(traj), 3))
        samples = {
            "vel_true": vel.astype(np.float32),
            "traj_id": traj.astype(np.int64),
            "vel_hat": (vel + 0.02 * rng.normal(size=vel.shape)).astype(np.float32),
        }
        res = logic.analyze_velocity_samples(
            samples, method="dreamwaq", seed_label="2", seed=0
        )
        self.assertEqual(len(res["rows"]), 1)
        self.assertEqual(res["rows"][0]["source"], "vel_mu")
        self.assertEqual(res["rows"][0]["method"], "DreamWaQ")

    def test_him_analyze_explicit(self):
        rng = np.random.default_rng(4)
        n_traj, steps = 20, 5
        traj = np.repeat(np.arange(n_traj), steps)
        vel = rng.normal(size=(len(traj), 3))
        samples = {
            "vel_true": vel.astype(np.float32),
            "traj_id": traj.astype(np.int64),
            "vel_hat": (vel + 0.02 * rng.normal(size=vel.shape)).astype(np.float32),
        }
        res = logic.analyze_velocity_samples(samples, method="him", seed_label="1", seed=0)
        self.assertEqual(res["rows"][0]["source"], "vel_hat")
        self.assertEqual(res["rows"][0]["method"], "HIM")

    def test_classify_gates(self):
        self.assertEqual(
            logic.classify_velocity_result(0.8, 0.0, n_identifiable=3),
            "velocity tahmin ediyor",
        )
        self.assertEqual(
            logic.classify_velocity_result(0.8, 0.0, n_identifiable=2),
            "identifiable boyutlarda tahmin ediyor",
        )
        self.assertEqual(
            logic.classify_velocity_result(0.2, 0.0), "velocity gösterilemedi"
        )
        self.assertEqual(
            logic.classify_velocity_result(0.9, 0.5), "velocity gösterilemedi"
        )
        self.assertIn(
            "protocol yetersiz",
            logic.classify_velocity_result(0.9, 0.0, n_identifiable=0),
        )

    def test_shuffle_r2_uses_same_identifiable_dims(self):
        # Construct true: high signal on vx/vy, near-constant vz
        rng = np.random.default_rng(0)
        n_traj, steps = 24, 6
        traj = np.repeat(np.arange(n_traj), steps)
        vel = rng.normal(size=(len(traj), 3))
        vel[:, 2] = 0.001  # near-constant vz
        hat = vel.copy()
        hat[:, :2] += 0.01 * rng.normal(size=(len(traj), 2))
        m = logic.explicit_velocity_metrics(hat, vel, traj, seed=0)
        tgt = m["target_std"]
        self.assertFalse(tgt["vz"]["identifiable"])
        # shuffle R² must be computed on same id dims (not dragged by vz alone)
        self.assertIn("shuffled_r2_per_dim", m)
        id_dims = m["identifiable_dims"]
        self.assertNotIn("vz", id_dims)
        # recompute manually
        shuf_manual = float(np.mean([m["shuffled_r2_per_dim"][d] for d in id_dims]))
        self.assertAlmostEqual(m["shuffled_r2"], shuf_manual, places=6)


# ---------------------------------------------------------------------------
# Adapter: explicit vel extracted, RMA has z_s/z_t
# ---------------------------------------------------------------------------


class TestAdapterVelFeatures(unittest.TestCase):
    def test_dreamwaq_extracts_vel_mu(self):
        sys.path.insert(0, str(REPO))
        from legged_gym.scripts.eval.probe_adapters.dreamwaq import DreamWaQAdapter
        import torch.nn as nn

        class VAE(nn.Module):
            def encode(self, h):
                B = h.shape[0]
                return (
                    torch.zeros(B, 16),
                    torch.zeros(B, 16),
                    h[:, :3],  # vel_mu from first 3 of history (for test)
                    torch.zeros(B, 3),
                )

        class AC(nn.Module):
            def __init__(self):
                super().__init__()
                self.vae = VAE()
                self.actor = nn.Linear(45 + 16 + 3, 12)

            def act_inference(self, obs, hist):
                return self.actor(
                    torch.cat([obs, torch.zeros(obs.shape[0], 16), hist[:, :3]], -1)
                )

        ac = AC()
        ad = DreamWaQAdapter()
        state = {"obs": torch.randn(4, 45), "history": torch.randn(4, 45)}
        lat = ad.extract_latent(ac, state)
        self.assertIn("vel_mu", lat)
        self.assertEqual(tuple(lat["vel_mu"].shape), (4, 3))
        # mass decoder features must NOT be vel
        feat = ad.decode_features(lat)
        self.assertEqual(tuple(feat.shape), (4, 16))

    def test_him_extracts_vel_hat(self):
        sys.path.insert(0, str(REPO))
        from legged_gym.scripts.eval.probe_adapters.him import HIMAdapter
        import torch.nn as nn

        class Est(nn.Module):
            def encode(self, h):
                B = h.shape[0]
                z = torch.nn.functional.normalize(torch.randn(B, 16), dim=-1)
                return h[:, :3], z

        class AC(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_one_step_obs = 45
                self.estimator = Est()
                self.actor = nn.Linear(45 + 3 + 16, 12)

            def act_inference(self, h):
                return torch.zeros(h.shape[0], 12)

        ac = AC()
        ad = HIMAdapter()
        h = torch.randn(4, 45 * 5)
        lat = ad.extract_latent(ac, {"obs": h, "obs_history": h})
        self.assertIn("vel_hat", lat)
        self.assertEqual(tuple(lat["vel_hat"].shape), (4, 3))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI(unittest.TestCase):
    def test_help(self):
        proc = subprocess.run(
            [sys.executable, str(EVAL / "probe_velocity.py"), "--help"],
            capture_output=True, text=True, env=_env(), cwd=str(REPO), timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("velocity", proc.stdout.lower())
        self.assertIn("--method", proc.stdout)

    def test_offline_cli(self):
        rng = np.random.default_rng(0)
        n_traj, steps = 20, 5
        traj = np.repeat(np.arange(n_traj), steps)
        vel = rng.normal(size=(len(traj), 3))
        hat = vel + 0.01 * rng.normal(size=vel.shape)
        with tempfile.TemporaryDirectory() as td:
            sp = os.path.join(td, "s.npz")
            np.savez(
                sp,
                vel_true=vel.astype(np.float32),
                traj_id=traj.astype(np.int64),
                vel_hat=hat.astype(np.float32),
            )
            out = os.path.join(td, "out")
            proc = subprocess.run(
                [
                    sys.executable, str(EVAL / "probe_velocity.py"),
                    "--analyze_only", "--samples", sp,
                    "--method", "him", "--seed_label", "1",
                    "--out_dir", out,
                ],
                capture_output=True, text=True, env=_env(), cwd=str(REPO), timeout=120,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            with open(os.path.join(out, "velocity_result.json")) as f:
                data = json.load(f)
            self.assertGreater(data["rows"][0]["r2_mean"], 0.9)

    def test_static_separate_from_mass_probe(self):
        """Velocity probe must not fold into mass table schema."""
        src = (EVAL / "probe_velocity.py").read_text()
        self.assertIn("velocity", src.lower())
        self.assertIn("Separate from the added-mass", src)
        # mass learn+use table keys should not be the primary product
        self.assertNotIn("delta_use", src)
        vlogic = (EVAL / "probe_velocity_logic.py").read_text()
        self.assertIn("z_s→v", vlogic or "z_s")  # source labels in analyzer
        self.assertIn("vel_mu", vlogic)
        self.assertIn("vel_hat", vlogic)
        self.assertIn("DEFAULT_VEL_COMMAND_SCHEDULE", vlogic)

    def test_multi_command_schedule_exists(self):
        sched = logic.DEFAULT_VEL_COMMAND_SCHEDULE
        self.assertGreaterEqual(len(sched), 4)
        # spans both vx and vy signs
        vxs = {c[0] for c in sched}
        vys = {c[1] for c in sched}
        self.assertTrue(any(x > 0 for x in vxs) and any(x < 0 for x in vxs))
        self.assertTrue(any(y > 0 for y in vys) and any(y < 0 for y in vys))


if __name__ == "__main__":
    unittest.main()
