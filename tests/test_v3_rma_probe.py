"""
FILE 4 — tests/test_v3_rma_probe.py
CPU-only, no env / GPU import required.

Tests:
 1. dr_normalize round-trip and OOD clamp.
 2. group_kfold_splits: physics_combo_id disjointness across folds.
 3. latent_R2 identities: z_s==z_t → 1.0; z_s==mean(z_t) → ~0.0.
 4. compute_decoder_metrics shapes and a linearly-encoded synthetic target
    recovering R2≈1 with ridge.
 5. Config YAML loads and has required top-level keys.

Run with:
    cd /home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex
    ./.venv/bin/python -m pytest tests/test_v3_rma_probe.py -q
  or:
    ./.venv/bin/python -m unittest tests/test_v3_rma_probe.py -v
"""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Import probe_rma_analyze DIRECTLY by file path to bypass legged_gym/__init__.py
# (which requires SIMULATOR env var and GPU imports).
# This satisfies the spec requirement that analyze.py imports without legged_gym.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
_ANALYZE_PATH = REPO_ROOT / "legged_gym" / "scripts" / "eval" / "probe_rma_analyze.py"

_spec   = importlib.util.spec_from_file_location("probe_rma_analyze", str(_ANALYZE_PATH))
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

dr_normalize          = _module.dr_normalize
dr_unnormalize        = _module.dr_unnormalize
enter_band_time       = _module.enter_band_time
group_kfold_splits    = _module.group_kfold_splits
latent_r2             = _module.latent_r2
latent_mse            = _module.latent_mse
compute_decoder_metrics = _module.compute_decoder_metrics
load_samples          = _module.load_samples
run_analysis          = _module.run_analysis

# ── Helper: build a minimal samples.npz ─────────────────────────────────────

def _make_samples_npz(path: str, N: int = 240, seed: int = 0):
    """
    Creates a synthetic samples.npz with the full schema required by probe_rma_analyze.
    Physics_combo_id is assigned so that there are multiple distinct groups
    (4 unique groups = axis_code*100 + value_index, with 4 combos).
    The z_s is a noisy copy of z_t so latent_R2 should be high.
    A linear target is embedded in z_s[:,0] so that the ridge decoder
    should find R2≈1 for that target.
    """
    rng = np.random.default_rng(seed)
    # 4 physics combos, 60 samples each
    n_combos = 4
    per_combo = N // n_combos
    physics_combo_id = np.repeat(np.array([0, 1, 2, 3]), per_combo).astype(np.int32)
    axis_code  = (physics_combo_id // 100 % 5).astype(np.int32)   # all 0 (friction)
    command_id = rng.integers(0, 6, size=N).astype(np.int32)

    z_t = rng.standard_normal((N, 8)).astype(np.float32)
    z_s = z_t + 0.05 * rng.standard_normal((N, 8)).astype(np.float32)

    # P5_raw: the added_mass target is a linear function of z_s[:,0]
    P5_raw       = rng.uniform(-1, 1, size=(N, 5)).astype(np.float32)
    # Plant a recoverable linear signal in added_mass (col 1) from z_s[:,0]
    P5_raw[:, 1] = (z_s[:, 0] * 2.0 + 0.5).astype(np.float32)

    P5_norm      = np.clip(2 * (P5_raw - np.array([0.5,  -2, -0.08, -0.08, -0.08]))
                           / np.array([0.75,  7,  0.16,  0.16,  0.16]) - 1, -1, 1).astype(np.float32)
    vel_raw      = rng.uniform(-1, 1, size=(N, 3)).astype(np.float32)
    teacher_action = rng.standard_normal((N, 12)).astype(np.float32)
    student_action = teacher_action + 0.1 * rng.standard_normal((N, 12)).astype(np.float32)
    obs          = rng.standard_normal((N, 45)).astype(np.float32)
    command      = rng.uniform(-1, 1, size=(N, 3)).astype(np.float32)
    fall         = rng.choice([False, True], size=N, p=[0.95, 0.05])
    tracking_lin_err = np.abs(rng.standard_normal(N)).astype(np.float32)
    tracking_ang_err = np.abs(rng.standard_normal(N)).astype(np.float32)
    step         = np.tile(np.arange(per_combo), n_combos).astype(np.int32)

    np.savez(
        path,
        P5_raw=P5_raw,
        P5_norm=P5_norm,
        vel_raw=vel_raw,
        z_t=z_t,
        z_s=z_s,
        teacher_action=teacher_action,
        student_action=student_action,
        obs=obs,
        command=command,
        command_id=command_id,
        axis_code=axis_code,
        physics_combo_id=physics_combo_id,
        fall=fall,
        tracking_lin_err=tracking_lin_err,
        tracking_ang_err=tracking_ang_err,
        step=step,
    )


# ═══════════════════════════════════════════════════════════════════════════
class TestDrNormalize(unittest.TestCase):

    def test_lo_maps_to_minus_one(self):
        lo, hi = -2.0, 5.0
        result = dr_normalize(np.array([-2.0]), lo, hi)
        self.assertAlmostEqual(float(result[0]), -1.0, places=6)

    def test_hi_maps_to_plus_one(self):
        lo, hi = -2.0, 5.0
        result = dr_normalize(np.array([5.0]), lo, hi)
        self.assertAlmostEqual(float(result[0]),  1.0, places=6)

    def test_midpoint_maps_to_zero(self):
        lo, hi = 0.5, 1.25
        mid = (lo + hi) / 2.0
        result = dr_normalize(np.array([mid]), lo, hi)
        self.assertAlmostEqual(float(result[0]), 0.0, places=6)

    def test_round_trip(self):
        lo, hi = -0.08, 0.08
        x = np.linspace(lo, hi, 20)
        normed     = dr_normalize(x, lo, hi)
        x_back     = dr_unnormalize(normed, lo, hi)
        np.testing.assert_allclose(x_back, x, atol=1e-6)

    def test_ood_clamp_positive(self):
        """mass +8 kg is OOD for range [-2, 5] → should saturate to +1."""
        lo, hi = -2.0, 5.0
        result = dr_normalize(np.array([8.0]), lo, hi)
        self.assertAlmostEqual(float(result[0]), 1.0, places=6)

    def test_ood_clamp_negative(self):
        """mass -5 kg is OOD for range [-2, 5] → should saturate to -1."""
        lo, hi = -2.0, 5.0
        result = dr_normalize(np.array([-5.0]), lo, hi)
        self.assertAlmostEqual(float(result[0]), -1.0, places=6)

    def test_array_shape_preserved(self):
        x = np.array([[0.5, 0.75], [0.9, 1.25]])
        out = dr_normalize(x, 0.5, 1.25)
        self.assertEqual(out.shape, x.shape)


# ═══════════════════════════════════════════════════════════════════════════
class TestGroupKFold(unittest.TestCase):

    def _make_groups(self, n_combos: int = 8, per_combo: int = 30):
        return np.repeat(np.arange(n_combos), per_combo)

    def test_train_test_groups_disjoint(self):
        groups = self._make_groups(n_combos=8, per_combo=30)
        for train_idx, test_idx in group_kfold_splits(groups, n_splits=5):
            tr_groups = set(groups[train_idx].tolist())
            te_groups = set(groups[test_idx].tolist())
            self.assertTrue(
                tr_groups.isdisjoint(te_groups),
                f"Overlap: {tr_groups & te_groups}"
            )

    def test_all_samples_covered(self):
        """Every sample should appear in exactly one test fold."""
        groups = self._make_groups(n_combos=6, per_combo=20)
        covered = np.zeros(len(groups), dtype=bool)
        for _, test_idx in group_kfold_splits(groups, n_splits=5):
            # No double-counting
            self.assertFalse(covered[test_idx].any(), "Some test indices already covered")
            covered[test_idx] = True
        self.assertTrue(covered.all(), "Not all samples appeared in a test fold")

    def test_n_splits_respected(self):
        groups = self._make_groups(n_combos=6, per_combo=10)
        folds = list(group_kfold_splits(groups, n_splits=5))
        self.assertLessEqual(len(folds), 5)
        self.assertGreaterEqual(len(folds), 2)

    def test_fewer_groups_than_splits(self):
        """With n_combos=3 and n_splits=5, should still work with ≤3 folds."""
        groups = self._make_groups(n_combos=3, per_combo=10)
        folds  = list(group_kfold_splits(groups, n_splits=5))
        self.assertLessEqual(len(folds), 3)
        self.assertGreaterEqual(len(folds), 1)

    def test_single_group_yields_nothing(self):
        groups = np.zeros(50, dtype=int)
        folds  = list(group_kfold_splits(groups, n_splits=5))
        self.assertEqual(len(folds), 0)

    def test_synthetic_disjointness_with_dict(self):
        """Simulate the structure expected in compute_decoder_metrics."""
        N = 200
        physics_combo_id = np.repeat(np.arange(10), 20)
        for tr, te in group_kfold_splits(physics_combo_id, n_splits=5):
            self.assertTrue(
                set(physics_combo_id[tr]).isdisjoint(set(physics_combo_id[te]))
            )


# ═══════════════════════════════════════════════════════════════════════════
class TestLatentR2(unittest.TestCase):

    def test_identical_latents_gives_one(self):
        rng = np.random.default_rng(7)
        z   = rng.standard_normal((500, 8))
        self.assertAlmostEqual(latent_r2(z, z), 1.0, places=6)

    def test_mean_predictor_gives_near_zero(self):
        rng = np.random.default_rng(7)
        z_t = rng.standard_normal((500, 8))
        z_s = np.full_like(z_t, z_t.mean())
        r2  = latent_r2(z_s, z_t)
        self.assertAlmostEqual(r2, 0.0, delta=0.05)

    def test_noisy_latent_gives_high_r2(self):
        rng = np.random.default_rng(7)
        z_t = rng.standard_normal((1000, 8))
        z_s = z_t + 0.01 * rng.standard_normal((1000, 8))
        r2  = latent_r2(z_s, z_t)
        self.assertGreater(r2, 0.99)

    def test_orthogonal_latent_gives_negative_r2(self):
        rng = np.random.default_rng(7)
        z_t = rng.standard_normal((500, 8))
        z_s = rng.standard_normal((500, 8))  # independent
        r2  = latent_r2(z_s, z_t)
        # Should be well below 1; typically negative or near 0
        self.assertLess(r2, 0.5)

    def test_constant_z_t(self):
        """When Var(z_t)~0 and z_s==z_t, R2=1."""
        z = np.ones((100, 8))
        self.assertAlmostEqual(latent_r2(z, z), 1.0, places=6)

    def test_latent_mse_identity(self):
        rng = np.random.default_rng(3)
        z_t = rng.standard_normal((200, 8))
        z_s = z_t + 0.5 * rng.standard_normal((200, 8))
        mse = latent_mse(z_s, z_t)
        self.assertGreater(mse, 0.0)
        # latent_R2 = 1 - mse / var(z_t)
        var_zt = float(np.var(z_t))
        expected_r2 = 1.0 - mse / var_zt
        self.assertAlmostEqual(latent_r2(z_s, z_t), expected_r2, places=10)


# ═══════════════════════════════════════════════════════════════════════════
class TestDecoderMetrics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        N   = 200
        # 5 physics combos, 40 samples each
        cls.groups = np.repeat(np.arange(5), 40)
        cls.X      = rng.standard_normal((N, 8))
        # Linear target: y = 2*X[:,0] - 0.5*X[:,1] + noise
        cls.y_lin  = 2.0 * cls.X[:, 0] - 0.5 * cls.X[:, 1] + 0.01 * rng.standard_normal(N)
        cls.y_rand = rng.standard_normal(N)

    def test_result_has_required_keys(self):
        res = compute_decoder_metrics(self.X, self.y_rand, self.groups, n_splits=3)
        for decoder in ("linear", "nonlinear"):
            self.assertIn(decoder, res)
            for key in ("R2", "nRMSE", "MAE", "spearman_rho", "calib_slope", "calib_intercept"):
                self.assertIn(key, res[decoder], f"Missing {key} in {decoder}")

    def test_oof_arrays_present(self):
        res = compute_decoder_metrics(self.X, self.y_rand, self.groups, n_splits=3)
        self.assertIn("y_true_oof",       res)
        self.assertIn("y_pred_linear",    res)
        self.assertIn("y_pred_nonlinear", res)
        self.assertIn("idx_oof",          res)
        self.assertEqual(len(res["y_true_oof"]), len(res["y_pred_linear"]))

    def test_r2_bounded(self):
        """R2 must be ≤ 1.0 (can be negative)."""
        res = compute_decoder_metrics(self.X, self.y_rand, self.groups, n_splits=3)
        for decoder in ("linear", "nonlinear"):
            r2 = res[decoder]["R2"]
            if not np.isnan(r2):
                self.assertLessEqual(r2, 1.0 + 1e-6)

    def test_linear_target_recovers_high_r2_with_ridge(self):
        """A linear target embedded in X should give R2 close to 1 for the ridge decoder."""
        res = compute_decoder_metrics(self.X, self.y_lin, self.groups, n_splits=5)
        r2  = res["linear"]["R2"]
        self.assertGreater(r2, 0.95,
                           f"Ridge R2={r2:.4f} on linear target; expected >0.95")

    def test_shuffled_label_gives_low_r2(self):
        """Randomly permuted y should give R2 near 0 or negative."""
        rng  = np.random.default_rng(99)
        perm = rng.permutation(len(self.y_lin))
        y_shuffled = self.y_lin[perm]
        res  = compute_decoder_metrics(self.X, y_shuffled, self.groups, n_splits=5)
        r2   = res["linear"]["R2"]
        self.assertLess(r2, 0.3,
                        f"Shuffled-label R2={r2:.4f}; expected < 0.3")

    def test_single_group_returns_nan(self):
        """When all samples share one group, metrics should be NaN."""
        X = np.random.randn(50, 4)
        y = np.random.randn(50)
        g = np.zeros(50, dtype=int)  # single group
        res = compute_decoder_metrics(X, y, g, n_splits=5)
        self.assertTrue(np.isnan(res["linear"]["R2"]))

    def test_nrmse_non_negative(self):
        res = compute_decoder_metrics(self.X, self.y_rand, self.groups, n_splits=3)
        for decoder in ("linear", "nonlinear"):
            nrmse = res[decoder]["nRMSE"]
            if not np.isnan(nrmse) and not np.isinf(nrmse):
                self.assertGreaterEqual(nrmse, 0.0)

    def test_mae_non_negative(self):
        res = compute_decoder_metrics(self.X, self.y_rand, self.groups, n_splits=3)
        for decoder in ("linear", "nonlinear"):
            mae = res[decoder]["MAE"]
            if not np.isnan(mae):
                self.assertGreaterEqual(mae, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
class TestEnterBandTime(unittest.TestCase):

    def test_already_in_band_at_t0(self):
        decoded = np.array([4.0, 4.1, 3.9, 4.0, 4.0])
        first, dwell = enter_band_time(decoded, target=4.0, band_frac=0.20, t0_idx=0)
        self.assertEqual(first, 0)
        self.assertGreater(dwell, 0.9)

    def test_never_enters_band(self):
        decoded = np.array([0.0, 0.0, 0.0, 0.0])
        first, dwell = enter_band_time(decoded, target=4.0, band_frac=0.20, t0_idx=0)
        self.assertIsNone(first)
        self.assertEqual(dwell, 0.0)

    def test_enters_band_after_switch(self):
        # Simulate step-function response that reaches target at index 5
        decoded = np.concatenate([np.zeros(5), np.full(10, 4.0)])
        first, dwell = enter_band_time(decoded, target=4.0, band_frac=0.20, t0_idx=0)
        self.assertEqual(first, 5)

    def test_t0_offset_respected(self):
        decoded = np.array([4.0, 0.0, 0.0, 4.0, 4.0])
        # If t0_idx=1, the 4.0 at index 0 should not count
        first, _ = enter_band_time(decoded, target=4.0, band_frac=0.20, t0_idx=1)
        self.assertIsNotNone(first)
        self.assertGreaterEqual(first, 1)


# ═══════════════════════════════════════════════════════════════════════════
class TestRunAnalysisSmoke(unittest.TestCase):
    """End-to-end smoke test: build a tiny synthetic npz and run analysis."""

    def test_run_analysis_produces_metrics_json_and_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = os.path.join(tmpdir, "samples.npz")
            _make_samples_npz(npz_path, N=160, seed=1)

            out_dir = os.path.join(tmpdir, "results")
            metrics = run_analysis(
                samples_path=npz_path,
                out_dir=out_dir,
                seed_label="test_seed",
                n_splits=3,
            )

            # probe_metrics.json written
            self.assertTrue(os.path.exists(os.path.join(out_dir, "probe_metrics.json")))
            # report.md written
            self.assertTrue(os.path.exists(os.path.join(out_dir, "report.md")))
            # figures dir created
            self.assertTrue(os.path.isdir(os.path.join(out_dir, "figures")))

            # Check structure
            self.assertIn("latent",   metrics)
            self.assertIn("decoders", metrics)
            lat = metrics["latent"]
            self.assertIn("latent_R2",  lat)
            self.assertIn("action_mae", lat)

    def test_decoder_block_has_expected_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = os.path.join(tmpdir, "samples.npz")
            _make_samples_npz(npz_path, N=160, seed=2)
            out_dir = os.path.join(tmpdir, "results2")
            metrics = run_analysis(npz_path, out_dir, n_splits=3)

            dec = metrics["decoders"]
            for fset in ["z_s", "z_t"]:
                self.assertIn(fset, dec)
                for tname in ["friction", "added_mass", "vx"]:
                    self.assertIn(tname, dec[fset])
                    self.assertIn("linear",    dec[fset][tname])
                    self.assertIn("nonlinear", dec[fset][tname])

    def test_load_samples_raises_on_missing_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = os.path.join(tmpdir, "bad.npz")
            np.savez(bad_path, z_s=np.zeros((10, 8)))  # missing most keys
            with self.assertRaises(KeyError):
                load_samples(bad_path)

    def test_latent_r2_high_in_smoke(self):
        """z_s is a noisy copy of z_t in the synthetic npz → R2 should be high."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = os.path.join(tmpdir, "samples.npz")
            _make_samples_npz(npz_path, N=200)
            metrics = run_analysis(npz_path, tmpdir, n_splits=3)
        r2 = metrics["latent"]["latent_R2"]
        self.assertGreater(r2, 0.95, f"Latent R2={r2:.4f} expected >0.95 for near-identical z_s/z_t")

    def test_added_mass_ridge_r2_high_for_linear_encoding(self):
        """added_mass is linearly encoded in z_s[:,0] → ridge R2 should be high."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = os.path.join(tmpdir, "samples.npz")
            _make_samples_npz(npz_path, N=200)
            metrics = run_analysis(npz_path, tmpdir, n_splits=4)
        r2 = metrics["decoders"]["z_s"]["added_mass"]["linear"]["R2"]
        self.assertGreater(r2, 0.85, f"z_s→added_mass ridge R2={r2:.4f} expected >0.85")


# ═══════════════════════════════════════════════════════════════════════════
class TestConfigYaml(unittest.TestCase):
    """Load v3_rma_latent_probe.yaml and verify required keys are present."""

    def _load_cfg(self):
        import yaml
        cfg_path = REPO_ROOT / "configs" / "eval" / "v3_rma_latent_probe.yaml"
        self.assertTrue(cfg_path.exists(), f"Config not found: {cfg_path}")
        with open(cfg_path) as f:
            return yaml.safe_load(f)

    def test_top_level_keys(self):
        cfg = self._load_cfg()
        for key in ("campaign", "output_root", "task", "seeds", "axes",
                    "commands", "per_point", "steps", "warmup", "stride",
                    "switch", "intervene", "smoke"):
            self.assertIn(key, cfg, f"Missing top-level key: {key}")

    def test_seeds_have_required_fields(self):
        cfg = self._load_cfg()
        for seed in cfg["seeds"]:
            self.assertIn("label",      seed)
            self.assertIn("load_run",   seed)
            self.assertIn("checkpoint", seed)

    def test_both_seeds_present_and_real(self):
        cfg = self._load_cfg()
        labels = [s["label"] for s in cfg["seeds"]]
        self.assertIn("seed_1", labels)
        self.assertIn("seed_2", labels)

    def test_axes_have_grids(self):
        cfg = self._load_cfg()
        for axis_name, ax_cfg in cfg["axes"].items():
            self.assertIn("grid", ax_cfg, f"{axis_name} missing 'grid'")
            self.assertGreater(len(ax_cfg["grid"]), 0)

    def test_switch_block_keys(self):
        cfg = self._load_cfg()
        sw = cfg["switch"]
        for key in ("command_id", "switch_step", "switch_from", "switch_to", "steps"):
            self.assertIn(key, sw, f"switch block missing '{key}'")

    def test_intervene_block_keys(self):
        cfg = self._load_cfg()
        iv = cfg["intervene"]
        for key in ("axis", "grid", "command_id", "modes"):
            self.assertIn(key, iv, f"intervene block missing '{key}'")

    def test_commands_indexed_0_to_5(self):
        cfg = self._load_cfg()
        cmds = cfg["commands"]
        for i in range(6):
            self.assertIn(i, cmds, f"commands[{i}] missing")

    def test_smoke_block_keys(self):
        cfg = self._load_cfg()
        smoke = cfg["smoke"]
        for key in ("per_point", "steps", "warmup"):
            self.assertIn(key, smoke, f"smoke block missing '{key}'")

    def test_task_is_go2_v3_rma(self):
        cfg = self._load_cfg()
        self.assertEqual(cfg["task"], "go2_v3_rma")

    def test_seed_1_load_run_path(self):
        cfg = self._load_cfg()
        seeds = {s["label"]: s for s in cfg["seeds"]}
        self.assertIn("Jul16_15-17-45", seeds["seed_1"]["load_run"])

    def test_seed_2_load_run_path(self):
        cfg = self._load_cfg()
        seeds = {s["label"]: s for s in cfg["seeds"]}
        self.assertIn("Jul16_15-17-46", seeds["seed_2"]["load_run"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
