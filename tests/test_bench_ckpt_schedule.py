"""CPU-only unit tests for benchmark_tripler Phase 0 helpers.

Drives the *shipped* resolution / schedule / headroom code paths (no reimplementation).
No Genesis / GPU required.

Run:
  SIMULATOR=genesis .venv/bin/python -m unittest tests.test_bench_ckpt_schedule -v
"""

from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("SIMULATOR", "genesis")

from legged_gym.scripts.eval.ckpt_utils import (  # noqa: E402
    ckpt_kind,
    normalize_ckpt_spec,
    resolve_checkpoint_path,
    sha256_file,
)
from legged_gym.scripts.eval.headroom import (  # noqa: E402
    FALL_GUARD_ABS,
    PRIMARY_CELLS,
    build_summary_primary,
    checkpoint_direction_flip,
    fall_guard_ok,
    gate_three_seeds,
    median_headroom,
    percent_headroom,
    seed_headrooms,
)
from legged_gym.scripts.eval.schedule_utils import (  # noqa: E402
    DEFAULT_PHASE_STEPS,
    DEFAULT_STEP_VX,
    DEFAULT_STEP_VY,
    DEFAULT_STEP_YAW,
    PHASE_NAMES,
    build_step_schedule,
    phase_command,
)
from legged_gym.utils.helpers import get_load_path  # noqa: E402


class TestNormalizeCkptSpec(unittest.TestCase):
    def test_best_latest_int(self):
        self.assertEqual(normalize_ckpt_spec("best"), "best")
        self.assertEqual(normalize_ckpt_spec("BEST"), "best")
        self.assertEqual(normalize_ckpt_spec("latest"), "latest")
        self.assertEqual(normalize_ckpt_spec(-1), -1)
        self.assertEqual(normalize_ckpt_spec("-1"), -1)
        self.assertEqual(normalize_ckpt_spec(3000), 3000)
        self.assertEqual(normalize_ckpt_spec("3000"), 3000)

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            normalize_ckpt_spec("model_best")
        with self.assertRaises(TypeError):
            normalize_ckpt_spec(3.14)  # type: ignore[arg-type]

    def test_ckpt_kind_labels(self):
        self.assertEqual(ckpt_kind("best"), "best")
        self.assertEqual(ckpt_kind("latest"), "latest")
        self.assertEqual(ckpt_kind(-1), "latest")
        self.assertEqual(ckpt_kind(3000), "3000")


class TestResolveCheckpointPath(unittest.TestCase):
    def _make_run(self, tmp: str) -> str:
        run = os.path.join(tmp, "Jul13_fake_run")
        os.makedirs(run)
        # empty placeholders — resolve only needs file existence
        for name in ("best.pt", "model_1000.pt", "model_3000.pt", "model_500.pt"):
            with open(os.path.join(run, name), "wb") as f:
                f.write(b"fake-ckpt-" + name.encode())
        return run

    def test_best_latest_int_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self._make_run(tmp)
            best = resolve_checkpoint_path(run, "best")
            self.assertTrue(best.endswith("best.pt"))
            self.assertTrue(os.path.isfile(best))

            latest = resolve_checkpoint_path(run, "latest")
            self.assertTrue(latest.endswith("model_3000.pt"), latest)

            latest_neg = resolve_checkpoint_path(run, -1)
            self.assertEqual(latest, latest_neg)

            m3 = resolve_checkpoint_path(run, 3000)
            self.assertTrue(m3.endswith("model_3000.pt"))
            m1 = resolve_checkpoint_path(run, "1000")
            self.assertTrue(m1.endswith("model_1000.pt"))

    def test_missing_file_fail_loud(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = os.path.join(tmp, "empty_run")
            os.makedirs(run)
            with open(os.path.join(run, "model_1.pt"), "wb") as f:
                f.write(b"x")
            with self.assertRaises(FileNotFoundError):
                resolve_checkpoint_path(run, "best")
            with self.assertRaises(FileNotFoundError):
                resolve_checkpoint_path(run, 9999)

    def test_missing_run_dir(self):
        with self.assertRaises(FileNotFoundError):
            resolve_checkpoint_path("/no/such/run/dir", "best")

    def test_get_load_path_uses_resolver(self):
        """Shipped helpers.get_load_path must hit the same resolver for best/int."""
        with tempfile.TemporaryDirectory() as tmp:
            exp = os.path.join(tmp, "go2_benchmark")
            run = self._make_run(exp)
            run_name = os.path.basename(run)
            path = get_load_path(exp, load_run=run_name, checkpoint="best")
            self.assertEqual(os.path.basename(path), "best.pt")
            path3 = get_load_path(exp, load_run=run_name, checkpoint=3000)
            self.assertEqual(os.path.basename(path3), "model_3000.pt")
            path_lat = get_load_path(exp, load_run=run_name, checkpoint=-1)
            self.assertEqual(os.path.basename(path_lat), "model_3000.pt")

    def test_sha256_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "blob.bin")
            with open(p, "wb") as f:
                f.write(b"benchmark-tripler-sha")
            a = sha256_file(p)
            b = sha256_file(p)
            self.assertEqual(a, b)
            self.assertEqual(len(a), 64)


class TestStepSchedule(unittest.TestCase):
    def test_six_phases_defaults(self):
        schedule, bounds = build_step_schedule()
        names = [n for n, _ in bounds]
        self.assertEqual(tuple(names), PHASE_NAMES)
        self.assertEqual(names, list(PHASE_NAMES))
        self.assertEqual(len(bounds), 6)
        self.assertEqual(schedule.shape, (6 * DEFAULT_PHASE_STEPS, 3))

    def test_phase_bounds_and_values(self):
        schedule, bounds = build_step_schedule(
            phase_steps=150, step_vx=1.0, step_vy=0.75, step_yaw=0.75,
        )
        expected = {
            "stand": (0.0, 0.0, 0.0),
            "forward": (1.0, 0.0, 0.0),
            "reverse": (-1.0, 0.0, 0.0),
            "lateral": (0.0, 0.75, 0.0),
            "yaw": (0.0, 0.0, 0.75),  # yaw on channel index 2
            "stop": (0.0, 0.0, 0.0),
        }
        for i, (name, start) in enumerate(bounds):
            self.assertEqual(start, i * 150)
            row = schedule[start]
            self.assertEqual(tuple(row.tolist()), expected[name], name)
            # entire phase constant
            self.assertTrue((schedule[start:start + 150] == row).all())

        # explicit channel-2 check for yaw phase
        yaw_start = dict(bounds)["yaw"]
        self.assertAlmostEqual(float(schedule[yaw_start, 2]), 0.75)
        self.assertAlmostEqual(float(schedule[yaw_start, 0]), 0.0)
        self.assertAlmostEqual(float(schedule[yaw_start, 1]), 0.0)

    def test_phase_command_helper(self):
        self.assertEqual(phase_command("yaw"), (0.0, 0.0, DEFAULT_STEP_YAW))
        self.assertEqual(phase_command("forward"), (DEFAULT_STEP_VX, 0.0, 0.0))
        self.assertEqual(phase_command("lateral"), (0.0, DEFAULT_STEP_VY, 0.0))

    def test_transient_cli_wrapper_uses_builder(self):
        """build_command_schedule(cli) must call the real schedule builder."""
        import types
        from legged_gym.scripts.eval.transient import build_command_schedule

        cli = types.SimpleNamespace(
            phase_steps=150, step_vx=1.0, step_vy=0.75, step_yaw=0.75,
        )
        schedule, bounds = build_command_schedule(cli)
        self.assertEqual([n for n, _ in bounds], list(PHASE_NAMES))
        yaw_start = dict(bounds)["yaw"]
        self.assertEqual(tuple(schedule[yaw_start].tolist()), (0.0, 0.0, 0.75))


class TestHeadroomFormulas(unittest.TestCase):
    def test_percent_headroom(self):
        # 20% improvement: (1.0 - 0.8) / 1.0 * 100
        self.assertAlmostEqual(percent_headroom(1.0, 0.8), 20.0)
        self.assertAlmostEqual(percent_headroom(0.5, 0.5), 0.0)
        self.assertAlmostEqual(percent_headroom(1.0, 1.2), -20.0)
        self.assertTrue(percent_headroom(0.0, 0.1) != percent_headroom(0.0, 0.1)
                        or True)  # non-finite
        import math
        self.assertTrue(math.isnan(percent_headroom(0.0, 0.1)))
        self.assertTrue(math.isnan(percent_headroom(float("nan"), 0.1)))

    def test_seed_headrooms_median(self):
        # Six identical cells: MLP=1.0, P5=0.8, P5V=0.7
        cells = PRIMARY_CELLS
        err_mlp = {c: 1.0 for c in cells}
        err_p5 = {c: 0.8 for c in cells}
        err_p5v = {c: 0.7 for c in cells}
        out = seed_headrooms(err_mlp, err_p5, err_p5v)
        self.assertAlmostEqual(out["H_P5"], 20.0)
        # (0.8-0.7)/0.8 * 100 = 12.5
        self.assertAlmostEqual(out["H_V"], 12.5)
        # (1.0-0.7)/1.0 * 100 = 30
        self.assertAlmostEqual(out["H_total"], 30.0)
        self.assertEqual(len(out["cells"]), 6)

    def test_gates(self):
        self.assertEqual(gate_three_seeds([15.0, 12.0, 20.0]), "pass")
        self.assertEqual(gate_three_seeds([-1.0, -2.0, -0.5]), "early-fail")
        self.assertEqual(gate_three_seeds([5.0, 6.0, 7.0]), "expand-to-seeds-4-5")  # below 10%
        self.assertEqual(
            gate_three_seeds([15.0, 12.0, 20.0], fall_guard_per_seed=[True, False, True]),
            "expand-to-seeds-4-5",
        )
        self.assertEqual(
            gate_three_seeds([15.0, 12.0, 20.0], checkpoint_flips_direction=True),
            "expand-to-seeds-4-5",
        )

    def test_fall_guard(self):
        self.assertTrue(fall_guard_ok(0.10, 0.10))
        self.assertTrue(fall_guard_ok(0.11, 0.10))  # 0.01 <= 0.02
        self.assertFalse(fall_guard_ok(0.13, 0.10))  # 0.03 > 0.02
        self.assertEqual(FALL_GUARD_ABS, 0.02)

    def test_checkpoint_direction_flip(self):
        self.assertFalse(checkpoint_direction_flip([10, 12, 15], [8, 9, 11]))
        self.assertTrue(checkpoint_direction_flip([10, 12, 15], [-5, -3, -1]))

    def test_checkpoint_sensitive_same_sign_gate_change(self):
        """P1 counterexample: best H=15% (pass) vs 3000 H=5% (expand) — both positive.

        Plan: gate decision change => checkpoint_sensitive; final must not stay pass.
        """
        from legged_gym.scripts.eval.headroom import (
            checkpoint_is_sensitive,
            finalize_gate_with_checkpoint,
            gate_three_seeds,
        )

        H_best = [15.0, 15.0, 15.0]
        H_3000 = [5.0, 5.0, 5.0]
        self.assertEqual(gate_three_seeds(H_best), "pass")
        self.assertEqual(gate_three_seeds(H_3000), "expand-to-seeds-4-5")
        # median signs agree (both > 0) — old sign-only check would miss this
        self.assertFalse(checkpoint_direction_flip(H_best, H_3000))
        sens, g_best, g_3000 = checkpoint_is_sensitive(H_best, H_3000)
        self.assertTrue(sens)
        self.assertEqual(g_best, "pass")
        self.assertEqual(g_3000, "expand-to-seeds-4-5")
        self.assertEqual(
            finalize_gate_with_checkpoint(g_best, g_3000, sens),
            "expand-to-seeds-4-5",
        )

        # Via build_summary_primary (A1 falls OK)
        cells = PRIMARY_CELLS
        per_seed = {}
        for s in (1, 2, 3):
            # Fabricate H via seed_headrooms: for P5 gate we need H_P5=15 on best path.
            # build_summary_primary takes H from per_seed and optional H_p5_3000 vectors.
            sh = seed_headrooms(
                {c: 1.0 for c in cells},
                {c: 0.85 for c in cells},  # h_p5 = 15%
                {c: 0.7 for c in cells},
            )
            sh["fall_mlp"] = 0.05
            sh["fall_p5"] = 0.04
            sh["fall_p5v"] = 0.04
            per_seed[s] = sh
        # Override stored H to exact 15 for clarity
        for s in per_seed:
            per_seed[s]["H_P5"] = 15.0
            per_seed[s]["H_V"] = 15.0  # keep V non-sensitive pass on both
            per_seed[s]["H_total"] = 30.0
        summary = build_summary_primary(
            per_seed,
            H_p5_3000=[5.0, 5.0, 5.0],
            H_v_3000=[15.0, 15.0, 15.0],
        )
        self.assertTrue(summary["checkpoint_sensitive_p5"])
        self.assertFalse(summary["checkpoint_sensitive_v"])
        self.assertEqual(summary["gate_p5_best"], "pass")
        self.assertEqual(summary["gate_p5_3000"], "expand-to-seeds-4-5")
        self.assertEqual(summary["gate_p5"], "expand-to-seeds-4-5")
        self.assertTrue(summary["expand_seeds_4_5"])

    def test_build_summary_primary(self):
        cells = PRIMARY_CELLS
        per_seed = {}
        for s in (1, 2, 3):
            sh = seed_headrooms(
                {c: 1.0 for c in cells},
                {c: 0.8 for c in cells},
                {c: 0.7 for c in cells},
            )
            sh["fall_mlp"] = 0.05
            sh["fall_p5"] = 0.04
            sh["fall_p5v"] = 0.04
            per_seed[s] = sh
        summary = build_summary_primary(per_seed)
        self.assertEqual(summary["gate_p5"], "pass")
        self.assertEqual(summary["gate_v"], "pass")
        self.assertFalse(summary["expand_seeds_4_5"])
        self.assertAlmostEqual(summary["median_H_P5"], 20.0)

        # Robust pair: both checkpoints pass → not sensitive, final pass
        summary_r = build_summary_primary(
            per_seed,
            H_p5_3000=[18.0, 20.0, 22.0],
            H_v_3000=[12.0, 13.0, 14.0],
        )
        self.assertFalse(summary_r["checkpoint_sensitive_p5"])
        self.assertEqual(summary_r["gate_p5"], "pass")
        self.assertEqual(summary_r["gate_p5_best"], "pass")
        self.assertEqual(summary_r["gate_p5_3000"], "pass")


class TestMedianHelper(unittest.TestCase):
    def test_median_headroom(self):
        self.assertAlmostEqual(median_headroom([10, 20, 30]), 20.0)


class TestSweepSavePayload(unittest.TestCase):
    """Regression: sweep must not pass duplicate kwargs into atomic_savez."""

    def test_merge_npz_payload_rejects_duplicates(self):
        from legged_gym.scripts.eval.provenance import merge_npz_payload
        with self.assertRaises(ValueError):
            merge_npz_payload({"axis": "added_mass"}, {"axis": "friction"})

    def test_build_sweep_save_payload_no_duplicate_keys(self):
        """Drives shipped build_sweep_save_payload with meta shaped like collect_run_meta."""
        import numpy as np
        from legged_gym.scripts.eval.sweep import build_sweep_save_payload
        from legged_gym.scripts.eval.provenance import atomic_savez, merge_npz_payload

        # Simulate meta that already embeds axis / command / in_dist / unit / steps / per_point
        meta = {
            "task": "go2_bench_mlp",
            "method": "MLP",
            "axis": "added_mass",
            "command_vx": 1.0,
            "command_vy": 0.0,
            "command_yaw": 0.0,
            "in_dist": np.array([-1.0, 1.0]),
            "unit": "kg",
            "steps": 2000,
            "per_point": 256,
            "warmup": 100,
            "ckpt_sha256": "abc",
            "eval_seed": 12345,
        }
        agg = {
            "tracking_lin_err_mean": np.array([0.2, 0.15, 0.18]),
            "fall_rate_mean": np.array([0.0, 0.0, 0.01]),
        }
        payload = build_sweep_save_payload(
            meta, agg, grid=np.array([-1.0, 0.0, 1.0]), label="MLP", seed=12345,
        )
        # every key unique
        self.assertEqual(len(payload), len(set(payload)))
        self.assertIn("axis", payload)
        self.assertIn("grid", payload)
        self.assertIn("tracking_lin_err_mean", payload)
        self.assertEqual(payload["axis"], "added_mass")
        # must be callable as atomic_savez(path, **payload) without TypeError
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sweep_out.npz")
            atomic_savez(path, **payload)
            self.assertTrue(os.path.isfile(path))
            with np.load(path, allow_pickle=True) as z:
                self.assertEqual(str(z["axis"]), "added_mass")
                self.assertEqual(list(z["grid"]), [-1.0, 0.0, 1.0])
                self.assertAlmostEqual(float(z["tracking_lin_err_mean"][1]), 0.15)

        # the OLD broken call shape must be rejected by merge helper
        with self.assertRaises(ValueError):
            merge_npz_payload(
                meta,
                {"axis": meta["axis"], "grid": np.array([-1.0, 0.0, 1.0]),
                 "command_vx": meta["command_vx"]},
            )

    def test_old_duplicate_kwargs_pattern_raises_typeerror(self):
        """Document the P0 failure mode: explicit kwargs + overlapping **meta."""
        def fake_savez(path, **arrays):
            return arrays

        meta = {"axis": "added_mass", "command_vx": 1.0, "steps": 2000, "per_point": 256}
        with self.assertRaises(TypeError):
            fake_savez(
                "out.npz",
                axis="added_mass",
                command_vx=1.0,
                steps=2000,
                per_point=256,
                **meta,
            )


class TestRunIdentity(unittest.TestCase):
    def test_resolve_load_run_fail_loud_cross_task(self):
        from legged_gym.scripts.eval.sweep import resolve_load_run
        with tempfile.TemporaryDirectory() as tmp:
            for name in (
                "Jul13_bench_mlp_genesis_seed1",
                "Jul13_bench_oracle_id_genesis_seed1",
            ):
                os.makedirs(os.path.join(tmp, name))
            with self.assertRaises(ValueError):
                resolve_load_run(
                    tmp, "bench_mlp", "Jul13_bench_oracle_id_genesis_seed1",
                )
            # matching run_name is ok
            chosen = resolve_load_run(
                tmp, "bench_mlp", "Jul13_bench_mlp_genesis_seed1",
            )
            self.assertEqual(chosen, "Jul13_bench_mlp_genesis_seed1")

    def test_verify_run_identity_manifest(self):
        import json
        from legged_gym.scripts.eval.provenance import verify_run_identity

        with tempfile.TemporaryDirectory() as tmp:
            run = os.path.join(tmp, "Jul13_bench_mlp_genesis_seed1")
            os.makedirs(run)
            with open(os.path.join(run, "run_manifest.json"), "w") as f:
                json.dump({"task": "go2_bench_mlp", "training_seed": 1}, f)
            man = verify_run_identity(
                run, expected_task="go2_bench_mlp", expected_run_name="bench_mlp",
                expected_training_seed=1,
            )
            self.assertEqual(man["task"], "go2_bench_mlp")
            with self.assertRaises(ValueError):
                verify_run_identity(
                    run, expected_task="go2_bench_oracle_id",
                    expected_run_name="bench_mlp",
                )
            with self.assertRaises(ValueError):
                verify_run_identity(
                    run, expected_task="go2_bench_mlp",
                    expected_run_name="bench_mlp", expected_training_seed=2,
                )


class TestAggregatePrimary(unittest.TestCase):
    def _write_tree(self, root: str):
        """Minimal synthetic Phase A tree driving the real aggregate CLI."""
        import numpy as np
        from legged_gym.scripts.eval.provenance import atomic_savez

        methods = {
            "mlp": "go2_bench_mlp",
            "p5": "go2_bench_oracle_id",
            "p5v": "go2_bench_oracle_id_vel",
        }
        # err levels so H_P5≈20, H_V≈12.5
        err = {"mlp": 1.0, "p5": 0.8, "p5v": 0.7}
        fall = {"mlp": 0.05, "p5": 0.04, "p5v": 0.04}
        for seed in (1, 2, 3):
            for mk, task in methods.items():
                # A1 indist
                p = os.path.join(root, "best", f"seed_{seed}", "indist", f"{task}.npz")
                os.makedirs(os.path.dirname(p), exist_ok=True)
                atomic_savez(p, task=task, fall_rate=fall[mk], tracking_lin_err=err[mk])
                for tree in ("best", "model_3000"):
                    for vx in (0.75, 1.0):
                        sp = os.path.join(
                            root, tree, f"seed_{seed}", "primary",
                            f"{task}_mass_vx{vx:g}.npz",
                        )
                        os.makedirs(os.path.dirname(sp), exist_ok=True)
                        # slight tree difference for 3000 so direction still positive
                        scale = 1.0 if tree == "best" else 1.02
                        e = err[mk] * scale
                        atomic_savez(
                            sp,
                            grid=np.array([-1.0, 0.0, 1.0]),
                            tracking_lin_err_mean=np.array([e, e, e]),
                            fall_rate_mean=np.array([0.0, 0.0, 0.0]),
                            axis="added_mass",
                            command_vx=vx,
                        )

    def test_aggregate_writes_summary_primary(self):
        import json
        from scripts.benchmark_tripler_aggregate import aggregate, main as agg_main

        with tempfile.TemporaryDirectory() as tmp:
            self._write_tree(tmp)
            summary = aggregate(tmp, require_3000=True)
            path = os.path.join(tmp, "aggregate", "summary_primary.json")
            self.assertTrue(os.path.isfile(path))
            with open(path) as f:
                loaded = json.load(f)
            self.assertEqual(loaded["gate_p5"], "pass")
            self.assertEqual(loaded["gate_v"], "pass")
            self.assertAlmostEqual(loaded["median_H_P5"], 20.0, places=5)
            for name in (
                "primary_cells.csv", "seed_headroom.csv",
                "safety.csv", "checkpoint_sensitivity.csv",
            ):
                self.assertTrue(os.path.isfile(os.path.join(tmp, "aggregate", name)))
            # CLI entry point
            rc = agg_main(["--artifact-root", tmp])
            self.assertEqual(rc, 0)


class TestAtomicSavez(unittest.TestCase):
    """Drives shipped atomic_savez / npz_is_valid (real write+read, no reimplementation)."""

    def test_atomic_savez_writes_readable_npz(self):
        from legged_gym.scripts.eval.provenance import atomic_savez, npz_is_valid
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cell_meta.npz")
            returned = atomic_savez(
                path,
                task="go2_bench_mlp",
                ckpt_sha256="abc123",
                tracking_lin_err=0.42,
                fall_rate=0.01,
                grid=np.array([-1.0, 0.0, 1.0]),
            )
            self.assertEqual(returned, os.path.abspath(path))
            self.assertTrue(os.path.isfile(path))
            # Must not leave the broken path.tmp / path.tmp.npz siblings
            self.assertFalse(os.path.isfile(path + ".tmp"))
            self.assertFalse(os.path.isfile(path + ".tmp.npz"))
            self.assertTrue(npz_is_valid(path, required_keys=("task", "tracking_lin_err", "fall_rate")))
            with np.load(path, allow_pickle=True) as z:
                self.assertEqual(str(z["task"]), "go2_bench_mlp")
                self.assertAlmostEqual(float(z["tracking_lin_err"]), 0.42)
                self.assertEqual(list(z["grid"]), [-1.0, 0.0, 1.0])

    def test_atomic_savez_partial_then_promote(self):
        """Mirrors orchestrator: write *.partial.npz then os.replace to final .npz."""
        from legged_gym.scripts.eval.provenance import atomic_savez, npz_is_valid
        import numpy as np
        from scripts.benchmark_tripler_orchestrator import partial_out_path

        with tempfile.TemporaryDirectory() as tmp:
            final = os.path.join(tmp, "go2_bench_mlp_mass_vx1.npz")
            partial = partial_out_path(final)
            self.assertTrue(partial.endswith(".partial.npz"))
            self.assertNotEqual(partial, final)

            atomic_savez(
                partial,
                tracking_lin_err_mean=np.array([0.1, 0.2, 0.3]),
                fall_rate_mean=np.array([0.0, 0.0, 0.01]),
                ckpt_sha256="deadbeef",
            )
            self.assertTrue(os.path.isfile(partial))
            self.assertFalse(os.path.isfile(final))
            self.assertTrue(npz_is_valid(
                partial, required_keys=("tracking_lin_err_mean", "fall_rate_mean", "ckpt_sha256"),
            ))
            os.replace(partial, final)
            self.assertTrue(os.path.isfile(final))
            self.assertFalse(os.path.isfile(partial))
            self.assertTrue(npz_is_valid(
                final, required_keys=("tracking_lin_err_mean", "fall_rate_mean"),
            ))

    def test_npz_is_valid_rejects_nan_and_missing(self):
        from legged_gym.scripts.eval.provenance import atomic_savez, npz_is_valid
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            good = os.path.join(tmp, "good.npz")
            bad_nan = os.path.join(tmp, "nan.npz")
            atomic_savez(good, tracking_lin_err=0.1, fall_rate=0.0)
            atomic_savez(bad_nan, tracking_lin_err=float("nan"), fall_rate=0.0)
            self.assertTrue(npz_is_valid(good, required_keys=("tracking_lin_err",)))
            self.assertFalse(npz_is_valid(bad_nan, required_keys=("tracking_lin_err",)))
            self.assertFalse(npz_is_valid(good, required_keys=("missing_key",)))
            self.assertFalse(npz_is_valid(os.path.join(tmp, "nope.npz")))


if __name__ == "__main__":
    unittest.main()
