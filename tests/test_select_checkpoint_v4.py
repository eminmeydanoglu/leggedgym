"""CPU-only contracts for the offline v4-terrain best_spnte.pt selector.

Monkeypatches the per-checkpoint eval step (``eval_fn``) with a fake
iteration -> spnte_lin mapping, so none of these tests touch Genesis / GPU
simulation -- they exercise selection, atomic publish, and provenance only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("SIMULATOR", "genesis")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from legged_gym.scripts.eval.select_checkpoint_v4 import (  # noqa: E402
    BEST_SPNTE_NAME,
    SELECTION_SIDECAR_NAME,
    CheckpointScore,
    candidate_iterations,
    materialize_best_spnte,
    run_selection,
    score_candidates,
    select_best,
    write_selection_sidecar,
)


def _write_model_checkpoint(run: Path, iteration: int, *, extra: dict | None = None) -> Path:
    checkpoint = {
        "iter": iteration,
        "model_state_dict": {"actor.weight": torch.tensor([float(iteration)])},
    }
    if extra:
        checkpoint.update(extra)
    path = run / f"model_{iteration}.pt"
    torch.save(checkpoint, path)
    return path


def _fake_eval_fn(spnte_by_iter: dict[int, float], extra_metrics: dict | None = None):
    """A fake eval_fn(checkpoint_path, iteration) -> metrics, driven purely by
    an injected iteration -> spnte_lin mapping (the monkeypatch seam)."""
    extra_metrics = extra_metrics or {}

    def _fn(checkpoint_path: Path, iteration: int) -> dict:
        base = {
            "spnte_lin": float(spnte_by_iter[iteration]),
            "spnte_yaw": 0.01,
            "fall_rate": 0.0,
            "tracking_lin_err": 0.05,
        }
        base.update(extra_metrics.get(iteration, {}))
        return base

    return _fn


class TestSelectCheckpointV4(unittest.TestCase):
    # --- pure selection logic ---------------------------------------------

    def test_select_best_picks_min_spnte(self):
        scores = [
            CheckpointScore(1000, Path("model_1000.pt"), {"spnte_lin": 0.30}),
            CheckpointScore(1400, Path("model_1400.pt"), {"spnte_lin": 0.10}),
            CheckpointScore(1800, Path("model_1800.pt"), {"spnte_lin": 0.20}),
        ]
        self.assertEqual(select_best(scores).iteration, 1400)

    def test_select_best_earliest_iteration_tie_break(self):
        scores = [
            CheckpointScore(2200, Path("model_2200.pt"), {"spnte_lin": 0.15}),
            CheckpointScore(1000, Path("model_1000.pt"), {"spnte_lin": 0.15}),
            CheckpointScore(1800, Path("model_1800.pt"), {"spnte_lin": 0.15}),
        ]
        self.assertEqual(select_best(scores).iteration, 1000)

    def test_select_best_raises_on_empty(self):
        with self.assertRaises(ValueError):
            select_best([])

    def test_score_candidates_requires_spnte_lin_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            _write_model_checkpoint(run, 1000)

            def bad_eval_fn(path, iteration):
                return {"tracking_lin_err": 0.1}  # missing spnte_lin

            with self.assertRaisesRegex(ValueError, "spnte_lin"):
                score_candidates(run, [1000], bad_eval_fn)

    def test_score_candidates_missing_checkpoint_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            _write_model_checkpoint(run, 1000)
            with self.assertRaises(FileNotFoundError):
                score_candidates(run, [1000, 2000], _fake_eval_fn({1000: 0.1, 2000: 0.1}))

    # --- candidate schedule --------------------------------------------------

    def test_candidate_iterations_explicit_list_overrides_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for it in range(200, 3001, 200):
                _write_model_checkpoint(run, it)
            self.assertEqual(
                candidate_iterations(run, iterations=[800, 2600]),
                [800, 2600],
            )

    def test_candidate_iterations_default_schedule_matches_v5_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for it in range(200, 3001, 200):
                _write_model_checkpoint(run, it)
            # min_iteration=1000, stride=500 (this module's CLI defaults, mirroring
            # v5_ued.yaml's selection schedule): floor each stride target to the
            # largest existing checkpoint <=, always include the final save.
            self.assertEqual(
                candidate_iterations(run, min_iteration=1000, stride=500),
                [1000, 1400, 1800, 2200, 2600, 3000],
            )

    def test_candidate_iterations_explicit_missing_checkpoint_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            _write_model_checkpoint(run, 1000)
            with self.assertRaises(FileNotFoundError):
                candidate_iterations(run, iterations=[1000, 9999])

    def test_candidate_iterations_no_checkpoints_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                candidate_iterations(Path(tmp))

    # --- atomic publish + provenance ------------------------------------------

    def test_run_selection_end_to_end_picks_min_spnte_and_publishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            spnte_by_iter = {1000: 0.30, 1400: 0.10, 1800: 0.25}
            for it in spnte_by_iter:
                _write_model_checkpoint(run, it)

            winner, best_path, sidecar_path, scores = run_selection(
                run, "go2_v4_mlp",
                iterations=list(spnte_by_iter),
                eval_fn=_fake_eval_fn(spnte_by_iter),
            )

            self.assertEqual(winner.iteration, 1400)
            self.assertEqual(best_path, run / BEST_SPNTE_NAME)
            self.assertTrue(best_path.is_file())
            self.assertEqual(len(scores), 3)

            published = torch.load(best_path, map_location="cpu", weights_only=False)
            self.assertEqual(published["iter"], 1400)
            self.assertTrue(torch.equal(
                published["model_state_dict"]["actor.weight"], torch.tensor([1400.0]),
            ))
            infos = published["infos"]
            self.assertEqual(infos["selection_metric"], "spnte_v1_offline")
            self.assertEqual(infos["selected_iteration"], 1400)
            self.assertAlmostEqual(infos["spnte_lin"], 0.10)
            self.assertEqual(infos["source_checkpoint"], "model_1400.pt")
            self.assertEqual(set(infos["eval_metrics_per_iter"]), {"1000", "1400", "1800"})
            self.assertAlmostEqual(infos["eval_metrics_per_iter"]["1000"]["spnte_lin"], 0.30)

    def test_publish_leaves_no_tmp_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            spnte_by_iter = {1000: 0.2, 1400: 0.1}
            for it in spnte_by_iter:
                _write_model_checkpoint(run, it)
            run_selection(
                run, "go2_v4_mlp",
                iterations=list(spnte_by_iter),
                eval_fn=_fake_eval_fn(spnte_by_iter),
            )
            self.assertFalse((run / (BEST_SPNTE_NAME + ".tmp")).exists())
            self.assertFalse((run / (SELECTION_SIDECAR_NAME + ".tmp")).exists())

    def test_never_creates_or_modifies_best_tracking(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            spnte_by_iter = {1000: 0.2, 1400: 0.1}
            for it in spnte_by_iter:
                _write_model_checkpoint(run, it)

            # Pre-existing best_tracking.pt (from the real V2 tracking selector
            # that DID run during training) must be byte-for-byte untouched.
            tracking_path = run / "best_tracking.pt"
            torch.save({"iter": 999, "model_state_dict": {"actor.weight": torch.tensor([999.0])}}, tracking_path)
            before = tracking_path.read_bytes()

            run_selection(
                run, "go2_v4_mlp",
                iterations=list(spnte_by_iter),
                eval_fn=_fake_eval_fn(spnte_by_iter),
            )

            self.assertEqual(tracking_path.read_bytes(), before)

        with tempfile.TemporaryDirectory() as tmp:
            # And when best_tracking.pt does NOT pre-exist, selection must not create it.
            run = Path(tmp)
            spnte_by_iter = {1000: 0.2, 1400: 0.1}
            for it in spnte_by_iter:
                _write_model_checkpoint(run, it)
            run_selection(
                run, "go2_v4_mlp",
                iterations=list(spnte_by_iter),
                eval_fn=_fake_eval_fn(spnte_by_iter),
            )
            self.assertFalse((run / "best_tracking.pt").exists())

    def test_sidecar_json_records_correct_winner_and_all_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            spnte_by_iter = {1000: 0.30, 1400: 0.10, 1800: 0.25}
            for it in spnte_by_iter:
                _write_model_checkpoint(run, it)

            _, best_path, sidecar_path, _ = run_selection(
                run, "go2_v4_mlp",
                iterations=list(spnte_by_iter),
                eval_fn=_fake_eval_fn(spnte_by_iter),
            )

            self.assertEqual(sidecar_path, run / SELECTION_SIDECAR_NAME)
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["task"], "go2_v4_mlp")
            self.assertEqual(payload["selected_iteration"], 1400)
            self.assertEqual(payload["source_checkpoint"], "model_1400.pt")
            self.assertEqual(payload["best_spnte_path"], str(best_path))
            self.assertEqual(set(payload["eval_metrics_per_iter"]), {"1000", "1400", "1800"})
            self.assertAlmostEqual(payload["eval_metrics_per_iter"]["1400"]["spnte_lin"], 0.10)

    def test_materialize_refuses_when_checkpoint_path_is_already_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            target = run / BEST_SPNTE_NAME
            torch.save({"iter": 1, "model_state_dict": {}}, target)
            winner = CheckpointScore(1, target, {"spnte_lin": 0.1})
            with self.assertRaises(ValueError):
                materialize_best_spnte(run, winner, [winner])

    def test_tie_break_publishes_earliest_iteration_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            spnte_by_iter = {1000: 0.20, 1400: 0.20, 1800: 0.20}
            for it in spnte_by_iter:
                _write_model_checkpoint(run, it)

            winner, best_path, _, _ = run_selection(
                run, "go2_v4_mlp",
                iterations=list(spnte_by_iter),
                eval_fn=_fake_eval_fn(spnte_by_iter),
            )
            self.assertEqual(winner.iteration, 1000)
            published = torch.load(best_path, map_location="cpu", weights_only=False)
            self.assertEqual(published["iter"], 1000)

    def test_write_selection_sidecar_is_atomic_and_overwritable(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            winner = CheckpointScore(1000, run / "model_1000.pt", {"spnte_lin": 0.1})
            path1 = write_selection_sidecar(run, "go2_v4_mlp", winner, [winner])
            self.assertTrue(path1.is_file())
            # Re-run must cleanly overwrite (not append / duplicate / leave temp).
            path2 = write_selection_sidecar(run, "go2_v4_mlp", winner, [winner])
            self.assertEqual(path1, path2)
            self.assertFalse((run / (SELECTION_SIDECAR_NAME + ".tmp")).exists())

    # --- CLI plumbing (mocked run_selection; no Genesis involved) -------------

    def test_main_cli_passes_args_through_to_run_selection(self):
        winner = CheckpointScore(1400, Path("model_1400.pt"), {"spnte_lin": 0.1, "spnte_yaw": 0.01,
                                                                 "fall_rate": 0.0, "tracking_lin_err": 0.05})
        fake_result = (winner, Path("/tmp/best_spnte.pt"), Path("/tmp/best_spnte_selection.json"), [winner])
        with mock.patch(
            "legged_gym.scripts.eval.select_checkpoint_v4.run_selection",
            return_value=fake_result,
        ) as mocked:
            from legged_gym.scripts.eval.select_checkpoint_v4 import main
            rc = main([
                "--run-dir", "/tmp/some_run_seed1",
                "--task", "go2_v4_mlp",
                "--min-iteration", "1000",
                "--stride", "500",
            ])
            self.assertEqual(rc, 0)
            mocked.assert_called_once()
            _, kwargs = mocked.call_args
            self.assertEqual(kwargs["min_iteration"], 1000)
            self.assertEqual(kwargs["stride"], 500)
            self.assertIsNone(kwargs["iterations"])
            self.assertEqual(mocked.call_args.args, ("/tmp/some_run_seed1", "go2_v4_mlp"))


if __name__ == "__main__":
    unittest.main()
