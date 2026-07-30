"""Focused contracts for V5 Uniform's read-only shadow metric telemetry."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import json
from types import SimpleNamespace

from legged_gym.utils.ued import TaskSpace, UniformEpisodeCurriculum
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.storage import RolloutStorage
from lpacr.analysis.shadow_metrics import analyze, load_frames, load_heldout_by_stage


def _curriculum() -> UniformEpisodeCurriculum:
    return UniformEpisodeCurriculum(TaskSpace(), stage_length_control_steps=10, seed=7)


def test_raw_gae_aggregation_excludes_standstill_and_preserves_sign_statistics():
    curriculum = _curriculum()
    curriculum.observe_shadow_gae(
        task_ids=np.asarray([[0, 0, 1], [0, 1, 1]], dtype=np.int64),
        standstill=np.asarray([[False, True, False], [False, False, False]]),
        raw_gae=np.asarray([[-2.0, 99.0, 4.0], [0.0, -3.0, 2.0]]),
        stage_indices=np.ones((2, 3), dtype=np.int64),
    )
    snapshot = curriculum.advance(10)
    assert snapshot is not None
    assert snapshot.gae_timestep_counts[:2].tolist() == [2, 3]
    np.testing.assert_allclose(snapshot.raw_gae_sums[:2], [-2.0, 3.0])
    np.testing.assert_allclose(snapshot.positive_gae_sums[:2], [0.0, 6.0])
    np.testing.assert_allclose(snapshot.absolute_gae_sums[:2], [2.0, 9.0])
    assert snapshot.positive_gae_counts[:2].tolist() == [0, 2]


def test_closed_snapshot_receives_rollout_gae_before_deferred_publication():
    curriculum = _curriculum()
    snapshot = curriculum.advance(10)
    assert snapshot is not None
    curriculum.observe_shadow_gae(
        task_ids=np.asarray([[3]], dtype=np.int64),
        standstill=np.asarray([[False]]),
        raw_gae=np.asarray([[2.5]]),
        stage_indices=np.asarray([[1]], dtype=np.int64),
    )
    assert snapshot.gae_timestep_counts[3] == 1
    assert snapshot.positive_gae_sums[3] == pytest.approx(2.5)


def test_completion_stage_provenance_and_success_are_not_censored_by_assignment_revision():
    curriculum = _curriculum()
    from legged_gym.utils.ued import EpisodeOutcomeBatch

    curriculum.observe(EpisodeOutcomeBatch(
        task_ids=np.asarray([4, 4]),
        assigned_revision=np.asarray([0, 0]),
        completion_revision=0,
        episodic_returns=np.asarray([1.0, 2.0]),
        episode_lengths=np.asarray([7, 9]),
        terminal_reasons=np.asarray(["timeout", "terminal"]),
    ))
    first = curriculum.advance(10)
    assert first is not None
    curriculum.observe(EpisodeOutcomeBatch(
        task_ids=np.asarray([4]),
        assigned_revision=np.asarray([0]),
        completion_revision=1,
        episodic_returns=np.asarray([3.0]),
        episode_lengths=np.asarray([11]),
        terminal_reasons=np.asarray(["timeout"]),
    ))
    second = curriculum.advance(20)
    assert second is not None
    assert second.completion_stage_episode_counts[4] == 1
    assert second.assigned_same_revision_counts[4] == 0
    assert second.cross_revision_completion_counts[4] == 1
    assert second.success_counts[4] == 1
    assert second.timeout_counts[4] == 1
    assert second.terminal_counts[4] == 0
    assert second.episode_length_sums[4] == pytest.approx(11.0)


def test_rollout_storage_retains_raw_gae_before_global_normalization():
    storage = RolloutStorage(1, 2, (1,), (1,), (1,), device="cpu")
    device = storage.rewards.device
    storage.rewards[:, 0, 0] = torch.tensor([1.0, 2.0], device=device)
    storage.values[:, 0, 0] = torch.tensor([0.5, 0.25], device=device)
    storage.dones[:, 0, 0] = torch.tensor([0, 1], dtype=torch.uint8, device=device)
    storage.compute_returns(torch.tensor([[4.0]], device=device), gamma=1.0, lam=1.0)
    raw = storage.raw_advantages[:, 0, 0]
    torch.testing.assert_close(raw, torch.tensor([2.5, 1.75], device=device))
    assert not torch.allclose(storage.advantages[:, 0, 0], raw)


def test_even_closed_stage_writes_checkpoint_manifest_after_update(tmp_path):
    curriculum = _curriculum()
    snapshot = curriculum.advance(10)
    assert snapshot is not None and snapshot.stage_index == 1
    env = SimpleNamespace(
        episode_curriculum=curriculum,
        consume_ued_stage_checkpoints=lambda: [snapshot],
    )
    runner = object.__new__(OnPolicyRunner)
    runner.env = env
    runner.cfg = {"ued_stage_checkpoint_interval": 2}
    runner.log_dir = str(tmp_path)
    (tmp_path / "run_manifest.json").write_text(json.dumps({"task": "go2_v5_uniform"}))
    saved = []
    runner.save = lambda path, iteration, infos: saved.append((path, iteration, infos))
    runner._save_ued_stage_checkpoints(17)
    assert len(saved) == 1
    checkpoint, iteration, infos = saved[0]
    assert checkpoint.endswith("shadow_stage_0000_iter_000017.pt")
    assert iteration == 17
    assert infos["shadow_stage_checkpoint"]["global_control_steps"] == 10
    manifest = json.loads((tmp_path / "shadow_stage_0000_iter_000017.json").read_text())
    assert manifest["closed_stage_index"] == 0
    assert manifest["ppo_completed_iteration"] == 17
    run_manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert run_manifest["shadow_stage_checkpoints"] == [manifest]


def test_shadow_analysis_uses_only_predeclared_future_horizons_and_manifest_join(tmp_path):
    n_cells = 24
    frames = []
    for stage in (1, 3, 5):
        signal = np.arange(n_cells, dtype=float) + stage
        frames.append({
            "metadata": {"frame": {"stage_index": stage}},
            "metrics": {
                "pvl": signal.tolist(), "abs_gae": (2.0 * signal).tolist(),
                "success_rate": (signal / signal.max()).tolist(),
                "frontier": (4.0 * signal / signal.max() * (1.0 - signal / signal.max())).tolist(),
                "completion_count": [10] * n_cells, "gae_timestep_count": [20] * n_cells,
                "raw_gae_sum": signal.tolist(), "episode_length_sum": [100] * n_cells,
                "sampling_probability": [1.0 / n_cells] * n_cells,
            },
        })
    frames_path = tmp_path / "frames.ndjson"
    frames_path.write_text("".join(json.dumps(row) + "\n" for row in frames))
    validation = tmp_path / "heldout"
    validation.mkdir()
    for stage, iteration in ((1, 10), (3, 30), (5, 50)):
        checkpoint_name = f"shadow_stage_{stage:04d}_iter_{iteration:06d}.pt"
        checkpoint = tmp_path / checkpoint_name
        checkpoint.write_bytes(f"checkpoint-{iteration}".encode())
        import hashlib
        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        (tmp_path / f"shadow_stage_{stage}.json").write_text(json.dumps({
            "snapshot_stage_index": stage, "ppo_completed_iteration": iteration,
            "checkpoint_file": checkpoint_name,
        }))
        cells = [
            {"cell_id": cell, "spnte_lin": 1.0 - cell * stage / 1000.0,
             "fall_rate": 0.5, "cell_success": bool(cell % 2)}
            for cell in range(n_cells)
        ]
        (validation / f"model_{iteration}.json").write_text(json.dumps({
            "checkpoint_iteration": iteration, "checkpoint_sha256": checkpoint_sha256,
            "validation_bank_fingerprint": "fixed-bank-v1",
            "provenance": {"shadow_stage": {"snapshot_stage_index": stage}},
            "scores": {"cells": cells},
        }))
    report = analyze(load_frames(frames_path), load_heldout_by_stage(tmp_path, validation),
                     n_bootstrap=5, n_permutation=5)
    assert report["horizons"]["2"]["paired_stages"] == [1, 3]
    assert report["horizons"]["4"]["paired_stages"] == [1]
    assert "sampling_probability" in report["horizons"]["2"]["signals"]
