"""Contracts for the revision-independent rolling-completion LP estimator."""
from __future__ import annotations

from copy import deepcopy
import os

import numpy as np
import pytest

# The project package selects a simulator at import time; this module exercises
# only the pure NumPy teacher core.
os.environ.setdefault("SIMULATOR", "genesis")

from legged_gym.utils.ued import EpisodeOutcomeBatch, LPACRLEpisodeCurriculum, TaskSpace
from legged_gym.utils.ued.checkpoint import SCHEMA_VERSION


def _outcomes(
    task_ids: int | list[int] | np.ndarray,
    returns: float | list[float] | np.ndarray,
    *,
    assigned_revision: int,
    completion_revision: int,
    completion_global_control_steps: int,
    length: int = 20,
) -> EpisodeOutcomeBatch:
    """Build one completion batch with explicit completion-time provenance."""
    ids = np.asarray(task_ids, dtype=np.int64)
    if ids.ndim == 0:
        ids = ids.reshape(1)
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(ids.shape, values, dtype=np.float64)
    assert ids.shape == values.shape
    return EpisodeOutcomeBatch(
        task_ids=ids,
        assigned_revision=np.full(ids.shape, assigned_revision, dtype=np.int64),
        completion_revision=completion_revision,
        completion_global_control_steps=completion_global_control_steps,
        episodic_returns=values,
        episode_lengths=np.full(ids.shape, length, dtype=np.int64),
        terminal_reasons=np.full(ids.shape, "timeout", dtype="U16"),
    )


def _rolling(*, window: int = 64, seed: int = 13, **kwargs) -> LPACRLEpisodeCurriculum:
    return LPACRLEpisodeCurriculum(
        TaskSpace(),
        stage_length_control_steps=10,
        beta=1.0,
        seed=seed,
        lp_estimator="rolling_completion",
        rolling_completion_window=window,
        **kwargs,
    )


def _advance(cur: LPACRLEpisodeCurriculum, step: int):
    snapshot = cur.advance(step)
    assert snapshot is not None
    return snapshot


def test_late_completion_appends_to_ring_without_stage_contamination():
    """A previous-revision completion is evidence for rolling LP, never a stage write."""
    task_id = 7
    cur = _rolling(window=2)
    cur.observe(_outcomes(
        task_id, 1.0, assigned_revision=0, completion_revision=0,
        completion_global_control_steps=8,
    ))
    _advance(cur, 10)
    assert cur.sampler_revision == 1

    old_stage_sums = cur._stage_return_sums.copy()
    old_stage_sq_sums = cur._stage_return_sq_sums.copy()
    old_stage_counts = cur._stage_episode_counts.copy()
    cur.observe(_outcomes(
        task_id, 99.0, assigned_revision=0, completion_revision=1,
        completion_global_control_steps=14,
        length=31,
    ))

    # The ring records *all* moving completions, including the late one and
    # its completion-time provenance.
    assert cur._completion_ring_count[task_id] == 2
    assert cur._completion_return_ring[task_id, :2].tolist() == [1.0, 99.0]
    assert cur._completion_length_ring[task_id, :2].tolist() == [20, 31]
    assert cur._completion_step_ring[task_id, :2].tolist() == [8, 14]
    assert cur.diagnostics()["late_outcome_count"] == 1
    assert cur.state_dict()["transition_occupancy"] == {"0:0": 1, "0:1": 1}

    # No late write can leak into either the just-closed or currently-open
    # stage accumulator.  In rolling mode these accumulators are intentionally
    # not a second source of LP evidence at all.
    np.testing.assert_array_equal(cur._stage_return_sums, old_stage_sums)
    np.testing.assert_array_equal(cur._stage_return_sq_sums, old_stage_sq_sums)
    np.testing.assert_array_equal(cur._stage_episode_counts, old_stage_counts)


def test_rolling_lp_is_zero_until_both_completion_windows_are_full():
    task_id = 3
    cur = _rolling(window=64)
    cur.observe(_outcomes(
        np.full(127, task_id, dtype=np.int64), np.arange(127, dtype=np.float64),
        assigned_revision=0, completion_revision=0, completion_global_control_steps=9,
    ))
    snapshot = _advance(cur, 10)

    assert cur._completion_ring_count[task_id] == 127
    assert not cur._rolling_ready_masks[task_id]
    assert snapshot.learning_progress[task_id] == 0.0
    assert snapshot.effective_learning_progress[task_id] == 0.0
    assert not snapshot.eligible_masks[task_id]
    # With no ready task, rolling LP supplies exactly the neutral score and
    # therefore preserves the bootstrap/uniform distribution.
    np.testing.assert_allclose(cur.probabilities(), 1.0 / cur._n)


def test_rolling_lp_matches_manual_two_window_return_means():
    task_id = 11
    previous = np.arange(64, dtype=np.float64)
    current = np.arange(100, 164, dtype=np.float64)
    cur = _rolling(window=64)
    cur.observe(_outcomes(
        np.full(128, task_id, dtype=np.int64), np.concatenate((previous, current)),
        assigned_revision=0, completion_revision=0, completion_global_control_steps=9,
    ))
    snapshot = _advance(cur, 10)

    expected_previous = float(previous.mean())
    expected_current = float(current.mean())
    expected_lp = expected_current - expected_previous
    assert cur._rolling_ready_masks[task_id]
    assert snapshot.previous_returns[task_id] == pytest.approx(expected_previous, abs=1e-12)
    assert snapshot.current_returns[task_id] == pytest.approx(expected_current, abs=1e-12)
    assert snapshot.learning_progress[task_id] == pytest.approx(expected_lp, abs=1e-12)
    assert snapshot.effective_learning_progress[task_id] == pytest.approx(expected_lp, abs=1e-12)


def test_rolling_lp_uses_chronological_tail_after_ring_wraparound():
    """The physical ring order must not become the LP order after wrap-around."""
    task_id = 2
    cur = _rolling(window=64)
    capacity = cur._completion_return_ring.shape[1]
    returns = np.arange(capacity + 128, dtype=np.float64)
    cur.observe(_outcomes(
        np.full(returns.shape, task_id, dtype=np.int64), returns,
        assigned_revision=0, completion_revision=0, completion_global_control_steps=9,
    ))
    snapshot = _advance(cur, 10)

    expected_previous = float(returns[-128:-64].mean())
    expected_current = float(returns[-64:].mean())
    assert capacity == 2048
    assert cur._completion_ring_count[task_id] == capacity
    assert cur._completion_ring_pos[task_id] == 128
    assert snapshot.previous_returns[task_id] == pytest.approx(expected_previous, abs=1e-12)
    assert snapshot.current_returns[task_id] == pytest.approx(expected_current, abs=1e-12)
    assert snapshot.learning_progress[task_id] == pytest.approx(expected_current - expected_previous, abs=1e-12)


def test_ring_observation_never_consumes_sampling_rng_or_changes_next_draw():
    """Appending a completion is logging/accounting, never an RNG operation."""
    logged = _rolling(seed=91)
    reference = _rolling(seed=91)
    before_rng = deepcopy(logged.state_dict()["rng_bit_generator_state"])
    logged.observe(_outcomes(
        np.zeros(128, dtype=np.int64), np.arange(128, dtype=np.float64),
        assigned_revision=0, completion_revision=0, completion_global_control_steps=7,
    ))
    assert logged.state_dict()["rng_bit_generator_state"] == before_rng
    # No advance has occurred, so the identical sampling distributions and RNG
    # states must yield byte-for-byte identical subsequent assignments.
    np.testing.assert_array_equal(
        logged.sample(512, global_control_steps=7).task_ids,
        reference.sample(512, global_control_steps=7).task_ids,
    )


def test_rolling_checkpoint_roundtrip_preserves_ring_lp_rng_and_fingerprint():
    task_id = 19
    source = _rolling(seed=123)
    initial_returns = np.concatenate((np.arange(64, dtype=np.float64), np.arange(100, 164, dtype=np.float64)))
    source.observe(_outcomes(
        np.full(128, task_id, dtype=np.int64), initial_returns,
        assigned_revision=0, completion_revision=0, completion_global_control_steps=9,
    ))
    _advance(source, 10)
    state = source.state_dict()
    # The rolling state is versioned with the curriculum's active checkpoint
    # schema. Later additive logging fields would advance that
    # schema independently; this contract must not pin it to a literal value.
    assert state["schema_version"] == SCHEMA_VERSION

    restored = _rolling(seed=999)
    restored.load_state_dict(state)
    for key in (
        "completion_return_ring", "completion_length_ring", "completion_step_ring",
        "completion_ring_pos", "completion_ring_count", "rolling_ready_masks",
        "rolling_previous_return_sems", "rolling_current_return_sems",
    ):
        np.testing.assert_array_equal(restored.state_dict()[key], state[key])

    # The next identical completion stream produces the identical rolling LP
    # and leaves the sampling stream aligned after load.
    follow_up = _outcomes(
        np.full(64, task_id, dtype=np.int64), np.arange(200, 264, dtype=np.float64),
        assigned_revision=1, completion_revision=1, completion_global_control_steps=19,
    )
    source.observe(follow_up)
    restored.observe(follow_up)
    source_snapshot = _advance(source, 20)
    restored_snapshot = _advance(restored, 20)
    np.testing.assert_array_equal(source_snapshot.learning_progress, restored_snapshot.learning_progress)
    np.testing.assert_array_equal(source.probabilities(), restored.probabilities())
    np.testing.assert_array_equal(
        source.sample(512, global_control_steps=20).task_ids,
        restored.sample(512, global_control_steps=20).task_ids,
    )

    with pytest.raises(ValueError, match="configuration"):
        _rolling(window=32).load_state_dict(state)
    with pytest.raises(ValueError, match="configuration"):
        LPACRLEpisodeCurriculum(TaskSpace(), stage_length_control_steps=10, beta=1.0).load_state_dict(state)


def test_rolling_estimator_keeps_epsilon_floor_and_probability_cap():
    epsilon = 0.03
    probability_cap = 0.08
    hot_task, neutral_task = 0, 1
    cur = _rolling(epsilon=epsilon, max_cell_probability=probability_cap)
    # Only the hot task has positive LP: its two completion windows are 0 then
    # 100.  The neutral task is ready too, with LP 0, so the cap is exercised
    # against a measured rather than merely missing comparison cell.
    ids = np.concatenate((
        np.full(128, hot_task, dtype=np.int64),
        np.full(128, neutral_task, dtype=np.int64),
    ))
    returns = np.concatenate((
        np.zeros(64, dtype=np.float64), np.full(64, 100.0),
        np.zeros(128, dtype=np.float64),
    ))
    cur.observe(_outcomes(
        ids, returns, assigned_revision=0, completion_revision=0,
        completion_global_control_steps=9,
    ))
    _advance(cur, 10)
    probabilities = cur.probabilities()

    assert np.isfinite(probabilities).all()
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-12)
    assert probabilities.max() <= probability_cap + 1e-12
    assert probabilities.min() >= epsilon / cur._n - 1e-12
    assert probabilities[hot_task] > probabilities[neutral_task]
