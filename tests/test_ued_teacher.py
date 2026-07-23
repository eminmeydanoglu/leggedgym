"""Acceptance tests for the pure NumPy UED teacher core."""
from __future__ import annotations

import ast
from copy import deepcopy
import os
from pathlib import Path

import numpy as np
import pytest

# The project package selects a simulator at import time; the pure teacher itself
# is verified separately below to contain no simulator imports.
os.environ.setdefault("SIMULATOR", "genesis")

from legged_gym.utils.ued import (
    ALPEpisodeCurriculum,
    EpisodeOutcomeBatch,
    LPACRLEpisodeCurriculum,
    TaskSpace,
    TaskSpec,
    UniformEpisodeCurriculum,
)


def _outcomes(task_ids, returns, *, valid=True, assigned_revision=0, completion_revision=0):
    task_ids = np.asarray(task_ids, dtype=np.int64)
    count = len(task_ids)
    return EpisodeOutcomeBatch(
        task_ids=task_ids,
        assigned_revision=np.full(count, assigned_revision, dtype=np.int64),
        completion_revision=completion_revision,
        episodic_returns=np.asarray(returns, dtype=np.float64),
        episode_lengths=np.full(count, 20, dtype=np.int64),
        terminal_reasons=np.full(count, "timeout", dtype="U16"),
        valid_for_curriculum=np.full(count, valid, dtype=bool),
    )


def _advance_with_return(curriculum, task_ids, returns, step):
    curriculum.observe(_outcomes(task_ids, returns, completion_revision=curriculum.sampler_revision))
    snapshot = curriculum.advance(step)
    assert snapshot is not None
    return snapshot


def test_task_space_round_trip_and_vectorized_batch():
    space = TaskSpace(builder_parameters={"tile_width": 6.0, "seed": 0})
    assert space.size == 84
    for task_id in range(space.size):
        assert space.encode(space.decode(task_id)) == task_id
    ids = np.array([0, 20, 21, 83], dtype=np.int64)
    batch = space.decode_batch(ids)
    assert batch.terrain_types.tolist() == [0, 5, 0, 5]
    assert batch.terrain_levels.tolist() == [0, 0, 0, 0]
    assert batch.vx_bins.tolist() == [0, 0, 1, 3]
    assert batch.vx_lower.tolist() == [0.0, 0.0, 0.5, 1.5]
    assert batch.vx_upper.tolist() == [0.5, 0.5, 1.0, 2.0]
    with pytest.raises(ValueError):
        space.encode(TaskSpec(5, 1, 0))
    assert space.fingerprint() == TaskSpace(builder_parameters={"seed": 0, "tile_width": 6.0}).fingerprint()


def test_task_space_rejects_lossy_identity_and_snapshots_fingerprint_inputs():
    builder = {"tile": {"spacing": 6.0}}
    space = TaskSpace(builder_parameters=builder)
    fingerprint = space.fingerprint()
    builder["tile"]["spacing"] = 7.0
    assert space.fingerprint() == fingerprint
    assert space.builder_parameters["tile"]["spacing"] == 6.0
    with pytest.raises(ValueError, match="integer"):
        space.encode(TaskSpec(0.5, 0, 0))
    with pytest.raises(ValueError, match="integer"):
        space.decode(0.0)
    with pytest.raises(ValueError, match="JSON-compatible"):
        TaskSpace(builder_parameters={"nan": float("nan")})
    with pytest.raises(ValueError, match="strictly increasing"):
        TaskSpace(velocity_bin_edges=(0.0, 0.5, float("nan"), 1.5, 2.0))


def test_uniform_sampling_and_local_rng_are_deterministic():
    first = UniformEpisodeCurriculum(TaskSpace(), stage_length_control_steps=10, seed=0)
    second = UniformEpisodeCurriculum(TaskSpace(), stage_length_control_steps=10, seed=0)
    a = first.sample(84_000, global_control_steps=0)
    b = second.sample(84_000, global_control_steps=0)
    np.testing.assert_array_equal(a.task_ids, b.task_ids)
    assert set(a.sources.tolist()) == {"bootstrap"}
    counts = np.bincount(a.task_ids, minlength=84)
    expected = len(a.task_ids) / 84
    # A fixed 84k sample has a per-cell standard deviation near 31; a 13%
    # bound is a conservative deterministic smoke threshold across 84 cells.
    assert np.max(np.abs(counts - expected)) < expected * 0.13
    np.testing.assert_allclose(first.probabilities(), np.full(84, 1 / 84))


def test_positive_lp_increases_probability_after_baseline_stage():
    curriculum = LPACRLEpisodeCurriculum(TaskSpace(), stage_length_control_steps=10, beta=0.2, seed=3)
    _advance_with_return(curriculum, [0, 1], [0.0, 0.0], 10)
    before = curriculum.probabilities()
    snapshot = _advance_with_return(curriculum, [0, 1], [1.0, 0.0], 20)
    after = curriculum.probabilities()
    assert after[0] > before[0]
    assert after[0] > after[1]
    assert snapshot.learning_progress[0] == pytest.approx(1.0)
    assert set(curriculum.sample(2, global_control_steps=20).sources.tolist()) == {"lp"}
    assert np.isfinite(after).all() and after.sum() == pytest.approx(1.0)


def test_negative_lp_is_absolute_only_for_alp():
    signed = LPACRLEpisodeCurriculum(TaskSpace(), stage_length_control_steps=10, beta=0.2, seed=4)
    absolute = ALPEpisodeCurriculum(TaskSpace(), stage_length_control_steps=10, beta=0.2, seed=4)
    for curriculum in (signed, absolute):
        _advance_with_return(curriculum, [0, 1], [1.0, 1.0], 10)
        _advance_with_return(curriculum, [0, 1], [0.0, 1.0], 20)
    assert signed.probabilities()[0] < signed.probabilities()[1]
    assert absolute.probabilities()[0] > absolute.probabilities()[1]


def test_missing_task_fallback_retains_its_previous_probability_and_invalid_is_ignored():
    curriculum = LPACRLEpisodeCurriculum(TaskSpace(), stage_length_control_steps=10, beta=0.2, seed=5)
    _advance_with_return(curriculum, [0, 1, 2], [0.0, 0.0, 0.0], 10)
    before = curriculum.probabilities()
    curriculum.observe(_outcomes([0, 1], [1.0, 0.0], completion_revision=1))
    curriculum.observe(_outcomes([0], [999.0], valid=False, completion_revision=1))
    snapshot = curriculum.advance(20)
    assert snapshot is not None
    after = curriculum.probabilities()
    assert after[2] == pytest.approx(before[2])
    assert np.isnan(snapshot.current_returns[2])
    assert snapshot.invalid_outcome_count == 1
    assert snapshot.transition_occupancy["0:1"] == 3


def test_diagnostics_and_assignment_completion_provenance():
    curriculum = UniformEpisodeCurriculum(TaskSpace(), stage_length_control_steps=10, seed=6)
    assignment = curriculum.sample(10, global_control_steps=0)
    curriculum.observe(_outcomes(assignment.task_ids, np.ones(10), assigned_revision=assignment.sampler_revision))
    snapshot = curriculum.advance(10)
    assert snapshot is not None
    d = curriculum.diagnostics()
    assert d["finite_probabilities"] is True
    assert d["entropy"] > 0 and d["effective_sample_size"] == pytest.approx(84.0)
    assert d["task_assignment_coverage"] > 0
    assert d["valid_completed_outcome_coverage"] > 0
    assert snapshot.transition_occupancy["0:0"] == 10


def test_checkpoint_continuation_and_fingerprint_rejection():
    source = LPACRLEpisodeCurriculum(TaskSpace(), stage_length_control_steps=10, beta=0.5, seed=0)
    _advance_with_return(source, [0], [0.0], 10)
    _advance_with_return(source, [0], [1.0], 20)
    state = source.state_dict()
    restored = LPACRLEpisodeCurriculum(TaskSpace(), stage_length_control_steps=10, beta=0.5, seed=999)
    restored.load_state_dict(state)
    np.testing.assert_array_equal(
        source.sample(100, global_control_steps=20).task_ids,
        restored.sample(100, global_control_steps=20).task_ids,
    )
    wrong_space = LPACRLEpisodeCurriculum(TaskSpace(builder_parameters={"changed": True}), stage_length_control_steps=10, beta=0.5)
    with pytest.raises(ValueError, match="fingerprint"):
        wrong_space.load_state_dict(state)
    wrong_algorithm = ALPEpisodeCurriculum(TaskSpace(), stage_length_control_steps=10, beta=0.5)
    with pytest.raises(ValueError, match="algorithm"):
        wrong_algorithm.load_state_dict(state)


def test_checkpoint_schema_configuration_and_counter_types_fail_closed():
    source = LPACRLEpisodeCurriculum(TaskSpace(), stage_length_control_steps=10, beta=0.5, seed=0)
    state = source.state_dict()
    wrong_config = LPACRLEpisodeCurriculum(TaskSpace(), stage_length_control_steps=10, beta=0.25)
    with pytest.raises(ValueError, match="configuration"):
        wrong_config.load_state_dict(state)
    wrong_schema = deepcopy(state)
    wrong_schema["schema_version"] = 2
    with pytest.raises(ValueError, match="schema"):
        source.load_state_dict(wrong_schema)
    fractional_counter = deepcopy(state)
    fractional_counter["stage_episode_counts"] = np.zeros(84, dtype=np.float64)
    with pytest.raises(ValueError, match="integers"):
        source.load_state_dict(fractional_counter)
    malformed_rng = deepcopy(state)
    malformed_rng["rng_bit_generator_state"] = {"bit_generator": "MT19937"}
    before = source.state_dict()
    with pytest.raises(ValueError, match="RNG"):
        source.load_state_dict(malformed_rng)
    assert source.state_dict()["rng_bit_generator_state"] == before["rng_bit_generator_state"]


def test_public_batch_contract_rejects_bad_outcomes():
    curriculum = UniformEpisodeCurriculum(TaskSpace(), stage_length_control_steps=10)
    with pytest.raises(ValueError, match="finite"):
        curriculum.observe(_outcomes([0], [np.nan]))
    with pytest.raises(ValueError, match="outside"):
        curriculum.observe(_outcomes([84], [0.0]))
    fractional_revision = _outcomes([0], [0.0])
    fractional_revision = EpisodeOutcomeBatch(
        **{**fractional_revision.__dict__, "assigned_revision": np.array([0.5])}
    )
    with pytest.raises(ValueError, match="assigned_revision"):
        curriculum.observe(fractional_revision)
    non_boolean_validity = _outcomes([0], [0.0])
    non_boolean_validity = EpisodeOutcomeBatch(
        **{**non_boolean_validity.__dict__, "valid_for_curriculum": np.array([1], dtype=np.int64)}
    )
    with pytest.raises(ValueError, match="boolean"):
        curriculum.observe(non_boolean_validity)


def test_public_scalar_inputs_and_softmax_remain_well_defined_at_extremes():
    curriculum = LPACRLEpisodeCurriculum(TaskSpace(), stage_length_control_steps=10, beta=1e-308)
    with pytest.raises(ValueError, match="count"):
        curriculum.sample(1.5, global_control_steps=0)
    with pytest.raises(ValueError, match="global_control_steps"):
        curriculum.advance(10.5)
    probabilities = curriculum._softmax(np.array([-1e308, 1e308]), curriculum.beta)
    np.testing.assert_allclose(probabilities, [0.0, 1.0])


def test_teacher_modules_have_no_forbidden_imports():
    # Only the TEACHER CORE (task_space/episode_curriculum/checkpoint, plus
    # the package __init__ that merely re-exports them) must stay free of
    # simulator/PPO imports (solun_plani.md §12: "Teacher modülü Genesis,
    # Torch, PPO veya policy import etmez"). genesis_adapter.py is
    # deliberately excluded: it is the documented Torch/Genesis *boundary*
    # (solun_plani.md §7 "GenesisUEDAdapter"), not the teacher itself, and is
    # SUPPOSED to import torch.
    root = Path(__file__).resolve().parents[1] / "legged_gym" / "utils" / "ued"
    teacher_core_files = ("__init__.py", "task_space.py", "episode_curriculum.py", "checkpoint.py")
    forbidden = {"genesis", "torch", "ppo", "policy"}
    for name in teacher_core_files:
        path = root / name
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = [name.name for name in node.names]
            elif isinstance(node, ast.ImportFrom):
                imports = [node.module or ""]
            else:
                continue
            for imported in imports:
                assert not any(part.lower() in forbidden for part in imported.split(".")), f"{path}: {imported}"
