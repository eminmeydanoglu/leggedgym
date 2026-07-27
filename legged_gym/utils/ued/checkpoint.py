"""Schema-v1 validation helpers for serialised UED curriculum state."""
from __future__ import annotations

from typing import Mapping


# v2 dropped the standstill-era ``invalid_outcome_count`` and
# ``valid_task_completion_counts`` fields: standstill is now a reserved mixture
# bucket that never reaches the curriculum, so every observed outcome is valid.
# v3 added per-cell return uncertainty and deterministic adaptive-temperature
# controller state. v4 adds two-stage sample-count admission, coverage
# diagnostics, and the adaptive-controller bootstrap flag.
SCHEMA_VERSION = 4


def validate_checkpoint_state(
    state: Mapping[str, object], *, algorithm: str, task_space_fingerprint: str, config_fingerprint: str
) -> None:
    """Fail closed before a curriculum state can be restored."""
    required = {
        "schema_version", "algorithm", "task_space_fingerprint", "config_fingerprint",
        "rng_bit_generator_state", "stage_index", "sampler_revision", "stage_start_global_steps",
        "probabilities", "previous_returns", "current_returns",
        "previous_return_sems", "current_return_sems", "learning_progress",
        "effective_learning_progress", "observed_masks", "stage_return_sums",
        "eligible_masks", "previous_stage_episode_counts",
        "stage_return_sq_sums", "stage_episode_counts", "task_assignment_counts",
        "task_completion_counts", "transition_occupancy", "source_label",
        "effective_beta", "target_ess", "signal_quality", "has_adaptive_signal",
        "ess_guard_uniform_mix",
    }
    missing = required.difference(state)
    if missing:
        raise ValueError(f"curriculum checkpoint is missing keys: {sorted(missing)}")
    if state["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported curriculum checkpoint schema")
    if state["algorithm"] != algorithm:
        raise ValueError("curriculum algorithm fingerprint mismatch")
    if state["task_space_fingerprint"] != task_space_fingerprint:
        raise ValueError("task-space fingerprint mismatch")
    if state["config_fingerprint"] != config_fingerprint:
        raise ValueError("curriculum configuration fingerprint mismatch")
