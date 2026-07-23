"""Schema-v1 validation helpers for serialised UED curriculum state."""
from __future__ import annotations

from typing import Mapping


SCHEMA_VERSION = 1


def validate_checkpoint_state(
    state: Mapping[str, object], *, algorithm: str, task_space_fingerprint: str, config_fingerprint: str
) -> None:
    """Fail closed before a curriculum state can be restored."""
    required = {
        "schema_version", "algorithm", "task_space_fingerprint", "config_fingerprint",
        "rng_bit_generator_state", "stage_index", "sampler_revision", "stage_start_global_steps",
        "probabilities", "previous_returns", "current_returns", "learning_progress", "observed_masks",
        "stage_return_sums", "stage_episode_counts", "task_assignment_counts", "task_completion_counts",
        "valid_task_completion_counts", "transition_occupancy", "invalid_outcome_count", "source_label",
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
