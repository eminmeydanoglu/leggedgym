"""Pure NumPy episode-task curricula for the finite V5 task space."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from typing import Mapping, Protocol

import numpy as np

from .checkpoint import SCHEMA_VERSION, validate_checkpoint_state
from .task_space import TaskSpace


@dataclass(frozen=True)
class TaskAssignmentBatch:
    task_ids: np.ndarray
    sampler_revision: int
    curriculum_stage: int
    probabilities: np.ndarray
    sources: np.ndarray


@dataclass(frozen=True)
class EpisodeOutcomeBatch:
    task_ids: np.ndarray
    assigned_revision: np.ndarray
    completion_revision: int
    episodic_returns: np.ndarray
    episode_lengths: np.ndarray
    terminal_reasons: np.ndarray
    valid_for_curriculum: np.ndarray


@dataclass(frozen=True)
class StageSnapshot:
    global_control_steps: int
    stage_index: int
    sampler_revision: int
    probabilities: np.ndarray
    previous_returns: np.ndarray
    current_returns: np.ndarray
    learning_progress: np.ndarray
    observed_masks: np.ndarray
    stage_episode_counts: np.ndarray
    task_assignment_counts: np.ndarray
    task_completion_counts: np.ndarray
    transition_occupancy: Mapping[str, int]
    diagnostics: Mapping[str, object]
    invalid_outcome_count: int


class EpisodeCurriculum(Protocol):
    def sample(self, count: int, *, global_control_steps: int) -> TaskAssignmentBatch: ...
    def observe(self, outcomes: EpisodeOutcomeBatch) -> None: ...
    def advance(self, global_control_steps: int) -> StageSnapshot | None: ...
    def probabilities(self) -> np.ndarray: ...
    def diagnostics(self) -> Mapping[str, object]: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: Mapping[str, object]) -> None: ...


class _FiniteEpisodeCurriculum:
    algorithm = "uniform"

    def __init__(
        self,
        task_space: TaskSpace,
        *,
        stage_length_control_steps: int,
        beta: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if stage_length_control_steps <= 0:
            raise ValueError("stage_length_control_steps must be positive")
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be finite and positive")
        self.task_space = task_space
        self.stage_length_control_steps = int(stage_length_control_steps)
        self.beta = float(beta)
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._n = task_space.size
        self._probabilities = np.full(self._n, 1.0 / self._n, dtype=np.float64)
        self._source_label = "bootstrap"
        self.stage_index = 0
        self.sampler_revision = 0
        self.stage_start_global_steps = 0
        self._previous_returns = np.full(self._n, np.nan, dtype=np.float64)
        self._current_returns = np.full(self._n, np.nan, dtype=np.float64)
        self._learning_progress = np.full(self._n, np.nan, dtype=np.float64)
        self._observed_masks = np.zeros(self._n, dtype=bool)
        self._stage_return_sums = np.zeros(self._n, dtype=np.float64)
        self._stage_episode_counts = np.zeros(self._n, dtype=np.int64)
        self._task_assignment_counts = np.zeros(self._n, dtype=np.int64)
        self._task_completion_counts = np.zeros(self._n, dtype=np.int64)
        self._valid_task_completion_counts = np.zeros(self._n, dtype=np.int64)
        self._transition_occupancy: dict[str, int] = {}
        self._invalid_outcome_count = 0
        self._snapshots: list[StageSnapshot] = []

    @property
    def config_fingerprint(self) -> str:
        payload = {"algorithm": self.algorithm, "stage_length_control_steps": self.stage_length_control_steps, "beta": self.beta}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @staticmethod
    def _non_negative_integer(value: object, *, name: str) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        return int(value)

    def sample(self, count: int, *, global_control_steps: int) -> TaskAssignmentBatch:
        count = self._non_negative_integer(count, name="count")
        self._non_negative_integer(global_control_steps, name="global_control_steps")
        task_ids = self._rng.choice(self._n, size=count, p=self._probabilities).astype(np.int64, copy=False)
        self._task_assignment_counts += np.bincount(task_ids, minlength=self._n)
        return TaskAssignmentBatch(
            task_ids,
            self.sampler_revision,
            self.stage_index,
            self._probabilities[task_ids].copy(),
            np.full(count, self._source_label, dtype="U10"),
        )

    def observe(self, outcomes: EpisodeOutcomeBatch) -> None:
        arrays = (
            outcomes.task_ids, outcomes.assigned_revision, outcomes.episodic_returns,
            outcomes.episode_lengths, outcomes.terminal_reasons, outcomes.valid_for_curriculum,
        )
        lengths = {np.asarray(value).shape for value in arrays}
        if len(lengths) != 1 or len(next(iter(lengths))) != 1:
            raise ValueError("outcome fields must be same-length one-dimensional arrays")
        task_ids = np.asarray(outcomes.task_ids)
        if not np.issubdtype(task_ids.dtype, np.integer) or np.any(task_ids < 0) or np.any(task_ids >= self._n):
            raise ValueError("outcome task_ids are outside the task space")
        assigned_revisions = np.asarray(outcomes.assigned_revision)
        if not np.issubdtype(assigned_revisions.dtype, np.integer) or np.any(assigned_revisions < 0):
            raise ValueError("outcome assigned_revision must be non-negative integers")
        completion_revision = self._non_negative_integer(
            outcomes.completion_revision, name="outcome completion_revision"
        )
        returns = np.asarray(outcomes.episodic_returns, dtype=np.float64)
        episode_lengths = np.asarray(outcomes.episode_lengths)
        if not np.issubdtype(episode_lengths.dtype, np.integer) or np.any(episode_lengths < 0):
            raise ValueError("outcome episode_lengths must be non-negative integers")
        if np.asarray(outcomes.valid_for_curriculum).dtype != np.bool_:
            raise ValueError("outcome valid_for_curriculum must be boolean")
        valid = np.asarray(outcomes.valid_for_curriculum, dtype=bool)
        if np.any(valid & ~np.isfinite(returns)):
            raise ValueError("valid curriculum outcomes require finite returns")
        self._task_completion_counts += np.bincount(task_ids.astype(np.int64), minlength=self._n)
        self._invalid_outcome_count += int((~valid).sum())
        for assigned in assigned_revisions:
            key = f"{int(assigned)}:{completion_revision}"
            self._transition_occupancy[key] = self._transition_occupancy.get(key, 0) + 1
        valid_ids = task_ids[valid].astype(np.int64, copy=False)
        self._valid_task_completion_counts += np.bincount(valid_ids, minlength=self._n)
        self._stage_return_sums += np.bincount(valid_ids, weights=returns[valid], minlength=self._n)
        self._stage_episode_counts += np.bincount(valid_ids, minlength=self._n)

    def _score(self, progress: np.ndarray) -> np.ndarray:
        return progress if self.algorithm == "lp_acrl" else np.abs(progress)

    @staticmethod
    def _softmax(values: np.ndarray, beta: float) -> np.ndarray:
        if not np.all(np.isfinite(values)):
            raise ValueError("non-finite learning progress")
        # Shift before scaling: scaling first can overflow even when the
        # resulting softmax is well-defined for finite inputs.
        with np.errstate(over="ignore", under="ignore"):
            shifted = (values - np.max(values)) / beta
            weights = np.exp(shifted, dtype=np.float64)
        total = weights.sum(dtype=np.float64)
        if not np.isfinite(total) or total <= 0:
            raise ValueError("invalid softmax normalizer")
        return weights / total

    def advance(self, global_control_steps: int) -> StageSnapshot | None:
        global_control_steps = self._non_negative_integer(global_control_steps, name="global_control_steps")
        if global_control_steps < self.stage_start_global_steps:
            raise ValueError("global_control_steps cannot move backwards")
        if global_control_steps - self.stage_start_global_steps < self.stage_length_control_steps:
            return None
        observed = self._stage_episode_counts > 0
        current = np.full(self._n, np.nan, dtype=np.float64)
        current[observed] = self._stage_return_sums[observed] / self._stage_episode_counts[observed]
        progress_mask = observed & self._observed_masks
        progress = np.full(self._n, np.nan, dtype=np.float64)
        progress[progress_mask] = current[progress_mask] - self._current_returns[progress_mask]
        if self.algorithm != "uniform" and np.any(progress_mask):
            retained = ~progress_mask
            retained_mass = float(self._probabilities[retained].sum())
            available = 1.0 - retained_mass
            if available > 0:
                updated = self._probabilities.copy()
                updated[progress_mask] = available * self._softmax(self._score(progress[progress_mask]), self.beta)
                self._probabilities = updated
                self._source_label = "lp" if self.algorithm == "lp_acrl" else "alp"
        if not np.all(np.isfinite(self._probabilities)) or not np.isclose(self._probabilities.sum(), 1.0):
            raise ValueError("curriculum probabilities are not finite and normalized")
        self._previous_returns = self._current_returns.copy()
        self._current_returns = current
        self._learning_progress = progress
        self._observed_masks = observed
        self.stage_index += 1
        self.sampler_revision += 1
        self.stage_start_global_steps = int(global_control_steps)
        snapshot = StageSnapshot(
            int(global_control_steps), self.stage_index, self.sampler_revision, self._probabilities.copy(),
            self._previous_returns.copy(), self._current_returns.copy(), self._learning_progress.copy(),
            self._observed_masks.copy(), self._stage_episode_counts.copy(), self._task_assignment_counts.copy(),
            self._task_completion_counts.copy(), dict(self._transition_occupancy), self.diagnostics(),
            self._invalid_outcome_count,
        )
        self._snapshots.append(snapshot)
        self._stage_return_sums.fill(0.0)
        self._stage_episode_counts.fill(0)
        self._invalid_outcome_count = 0
        return snapshot

    def probabilities(self) -> np.ndarray:
        return self._probabilities.copy()

    def diagnostics(self) -> Mapping[str, object]:
        p = self._probabilities
        return {
            "finite_probabilities": bool(np.all(np.isfinite(p))),
            "entropy": float(-np.sum(np.where(p > 0, p * np.log(p), 0.0))),
            "effective_sample_size": float(1.0 / np.sum(np.square(p))),
            "max_cell_probability": float(np.max(p)),
            "task_assignment_coverage": float(np.count_nonzero(self._task_assignment_counts) / self._n),
            "valid_completed_outcome_coverage": float(np.count_nonzero(self._valid_task_completion_counts) / self._n),
        }

    def state_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION, "algorithm": self.algorithm,
            "task_space_fingerprint": self.task_space.fingerprint(), "config_fingerprint": self.config_fingerprint,
            "stage_index": self.stage_index, "sampler_revision": self.sampler_revision,
            "stage_start_global_steps": self.stage_start_global_steps, "probabilities": self._probabilities.copy(),
            "previous_returns": self._previous_returns.copy(), "current_returns": self._current_returns.copy(),
            "learning_progress": self._learning_progress.copy(), "observed_masks": self._observed_masks.copy(),
            "stage_return_sums": self._stage_return_sums.copy(), "stage_episode_counts": self._stage_episode_counts.copy(),
            "task_assignment_counts": self._task_assignment_counts.copy(), "task_completion_counts": self._task_completion_counts.copy(),
            "valid_task_completion_counts": self._valid_task_completion_counts.copy(),
            "transition_occupancy": dict(self._transition_occupancy),
            "invalid_outcome_count": self._invalid_outcome_count,
            "source_label": self._source_label,
            "rng_bit_generator_state": deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        validate_checkpoint_state(state, algorithm=self.algorithm, task_space_fingerprint=self.task_space.fingerprint(), config_fingerprint=self.config_fingerprint)
        arrays = ("probabilities", "previous_returns", "current_returns", "learning_progress", "observed_masks", "stage_return_sums", "stage_episode_counts", "task_assignment_counts", "task_completion_counts", "valid_task_completion_counts")
        count_arrays = {"stage_episode_counts", "task_assignment_counts", "task_completion_counts", "valid_task_completion_counts"}
        loaded_arrays: dict[str, np.ndarray] = {}
        for name in arrays:
            value = np.asarray(state[name])
            if value.shape != (self._n,):
                raise ValueError(f"checkpoint {name} has the wrong task-space shape")
            if name == "observed_masks":
                if value.dtype != np.bool_:
                    raise ValueError("checkpoint observed_masks must be boolean")
            elif name in count_arrays:
                if not np.issubdtype(value.dtype, np.integer) or np.any(value < 0):
                    raise ValueError(f"checkpoint {name} must be non-negative integers")
            else:
                if not np.issubdtype(value.dtype, np.number) or np.any(np.isinf(value)):
                    raise ValueError(f"checkpoint {name} must be numeric without infinities")
            loaded_arrays[name] = value.copy()
        probabilities = np.asarray(loaded_arrays["probabilities"], dtype=np.float64)
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("checkpoint probabilities are invalid")
        stage_index = self._non_negative_integer(state["stage_index"], name="checkpoint stage_index")
        sampler_revision = self._non_negative_integer(state["sampler_revision"], name="checkpoint sampler_revision")
        stage_start_global_steps = self._non_negative_integer(state["stage_start_global_steps"], name="checkpoint stage_start_global_steps")
        try:
            occupancy = dict(state["transition_occupancy"])
        except (TypeError, ValueError) as exc:
            raise ValueError("checkpoint transition_occupancy must be a mapping") from exc
        transition_occupancy = {
            str(key): self._non_negative_integer(value, name="checkpoint transition occupancy")
            for key, value in occupancy.items()
        }
        invalid_outcome_count = self._non_negative_integer(state["invalid_outcome_count"], name="checkpoint invalid_outcome_count")
        source_label = str(state["source_label"])
        allowed_sources = {"bootstrap"} if self.algorithm == "uniform" else {"bootstrap", "lp" if self.algorithm == "lp_acrl" else "alp"}
        if source_label not in allowed_sources:
            raise ValueError("checkpoint source_label does not match curriculum algorithm")
        rng_state = deepcopy(state["rng_bit_generator_state"])
        try:
            np.random.PCG64().state = rng_state
        except (TypeError, ValueError) as exc:
            raise ValueError("checkpoint RNG state is not a valid PCG64 state") from exc

        self._probabilities = probabilities
        for name, value in loaded_arrays.items():
            if name != "probabilities":
                setattr(self, "_" + name, value)
        self.stage_index = stage_index
        self.sampler_revision = sampler_revision
        self.stage_start_global_steps = stage_start_global_steps
        self._transition_occupancy = transition_occupancy
        self._invalid_outcome_count = invalid_outcome_count
        self._source_label = source_label
        self._rng.bit_generator.state = rng_state


class UniformEpisodeCurriculum(_FiniteEpisodeCurriculum):
    algorithm = "uniform"


class LPACRLEpisodeCurriculum(_FiniteEpisodeCurriculum):
    algorithm = "lp_acrl"


class ALPEpisodeCurriculum(_FiniteEpisodeCurriculum):
    algorithm = "alp"


Uniform = UniformEpisodeCurriculum
LPACRL = LPACRLEpisodeCurriculum
ALP = ALPEpisodeCurriculum
