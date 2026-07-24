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
    """Completed *moving-task* episodes handed back to the curriculum.

    Reserved-standstill episodes never appear here: the environment splits them
    off before calling ``observe`` (see the standstill mixture in
    ``legged_robot._assign_ued_batch``), so every outcome in this batch is a
    real LP cell with a finite return.  There is no per-episode validity flag.

    Learning-progress admission (see :meth:`_FiniteEpisodeCurriculum.observe`):
    only outcomes with ``assigned_revision`` equal to the currently open
    ``sampler_revision`` and ``episode_lengths > 0`` update stage return
    averages.  Late completions (assigned under a previous revision) stay in
    provenance counters but do not contaminate the open stage.
    """

    task_ids: np.ndarray
    assigned_revision: np.ndarray
    completion_revision: int
    episodic_returns: np.ndarray
    episode_lengths: np.ndarray
    terminal_reasons: np.ndarray


@dataclass(frozen=True)
class StageSnapshot:
    global_control_steps: int
    stage_index: int
    sampler_revision: int
    probabilities: np.ndarray
    previous_returns: np.ndarray
    current_returns: np.ndarray
    current_return_sems: np.ndarray
    learning_progress: np.ndarray
    effective_learning_progress: np.ndarray
    observed_masks: np.ndarray
    stage_episode_counts: np.ndarray
    task_assignment_counts: np.ndarray
    task_completion_counts: np.ndarray
    transition_occupancy: Mapping[str, int]
    diagnostics: Mapping[str, object]


class EpisodeCurriculum(Protocol):
    def sample(self, count: int, *, global_control_steps: int) -> TaskAssignmentBatch: ...
    def draw_placements(self, count: int, *, global_control_steps: int) -> TaskAssignmentBatch: ...
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
        epsilon: float = 0.0,
        temperature_mode: str = "fixed",
        beta_min: float = 0.75,
        beta_max: float = 8.0,
        beta_ema: float = 0.8,
        target_ess_ratio_min: float = 0.5,
        max_cell_probability: float = 0.08,
        min_stage_episodes_for_lp: int = 16,
        confidence_scale: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if stage_length_control_steps <= 0:
            raise ValueError("stage_length_control_steps must be positive")
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be finite and positive")
        if not np.isfinite(epsilon) or not (0.0 <= epsilon < 1.0):
            raise ValueError("epsilon must be finite and in [0, 1)")
        if temperature_mode not in {"fixed", "adaptive_ess"}:
            raise ValueError("temperature_mode must be 'fixed' or 'adaptive_ess'")
        if not np.isfinite(beta_min) or not np.isfinite(beta_max) or not (0 < beta_min <= beta_max):
            raise ValueError("beta bounds must be finite, positive, and ordered")
        if not np.isfinite(beta_ema) or not (0.0 <= beta_ema < 1.0):
            raise ValueError("beta_ema must be in [0, 1)")
        if not np.isfinite(target_ess_ratio_min) or not (0.0 < target_ess_ratio_min <= 1.0):
            raise ValueError("target_ess_ratio_min must be in (0, 1]")
        if not np.isfinite(max_cell_probability) or not (1.0 / task_space.size <= max_cell_probability <= 1.0):
            raise ValueError("max_cell_probability must be in [1/task_count, 1]")
        if isinstance(min_stage_episodes_for_lp, bool) or min_stage_episodes_for_lp < 2:
            raise ValueError("min_stage_episodes_for_lp must be an integer >= 2")
        if not np.isfinite(confidence_scale) or confidence_scale <= 0:
            raise ValueError("confidence_scale must be finite and positive")
        self.task_space = task_space
        self.stage_length_control_steps = int(stage_length_control_steps)
        self.beta = float(beta)
        # Uniform exploration floor mixed into the Eq. 7 softmax (0 disables it,
        # keeping the update byte-identical to the paper).  See ``advance``.
        self.epsilon = float(epsilon)
        self.temperature_mode = temperature_mode
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.beta_ema = float(beta_ema)
        self.target_ess_ratio_min = float(target_ess_ratio_min)
        self.max_cell_probability = float(max_cell_probability)
        self.min_stage_episodes_for_lp = int(min_stage_episodes_for_lp)
        self.confidence_scale = float(confidence_scale)
        self._effective_beta = (
            float(np.clip(beta, beta_min, beta_max))
            if temperature_mode == "adaptive_ess"
            else float(beta)
        )
        self._target_ess = float(task_space.size)
        self._signal_quality = 0.0
        self._ess_guard_uniform_mix = 0.0
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._n = task_space.size
        self._probabilities = np.full(self._n, 1.0 / self._n, dtype=np.float64)
        self._source_label = "bootstrap"
        self.stage_index = 0
        self.sampler_revision = 0
        self.stage_start_global_steps = 0
        self._previous_returns = np.full(self._n, np.nan, dtype=np.float64)
        self._current_returns = np.full(self._n, np.nan, dtype=np.float64)
        self._previous_return_sems = np.full(self._n, np.nan, dtype=np.float64)
        self._current_return_sems = np.full(self._n, np.nan, dtype=np.float64)
        self._learning_progress = np.full(self._n, np.nan, dtype=np.float64)
        self._effective_learning_progress = np.full(self._n, np.nan, dtype=np.float64)
        self._observed_masks = np.zeros(self._n, dtype=bool)
        self._stage_return_sums = np.zeros(self._n, dtype=np.float64)
        self._stage_return_sq_sums = np.zeros(self._n, dtype=np.float64)
        self._stage_episode_counts = np.zeros(self._n, dtype=np.int64)
        self._task_assignment_counts = np.zeros(self._n, dtype=np.int64)
        self._task_completion_counts = np.zeros(self._n, dtype=np.int64)
        self._transition_occupancy: dict[str, int] = {}
        # Completions assigned under a previous sampler_revision (stage already
        # closed).  Tracked for diagnostics only; never enter stage return sums.
        self._late_outcome_count = 0
        self._snapshots: list[StageSnapshot] = []

    @property
    def config_fingerprint(self) -> str:
        payload = {
            "algorithm": self.algorithm,
            "stage_length_control_steps": self.stage_length_control_steps,
            "beta": self.beta,
            "epsilon": self.epsilon,
            "temperature_mode": self.temperature_mode,
            "beta_min": self.beta_min,
            "beta_max": self.beta_max,
            "beta_ema": self.beta_ema,
            "target_ess_ratio_min": self.target_ess_ratio_min,
            "max_cell_probability": self.max_cell_probability,
            "min_stage_episodes_for_lp": self.min_stage_episodes_for_lp,
            "confidence_scale": self.confidence_scale,
        }
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

    def draw_placements(self, count: int, *, global_control_steps: int) -> TaskAssignmentBatch:
        """Draw terrain/vx cells for reserved-standstill episodes.

        Identical draw to :meth:`sample` -- LP-weighted, so standing exposure
        rides the same easy->hard ordering the curriculum discovers -- but
        bookkeeping-free: these draws only decide *where* a standstill episode
        stands.  They are never counted as curriculum assignments and their
        returns never feed learning progress.  The reserved bucket lives beside
        the LP task space, not inside it.
        """
        count = self._non_negative_integer(count, name="count")
        self._non_negative_integer(global_control_steps, name="global_control_steps")
        task_ids = self._rng.choice(self._n, size=count, p=self._probabilities).astype(np.int64, copy=False)
        return TaskAssignmentBatch(
            task_ids,
            self.sampler_revision,
            self.stage_index,
            self._probabilities[task_ids].copy(),
            np.full(count, "standstill", dtype="U10"),
        )

    def observe(self, outcomes: EpisodeOutcomeBatch) -> None:
        """Ingest completed moving-task episodes.

        Provenance (completion counts, transition occupancy) records every
        outcome.  Stage return / LP accumulators only accept outcomes assigned
        under the currently open ``sampler_revision``: late completions
        (previous revision) are censored from LP rather than written into the
        open stage, and assigned revisions ahead of the open sampler are
        rejected fail-closed.

        Callers pass only genuinely-run episodes; never-stepped startup ghosts
        are filtered upstream at the episode-lifecycle boundary (see
        ``LeggedRobot._observe_ued_outcomes``), not re-checked here.
        """
        arrays = (
            outcomes.task_ids, outcomes.assigned_revision, outcomes.episodic_returns,
            outcomes.episode_lengths, outcomes.terminal_reasons,
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
        if np.any(assigned_revisions > self.sampler_revision):
            raise ValueError(
                "outcome assigned_revision cannot be ahead of the open sampler revision"
            )
        completion_revision = self._non_negative_integer(
            outcomes.completion_revision, name="outcome completion_revision"
        )
        returns = np.asarray(outcomes.episodic_returns, dtype=np.float64)
        episode_lengths = np.asarray(outcomes.episode_lengths)
        if not np.issubdtype(episode_lengths.dtype, np.integer) or np.any(episode_lengths < 0):
            raise ValueError("outcome episode_lengths must be non-negative integers")
        # Every outcome is a real moving-task episode (standstill is split off
        # upstream), so all returns must be finite.  LP admission is narrower.
        if np.any(~np.isfinite(returns)):
            raise ValueError("curriculum outcomes require finite returns")
        ids = task_ids.astype(np.int64, copy=False)
        self._task_completion_counts += np.bincount(ids, minlength=self._n)
        for assigned in assigned_revisions:
            key = f"{int(assigned)}:{completion_revision}"
            self._transition_occupancy[key] = self._transition_occupancy.get(key, 0) + 1

        same_stage = assigned_revisions == self.sampler_revision
        late = ~same_stage
        self._late_outcome_count += int(late.sum())
        admitted = same_stage
        if not np.any(admitted):
            return
        admitted_ids = ids[admitted]
        admitted_returns = returns[admitted]
        self._stage_return_sums += np.bincount(
            admitted_ids, weights=admitted_returns, minlength=self._n
        )
        self._stage_return_sq_sums += np.bincount(
            admitted_ids, weights=np.square(admitted_returns), minlength=self._n
        )
        self._stage_episode_counts += np.bincount(admitted_ids, minlength=self._n)

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

    def _mix_epsilon(self, weights: np.ndarray) -> np.ndarray:
        if self.epsilon <= 0.0:
            return weights
        return (1.0 - self.epsilon) * weights + self.epsilon / self._n

    def _cap_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        """Project onto the simplex with a hard per-cell upper bound."""
        capped = probabilities.astype(np.float64, copy=True)
        free = np.ones(self._n, dtype=bool)
        remaining = 1.0
        while True:
            free_total = float(capped[free].sum())
            if free_total <= 0.0:
                break
            capped[free] *= remaining / free_total
            newly_capped = free & (capped > self.max_cell_probability)
            if not np.any(newly_capped):
                break
            capped[newly_capped] = self.max_cell_probability
            free[newly_capped] = False
            remaining = 1.0 - float(capped[~free].sum())
        return capped / capped.sum(dtype=np.float64)

    def _distribution(self, scores: np.ndarray, beta: float) -> np.ndarray:
        return self._cap_probabilities(
            self._mix_epsilon(self._softmax(scores, beta))
        )

    @staticmethod
    def _effective_sample_size(probabilities: np.ndarray) -> float:
        return float(1.0 / np.sum(np.square(probabilities), dtype=np.float64))

    def _ensure_minimum_ess(
        self, probabilities: np.ndarray, target_ess: float
    ) -> np.ndarray:
        """Blend toward uniform only when beta_max cannot satisfy the ESS guard."""
        if self._effective_sample_size(probabilities) >= target_ess:
            self._ess_guard_uniform_mix = 0.0
            return probabilities
        uniform = np.full(self._n, 1.0 / self._n, dtype=np.float64)
        lo, hi = 0.0, 1.0
        for _ in range(64):
            mid = (lo + hi) / 2.0
            mixed = (1.0 - mid) * probabilities + mid * uniform
            if self._effective_sample_size(mixed) < target_ess:
                lo = mid
            else:
                hi = mid
        self._ess_guard_uniform_mix = hi
        return (1.0 - hi) * probabilities + hi * uniform

    def _solve_beta_for_ess(self, scores: np.ndarray, target_ess: float) -> float:
        """Find the temperature whose epsilon-mixed distribution hits target ESS."""
        if np.ptp(scores) <= np.finfo(np.float64).eps:
            return self.beta_max

        def ess(beta: float) -> float:
            return self._effective_sample_size(self._distribution(scores, beta))

        if ess(self.beta_min) >= target_ess:
            return self.beta_min
        if ess(self.beta_max) <= target_ess:
            return self.beta_max
        lo, hi = self.beta_min, self.beta_max
        for _ in range(64):
            mid = float(np.sqrt(lo * hi))
            if ess(mid) < target_ess:
                lo = mid
            else:
                hi = mid
        return float(np.sqrt(lo * hi))

    def _adaptive_scores(
        self,
        progress: np.ndarray,
        progress_mask: np.ndarray,
        current_sems: np.ndarray,
    ) -> np.ndarray:
        eligible = (
            progress_mask
            & (self._stage_episode_counts >= self.min_stage_episodes_for_lp)
            & np.isfinite(current_sems)
            & np.isfinite(self._current_return_sems)
        )
        lp_sem = np.full(self._n, np.nan, dtype=np.float64)
        lp_sem[eligible] = np.sqrt(
            np.square(current_sems[eligible])
            + np.square(self._current_return_sems[eligible])
        )
        reliability = np.zeros(self._n, dtype=np.float64)
        magnitude = np.abs(progress[eligible])
        reliability[eligible] = magnitude / (
            magnitude + self.confidence_scale * lp_sem[eligible] + np.finfo(np.float64).eps
        )
        effective = np.zeros(self._n, dtype=np.float64)
        effective[eligible] = progress[eligible] * reliability[eligible]
        self._signal_quality = float(np.median(reliability[eligible])) if np.any(eligible) else 0.0
        self._target_ess = float(
            self._n
            - self._signal_quality
            * (self._n - self.target_ess_ratio_min * self._n)
        )
        return self._score(effective)

    def advance(self, global_control_steps: int) -> StageSnapshot | None:
        global_control_steps = self._non_negative_integer(global_control_steps, name="global_control_steps")
        if global_control_steps < self.stage_start_global_steps:
            raise ValueError("global_control_steps cannot move backwards")
        if global_control_steps - self.stage_start_global_steps < self.stage_length_control_steps:
            return None
        observed = self._stage_episode_counts > 0
        current = np.full(self._n, np.nan, dtype=np.float64)
        current[observed] = self._stage_return_sums[observed] / self._stage_episode_counts[observed]
        current_sems = np.full(self._n, np.nan, dtype=np.float64)
        variance_mask = self._stage_episode_counts >= 2
        counts = self._stage_episode_counts[variance_mask].astype(np.float64)
        centered_ss = (
            self._stage_return_sq_sums[variance_mask]
            - np.square(self._stage_return_sums[variance_mask]) / counts
        )
        sample_variances = np.maximum(centered_ss / (counts - 1.0), 0.0)
        current_sems[variance_mask] = np.sqrt(sample_variances / counts)
        # Eq. 6: learning progress is only defined for a cell with a real return
        # in TWO consecutive stages.  A cell without that delta is imputed LP = 0
        # -- the neutral value, not a fabricated reward: it writes nothing to the
        # return accumulators and yields the reference softmax weight e^{0/beta},
        # so an unobserved cell is neither boosted nor starved.
        progress_mask = observed & self._observed_masks
        progress = np.zeros(self._n, dtype=np.float64)
        progress[progress_mask] = current[progress_mask] - self._current_returns[progress_mask]
        if self.algorithm != "uniform":
            # Eq. 7: rebuild the WHOLE distribution as a softmax of LP over all of
            # T.  No probability is carried across stages -- the previous c_j is
            # not an input -- so a cell can never be permanently starved (an
            # absorbing state the old freeze-retained-mass update could reach).
            if self.temperature_mode == "adaptive_ess":
                scores = self._adaptive_scores(progress, progress_mask, current_sems)
                solved_beta = self._solve_beta_for_ess(scores, self._target_ess)
                # Temperature is multiplicative, so smooth in log space.
                smoothed_beta = float(np.exp(
                    self.beta_ema * np.log(self._effective_beta)
                    + (1.0 - self.beta_ema) * np.log(solved_beta)
                ))
                # EMA may lag dangerously when LP scale jumps.  Never use a
                # temperature below the one that satisfies the current stage's
                # ESS target; lag is allowed only in the safer, more-uniform
                # direction.
                self._effective_beta = max(smoothed_beta, solved_beta)
            else:
                scores = self._score(progress)
                self._effective_beta = self.beta
                self._target_ess = float(self._n)
                self._signal_quality = 0.0
            weights = (
                self._distribution(scores, self._effective_beta)
                if self.temperature_mode == "adaptive_ess"
                else self._mix_epsilon(self._softmax(scores, self._effective_beta))
            )
            if self.temperature_mode == "adaptive_ess":
                weights = self._ensure_minimum_ess(weights, self._target_ess)
            self._probabilities = weights
            self._source_label = "lp" if self.algorithm == "lp_acrl" else "alp"
        if not np.all(np.isfinite(self._probabilities)) or not np.isclose(self._probabilities.sum(), 1.0):
            raise ValueError("curriculum probabilities are not finite and normalized")
        # Report imputed cells as NaN LP so downstream analysis can tell an
        # imputed 0 from a genuinely measured zero learning progress.
        progress_report = np.full(self._n, np.nan, dtype=np.float64)
        progress_report[progress_mask] = progress[progress_mask]
        self._previous_returns = self._current_returns.copy()
        self._current_returns = current
        self._previous_return_sems = self._current_return_sems.copy()
        self._current_return_sems = current_sems
        self._learning_progress = progress_report
        effective_report = np.full(self._n, np.nan, dtype=np.float64)
        if self.algorithm != "uniform":
            effective_report[progress_mask] = scores[progress_mask]
        self._effective_learning_progress = effective_report
        self._observed_masks = observed
        self.stage_index += 1
        self.sampler_revision += 1
        self.stage_start_global_steps = int(global_control_steps)
        snapshot = StageSnapshot(
            int(global_control_steps), self.stage_index, self.sampler_revision, self._probabilities.copy(),
            self._previous_returns.copy(), self._current_returns.copy(), self._current_return_sems.copy(),
            self._learning_progress.copy(), self._effective_learning_progress.copy(),
            self._observed_masks.copy(), self._stage_episode_counts.copy(), self._task_assignment_counts.copy(),
            self._task_completion_counts.copy(), dict(self._transition_occupancy), self.diagnostics(),
        )
        self._snapshots.append(snapshot)
        self._stage_return_sums.fill(0.0)
        self._stage_return_sq_sums.fill(0.0)
        self._stage_episode_counts.fill(0)
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
            "min_cell_probability": float(np.min(p)),
            "temperature_mode": self.temperature_mode,
            "effective_beta": float(self._effective_beta),
            "target_ess": float(self._target_ess),
            "signal_quality": float(self._signal_quality),
            "ess_guard_uniform_mix": float(self._ess_guard_uniform_mix),
            "task_assignment_coverage": float(np.count_nonzero(self._task_assignment_counts) / self._n),
            "completed_outcome_coverage": float(np.count_nonzero(self._task_completion_counts) / self._n),
            "late_outcome_count": int(self._late_outcome_count),
        }

    def state_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION, "algorithm": self.algorithm,
            "task_space_fingerprint": self.task_space.fingerprint(), "config_fingerprint": self.config_fingerprint,
            "stage_index": self.stage_index, "sampler_revision": self.sampler_revision,
            "stage_start_global_steps": self.stage_start_global_steps, "probabilities": self._probabilities.copy(),
            "previous_returns": self._previous_returns.copy(), "current_returns": self._current_returns.copy(),
            "previous_return_sems": self._previous_return_sems.copy(), "current_return_sems": self._current_return_sems.copy(),
            "learning_progress": self._learning_progress.copy(), "observed_masks": self._observed_masks.copy(),
            "effective_learning_progress": self._effective_learning_progress.copy(),
            "stage_return_sums": self._stage_return_sums.copy(), "stage_return_sq_sums": self._stage_return_sq_sums.copy(),
            "stage_episode_counts": self._stage_episode_counts.copy(),
            "task_assignment_counts": self._task_assignment_counts.copy(), "task_completion_counts": self._task_completion_counts.copy(),
            "transition_occupancy": dict(self._transition_occupancy),
            "source_label": self._source_label,
            "effective_beta": self._effective_beta, "target_ess": self._target_ess,
            "signal_quality": self._signal_quality,
            "ess_guard_uniform_mix": self._ess_guard_uniform_mix,
            "rng_bit_generator_state": deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        validate_checkpoint_state(state, algorithm=self.algorithm, task_space_fingerprint=self.task_space.fingerprint(), config_fingerprint=self.config_fingerprint)
        arrays = (
            "probabilities", "previous_returns", "current_returns",
            "previous_return_sems", "current_return_sems", "learning_progress",
            "effective_learning_progress", "observed_masks", "stage_return_sums",
            "stage_return_sq_sums", "stage_episode_counts",
            "task_assignment_counts", "task_completion_counts",
        )
        count_arrays = {"stage_episode_counts", "task_assignment_counts", "task_completion_counts"}
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
        effective_beta = float(state["effective_beta"])
        target_ess = float(state["target_ess"])
        signal_quality = float(state["signal_quality"])
        ess_guard_uniform_mix = float(state["ess_guard_uniform_mix"])
        valid_beta = (
            self.beta_min <= effective_beta <= self.beta_max
            if self.temperature_mode == "adaptive_ess"
            else effective_beta > 0
        )
        if not np.isfinite(effective_beta) or not valid_beta:
            raise ValueError("checkpoint effective_beta is invalid")
        if not np.isfinite(target_ess) or not (1.0 <= target_ess <= self._n):
            raise ValueError("checkpoint target_ess is invalid")
        if not np.isfinite(signal_quality) or not (0.0 <= signal_quality <= 1.0):
            raise ValueError("checkpoint signal_quality is invalid")
        if not np.isfinite(ess_guard_uniform_mix) or not (0.0 <= ess_guard_uniform_mix <= 1.0):
            raise ValueError("checkpoint ess_guard_uniform_mix is invalid")
        try:
            occupancy = dict(state["transition_occupancy"])
        except (TypeError, ValueError) as exc:
            raise ValueError("checkpoint transition_occupancy must be a mapping") from exc
        transition_occupancy = {
            str(key): self._non_negative_integer(value, name="checkpoint transition occupancy")
            for key, value in occupancy.items()
        }
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
        self._effective_beta = effective_beta
        self._target_ess = target_ess
        self._signal_quality = signal_quality
        self._ess_guard_uniform_mix = ess_guard_uniform_mix
        self._transition_occupancy = transition_occupancy
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
