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


COMPLETION_RING_CAPACITY = 2_048


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

    The legacy stage estimator admits only outcomes with
    ``assigned_revision == sampler_revision`` to its stage return averages;
    late completions remain provenance-only there.  The rolling-completion
    estimator instead records every real moving completion in its own per-task
    ring, with no write to any stage accumulator.
    """

    task_ids: np.ndarray
    assigned_revision: np.ndarray
    completion_revision: int
    episodic_returns: np.ndarray
    episode_lengths: np.ndarray
    terminal_reasons: np.ndarray
    # All outcomes in one lifecycle batch complete at the same control step.
    # The default keeps direct core callers written before rolling completion
    # provenance compatible; the environment supplies the real value.
    completion_global_control_steps: int = 0


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
    eligible_masks: np.ndarray
    previous_stage_episode_counts: np.ndarray
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
        lp_estimator: str = "stage",
        rolling_completion_window: int = 64,
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
        if (
            isinstance(min_stage_episodes_for_lp, bool)
            or not isinstance(min_stage_episodes_for_lp, Integral)
            or min_stage_episodes_for_lp < 2
        ):
            raise ValueError("min_stage_episodes_for_lp must be an integer >= 2")
        if not np.isfinite(confidence_scale) or confidence_scale <= 0:
            raise ValueError("confidence_scale must be finite and positive")
        if not isinstance(lp_estimator, str) or lp_estimator not in {"stage", "rolling_completion"}:
            raise ValueError("lp_estimator must be 'stage' or 'rolling_completion'")
        if (
            isinstance(rolling_completion_window, bool)
            or not isinstance(rolling_completion_window, Integral)
            or rolling_completion_window < 1
            or 2 * rolling_completion_window > COMPLETION_RING_CAPACITY
        ):
            raise ValueError(
                "rolling_completion_window must be an integer >= 1 with "
                f"2 * K <= {COMPLETION_RING_CAPACITY}"
            )
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
        self._lp_estimator = lp_estimator
        self._rolling_window_size = int(rolling_completion_window)
        self._effective_beta = (
            float(np.clip(beta, beta_min, beta_max))
            if temperature_mode == "adaptive_ess"
            else float(beta)
        )
        self._target_ess = float(task_space.size)
        self._signal_quality = 0.0
        self._has_adaptive_signal = False
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
        self._eligible_masks = np.zeros(self._n, dtype=bool)
        self._previous_stage_episode_counts = np.zeros(self._n, dtype=np.int64)
        self._stage_return_sums = np.zeros(self._n, dtype=np.float64)
        self._stage_return_sq_sums = np.zeros(self._n, dtype=np.float64)
        self._stage_episode_counts = np.zeros(self._n, dtype=np.int64)
        self._completion_return_ring = np.zeros(
            (self._n, COMPLETION_RING_CAPACITY), dtype=np.float64
        )
        self._completion_length_ring = np.zeros(
            (self._n, COMPLETION_RING_CAPACITY), dtype=np.int64
        )
        self._completion_step_ring = np.zeros(
            (self._n, COMPLETION_RING_CAPACITY), dtype=np.int64
        )
        self._completion_ring_pos = np.zeros(self._n, dtype=np.int64)
        self._completion_ring_count = np.zeros(self._n, dtype=np.int64)
        self._rolling_ready_masks = np.zeros(self._n, dtype=bool)
        self._rolling_previous_return_sems = np.full(self._n, np.nan, dtype=np.float64)
        self._rolling_current_return_sems = np.full(self._n, np.nan, dtype=np.float64)
        self._rolling_reward_per_step_shadow = float("nan")
        self._task_assignment_counts = np.zeros(self._n, dtype=np.int64)
        self._task_completion_counts = np.zeros(self._n, dtype=np.int64)
        self._transition_occupancy: dict[str, int] = {}
        # Completions assigned under a previous sampler_revision (stage already
        # closed).  Tracked for diagnostics only; never enter stage return sums.
        self._late_outcome_count = 0
        # Curriculum-quality diagnostics (computed each advance, exposed via
        # ``diagnostics``).  Previous-stage sampling distribution is held in
        # memory only -- intentionally NOT persisted, so the first post-resume
        # stage degrades ``top10_overlap_prev`` to NaN rather than comparing
        # against a stale distribution.
        self._prev_stage_probabilities: np.ndarray | None = None
        self._top10_overlap_prev = float("nan")
        self._lp_reliability_median = float("nan")
        self._sampled_lp_mass = float("nan")
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
            "lp_estimator": self._lp_estimator,
            "rolling_completion_window": self._rolling_window_size,
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
        outcome.  The legacy ``stage`` estimator only admits outcomes assigned
        under the currently open ``sampler_revision``.  In
        ``rolling_completion`` mode every real moving completion is appended to
        its task's ring, independent of assignment/completion revision; it
        never enters either an old or the open stage accumulator.

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
        completion_global_control_steps = self._non_negative_integer(
            outcomes.completion_global_control_steps,
            name="outcome completion_global_control_steps",
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

        if self._lp_estimator == "rolling_completion":
            self._append_rolling_completions(
                ids, returns, episode_lengths.astype(np.int64, copy=False),
                completion_global_control_steps,
            )
            return
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

    def _append_rolling_completions(
        self,
        task_ids: np.ndarray,
        returns: np.ndarray,
        lengths: np.ndarray,
        completion_global_control_steps: int,
    ) -> None:
        """Append completions in lifecycle order without consuming RNG state."""
        for task_id, episodic_return, episode_length in zip(task_ids, returns, lengths):
            index = int(self._completion_ring_pos[task_id])
            self._completion_return_ring[task_id, index] = episodic_return
            self._completion_length_ring[task_id, index] = episode_length
            self._completion_step_ring[task_id, index] = completion_global_control_steps
            self._completion_ring_pos[task_id] = (index + 1) % COMPLETION_RING_CAPACITY
            self._completion_ring_count[task_id] = min(
                int(self._completion_ring_count[task_id]) + 1,
                COMPLETION_RING_CAPACITY,
            )
        self._rolling_ready_masks = (
            self._completion_ring_count >= 2 * self._rolling_window_size
        )

    @staticmethod
    def _window_sem(values: np.ndarray) -> float:
        if values.size < 2:
            return float("nan")
        return float(np.std(values, ddof=1) / np.sqrt(values.size))

    def _rolling_window_stats(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return two chronological completion-window means/SEMs and shadow RPS.

        The physical ring layout is not chronological after wrap-around.  For a
        task with a full ring, ``ring_pos`` is exactly its oldest item, so the
        last ``2K`` indices can be built modulo capacity without copying or
        touching the sampler RNG.
        """
        previous = np.full(self._n, np.nan, dtype=np.float64)
        current = np.full(self._n, np.nan, dtype=np.float64)
        previous_sems = np.full(self._n, np.nan, dtype=np.float64)
        current_sems = np.full(self._n, np.nan, dtype=np.float64)
        previous_rps = np.full(self._n, np.nan, dtype=np.float64)
        current_rps = np.full(self._n, np.nan, dtype=np.float64)
        window = self._rolling_window_size
        ready_ids = np.flatnonzero(
            self._completion_ring_count >= 2 * window
        )
        for task_id in ready_ids:
            end = int(self._completion_ring_pos[task_id])
            indices = (np.arange(-2 * window, 0, dtype=np.int64) + end) % COMPLETION_RING_CAPACITY
            returns = self._completion_return_ring[task_id, indices]
            lengths = self._completion_length_ring[task_id, indices]
            previous_values = returns[:window]
            current_values = returns[window:]
            previous[task_id] = float(np.mean(previous_values))
            current[task_id] = float(np.mean(current_values))
            previous_sems[task_id] = self._window_sem(previous_values)
            current_sems[task_id] = self._window_sem(current_values)
            reward_per_step = returns / np.maximum(lengths, 1)
            previous_rps[task_id] = float(np.mean(reward_per_step[:window]))
            current_rps[task_id] = float(np.mean(reward_per_step[window:]))
        return previous, current, previous_sems, current_sems, previous_rps, current_rps

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

    @staticmethod
    def _top10_overlap(current: np.ndarray, previous: np.ndarray, k: int = 10) -> float:
        """|top-k(current) ∩ top-k(previous)| / k over two distributions."""
        k = min(k, current.shape[0])
        current_top = set(np.argsort(current, kind="stable")[-k:].tolist())
        previous_top = set(np.argsort(previous, kind="stable")[-k:].tolist())
        return float(len(current_top & previous_top)) / float(k)

    def _update_curriculum_diagnostics(
        self,
        progress: np.ndarray,
        progress_mask: np.ndarray,
        current_sems: np.ndarray,
        *,
        eligible_masks: np.ndarray | None = None,
        previous_sems: np.ndarray | None = None,
    ) -> None:
        """Compute side-effect-free curriculum-quality diagnostics.

        Runs in BOTH temperature modes and never mutates the sampling
        distribution.  Called from ``advance`` after ``self._probabilities`` is
        finalized but before the previous-stage returns/sems/counts are rolled
        forward, so ``self._current_return_sems`` and
        ``self._previous_stage_episode_counts`` still hold the previous stage's
        values (matching the adaptive path's ``lp_sem`` inputs).
        """
        # top10_overlap_prev: NaN on the first real stage (no prior stage
        # distribution to compare against), and NaN on the first post-resume
        # stage because ``_prev_stage_probabilities`` is not checkpointed.
        if self._prev_stage_probabilities is not None:
            self._top10_overlap_prev = self._top10_overlap(
                self._probabilities, self._prev_stage_probabilities
            )
        else:
            self._top10_overlap_prev = float("nan")
        self._prev_stage_probabilities = self._probabilities.copy()

        # lp_reliability_median: median over eligible cells of
        # |LP| / (|LP| + lp_sem), lp_sem = sqrt(cur_sem^2 + prev_sem^2).
        eligible = (
            self._episode_gate(progress_mask, current_sems)
            if eligible_masks is None
            else eligible_masks
        )
        previous_sems = (
            self._current_return_sems if previous_sems is None else previous_sems
        )
        if np.any(eligible):
            lp_sem = np.sqrt(
                np.square(current_sems[eligible])
                + np.square(previous_sems[eligible])
            )
            magnitude = np.abs(progress[eligible])
            reliability = magnitude / (magnitude + lp_sem + np.finfo(np.float64).eps)
            self._lp_reliability_median = float(np.median(reliability))
        else:
            self._lp_reliability_median = float("nan")

        # sampled_lp_mass: fraction of TRUSTED positive LP the distribution
        # targets.  Restricted to gated cells so the metric answers "of the LP
        # we believe, how much are we sampling"; scored against raw LP it would
        # be dominated by one-episode noise and unreadable by construction.
        positive_lp = np.where(eligible, np.maximum(progress, 0.0), 0.0)
        total_positive = float(positive_lp.sum(dtype=np.float64))
        if total_positive > 0.0:
            self._sampled_lp_mass = float(
                np.sum(self._probabilities * positive_lp, dtype=np.float64) / total_positive
            )
        else:
            self._sampled_lp_mass = float("nan")

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

    def _episode_gate(
        self, progress_mask: np.ndarray, current_sems: np.ndarray
    ) -> np.ndarray:
        """Cells whose LP rests on enough episodes in BOTH stages to be real."""
        return (
            progress_mask
            & (self._stage_episode_counts >= self.min_stage_episodes_for_lp)
            & (self._previous_stage_episode_counts >= self.min_stage_episodes_for_lp)
            & np.isfinite(current_sems)
            & np.isfinite(self._current_return_sems)
        )

    def _adaptive_scores(
        self,
        progress: np.ndarray,
        progress_mask: np.ndarray,
        current_sems: np.ndarray,
        *,
        eligible_masks: np.ndarray | None = None,
        previous_sems: np.ndarray | None = None,
    ) -> np.ndarray:
        eligible = (
            self._episode_gate(progress_mask, current_sems)
            if eligible_masks is None
            else eligible_masks
        )
        self._eligible_masks = eligible
        previous_sems = (
            self._current_return_sems if previous_sems is None else previous_sems
        )
        lp_sem = np.full(self._n, np.nan, dtype=np.float64)
        lp_sem[eligible] = np.sqrt(
            np.square(current_sems[eligible])
            + np.square(previous_sems[eligible])
        )
        reliability = np.zeros(self._n, dtype=np.float64)
        magnitude = np.abs(progress[eligible])
        reliability[eligible] = magnitude / (
            magnitude + self.confidence_scale * lp_sem[eligible] + np.finfo(np.float64).eps
        )
        effective = np.zeros(self._n, dtype=np.float64)
        effective[eligible] = progress[eligible] * reliability[eligible]
        eligible_fraction = float(np.count_nonzero(eligible) / self._n)
        eligible_quality = (
            float(np.median(reliability[eligible])) if np.any(eligible) else 0.0
        )
        # Confidence in a few well-measured hot cells is not confidence in the
        # whole curriculum.  Coverage therefore gates the global controller.
        self._signal_quality = eligible_quality * eligible_fraction
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
        previous_stage_episode_counts = self._previous_stage_episode_counts.copy()
        if self._lp_estimator == "rolling_completion":
            (
                previous,
                current,
                previous_sems,
                current_sems,
                previous_rps,
                current_rps,
            ) = self._rolling_window_stats()
            observed = self._completion_ring_count > 0
            progress_mask = self._completion_ring_count >= 2 * self._rolling_window_size
            progress = np.zeros(self._n, dtype=np.float64)
            progress[progress_mask] = current[progress_mask] - previous[progress_mask]
            rolling_adaptive_eligible = (
                progress_mask
                & np.isfinite(current_sems)
                & np.isfinite(previous_sems)
            )
        else:
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
            # Eq. 6: LP is defined only for a cell with a real return in two
            # consecutive *stages*.  This branch deliberately preserves the
            # legacy estimator's exact admission and missing-data semantics.
            progress_mask = observed & self._observed_masks
            progress = np.zeros(self._n, dtype=np.float64)
            progress[progress_mask] = current[progress_mask] - self._current_returns[progress_mask]
            previous = self._current_returns
            previous_sems = self._current_return_sems
            previous_rps = np.full(self._n, np.nan, dtype=np.float64)
            current_rps = np.full(self._n, np.nan, dtype=np.float64)
            rolling_adaptive_eligible = None
        if self._lp_estimator == "rolling_completion" and self.algorithm == "uniform":
            self._eligible_masks = progress_mask.copy()
        if self.algorithm != "uniform":
            # Eq. 7: rebuild the WHOLE distribution as a softmax of LP over all of
            # T.  No probability is carried across stages -- the previous c_j is
            # not an input -- so a cell can never be permanently starved (an
            # absorbing state the old freeze-retained-mass update could reach).
            if self.temperature_mode == "adaptive_ess":
                scores = self._adaptive_scores(
                    progress,
                    progress_mask,
                    current_sems,
                    eligible_masks=rolling_adaptive_eligible,
                    previous_sems=previous_sems,
                )
                solved_beta = self._solve_beta_for_ess(scores, self._target_ess)
                if self._signal_quality > 0.0 and not self._has_adaptive_signal:
                    # Bootstrap/uniform stages contain no LP information and
                    # must not seed the EMA at beta_max.
                    smoothed_beta = solved_beta
                    self._has_adaptive_signal = True
                else:
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
                # The stage estimator's episode gate is not an adaptive-mode extra.  An LP built
                # from a one-episode mean is noise whose magnitude (|LP| ~ 6-9
                # here) dwarfs the steady-state signal (~0.5), and a bare
                # softmax at beta=1 turns e^{noise} into a runaway: the cell
                # with the loudest measurement error takes ~90% of the mass,
                # floods, regresses, and hands the mass to the next 1-episode
                # cell.  Gated-out cells fall back to the same neutral LP = 0
                # imputation unobserved cells get, so they keep the reference
                # weight e^0 rather than hijacking the distribution.
                self._eligible_masks = (
                    progress_mask
                    if self._lp_estimator == "rolling_completion"
                    else self._episode_gate(progress_mask, current_sems)
                )
                scores = self._score(np.where(self._eligible_masks, progress, 0.0))
                self._effective_beta = self.beta
                # Inert sentinels: nothing in this branch reads them, but the
                # checkpoint contract requires finite values, so they stay in
                # state and are suppressed at the reporting boundary instead
                # (see diagnostics()).
                self._target_ess = float(self._n)
                self._signal_quality = 0.0
            # The per-cell cap is a safety bound on the sampler, not on one
            # temperature controller; both modes go through _distribution.
            weights = self._distribution(scores, self._effective_beta)
            if self.temperature_mode == "adaptive_ess":
                weights = self._ensure_minimum_ess(weights, self._target_ess)
            self._probabilities = weights
            self._source_label = "lp" if self.algorithm == "lp_acrl" else "alp"
        if not np.all(np.isfinite(self._probabilities)) or not np.isclose(self._probabilities.sum(), 1.0):
            raise ValueError("curriculum probabilities are not finite and normalized")
        # Curriculum-quality diagnostics: computed against the previous stage's
        # state, which is still intact here (rolled forward just below).
        self._update_curriculum_diagnostics(
            progress,
            progress_mask,
            current_sems,
            eligible_masks=(
                self._eligible_masks if self._lp_estimator == "rolling_completion" else None
            ),
            previous_sems=previous_sems,
        )
        if self._lp_estimator == "rolling_completion":
            # Unlike the legacy stage path, incomplete rolling windows have an
            # explicit neutral LP score of 0 (not an unobserved-stage NaN).
            progress_report = progress.copy()
            self._previous_returns = previous.copy()
            self._current_returns = current.copy()
            self._previous_return_sems = previous_sems.copy()
            self._current_return_sems = current_sems.copy()
            self._rolling_previous_return_sems = previous_sems.copy()
            self._rolling_current_return_sems = current_sems.copy()
            ready_rps = current_rps[progress_mask] - previous_rps[progress_mask]
            self._rolling_reward_per_step_shadow = (
                float(np.mean(ready_rps)) if ready_rps.size else float("nan")
            )
            self._learning_progress = progress_report
            self._effective_learning_progress = (
                scores.copy() if self.algorithm != "uniform" else progress_report.copy()
            )
        else:
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
        closed_stage_episode_counts = self._stage_episode_counts.copy()
        self._previous_stage_episode_counts = closed_stage_episode_counts
        self.stage_index += 1
        self.sampler_revision += 1
        self.stage_start_global_steps = int(global_control_steps)
        snapshot = StageSnapshot(
            int(global_control_steps), self.stage_index, self.sampler_revision, self._probabilities.copy(),
            self._previous_returns.copy(), self._current_returns.copy(), self._current_return_sems.copy(),
            self._learning_progress.copy(), self._effective_learning_progress.copy(),
            self._observed_masks.copy(), self._eligible_masks.copy(),
            previous_stage_episode_counts, closed_stage_episode_counts,
            self._task_assignment_counts.copy(),
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
        adaptive = self.temperature_mode == "adaptive_ess"
        return {
            "finite_probabilities": bool(np.all(np.isfinite(p))),
            "entropy": float(-np.sum(np.where(p > 0, p * np.log(p), 0.0))),
            "effective_sample_size": float(1.0 / np.sum(np.square(p))),
            "max_cell_probability": float(np.max(p)),
            "min_cell_probability": float(np.min(p)),
            "temperature_mode": self.temperature_mode,
            "lp_estimator": self._lp_estimator,
            "rolling_completion_window": int(self._rolling_window_size),
            "rolling_completion_total": int(self._completion_ring_count.sum()),
            "rolling_ready_task_count": int(np.count_nonzero(self._rolling_ready_masks)),
            "rolling_lp_nonzero_task_count": int(
                np.count_nonzero(self._rolling_ready_masks & (self._learning_progress != 0.0))
            ),
            "rolling_reward_per_step_shadow": float(self._rolling_reward_per_step_shadow),
            "effective_beta": float(self._effective_beta),
            # Both belong to the adaptive controller, which is dormant in fixed
            # mode (_ensure_minimum_ess runs only there).  Their inert sentinels
            # read as live values on a dashboard -- "target ESS 84" says we are
            # aiming for the uniform this mode exists to escape, and "signal
            # quality 0" contradicts a measured LP reliability of ~0.9 -- so
            # report them as NaN, which renders as n/a.
            "target_ess": float(self._target_ess) if adaptive else float("nan"),
            "signal_quality": float(self._signal_quality) if adaptive else float("nan"),
            "eligible_cell_count": int(np.count_nonzero(self._eligible_masks)),
            "eligible_fraction": float(np.count_nonzero(self._eligible_masks) / self._n),
            "ess_guard_uniform_mix": float(self._ess_guard_uniform_mix),
            "task_assignment_coverage": float(np.count_nonzero(self._task_assignment_counts) / self._n),
            "completed_outcome_coverage": float(np.count_nonzero(self._task_completion_counts) / self._n),
            "late_outcome_count": int(self._late_outcome_count),
            # Curriculum-quality diagnostics (see _update_curriculum_diagnostics).
            "top10_overlap_prev": float(self._top10_overlap_prev),
            "tv_distance_uniform": float(0.5 * np.sum(np.abs(p - 1.0 / self._n), dtype=np.float64)),
            "lp_reliability_median": float(self._lp_reliability_median),
            "sampled_lp_mass": float(self._sampled_lp_mass),

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
            "eligible_masks": self._eligible_masks.copy(),
            "lp_estimator": self._lp_estimator,
            "rolling_completion_window": self._rolling_window_size,
            "previous_stage_episode_counts": self._previous_stage_episode_counts.copy(),
            "stage_return_sums": self._stage_return_sums.copy(), "stage_return_sq_sums": self._stage_return_sq_sums.copy(),
            "stage_episode_counts": self._stage_episode_counts.copy(),
            "completion_return_ring": self._completion_return_ring.copy(),
            "completion_length_ring": self._completion_length_ring.copy(),
            "completion_step_ring": self._completion_step_ring.copy(),
            "completion_ring_pos": self._completion_ring_pos.copy(),
            "completion_ring_count": self._completion_ring_count.copy(),
            "rolling_ready_masks": self._rolling_ready_masks.copy(),
            "rolling_previous_return_sems": self._rolling_previous_return_sems.copy(),
            "rolling_current_return_sems": self._rolling_current_return_sems.copy(),
            "task_assignment_counts": self._task_assignment_counts.copy(), "task_completion_counts": self._task_completion_counts.copy(),
            "transition_occupancy": dict(self._transition_occupancy),
            "source_label": self._source_label,
            "effective_beta": self._effective_beta, "target_ess": self._target_ess,
            "signal_quality": self._signal_quality,
            "has_adaptive_signal": self._has_adaptive_signal,
            "ess_guard_uniform_mix": self._ess_guard_uniform_mix,
            "rng_bit_generator_state": deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        validate_checkpoint_state(state, algorithm=self.algorithm, task_space_fingerprint=self.task_space.fingerprint(), config_fingerprint=self.config_fingerprint)
        if state["lp_estimator"] != self._lp_estimator:
            raise ValueError("checkpoint lp_estimator does not match curriculum configuration")
        checkpoint_window = self._non_negative_integer(
            state["rolling_completion_window"],
            name="checkpoint rolling_completion_window",
        )
        if checkpoint_window != self._rolling_window_size:
            raise ValueError(
                "checkpoint rolling_completion_window does not match curriculum configuration"
            )
        arrays = (
            "probabilities", "previous_returns", "current_returns",
            "previous_return_sems", "current_return_sems", "learning_progress",
            "effective_learning_progress", "observed_masks", "stage_return_sums",
            "eligible_masks", "previous_stage_episode_counts",
            "stage_return_sq_sums", "stage_episode_counts",
            "task_assignment_counts", "task_completion_counts",
        )
        count_arrays = {
            "previous_stage_episode_counts", "stage_episode_counts",
            "task_assignment_counts", "task_completion_counts",
        }
        loaded_arrays: dict[str, np.ndarray] = {}
        for name in arrays:
            value = np.asarray(state[name])
            if value.shape != (self._n,):
                raise ValueError(f"checkpoint {name} has the wrong task-space shape")
            if name in {"observed_masks", "eligible_masks"}:
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
        ring_arrays = (
            "completion_return_ring", "completion_length_ring", "completion_step_ring",
        )
        loaded_rings: dict[str, np.ndarray] = {}
        expected_ring_shape = (self._n, COMPLETION_RING_CAPACITY)
        for name in ring_arrays:
            value = np.asarray(state[name])
            if value.shape != expected_ring_shape:
                raise ValueError(f"checkpoint {name} has the wrong ring shape")
            if name == "completion_return_ring":
                if not np.issubdtype(value.dtype, np.number) or not np.all(np.isfinite(value)):
                    raise ValueError("checkpoint completion_return_ring must be finite numeric values")
                loaded_rings[name] = value.astype(np.float64, copy=True)
            else:
                if not np.issubdtype(value.dtype, np.integer) or np.any(value < 0):
                    raise ValueError(f"checkpoint {name} must be non-negative integers")
                loaded_rings[name] = value.astype(np.int64, copy=True)
        ring_vectors = (
            "completion_ring_pos", "completion_ring_count", "rolling_ready_masks",
            "rolling_previous_return_sems", "rolling_current_return_sems",
        )
        loaded_ring_vectors: dict[str, np.ndarray] = {}
        for name in ring_vectors:
            value = np.asarray(state[name])
            if value.shape != (self._n,):
                raise ValueError(f"checkpoint {name} has the wrong task-space shape")
            if name == "rolling_ready_masks":
                if value.dtype != np.bool_:
                    raise ValueError("checkpoint rolling_ready_masks must be boolean")
            elif name in {"completion_ring_pos", "completion_ring_count"}:
                if not np.issubdtype(value.dtype, np.integer) or np.any(value < 0):
                    raise ValueError(f"checkpoint {name} must be non-negative integers")
            elif not np.issubdtype(value.dtype, np.number) or np.any(np.isinf(value)):
                raise ValueError(f"checkpoint {name} must be numeric without infinities")
            loaded_ring_vectors[name] = value.copy()
        ring_pos = loaded_ring_vectors["completion_ring_pos"]
        ring_count = loaded_ring_vectors["completion_ring_count"]
        if np.any(ring_pos >= COMPLETION_RING_CAPACITY):
            raise ValueError("checkpoint completion_ring_pos is outside the ring")
        if np.any(ring_count > COMPLETION_RING_CAPACITY):
            raise ValueError("checkpoint completion_ring_count exceeds ring capacity")
        expected_ready = ring_count >= 2 * self._rolling_window_size
        if not np.array_equal(loaded_ring_vectors["rolling_ready_masks"], expected_ready):
            raise ValueError("checkpoint rolling_ready_masks do not match ring counts")
        stage_index = self._non_negative_integer(state["stage_index"], name="checkpoint stage_index")
        sampler_revision = self._non_negative_integer(state["sampler_revision"], name="checkpoint sampler_revision")
        stage_start_global_steps = self._non_negative_integer(state["stage_start_global_steps"], name="checkpoint stage_start_global_steps")
        effective_beta = float(state["effective_beta"])
        target_ess = float(state["target_ess"])
        signal_quality = float(state["signal_quality"])
        has_adaptive_signal = state["has_adaptive_signal"]
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
        if not isinstance(has_adaptive_signal, (bool, np.bool_)):
            raise ValueError("checkpoint has_adaptive_signal must be boolean")
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
        for name, value in loaded_rings.items():
            setattr(self, "_" + name, value)
        for name, value in loaded_ring_vectors.items():
            setattr(self, "_" + name, value)
        self.stage_index = stage_index
        self.sampler_revision = sampler_revision
        self.stage_start_global_steps = stage_start_global_steps
        self._effective_beta = effective_beta
        self._target_ess = target_ess
        self._signal_quality = signal_quality
        self._has_adaptive_signal = bool(has_adaptive_signal)
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
