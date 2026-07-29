"""Pure NumPy type-balanced speed x terrain-level frontier curriculum."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Mapping

import numpy as np

from legged_gym.utils.ued.episode_curriculum import StageSnapshot, TaskAssignmentBatch
from .task_space import FrontierTaskSpec, V4FrontierTaskSpace


SCHEMA_VERSION = 3

# Bucket order used by the per-cell source counters below.
SAMPLING_SOURCES = ("frontier", "replay", "uniform")

# Cell lifecycle, exposed as one categorical layer.  Ordered so that a plain
# numeric render still reads as "how far along is this cell".
CELL_STATE_LOCKED = 0
CELL_STATE_FRONTIER = 1
CELL_STATE_MASTERED = 2
CELL_STATE_UNSTABLE = 3
CELL_STATE_NAMES = ("locked", "frontier", "mastered", "unstable")


@dataclass(frozen=True)
class FrontierOutcomeBatch:
    task_ids: np.ndarray
    assigned_revision: np.ndarray
    completion_revision: int
    episodic_returns: np.ndarray
    episode_lengths: np.ndarray
    terminal_reasons: np.ndarray
    successes: np.ndarray
    mean_linear_errors: np.ndarray
    mean_yaw_errors: np.ndarray
    completion_global_control_steps: int = 0
    on_tile_fractions: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    first_exit_steps: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    max_abs_dx: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    max_abs_dy: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )


class FrontierCurriculum:
    """Success-gated frontier over V4-style episode starting conditions.

    A cell identifies the terrain family/level where an episode starts plus its
    |vx| bin.  The static V4 heightfield remains traversable, so occupancy is
    observability rather than a claim that the full episode stayed on one tile.
    """

    algorithm = "frontier"

    def __init__(
        self,
        task_space: V4FrontierTaskSpace,
        *,
        update_interval_control_steps: int = 2_000,
        window_size: int = 32,
        min_episodes: int = 24,
        mastery_threshold: float = 0.80,
        unstable_threshold: float = 0.55,
        mastery_updates: int = 2,
        frontier_fraction: float = 0.60,
        replay_fraction: float = 0.30,
        uniform_fraction: float = 0.10,
        linear_error_threshold: float = 0.35,
        yaw_error_threshold: float = 0.40,
        seed: int | None = None,
    ) -> None:
        fractions = np.asarray(
            [frontier_fraction, replay_fraction, uniform_fraction], dtype=np.float64
        )
        if update_interval_control_steps <= 0:
            raise ValueError("update_interval_control_steps must be positive")
        if window_size < 2 or min_episodes < 2 or min_episodes > window_size:
            raise ValueError("require 2 <= min_episodes <= window_size")
        if not 0.0 < unstable_threshold < mastery_threshold < 1.0:
            raise ValueError("require 0 < unstable_threshold < mastery_threshold < 1")
        if mastery_updates < 1:
            raise ValueError("mastery_updates must be positive")
        if np.any(fractions < 0.0) or not np.isclose(fractions.sum(), 1.0):
            raise ValueError("sampling fractions must be non-negative and sum to one")
        if linear_error_threshold <= 0.0 or yaw_error_threshold <= 0.0:
            raise ValueError("tracking-error thresholds must be positive")

        self.task_space = task_space
        self.update_interval_control_steps = int(update_interval_control_steps)
        self.window_size = int(window_size)
        self.min_episodes = int(min_episodes)
        self.mastery_threshold = float(mastery_threshold)
        self.unstable_threshold = float(unstable_threshold)
        self.mastery_updates = int(mastery_updates)
        self.bucket_fractions = fractions
        self.linear_error_threshold = float(linear_error_threshold)
        self.yaw_error_threshold = float(yaw_error_threshold)
        self.rng = np.random.Generator(np.random.PCG64(seed))

        shape = (
            task_space.num_families,
            task_space.num_speed_bins,
            task_space.NUM_LEVELS,
        )
        self._shape = shape
        self._unlocked = np.zeros(shape, dtype=bool)
        self._mastered = np.zeros(shape, dtype=bool)
        self._unstable = np.zeros(shape, dtype=bool)
        self._consecutive_mastery = np.zeros(shape, dtype=np.int64)
        self._success_ring = np.zeros(shape + (window_size,), dtype=np.bool_)
        self._ring_pos = np.zeros(shape, dtype=np.int64)
        self._ring_count = np.zeros(shape, dtype=np.int64)
        self._success_probability = np.full(shape, 0.5, dtype=np.float64)
        self._unlocked[:, 0, 0] = True

        # Per-stage completions gate the mastery vote in ``_update_states``; the
        # rest of this block is observability that never feeds a decision, so it
        # can grow freely without changing the run.
        self._stage_completion_counts = np.zeros(shape, dtype=np.int64)
        self._previous_success_probability = np.full(shape, np.nan, dtype=np.float64)
        self._ring_count_at_last_update = np.zeros(shape, dtype=np.int64)
        self._source_counts = np.zeros(shape + (len(SAMPLING_SOURCES),), dtype=np.int64)
        self._standstill_place_counts = np.zeros(shape, dtype=np.int64)
        self._cell_completion_counts = np.zeros(shape, dtype=np.int64)
        self._previous_stage_completion_counts = np.zeros(shape, dtype=np.int64)
        self._unlocked_at_stage = np.full(shape, -1, dtype=np.int64)
        self._mastered_at_stage = np.full(shape, -1, dtype=np.int64)
        self._unlocked_at_stage[:, 0, 0] = 0
        # Continuous episode signals the mastery rule throws away.  An EWMA over
        # the same horizon as the success window keeps them comparable.
        self._ewma_alpha = 2.0 / (self.window_size + 1.0)
        self._ewma_linear_error = np.full(shape, np.nan, dtype=np.float64)
        self._ewma_yaw_error = np.full(shape, np.nan, dtype=np.float64)
        self._ewma_episode_length = np.full(shape, np.nan, dtype=np.float64)
        self._ewma_episodic_return = np.full(shape, np.nan, dtype=np.float64)
        self._ewma_timeout = np.full(shape, np.nan, dtype=np.float64)
        self._ewma_on_tile_fraction = np.full(shape, np.nan, dtype=np.float64)
        self._ewma_left_tile = np.full(shape, np.nan, dtype=np.float64)
        self._ewma_first_exit_step = np.full(shape, np.nan, dtype=np.float64)
        self._ewma_max_abs_dx = np.full(shape, np.nan, dtype=np.float64)
        self._ewma_max_abs_dy = np.full(shape, np.nan, dtype=np.float64)

        self.stage_index = 0
        self.sampler_revision = 0
        self.stage_start_global_steps = 0
        self._assignment_counts = np.zeros(task_space.size, dtype=np.int64)
        self._completion_counts = np.zeros(task_space.size, dtype=np.int64)
        self._last_probabilities = self._compute_task_probabilities()
        self._config_fingerprint = self._make_config_fingerprint()

    @property
    def config_fingerprint(self) -> str:
        return self._config_fingerprint

    @property
    def adapter_class(self):
        from .genesis_adapter import GenesisFrontierAdapter
        return GenesisFrontierAdapter

    @property
    def adapter_kwargs(self) -> dict[str, float]:
        builder = self.task_space.builder_parameters
        return {
            "linear_error_threshold": self.linear_error_threshold,
            "yaw_error_threshold": self.yaw_error_threshold,
            "terrain_length": float(builder.get("terrain_length", float("inf"))),
            "terrain_width": float(builder.get("terrain_width", float("inf"))),
        }

    def _make_config_fingerprint(self) -> str:
        payload = {
            "update_interval_control_steps": self.update_interval_control_steps,
            "window_size": self.window_size,
            "min_episodes": self.min_episodes,
            "mastery_threshold": self.mastery_threshold,
            "unstable_threshold": self.unstable_threshold,
            "mastery_updates": self.mastery_updates,
            "bucket_fractions": self.bucket_fractions.tolist(),
            "linear_error_threshold": self.linear_error_threshold,
            "yaw_error_threshold": self.yaw_error_threshold,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _cell_for_task(self, task_id: int) -> tuple[int, int, int]:
        spec = self.task_space.decode(int(task_id))
        return (
            self.task_space.family_for_column(spec.terrain_column),
            spec.speed_bin,
            spec.terrain_level,
        )

    def observe(self, outcomes: FrontierOutcomeBatch) -> None:
        ids = np.asarray(outcomes.task_ids)
        success = np.asarray(outcomes.successes)
        if ids.ndim != 1 or success.shape != ids.shape or success.dtype != np.bool_:
            raise ValueError("frontier outcomes require aligned task_ids and boolean successes")
        if np.any(ids < 0) or np.any(ids >= self.task_space.size):
            raise ValueError("frontier outcome task_id is outside the task space")
        np.add.at(self._completion_counts, ids, 1)
        signals = self._episode_signals(outcomes, len(ids))
        for index, (task_id, solved) in enumerate(zip(ids, success)):
            cell = self._cell_for_task(int(task_id))
            pos = int(self._ring_pos[cell])
            self._success_ring[cell + (pos,)] = bool(solved)
            self._ring_pos[cell] = (pos + 1) % self.window_size
            self._ring_count[cell] = min(int(self._ring_count[cell]) + 1, self.window_size)
            self._cell_completion_counts[cell] += 1
            self._stage_completion_counts[cell] += 1
            for array, column in signals:
                self._update_ewma(array, cell, float(column[index]))

    def _episode_signals(self, outcomes: FrontierOutcomeBatch, count: int):
        """Pair each EWMA accumulator with its per-episode column.

        Columns that a caller left unsized are skipped rather than guessed;
        observability must never be able to break ``observe``.
        """
        candidates = (
            (self._ewma_linear_error, outcomes.mean_linear_errors),
            (self._ewma_yaw_error, outcomes.mean_yaw_errors),
            (self._ewma_episode_length, outcomes.episode_lengths),
            (self._ewma_episodic_return, outcomes.episodic_returns),
            (self._ewma_timeout, np.asarray(outcomes.terminal_reasons) == "timeout"),
            (self._ewma_on_tile_fraction, outcomes.on_tile_fractions),
            (
                self._ewma_left_tile,
                np.asarray(outcomes.on_tile_fractions, dtype=np.float64) < 1.0,
            ),
            (self._ewma_first_exit_step, outcomes.first_exit_steps),
            (self._ewma_max_abs_dx, outcomes.max_abs_dx),
            (self._ewma_max_abs_dy, outcomes.max_abs_dy),
        )
        signals = []
        for array, column in candidates:
            values = np.asarray(column, dtype=np.float64).reshape(-1)
            if values.size == count:
                signals.append((array, values))
        return tuple(signals)

    def _update_ewma(self, array: np.ndarray, cell: tuple[int, int, int], value: float) -> None:
        if not np.isfinite(value):
            return
        current = array[cell]
        array[cell] = value if not np.isfinite(current) else (
            current + self._ewma_alpha * (value - current)
        )

    def _balanced_families(self, count: int) -> np.ndarray:
        base, remainder = divmod(count, self.task_space.num_families)
        families = np.repeat(np.arange(self.task_space.num_families), base)
        if remainder:
            families = np.concatenate(
                (families, self.rng.choice(self.task_space.num_families, remainder, replace=False))
            )
        self.rng.shuffle(families)
        return families

    def _sample_cell(self, family: int) -> tuple[int, int, str]:
        unlocked = np.argwhere(self._unlocked[family])
        frontier = np.argwhere(self._unlocked[family] & ~self._mastered[family])
        replay = np.argwhere(self._mastered[family])
        bucket = int(self.rng.choice(3, p=self.bucket_fractions))
        if bucket == 0 and len(frontier):
            pool, source = frontier, "frontier"
        elif bucket == 1 and len(replay):
            weights = 1.0 + 2.0 * self._unstable[family][tuple(replay.T)]
            index = int(self.rng.choice(len(replay), p=weights / weights.sum()))
            speed, level = replay[index]
            return int(speed), int(level), "replay"
        else:
            pool, source = unlocked, "uniform"
        speed, level = pool[int(self.rng.integers(len(pool)))]
        return int(speed), int(level), source

    def _draw(self, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Shared RNG core for assignments and standstill placements."""
        if count < 0:
            raise ValueError("sample count must be non-negative")
        families = self._balanced_families(count)
        task_ids = np.empty(count, dtype=np.int64)
        sources = np.empty(count, dtype="U16")
        cells = np.empty((count, 3), dtype=np.int64)
        for index, family in enumerate(families):
            speed, level, source = self._sample_cell(int(family))
            columns = self.task_space.columns_for_family(int(family))
            column = int(columns[int(self.rng.integers(len(columns)))])
            task_ids[index] = self.task_space.encode(FrontierTaskSpec(column, level, speed))
            sources[index] = source
            cells[index] = (int(family), speed, level)
        return task_ids, sources, cells

    def sample(self, count: int, *, global_control_steps: int) -> TaskAssignmentBatch:
        task_ids, sources, cells = self._draw(count)
        np.add.at(self._assignment_counts, task_ids, 1)
        if count:
            bucket = np.asarray(
                [SAMPLING_SOURCES.index(str(name)) for name in sources], dtype=np.int64
            )
            np.add.at(
                self._source_counts,
                (cells[:, 0], cells[:, 1], cells[:, 2], bucket),
                1,
            )
        probabilities = self._last_probabilities[task_ids]
        return TaskAssignmentBatch(
            task_ids=task_ids,
            sampler_revision=self.sampler_revision,
            curriculum_stage=self.stage_index,
            probabilities=probabilities.copy(),
            sources=sources,
        )

    def draw_placements(self, count: int, *, global_control_steps: int) -> TaskAssignmentBatch:
        # Placement only decides where a reserved standstill episode is born;
        # it is not a moving-task assignment and must not inflate cell coverage.
        task_ids, _, cells = self._draw(count)
        if count:
            np.add.at(self._standstill_place_counts, tuple(cells.T), 1)
        return TaskAssignmentBatch(
            task_ids=task_ids,
            sampler_revision=self.sampler_revision,
            curriculum_stage=self.stage_index,
            probabilities=self._last_probabilities[task_ids].copy(),
            sources=np.full(count, "standstill_place", dtype="U16"),
        )

    def _update_states(self) -> None:
        counts = self._ring_count
        successes = self._success_ring.sum(axis=-1, dtype=np.int64)
        # Beta(1,1) posterior mean prevents one rollout from reading as 0 or 1.
        # Keep the pre-update value so the dashboard can show a real per-stage
        # delta; the frontier rule itself has no learning-progress signal.
        self._previous_success_probability = np.where(
            self._ring_count_at_last_update > 0, self._success_probability, np.nan
        )
        self._success_probability = (successes + 1.0) / (counts + 2.0)
        self._ring_count_at_last_update = counts.copy()
        eligible = self._unlocked & (counts >= self.min_episodes)
        # A rolling window can remain unchanged across multiple curriculum
        # updates.  Only let a stage vote on mastery when that cell collected
        # enough new evidence during this stage; otherwise the same completed
        # episodes could satisfy ``mastery_updates`` more than once.
        evaluated = eligible & (self._stage_completion_counts >= self.min_episodes)
        above = evaluated & (self._success_probability >= self.mastery_threshold)
        self._consecutive_mastery[above] += 1
        self._consecutive_mastery[
            evaluated & (self._success_probability < self.mastery_threshold)
        ] = 0
        newly_mastered = (
            self._unlocked
            & ~self._mastered
            & (self._consecutive_mastery >= self.mastery_updates)
        )
        # ``stage_index`` is incremented by ``advance`` right after this call, so
        # the stamp names the stage whose data caused the transition.
        stamp = self.stage_index + 1
        for family, speed, level in np.argwhere(newly_mastered):
            self._mastered[family, speed, level] = True
            self._mastered_at_stage[family, speed, level] = stamp
            for neighbor in (
                (family, speed + 1, level) if speed + 1 < self.task_space.num_speed_bins else None,
                (family, speed, level + 1) if level + 1 < self.task_space.NUM_LEVELS else None,
            ):
                if neighbor is None or self._unlocked[neighbor]:
                    continue
                self._unlocked[neighbor] = True
                self._unlocked_at_stage[neighbor] = stamp
        self._unstable = (
            self._mastered & (counts >= self.min_episodes)
            & (self._success_probability < self.unstable_threshold)
        )

    def _compute_task_probabilities(self) -> np.ndarray:
        probabilities = np.zeros(self.task_space.size, dtype=np.float64)
        for family in range(self.task_space.num_families):
            cell_probabilities = self._cell_probabilities(family)
            columns = self.task_space.columns_for_family(family)
            for speed, level in np.argwhere(cell_probabilities > 0.0):
                for column in columns:
                    task_id = self.task_space.encode(
                        FrontierTaskSpec(int(column), int(level), int(speed))
                    )
                    probabilities[task_id] = (
                        cell_probabilities[speed, level]
                        / self.task_space.num_families
                        / len(columns)
                    )
        if not np.isclose(probabilities.sum(), 1.0):
            raise RuntimeError("frontier task probabilities are not normalized")
        return probabilities

    def _cell_probabilities(self, family: int) -> np.ndarray:
        """Exact marginal over cells after empty-bucket fallback."""
        distribution = np.zeros(
            (self.task_space.num_speed_bins, self.task_space.NUM_LEVELS),
            dtype=np.float64,
        )
        unlocked = self._unlocked[family]
        frontier = unlocked & ~self._mastered[family]
        replay = self._mastered[family]

        def add_uniform(mask: np.ndarray, mass: float) -> None:
            count = int(mask.sum())
            if count:
                distribution[mask] += mass / count

        if np.any(frontier):
            add_uniform(frontier, float(self.bucket_fractions[0]))
        else:
            add_uniform(unlocked, float(self.bucket_fractions[0]))

        if np.any(replay):
            replay_weights = (
                replay.astype(np.float64)
                * (1.0 + 2.0 * self._unstable[family].astype(np.float64))
            )
            distribution += (
                float(self.bucket_fractions[1]) * replay_weights / replay_weights.sum()
            )
        else:
            add_uniform(unlocked, float(self.bucket_fractions[1]))

        add_uniform(unlocked, float(self.bucket_fractions[2]))
        return distribution

    def advance(self, global_control_steps: int) -> StageSnapshot | None:
        if global_control_steps < self.stage_start_global_steps:
            raise ValueError("global control step cannot move backwards")
        if global_control_steps - self.stage_start_global_steps < self.update_interval_control_steps:
            return None
        self._update_states()
        self.stage_index += 1
        self.sampler_revision += 1
        self.stage_start_global_steps = int(global_control_steps)
        self._last_probabilities = self._compute_task_probabilities()
        flat_p = np.full(self.task_space.size, np.nan, dtype=np.float64)
        flat_counts = np.zeros(self.task_space.size, dtype=np.int64)
        for task_id in range(self.task_space.size):
            cell = self._cell_for_task(task_id)
            flat_p[task_id] = self._success_probability[cell]
            flat_counts[task_id] = self._ring_count[cell]
        # Roll the per-stage episode counters: after this call the "previous"
        # buffer describes the stage this snapshot closes.
        self._previous_stage_completion_counts = self._stage_completion_counts
        self._stage_completion_counts = np.zeros(self._shape, dtype=np.int64)
        return StageSnapshot(
            global_control_steps=int(global_control_steps),
            stage_index=self.stage_index,
            sampler_revision=self.sampler_revision,
            probabilities=self._last_probabilities.copy(),
            previous_returns=flat_p.copy(),
            current_returns=flat_p.copy(),
            current_return_sems=np.full(self.task_space.size, np.nan),
            learning_progress=np.full(self.task_space.size, np.nan),
            effective_learning_progress=np.full(self.task_space.size, np.nan),
            observed_masks=flat_counts > 0,
            eligible_masks=flat_counts >= self.min_episodes,
            previous_stage_episode_counts=flat_counts.copy(),
            stage_episode_counts=flat_counts,
            task_assignment_counts=self._assignment_counts.copy(),
            task_completion_counts=self._completion_counts.copy(),
            transition_occupancy={},
            diagnostics=self.diagnostics(),
        )

    def probabilities(self) -> np.ndarray:
        return self._last_probabilities.copy()

    def cell_states(self) -> np.ndarray:
        """Categorical lifecycle per cell; see the ``CELL_STATE_*`` constants."""
        states = np.full(self._shape, CELL_STATE_LOCKED, dtype=np.int64)
        states[self._unlocked & ~self._mastered] = CELL_STATE_FRONTIER
        states[self._mastered] = CELL_STATE_MASTERED
        states[self._unstable] = CELL_STATE_UNSTABLE
        return states

    def cell_probabilities(self) -> np.ndarray:
        """Sampling mass per (family, speed, level) cell, summing to one.

        Replicas are an implementation detail of the V4 terrain bank, not a
        curriculum axis, so this is the honest resolution of the decision.
        """
        distribution = np.zeros(self._shape, dtype=np.float64)
        for family in range(self.task_space.num_families):
            distribution[family] = (
                self._cell_probabilities(family) / self.task_space.num_families
            )
        return distribution

    def cell_metrics(self) -> dict[str, np.ndarray]:
        """Every per-cell quantity this curriculum knows, ``(family, speed, level)``.

        Nothing here feeds a decision -- it exists so the sampling distribution,
        the frontier and the episode signals behind them can be inspected after
        the fact.  Undefined entries are NaN rather than a fabricated zero.
        """
        source_counts = self._source_counts.astype(np.float64)
        source_total = source_counts.sum(axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            shares = np.where(
                source_total[..., None] > 0.0,
                source_counts / np.maximum(source_total[..., None], 1.0),
                np.nan,
            )
        never = -1
        unlocked_at = np.where(
            self._unlocked_at_stage == never, np.nan, self._unlocked_at_stage
        )
        mastered_at = np.where(
            self._mastered_at_stage == never, np.nan, self._mastered_at_stage
        )
        window_successes = self._success_ring.sum(axis=-1, dtype=np.int64)
        metrics = {
            "state": self.cell_states().astype(np.float64),
            "unlocked": self._unlocked.astype(np.float64),
            "mastered": self._mastered.astype(np.float64),
            "unstable": self._unstable.astype(np.float64),
            "success_probability": np.where(
                self._ring_count > 0, self._success_probability, np.nan
            ),
            "success_probability_delta": (
                self._success_probability - self._previous_success_probability
            ),
            "consecutive_mastery": self._consecutive_mastery.astype(np.float64),
            "window_episode_count": self._ring_count.astype(np.float64),
            "window_success_count": window_successes.astype(np.float64),
            "window_fill_fraction": self._ring_count / float(self.window_size),
            "episodes_until_eligible": np.maximum(
                0, self.min_episodes - self._ring_count
            ).astype(np.float64),
            "eligible": (self._unlocked & (self._ring_count >= self.min_episodes)).astype(
                np.float64
            ),
            "sampling_probability": self.cell_probabilities(),
            "assignment_count": source_total,
            "completion_count": self._cell_completion_counts.astype(np.float64),
            "completion_count_last_stage": (
                self._previous_stage_completion_counts.astype(np.float64)
            ),
            "standstill_placement_count": self._standstill_place_counts.astype(np.float64),
            "unlocked_at_stage": unlocked_at,
            "mastered_at_stage": mastered_at,
            "stages_since_unlock": np.where(
                np.isnan(unlocked_at), np.nan, self.stage_index - unlocked_at
            ),
            "stages_since_mastery": np.where(
                np.isnan(mastered_at), np.nan, self.stage_index - mastered_at
            ),
            "mean_linear_error": self._ewma_linear_error.copy(),
            "mean_yaw_error": self._ewma_yaw_error.copy(),
            "mean_episode_length": self._ewma_episode_length.copy(),
            "mean_episodic_return": self._ewma_episodic_return.copy(),
            "timeout_fraction": self._ewma_timeout.copy(),
            "on_tile_fraction": self._ewma_on_tile_fraction.copy(),
            "left_tile_fraction": self._ewma_left_tile.copy(),
            "mean_first_exit_step": self._ewma_first_exit_step.copy(),
            "max_abs_dx": self._ewma_max_abs_dx.copy(),
            "max_abs_dy": self._ewma_max_abs_dy.copy(),
        }
        for index, name in enumerate(SAMPLING_SOURCES):
            metrics[f"source_{name}_count"] = source_counts[..., index].copy()
            metrics[f"source_{name}_share"] = shares[..., index].copy()
        return metrics

    def frontier_edge(self) -> dict[str, list[int]]:
        """Deepest unlocked level and speed bin per family (-1 when nothing is)."""
        levels, speeds = [], []
        for family in range(self.task_space.num_families):
            unlocked = np.argwhere(self._unlocked[family])
            if len(unlocked) == 0:
                levels.append(-1)
                speeds.append(-1)
                continue
            speeds.append(int(unlocked[:, 0].max()))
            levels.append(int(unlocked[:, 1].max()))
        return {"max_unlocked_level": levels, "max_unlocked_speed_bin": speeds}

    def diagnostics(self) -> Mapping[str, object]:
        frontier = self._unlocked & ~self._mastered
        frontier_probabilities = np.isfinite(self._success_probability) & frontier
        source_totals = self._source_counts.sum(axis=(0, 1, 2)).astype(np.float64)
        assigned = float(source_totals.sum())
        edge = self.frontier_edge()
        return {
            "algorithm": self.algorithm,
            "unlocked_cell_count": int(self._unlocked.sum()),
            "mastered_cell_count": int(self._mastered.sum()),
            "unstable_cell_count": int(self._unstable.sum()),
            "frontier_cell_count": int(frontier.sum()),
            "eligible_cell_count": int(
                (self._unlocked & (self._ring_count >= self.min_episodes)).sum()
            ),
            "cell_count": int(self._unlocked.size),
            "completion_count": int(self._completion_counts.sum()),
            "assignment_count": int(assigned),
            "standstill_placement_count": int(self._standstill_place_counts.sum()),
            "terrain_family_count": self.task_space.num_families,
            "mean_frontier_success_probability": (
                float(self._success_probability[frontier_probabilities].mean())
                if np.any(frontier_probabilities)
                else float("nan")
            ),
            **{
                f"source_{name}_share": (
                    float(source_totals[index] / assigned) if assigned else float("nan")
                )
                for index, name in enumerate(SAMPLING_SOURCES)
            },
            **edge,
        }

    def _observability_shapes(self) -> dict[str, tuple[int, ...]]:
        """Arrays added for inspection that must survive a resume.

        ``stage_completion_counts`` is the exception that is not merely
        diagnostic: ``_update_states`` gates the mastery vote on it, so dropping
        it from a checkpoint would let a resumed run vote twice on one window.
        """
        return {
            "previous_success_probability": self._shape,
            "ring_count_at_last_update": self._shape,
            "source_counts": self._shape + (len(SAMPLING_SOURCES),),
            "standstill_place_counts": self._shape,
            "cell_completion_counts": self._shape,
            "stage_completion_counts": self._shape,
            "previous_stage_completion_counts": self._shape,
            "unlocked_at_stage": self._shape,
            "mastered_at_stage": self._shape,
            "ewma_linear_error": self._shape,
            "ewma_yaw_error": self._shape,
            "ewma_episode_length": self._shape,
            "ewma_episodic_return": self._shape,
            "ewma_timeout": self._shape,
            "ewma_on_tile_fraction": self._shape,
            "ewma_left_tile": self._shape,
            "ewma_first_exit_step": self._shape,
            "ewma_max_abs_dx": self._shape,
            "ewma_max_abs_dy": self._shape,
        }

    def state_dict(self) -> dict:
        return {
            **{
                name: getattr(self, "_" + name).copy()
                for name in self._observability_shapes()
            },
            "schema_version": SCHEMA_VERSION,
            "algorithm": self.algorithm,
            "task_space_fingerprint": self.task_space.fingerprint(),
            "config_fingerprint": self._config_fingerprint,
            "stage_index": self.stage_index,
            "sampler_revision": self.sampler_revision,
            "stage_start_global_steps": self.stage_start_global_steps,
            "unlocked": self._unlocked.copy(),
            "mastered": self._mastered.copy(),
            "unstable": self._unstable.copy(),
            "consecutive_mastery": self._consecutive_mastery.copy(),
            "success_ring": self._success_ring.copy(),
            "ring_pos": self._ring_pos.copy(),
            "ring_count": self._ring_count.copy(),
            "success_probability": self._success_probability.copy(),
            "assignment_counts": self._assignment_counts.copy(),
            "completion_counts": self._completion_counts.copy(),
            "last_probabilities": self._last_probabilities.copy(),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        expected = {
            "schema_version": SCHEMA_VERSION,
            "algorithm": self.algorithm,
            "task_space_fingerprint": self.task_space.fingerprint(),
            "config_fingerprint": self._config_fingerprint,
        }
        for name, value in expected.items():
            if state.get(name) != value:
                raise ValueError(f"frontier checkpoint {name} mismatch")
        shaped = {
            "unlocked": self._shape,
            "mastered": self._shape,
            "unstable": self._shape,
            "consecutive_mastery": self._shape,
            "ring_pos": self._shape,
            "ring_count": self._shape,
            "success_probability": self._shape,
            "success_ring": self._shape + (self.window_size,),
            "assignment_counts": (self.task_space.size,),
            "completion_counts": (self.task_space.size,),
            "last_probabilities": (self.task_space.size,),
            **self._observability_shapes(),
        }
        loaded = {}
        for name, shape in shaped.items():
            value = np.asarray(state[name])
            if value.shape != shape:
                raise ValueError(f"frontier checkpoint {name} has wrong shape")
            loaded[name] = value.copy()
        for name, value in loaded.items():
            setattr(self, "_" + name, value)
        self.stage_index = int(state["stage_index"])
        self.sampler_revision = int(state["sampler_revision"])
        self.stage_start_global_steps = int(state["stage_start_global_steps"])
        self.rng.bit_generator.state = state["rng_state"]
