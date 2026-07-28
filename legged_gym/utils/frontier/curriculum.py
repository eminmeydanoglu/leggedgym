"""Pure NumPy type-balanced speed x terrain-level frontier curriculum."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

import numpy as np

from legged_gym.utils.ued.episode_curriculum import StageSnapshot, TaskAssignmentBatch
from .task_space import FrontierTaskSpec, V4FrontierTaskSpace


SCHEMA_VERSION = 1


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


class FrontierCurriculum:
    """Success-gated frontier shared across replicas of each terrain family."""

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
        return {
            "linear_error_threshold": self.linear_error_threshold,
            "yaw_error_threshold": self.yaw_error_threshold,
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
        for task_id, solved in zip(ids, success):
            cell = self._cell_for_task(int(task_id))
            pos = int(self._ring_pos[cell])
            self._success_ring[cell + (pos,)] = bool(solved)
            self._ring_pos[cell] = (pos + 1) % self.window_size
            self._ring_count[cell] = min(int(self._ring_count[cell]) + 1, self.window_size)

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

    def sample(self, count: int, *, global_control_steps: int) -> TaskAssignmentBatch:
        if count < 0:
            raise ValueError("sample count must be non-negative")
        families = self._balanced_families(count)
        task_ids = np.empty(count, dtype=np.int64)
        sources = np.empty(count, dtype="U16")
        for index, family in enumerate(families):
            speed, level, source = self._sample_cell(int(family))
            columns = self.task_space.columns_for_family(int(family))
            column = int(columns[int(self.rng.integers(len(columns)))])
            task_ids[index] = self.task_space.encode(FrontierTaskSpec(column, level, speed))
            sources[index] = source
        np.add.at(self._assignment_counts, task_ids, 1)
        probabilities = self._last_probabilities[task_ids]
        return TaskAssignmentBatch(
            task_ids=task_ids,
            sampler_revision=self.sampler_revision,
            curriculum_stage=self.stage_index,
            probabilities=probabilities.copy(),
            sources=sources,
        )

    def draw_placements(self, count: int, *, global_control_steps: int) -> TaskAssignmentBatch:
        batch = self.sample(count, global_control_steps=global_control_steps)
        # Placement only decides where a reserved standstill episode is born;
        # it is not a moving-task assignment and must not inflate cell coverage.
        np.add.at(self._assignment_counts, batch.task_ids, -1)
        return TaskAssignmentBatch(
            task_ids=batch.task_ids,
            sampler_revision=batch.sampler_revision,
            curriculum_stage=batch.curriculum_stage,
            probabilities=batch.probabilities,
            sources=np.full(count, "standstill_place", dtype="U16"),
        )

    def _update_states(self) -> None:
        counts = self._ring_count
        successes = self._success_ring.sum(axis=-1, dtype=np.int64)
        # Beta(1,1) posterior mean prevents one rollout from reading as 0 or 1.
        self._success_probability = (successes + 1.0) / (counts + 2.0)
        eligible = self._unlocked & (counts >= self.min_episodes)
        above = eligible & (self._success_probability >= self.mastery_threshold)
        self._consecutive_mastery[above] += 1
        self._consecutive_mastery[self._unlocked & ~above] = 0
        newly_mastered = (
            self._unlocked
            & ~self._mastered
            & (self._consecutive_mastery >= self.mastery_updates)
        )
        for family, speed, level in np.argwhere(newly_mastered):
            self._mastered[family, speed, level] = True
            if speed + 1 < self.task_space.num_speed_bins:
                self._unlocked[family, speed + 1, level] = True
            if level + 1 < self.task_space.NUM_LEVELS:
                self._unlocked[family, speed, level + 1] = True
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

    def diagnostics(self) -> Mapping[str, object]:
        return {
            "algorithm": self.algorithm,
            "unlocked_cell_count": int(self._unlocked.sum()),
            "mastered_cell_count": int(self._mastered.sum()),
            "unstable_cell_count": int(self._unstable.sum()),
            "completion_count": int(self._completion_counts.sum()),
            "terrain_family_count": self.task_space.num_families,
        }

    def state_dict(self) -> dict:
        return {
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
