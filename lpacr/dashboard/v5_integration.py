"""Narrow V5/UED-to-Curriculum-Atlas bridge.

This module is intentionally the only place that knows both the V5 teacher's
``StageSnapshot`` shape and the generic dashboard transport.  Import it only
when the dashboard is explicitly enabled: ordinary training neither imports
the dashboard nor starts a background thread.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .plugger import CurriculumDashboardPlugger, TaskSpace as DashboardTaskSpace


def _velocity_labels(edges: tuple[float, ...]) -> list[str]:
    return [f"{lower:g}–{upper:g} m/s" for lower, upper in zip(edges, edges[1:])]


def is_frontier_task_space(ued_task_space: Any) -> bool:
    return hasattr(ued_task_space, "FAMILY_COLUMNS")


def is_v7_semantic_task_space(ued_task_space: Any) -> bool:
    """V7 stores LP directly in its semantic 240-cell identity."""
    return (
        type(ued_task_space).__module__.startswith("legged_gym.utils.v7")
        and hasattr(ued_task_space, "terrain_type_names")
        and hasattr(ued_task_space, "NUM_LEVELS")
    )


def is_v7_velocity_task_space(ued_task_space: Any) -> bool:
    """The flat V7 source has only an absolute-velocity axis."""
    return (
        type(ued_task_space).__module__.startswith("legged_gym.utils.v7")
        and not hasattr(ued_task_space, "terrain_type_names")
    )


def frontier_dashboard_task_space(ued_task_space: Any) -> DashboardTaskSpace:
    """Publish V6 at the resolution the curriculum actually decides on.

    The frontier's state lives on ``(family, speed_bin, level)``.  Terrain
    physical columns are interchangeable starting-terrain replicas of one V4
    family, drawn uniformly *after* the semantic cell is chosen.  Their counts
    are uneven in the native 10-column bank, so putting them on an axis would
    misrepresent the 240 real semantic decisions. Replica coverage is reported in the
    frame metadata instead.  ``vx_bin`` stays first so C-order flattening keeps
    matching the V5 convention of velocity as the outermost dimension.
    """
    return DashboardTaskSpace(
        dimensions=("vx_bin", "starting_terrain_family", "starting_terrain_level"),
        coordinates={
            "vx_bin": _velocity_labels(tuple(ued_task_space.velocity_bin_edges)),
            "starting_terrain_family": list(ued_task_space.terrain_type_names),
            "starting_terrain_level": [
                f"L{level + 1}" for level in range(int(ued_task_space.NUM_LEVELS))
            ],
        },
    )


def dashboard_task_space(ued_task_space: Any) -> DashboardTaskSpace:
    """Expose V5 or the V4-frontier support without inventing invalid cells.

    V5 has 21 valid terrain cells (five terrain types at four levels plus one
    flat cell), crossed with four vx bins.  A synthetic 6×4 terrain grid would
    display 12 impossible tasks.  ``terrain_cell`` keeps the support exact and
    retaining ``vx_bin`` first makes C-order flattening identical to V5's
    stable ``task_id = vx_bin * 21 + terrain_cell`` encoding.
    """
    if is_frontier_task_space(ued_task_space):
        return frontier_dashboard_task_space(ued_task_space)
    if is_v7_velocity_task_space(ued_task_space):
        return DashboardTaskSpace(
            dimensions=("vx_bin",),
            coordinates={
                "vx_bin": _velocity_labels(tuple(ued_task_space.velocity_bin_edges)),
            },
        )
    terrain_cells = []
    for task_id in range(21):
        spec = ued_task_space.decode(task_id)
        terrain_cells.append(
            f"{ued_task_space.terrain_type_names[spec.terrain_type]} · "
            f"L{spec.terrain_level + 1}"
        )
    return DashboardTaskSpace(
        dimensions=("vx_bin", "terrain_cell"),
        coordinates={
            "vx_bin": _velocity_labels(tuple(ued_task_space.velocity_bin_edges)),
            "terrain_cell": terrain_cells,
        },
    )


def _safe_run_id(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip(".-")
    return (value or "v5-ued")[:80]


class _DashboardBridge:
    """Shared plumbing: resolve the run id and open one plugger."""

    source = "leggedgym_v5_ued"
    task_id_layout = "task_id = vx_bin * 21 + terrain_cell"

    def __init__(
        self,
        *,
        env: Any,
        task: str,
        training_seed: int | None,
        server_url: str | None,
        run_id: str | None = None,
        local_dir: str | None = None,
    ) -> None:
        curriculum = getattr(env, "episode_curriculum", None)
        adapter = getattr(env, "ued_adapter", None)
        if curriculum is None or adapter is None:
            raise ValueError("UED dashboard requires an enabled UED environment")
        self.env = env
        self.curriculum = curriculum
        self.adapter = adapter
        raw_run_id = run_id or f"{task}-{Path(str(getattr(env, 'dashboard_log_dir', task))).name}"
        self.run_id = _safe_run_id(raw_run_id)
        # Prefer an explicit local_dir; fall back to <log_dir>/curriculum_atlas
        # so sample probabilities land next to model_*.pt even when Atlas is off.
        resolved_local = local_dir
        if resolved_local is None:
            log_dir = getattr(env, "dashboard_log_dir", None) or getattr(env, "log_dir", None)
            if log_dir:
                resolved_local = str(Path(log_dir) / "curriculum_atlas")
        self.local_dir = resolved_local
        self.plugger = CurriculumDashboardPlugger(
            self.run_id,
            dashboard_task_space(curriculum.task_space),
            server_url=server_url,
            local_dir=resolved_local,
            metadata={
                "source": self.source,
                "task": task,
                "training_seed": training_seed,
                "curriculum_algorithm": curriculum.algorithm,
                "task_space_fingerprint": curriculum.task_space.fingerprint(),
                "curriculum_config_fingerprint": curriculum.config_fingerprint,
                "task_id_layout": self.task_id_layout,
            },
        )

    def _standstill(self) -> dict:
        return dict(self.adapter.standstill_diagnostics())

    def publish(self, snapshot: Any) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        self.plugger.close()


class V5DashboardBridge(_DashboardBridge):
    """Publish completed V5 curriculum stages through the generic plugger."""

    def publish(self, snapshot: Any) -> bool:
        """Accept the actual snapshot returned by ``EpisodeCurriculum.advance``."""
        standstill = self.adapter.standstill_diagnostics()
        return self.plugger.log(
            snapshot.global_control_steps,
            {
                "performance": snapshot.current_returns,
                "performance_sem": snapshot.current_return_sems,
                "learning_progress": snapshot.learning_progress,
                "effective_learning_progress": snapshot.effective_learning_progress,
                "eligible_for_lp": snapshot.eligible_masks.astype(np.float64),
                "previous_stage_episode_count": snapshot.previous_stage_episode_counts,
                "sampling_probability": snapshot.probabilities,
                "stage_episode_count": snapshot.stage_episode_counts,
                "task_assignment_count": snapshot.task_assignment_counts,
                "task_completion_count": snapshot.task_completion_counts,
            },
            frame_metadata={
                "stage_index": snapshot.stage_index,
                "sampler_revision": snapshot.sampler_revision,
                "observed_cell_count": int(snapshot.observed_masks.sum()),
                "diagnostics": dict(snapshot.diagnostics),
                "standstill": dict(standstill),
            },
        )


class V7FlatDashboardBridge(V5DashboardBridge):
    """Flat V7 source prior: four |vx| cells, no terrain reshape."""

    source = "leggedgym_v7_flat_lpacrl"
    task_id_layout = "cell = abs_vx_bin (4 bins, no terrain)"


class V7DashboardBridge(V5DashboardBridge):
    """Publish V7 LP/Uniform snapshots in the semantic V6 atlas geometry."""

    source = "leggedgym_v7_lpacrl"
    task_id_layout = "cell = (starting_terrain_family, abs_vx_bin, starting_terrain_level)"

    def _to_frame_order(self, values: np.ndarray) -> np.ndarray:
        space = self.curriculum.task_space
        array = np.asarray(values)
        expected = space.num_families * space.num_speed_bins * space.NUM_LEVELS
        if array.size != expected:
            raise ValueError(f"V7 snapshot has {array.size} cells, expected {expected}")
        # V7's stable integer identity is (family, speed, level); Atlas keeps
        # velocity first, exactly like the V6 frontier display contract.
        return np.transpose(
            array.reshape(space.num_families, space.num_speed_bins, space.NUM_LEVELS),
            (1, 0, 2),
        )

    def publish(self, snapshot: Any) -> bool:
        standstill = self.adapter.standstill_diagnostics()
        fields = {
            "performance": snapshot.current_returns,
            "performance_sem": snapshot.current_return_sems,
            "learning_progress": snapshot.learning_progress,
            "effective_learning_progress": snapshot.effective_learning_progress,
            "eligible_for_lp": snapshot.eligible_masks.astype(np.float64),
            "previous_stage_episode_count": snapshot.previous_stage_episode_counts,
            "sampling_probability": snapshot.probabilities,
            "stage_episode_count": snapshot.stage_episode_counts,
            "task_assignment_count": snapshot.task_assignment_counts,
            "task_completion_count": snapshot.task_completion_counts,
        }
        return self.plugger.log(
            snapshot.global_control_steps,
            {name: self._to_frame_order(values) for name, values in fields.items()},
            frame_metadata={
                "stage_index": snapshot.stage_index,
                "sampler_revision": snapshot.sampler_revision,
                "observed_cell_count": int(snapshot.observed_masks.sum()),
                "diagnostics": dict(snapshot.diagnostics),
                "standstill": dict(standstill),
            },
        )


def _cell_state_names() -> list[str]:
    """Imported lazily so the dashboard package stays usable without legged_gym."""
    from legged_gym.utils.frontier.curriculum import CELL_STATE_NAMES

    return list(CELL_STATE_NAMES)


def _replica_balance(task_space: Any, task_counts: np.ndarray) -> dict[str, object]:
    """Per-column assignment totals, so a replica skew stays detectable.

    Replicas are off the atlas by design; this keeps the evidence that they are
    in fact drawn uniformly instead of asking the reader to trust it.
    """
    counts = np.asarray(task_counts, dtype=np.float64)
    columns = np.zeros(int(task_space.NUM_COLUMNS), dtype=np.float64)
    for task_id, value in enumerate(counts):
        columns[task_space.decode(task_id).terrain_column] += value
    ratios = []
    for family in range(task_space.num_families):
        pair = [columns[column] for column in task_space.columns_for_family(family)]
        total = sum(pair)
        ratios.append(float(max(pair) / total) if total else float("nan"))
    return {
        "column_assignment_counts": columns.tolist(),
        "max_replica_share": ratios,
    }


class FrontierDashboardBridge(_DashboardBridge):
    """Publish V6 frontier stages at ``(vx_bin, family, level)`` resolution."""

    source = "leggedgym_frontier"
    task_id_layout = (
        "cell = (abs_vx_bin, starting_terrain_family, starting_terrain_level)"
    )

    @staticmethod
    def _to_frame_order(values: np.ndarray) -> np.ndarray:
        """``(family, speed, level)`` → the frame's ``(vx_bin, family, level)``."""
        return np.transpose(np.asarray(values), (1, 0, 2))

    def publish(self, snapshot: Any) -> bool:
        cells = self.curriculum.cell_metrics()
        metrics = {name: self._to_frame_order(values) for name, values in cells.items()}
        return self.plugger.log(
            snapshot.global_control_steps,
            metrics,
            frame_metadata={
                "stage_index": snapshot.stage_index,
                "sampler_revision": snapshot.sampler_revision,
                "observed_cell_count": int(np.count_nonzero(cells["window_episode_count"])),
                "cell_state_names": _cell_state_names(),
                "diagnostics": dict(snapshot.diagnostics),
                "standstill": self._standstill(),
                "replica_balance": _replica_balance(
                    self.curriculum.task_space, snapshot.task_assignment_counts
                ),
            },
        )


def create_v5_dashboard_bridge(
    env: Any,
    *,
    task: str,
    training_seed: int | None,
    server_url: str | None = "http://127.0.0.1:8765",
    run_id: str | None = None,
    local_dir: str | None = None,
) -> _DashboardBridge | None:
    """Create a bridge for UED arms; handcrafted/non-UED tasks are a no-op.

    V5 and V6 publish different metric sets on different task spaces, so the
    curriculum picks the bridge; the dashboard reads both from the frame.

    ``server_url=None`` is local-only mode: stage frames (including
    ``sampling_probability``) are still written under ``local_dir`` / the run's
    ``curriculum_atlas/`` directory so headless cluster jobs keep an analysis
    trail without a live Node Atlas process.
    """
    if not bool(getattr(getattr(env, "cfg", None), "env", None) and getattr(env.cfg.env, "ued_enabled", False)):
        return None
    curriculum = getattr(env, "episode_curriculum", None)
    bridge_class = V5DashboardBridge
    if curriculum is not None and is_v7_semantic_task_space(curriculum.task_space):
        bridge_class = V7DashboardBridge
    elif curriculum is not None and is_v7_velocity_task_space(curriculum.task_space):
        bridge_class = V7FlatDashboardBridge
    elif curriculum is not None and is_frontier_task_space(curriculum.task_space):
        bridge_class = FrontierDashboardBridge
    return bridge_class(
        env=env,
        task=task,
        training_seed=training_seed,
        server_url=server_url,
        run_id=run_id,
        local_dir=local_dir,
    )
