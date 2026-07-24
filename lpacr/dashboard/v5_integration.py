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

from .plugger import CurriculumDashboardPlugger, TaskSpace as DashboardTaskSpace


def _velocity_labels(edges: tuple[float, ...]) -> list[str]:
    return [f"{lower:g}–{upper:g} m/s" for lower, upper in zip(edges, edges[1:])]


def dashboard_task_space(ued_task_space: Any) -> DashboardTaskSpace:
    """Expose V5's irregular 84-cell support without inventing invalid cells.

    V5 has 21 valid terrain cells (five terrain types at four levels plus one
    flat cell), crossed with four vx bins.  A synthetic 6×4 terrain grid would
    display 12 impossible tasks.  ``terrain_cell`` keeps the support exact and
    retaining ``vx_bin`` first makes C-order flattening identical to V5's
    stable ``task_id = vx_bin * 21 + terrain_cell`` encoding.
    """
    terrain_cells = []
    for task_id in range(21):
        spec = ued_task_space.decode(task_id)
        terrain_cells.append(
            f"{ued_task_space.terrain_type_names[spec.terrain_type]} · L{spec.terrain_level + 1}"
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


class V5DashboardBridge:
    """Publish completed V5 curriculum stages through the generic plugger."""

    def __init__(
        self,
        *,
        env: Any,
        task: str,
        training_seed: int | None,
        server_url: str,
        run_id: str | None = None,
    ) -> None:
        curriculum = getattr(env, "episode_curriculum", None)
        adapter = getattr(env, "ued_adapter", None)
        if curriculum is None or adapter is None:
            raise ValueError("V5 dashboard requires an enabled UED environment")
        self.env = env
        self.curriculum = curriculum
        self.adapter = adapter
        raw_run_id = run_id or f"{task}-{Path(str(getattr(env, 'dashboard_log_dir', task))).name}"
        self.run_id = _safe_run_id(raw_run_id)
        task_space = dashboard_task_space(curriculum.task_space)
        self.plugger = CurriculumDashboardPlugger(
            self.run_id,
            task_space,
            server_url=server_url,
            metadata={
                "source": "leggedgym_v5_ued",
                "task": task,
                "training_seed": training_seed,
                "curriculum_algorithm": curriculum.algorithm,
                "task_space_fingerprint": curriculum.task_space.fingerprint(),
                "curriculum_config_fingerprint": curriculum.config_fingerprint,
                "task_id_layout": "task_id = vx_bin * 21 + terrain_cell",
            },
        )

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

    def close(self) -> None:
        self.plugger.close()


def create_v5_dashboard_bridge(
    env: Any,
    *,
    task: str,
    training_seed: int | None,
    server_url: str,
    run_id: str | None = None,
) -> V5DashboardBridge | None:
    """Create a bridge for UED arms; handcrafted/non-UED tasks are a no-op."""
    if not bool(getattr(getattr(env, "cfg", None), "env", None) and getattr(env.cfg.env, "ued_enabled", False)):
        return None
    return V5DashboardBridge(
        env=env,
        task=task,
        training_seed=training_seed,
        server_url=server_url,
        run_id=run_id,
    )
