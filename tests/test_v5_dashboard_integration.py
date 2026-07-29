"""Contracts for publishing *actual* V5 UED stage snapshots to the dashboard."""
from __future__ import annotations

import os
os.environ.setdefault("SIMULATOR", "genesis")

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.frontier.curriculum import FrontierCurriculum, FrontierOutcomeBatch
from legged_gym.utils.frontier.task_space import FrontierTaskSpec, V4FrontierTaskSpace
from legged_gym.utils.ued import EpisodeOutcomeBatch, LPACRLEpisodeCurriculum, TaskSpace
from lpacr.dashboard.v5_integration import (
    FrontierDashboardBridge,
    V5DashboardBridge,
    create_v5_dashboard_bridge,
    dashboard_task_space,
)


class _Plugger:
    instances = []

    def __init__(self, run_id, task_space, **kwargs):
        self.run_id = run_id
        self.task_space = task_space
        self.kwargs = kwargs
        self.calls = []
        self.closed = False
        self.instances.append(self)

    def log(self, step, metrics, **kwargs):
        self.calls.append((step, metrics, kwargs))
        return True

    def close(self):
        self.closed = True


def _snapshot():
    space = TaskSpace()
    curriculum = LPACRLEpisodeCurriculum(space, stage_length_control_steps=10, beta=5.0, seed=9)
    ids = np.asarray([0, 21], dtype=np.int64)
    curriculum.observe(EpisodeOutcomeBatch(
        task_ids=ids,
        assigned_revision=np.zeros(2, dtype=np.int64),
        completion_revision=0,
        episodic_returns=np.asarray([3.0, 5.0]),
        episode_lengths=np.ones(2, dtype=np.int64),
        terminal_reasons=np.asarray(["timeout", "terminal"]),
    ))
    snapshot = curriculum.advance(10)
    assert snapshot is not None
    return curriculum, snapshot


def test_dashboard_task_space_is_exact_v5_support_and_task_order():
    space = dashboard_task_space(TaskSpace())
    assert space.size == 84
    assert space.dimensions == ("vx_bin", "terrain_cell")
    assert space.coordinates["terrain_cell"][0] == "stairs_up · L1"
    assert space.coordinates["terrain_cell"][-1] == "flat · L1"


def test_real_stage_snapshot_is_forwarded_as_one_dashboard_frame():
    curriculum, snapshot = _snapshot()
    env = SimpleNamespace(
        cfg=SimpleNamespace(env=SimpleNamespace(ued_enabled=True)),
        episode_curriculum=curriculum,
        ued_adapter=SimpleNamespace(standstill_diagnostics=lambda: {"standstill_episode_count": 2.0}),
    )
    with patch("lpacr.dashboard.v5_integration.CurriculumDashboardPlugger", _Plugger):
        bridge = V5DashboardBridge(
            env=env, task="go2_v5_lpacrl", training_seed=17,
            server_url="http://127.0.0.1:8765", run_id="v5 integration",
        )
        assert bridge.publish(snapshot)
        plugger = _Plugger.instances[-1]

    assert plugger.run_id == "v5-integration"
    assert plugger.kwargs["metadata"]["curriculum_algorithm"] == "lp_acrl"
    assert len(plugger.calls) == 1
    step, metrics, kwargs = plugger.calls[0]
    assert step == 10
    np.testing.assert_allclose(metrics["sampling_probability"], snapshot.probabilities)
    np.testing.assert_allclose(metrics["performance"][:2], snapshot.current_returns[:2])
    assert kwargs["frame_metadata"]["stage_index"] == snapshot.stage_index
    assert kwargs["frame_metadata"]["standstill"]["standstill_episode_count"] == 2.0


def test_listener_receives_the_committed_stage_snapshot_and_observer_is_optional():
    _, snapshot = _snapshot()
    received = []
    shim = SimpleNamespace(_ued_snapshot_listener=received.append)
    LeggedRobot._emit_ued_snapshot(shim, snapshot)
    assert received == [snapshot]
    LeggedRobot.set_ued_snapshot_listener(shim, None)
    LeggedRobot._emit_ued_snapshot(shim, snapshot)
    assert received == [snapshot]


def _frontier_env(family: int = 3):
    space = V4FrontierTaskSpace()
    curriculum = FrontierCurriculum(
        space, update_interval_control_steps=10, min_episodes=4, window_size=4, seed=5
    )
    column = space.columns_for_family(family)[0]
    task_id = space.encode(FrontierTaskSpec(column, 0, 0))
    curriculum.sample(60, global_control_steps=0)
    curriculum.observe(FrontierOutcomeBatch(
        task_ids=np.full(4, task_id, dtype=np.int64),
        assigned_revision=np.zeros(4, dtype=np.int64),
        completion_revision=0,
        episodic_returns=np.full(4, 2.0),
        episode_lengths=np.full(4, 100, dtype=np.int64),
        terminal_reasons=np.full(4, "timeout", dtype="U16"),
        successes=np.ones(4, dtype=np.bool_),
        mean_linear_errors=np.full(4, 0.2),
        mean_yaw_errors=np.full(4, 0.1),
    ))
    snapshot = curriculum.advance(10)
    env = SimpleNamespace(
        cfg=SimpleNamespace(env=SimpleNamespace(ued_enabled=True)),
        episode_curriculum=curriculum,
        ued_adapter=SimpleNamespace(standstill_diagnostics=lambda: {"standstill_episode_count": 1.0}),
    )
    return env, curriculum, snapshot, space


def test_frontier_arm_selects_the_v6_bridge_and_v5_arm_keeps_its_own():
    env, _, _, _ = _frontier_env()
    with patch("lpacr.dashboard.v5_integration.CurriculumDashboardPlugger", _Plugger):
        bridge = create_v5_dashboard_bridge(
            env, task="go2_v6_frontier", training_seed=1,
            server_url="http://127.0.0.1:8765", run_id="v6-run",
        )
    assert isinstance(bridge, FrontierDashboardBridge)
    assert _Plugger.instances[-1].kwargs["metadata"]["source"] == "leggedgym_frontier"

    curriculum, _ = _snapshot()
    v5_env = SimpleNamespace(
        cfg=SimpleNamespace(env=SimpleNamespace(ued_enabled=True)),
        episode_curriculum=curriculum,
        ued_adapter=SimpleNamespace(standstill_diagnostics=dict),
    )
    with patch("lpacr.dashboard.v5_integration.CurriculumDashboardPlugger", _Plugger):
        v5_bridge = create_v5_dashboard_bridge(
            v5_env, task="go2_v5_lpacrl", training_seed=1,
            server_url="http://127.0.0.1:8765", run_id="v5-run",
        )
    assert isinstance(v5_bridge, V5DashboardBridge)
    assert _Plugger.instances[-1].kwargs["metadata"]["source"] == "leggedgym_v5_ued"


def test_frontier_frame_is_cell_shaped_and_transposed_into_vx_major_order():
    env, curriculum, snapshot, space = _frontier_env(family=3)
    with patch("lpacr.dashboard.v5_integration.CurriculumDashboardPlugger", _Plugger):
        bridge = FrontierDashboardBridge(
            env=env, task="go2_v6_frontier", training_seed=1,
            server_url="http://127.0.0.1:8765", run_id="v6-frame",
        )
        assert bridge.publish(snapshot)
    step, metrics, kwargs = _Plugger.instances[-1].calls[0]
    assert step == snapshot.global_control_steps

    cells = curriculum.cell_metrics()
    levels = space.NUM_LEVELS
    families = space.num_families
    for name, values in metrics.items():
        assert values.shape == (space.num_speed_bins, families, levels), name
        # (vx_bin, family, level) must be the (family, vx_bin, level) transpose.
        np.testing.assert_allclose(
            values.reshape(-1)[(0 * families + 3) * levels + 0],
            cells[name][3, 0, 0],
            equal_nan=True,
            err_msg=name,
        )
    assert metrics["state"][0, 3, 0] == 1.0  # observed but not yet mastered

    metadata = kwargs["frame_metadata"]
    assert metadata["cell_state_names"][2] == "mastered"
    assert metadata["standstill"]["standstill_episode_count"] == 1.0
    assert metadata["diagnostics"]["algorithm"] == "frontier"
    balance = metadata["replica_balance"]
    assert len(balance["column_assignment_counts"]) == 10
    assert sum(balance["column_assignment_counts"]) == 60
    assert all(0.0 <= share <= 1.0 for share in balance["max_replica_share"])


def test_handcrafted_arm_is_a_meaningful_dashboard_noop():
    env = SimpleNamespace(cfg=SimpleNamespace(env=SimpleNamespace(ued_enabled=False)))
    assert create_v5_dashboard_bridge(
        env, task="go2_v5_handcrafted", training_seed=17,
        server_url="http://127.0.0.1:8765",
    ) is None
