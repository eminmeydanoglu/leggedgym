"""Contracts for publishing *actual* V5 UED stage snapshots to the dashboard."""
from __future__ import annotations

import os
os.environ.setdefault("SIMULATOR", "genesis")

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.ued import EpisodeOutcomeBatch, LPACRLEpisodeCurriculum, TaskSpace
from lpacr.dashboard.v5_integration import (
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


def test_handcrafted_arm_is_a_meaningful_dashboard_noop():
    env = SimpleNamespace(cfg=SimpleNamespace(env=SimpleNamespace(ued_enabled=False)))
    assert create_v5_dashboard_bridge(
        env, task="go2_v5_handcrafted", training_seed=17,
        server_url="http://127.0.0.1:8765",
    ) is None
