from __future__ import annotations

import numpy as np
import torch

from legged_gym.envs.go2.go2_v6_frontier_config import (
    Go2V6FrontierCfg,
    build_frontier_teacher,
)
from legged_gym.utils.frontier.curriculum import FrontierCurriculum, FrontierOutcomeBatch
from legged_gym.utils.frontier.task_space import (
    FrontierTaskSpec,
    V4FrontierTaskSpace,
)
from legged_gym.utils.frontier.genesis_adapter import GenesisFrontierAdapter
from legged_gym.utils.ued.episode_curriculum import TaskAssignmentBatch
from lpacr.dashboard.v5_integration import dashboard_task_space


def _outcomes(task_ids: np.ndarray, successes: np.ndarray, revision: int = 0):
    count = len(task_ids)
    return FrontierOutcomeBatch(
        task_ids=np.asarray(task_ids, dtype=np.int64),
        assigned_revision=np.full(count, revision, dtype=np.int64),
        completion_revision=revision,
        episodic_returns=np.zeros(count, dtype=np.float64),
        episode_lengths=np.full(count, 100, dtype=np.int64),
        terminal_reasons=np.full(count, "timeout", dtype="U16"),
        successes=np.asarray(successes, dtype=np.bool_),
        mean_linear_errors=np.zeros(count, dtype=np.float64),
        mean_yaw_errors=np.zeros(count, dtype=np.float64),
    )


def test_v4_frontier_task_space_matches_real_column_families():
    space = V4FrontierTaskSpace()
    assert space.FAMILY_COLUMNS == (
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)
    )
    assert space.size == 12 * 10 * 4
    assert space.ABS_SPEED_BIN_EDGES == (0.2, 0.5, 1.0, 1.5, 2.0)

    spec = FrontierTaskSpec(terrain_column=11, terrain_level=9, speed_bin=3)
    assert space.decode(space.encode(spec)) == spec
    decoded = space.decode_batch(np.asarray([space.encode(spec)]))
    assert decoded.terrain_types.tolist() == [11]
    assert decoded.terrain_levels.tolist() == [9]
    assert decoded.vx_lower.tolist() == [1.5]
    assert decoded.vx_upper.tolist() == [2.0]


def test_sampling_balances_semantic_families_not_replica_columns():
    cfg = Go2V6FrontierCfg()
    curriculum, space = build_frontier_teacher(cfg)
    batch = curriculum.sample(6_000, global_control_steps=0)
    families = np.asarray(
        [space.family_for_column(space.decode(task_id).terrain_column) for task_id in batch.task_ids]
    )
    np.testing.assert_array_equal(np.bincount(families, minlength=6), np.full(6, 1_000))
    assert {
        (space.decode(task_id).speed_bin, space.decode(task_id).terrain_level)
        for task_id in batch.task_ids
    } == {(0, 0)}


def test_mastery_requires_two_updates_then_unlocks_axis_neighbors_only():
    space = V4FrontierTaskSpace()
    curriculum = FrontierCurriculum(
        space,
        update_interval_control_steps=2_000,
        min_episodes=8,
        window_size=8,
        mastery_updates=2,
        seed=0,
    )
    task_id = space.encode(FrontierTaskSpec(terrain_column=0, terrain_level=0, speed_bin=0))
    curriculum.observe(_outcomes(np.full(8, task_id), np.ones(8, dtype=np.bool_)))

    stage = curriculum.update_interval_control_steps
    first = curriculum.advance(stage)
    assert first is not None
    assert curriculum.diagnostics()["mastered_cell_count"] == 0
    second = curriculum.advance(2 * stage)
    assert second is not None
    assert curriculum.diagnostics()["mastered_cell_count"] == 1
    # Initial cells for six families + the two axis neighbors opened for family 0.
    assert curriculum.diagnostics()["unlocked_cell_count"] == 8
    assert curriculum._unlocked[0, 1, 0]
    assert curriculum._unlocked[0, 0, 1]
    assert not curriculum._unlocked[0, 1, 1]


def test_frontier_checkpoint_round_trip_preserves_next_draw():
    cfg = Go2V6FrontierCfg()
    first, _ = build_frontier_teacher(cfg)
    first.sample(37, global_control_steps=0)
    state = first.state_dict()

    restored, _ = build_frontier_teacher(cfg)
    restored.load_state_dict(state)
    expected = first.sample(101, global_control_steps=0)
    actual = restored.sample(101, global_control_steps=0)
    np.testing.assert_array_equal(actual.task_ids, expected.task_ids)
    np.testing.assert_array_equal(actual.sources, expected.sources)


def test_standstill_placements_do_not_inflate_moving_assignment_counts():
    cfg = Go2V6FrontierCfg()
    curriculum, _ = build_frontier_teacher(cfg)
    before = int(curriculum._assignment_counts.sum())
    placements = curriculum.draw_placements(97, global_control_steps=0)
    assert int(curriculum._assignment_counts.sum()) == before
    assert set(placements.sources) == {"standstill_place"}


def test_v6_config_exposes_symmetric_vx_and_v4_grid():
    cfg = Go2V6FrontierCfg()
    assert cfg.commands.ranges.lin_vel_x == [-2.0, 2.0]
    assert cfg.terrain.num_rows == 10
    assert cfg.terrain.num_cols == 12
    assert cfg.terrain.curriculum is True
    assert cfg.terrain.terrain_replica_variation == 0.10
    assert cfg.curriculum.algorithm == "frontier"
    curriculum, _ = build_frontier_teacher(cfg)
    assert curriculum.adapter_kwargs == {
        "linear_error_threshold": 0.35,
        "yaw_error_threshold": 0.40,
    }


def test_frontier_dashboard_shape_matches_480_task_encoding():
    space = V4FrontierTaskSpace()
    dashboard = dashboard_task_space(space)
    assert dashboard.size == 480
    assert dashboard.dimensions == ("vx_bin", "terrain_cell")
    assert len(dashboard.coordinates["terrain_cell"]) == 120
    assert dashboard.coordinates["terrain_cell"][0] == "slope_up r1 · L1"
    assert dashboard.coordinates["terrain_cell"][119] == "discrete_obstacles r2 · L10"


class _FrontierSimulator:
    def __init__(self, count: int):
        self.custom_origins = True
        self._terrain_origins = torch.zeros(10, 12, 3)
        self._terrain_levels = torch.zeros(count, dtype=torch.long)
        self._terrain_types = torch.zeros(count, dtype=torch.long)
        self._env_origins = torch.zeros(count, 3)
        self.base_lin_vel = torch.zeros(count, 3)
        self.base_ang_vel = torch.zeros(count, 3)

    @property
    def terrain_levels(self):
        return self._terrain_levels

    @property
    def terrain_types(self):
        return self._terrain_types

    @property
    def env_origins(self):
        return self._env_origins


def test_frontier_adapter_samples_persistent_symmetric_vx_and_binary_success():
    torch.manual_seed(9)
    count = 240
    space = V4FrontierTaskSpace()
    simulator = _FrontierSimulator(count)
    commands = torch.zeros(count, 4)
    adapter = GenesisFrontierAdapter(
        task_space=space,
        simulator=simulator,
        commands=commands,
        command_ranges={
            "lin_vel_x": [-2.0, 2.0],
            "lin_vel_y": [-1.0, 1.0],
            "ang_vel_yaw": [-1.0, 1.0],
        },
        device="cpu",
    )
    task_id = space.encode(FrontierTaskSpec(0, 0, 0))
    assignments = TaskAssignmentBatch(
        task_ids=np.full(count, task_id, dtype=np.int64),
        sampler_revision=0,
        curriculum_stage=0,
        probabilities=np.full(count, 1.0 / count),
        sources=np.full(count, "frontier"),
    )
    ids = torch.arange(count)
    adapter.assign(ids, assignments)
    signs = adapter.active_vx_sign.clone()
    assert torch.any(signs < 0)
    assert torch.any(signs > 0)
    for _ in range(4):
        adapter.resample_commands_within_active_bin(ids)
        assert torch.equal(torch.sign(commands[:, 0]), signs)
        assert torch.all(torch.abs(commands[:, 0]) >= 0.2)
        assert torch.all(torch.abs(commands[:, 0]) < 0.5)

    # Perfect tracking for one step plus a timeout is a success; a terminal is not.
    simulator.base_lin_vel[:, :2] = commands[:, :2]
    simulator.base_ang_vel[:, 2] = commands[:, 2]
    adapter.record_step(torch.zeros(count))
    timed_out = torch.ones(count, dtype=torch.bool)
    timed_out[0] = False
    outcome = adapter.collect_outcomes(
        torch.tensor([0, 1]), completion_revision=0, timed_out=timed_out
    )
    assert outcome.successes.tolist() == [False, True]
