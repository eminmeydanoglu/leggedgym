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
        (0,), (1,), (2,), (3, 4, 5), (6, 7), (8, 9)
    )
    assert space.size == 10 * 10 * 4
    assert space.ABS_SPEED_BIN_EDGES == (0.2, 0.5, 1.0, 1.5, 2.0)

    spec = FrontierTaskSpec(terrain_column=9, terrain_level=9, speed_bin=3)
    assert space.decode(space.encode(spec)) == spec
    decoded = space.decode_batch(np.asarray([space.encode(spec)]))
    assert decoded.terrain_types.tolist() == [9]
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


def test_mastery_requires_fresh_evidence_in_two_updates_then_unlocks_axis_neighbors_only():
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

    # Advancing the curriculum without a fresh completion must not count the
    # unchanged rolling window as a second mastery vote.
    second = curriculum.advance(2 * stage)
    assert second is not None
    assert curriculum.diagnostics()["mastered_cell_count"] == 0

    curriculum.observe(_outcomes(np.full(8, task_id), np.ones(8, dtype=np.bool_)))
    third = curriculum.advance(3 * stage)
    assert third is not None
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
    assert cfg.terrain.num_cols == 10
    assert cfg.terrain.curriculum is True
    assert cfg.terrain.terrain_replica_variation == 0.10
    assert cfg.curriculum.algorithm == "frontier"
    curriculum, _ = build_frontier_teacher(cfg)
    assert curriculum.adapter_kwargs == {
        "linear_error_threshold": 0.35,
        "yaw_error_threshold": 0.40,
        "terrain_length": 8.0,
        "terrain_width": 8.0,
    }


def test_frontier_dashboard_publishes_the_240_cell_decision_grid():
    """Replicas are drawn after the cell, so they are not an atlas axis."""
    space = V4FrontierTaskSpace()
    dashboard = dashboard_task_space(space)
    assert dashboard.size == 240
    assert dashboard.dimensions == (
        "vx_bin", "starting_terrain_family", "starting_terrain_level"
    )
    assert dashboard.coordinates["starting_terrain_family"] == list(
        space.TERRAIN_FAMILIES
    )
    assert dashboard.coordinates["starting_terrain_level"][0] == "L1"
    assert dashboard.coordinates["starting_terrain_level"][-1] == "L10"
    assert dashboard.coordinates["vx_bin"][0] == "0.2–0.5 m/s"


def _mastered_curriculum(family: int = 2):
    """Drive one family's origin cell to mastery so every state appears."""
    space = V4FrontierTaskSpace()
    curriculum = FrontierCurriculum(
        space,
        update_interval_control_steps=1_000,
        min_episodes=8,
        window_size=8,
        mastery_updates=2,
        seed=3,
    )
    column = space.columns_for_family(family)[0]
    task_id = space.encode(FrontierTaskSpec(column, 0, 0))
    curriculum.sample(120, global_control_steps=0)
    curriculum.observe(_outcomes(np.full(8, task_id), np.ones(8, dtype=np.bool_)))
    curriculum.advance(1_000)
    curriculum.observe(_outcomes(np.full(8, task_id), np.ones(8, dtype=np.bool_)))
    snapshot = curriculum.advance(2_000)
    return curriculum, space, snapshot, task_id


def test_cell_metrics_expose_state_coverage_and_sampling_at_cell_resolution():
    curriculum, space, _, _ = _mastered_curriculum(family=2)
    metrics = curriculum.cell_metrics()
    shape = (space.num_families, space.num_speed_bins, space.NUM_LEVELS)
    assert all(value.shape == shape for value in metrics.values())

    # Mastering (2, speed 0, level 0) opens exactly its two axis neighbours.
    assert metrics["state"][2, 0, 0] == 2.0  # mastered
    assert metrics["state"][2, 1, 0] == 1.0  # frontier
    assert metrics["state"][2, 0, 1] == 1.0  # frontier
    assert metrics["state"][2, 1, 1] == 0.0  # locked
    assert metrics["mastered_at_stage"][2, 0, 0] == 2
    assert np.isnan(metrics["mastered_at_stage"][2, 1, 0])
    assert metrics["unlocked_at_stage"][2, 0, 0] == 0

    # Sampling mass is a real distribution over the 240 decision cells.
    assert np.isclose(metrics["sampling_probability"].sum(), 1.0)
    assert metrics["sampling_probability"][2, 0, 0] > 0.0
    assert metrics["sampling_probability"][2, 1, 1] == 0.0

    # Unobserved cells stay NaN instead of reading as a confident zero.
    assert metrics["window_episode_count"][2, 0, 0] == 8.0
    assert np.isnan(metrics["success_probability"][0, 0, 1])
    assert metrics["success_probability"][2, 0, 0] > 0.8
    assert metrics["episodes_until_eligible"][2, 0, 0] == 0.0
    assert metrics["episodes_until_eligible"][0, 0, 1] == 8.0


def test_cell_metrics_track_sampling_bucket_and_standstill_provenance():
    cfg = Go2V6FrontierCfg()
    curriculum, space = build_frontier_teacher(cfg)
    curriculum.sample(600, global_control_steps=0)
    metrics = curriculum.cell_metrics()
    counts = np.stack([
        metrics[f"source_{name}_count"] for name in ("frontier", "replay", "uniform")
    ])
    assert counts.sum() == 600
    # Only the six origin cells are unlocked, so every draw lands there.
    assert metrics["assignment_count"][:, 0, 0].sum() == 600
    shares = np.stack([
        metrics[f"source_{name}_share"][:, 0, 0] for name in ("frontier", "replay", "uniform")
    ])
    np.testing.assert_allclose(shares.sum(axis=0), np.ones(6))
    # No cell is mastered yet, so the replay bucket falls back to uniform.
    assert np.all(shares[1] == 0.0)
    assert np.allclose(shares[0], 0.6, atol=0.1)

    before = metrics["assignment_count"].copy()
    curriculum.draw_placements(120, global_control_steps=0)
    after = curriculum.cell_metrics()
    np.testing.assert_array_equal(after["assignment_count"], before)
    assert after["standstill_placement_count"].sum() == 120


def test_cell_metrics_carry_the_episode_signals_the_mastery_rule_discards():
    space = V4FrontierTaskSpace()
    curriculum = FrontierCurriculum(space, min_episodes=2, window_size=4, seed=1)
    task_id = space.encode(FrontierTaskSpec(0, 0, 0))
    outcomes = _outcomes(np.full(4, task_id), np.ones(4, dtype=np.bool_))
    outcomes = FrontierOutcomeBatch(
        **{
            **outcomes.__dict__,
            "mean_linear_errors": np.full(4, 0.25),
            "mean_yaw_errors": np.full(4, 0.10),
            "episodic_returns": np.full(4, 7.0),
        }
    )
    curriculum.observe(outcomes)
    metrics = curriculum.cell_metrics()
    assert np.isclose(metrics["mean_linear_error"][0, 0, 0], 0.25)
    assert np.isclose(metrics["mean_yaw_error"][0, 0, 0], 0.10)
    assert np.isclose(metrics["mean_episodic_return"][0, 0, 0], 7.0)
    assert np.isclose(metrics["timeout_fraction"][0, 0, 0], 1.0)
    assert np.isnan(metrics["mean_linear_error"][1, 0, 0])


def test_diagnostics_report_the_frontier_edge_and_bucket_shares():
    curriculum, _, snapshot, _ = _mastered_curriculum(family=2)
    diagnostics = dict(snapshot.diagnostics)
    assert diagnostics["mastered_cell_count"] == 1
    assert diagnostics["frontier_cell_count"] == 7
    assert diagnostics["cell_count"] == 240
    assert diagnostics["max_unlocked_level"][2] == 1
    assert diagnostics["max_unlocked_speed_bin"][2] == 1
    assert diagnostics["max_unlocked_level"][0] == 0
    assert 0.0 <= diagnostics["source_frontier_share"] <= 1.0


def test_observability_state_survives_a_checkpoint_round_trip():
    cfg = Go2V6FrontierCfg()
    first, space = build_frontier_teacher(cfg)
    first.sample(200, global_control_steps=0)
    first.draw_placements(40, global_control_steps=0)
    task_id = space.encode(FrontierTaskSpec(0, 0, 0))
    first.observe(_outcomes(np.full(30, task_id), np.ones(30, dtype=np.bool_)))
    first.advance(first.update_interval_control_steps)

    restored, _ = build_frontier_teacher(cfg)
    restored.load_state_dict(first.state_dict())
    expected = first.cell_metrics()
    actual = restored.cell_metrics()
    assert set(actual) == set(expected)
    for name, values in expected.items():
        np.testing.assert_allclose(actual[name], values, equal_nan=True, err_msg=name)


class _FrontierSimulator:
    def __init__(self, count: int):
        self.custom_origins = True
        self._terrain_origins = torch.zeros(10, 10, 3)
        self._terrain_levels = torch.zeros(count, dtype=torch.long)
        self._terrain_types = torch.zeros(count, dtype=torch.long)
        self._env_origins = torch.zeros(count, 3)
        self.base_pos = torch.zeros(count, 3)
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
        terrain_length=8.0,
        terrain_width=8.0,
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
    simulator.base_pos[0, 0] = 41.0
    adapter.record_step(torch.zeros(count))
    timed_out = torch.ones(count, dtype=torch.bool)
    timed_out[0] = False
    outcome = adapter.collect_outcomes(
        torch.tensor([0, 1]), completion_revision=0, timed_out=timed_out
    )
    assert outcome.successes.tolist() == [False, True]
    np.testing.assert_allclose(outcome.on_tile_fractions, [0.0, 1.0])
    np.testing.assert_allclose(outcome.first_exit_steps[:1], [1.0])
    assert np.isnan(outcome.first_exit_steps[1])
    np.testing.assert_allclose(outcome.max_abs_dx, [41.0, 0.0])
    np.testing.assert_allclose(outcome.max_abs_dy, [0.0, 0.0])
