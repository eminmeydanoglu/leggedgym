"""Pure-torch contracts for the V7 shared-HIM and flat-prior protocol."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from legged_gym.envs.go2.go2_v6_frontier_config import (
    Go2V6FrontierHIMFixedCfg,
    build_frontier_teacher,
)
from legged_gym.envs.go2.go2_v7_lpacrl_him_config import (
    Go2V7FlatLPACRLHIMCfg,
    Go2V7LPACRLHIMCfg,
    Go2V7UniformHIMCfg,
    build_v7_teacher,
    v6_terrain_builder_parameters,
)
from legged_gym.utils.ued import EpisodeOutcomeBatch, TaskAssignmentBatch
from legged_gym.utils.v7.genesis_adapter import (
    GenesisV7FlatAdapter,
    GenesisV7SemanticAdapter,
)
from legged_gym.utils.v7.task_space import (
    V7SemanticTaskSpace,
    V7TerrainTaskSpec,
    V7VelocityTaskSpace,
)
from rsl_rl.modules.him_actor_critic import HIMActorCritic
from rsl_rl.runners.eval_adapter import load_deploy_components


class _TerrainSimulator:
    custom_origins = True

    def __init__(self, n: int = 16):
        self._terrain_origins = torch.zeros(10, 10, 3)
        self.terrain_types = torch.zeros(n, dtype=torch.long)
        self.terrain_levels = torch.zeros(n, dtype=torch.long)
        self.env_origins = torch.zeros(n, 3)
        self.base_pos = torch.zeros(n, 3)
        self.base_lin_vel = torch.zeros(n, 3)
        self.base_ang_vel = torch.zeros(n, 3)


class _FlatSimulator:
    custom_origins = False


def _assignments(task_id: int, n: int, revision: int = 0) -> TaskAssignmentBatch:
    return TaskAssignmentBatch(
        task_ids=np.full(n, task_id, dtype=np.int64),
        sampler_revision=revision,
        curriculum_stage=0,
        probabilities=np.full(n, 1.0, dtype=np.float64),
        sources=np.full(n, "test", dtype="U10"),
    )


def _outcomes(task_ids: np.ndarray, returns: np.ndarray, revision: int = 0) -> EpisodeOutcomeBatch:
    count = len(task_ids)
    return EpisodeOutcomeBatch(
        task_ids=task_ids.astype(np.int64),
        assigned_revision=np.full(count, revision, dtype=np.int64),
        completion_revision=revision,
        episodic_returns=returns.astype(np.float64),
        episode_lengths=np.ones(count, dtype=np.int64),
        terminal_reasons=np.full(count, "timeout", dtype="U16"),
        completion_global_control_steps=revision + 1,
    )


def test_v7_him_contract_reuses_v6_terrain_commands_and_history_schema():
    v6, v7 = Go2V6FrontierHIMFixedCfg(), Go2V7LPACRLHIMCfg()
    assert v6_terrain_builder_parameters(v6) == v6_terrain_builder_parameters(v7)
    assert v6.commands.ranges.lin_vel_x == v7.commands.ranges.lin_vel_x == [-2.0, 2.0]
    assert v6.commands.ranges.lin_vel_y == v7.commands.ranges.lin_vel_y == [-1.0, 1.0]
    assert v6.commands.ranges.ang_vel_yaw == v7.commands.ranges.ang_vel_yaw == [-1.0, 1.0]
    assert v6.commands.heading_command is v7.commands.heading_command is False
    for attr in ("num_one_step_obs", "frame_stack", "num_observations", "c_frame_stack", "num_single_critic_obs", "num_privileged_obs"):
        assert getattr(v6.env, attr) == getattr(v7.env, attr)


def test_v7_semantic_task_space_is_240_cells_and_keeps_v6_family_columns():
    cfg = Go2V7LPACRLHIMCfg()
    _, v7_space = build_v7_teacher(cfg)
    _, v6_space = build_frontier_teacher(Go2V6FrontierHIMFixedCfg())
    assert isinstance(v7_space, V7SemanticTaskSpace)
    assert v7_space.size == 6 * 4 * 10 == 240
    assert v7_space.builder_parameters == v6_space.builder_parameters
    assert v7_space.terrain_bank_fingerprint() == v6_space.fingerprint()
    task_id = v7_space.encode(V7TerrainTaskSpec(terrain_family=3, terrain_level=7, vx_bin=2))
    assert v7_space.decode(task_id) == V7TerrainTaskSpec(3, 7, 2)
    assert v7_space.columns_for_family(3) == (3, 4, 5)


def test_semantic_adapter_samples_only_physical_columns_in_the_active_family():
    space = V7SemanticTaskSpace()
    n = 32
    commands = torch.zeros(n, 4)
    adapter = GenesisV7SemanticAdapter(
        task_space=space,
        simulator=_TerrainSimulator(n),
        commands=commands,
        command_ranges={},
        device="cpu",
    )
    task_id = space.encode(V7TerrainTaskSpec(terrain_family=3, terrain_level=7, vx_bin=1))
    adapter.assign(torch.arange(n), _assignments(task_id, n))
    assert torch.all(adapter.active_task_id == task_id)
    assert set(adapter.active_physical_column.tolist()).issubset({3, 4, 5})
    assert torch.all(adapter.simulator.terrain_levels == 7)


def test_flat_prior_has_four_velocity_cells_and_never_requires_terrain_origins():
    curriculum, space = build_v7_teacher(Go2V7FlatLPACRLHIMCfg())
    assert isinstance(space, V7VelocityTaskSpace)
    assert space.size == 4
    assert curriculum.adapter_class is GenesisV7FlatAdapter
    n = 8
    commands = torch.zeros(n, 4)
    adapter = GenesisV7FlatAdapter(
        task_space=space,
        simulator=_FlatSimulator(),
        commands=commands,
        command_ranges={},
        device="cpu",
    )
    adapter.assign(torch.arange(n), _assignments(2, n))
    sign = adapter.active_vx_sign.clone()
    adapter.resample_commands_within_active_bin(torch.arange(n))
    first = commands[:, 0].clone()
    adapter.resample_commands_within_active_bin(torch.arange(n))
    assert torch.equal(adapter.active_vx_sign, sign)
    assert torch.all((commands[:, 0] * sign) >= 1.0)
    assert torch.all((commands[:, 0] * sign) < 1.5)
    assert torch.all((first * sign) >= 1.0)


def test_standstill_returns_do_not_reach_flat_lp_and_return_lp_changes_sampling():
    space = V7VelocityTaskSpace()
    from legged_gym.utils.v7.curriculum import V7LPACRLCurriculum

    cur = V7LPACRLCurriculum(
        space,
        stage_length_control_steps=1,
        beta=0.2,
        epsilon=0.0,
        max_cell_probability=1.0,
        min_stage_episodes_for_lp=2,
        lp_estimator="stage",
        seed=3,
        flat=True,
    )
    n = 3
    adapter = GenesisV7FlatAdapter(
        task_space=space,
        simulator=_FlatSimulator(),
        commands=torch.zeros(n, 4),
        command_ranges={},
        device="cpu",
    )
    adapter.assign_standstill(torch.arange(n), _assignments(0, n))
    adapter.episode_return[:] = 999.0
    adapter.episode_length[:] = 1
    adapter.record_standstill_outcomes(torch.arange(n))
    assert cur._task_completion_counts.sum() == 0
    ids = np.repeat(np.arange(4), 2)
    cur.observe(_outcomes(ids, np.full(8, 1.0), revision=0))
    cur.advance(1)
    cur.observe(_outcomes(ids, np.where(ids == 2, 6.0, 1.0), revision=1))
    snapshot = cur.advance(2)
    assert snapshot is not None
    assert snapshot.probabilities[2] > snapshot.probabilities[0]


def test_uniform_him_teacher_stays_uniform_after_outcomes():
    cur, space = build_v7_teacher(Go2V7UniformHIMCfg())
    ids = np.arange(space.size, dtype=np.int64)
    cur.observe(_outcomes(ids, np.linspace(0.0, 1.0, space.size)))
    snapshot = cur.advance(cur.stage_length_control_steps)
    assert snapshot is not None
    assert np.allclose(snapshot.probabilities, 1.0 / space.size)


def test_flat_source_budget_and_artifact_protocol_are_fixed():
    from legged_gym.envs.go2.go2_v7_lpacrl_him_config import Go2V7FlatLPACRLHIMCfgPPO

    runner = Go2V7FlatLPACRLHIMCfgPPO.runner
    assert runner.max_iterations == 1000
    assert runner.save_interval == 1000
    assert runner.eval_interval == 0


def test_v7_dashboard_geometry_keeps_semantic_240_cells_and_flat_velocity_axis():
    from lpacr.dashboard.v5_integration import V7DashboardBridge, dashboard_task_space

    semantic = V7SemanticTaskSpace()
    atlas_space = dashboard_task_space(semantic)
    assert atlas_space.dimensions == ("vx_bin", "starting_terrain_family", "starting_terrain_level")
    assert tuple(len(atlas_space.coordinates[axis]) for axis in atlas_space.dimensions) == (4, 6, 10)
    bridge = V7DashboardBridge.__new__(V7DashboardBridge)
    bridge.curriculum = SimpleNamespace(task_space=semantic)
    grid = bridge._to_frame_order(np.arange(semantic.size))
    assert grid.shape == (4, 6, 10)
    assert grid[2, 3, 7] == semantic.encode(V7TerrainTaskSpec(3, 7, 2))
    flat_space = dashboard_task_space(V7VelocityTaskSpace())
    assert flat_space.dimensions == ("vx_bin",)
    assert len(flat_space.coordinates["vx_bin"]) == 4


def _him() -> HIMActorCritic:
    return HIMActorCritic(
        270, 265, 45, 12,
        actor_hidden_dims=[8, 8],
        critic_hidden_dims=[8, 8],
        enc_hidden_dims=[8, 4, 2],
        tar_hidden_dims=[8, 4],
        num_prototype=4,
    )


def test_flat_init_loads_only_him_actor_estimator_and_rejects_incompatible_source(tmp_path):
    from legged_gym.scripts.train import load_v7_flat_initialization

    source, target = _him(), _him()
    source_dir = tmp_path / "flat"
    source_dir.mkdir()
    checkpoint = source_dir / "model_1000.pt"
    torch.save({"model_state_dict": source.state_dict(), "iter": 1000}, checkpoint)
    (source_dir / "run_manifest.json").write_text(json.dumps({
        "task": "go2_v7_flat_lpacrl_him", "training_seed": 11,
    }))

    critic_before = {key: value.detach().clone() for key, value in target.critic.state_dict().items()}

    class _Runner:
        device = "cpu"

        def deploy_state_prefixes(self):
            return ("actor.", "estimator.")

        def load_deploy_state(self, path, map_location=None):
            state = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
            load_deploy_components(target, state, self.deploy_state_prefixes(), path)

    provenance = load_v7_flat_initialization(_Runner(), "go2_v7_lpacrl_him", str(checkpoint))
    assert provenance["source_task"] == "go2_v7_flat_lpacrl_him"
    assert provenance["source_iteration"] == 1000
    assert provenance["source_seed"] == 11
    assert all(torch.equal(target.critic.state_dict()[key], value) for key, value in critic_before.items())
    assert all(torch.equal(target.actor.state_dict()[key], value) for key, value in source.actor.state_dict().items())

    actor_only = {f"actor.{key}": value for key, value in source.actor.state_dict().items()}
    torch.save({"model_state_dict": actor_only, "iter": 1000}, checkpoint)
    with pytest.raises(RuntimeError, match="estimator"):
        load_v7_flat_initialization(_Runner(), "go2_v7_lpacrl_him", str(checkpoint))
