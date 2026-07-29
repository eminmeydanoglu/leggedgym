"""Pure-torch contracts for the V6 frontier command and play geometry boundary."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from legged_gym.envs.go2.go2_v6_frontier_config import Go2V6FrontierCfg
from legged_gym.utils.frontier.genesis_adapter import GenesisFrontierAdapter
from legged_gym.utils.frontier.task_space import FrontierTaskSpec, V4FrontierTaskSpace
from legged_gym.utils.ued.episode_curriculum import TaskAssignmentBatch


class _Simulator:
    """The narrow simulator surface required by the assignment adapter."""

    def __init__(self, count: int):
        self.custom_origins = True
        self._terrain_origins = torch.zeros(10, 10, 3)
        self.terrain_types = torch.zeros(count, dtype=torch.long)
        self.terrain_levels = torch.zeros(count, dtype=torch.long)
        self.env_origins = torch.zeros(count, 3)


def _assignments(task_ids: np.ndarray) -> TaskAssignmentBatch:
    ids = np.asarray(task_ids, dtype=np.int64)
    return TaskAssignmentBatch(
        ids, sampler_revision=0, curriculum_stage=0,
        probabilities=np.ones(len(ids), dtype=np.float64),
        sources=np.full(len(ids), "test", dtype="U16"),
    )


def _adapter(count: int = 64) -> tuple[GenesisFrontierAdapter, torch.Tensor, V4FrontierTaskSpace]:
    commands = torch.zeros(count, 4)
    space = V4FrontierTaskSpace()
    return (
        GenesisFrontierAdapter(
            task_space=space, simulator=_Simulator(count), commands=commands,
            command_ranges={"lin_vel_x": [-2.0, 2.0]}, device="cpu",
        ),
        commands,
        space,
    )


def test_v6_moving_command_resamples_keep_episode_sign_and_active_bin():
    adapter, commands, space = _adapter()
    ids = torch.arange(64)
    task = space.encode(FrontierTaskSpec(terrain_column=0, terrain_level=0, speed_bin=2))
    torch.manual_seed(17)
    adapter.assign(ids, _assignments(np.full(64, task, dtype=np.int64)))
    signs_at_birth = adapter.active_vx_sign.clone()

    # vy/yaw are nuisance commands owned by the surrounding command sampler;
    # frontier vx resampling must neither clear nor take them out of contract.
    commands[:, 1] = torch.linspace(-1.0, 1.0, 64)
    commands[:, 2] = torch.linspace(1.0, -1.0, 64)
    for _ in range(12):
        adapter.resample_commands_within_active_bin(ids)
        assert torch.equal(adapter.active_vx_sign, signs_at_birth)
        assert torch.all(torch.sign(commands[:, 0]) == signs_at_birth)
        assert torch.all(torch.abs(commands[:, 0]) >= adapter.active_vx_lower)
        assert torch.all(torch.abs(commands[:, 0]) < adapter.active_vx_upper)
        assert torch.all(commands[:, 1].abs() <= 1.0)
        assert torch.all(commands[:, 2].abs() <= 1.0)


def test_v6_moving_assignment_cannot_become_standstill_from_a_command_threshold():
    adapter, commands, space = _adapter(128)
    ids = torch.arange(128)
    task = space.encode(FrontierTaskSpec(terrain_column=0, terrain_level=0, speed_bin=0))
    adapter.assign(ids, _assignments(np.full(128, task, dtype=np.int64)))
    moving_ids = adapter.apply_standstill_hold(ids)
    adapter.resample_commands_within_active_bin(moving_ids)

    assert torch.equal(moving_ids, ids)
    assert not torch.any(adapter.episode_standstill)
    # The lowest active bin begins at 0.2, so no resample can be thresholded
    # into the reserved standstill bucket.
    assert torch.all(commands[:, 0].abs() >= 0.2)


def test_v6_standstill_is_zero_for_the_entire_episode_across_resamples():
    adapter, commands, space = _adapter(8)
    ids = torch.arange(8)
    task = space.encode(FrontierTaskSpec(terrain_column=0, terrain_level=0, speed_bin=1))
    adapter.assign_standstill(ids, _assignments(np.full(8, task, dtype=np.int64)))
    for _ in range(10):
        commands[:, :3] = 0.73
        assert adapter.apply_standstill_hold(ids).numel() == 0
        assert torch.all(commands[:, :3] == 0.0)


def test_v6_vx_sign_is_approximately_balanced_across_assignments():
    adapter, _, space = _adapter(10_000)
    ids = torch.arange(10_000)
    task = space.encode(FrontierTaskSpec(terrain_column=0, terrain_level=0, speed_bin=1))
    torch.manual_seed(9)
    adapter.assign(ids, _assignments(np.full(10_000, task, dtype=np.int64)))
    positive_fraction = float((adapter.active_vx_sign > 0).float().mean())
    assert abs(positive_fraction - 0.5) < 0.03


def test_v6_training_geometry_is_native_8m_and_train_play_mode_copies_live_contract():
    from legged_gym.scripts import play as play_mod

    cfg = Go2V6FrontierCfg()
    assert (cfg.terrain.terrain_length, cfg.terrain.terrain_width) == (8.0, 8.0)

    terrain = SimpleNamespace(
        mesh_type="plane", curriculum=True, selected=False, terrain_kwargs={"type": "x"},
        num_rows=1, num_cols=1, border_size=2.0, max_init_terrain_level=0,
        fixed_terrain_level=None, terrain_length=1.0, terrain_width=1.0,
        platform_size=1.0, horizontal_scale=0.1, simplify_mesh=True, mode=None,
        taxonomy_showcase=False, ued_training_grid=True,
        terrain_proportions=[1.0], terrain_curriculum_difficulty={},
    )
    play_cfg = SimpleNamespace(terrain=terrain)
    play_mod.configure_play_terrain(play_cfg, "v6", v6_tile_size="train")

    assert play_cfg.terrain.terrain_length == cfg.terrain.terrain_length
    assert play_cfg.terrain.terrain_width == cfg.terrain.terrain_width
    assert play_cfg.terrain.platform_size == cfg.terrain.platform_size
    assert play_cfg.terrain.terrain_curriculum_difficulty == cfg.terrain.terrain_curriculum_difficulty
    assert play_cfg.terrain.terrain_replica_variation == cfg.terrain.terrain_replica_variation
