"""Registration and composition contract for all four go2_v6_frontier arms."""
from __future__ import annotations

import numpy as np
import pytest

import legged_gym.envs  # noqa: F401  -- register tasks
from legged_gym.envs.go2.go2_v3_dreamwaq import Go2V3Dreamwaq
from legged_gym.envs.go2.go2_v3_him_fixed import Go2V3HIMFixed
from legged_gym.envs.go2.go2_v3_mlp import Go2V3Mlp
from legged_gym.envs.go2.go2_v3_superset_oracle import Go2V3SupersetOracle
from legged_gym.envs.go2.go2_v6_frontier_config import (
    Go2V6FrontierCfg,
    Go2V6FrontierDreamwaqCfg,
    Go2V6FrontierHIMFixedCfg,
    Go2V6FrontierSupersetOracleCfg,
    build_frontier_teacher,
)
from legged_gym.utils.task_registry import task_registry

# Pinned digests for the mlp arm.  Any terrain/knob change that alters the
# checkpoint resume identity must update these literals deliberately.
MLP_TASK_SPACE_FINGERPRINT = (
    "8df54f32907b03362ec567ed8b05995a94c4496dea7fb0720ab19f72a86175ad"
)
MLP_CONFIG_FINGERPRINT = (
    "ef4777b83658e71be31a5ebee78789107b66de8aeb5521aa820ce31885a5d025"
)

V6_ARMS = {
    "go2_v6_frontier": {
        "env_cls": Go2V3Mlp,
        "cfg_cls": Go2V6FrontierCfg,
        "num_observations": 45,
        "num_privileged_obs": 48,
    },
    "go2_v6_frontier_mlp": {
        "env_cls": Go2V3Mlp,
        "cfg_cls": Go2V6FrontierCfg,
        "num_observations": 45,
        "num_privileged_obs": 48,
    },
    "go2_v6_frontier_oracle": {
        "env_cls": Go2V3SupersetOracle,
        "cfg_cls": Go2V6FrontierSupersetOracleCfg,
        "num_observations": 45 * 20 + 3 + 5 + 187,  # 1095
        "num_privileged_obs": 45 * 20 + 3 + 5 + 187,
    },
    "go2_v6_frontier_dreamwaq": {
        "env_cls": Go2V3Dreamwaq,
        "cfg_cls": Go2V6FrontierDreamwaqCfg,
        "num_observations": 45,
        "num_privileged_obs": 5 * 53,  # 265
    },
    "go2_v6_frontier_him": {
        "env_cls": Go2V3HIMFixed,
        "cfg_cls": Go2V6FrontierHIMFixedCfg,
        "num_observations": 6 * 45,  # 270
        "num_privileged_obs": 5 * 53,  # 265
    },
}

# Distinct architecture arms (mlp alias shares the default cfg).
V6_ARCHITECTURE_ARMS = (
    "go2_v6_frontier",
    "go2_v6_frontier_oracle",
    "go2_v6_frontier_dreamwaq",
    "go2_v6_frontier_him",
)


def _assert_v6_substrate(cfg) -> None:
    """Frontier terrain / commands / curriculum must not be shadowed."""
    assert cfg.terrain.num_rows == 10
    assert cfg.terrain.num_cols == 10
    assert list(cfg.terrain.terrain_proportions) == pytest.approx(
        [0.2, 0.1, 0.25, 0.25, 0.2]
    )
    assert (
        cfg.terrain.terrain_curriculum_difficulty["rough_height"]
        == "0.01 + 0.10 * difficulty"
    )
    assert cfg.terrain.curriculum is True
    assert cfg.terrain.selected is False
    assert cfg.terrain.ued_training_grid is False
    assert cfg.terrain.terrain_replica_variation == 0.10
    assert cfg.commands.ranges.lin_vel_x == [-2.0, 2.0]
    assert cfg.commands.legacy_performance_command_curriculum_enabled is False
    assert cfg.commands.per_env_standstill is True
    assert cfg.curriculum.algorithm == "frontier"
    assert cfg.env.ued_enabled is True
    assert cfg.env.num_actions == 12


@pytest.mark.parametrize("task_name", list(V6_ARMS))
def test_v6_arm_registration_and_env_class(task_name):
    expected = V6_ARMS[task_name]
    assert task_name in task_registry.task_classes
    assert task_registry.get_task_class(task_name) is expected["env_cls"]
    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    assert isinstance(env_cfg, expected["cfg_cls"])
    assert train_cfg.runner.experiment_name == "go2_v6_frontier"
    assert train_cfg.runner.command_schedule is None


@pytest.mark.parametrize("task_name", list(V6_ARMS))
def test_v6_arm_instantiated_cfg_composition(task_name):
    expected = V6_ARMS[task_name]
    # Instantiate via the cfg class (not only the registry singleton) so the
    # nested BaseConfig walk is exercised independently of registration order.
    cfg = expected["cfg_cls"]()
    _assert_v6_substrate(cfg)
    assert cfg.env.num_observations == expected["num_observations"]
    assert cfg.env.num_privileged_obs == expected["num_privileged_obs"]

    # Registry-resolved cfg must agree (task_registry may deep-copy).
    env_cfg, _ = task_registry.get_cfgs(name=task_name)
    _assert_v6_substrate(env_cfg)
    assert env_cfg.env.num_observations == expected["num_observations"]
    assert env_cfg.env.num_privileged_obs == expected["num_privileged_obs"]


def test_oracle_privileged_height_map_plumbing():
    cfg = Go2V6FrontierSupersetOracleCfg()
    assert cfg.env.oracle_height_map is True
    assert cfg.env.height_map_dim == 187
    assert cfg.env.num_observations == 45 * 20 + 3 + 5 + 187
    assert cfg.env.num_privileged_obs == cfg.env.num_observations
    # Height-map sampling depends on the V4 terrain measure grid.
    assert cfg.terrain.measure_heights is True
    assert len(cfg.terrain.measured_points_x) == 17
    assert len(cfg.terrain.measured_points_y) == 11


def test_dreamwaq_and_him_obs_contracts():
    dream = Go2V6FrontierDreamwaqCfg()
    assert dream.env.frame_stack == 5
    assert dream.env.num_latent_dims == 16
    assert dream.env.num_explicit_dims == 3
    assert dream.env.num_decoder_output == 45
    assert dream.env.num_single_critic_obs == 3 + 45 + 5
    assert dream.env.num_privileged_obs == 5 * 53

    him = Go2V6FrontierHIMFixedCfg()
    assert him.env.num_one_step_obs == 45
    assert him.env.frame_stack == 6
    assert him.env.num_observations == 6 * 45
    assert him.env.c_frame_stack == 5
    assert him.env.num_single_critic_obs == 3 + 45 + 5
    assert him.env.num_privileged_obs == 5 * 53


@pytest.mark.parametrize("task_name", V6_ARCHITECTURE_ARMS)
def test_build_frontier_teacher_succeeds_and_shares_task_space_fingerprint(task_name):
    cfg = V6_ARMS[task_name]["cfg_cls"]()
    curriculum, task_space = build_frontier_teacher(cfg)
    assert task_space.fingerprint() == MLP_TASK_SPACE_FINGERPRINT
    assert curriculum.config_fingerprint == MLP_CONFIG_FINGERPRINT
    # seed=None must fall back cleanly (getattr env_cfg.seed / 0).
    assert cfg.curriculum.seed is None


def test_mlp_fingerprints_unchanged():
    cfg = Go2V6FrontierCfg()
    curriculum, task_space = build_frontier_teacher(cfg)
    assert task_space.fingerprint() == MLP_TASK_SPACE_FINGERPRINT
    assert curriculum.config_fingerprint == MLP_CONFIG_FINGERPRINT


def test_cross_arm_checkpoint_load():
    """A teacher state saved from mlp must load into any other architecture arm."""
    mlp_cfg = Go2V6FrontierCfg()
    mlp_teacher, _ = build_frontier_teacher(mlp_cfg)
    mlp_teacher.sample(64, global_control_steps=0)
    state = mlp_teacher.state_dict()

    # Reference draw after restoring the same mlp state.
    mlp_ref, _ = build_frontier_teacher(mlp_cfg)
    mlp_ref.load_state_dict(state)
    expected = mlp_ref.sample(17, global_control_steps=0)

    for task_name in (
        "go2_v6_frontier_oracle",
        "go2_v6_frontier_dreamwaq",
        "go2_v6_frontier_him",
    ):
        other_cfg = V6_ARMS[task_name]["cfg_cls"]()
        other_teacher, _ = build_frontier_teacher(other_cfg)
        other_teacher.load_state_dict(state)
        actual = other_teacher.sample(17, global_control_steps=0)
        np.testing.assert_array_equal(actual.task_ids, expected.task_ids)
        np.testing.assert_array_equal(actual.sources, expected.sources)


@pytest.mark.parametrize("n", [1, 4, 7, 24, 4096])
def test_sample_remainder_path(n):
    cfg = Go2V6FrontierCfg()
    curriculum, _ = build_frontier_teacher(cfg)
    batch = curriculum.sample(n, global_control_steps=0)
    assert len(batch.task_ids) == n
    assert len(batch.sources) == n
    assert batch.task_ids.dtype == np.int64


def test_runner_names_per_arm():
    _, mlp_train = task_registry.get_cfgs(name="go2_v6_frontier")
    _, oracle_train = task_registry.get_cfgs(name="go2_v6_frontier_oracle")
    _, dream_train = task_registry.get_cfgs(name="go2_v6_frontier_dreamwaq")
    _, him_train = task_registry.get_cfgs(name="go2_v6_frontier_him")
    assert mlp_train.runner.run_name.startswith("v6_v4_frontier")
    assert "oracle" in oracle_train.runner.run_name
    assert "dreamwaq" in dream_train.runner.run_name
    assert "him" in him_train.runner.run_name
    for train in (mlp_train, oracle_train, dream_train, him_train):
        assert train.runner.experiment_name == "go2_v6_frontier"
        assert train.runner.command_schedule is None


def test_online_eval_disabled_on_every_arm():
    """Online eval drives the training env, so it must stay off under UED.

    The V3 method runners (oracle / dreamwaq / him) default to
    ``eval_interval=200``.  Left enabled, those eval resets reach
    ``_observe_ued_outcomes`` and land in the frontier success rings, so a
    cell could be promoted on episodes the curriculum never sampled -- and
    only three of the four arms would be contaminated, which breaks the
    comparison outright.  The MLP arm inherits 0 from ``Go2V5CfgPPO``.
    """
    for task in (
        "go2_v6_frontier",
        "go2_v6_frontier_mlp",
        "go2_v6_frontier_oracle",
        "go2_v6_frontier_dreamwaq",
        "go2_v6_frontier_him",
    ):
        _, train = task_registry.get_cfgs(name=task)
        assert getattr(train.runner, "eval_interval", 0) == 0, (
            f"{task} would run online eval against the UED training env"
        )
