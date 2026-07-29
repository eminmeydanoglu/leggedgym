"""V4-native, type-balanced speed x terrain-level frontier curriculum.

This is intentionally a new experiment identity.  It reuses the V4 terrain
builder, locomotion substrate, domain randomization and the proven V5
assignment/reset boundary, but it does not use learning progress.

Four architecture arms share the same frontier curriculum substrate
(``Go2V6FrontierCfg`` terrain / commands / curriculum) and only differ in
observation contract + PPO policy class, matching the V4 pattern in
``go2_v4_config.py``.
"""
from __future__ import annotations

from legged_gym.envs.base.common_cfgs import get_simulator_suffix
from legged_gym.envs.go2.go2_v5_config import Go2V5CommonCfg, Go2V5CfgPPO
from legged_gym.envs.go2.go2_v3_dreamwaq.go2_v3_dreamwaq_config import Go2V3DreamwaqCfgPPO
from legged_gym.envs.go2.go2_v3_him_fixed.go2_v3_him_fixed_config import Go2V3HIMFixedCfgPPO
from legged_gym.envs.go2.go2_v3_superset_oracle.go2_v3_superset_oracle_config import (
    Go2V3SupersetOracleCfgPPO,
)


class Go2V6FrontierCfg(Go2V5CommonCfg):
    """MLP arm (default ``go2_v6_frontier``): 45-D noisy proprio / 48-D critic."""

    class env(Go2V5CommonCfg.env):
        ued_enabled = True

    class terrain(Go2V5CommonCfg.terrain):
        # Use V4's native 10x10 traversable terrain bank and interpret the
        # assigned cell as the episode's starting terrain, exactly as the
        # game-inspired curriculum does.  Robots may cross tile boundaries;
        # occupancy is logged for audit but never causes reward or termination.
        curriculum = True
        selected = False
        ued_training_grid = False
        num_rows = 10
        num_cols = 10
        max_init_terrain_level = 1
        terrain_length = 8.0
        terrain_width = 8.0
        platform_size = 4.0
        terrain_proportions = [0.2, 0.1, 0.25, 0.25, 0.2]
        # Physical columns within one semantic family sample nearby severities;
        # rough/discrete columns also retain independent stochastic layouts.
        terrain_replica_variation = 0.10
        # Unlike legacy V4's fixed ±5 cm rough tile, make rough severity follow
        # the same ten-level principle: L0=±1 cm through L9=±10 cm.
        terrain_curriculum_difficulty = {
            **Go2V5CommonCfg.terrain.terrain_curriculum_difficulty,
            "rough_height": "0.01 + 0.10 * difficulty",
        }

    class commands(Go2V5CommonCfg.commands):
        # The frontier adapter samples a magnitude in [0.2, 2.0] and chooses a
        # persistent episode sign with equal probability.  These outer bounds
        # remain truthful for validation, SPNTE normalization and manifests.
        legacy_performance_command_curriculum_enabled = False
        per_env_standstill = True

        class ranges(Go2V5CommonCfg.commands.ranges):
            lin_vel_x = [-2.0, 2.0]

    class curriculum(Go2V5CommonCfg.curriculum):
        algorithm = "frontier"
        update_interval_control_steps = 2_000
        window_size = 32
        min_episodes = 24
        mastery_threshold = 0.80
        unstable_threshold = 0.55
        mastery_updates = 2
        frontier_fraction = 0.60
        replay_fraction = 0.30
        uniform_fraction = 0.10
        linear_error_threshold = 0.35
        yaw_error_threshold = 0.40


# ---------------------------------------------------------------------------
# Architecture arms: derive from Go2V6FrontierCfg and re-declare class env
# with the V4 method observation contract + ued_enabled=True.  Explicit
# re-declaration avoids diamond inheritance silently picking the wrong
# terrain / commands / curriculum from a V4 method cfg.
# ---------------------------------------------------------------------------

class Go2V6FrontierSupersetOracleCfg(Go2V6FrontierCfg):
    """Oracle arm: 20-frame proprio history + true vel/P5 + 17x11 height map."""

    class env(Go2V6FrontierCfg.env):
        ued_enabled = True
        num_single_obs = 45
        frame_stack = 20
        height_map_dim = 17 * 11
        oracle_height_map = True
        # [20 x noisy proprio, true velocity, true P5, true height map]
        num_observations = num_single_obs * frame_stack + 3 + 5 + height_map_dim
        num_privileged_obs = num_observations
        num_actions = 12


class Go2V6FrontierDreamwaqCfg(Go2V6FrontierCfg):
    """DreamWaQ arm: 45-D step obs, 5-frame history, VAE critic stack."""

    class env(Go2V6FrontierCfg.env):
        ued_enabled = True
        num_observations = 45
        frame_stack = 5
        num_history_obs = num_observations * frame_stack
        num_latent_dims = 16
        num_explicit_dims = 3
        num_decoder_output = num_observations
        c_frame_stack = 5
        num_single_critic_obs = 3 + num_observations + 5
        num_privileged_obs = c_frame_stack * num_single_critic_obs
        num_actions = 12


class Go2V6FrontierHIMFixedCfg(Go2V6FrontierCfg):
    """HIM-fixed arm: 6x45 actor history, 5x53 critic history."""

    class env(Go2V6FrontierCfg.env):
        ued_enabled = True
        num_one_step_obs = 45
        frame_stack = 6
        num_observations = frame_stack * num_one_step_obs
        c_frame_stack = 5
        num_single_critic_obs = 3 + num_one_step_obs + 5
        num_privileged_obs = c_frame_stack * num_single_critic_obs
        num_actions = 12


# ---------------------------------------------------------------------------
# PPO / runner cfgs
# ---------------------------------------------------------------------------

class Go2V6FrontierCfgPPO(Go2V5CfgPPO):
    class runner(Go2V5CfgPPO.runner):
        experiment_name = "go2_v6_frontier"
        run_name = "v6_v4_frontier" + get_simulator_suffix()
        command_schedule = None


class Go2V6FrontierSupersetOracleCfgPPO(Go2V3SupersetOracleCfgPPO):
    class runner(Go2V3SupersetOracleCfgPPO.runner):
        experiment_name = "go2_v6_frontier"
        run_name = "v6_v4_frontier_oracle" + get_simulator_suffix()
        command_schedule = None
        # The V3 method runners inherit online in-distribution eval
        # (eval_interval=200), which drives the *training* env.  Under a UED
        # teacher those extra resets flow through _observe_ued_outcomes into
        # the frontier success rings, so mastery would be decided partly on
        # episodes the curriculum never sampled.  V5 disabled it for exactly
        # this reason (go2_v5_config.py); every v6 arm must match the MLP arm,
        # which already gets eval_interval=0 from Go2V5CfgPPO -- otherwise the
        # four arms are not comparable.
        eval_interval = 0


class Go2V6FrontierDreamwaqCfgPPO(Go2V3DreamwaqCfgPPO):
    class runner(Go2V3DreamwaqCfgPPO.runner):
        experiment_name = "go2_v6_frontier"
        run_name = "v6_v4_frontier_dreamwaq" + get_simulator_suffix()
        command_schedule = None
        # See Go2V6FrontierSupersetOracleCfgPPO: online eval must stay off so
        # it cannot inject episodes into the frontier success rings.
        eval_interval = 0


class Go2V6FrontierHIMFixedCfgPPO(Go2V3HIMFixedCfgPPO):
    class runner(Go2V3HIMFixedCfgPPO.runner):
        experiment_name = "go2_v6_frontier"
        run_name = "v6_v4_frontier_him" + get_simulator_suffix()
        command_schedule = None
        # See Go2V6FrontierSupersetOracleCfgPPO: online eval must stay off so
        # it cannot inject episodes into the frontier success rings.
        eval_interval = 0


def build_frontier_teacher(env_cfg):
    """Construct the checkpointable frontier teacher and V4 task space."""
    from legged_gym.utils.frontier import FrontierCurriculum, V4FrontierTaskSpace

    terrain = env_cfg.terrain
    builder_parameters = {
        "builder": "v4_generators_frontier_starting_terrain_10x10_v2",
        "num_rows": int(terrain.num_rows),
        "num_cols": int(terrain.num_cols),
        "terrain_proportions": tuple(float(x) for x in terrain.terrain_proportions),
        "terrain_length": float(terrain.terrain_length),
        "terrain_width": float(terrain.terrain_width),
        "horizontal_scale": float(terrain.horizontal_scale),
        "vertical_scale": float(terrain.vertical_scale),
        "terrain_curriculum_difficulty": terrain.terrain_curriculum_difficulty,
        "terrain_replica_variation": float(terrain.terrain_replica_variation),
    }
    task_space = V4FrontierTaskSpace(builder_parameters=builder_parameters)
    seed = env_cfg.curriculum.seed
    if seed is None:
        seed = int(getattr(env_cfg, "seed", 0) or 0)
    curriculum = FrontierCurriculum(
        task_space,
        update_interval_control_steps=int(
            env_cfg.curriculum.update_interval_control_steps
        ),
        window_size=int(env_cfg.curriculum.window_size),
        min_episodes=int(env_cfg.curriculum.min_episodes),
        mastery_threshold=float(env_cfg.curriculum.mastery_threshold),
        unstable_threshold=float(env_cfg.curriculum.unstable_threshold),
        mastery_updates=int(env_cfg.curriculum.mastery_updates),
        frontier_fraction=float(env_cfg.curriculum.frontier_fraction),
        replay_fraction=float(env_cfg.curriculum.replay_fraction),
        uniform_fraction=float(env_cfg.curriculum.uniform_fraction),
        linear_error_threshold=float(env_cfg.curriculum.linear_error_threshold),
        yaw_error_threshold=float(env_cfg.curriculum.yaw_error_threshold),
        seed=seed,
    )
    return curriculum, task_space
