"""V4-native, type-balanced speed x terrain-level frontier curriculum.

This is intentionally a new experiment identity.  It reuses the V4 terrain
builder, locomotion substrate, domain randomization and the proven V5
assignment/reset boundary, but it does not use learning progress.
"""
from __future__ import annotations

from legged_gym.envs.base.common_cfgs import get_simulator_suffix
from legged_gym.envs.go2.go2_v5_config import Go2V5CommonCfg, Go2V5CfgPPO


class Go2V6FrontierCfg(Go2V5CommonCfg):
    class env(Go2V5CommonCfg.env):
        ued_enabled = True

    class terrain(Go2V5CommonCfg.terrain):
        # Use V4 Terrain.curiculum() and the same generators/severity formulas.
        # Twelve columns give every semantic family two physical replicas:
        # slope up/down, rough, stairs up/down and discrete obstacles.
        # The episode curriculum owns level changes, so the legacy per-env
        # distance promotion is bypassed whenever the adapter is active.
        curriculum = True
        selected = False
        ued_training_grid = False
        num_rows = 10
        num_cols = 12
        max_init_terrain_level = 1
        terrain_length = 8.0
        terrain_width = 8.0
        platform_size = 4.0
        terrain_proportions = [1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6]
        # Each of the two columns per family samples a nearby severity inside
        # its level band.  Rough/discrete also receive independent layouts from
        # their existing stochastic V4 generators.
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


class Go2V6FrontierCfgPPO(Go2V5CfgPPO):
    class runner(Go2V5CfgPPO.runner):
        experiment_name = "go2_v6_frontier"
        run_name = "v6_v4_frontier" + get_simulator_suffix()
        command_schedule = None


def build_frontier_teacher(env_cfg):
    """Construct the checkpointable frontier teacher and V4 task space."""
    from legged_gym.utils.frontier import FrontierCurriculum, V4FrontierTaskSpace

    terrain = env_cfg.terrain
    builder_parameters = {
        "builder": "v4_generators_frontier_10x12_v1",
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
