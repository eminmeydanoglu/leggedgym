"""V7: common V6-HIM substrate with a return-LP semantic terrain teacher.

Only the teacher differs between ``go2_v6_frontier_him`` and the two moving
V7 tasks.  They retain the V6 native V4 terrain bank, velocity support,
HIM histories, locomotion reward/DR and PPO contract exactly.
"""
from __future__ import annotations

from legged_gym.envs.base.common_cfgs import get_simulator_suffix
from legged_gym.envs.go2.go2_v6_frontier_config import (
    Go2V6FrontierHIMFixedCfg,
    Go2V6FrontierHIMFixedCfgPPO,
)


class _Go2V7HIMCfg(Go2V6FrontierHIMFixedCfg):
    """V6-HIM physical/observation contract; V7 owns only teacher identity."""

    class env(Go2V6FrontierHIMFixedCfg.env):
        ued_enabled = True

    class curriculum(Go2V6FrontierHIMFixedCfg.curriculum):
        algorithm = "lp_acrl"
        # Keep V5's proven safeguards while changing only task-space identity.
        lp_estimator = "rolling_completion"


class Go2V7LPACRLHIMCfg(_Go2V7HIMCfg):
    """Default V7 return/LP arm over 240 semantic terrain-speed cells."""

    class curriculum(_Go2V7HIMCfg.curriculum):
        algorithm = "lp_acrl"


class Go2V7UniformHIMCfg(_Go2V7HIMCfg):
    """Matched V7 control whose moving semantic cells remain uniform."""

    class curriculum(_Go2V7HIMCfg.curriculum):
        algorithm = "uniform"


class Go2V7FlatLPACRLHIMCfg(Go2V7LPACRLHIMCfg):
    """1000-iteration flat LP source prior; no terrain grid or teleporting."""

    class curriculum(Go2V7LPACRLHIMCfg.curriculum):
        # The inherited 0.25 cap was tuned for the 240-cell semantic space,
        # where it leaves ample room above uniform (1/240).  On the flat prior
        # there are only four |vx| cells, so a 0.25 per-cell cap equals uniform
        # (4 x 0.25 = 1.0): every LP-weighted distribution would be clamped flat
        # and the source prior would degenerate into go2_v7_uniform.  Lift the
        # cap so learning progress can actually concentrate on a velocity bin.
        max_cell_probability = 1.0

    class terrain(Go2V7LPACRLHIMCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        selected = False
        ued_training_grid = False


class _Go2V7HIMCfgPPO(Go2V6FrontierHIMFixedCfgPPO):
    class runner(Go2V6FrontierHIMFixedCfgPPO.runner):
        experiment_name = "go2_v7_lpacrl_him"
        command_schedule = None
        # No training-env eval: it would add outcomes to a live teacher.
        eval_interval = 0


class Go2V7LPACRLHIMCfgPPO(_Go2V7HIMCfgPPO):
    class runner(_Go2V7HIMCfgPPO.runner):
        run_name = "v7_lp_acrl_him" + get_simulator_suffix()


class Go2V7UniformHIMCfgPPO(_Go2V7HIMCfgPPO):
    class runner(_Go2V7HIMCfgPPO.runner):
        run_name = "v7_uniform_him" + get_simulator_suffix()


class Go2V7FlatLPACRLHIMCfgPPO(_Go2V7HIMCfgPPO):
    class runner(_Go2V7HIMCfgPPO.runner):
        experiment_name = "go2_v7_flat_lpacrl_him"
        run_name = "v7_flat_lp_acrl_him" + get_simulator_suffix()
        max_iterations = 1000
        # Intermediate saves every 200 iterations.  The Jul29 seed-1 run kept
        # only model_1000.pt, so when its LR floored at 1e-5 around iteration
        # 150 there was no earlier checkpoint left to inspect or roll back to.
        # There is still no best_* selection because eval is disabled above.
        save_interval = 200


def v6_terrain_builder_parameters(env_cfg) -> dict[str, object]:
    """The builder identity deliberately mirrors ``build_frontier_teacher``."""
    terrain = env_cfg.terrain
    return {
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


def build_v7_teacher(env_cfg):
    """Build a V7 teacher without modifying V5 or V6 task/teacher state."""
    from legged_gym.utils.v7 import (
        V7LPACRLCurriculum,
        V7SemanticTaskSpace,
        V7UniformCurriculum,
        V7VelocityTaskSpace,
    )

    algorithm = str(env_cfg.curriculum.algorithm)
    curriculum_class = {
        "lp_acrl": V7LPACRLCurriculum,
        "uniform": V7UniformCurriculum,
    }.get(algorithm)
    if curriculum_class is None:
        raise ValueError("V7 expects curriculum.algorithm 'lp_acrl' or 'uniform'")
    flat = getattr(env_cfg.terrain, "mesh_type", None) == "plane"
    if flat:
        task_space = V7VelocityTaskSpace(
            builder_parameters={"builder": "plane", "terrain_teleport": False}
        )
    else:
        task_space = V7SemanticTaskSpace(
            builder_parameters=v6_terrain_builder_parameters(env_cfg)
        )
    seed = env_cfg.curriculum.seed
    if seed is None:
        seed = int(getattr(env_cfg, "seed", 0) or 0)
    curriculum = curriculum_class(
        task_space,
        stage_length_control_steps=int(env_cfg.curriculum.stage_length_control_steps),
        beta=float(env_cfg.curriculum.beta),
        epsilon=float(env_cfg.curriculum.epsilon),
        temperature_mode=str(env_cfg.curriculum.temperature_mode),
        beta_min=float(env_cfg.curriculum.beta_min),
        beta_max=float(env_cfg.curriculum.beta_max),
        beta_ema=float(env_cfg.curriculum.beta_ema),
        target_ess_ratio_min=float(env_cfg.curriculum.target_ess_ratio_min),
        max_cell_probability=float(env_cfg.curriculum.max_cell_probability),
        min_stage_episodes_for_lp=int(env_cfg.curriculum.min_stage_episodes_for_lp),
        confidence_scale=float(env_cfg.curriculum.confidence_scale),
        lp_estimator=str(env_cfg.curriculum.lp_estimator),
        rolling_completion_window=int(env_cfg.curriculum.rolling_completion_window),
        seed=seed,
        flat=flat,
    )
    return curriculum, task_space
