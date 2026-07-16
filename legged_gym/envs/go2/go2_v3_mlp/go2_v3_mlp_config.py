from legged_gym.envs.base.common_cfgs import Go2BenchmarkV3CommonCfg, get_simulator_suffix
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO


class Go2V3MlpCfg(Go2BenchmarkV3CommonCfg):
    class env(Go2BenchmarkV3CommonCfg.env):
        num_observations = 45
        num_privileged_obs = 48  # clean proprio + true base velocity for critic
        num_actions = 12


class Go2V3CfgPPO(LeggedRobotCfgPPO):
    """Shared V3 PPO and checkpoint-selection protocol."""
    class runner(LeggedRobotCfgPPO.runner):
        experiment_name = "go2_v3_dynamic"
        run_name = "v3_mlp" + get_simulator_suffix()
        save_interval = 200
        max_iterations = 3000
        command_schedule = [
            {"start_iteration": 0, "lin_vel_x": [-0.5, 0.5]},
            {"start_iteration": 500, "lin_vel_x": [-1.0, 1.0]},
            {"start_iteration": 1500, "lin_vel_x": [-1.0, 2.0]},
        ]
        # Training-internal, fixed-seed validation selects a safe checkpoint.
        # It is not the external V3 scorecard/evaluation campaign.
        eval_interval = 200
        eval_steps = 1100
        eval_warmup = 50
        eval_seed = 12345
        eval_fall_guard = 0.05


class Go2V3MlpCfgPPO(Go2V3CfgPPO):
    pass
