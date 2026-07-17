"""Corrected HIM configuration on the shared V3 dynamic-physics contract."""

from legged_gym.envs.base.common_cfgs import Go2BenchmarkV3CommonCfg, get_simulator_suffix
from legged_gym.envs.go2.go2_bench_him.go2_bench_him_config import Go2BenchHIMCfgPPO


class Go2V3HIMFixedCfg(Go2BenchmarkV3CommonCfg):
    class env(Go2BenchmarkV3CommonCfg.env):
        num_one_step_obs = 45
        frame_stack = 6
        num_observations = frame_stack * num_one_step_obs
        c_frame_stack = 5
        # Fixed HIM keeps its 6x45 actor history and 5x53 critic history.
        num_single_critic_obs = 3 + num_one_step_obs + 5
        num_privileged_obs = c_frame_stack * num_single_critic_obs
        num_actions = 12


class Go2V3HIMFixedCfgPPO(Go2BenchHIMCfgPPO):
    """Retain the corrected HIM estimator and PPO coupling from HIM-fixed."""

    class runner(Go2BenchHIMCfgPPO.runner):
        experiment_name = "go2_v3_dynamic"
        run_name = "v3_him_fixed" + get_simulator_suffix()
        command_schedule = [
            {"start_iteration": 0, "lin_vel_x": [-0.5, 0.5]},
            {"start_iteration": 500, "lin_vel_x": [-1.0, 1.0]},
            {"start_iteration": 1500, "lin_vel_x": [-1.0, 2.0]},
        ]
