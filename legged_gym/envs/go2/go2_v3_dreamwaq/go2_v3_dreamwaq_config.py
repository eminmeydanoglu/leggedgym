"""DreamWaQ configuration on the shared V3 dynamic-physics contract."""

from legged_gym.envs.base.common_cfgs import Go2BenchmarkV3CommonCfg, get_simulator_suffix
from legged_gym.envs.go2.go2_bench_dreamwaq.go2_bench_dreamwaq_config import Go2BenchDreamwaqCfgPPO


class Go2V3DreamwaqCfg(Go2BenchmarkV3CommonCfg):
    class env(Go2BenchmarkV3CommonCfg.env):
        num_observations = 45
        frame_stack = 5
        num_history_obs = int(num_observations * frame_stack)
        num_latent_dims = 16
        num_explicit_dims = 3
        num_decoder_output = num_observations
        c_frame_stack = 5
        # Five clean [base_lin_vel, proprio, P5] frames for the critic.
        num_single_critic_obs = 3 + num_observations + 5
        num_privileged_obs = c_frame_stack * num_single_critic_obs
        num_actions = 12


class Go2V3DreamwaqCfgPPO(Go2BenchDreamwaqCfgPPO):
    """Keep DreamWaQ's VAE settings; replace only shared V3 runner settings."""

    class runner(Go2BenchDreamwaqCfgPPO.runner):
        experiment_name = "go2_v3_dynamic"
        run_name = "v3_dreamwaq" + get_simulator_suffix()
        command_schedule = [
            {"start_iteration": 0, "lin_vel_x": [-0.5, 0.5]},
            {"start_iteration": 500, "lin_vel_x": [-1.0, 1.0]},
            {"start_iteration": 1500, "lin_vel_x": [-1.0, 2.0]},
        ]
