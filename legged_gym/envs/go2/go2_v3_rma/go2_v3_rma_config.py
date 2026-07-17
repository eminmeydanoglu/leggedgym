"""RMA configuration on the shared V3 dynamic-physics contract."""

from legged_gym.envs.base.common_cfgs import Go2BenchmarkV3CommonCfg, get_simulator_suffix
from legged_gym.envs.go2.go2_bench_rma.go2_bench_rma_config import Go2BenchRMACfgPPO


class Go2V3RMACfg(Go2BenchmarkV3CommonCfg):
    class env(Go2BenchmarkV3CommonCfg.env):
        num_observations = 45
        frame_stack = 20
        num_history_obs = int(num_observations * frame_stack)
        # Teacher label / privilege encoder input: [P5, base_lin_vel].
        num_privileged_obs = 8
        num_latent_dims = num_privileged_obs
        # Asymmetric critic: five clean [proprio, P5, velocity] frames.
        c_frame_stack = 5
        num_single_critic_obs = num_observations + 5 + 3
        num_critic_obs = c_frame_stack * num_single_critic_obs
        num_actions = 12


class Go2V3RMACfgPPO(Go2BenchRMACfgPPO):
    """Keep RMA's teacher/student architecture; replace only V3 protocol knobs."""

    class runner(Go2BenchRMACfgPPO.runner):
        experiment_name = "go2_v3_dynamic"
        run_name = "v3_rma" + get_simulator_suffix()
        command_schedule = [
            {"start_iteration": 0, "lin_vel_x": [-0.5, 0.5]},
            {"start_iteration": 500, "lin_vel_x": [-1.0, 1.0]},
            {"start_iteration": 1500, "lin_vel_x": [-1.0, 2.0]},
        ]
