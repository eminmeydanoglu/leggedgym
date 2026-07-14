from legged_gym import *
from legged_gym.envs.base.template_cfgs import LeggedRobotEECfgPPO
from legged_gym.envs.base.common_cfgs import Go2BenchmarkCommonCfg, get_simulator_suffix


class Go2BenchSysIDCfg(Go2BenchmarkCommonCfg):
    """Explicit SysID on the frozen benchmark substrate. Inherits the frozen flat
    terrain, frozen band DR, reward scales and commands from
    Go2BenchmarkCommonCfg; adds the explicit-estimator obs dims."""
    class env(Go2BenchmarkCommonCfg.env):
        num_single_obs = 45
        frame_stack = 20                                    # estimator feature history
        num_estimator_features = int(num_single_obs * frame_stack)  # 900
        num_estimator_labels = 8                            # [base_lin_vel(3), P(5)]
        c_frame_stack = 5
        num_single_critic_obs = num_single_obs + num_estimator_labels  # [obs, labels] = 53
        num_privileged_obs = c_frame_stack * num_single_critic_obs     # 265 (= critic obs)
        num_actions = 12


class Go2BenchSysIDCfgPPO(LeggedRobotEECfgPPO):
    """Runner=EERunner, policy=ActorCriticEE, algo=PPO_EE. Benchmark budget."""
    class policy(LeggedRobotEECfgPPO.policy):
        critic_hidden_dims = [1024, 256, 128]
        estimator_hidden_dims = [256, 128]
        # The estimator alone consumes the 900D history. The actor receives the
        # current proprioceptive frame plus the estimated [V, P5] labels.
        num_actor_obs = 45

    class algorithm(LeggedRobotEECfgPPO.algorithm):
        estimator_lr = 2.e-4
        num_estimator_epochs = 1

    class runner(LeggedRobotEECfgPPO.runner):
        experiment_name = 'go2_benchmark'
        run_name = 'bench_sysid' + get_simulator_suffix()
        save_interval = 200
        max_iterations = 3000
        command_schedule = [
            {"start_iteration": 0, "lin_vel_x": [-0.5, 0.5]},
            {"start_iteration": 500, "lin_vel_x": [-1.0, 1.0]},
        ]
        eval_interval = 200
        eval_steps = 1100
        eval_warmup = 50
        eval_seed = 12345
        eval_fall_guard = 0.05
        # Native five-frame critic is intentionally retained. Results are a
        # method-package comparison, not a controlled single-critic ablation.
        critic_contract = "stacked_5x53_265d"
