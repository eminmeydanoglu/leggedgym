from legged_gym.envs.base.common_cfgs import Go2BenchmarkV3CommonCfg, get_simulator_suffix
from legged_gym.envs.base.template_cfgs import LeggedRobotEECfgPPO


class Go2V3SysIDCfg(Go2BenchmarkV3CommonCfg):
    class env(Go2BenchmarkV3CommonCfg.env):
        num_single_obs = 45
        frame_stack = 20
        num_estimator_features = int(num_single_obs * frame_stack)
        num_estimator_labels = 8  # true base velocity (3) + P5
        c_frame_stack = 5
        num_single_critic_obs = num_single_obs + num_estimator_labels
        num_privileged_obs = c_frame_stack * num_single_critic_obs
        num_actions = 12


class Go2V3SysIDCfgPPO(LeggedRobotEECfgPPO):
    class policy(LeggedRobotEECfgPPO.policy):
        critic_hidden_dims = [1024, 256, 128]
        estimator_hidden_dims = [256, 128]
        num_actor_obs = 45

    class algorithm(LeggedRobotEECfgPPO.algorithm):
        estimator_lr = 2.e-4
        num_estimator_epochs = 1

    class runner(LeggedRobotEECfgPPO.runner):
        experiment_name = "go2_v3_dynamic"
        run_name = "v3_sysid" + get_simulator_suffix()
        save_interval = 200
        max_iterations = 3000
        command_schedule = [
            {"start_iteration": 0, "lin_vel_x": [-0.5, 0.5]},
            {"start_iteration": 500, "lin_vel_x": [-1.0, 1.0]},
            {"start_iteration": 1500, "lin_vel_x": [-1.0, 2.0]},
        ]
        eval_interval = 200
        eval_steps = 1100
        eval_warmup = 50
        eval_seed = 12345
        eval_fall_guard = 0.05
        critic_contract = "stacked_5x53_265d"
