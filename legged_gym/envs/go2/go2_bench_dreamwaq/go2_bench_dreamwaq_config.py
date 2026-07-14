from legged_gym import *
from legged_gym.envs.base.template_cfgs import LeggedRobotDreamwaqCfgPPO
from legged_gym.envs.base.common_cfgs import Go2BenchmarkCommonCfg, get_simulator_suffix


class Go2BenchDreamwaqCfg(Go2BenchmarkCommonCfg):
    """DreamWaQ on the frozen benchmark substrate. Inherits the frozen flat
    terrain, frozen band DR, reward scales and commands from
    Go2BenchmarkCommonCfg; adds the DreamWaQ VAE obs dims."""
    class env(Go2BenchmarkCommonCfg.env):
        num_envs = 4096                                     # fairness: match the family
        num_observations = 45
        frame_stack = 5                                     # VAE encoder input frames
        num_history_obs = int(num_observations * frame_stack)   # 225
        num_latent_dims = 16                                # VAE latent
        num_explicit_dims = 3                               # base_lin_vel
        num_decoder_output = num_observations               # 45
        c_frame_stack = 5
        num_single_critic_obs = 3 + num_observations + 5    # [base_lin_vel, obs, P] = 53
        num_privileged_obs = c_frame_stack * num_single_critic_obs  # 265
        num_actions = 12


class Go2BenchDreamwaqCfgPPO(LeggedRobotDreamwaqCfgPPO):
    """Runner=DreamWaQRunner, policy=ActorCriticDreamWaQ, algo=PPO_DreamWaQ.
    Benchmark training budget."""
    class policy(LeggedRobotDreamwaqCfgPPO.policy):
        critic_hidden_dims = [1024, 256, 128]
        encoder_hidden_dims = [256, 128]
        decoder_hidden_dims = [256, 128]

    class algorithm(LeggedRobotDreamwaqCfgPPO.algorithm):
        encoder_lr = 2.e-4
        num_encoder_epochs = 1
        vae_kld_weight = 2.0

    class runner(LeggedRobotDreamwaqCfgPPO.runner):
        experiment_name = 'go2_benchmark'
        run_name = 'bench_dreamwaq' + get_simulator_suffix()
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
