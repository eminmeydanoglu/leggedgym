from legged_gym import *
from legged_gym.envs.base.template_cfgs import LeggedRobotTSCfgPPO
from legged_gym.envs.base.common_cfgs import Go2BenchmarkCommonCfg, get_simulator_suffix


class Go2BenchRMACfg(Go2BenchmarkCommonCfg):
    """RMA on the frozen benchmark substrate. Inherits the frozen flat terrain,
    frozen band DR (friction [0.5,1.25] / mass [-1,1] / com +-0.03, pd_gain &
    ctrl_delay OFF), reward scales and commands from Go2BenchmarkCommonCfg. Only
    the teacher-student obs dims are added here."""
    class env(Go2BenchmarkCommonCfg.env):
        num_observations = 45
        frame_stack = 20                                    # obs_history stacking
        num_history_obs = int(num_observations * frame_stack)   # 900
        num_privileged_obs = 8                              # [P(5), base_lin_vel(3)]
        num_latent_dims = num_privileged_obs                # z dim = 8
        c_frame_stack = 5
        num_single_critic_obs = num_observations + 5 + 3    # [obs, P, base_lin_vel] = 53
        num_critic_obs = c_frame_stack * num_single_critic_obs  # 265
        num_actions = 12


class Go2BenchRMACfgPPO(LeggedRobotTSCfgPPO):
    """Runner=TSRunner, policy=ActorCriticTS, algo=PPO_TS (from template).
    Benchmark training budget: same experiment_name/iterations as the rest of
    the go2_benchmark family."""
    class policy(LeggedRobotTSCfgPPO.policy):
        critic_hidden_dims = [1024, 256, 128]
        privilege_encoder_hidden_dims = [256, 128]
        history_encoder_type = "MLP"
        history_encoder_hidden_dims = [256, 128]
        history_encoder_channel_dims = [1, 1, 1, 1]
        history_encoder_dilation = [1, 1, 2, 1]
        history_encoder_stride = [1, 2, 1, 2]
        history_encoder_final_layer_dim = 128
        kernel_size = 5

    class algorithm(LeggedRobotTSCfgPPO.algorithm):
        encoder_lr = 2.e-4
        num_encoder_epochs = 2

    class runner(LeggedRobotTSCfgPPO.runner):
        experiment_name = 'go2_benchmark'
        run_name = 'bench_rma' + get_simulator_suffix()
        save_interval = 200
        max_iterations = 3000
        # --- iteration-based command schedule (shared by all benchmark methods) ---
        command_schedule = [
            {"start_iteration": 0,   "lin_vel_x": [-0.5, 0.5]},
            {"start_iteration": 500, "lin_vel_x": [-1.0, 1.0]},
        ]
        # --- in-distribution eval + best_tracking.pt (shared by all methods) ---
        # Eval runs the DEPLOYED student (history encoder), so the checkpoint is
        # picked on what ships, not on the privileged teacher (RMA blocker).
        eval_interval = 200      # iters between evals (0 disables); aligns with save_interval
        eval_steps = 1100        # >= max_episode_length+1 (1001) so returns are complete
        eval_warmup = 50         # unrecorded settling steps
        eval_seed = 12345        # fixed -> comparable across checkpoints within a run
        eval_fall_guard = 0.05   # V2 safe fixed-window fall-rate threshold
        # 1.8 critic-contract disclosure: RMA trains on a 5x53=265D stacked
        # privileged critic (its native design), NOT the 53D single-frame critic
        # used by MLP/P5/HIM. Recorded so the family is reported as a
        # method-package comparison, not a controlled single-critic ablation.
        critic_contract = "stacked_5x53_265d"
