from legged_gym import *
from legged_gym.envs.base.template_cfgs import LeggedRobotHIMCfgPPO
from legged_gym.envs.base.common_cfgs import Go2BenchmarkCommonCfg, get_simulator_suffix


class Go2BenchHIMCfg(Go2BenchmarkCommonCfg):
    """HIM on the frozen benchmark substrate. Inherits the frozen flat terrain,
    frozen band DR, reward scales and commands from Go2BenchmarkCommonCfg; adds
    the HIM history / single-frame-critic obs dims."""
    class env(Go2BenchmarkCommonCfg.env):
        num_envs = 4096                                     # fairness: match the family
        num_one_step_obs = 45                               # single proprio frame
        frame_stack = 6                                     # temporal_steps for the estimator
        num_observations = frame_stack * num_one_step_obs   # 270 (actor obs = history)
        num_privileged_obs = 3 + num_one_step_obs + 5       # [base_lin_vel, one_step, P] = 53
        num_actions = 12


class Go2BenchHIMCfgPPO(LeggedRobotHIMCfgPPO):
    """Runner=HIMRunner, policy=HIMActorCritic, algo=PPO_HIM. Benchmark budget.

    Because HIM keeps the standard contract, its runner reuses the base
    OnPolicyRunner fairness machinery -- so the command_schedule + eval_* block
    is copied verbatim from Go2BenchCfgPPO.runner (go2_bench_mlp_config.py)."""
    class policy(LeggedRobotHIMCfgPPO.policy):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        enc_hidden_dims = [128, 64, 16]   # latent = 16 -> actor input = 45 + 3 + 16 = 64
        tar_hidden_dims = [128, 64]
        num_prototype = 32
        temperature = 3.0
        learning_rate = 1.e-3
        max_grad_norm = 10.0

    class runner(LeggedRobotHIMCfgPPO.runner):
        experiment_name = 'go2_benchmark'
        run_name = 'bench_him' + get_simulator_suffix()
        save_interval = 200
        max_iterations = 3000
        # --- iteration-based command schedule (shared by all benchmark methods) ---
        command_schedule = [
            {"start_iteration": 0,   "lin_vel_x": [-0.5, 0.5]},
            {"start_iteration": 500, "lin_vel_x": [-1.0, 1.0]},
        ]
        # --- in-distribution eval + best_tracking.pt (shared by all methods) ---
        eval_interval = 200      # iters between evals (0 disables); aligns with save_interval
        eval_steps = 1100        # >= max_episode_length+1 (1001) so returns are complete
        eval_warmup = 50         # unrecorded settling steps
        eval_seed = 12345        # fixed -> comparable across checkpoints within a run
        eval_fall_guard = 0.05   # V2 safe fixed-window fall-rate threshold
