from legged_gym import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO
from legged_gym.envs.base.common_cfgs import Go2BenchmarkCommonCfg, get_simulator_suffix


class Go2BenchMlpCfg(Go2BenchmarkCommonCfg):
    """DR + MLP baseline -- the main comparison point ('saf DR')."""
    class env(Go2BenchmarkCommonCfg.env):
        num_observations = 45
        # Asymmetric critic gets the true base linear velocity, while the
        # deployable MLP actor remains strictly proprioceptive.
        num_privileged_obs = 45 + 3
        num_actions = 12


class Go2BenchCfgPPO(LeggedRobotCfgPPO):
    """Shared PPO settings for the whole benchmark family: SAME training budget
    for every method (fairness). No-DR / Oracle inherit and only change run_name."""
    class runner(LeggedRobotCfgPPO.runner):
        experiment_name = 'go2_benchmark'
        run_name = 'bench_mlp' + get_simulator_suffix()
        save_interval = 200
        max_iterations = 3000
        # --- iteration-based command schedule (shared by all benchmark methods) ---
        # Replaces the performance-based curriculum (commands.curriculum=False).
        # Every method sees the SAME lin_vel_x distribution at the SAME training
        # stage, so the P5 advantage cannot be confounded by curriculum timing.
        #   it 0-499   : lin_vel_x in [-0.5, 0.5]
        #   it 500-end : lin_vel_x in [-1.0, 1.0]  (== validation field)
        command_schedule = [
            {"start_iteration": 0,   "lin_vel_x": [-0.5, 0.5]},
            {"start_iteration": 500, "lin_vel_x": [-1.0, 1.0]},
        ]
        # --- in-distribution eval + best_tracking.pt (shared by all methods) ---
        # Frozen, deterministic eval on the training distribution: logs Eval/* and
        # writes best_tracking.pt (V2 safety-first tracking selection).
        # Same protocol for nodr/mlp/oracle so checkpoint selection stays fair.
        eval_interval = 200      # iters between evals (0 disables); aligns with save_interval
        eval_steps = 1100        # >= max_episode_length+1 (1001) so returns are complete
        eval_warmup = 50         # unrecorded settling steps
        eval_seed = 12345        # fixed -> comparable across checkpoints within a run
        eval_fall_guard = 0.05   # V2 safe fixed-window fall-rate threshold
