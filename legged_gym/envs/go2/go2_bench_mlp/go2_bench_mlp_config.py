from legged_gym import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO
from legged_gym.envs.base.common_cfgs import Go2BenchmarkCommonCfg, get_simulator_suffix


class Go2BenchMlpCfg(Go2BenchmarkCommonCfg):
    """DR + MLP baseline -- the main comparison point ('saf DR')."""
    class env(Go2BenchmarkCommonCfg.env):
        num_observations = 45
        num_privileged_obs = None
        num_actions = 12


class Go2BenchCfgPPO(LeggedRobotCfgPPO):
    """Shared PPO settings for the whole benchmark family: SAME training budget
    for every method (fairness). No-DR / Oracle inherit and only change run_name."""
    class runner(LeggedRobotCfgPPO.runner):
        experiment_name = 'go2_benchmark'
        run_name = 'bench_mlp' + get_simulator_suffix()
        save_interval = 200
        max_iterations = 3000
        # --- in-distribution eval + best.pt (shared by all benchmark methods) ---
        # Frozen, deterministic eval on the training distribution: logs Eval/* and
        # writes best.pt (argmax mean_return, hard-demoted if fall_rate > guard).
        # Same protocol for nodr/mlp/oracle so best.pt selection stays fair.
        eval_interval = 200      # iters between evals (0 disables); aligns with save_interval
        eval_steps = 1100        # >= max_episode_length+1 (1001) so returns are complete
        eval_warmup = 50         # unrecorded settling steps
        eval_seed = 12345        # fixed -> comparable across checkpoints within a run
        eval_fall_guard = 0.25   # fall_rate above this demotes a checkpoint for best.pt
