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
