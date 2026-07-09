from legged_gym import *
from legged_gym.envs.base.common_cfgs import get_simulator_suffix
from legged_gym.envs.go2.go2_bench_mlp.go2_bench_mlp_config import Go2BenchMlpCfg, Go2BenchCfgPPO


class Go2BenchOracleCfg(Go2BenchMlpCfg):
    """Oracle / privileged policy -- the ceiling. Actor additionally sees the TRUE
    physics params P = [friction(1), added_mass(1), com_bias(3)] = 5 dims.
    'If identification were perfect, how well could you do?'"""
    class env(Go2BenchMlpCfg.env):
        num_observations = 45 + 5   # 45 proprio + privileged P
        num_privileged_obs = None
        num_actions = 12


class Go2BenchOracleCfgPPO(Go2BenchCfgPPO):
    class runner(Go2BenchCfgPPO.runner):
        run_name = 'bench_oracle' + get_simulator_suffix()
