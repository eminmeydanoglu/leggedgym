from legged_gym import *
from legged_gym.envs.base.common_cfgs import get_simulator_suffix
from legged_gym.envs.go2.go2_bench_mlp.go2_bench_mlp_config import Go2BenchMlpCfg, Go2BenchCfgPPO


class Go2BenchMlpWideCfg(Go2BenchMlpCfg):
    """Wide matched control: same physics distribution as the wide oracle."""
    class domain_rand(Go2BenchMlpCfg.domain_rand):
        friction_range = [0.1, 2.5]
        added_mass_range = [-2.0, 5.0]


class Go2BenchMlpWideCfgPPO(Go2BenchCfgPPO):
    class runner(Go2BenchCfgPPO.runner):
        run_name = 'bench_mlp_wide' + get_simulator_suffix()
