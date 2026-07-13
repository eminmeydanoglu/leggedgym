from legged_gym import *
from legged_gym.envs.base.common_cfgs import get_simulator_suffix
from legged_gym.envs.go2.go2_bench_oracle.go2_bench_oracle_config import (
    Go2BenchOracleCfg, Go2BenchOracleCfgPPO,
)


class Go2BenchOracleIDCfg(Go2BenchOracleCfg):
    """Matched-distribution oracle: isolates the value of observing true P."""
    class env(Go2BenchOracleCfg.env):
        # Actor: [proprio(45), P(5)]. Critic additionally gets true base_lin_vel.
        num_privileged_obs = 45 + 5 + 3

    class domain_rand(Go2BenchOracleCfg.domain_rand):
        friction_range = [0.5, 1.25]
        added_mass_range = [-1.0, 1.0]


class Go2BenchOracleIDCfgPPO(Go2BenchOracleCfgPPO):
    class runner(Go2BenchOracleCfgPPO.runner):
        run_name = 'bench_oracle_id' + get_simulator_suffix()
