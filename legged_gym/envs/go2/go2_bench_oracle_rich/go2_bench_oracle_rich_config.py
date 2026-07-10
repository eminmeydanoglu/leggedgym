from legged_gym import *
from legged_gym.envs.base.common_cfgs import get_simulator_suffix
from legged_gym.envs.go2.go2_bench_oracle_id.go2_bench_oracle_id_config import (
    Go2BenchOracleIDCfg, Go2BenchOracleIDCfgPPO,
)


class Go2BenchOracleRichCfg(Go2BenchOracleIDCfg):
    """Rich-P oracle: matched-distribution oracle with an ENRICHED P vector.

    Inherits the NARROW oracle_id physics band (friction [0.5,1.25], mass [-1,1]),
    and additionally randomizes pd-gain and control latency. The actor observes
        P = [friction(1), added_mass(1), com_bias(3), pd_gain(1), ctrl_delay(1)] = 7
    Paired with go2_bench_mlp_rich (identical distribution, no P) to isolate the
    value of observing the two EXTRA physics params on top of friction/mass/com.
    """
    class env(Go2BenchOracleIDCfg.env):
        num_observations = 45 + 7   # 45 proprio + 7-dim privileged P

    class domain_rand(Go2BenchOracleIDCfg.domain_rand):
        randomize_pd_gain = True
        pd_gain_scalar = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]
        randomize_ctrl_delay = True
        ctrl_delay_step_range = [0, 4]


class Go2BenchOracleRichCfgPPO(Go2BenchOracleIDCfgPPO):
    class runner(Go2BenchOracleIDCfgPPO.runner):
        run_name = 'bench_oracle_rich' + get_simulator_suffix()
