from legged_gym import *
from legged_gym.envs.base.common_cfgs import get_simulator_suffix
from legged_gym.envs.go2.go2_bench_oracle_id.go2_bench_oracle_id_config import (
    Go2BenchOracleIDCfg, Go2BenchOracleIDCfgPPO,
)


class Go2BenchOracleIDVelCfg(Go2BenchOracleIDCfg):
    """Go2BenchOracleID + true base_lin_vel(3) appended to the observation.

    Inherits EVERYTHING from Go2BenchOracleID (narrow matched DR band
    friction [0.5,1.25] / mass [-1,1], reward, command schedule, PPO budget);
    the ONLY change is the observation width: 45 proprio + 5 P + 3 base_lin_vel.
    """
    class env(Go2BenchOracleIDCfg.env):
        num_observations = 45 + 5 + 3   # proprio + P + base_lin_vel
        # Its actor observation already contains true base_lin_vel, so the
        # separate critic tensor has the same 53-dimensional layout.
        num_privileged_obs = 45 + 5 + 3
        num_actions = 12


class Go2BenchOracleIDVelCfgPPO(Go2BenchOracleIDCfgPPO):
    class runner(Go2BenchOracleIDCfgPPO.runner):
        run_name = 'bench_oracle_id_vel' + get_simulator_suffix()
