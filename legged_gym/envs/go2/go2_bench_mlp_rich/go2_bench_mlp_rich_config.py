from legged_gym import *
from legged_gym.envs.base.common_cfgs import get_simulator_suffix
from legged_gym.envs.go2.go2_bench_mlp.go2_bench_mlp_config import Go2BenchMlpCfg, Go2BenchCfgPPO


class Go2BenchMlpRichCfg(Go2BenchMlpCfg):
    """Rich-P DR-MLP: hidden-P baseline for the enriched-P study cell.

    Same NARROW friction/mass/com band as go2_bench_mlp, but ADDITIONALLY
    randomizes pd-gain (kp/kd scale) and control latency -- the two axes we add
    to the oracle's privileged P. This is the no-P matched control for
    go2_bench_oracle_rich: same physics distribution, differs only in whether the
    policy observes P. Does NOT touch the frozen 5-task factorial
    (mlp / oracle_id / mlp_wide / oracle); trained in a separate Wave-2 run.
    """
    class domain_rand(Go2BenchMlpCfg.domain_rand):
        randomize_pd_gain = True
        pd_gain_scalar = True          # single per-env gain scale (clean P label)
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]
        randomize_ctrl_delay = True
        ctrl_delay_step_range = [0, 4]  # 0..4 control steps of latency


class Go2BenchMlpRichCfgPPO(Go2BenchCfgPPO):
    class runner(Go2BenchCfgPPO.runner):
        run_name = 'bench_mlp_rich' + get_simulator_suffix()
