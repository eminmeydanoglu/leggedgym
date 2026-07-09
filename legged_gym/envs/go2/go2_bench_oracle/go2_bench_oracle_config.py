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

    class domain_rand(Go2BenchMlpCfg.domain_rand):
        # The oracle is the ceiling/reference, NOT a method under test. It must
        # be competent -- and see UN-CLAMPED privileged labels -- across the
        # ENTIRE eval sweep, including what is OOD for the band-trained methods.
        # Widen its training ranges to cover the eval grids in
        # scripts/eval/dr_axes.py. dr_normalize uses these same ranges, so this
        # also removes the [-1,1] clamp saturation at OOD grid points.
        # (The other benchmark methods keep the narrow [0.5,1.25] / [-1,1] band.)
        friction_range = [0.1, 2.5]     # eval grid spans 0.1 .. 2.5
        added_mass_range = [-2.0, 5.0]  # eval grid spans -2 .. 5


class Go2BenchOracleCfgPPO(Go2BenchCfgPPO):
    class runner(Go2BenchCfgPPO.runner):
        run_name = 'bench_oracle' + get_simulator_suffix()
