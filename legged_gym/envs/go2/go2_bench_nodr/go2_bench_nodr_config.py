from legged_gym import *
from legged_gym.envs.base.common_cfgs import Go2BenchmarkCommonCfg, get_simulator_suffix
from legged_gym.envs.go2.go2_bench_mlp.go2_bench_mlp_config import Go2BenchMlpCfg, Go2BenchCfgPPO


class Go2BenchNoDRCfg(Go2BenchMlpCfg):
    """No-DR MLP -- the floor. DR fully disabled; everything else identical.
    Shows how far a memoryless policy collapses OOD when it never saw variation."""
    class domain_rand(Go2BenchmarkCommonCfg.domain_rand):
        randomize_friction = False
        randomize_base_mass = False
        randomize_com_displacement = False
        push_robots = False


class Go2BenchNoDRCfgPPO(Go2BenchCfgPPO):
    class runner(Go2BenchCfgPPO.runner):
        run_name = 'bench_nodr' + get_simulator_suffix()
