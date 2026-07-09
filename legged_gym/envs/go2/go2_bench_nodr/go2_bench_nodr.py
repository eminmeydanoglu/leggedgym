from legged_gym import *

from legged_gym.envs.go2.go2_bench_mlp.go2_bench_mlp import Go2BenchMlp


class Go2BenchNoDR(Go2BenchMlp):
    """No-DR baseline env -- same as the MLP baseline; DR is disabled via config."""
    pass
