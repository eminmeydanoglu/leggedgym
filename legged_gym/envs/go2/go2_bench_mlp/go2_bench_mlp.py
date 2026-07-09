from legged_gym import *

from legged_gym.envs.go2.go2 import GO2


class Go2BenchMlp(GO2):
    """DR + MLP baseline env. Identical dynamics/obs to GO2; exists as its own
    task so the whole benchmark family shares one frozen substrate
    (Go2BenchmarkCommonCfg in common_cfgs.py)."""
    pass
