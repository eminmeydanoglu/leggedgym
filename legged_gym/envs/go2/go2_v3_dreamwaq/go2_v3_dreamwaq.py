"""V3 dynamic-physics variant of DreamWaQ."""

from legged_gym.envs.go2.go2_bench_dreamwaq.go2_bench_dreamwaq import Go2BenchDreamwaq
from legged_gym.envs.go2.go2_v3_physics import Go2V3PhysicsResampleMixin


class Go2V3Dreamwaq(Go2V3PhysicsResampleMixin, Go2BenchDreamwaq):
    """DreamWaQ with the common V3 distribution and one live mass/CoM switch."""

    pass
