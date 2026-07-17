"""Corrected HIM implementation on the shared V3 dynamic-physics substrate."""

from legged_gym.envs.go2.go2_bench_him.go2_bench_him import Go2BenchHIM
from legged_gym.envs.go2.go2_v3_physics import Go2V3PhysicsResampleMixin


class Go2V3HIMFixed(Go2V3PhysicsResampleMixin, Go2BenchHIM):
    """The fixed HIM estimator with V3 mass/CoM resampling enabled."""

    pass
