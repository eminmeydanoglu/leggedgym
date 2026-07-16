from legged_gym.envs.go2.go2_bench_sysid.go2_bench_sysid import Go2BenchSysID
from legged_gym.envs.go2.go2_v3_physics import Go2V3PhysicsResampleMixin


class Go2V3SysID(Go2V3PhysicsResampleMixin, Go2BenchSysID):
    """V3 explicit SysID: 20-frame history estimates true velocity plus P5."""

    pass
