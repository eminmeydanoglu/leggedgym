# Go2 HIM arm of the go2_rl_gym (wty-yy) port.
# Identical terrain / curriculum / reward machinery as the MoE-CTS arm
# (WtyCurriculumMixin); the observation contract is inherited from
# Go2BenchHIM, so the policy architecture is swapped between MoE-CTS and HIM
# by task name only.

from legged_gym.envs.go2.go2_bench_him.go2_bench_him import Go2BenchHIM
from legged_gym.envs.go2.go2_moects.wty_curriculum_mixin import WtyCurriculumMixin


class Go2MoECTSHIM(WtyCurriculumMixin, Go2BenchHIM):
    """HIM arm: go2_rl_gym curriculum substrate with the HIM obs contract.

    Inherits from Go2BenchHIM: noisy 6x45 actor history obs_buf (270,
    newest-first), 5x53 privileged critic frames (265) and the standard
    5-tuple step contract that HIMRunner / the eval harness expect. The mixin
    overrides only the curriculum / command / reward methods, never
    compute_observations.
    """
