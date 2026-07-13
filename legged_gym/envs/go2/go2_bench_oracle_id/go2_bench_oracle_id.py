from legged_gym.envs.go2.go2_bench_oracle.go2_bench_oracle import Go2BenchOracle

import torch


class Go2BenchOracleID(Go2BenchOracle):
    """Narrow-band oracle with a velocity-aware asymmetric critic.

    The actor remains ``[proprio(45), P(5)]``. The critic receives that actor
    observation plus the noise-free true base linear velocity.
    """

    def compute_observations(self):
        super().compute_observations()
        base_lin_vel = self.simulator.base_lin_vel * self.obs_scales.lin_vel
        self.privileged_obs_buf = torch.cat((self.obs_buf, base_lin_vel), dim=-1)
