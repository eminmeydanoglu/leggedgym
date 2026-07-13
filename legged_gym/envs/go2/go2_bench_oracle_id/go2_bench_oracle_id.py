from legged_gym.envs.go2.go2_bench_oracle.go2_bench_oracle import Go2BenchOracle

import torch


class Go2BenchOracleID(Go2BenchOracle):
    """Narrow-band oracle with a velocity-aware asymmetric critic.

    The actor receives ``[noisy_proprio(45), P(5)]``. The critic receives the
    corresponding clean proprio block plus the noise-free P5 and true base
    linear velocity.
    """

    def compute_observations(self):
        super().compute_observations()
        clean_proprio = torch.cat((
            self.commands[:, :3] * self.commands_scale,
            self.simulator.projected_gravity,
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,
            (self.simulator.dof_pos - self.simulator.default_dof_pos)
                * self.obs_scales.dof_pos,
            self.simulator.dof_vel * self.obs_scales.dof_vel,
            self.actions,
        ), dim=-1)
        # P5 is the clean, noise-free suffix produced by Go2BenchOracle.
        p5 = self.obs_buf[:, 45:50]
        base_lin_vel = self.simulator.base_lin_vel * self.obs_scales.lin_vel
        self.privileged_obs_buf = torch.cat((clean_proprio, p5, base_lin_vel), dim=-1)
