from legged_gym import *

import torch

from legged_gym.envs.go2.go2_bench_oracle.go2_bench_oracle import Go2BenchOracle


class Go2BenchOracleIDVel(Go2BenchOracle):
    """Narrow-matched oracle that ALSO sees the TRUE base linear velocity.

    Identical to Go2BenchOracleID (same narrow DR band, reward, command schedule,
    PPO budget) EXCEPT the actor/critic observation additionally appends the true
    base_lin_vel(3) AFTER the P(5) block:

        obs = [ proprio(45), P(5), base_lin_vel(3) ] = 53

    Rationale: the implicit/explicit adaptation methods (RMA / DreamWaQ / SysID)
    all regress base_lin_vel in addition to P. For the oracle to be a genuine
    ceiling of *what those methods try to estimate*, it should see that velocity
    state too -- not just the static physics params.

    Design choices so the comparison against Go2BenchOracleID is clean:
      * The first 50 dims are byte-identical to Go2BenchOracleID (proprio + P), so
        the ONLY difference is the appended velocity block.
      * base_lin_vel is scaled by obs_scales.lin_vel -- the SAME scaling the
        RMA/DreamWaQ/SysID envs use for their velocity label.
      * The velocity block is NOISE-FREE (oracle truth), like the P block.
      * Ordering [obs, P, base_lin_vel] matches RMA's critic layout.

    Only compute_observations is overridden; everything else is Go2BenchOracle.
    Must match num_observations = 45 + 5 + 3 in Go2BenchOracleIDVelCfg.
    """

    def compute_observations(self):
        # 45-dim proprio base -- identical to GO2 / Go2BenchOracle
        base = torch.cat((
            self.commands[:, :3] * self.commands_scale,                     # 3
            self.simulator.projected_gravity,                               # 3
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,          # 3
            (self.simulator.dof_pos - self.simulator.default_dof_pos)
                * self.obs_scales.dof_pos,                                  # 12
            self.simulator.dof_vel * self.obs_scales.dof_vel,               # 12
            self.actions,                                                   # 12
        ), dim=-1)
        # proprio noise, same scales as GO2 (noise_scale_vec[:45] holds them)
        if self.add_noise:
            base += (2 * torch.rand_like(base) - 1) * self.noise_scale_vec[:base.shape[1]]

        # privileged physics params P (5) -- NOISE-FREE (oracle knows the truth)
        # [friction, added_base_mass, com_x, com_y, com_z]
        privileged = torch.cat((
            self.simulator.dr_friction_values.view(self.num_envs, -1),      # 1
            self.simulator.dr_added_base_mass.view(self.num_envs, -1),      # 1
            self.simulator.dr_base_com_bias.view(self.num_envs, -1),        # 3
        ), dim=-1)

        # true base linear velocity (3) -- NOISE-FREE, scaled like the estimators
        base_lin_vel = self.simulator.base_lin_vel * self.obs_scales.lin_vel  # 3

        self.obs_buf = torch.cat((base, privileged, base_lin_vel), dim=-1)
        # The actor already sees the complete privileged state; keep a separate
        # critic buffer so the runner takes the same asymmetric-critic path as
        # the other benchmark variants.
        self.privileged_obs_buf = self.obs_buf
