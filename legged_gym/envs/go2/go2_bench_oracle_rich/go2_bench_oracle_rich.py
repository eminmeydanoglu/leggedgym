from legged_gym import *

import torch

from legged_gym.envs.go2.go2_bench_oracle_id.go2_bench_oracle_id import Go2BenchOracleID


class Go2BenchOracleRich(Go2BenchOracleID):
    """Oracle env with an ENRICHED 7-dim privileged block appended to the 45-dim
    proprio observation, noise-free:
        P = [friction(1), added_mass(1), com_bias(3), pd_gain(1), ctrl_delay(1)]
    All 7 dims are normalised by dr_normalize using this task's OWN DR ranges.
    Must match num_observations = 45 + 7 in Go2BenchOracleRichCfg.

    Only compute_observations is overridden: the 45-dim proprio block and its noise
    are identical to GO2; the 7-dim privileged block is appended without noise.
    """

    def compute_observations(self):
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
        # privileged physics params -- NOISE-FREE (the oracle knows the truth)
        privileged = torch.cat((
            self.simulator.dr_friction_values.view(self.num_envs, -1),      # 1
            self.simulator.dr_added_base_mass.view(self.num_envs, -1),      # 1
            self.simulator.dr_base_com_bias.view(self.num_envs, -1),        # 3
            self.simulator.dr_kp_scale_scalar.view(self.num_envs, -1),      # 1
            self.simulator.dr_ctrl_delay.view(self.num_envs, -1),          # 1
        ), dim=-1)
        self.obs_buf = torch.cat((base, privileged), dim=-1)
