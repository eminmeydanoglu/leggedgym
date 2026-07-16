import torch

from legged_gym.envs.go2.go2 import GO2
from legged_gym.envs.go2.go2_v3_physics import Go2V3PhysicsResampleMixin


class Go2V3Mlp(Go2V3PhysicsResampleMixin, GO2):
    """Memoryless DR baseline for the V3 dynamic-physics training contract."""

    def compute_observations(self):
        clean_proprio = torch.cat((
            self.commands[:, :3] * self.commands_scale,
            self.simulator.projected_gravity,
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,
            (self.simulator.dof_pos - self.simulator.default_dof_pos)
            * self.obs_scales.dof_pos,
            self.simulator.dof_vel * self.obs_scales.dof_vel,
            self.actions,
        ), dim=-1)
        self.obs_buf = clean_proprio.clone()
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        base_lin_vel = self.simulator.base_lin_vel * self.obs_scales.lin_vel
        self.privileged_obs_buf = torch.cat((clean_proprio, base_lin_vel), dim=-1)
