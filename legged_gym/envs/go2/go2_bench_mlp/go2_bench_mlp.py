from legged_gym import *

import torch

from legged_gym.envs.go2.go2 import GO2


class Go2BenchMlp(GO2):
    """DR + MLP baseline with an asymmetric velocity-aware critic.

    The deployable actor receives the noisy 45-dimensional GO2 proprioceptive
    observation. The asymmetric critic receives its clean counterpart plus the
    noise-free true base linear velocity: ``[clean_proprio(45), base_lin_vel(3)]``.
    """

    def compute_observations(self):
        # Keep the actor observation byte-for-byte aligned with GO2's 45-dim layout.
        # Retain a clean copy for the asymmetric critic, matching the benchmark
        # adaptation methods' critic-information protocol.
        clean_proprio = torch.cat((
            self.commands[:, :3] * self.commands_scale,                    # 3
            self.simulator.projected_gravity,                               # 3
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,          # 3
            (self.simulator.dof_pos - self.simulator.default_dof_pos)
                * self.obs_scales.dof_pos,                                  # 12
            self.simulator.dof_vel * self.obs_scales.dof_vel,               # 12
            self.actions,                                                    # 12
        ), dim=-1)

        self.obs_buf = clean_proprio.clone()
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        # Do not call GO2's generic privileged-observation path: its layout is
        # broader than this benchmark's intended critic input.
        base_lin_vel = self.simulator.base_lin_vel * self.obs_scales.lin_vel
        self.privileged_obs_buf = torch.cat((clean_proprio, base_lin_vel), dim=-1)
