from legged_gym import *

import torch

from legged_gym.envs.base.legged_robot_dreamwaq import LeggedRobotDreamwaq
from legged_gym.utils.math_utils import torch_rand_float


class Go2BenchDreamwaq(LeggedRobotDreamwaq):
    """DreamWaQ on the frozen flat benchmark substrate.

    Rebound from go2_dreamwaq (Go2RoughCommonCfg) to Go2BenchmarkCommonCfg. The
    rough explicit labels (link-contact 17 + foot-clearance 4) and the height
    map / dr pd-gain critic block do not exist on flat ground and are dropped.

    Buffer layout (GO2 order):
        obs_buf              = 45  proprio
        explicit_labels_buf  = base_lin_vel(3)              (VAE explicit head target)
        next_state_buf       = 45  proprio next-state       (VAE decoder target)
        single critic obs    = [base_lin_vel(3), obs_buf(45), P(5)] = 53
        privileged_obs_buf   = critic single x c_frame_stack = 265
        obs_history          = obs_buf x frame_stack

    Canonical P = [dr_friction_values(1), dr_added_base_mass(1),
    dr_base_com_bias(3)] = 5 (normalised band-DR, single source of truth).
    """

    def compute_observations(self):
        # 45-dim proprio base -- GO2 order
        self.obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,                     # 3
            self.simulator.projected_gravity,                               # 3
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,          # 3
            (self.simulator.dof_pos - self.simulator.default_dof_pos)
                * self.obs_scales.dof_pos,                                  # 12
            self.simulator.dof_vel * self.obs_scales.dof_vel,               # 12
            self.actions,                                                   # 12
        ), dim=-1)

        # canonical privileged physics vector P (5)
        P = torch.cat((
            self.simulator.dr_friction_values.view(self.num_envs, -1),      # 1
            self.simulator.dr_added_base_mass.view(self.num_envs, -1),      # 1
            self.simulator.dr_base_com_bias.view(self.num_envs, -1),        # 3
        ), dim=-1)
        base_lin_vel = self.simulator.base_lin_vel * self.obs_scales.lin_vel  # 3

        # Critic observation (clean) = [base_lin_vel, obs_buf, P] = 53
        critic_obs = torch.cat((base_lin_vel, self.obs_buf, P), dim=-1)
        self.critic_obs_deque.append(critic_obs)
        self.privileged_obs_buf = torch.cat(
            [self.critic_obs_deque[i] for i in range(self.critic_obs_deque.maxlen)],
            dim=-1,
        )

        # add noise to the actor obs only
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        # push (noisy) obs_buf to obs_history
        self.obs_history_deque.append(self.obs_buf)
        self.obs_history = torch.cat(
            [self.obs_history_deque[i] for i in range(self.obs_history_deque.maxlen)],
            dim=-1,
        )

        # next state (VAE decoder target) -- proprio with scaled actions
        self.next_state_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,                     # 3
            self.simulator.projected_gravity,                               # 3
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,          # 3
            (self.simulator.dof_pos - self.simulator.default_dof_pos)
                * self.obs_scales.dof_pos,                                  # 12
            self.simulator.dof_vel * self.obs_scales.dof_vel,               # 12
            self.actions * self.cfg.control.action_scale,                   # 12
        ), dim=-1)

        # explicit info label (VAE explicit head target) = base_lin_vel (3)
        self.explicit_labels_buf = base_lin_vel

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environments (GO2 style). """
        dof_pos = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float,
                              device=self.device, requires_grad=False)
        dof_vel = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float,
                              device=self.device, requires_grad=False)
        dof_pos[:, [0, 3, 6, 9]] = self.simulator.default_dof_pos[:, [0, 3, 6, 9]] + \
            torch_rand_float(-0.2, 0.2, (len(env_ids), 4), self.device)
        dof_pos[:, [1, 4, 7, 10]] = self.simulator.default_dof_pos[:, [1, 4, 7, 10]] + \
            torch_rand_float(-0.4, 0.4, (len(env_ids), 4), self.device)
        dof_pos[:, [2, 5, 8, 11]] = self.simulator.default_dof_pos[:, [2, 5, 8, 11]] + \
            torch_rand_float(-0.4, 0.4, (len(env_ids), 4), self.device)
        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)

    def _get_noise_scale_vec(self):
        """ Noise scale vector for the 45-dim GO2 proprio obs (identical to GO2). """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = 0.  # commands
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[9:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[33:45] = 0.  # previous actions
        return noise_vec
