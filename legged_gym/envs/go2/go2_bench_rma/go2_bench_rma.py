from legged_gym import *

import torch

from legged_gym.envs.base.legged_robot_ts import LeggedRobotTS
from legged_gym.utils.math_utils import torch_rand_float


class Go2BenchRMA(LeggedRobotTS):
    """RMA (teacher-student) on the frozen flat benchmark substrate.

    Rebound from go2_ts (Go2RoughCommonCfg) to Go2BenchmarkCommonCfg: the rough
    substrate's height-map / link-contact / terrain-around-feet buffers do NOT
    exist on flat ground, so the rough compute_observations and its
    height_around_feet reward overrides are dropped. Only the flat-safe pieces
    are kept; rewards fall back to the base LeggedRobot implementations.

    Observation layout (GO2 order, identical to the rest of the benchmark family):
        obs_buf              = 45  proprio
        privileged_obs_buf   = [P(5), base_lin_vel(3)] = 8   (privilege encoder input)
        single critic obs    = [obs_buf(45), P(5), base_lin_vel(3)] = 53
        obs_history          = obs_buf x frame_stack

    Canonical P = [dr_friction_values(1), dr_added_base_mass(1),
    dr_base_com_bias(3)] = 5, the SAME normalised band-DR vector the oracle
    consumes and the estimator/probe regresses (single source of truth).
    """

    def compute_observations(self):
        # 45-dim proprio base -- GO2 order (commands, proj_grav, ang_vel, dof_pos, dof_vel, actions)
        self.obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,                     # 3
            self.simulator.projected_gravity,                               # 3
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,          # 3
            (self.simulator.dof_pos - self.simulator.default_dof_pos)
                * self.obs_scales.dof_pos,                                  # 12
            self.simulator.dof_vel * self.obs_scales.dof_vel,               # 12
            self.actions,                                                   # 12
        ), dim=-1)

        # canonical privileged physics vector P (5), normalised band-DR accessors
        P = torch.cat((
            self.simulator.dr_friction_values.view(self.num_envs, -1),      # 1
            self.simulator.dr_added_base_mass.view(self.num_envs, -1),      # 1
            self.simulator.dr_base_com_bias.view(self.num_envs, -1),        # 3
        ), dim=-1)
        base_lin_vel = self.simulator.base_lin_vel * self.obs_scales.lin_vel  # 3

        # Critic observation (clean, un-noised) = [obs_buf, P, base_lin_vel] = 53
        critic_obs = torch.cat((self.obs_buf, P, base_lin_vel), dim=-1)
        self.critic_obs_deque.append(critic_obs)
        self.critic_obs_buf = torch.cat(
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

        # Privileged observation (privilege encoder input) = [P, base_lin_vel] = 8
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.cat((P, base_lin_vel), dim=-1)

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
