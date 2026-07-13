from legged_gym import *

import torch

from legged_gym.envs.base.legged_robot_ee import LeggedRobotEE
from legged_gym.utils.math_utils import torch_rand_float


class Go2BenchSysID(LeggedRobotEE):
    """Explicit SysID (explicit estimator) on the frozen flat benchmark substrate.

    Rebound from go2_ee (Go2RoughCommonCfg) to Go2BenchmarkCommonCfg. The rough
    estimator labels (link-contact 17 + foot-clearance 4) and the height map /
    dr pd-gain critic block are dropped -- they do not exist on flat ground.

    Buffer layout (GO2 order):
        obs_buf (local)        = 45  proprio
        estimator_labels_buf   = [base_lin_vel(3), P(5)] = 8   (explicit estimator target)
        single critic obs      = [obs_buf(45), estimator_labels(8)] = 53
        privileged_obs_buf     = critic single x c_frame_stack = 265  (= critic obs)
        estimator_features_buf = obs_buf x frame_stack = 900   (estimator input)

    Per the study decision, the SysID target is BOTH the velocity state and the
    physics parameters: base_lin_vel(3) + canonical P(5).
    """

    def compute_observations(self):
        # 45-dim proprio base -- GO2 order (local; EE keeps obs history, not self.obs_buf)
        obs_buf = torch.cat((
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

        # Estimator labels (regression target) = [base_lin_vel(3), P(5)] = 8
        self.estimator_labels_buf = torch.cat((base_lin_vel, P), dim=-1)

        # Critic observation (clean) = [obs_buf, estimator_labels] = 53
        single_critic_obs = torch.cat((obs_buf, self.estimator_labels_buf), dim=-1)
        self.critic_obs_deque.append(single_critic_obs)
        self.privileged_obs_buf = torch.cat(
            [self.critic_obs_deque[i] for i in range(self.critic_obs_deque.maxlen)],
            dim=-1,
        )

        # add noise to the actor obs only
        if self.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec

        # push (noisy) obs_buf to the estimator feature history
        self.obs_history_deque.append(obs_buf)
        self.estimator_features_buf = torch.cat(
            [self.obs_history_deque[i] for i in range(self.obs_history_deque.maxlen)],
            dim=-1,
        )

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
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
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
