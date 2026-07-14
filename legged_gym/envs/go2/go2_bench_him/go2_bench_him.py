from legged_gym import *

import torch
from collections import deque

from legged_gym.envs.go2.go2 import GO2


class Go2BenchHIM(GO2):
    """Hybrid Internal Model (HIM) on the frozen flat benchmark substrate.

    Unlike DreamWaQ/RMA (which broke the shared single-tensor eval contract),
    HIM follows the HIMLoco structure: the actor observation IS the stacked
    noisy history and the estimator lives inside the actor-critic. So
    ``get_observations()`` returns a single tensor, ``step()`` returns the
    standard 5-tuple, and ``policy(obs)`` works across the whole eval harness
    with zero edits -- HIM keeps the fair command_schedule + Eval-V2 machinery.

    Buffer layout (GO2 order):
        one_step_obs (45)  = [commands*scale(3), projected_gravity(3),
                              base_ang_vel*scale(3), (dof_pos-default)*scale(12),
                              dof_vel*scale(12), actions(12)]
        obs_buf (270)      = noisy one_step history, frame_stack=6, NEWEST-FIRST
                             so obs_buf[:, :45] is the current frame (the actor's
                             current-frame slice; estimator encodes the whole 270)
        privileged_obs_buf (265) = 5 newest-first frames of
                             [base_lin_vel(3), clean one_step_obs(45), P(5)].
                             The newest 53D frame is also the estimator target.

    Canonical P = [dr_friction_values(1), dr_added_base_mass(1),
    dr_base_com_bias(3)] = 5 (normalised band-DR, single source of truth).
    """

    def _one_step_obs(self):
        return torch.cat((
            self.commands[:, :3] * self.commands_scale,                     # 3
            self.simulator.projected_gravity,                               # 3
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,          # 3
            (self.simulator.dof_pos - self.simulator.default_dof_pos)
                * self.obs_scales.dof_pos,                                  # 12
            self.simulator.dof_vel * self.obs_scales.dof_vel,               # 12
            self.actions,                                                   # 12
        ), dim=-1)

    def compute_observations(self):
        # clean current one-step proprio frame (GO2 order)
        one_step_obs = self._one_step_obs()

        # canonical privileged physics vector P (5)
        P = torch.cat((
            self.simulator.dr_friction_values.view(self.num_envs, -1),      # 1
            self.simulator.dr_added_base_mass.view(self.num_envs, -1),      # 1
            self.simulator.dr_base_com_bias.view(self.num_envs, -1),        # 3
        ), dim=-1)
        base_lin_vel = self.simulator.base_lin_vel * self.obs_scales.lin_vel  # 3

        # Clean critic frame = [base_lin_vel, one_step_obs, P] = 53. Stack five
        # frames newest-first, matching the other adaptive benchmark critics.
        # HIMEstimator.update consumes the first (current) frame as its target.
        critic_frame = torch.cat((base_lin_vel, one_step_obs, P), dim=-1)
        self.critic_history_deque.append(critic_frame)
        self.privileged_obs_buf = torch.cat(
            [self.critic_history_deque[self.c_frame_stack - 1 - i]
             for i in range(self.c_frame_stack)],
            dim=-1,
        )

        # noisy current frame -> actor history (noise on the actor obs only)
        noisy_obs = one_step_obs.clone()
        if self.add_noise:
            noisy_obs += (2 * torch.rand_like(noisy_obs) - 1) * self.noise_scale_vec

        # newest-first stacking: obs_buf[:, :num_one_step_obs] == current frame
        self.obs_history_deque.append(noisy_obs)
        self.obs_buf = torch.cat(
            [self.obs_history_deque[self.frame_stack - 1 - i] for i in range(self.frame_stack)],
            dim=-1,
        )

    def _get_noise_scale_vec(self):
        """Noise scale vector for the 45-dim GO2 one-step obs (per-frame)."""
        noise_vec = torch.zeros(self.num_one_step_obs, device=self.device)
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

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.num_one_step_obs = self.cfg.env.num_one_step_obs
        self.frame_stack = self.cfg.env.frame_stack
        self.c_frame_stack = self.cfg.env.c_frame_stack
        self.num_single_critic_obs = self.cfg.env.num_single_critic_obs

    def _init_buffers(self):
        super()._init_buffers()
        # actor observation history (noisy one-step frames)
        self.obs_history_deque = deque(maxlen=self.frame_stack)
        for _ in range(self.frame_stack):
            self.obs_history_deque.append(
                torch.zeros(
                    self.num_envs,
                    self.num_one_step_obs,
                    dtype=torch.float,
                    device=self.device,
                )
            )
        self.critic_history_deque = deque(maxlen=self.c_frame_stack)
        for _ in range(self.c_frame_stack):
            self.critic_history_deque.append(
                torch.zeros(
                    self.num_envs,
                    self.num_single_critic_obs,
                    dtype=torch.float,
                    device=self.device,
                )
            )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # clear obs history for the envs that are reset
        for i in range(self.obs_history_deque.maxlen):
            self.obs_history_deque[i][env_ids] *= 0
        for i in range(self.critic_history_deque.maxlen):
            self.critic_history_deque[i][env_ids] *= 0
