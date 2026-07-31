# Go2 MoE-CTS arm of the go2_rl_gym (wty-yy) port.
# Student/privileged observation composition follows the vendored
# go2_rl_gym/legged_gym/envs/go2/go2_env.py exactly (45 / 263 dims); the
# history/critic buffer plumbing mirrors the host Go2CTS / LeggedRobotTS.

from legged_gym import *

import torch

from legged_gym.envs.base.legged_robot_cts import LeggedRobotCTS
from legged_gym.envs.go2.go2_moects.wty_curriculum_mixin import WtyCurriculumMixin


class Go2MoECTS(WtyCurriculumMixin, LeggedRobotCTS):
    """MoE-CTS arm: vendored obs contract on the ported curriculum substrate.

    Observation contract (vendored go2_env.py):
        obs_buf (45)             = [base_ang_vel(3), projected_gravity(3),
                                    commands(3), dof_pos(12), dof_vel(12),
                                    actions(12)] + observation noise
        privileged_obs_buf (263) = [base_lin_vel(3), <clean 45>,
                                    feet_contact_forces(4), dof_torques(12),
                                    dof_acc(12), height_measurements(187)]
        obs_history (225)        = frame_stack(5) x noisy 45, flat
        critic_obs_buf (263)     = privileged_obs_buf, single frame
                                   (c_frame_stack=1); ActorCriticMoECTS
                                   derives the teacher latent from it in
                                   evaluate(), so the layouts must match.
    """

    def compute_observations(self):
        # student obs (45), clean -- noise is added after the privileged obs
        # is built so privileged/critic obs stay clean (vendored ordering).
        self.obs_buf = torch.cat((
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,              # 3
            self.simulator.projected_gravity,                                   # 3
            self.commands[:, :3] * self.commands_scale,                         # 3
            (self.simulator.dof_pos - self.simulator.default_dof_pos)
                * self.obs_scales.dof_pos,                                      # 12
            self.simulator.dof_vel * self.obs_scales.dof_vel,                   # 12
            self.actions,                                                       # 12
        ), dim=-1)

        heights = torch.clip(self.simulator.base_pos[:, 2].unsqueeze(
            1) - 0.5 - self.simulator.measured_heights, -1, 1.0) * self.obs_scales.height_measurements

        # privileged obs (263), vendored layout: lin_vel + clean 45 + contact
        # forces + normalized torques + dof accelerations + height scan
        privileged_obs = torch.cat((
            self.simulator.base_lin_vel * self.obs_scales.lin_vel,              # 3
            self.obs_buf,                                                       # 45
            self.feet_force_norm * 1e-3,                                        # 4
            self.simulator.torques / self.simulator.torque_limits,              # 12
            (self.simulator.last_dof_vel - self.simulator.dof_vel) / self.dt * 1e-4,  # 12
            heights,                                                            # 187
        ), dim=-1)
        self.privileged_obs_buf = privileged_obs

        # critic obs == privileged obs (single frame, c_frame_stack=1)
        self.critic_obs_deque.append(privileged_obs)
        self.critic_obs_buf = torch.cat(
            [self.critic_obs_deque[i]
                for i in range(self.critic_obs_deque.maxlen)],
            dim=-1,
        )

        # noise on the student obs only
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        # push obs_buf to obs_history (host LeggedRobotTS convention)
        self.obs_history_deque.append(self.obs_buf)
        self.obs_history = torch.cat(
            [self.obs_history_deque[i]
                for i in range(self.obs_history_deque.maxlen)],
            dim=-1,
        )

    def _get_noise_scale_vec(self):
        """Noise vector for the vendored 45-dim obs layout
        [vendored go2_env.py _get_noise_scale_vec]:
        [ang_vel(3), gravity(3), commands(3), dof_pos(12), dof_vel(12), actions(12)].
        [NOTE]: Must be adapted when changing the observations structure
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.  # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0.  # previous actions
        return noise_vec
