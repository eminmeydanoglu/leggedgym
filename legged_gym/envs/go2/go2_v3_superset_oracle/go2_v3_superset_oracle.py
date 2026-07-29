"""Superset oracle for V3: the full SysID information set plus truth labels."""

from collections import deque

import torch

from legged_gym.envs.go2.go2 import GO2
from legged_gym.envs.go2.go2_v3_physics import Go2V3PhysicsResampleMixin


class Go2V3SupersetOracle(Go2V3PhysicsResampleMixin, GO2):
    """Actor sees noisy proprio history and the true current ``[velocity, P5]``.

    This is intentionally a training ceiling, not a deployable method.  It sees
    the exact 20-frame information stream used by SysID's estimator and the two
    quantities that estimator has to infer, so SysID cannot have a structural
    information advantage over it.
    """

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.num_single_obs = cfg.env.num_single_obs
        self.frame_stack = cfg.env.frame_stack

    def _init_buffers(self):
        super()._init_buffers()
        self.obs_history_deque = deque(maxlen=self.frame_stack)
        self.clean_history_deque = deque(maxlen=self.frame_stack)
        for history in (self.obs_history_deque, self.clean_history_deque):
            for _ in range(self.frame_stack):
                history.append(torch.zeros(
                    self.num_envs,
                    self.num_single_obs,
                    dtype=torch.float,
                    device=self.device,
                ))

    def _one_step_proprio(self):
        return torch.cat((
            self.commands[:, :3] * self.commands_scale,
            self.simulator.projected_gravity,
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,
            (self.simulator.dof_pos - self.simulator.default_dof_pos)
            * self.obs_scales.dof_pos,
            self.simulator.dof_vel * self.obs_scales.dof_vel,
            self.actions,
        ), dim=-1)

    def _p5_and_velocity(self):
        p5 = torch.cat((
            self.simulator.dr_friction_values.view(self.num_envs, -1),
            self.simulator.dr_added_base_mass.view(self.num_envs, -1),
            self.simulator.dr_base_com_bias.view(self.num_envs, -1),
        ), dim=-1)
        velocity = self.simulator.base_lin_vel * self.obs_scales.lin_vel
        return p5, velocity

    def _terrain_height_map(self):
        """Noise-free, yaw-aligned local height map for the V4 oracle only."""
        if not self.cfg.terrain.measure_heights:
            raise RuntimeError("oracle_height_map requires terrain.measure_heights=True")
        return torch.clip(
            self.simulator.base_pos[:, 2].unsqueeze(1) - 0.5
            - self.simulator.measured_heights,
            -1.0,
            1.0,
        ) * self.obs_scales.height_measurements

    def compute_observations(self):
        clean_current = self._one_step_proprio()
        noisy_current = clean_current.clone()
        if self.add_noise:
            noisy_current += (2 * torch.rand_like(noisy_current) - 1) * self.noise_scale_vec

        self.obs_history_deque.append(noisy_current)
        self.clean_history_deque.append(clean_current)
        noisy_history = torch.cat(list(self.obs_history_deque), dim=-1)
        clean_history = torch.cat(list(self.clean_history_deque), dim=-1)
        p5, velocity = self._p5_and_velocity()

        actor_parts = (noisy_history, velocity, p5)
        critic_parts = (clean_history, velocity, p5)
        if getattr(self.cfg.env, "oracle_height_map", False):
            # This is privileged perception, not a learned terrain estimate.
            height_map = self._terrain_height_map()
            actor_parts += (height_map,)
            critic_parts += (height_map,)

        self.obs_buf = torch.cat(actor_parts, dim=-1)
        self.privileged_obs_buf = torch.cat(critic_parts, dim=-1)

    def _get_noise_scale_vec(self):
        noise_vec = torch.zeros(self.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        scales = self.cfg.noise.noise_scales
        level = self.cfg.noise.noise_level
        noise_vec[:3] = 0.0
        noise_vec[3:6] = scales.gravity * level
        noise_vec[6:9] = scales.ang_vel * level * self.obs_scales.ang_vel
        noise_vec[9:21] = scales.dof_pos * level * self.obs_scales.dof_pos
        noise_vec[21:33] = scales.dof_vel * level * self.obs_scales.dof_vel
        noise_vec[33:45] = 0.0
        return noise_vec

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # Snapshot the deques: compute_observations() may append (or a Viser
        # GUI worker may race the main step loop) while we zero history slots.
        for history in (self.obs_history_deque, self.clean_history_deque):
            for frame in list(history):
                frame[env_ids] = 0.0
