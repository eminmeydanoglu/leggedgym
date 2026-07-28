"""Genesis boundary for symmetric-|vx| frontier tasks and success metrics."""
from __future__ import annotations

import numpy as np
import torch

from legged_gym.utils.ued.genesis_adapter import GenesisUEDAdapter
from .curriculum import FrontierOutcomeBatch


class GenesisFrontierAdapter(GenesisUEDAdapter):
    """Reuse the proven teleport lifecycle while measuring binary success."""

    def __init__(
        self,
        *,
        linear_error_threshold: float = 0.35,
        yaw_error_threshold: float = 0.40,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.linear_error_threshold = float(linear_error_threshold)
        self.yaw_error_threshold = float(yaw_error_threshold)
        self.active_vx_sign = torch.ones(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.episode_linear_error = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.episode_yaw_error = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

    def assign(self, env_ids: torch.Tensor, assignments) -> None:
        ids = self._teleport(env_ids, assignments)
        self.episode_standstill[ids] = False
        signs = torch.where(
            torch.rand(len(ids), device=self.device) < 0.5,
            -torch.ones(len(ids), device=self.device),
            torch.ones(len(ids), device=self.device),
        )
        self.active_vx_sign[ids] = signs

    def resample_commands_within_active_bin(self, env_ids: torch.Tensor) -> None:
        ids = self._env_ids(env_ids)
        if torch.any(self.active_task_id[ids] < 0):
            raise RuntimeError("cannot sample a frontier command before assigning a task")
        magnitude = self.active_vx_lower[ids] + (
            self.active_vx_upper[ids] - self.active_vx_lower[ids]
        ) * torch.rand(len(ids), device=self.device)
        self.commands[ids, 0] = self.active_vx_sign[ids] * magnitude

    def record_step(self, rewards: torch.Tensor) -> None:
        super().record_step(rewards)
        active = (self.active_task_id >= 0) & ~self.episode_standstill
        if not torch.any(active):
            return
        linear_command = self.commands[active, :2]
        linear_error = torch.linalg.vector_norm(
            linear_command - self.simulator.base_lin_vel[active, :2], dim=1
        )
        linear_scale = torch.clamp(
            torch.linalg.vector_norm(linear_command, dim=1), min=0.2
        )
        yaw_error = torch.abs(
            self.commands[active, 2] - self.simulator.base_ang_vel[active, 2]
        ) / torch.clamp(torch.abs(self.commands[active, 2]), min=0.2)
        self.episode_linear_error[active] += linear_error / linear_scale
        self.episode_yaw_error[active] += yaw_error

    def collect_outcomes(
        self,
        env_ids: torch.Tensor,
        *,
        completion_revision: int,
        completion_global_control_steps: int = 0,
        timed_out: torch.Tensor,
    ) -> FrontierOutcomeBatch:
        base = super().collect_outcomes(
            env_ids,
            completion_revision=completion_revision,
            completion_global_control_steps=completion_global_control_steps,
            timed_out=timed_out,
        )
        ids = self._env_ids(env_ids)
        lengths = torch.clamp(self.episode_length[ids], min=1).float()
        mean_linear = self.episode_linear_error[ids] / lengths
        mean_yaw = self.episode_yaw_error[ids] / lengths
        survived = timed_out[ids]
        successes = (
            survived
            & (mean_linear <= self.linear_error_threshold)
            & (mean_yaw <= self.yaw_error_threshold)
        )
        return FrontierOutcomeBatch(
            task_ids=base.task_ids,
            assigned_revision=base.assigned_revision,
            completion_revision=base.completion_revision,
            completion_global_control_steps=base.completion_global_control_steps,
            episodic_returns=base.episodic_returns,
            episode_lengths=base.episode_lengths,
            terminal_reasons=base.terminal_reasons,
            successes=successes.detach().cpu().numpy().astype(np.bool_, copy=True),
            mean_linear_errors=mean_linear.detach().cpu().numpy().astype(np.float64, copy=True),
            mean_yaw_errors=mean_yaw.detach().cpu().numpy().astype(np.float64, copy=True),
        )

    def clear_episode_accumulators(self, env_ids: torch.Tensor) -> None:
        super().clear_episode_accumulators(env_ids)
        ids = self._env_ids(env_ids)
        self.episode_linear_error[ids] = 0.0
        self.episode_yaw_error[ids] = 0.0
