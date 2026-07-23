"""Torch/Genesis boundary for applying immutable UED task assignments.

The curriculum itself remains pure NumPy.  This module only translates its
stable task identities into per-environment Genesis state and reports completed
episodes back using the assignment that was active before a reset.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch

from .episode_curriculum import EpisodeOutcomeBatch, TaskAssignmentBatch
from .task_space import TaskSpace


class GenesisUEDAdapter:
    """Own per-environment UED provenance at the Genesis/Torch boundary."""

    def __init__(
        self,
        *,
        task_space: TaskSpace,
        simulator: object,
        commands: torch.Tensor,
        command_ranges: Mapping[str, Sequence[float]],
        device: torch.device | str,
    ) -> None:
        if commands.ndim != 2 or commands.shape[1] < 1:
            raise ValueError("commands must be a [num_envs, >=1] tensor")
        if not getattr(simulator, "custom_origins", False) or not hasattr(simulator, "_terrain_origins"):
            raise ValueError("UED terrain teleport requires custom terrain origins")
        self.task_space = task_space
        self.simulator = simulator
        self.commands = commands
        self.command_ranges = command_ranges
        self.device = torch.device(device)
        self.num_envs = int(commands.shape[0])
        self.active_task_id = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.active_sampler_revision = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.active_vx_lower = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.active_vx_upper = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.episode_return = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.episode_length = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        # False until an assignment starts an episode; a standstill also makes
        # the rollout curriculum-invalid while PPO still consumes it.
        self.episode_valid = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # Standstill is a once-per-episode decision (drawn only at reset).  Mid-
        # episode command resamples must re-apply zeros for these envs and must
        # never re-roll the Bernoulli, or the episode becomes a mixed
        # standstill/motion return that the binary valid flag cannot describe.
        self.episode_standstill = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

    def record_step(self, rewards: torch.Tensor) -> None:
        """Accumulate the actual clipped reward for every active environment."""
        if rewards.shape != (self.num_envs,):
            raise ValueError("rewards must have one value per environment")
        active = self.active_task_id >= 0
        self.episode_return[active] += rewards[active]
        self.episode_length[active] += 1

    def collect_outcomes(
        self,
        env_ids: torch.Tensor,
        *,
        completion_revision: int,
        timed_out: torch.Tensor,
    ) -> EpisodeOutcomeBatch:
        """Copy outcomes for *old* assignments before they are replaced."""
        ids = self._env_ids(env_ids)
        if torch.any(self.active_task_id[ids] < 0):
            raise RuntimeError("cannot collect a UED outcome before assigning a task")
        if timed_out.shape != (self.num_envs,):
            raise ValueError("timed_out must have one value per environment")
        reasons = np.where(
            timed_out[ids].detach().cpu().numpy(), "timeout", "terminal"
        ).astype("U16")
        return EpisodeOutcomeBatch(
            task_ids=self.active_task_id[ids].detach().cpu().numpy().astype(np.int64, copy=True),
            assigned_revision=self.active_sampler_revision[ids].detach().cpu().numpy().astype(np.int64, copy=True),
            completion_revision=int(completion_revision),
            episodic_returns=self.episode_return[ids].detach().cpu().numpy().astype(np.float64, copy=True),
            episode_lengths=self.episode_length[ids].detach().cpu().numpy().astype(np.int64, copy=True),
            terminal_reasons=reasons,
            valid_for_curriculum=self.episode_valid[ids].detach().cpu().numpy().astype(bool, copy=True),
        )

    def assign(self, env_ids: torch.Tensor, assignments: TaskAssignmentBatch) -> None:
        """Atomically apply one decoded assignment per environment by teleport.

        This deliberately changes only tensors already owned by the simulator;
        it never calls terrain construction or changes the static heightfield.
        """
        ids = self._env_ids(env_ids)
        task_ids = np.asarray(assignments.task_ids)
        if task_ids.shape != (len(ids),) or not np.issubdtype(task_ids.dtype, np.integer):
            raise ValueError("assignment task_ids must match env_ids")
        decoded = self.task_space.decode_batch(task_ids)
        terrain_types = torch.as_tensor(decoded.terrain_types, device=self.device, dtype=torch.long)
        terrain_levels = torch.as_tensor(decoded.terrain_levels, device=self.device, dtype=torch.long)
        origins = self.simulator._terrain_origins
        if torch.any(terrain_levels >= origins.shape[0]) or torch.any(terrain_types >= origins.shape[1]):
            raise ValueError("training terrain grid does not cover the task space")
        # The same decoded batch drives type, level, origin and velocity bounds.
        self.simulator.terrain_types[ids] = terrain_types
        self.simulator.terrain_levels[ids] = terrain_levels
        self.simulator.env_origins[ids] = origins[terrain_levels, terrain_types]
        self.active_task_id[ids] = torch.as_tensor(task_ids, device=self.device, dtype=torch.long)
        self.active_sampler_revision[ids] = int(assignments.sampler_revision)
        self.active_vx_lower[ids] = torch.as_tensor(decoded.vx_lower, device=self.device, dtype=torch.float32)
        self.active_vx_upper[ids] = torch.as_tensor(decoded.vx_upper, device=self.device, dtype=torch.float32)
        self.episode_valid[ids] = True
        self.episode_standstill[ids] = False

    def clear_episode_accumulators(self, env_ids: torch.Tensor) -> None:
        """Start the newly teleported episode after its root state is reset."""
        ids = self._env_ids(env_ids)
        self.episode_return[ids] = 0.0
        self.episode_length[ids] = 0

    def resample_commands_within_active_bin(self, env_ids: torch.Tensor) -> None:
        """Sample only forward velocity; each value stays in its active bin."""
        ids = self._env_ids(env_ids)
        if torch.any(self.active_task_id[ids] < 0):
            raise RuntimeError("cannot sample a UED command before assigning a task")
        lower = self.active_vx_lower[ids]
        upper = self.active_vx_upper[ids]
        self.commands[ids, 0] = lower + (upper - lower) * torch.rand(len(ids), device=self.device)

    def mark_standstill(self, env_ids: torch.Tensor) -> None:
        """Latch standstill for the rest of the episode; LP/ALP skip this return."""
        ids = self._env_ids(env_ids)
        self.episode_valid[ids] = False
        self.episode_standstill[ids] = True

    def apply_standstill_hold(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Re-zero commands for envs already latched as standstill this episode.

        Returns the subset of ``env_ids`` that are *not* on a standstill hold
        (safe to draw a fresh non-zero command for).
        """
        ids = self._env_ids(env_ids)
        hold = self.episode_standstill[ids]
        if torch.any(hold):
            self.commands[ids[hold], :3] = 0.0
        return ids[~hold]

    def _env_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        if torch.any(ids < 0) or torch.any(ids >= self.num_envs):
            raise ValueError("env_ids are out of range")
        return ids
