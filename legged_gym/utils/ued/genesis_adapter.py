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

    # Terrain curricula require the static grid supplied by Genesis.  The V7
    # flat-prior adapter deliberately overrides this because it has a velocity
    # task space only and must work with ``terrain.mesh_type = 'plane'``.
    requires_terrain_origins = True

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
        if self.requires_terrain_origins and (
            not getattr(simulator, "custom_origins", False)
            or not hasattr(simulator, "_terrain_origins")
        ):
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
        # Birth label, not a post-hoc veto: an env is set standstill (True) or
        # moving (False) once, at assignment.  Standstill episodes are a reserved
        # mixture bucket -- they run beside the LP task space, hold a zero command
        # for the whole episode, and their return updates the standstill bucket
        # instead of any curriculum cell.  PPO still learns from them.
        self.episode_standstill = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # Reserved standstill bucket metrics (never part of learning progress).
        self._standstill_episode_count = 0
        self._standstill_return_sum = 0.0
        self._standstill_length_sum = 0

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
        completion_global_control_steps: int = 0,
        timed_out: torch.Tensor,
    ) -> EpisodeOutcomeBatch:
        """Copy moving-task outcomes for *old* assignments before replacement.

        The caller passes only the moving envs among the completing batch;
        reserved-standstill envs are collected separately by
        :meth:`record_standstill_outcomes`.
        """
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
            # Shared control-step clock at the lifecycle completion boundary;
            # every member of one reset batch completes at the same tick.
            completion_global_control_steps=int(completion_global_control_steps),
            episodic_returns=self.episode_return[ids].detach().cpu().numpy().astype(np.float64, copy=True),
            episode_lengths=self.episode_length[ids].detach().cpu().numpy().astype(np.int64, copy=True),
            terminal_reasons=reasons,
        )

    def assign(self, env_ids: torch.Tensor, assignments: TaskAssignmentBatch) -> None:
        """Start one moving-task episode per environment by teleport.

        This deliberately changes only tensors already owned by the simulator;
        it never calls terrain construction or changes the static heightfield.
        """
        ids = self._teleport(env_ids, assignments)
        self.episode_standstill[ids] = False

    def assign_standstill(self, env_ids: torch.Tensor, placements: TaskAssignmentBatch) -> None:
        """Start one reserved-standstill episode per environment.

        The placement cells (drawn LP-weighted, see
        ``EpisodeCurriculum.draw_placements``) decide only *where* the robot
        stands: the env is teleported there and its command is held at zero.
        The outcome is never attributed to that cell -- the episode is born
        standstill and reported to the standstill bucket, not the curriculum.
        """
        ids = self._teleport(env_ids, placements)
        self.commands[ids, :3] = 0.0
        self.episode_standstill[ids] = True

    def _teleport(self, env_ids: torch.Tensor, batch: TaskAssignmentBatch) -> torch.Tensor:
        """Shared atomic teleport for both moving and standstill placements."""
        ids = self._env_ids(env_ids)
        task_ids = np.asarray(batch.task_ids)
        if task_ids.shape != (len(ids),) or not np.issubdtype(task_ids.dtype, np.integer):
            raise ValueError("placement task_ids must match env_ids")
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
        self.active_sampler_revision[ids] = int(batch.sampler_revision)
        self.active_vx_lower[ids] = torch.as_tensor(decoded.vx_lower, device=self.device, dtype=torch.float32)
        self.active_vx_upper[ids] = torch.as_tensor(decoded.vx_upper, device=self.device, dtype=torch.float32)
        return ids

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

    def record_standstill_outcomes(self, env_ids: torch.Tensor) -> None:
        """Fold completed standstill episodes into the reserved-bucket metrics.

        This is the standstill counterpart of :meth:`collect_outcomes`; its
        returns never touch learning progress.
        """
        ids = self._env_ids(env_ids)
        if not len(ids):
            return
        self._standstill_episode_count += int(len(ids))
        self._standstill_return_sum += float(self.episode_return[ids].sum())
        self._standstill_length_sum += int(self.episode_length[ids].sum())

    def standstill_diagnostics(self) -> Mapping[str, float]:
        """Cumulative reserved-bucket metrics, safe to log beside the curriculum."""
        count = self._standstill_episode_count
        return {
            "standstill_episode_count": float(count),
            "standstill_mean_return": self._standstill_return_sum / count if count else 0.0,
            "standstill_mean_length": self._standstill_length_sum / count if count else 0.0,
        }

    def apply_standstill_hold(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Keep reserved-standstill envs at a zero command; return the movers.

        Standstill is a whole-episode birth label, so every command (re)sample
        must re-zero these envs.  The returned subset of ``env_ids`` is the
        moving envs, safe to draw a fresh non-zero command for.
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
