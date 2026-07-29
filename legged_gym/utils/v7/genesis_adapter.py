"""Genesis boundaries for V7 semantic-terrain and flat velocity curricula."""
from __future__ import annotations

import numpy as np
import torch

from legged_gym.utils.frontier.genesis_adapter import GenesisFrontierAdapter
from legged_gym.utils.ued.genesis_adapter import GenesisUEDAdapter


class GenesisV7SemanticAdapter(GenesisFrontierAdapter):
    """Apply a semantic V7 task and sample its physical V6 column at reset."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.active_physical_column = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )

    def _teleport(self, env_ids: torch.Tensor, batch) -> torch.Tensor:
        ids = self._env_ids(env_ids)
        task_ids = np.asarray(batch.task_ids)
        if task_ids.shape != (len(ids),) or not np.issubdtype(task_ids.dtype, np.integer):
            raise ValueError("placement task_ids must match env_ids")
        decoded = self.task_space.decode_batch(task_ids)
        columns = np.empty(len(ids), dtype=np.int64)
        # Physical replica sampling is intentionally outside curriculum state:
        # LP observes family × |vx| × level, never one replica's layout noise.
        for family in np.unique(decoded.terrain_families):
            positions = np.flatnonzero(decoded.terrain_families == family)
            family_columns = self.task_space.columns_for_family(int(family))
            choices = torch.randint(
                len(family_columns), (len(positions),), device=self.device
            ).cpu().numpy()
            columns[positions] = np.asarray(family_columns, dtype=np.int64)[choices]
        levels = torch.as_tensor(decoded.terrain_levels, device=self.device, dtype=torch.long)
        physical_columns = torch.as_tensor(columns, device=self.device, dtype=torch.long)
        origins = self.simulator._terrain_origins
        if torch.any(levels >= origins.shape[0]) or torch.any(physical_columns >= origins.shape[1]):
            raise ValueError("training terrain grid does not cover the V7 task space")
        self.simulator.terrain_types[ids] = physical_columns
        self.simulator.terrain_levels[ids] = levels
        self.simulator.env_origins[ids] = origins[levels, physical_columns]
        self.active_task_id[ids] = torch.as_tensor(task_ids, device=self.device, dtype=torch.long)
        self.active_sampler_revision[ids] = int(batch.sampler_revision)
        self.active_vx_lower[ids] = torch.as_tensor(decoded.vx_lower, device=self.device, dtype=torch.float32)
        self.active_vx_upper[ids] = torch.as_tensor(decoded.vx_upper, device=self.device, dtype=torch.float32)
        self.active_physical_column[ids] = physical_columns
        return ids


class GenesisV7FlatAdapter(GenesisUEDAdapter):
    """Velocity-only UED adapter that never touches terrain origins or teleports."""

    requires_terrain_origins = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.active_vx_sign = torch.ones(
            self.num_envs, dtype=torch.float32, device=self.device
        )

    def _teleport(self, env_ids: torch.Tensor, batch) -> torch.Tensor:
        ids = self._env_ids(env_ids)
        task_ids = np.asarray(batch.task_ids)
        if task_ids.shape != (len(ids),) or not np.issubdtype(task_ids.dtype, np.integer):
            raise ValueError("placement task_ids must match env_ids")
        decoded = self.task_space.decode_batch(task_ids)
        self.active_task_id[ids] = torch.as_tensor(task_ids, device=self.device, dtype=torch.long)
        self.active_sampler_revision[ids] = int(batch.sampler_revision)
        self.active_vx_lower[ids] = torch.as_tensor(decoded.vx_lower, device=self.device, dtype=torch.float32)
        self.active_vx_upper[ids] = torch.as_tensor(decoded.vx_upper, device=self.device, dtype=torch.float32)
        return ids

    def assign(self, env_ids: torch.Tensor, assignments) -> None:
        ids = self._teleport(env_ids, assignments)
        self.episode_standstill[ids] = False
        self.active_vx_sign[ids] = torch.where(
            torch.rand(len(ids), device=self.device) < 0.5,
            -torch.ones(len(ids), device=self.device),
            torch.ones(len(ids), device=self.device),
        )

    def resample_commands_within_active_bin(self, env_ids: torch.Tensor) -> None:
        ids = self._env_ids(env_ids)
        if torch.any(self.active_task_id[ids] < 0):
            raise RuntimeError("cannot sample a V7-flat command before assigning a task")
        magnitude = self.active_vx_lower[ids] + (
            self.active_vx_upper[ids] - self.active_vx_lower[ids]
        ) * torch.rand(len(ids), device=self.device)
        self.commands[ids, 0] = self.active_vx_sign[ids] * magnitude
