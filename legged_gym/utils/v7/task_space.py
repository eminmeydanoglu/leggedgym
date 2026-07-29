"""Stable V7 task identities.

V6 owns a physical-column task space because its frontier state is stored at
the semantic-family level while its assignments are concrete terrain tiles.
V7 instead makes the semantic cell the persisted LP identity and samples a
physical V4 column only when that identity is applied to Genesis.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from typing import Mapping

import numpy as np

from legged_gym.utils.frontier.task_space import V4FrontierTaskSpace


@dataclass(frozen=True)
class V7TerrainTaskSpec:
    terrain_family: int
    terrain_level: int
    vx_bin: int


@dataclass(frozen=True)
class V7TerrainTaskBatch:
    terrain_families: np.ndarray
    terrain_levels: np.ndarray
    vx_bins: np.ndarray
    vx_lower: np.ndarray
    vx_upper: np.ndarray


@dataclass(frozen=True)
class V7VelocityTaskSpec:
    vx_bin: int


@dataclass(frozen=True)
class V7VelocityTaskBatch:
    vx_bins: np.ndarray
    vx_lower: np.ndarray
    vx_upper: np.ndarray


class _BaseV7TaskSpace:
    VELOCITY_BIN_EDGES = V4FrontierTaskSpace.ABS_SPEED_BIN_EDGES

    @staticmethod
    def _integer(value: object, *, name: str) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer")
        return int(value)

    @classmethod
    def _validate_ids(cls, task_ids: np.ndarray, size: int) -> np.ndarray:
        ids = np.asarray(task_ids)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("task_ids must be a one-dimensional integer array")
        if np.any(ids < 0) or np.any(ids >= size):
            raise ValueError("task_ids contains an out-of-range identity")
        return ids.astype(np.int64, copy=False)


class V7SemanticTaskSpace(_BaseV7TaskSpace):
    """6 semantic terrain families × 4 |vx| bins × 10 terrain levels."""

    TERRAIN_FAMILIES = V4FrontierTaskSpace.TERRAIN_FAMILIES
    FAMILY_COLUMNS = V4FrontierTaskSpace.FAMILY_COLUMNS
    NUM_LEVELS = V4FrontierTaskSpace.NUM_LEVELS

    def __init__(self, *, builder_parameters: Mapping[str, object] | None = None) -> None:
        self.terrain_type_names = self.TERRAIN_FAMILIES
        self.velocity_bin_edges = self.VELOCITY_BIN_EDGES
        self._builder_parameters = deepcopy(dict(builder_parameters or {}))
        payload = {
            "task_identity": "v7_semantic_family_abs_vx_level",
            "terrain_families": self.TERRAIN_FAMILIES,
            "family_columns": self.FAMILY_COLUMNS,
            "velocity_bin_edges": self.VELOCITY_BIN_EDGES,
            "num_levels": self.NUM_LEVELS,
            "vx_sign_sampling": "episode_rademacher_0.5",
            "builder_parameters": self._builder_parameters,
        }
        self._fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        # Keep the exact V6 physical-bank identity available for paired-run
        # provenance. ``fingerprint()`` below intentionally remains distinct:
        # it identifies V7's semantic curriculum state, not V6's 400 physical
        # assignment IDs.
        self._terrain_bank_fingerprint = V4FrontierTaskSpace(
            builder_parameters=self._builder_parameters
        ).fingerprint()

    @property
    def size(self) -> int:
        return self.num_families * self.num_speed_bins * self.NUM_LEVELS

    @property
    def num_families(self) -> int:
        return len(self.TERRAIN_FAMILIES)

    @property
    def num_speed_bins(self) -> int:
        return len(self.VELOCITY_BIN_EDGES) - 1

    @property
    def builder_parameters(self) -> Mapping[str, object]:
        return deepcopy(self._builder_parameters)

    def columns_for_family(self, family: int) -> tuple[int, ...]:
        family = self._integer(family, name="terrain_family")
        if not 0 <= family < self.num_families:
            raise ValueError("terrain_family is outside the task space")
        return self.FAMILY_COLUMNS[family]

    def encode(self, spec: V7TerrainTaskSpec) -> int:
        if not isinstance(spec, V7TerrainTaskSpec):
            raise TypeError("spec must be a V7TerrainTaskSpec")
        family = self._integer(spec.terrain_family, name="terrain_family")
        level = self._integer(spec.terrain_level, name="terrain_level")
        speed = self._integer(spec.vx_bin, name="vx_bin")
        if not 0 <= family < self.num_families:
            raise ValueError("terrain_family is outside the task space")
        if not 0 <= level < self.NUM_LEVELS:
            raise ValueError("terrain_level is outside the task space")
        if not 0 <= speed < self.num_speed_bins:
            raise ValueError("vx_bin is outside the task space")
        return (family * self.num_speed_bins + speed) * self.NUM_LEVELS + level

    def decode(self, task_id: int) -> V7TerrainTaskSpec:
        task_id = self._integer(task_id, name="task_id")
        if not 0 <= task_id < self.size:
            raise ValueError("task_id is outside the task space")
        family_speed, level = divmod(task_id, self.NUM_LEVELS)
        family, speed = divmod(family_speed, self.num_speed_bins)
        return V7TerrainTaskSpec(family, level, speed)

    def decode_batch(self, task_ids: np.ndarray) -> V7TerrainTaskBatch:
        ids = self._validate_ids(task_ids, self.size)
        family_speed, levels = np.divmod(ids, self.NUM_LEVELS)
        families, speed = np.divmod(family_speed, self.num_speed_bins)
        edges = np.asarray(self.VELOCITY_BIN_EDGES, dtype=np.float64)
        return V7TerrainTaskBatch(families, levels, speed, edges[speed], edges[speed + 1])

    def fingerprint(self) -> str:
        return self._fingerprint

    def terrain_bank_fingerprint(self) -> str:
        """Fingerprint of the shared V6 physical 10×10 terrain bank."""
        return self._terrain_bank_fingerprint


class V7VelocityTaskSpace(_BaseV7TaskSpace):
    """Flat-prior support: the four moving |vx| bins and no terrain identity."""

    def __init__(self, *, builder_parameters: Mapping[str, object] | None = None) -> None:
        self.velocity_bin_edges = self.VELOCITY_BIN_EDGES
        self._builder_parameters = deepcopy(dict(builder_parameters or {}))
        payload = {
            "task_identity": "v7_flat_abs_vx",
            "velocity_bin_edges": self.VELOCITY_BIN_EDGES,
            "vx_sign_sampling": "episode_rademacher_0.5",
            "builder_parameters": self._builder_parameters,
        }
        self._fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def size(self) -> int:
        return len(self.VELOCITY_BIN_EDGES) - 1

    @property
    def builder_parameters(self) -> Mapping[str, object]:
        return deepcopy(self._builder_parameters)

    def encode(self, spec: V7VelocityTaskSpec) -> int:
        if not isinstance(spec, V7VelocityTaskSpec):
            raise TypeError("spec must be a V7VelocityTaskSpec")
        vx_bin = self._integer(spec.vx_bin, name="vx_bin")
        if not 0 <= vx_bin < self.size:
            raise ValueError("vx_bin is outside the task space")
        return vx_bin

    def decode(self, task_id: int) -> V7VelocityTaskSpec:
        task_id = self._integer(task_id, name="task_id")
        if not 0 <= task_id < self.size:
            raise ValueError("task_id is outside the task space")
        return V7VelocityTaskSpec(task_id)

    def decode_batch(self, task_ids: np.ndarray) -> V7VelocityTaskBatch:
        ids = self._validate_ids(task_ids, self.size)
        edges = np.asarray(self.VELOCITY_BIN_EDGES, dtype=np.float64)
        return V7VelocityTaskBatch(ids, edges[ids], edges[ids + 1])

    def fingerprint(self) -> str:
        return self._fingerprint
