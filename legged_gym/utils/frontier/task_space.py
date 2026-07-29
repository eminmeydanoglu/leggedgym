"""Finite task identity for the V4 terrain-column frontier curriculum."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class FrontierTaskSpec:
    terrain_column: int
    terrain_level: int
    speed_bin: int


@dataclass(frozen=True)
class FrontierTaskBatch:
    terrain_types: np.ndarray
    terrain_levels: np.ndarray
    vx_bins: np.ndarray
    vx_lower: np.ndarray
    vx_upper: np.ndarray


class V4FrontierTaskSpace:
    """The real V4 10x10 starting-terrain bank crossed with four |vx| bins.

    Columns are physical starting-terrain replicas.  Their counts are uneven
    under V4's native proportions, so the curriculum balances six semantic
    families first and chooses one of that family's columns second.
    """

    TERRAIN_FAMILIES = (
        "slope_up",
        "slope_down",
        "rough",
        "stairs_up",
        "stairs_down",
        "discrete_obstacles",
    )
    # Exact mapping produced by Terrain.curiculum() with V4 proportions
    # [0.2, 0.1, 0.25, 0.25, 0.2] on ten columns.  The generator splits the
    # slope and stairs bands by sign; stairs therefore have 3/2 replicas.
    FAMILY_COLUMNS = (
        (0,),
        (1,),
        (2,),
        (3, 4, 5),
        (6, 7),
        (8, 9),
    )
    ABS_SPEED_BIN_EDGES = (0.2, 0.5, 1.0, 1.5, 2.0)
    NUM_LEVELS = 10
    NUM_COLUMNS = 10

    def __init__(self, *, builder_parameters: Mapping[str, object] | None = None) -> None:
        self.terrain_type_names = self.TERRAIN_FAMILIES
        self.velocity_bin_edges = self.ABS_SPEED_BIN_EDGES
        self._builder_parameters = deepcopy(dict(builder_parameters or {}))
        payload = {
            "terrain_families": self.TERRAIN_FAMILIES,
            "family_columns": self.FAMILY_COLUMNS,
            "abs_speed_bin_edges": self.ABS_SPEED_BIN_EDGES,
            "vx_sign_sampling": "episode_rademacher_0.5",
            "num_levels": self.NUM_LEVELS,
            "builder_parameters": self._builder_parameters,
        }
        self._fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self._column_to_family = np.empty(self.NUM_COLUMNS, dtype=np.int64)
        for family, columns in enumerate(self.FAMILY_COLUMNS):
            self._column_to_family[np.asarray(columns, dtype=np.int64)] = family

    @property
    def size(self) -> int:
        return self.NUM_COLUMNS * self.NUM_LEVELS * (len(self.ABS_SPEED_BIN_EDGES) - 1)

    @property
    def num_families(self) -> int:
        return len(self.TERRAIN_FAMILIES)

    @property
    def num_speed_bins(self) -> int:
        return len(self.ABS_SPEED_BIN_EDGES) - 1

    @property
    def builder_parameters(self) -> Mapping[str, object]:
        return deepcopy(self._builder_parameters)

    def family_for_column(self, column: int) -> int:
        if isinstance(column, (bool, np.bool_)) or not isinstance(column, Integral):
            raise ValueError("terrain column must be an integer")
        column = int(column)
        if not 0 <= column < self.NUM_COLUMNS:
            raise ValueError("terrain column is outside the V4 bank")
        return int(self._column_to_family[column])

    def columns_for_family(self, family: int) -> tuple[int, ...]:
        if isinstance(family, (bool, np.bool_)) or not isinstance(family, Integral):
            raise ValueError("terrain family must be an integer")
        family = int(family)
        if not 0 <= family < self.num_families:
            raise ValueError("terrain family is outside the task space")
        return self.FAMILY_COLUMNS[family]

    def encode(self, spec: FrontierTaskSpec) -> int:
        if not isinstance(spec, FrontierTaskSpec):
            raise TypeError("spec must be FrontierTaskSpec")
        column = int(spec.terrain_column)
        level = int(spec.terrain_level)
        speed = int(spec.speed_bin)
        if not 0 <= column < self.NUM_COLUMNS:
            raise ValueError("terrain_column is outside the task space")
        if not 0 <= level < self.NUM_LEVELS:
            raise ValueError("terrain_level is outside the task space")
        if not 0 <= speed < self.num_speed_bins:
            raise ValueError("speed_bin is outside the task space")
        return (speed * self.NUM_LEVELS + level) * self.NUM_COLUMNS + column

    def decode(self, task_id: int) -> FrontierTaskSpec:
        if isinstance(task_id, (bool, np.bool_)) or not isinstance(task_id, Integral):
            raise ValueError("task_id must be an integer")
        task_id = int(task_id)
        if not 0 <= task_id < self.size:
            raise ValueError("task_id is outside the task space")
        speed_level, column = divmod(task_id, self.NUM_COLUMNS)
        speed, level = divmod(speed_level, self.NUM_LEVELS)
        return FrontierTaskSpec(column, level, speed)

    def decode_batch(self, task_ids: np.ndarray) -> FrontierTaskBatch:
        ids = np.asarray(task_ids)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("task_ids must be a one-dimensional integer array")
        if np.any(ids < 0) or np.any(ids >= self.size):
            raise ValueError("task_ids contains an out-of-range identity")
        speed_level, columns = np.divmod(ids.astype(np.int64, copy=False), self.NUM_COLUMNS)
        speed, levels = np.divmod(speed_level, self.NUM_LEVELS)
        edges = np.asarray(self.ABS_SPEED_BIN_EDGES, dtype=np.float64)
        return FrontierTaskBatch(
            terrain_types=columns,
            terrain_levels=levels,
            vx_bins=speed,
            vx_lower=edges[speed],
            vx_upper=edges[speed + 1],
        )

    def fingerprint(self) -> str:
        return self._fingerprint
