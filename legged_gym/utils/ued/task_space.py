"""Immutable finite task identity for the UED teacher.

This module deliberately knows only the discrete experiment support.  Applying a
task to a simulator is the adapter's responsibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
from numbers import Integral
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class TaskSpec:
    """One moving-task coordinate in the frozen V5 support."""

    terrain_type: int
    terrain_level: int
    vx_bin: int


@dataclass(frozen=True)
class TaskSpecBatch:
    """Vectorized decoded task payload used by a simulator adapter."""

    terrain_types: np.ndarray
    terrain_levels: np.ndarray
    vx_bins: np.ndarray
    vx_lower: np.ndarray
    vx_upper: np.ndarray


class TaskSpace:
    """The canonical 84-cell moving-task grid with stable integer identities."""

    TERRAIN_TYPE_NAMES = (
        "stairs_up", "stairs_down", "slope_up", "slope_down", "rough", "flat",
    )
    VELOCITY_BIN_EDGES = (0.0, 0.5, 1.0, 1.5, 2.0)

    def __init__(
        self,
        *,
        terrain_type_names: tuple[str, ...] = TERRAIN_TYPE_NAMES,
        velocity_bin_edges: tuple[float, ...] = VELOCITY_BIN_EDGES,
        builder_parameters: Mapping[str, object] | None = None,
    ) -> None:
        if (
            len(terrain_type_names) != 6
            or len(set(terrain_type_names)) != 6
            or not all(isinstance(name, str) and name for name in terrain_type_names)
        ):
            raise ValueError("TaskSpace requires six distinct terrain types")
        try:
            velocity_bin_edges = tuple(float(value) for value in velocity_bin_edges)
        except (TypeError, ValueError) as exc:
            raise ValueError("velocity_bin_edges must be numeric") from exc
        if (
            len(velocity_bin_edges) != 5
            or not all(np.isfinite(velocity_bin_edges))
            or any(b >= a for b, a in zip(velocity_bin_edges, velocity_bin_edges[1:]))
        ):
            raise ValueError("velocity_bin_edges must be five strictly increasing values")
        self.terrain_type_names = tuple(terrain_type_names)
        self.velocity_bin_edges = velocity_bin_edges
        # Builder parameters participate in the persisted identity.  Keep an
        # internal snapshot so a caller mutating its input after construction
        # cannot silently change the identity of an existing task space.
        self._builder_parameters = deepcopy(dict(builder_parameters or {}))
        self._fingerprint_payload = {
            "terrain_type_names": self.terrain_type_names,
            "velocity_bin_edges": self.velocity_bin_edges,
            "builder_parameters": self._builder_parameters,
        }
        try:
            self._fingerprint = hashlib.sha256(
                json.dumps(
                    self._fingerprint_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
            ).hexdigest()
        except (TypeError, ValueError) as exc:
            raise ValueError("builder_parameters must be finite JSON-compatible data") from exc
        self._terrain_configs = tuple(
            [(terrain_type, level) for terrain_type in range(5) for level in range(4)]
            + [(5, 0)]
        )
        self._config_to_index = {spec: index for index, spec in enumerate(self._terrain_configs)}

    @property
    def size(self) -> int:
        return len(self._terrain_configs) * (len(self.velocity_bin_edges) - 1)

    @property
    def builder_parameters(self) -> Mapping[str, object]:
        """A defensive copy of the deterministic builder description."""
        return deepcopy(self._builder_parameters)

    @staticmethod
    def _integer(value: object, *, name: str) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer")
        return int(value)

    def encode(self, spec: TaskSpec) -> int:
        if not isinstance(spec, TaskSpec):
            raise TypeError("spec must be a TaskSpec")
        terrain_type = self._integer(spec.terrain_type, name="terrain_type")
        terrain_level = self._integer(spec.terrain_level, name="terrain_level")
        vx_bin = self._integer(spec.vx_bin, name="vx_bin")
        if not 0 <= vx_bin < len(self.velocity_bin_edges) - 1:
            raise ValueError("vx_bin is outside the task space")
        try:
            terrain_index = self._config_to_index[(terrain_type, terrain_level)]
        except KeyError as exc:
            raise ValueError("terrain type/level is outside the task space") from exc
        return vx_bin * len(self._terrain_configs) + terrain_index

    def decode(self, task_id: int) -> TaskSpec:
        task_id = self._integer(task_id, name="task_id")
        if not 0 <= task_id < self.size:
            raise ValueError("task_id is outside the task space")
        vx_bin, terrain_index = divmod(task_id, len(self._terrain_configs))
        terrain_type, terrain_level = self._terrain_configs[terrain_index]
        return TaskSpec(terrain_type=terrain_type, terrain_level=terrain_level, vx_bin=vx_bin)

    def decode_batch(self, task_ids: np.ndarray) -> TaskSpecBatch:
        ids = np.asarray(task_ids)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("task_ids must be a one-dimensional integer array")
        if np.any(ids < 0) or np.any(ids >= self.size):
            raise ValueError("task_ids contains an out-of-range identity")
        ids = ids.astype(np.int64, copy=False)
        vx_bins, terrain_indices = np.divmod(ids, len(self._terrain_configs))
        terrain = np.asarray(self._terrain_configs, dtype=np.int64)[terrain_indices]
        lower = np.asarray(self.velocity_bin_edges, dtype=np.float64)[vx_bins]
        upper = np.asarray(self.velocity_bin_edges[1:], dtype=np.float64)[vx_bins]
        return TaskSpecBatch(terrain[:, 0], terrain[:, 1], vx_bins, lower, upper)

    def fingerprint(self) -> str:
        return self._fingerprint
