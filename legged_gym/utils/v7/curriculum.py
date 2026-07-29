"""V7 UED teachers sharing V5's tested LP/Uniform implementation."""
from __future__ import annotations

from legged_gym.utils.ued.episode_curriculum import (
    LPACRLEpisodeCurriculum,
    UniformEpisodeCurriculum,
)


class _V7AdapterMixin:
    def __init__(self, *args, flat: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._v7_flat = bool(flat)

    @property
    def adapter_class(self):
        from .genesis_adapter import GenesisV7FlatAdapter, GenesisV7SemanticAdapter
        return GenesisV7FlatAdapter if self._v7_flat else GenesisV7SemanticAdapter


class V7LPACRLCurriculum(_V7AdapterMixin, LPACRLEpisodeCurriculum):
    """Rolling-completion return LP over either 240 semantic or four flat cells."""


class V7UniformCurriculum(_V7AdapterMixin, UniformEpisodeCurriculum):
    """Uniform V7 control, retaining identical UED lifecycle/provenance."""
