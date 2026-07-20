"""Adapter protocol for RMA / DreamWaQ / HIM physics-use probe."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch


class ProbeAdapter:
    """Thin interface between the shared runner and a method's latent/act paths.

    Subclasses implement pure tensor operations on actor_critic + env state so
    unit tests can exercise them without a live Genesis scene.
    """

    name: str = "base"
    # RMA has a privileged teacher; DreamWaQ/HIM use within/cross latent swaps.
    has_teacher: bool = False

    def extract_latent(
        self, actor_critic: Any, state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Return method-specific latent tensors for logging/decode.

        Common keys:
          - latent: implicit physics latent used by the actor (decode target)
          - vel: optional velocity estimate (NOT given to mass decoder)
        RMA also returns z_s, z_t.
        """
        raise NotImplementedError

    def act_normal(
        self, actor_critic: Any, state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        raise NotImplementedError

    def act_control(
        self,
        actor_critic: Any,
        state: Dict[str, torch.Tensor],
        donors: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Same-physics control: teacher_correct (RMA) or within-mass swap."""
        raise NotImplementedError

    def act_wrong(
        self,
        actor_critic: Any,
        state: Dict[str, torch.Tensor],
        donors: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Wrong-physics: teacher_wrong (RMA) or cross-mass swap."""
        raise NotImplementedError

    def decode_features(
        self, latents: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Feature matrix for mass decoder (implicit latent only)."""
        if "latent" in latents:
            return latents["latent"]
        raise KeyError("latents missing 'latent'")

    def teacher_features(
        self, latents: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Optional teacher latent for RMA teacher mass R²."""
        return latents.get("z_t")


def get_adapter(method: str) -> ProbeAdapter:
    method = method.lower().replace("-", "_")
    if method in ("rma", "go2_v3_rma", "v3_rma"):
        from .rma import RMAAdapter
        return RMAAdapter()
    if method in ("dreamwaq", "go2_v3_dreamwaq", "v3_dreamwaq", "dw"):
        from .dreamwaq import DreamWaQAdapter
        return DreamWaQAdapter()
    if method in ("him", "go2_v3_him_fixed", "him_fixed", "v3_him"):
        from .him import HIMAdapter
        return HIMAdapter()
    raise ValueError(f"unknown probe method: {method!r}")
