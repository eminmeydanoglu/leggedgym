"""DreamWaQ adapter: posterior-mean latent_mu + vel_mu; swap only latent_mu."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from legged_gym.scripts.eval.probe_physics_logic import swap_implicit_latent


def encode_posterior_mean(
    actor_critic: Any, history: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deploy path: VAE.encode → (latent_mu, vel_mu). No stochastic sample."""
    latent_mu, _, vel_mu, _ = actor_critic.vae.encode(history)
    return latent_mu, vel_mu


def actor_from_parts(
    actor_critic: Any,
    obs: torch.Tensor,
    latent_mu: torch.Tensor,
    vel_mu: torch.Tensor,
) -> torch.Tensor:
    mean_out = torch.cat([latent_mu, vel_mu], dim=-1)
    return actor_critic.actor(torch.cat([obs, mean_out], dim=-1))


class DreamWaQAdapter:
    name = "DreamWaQ"
    has_teacher = False
    # Opposite-mass latent population sensitivity (not pure mass-only causal).
    use_test_kind = "student_latent_swap"

    def extract_latent(
        self, actor_critic: Any, state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        history = state["history"]
        latent_mu, vel_mu = encode_posterior_mean(actor_critic, history)
        return {
            "latent": latent_mu,
            "latent_mu": latent_mu,
            "vel_mu": vel_mu,
            "vel": vel_mu,
            "obs": state["obs"],
            "history": history,
        }

    def act_normal(
        self, actor_critic: Any, state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # Match deploy: act_inference uses posterior means.
        return actor_critic.act_inference(state["obs"], state["history"])

    def act_control(
        self,
        actor_critic: Any,
        state: Dict[str, torch.Tensor],
        donors: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Within-mass swap: replace latent_mu only; keep own vel_mu."""
        if donors is None or "within_idx" not in donors:
            raise KeyError("DreamWaQ act_control needs donors['within_idx']")
        latents = self.extract_latent(actor_critic, state)
        # Prefer precomputed donor bank if provided (same step, all envs).
        bank = donors.get("latent_bank", latents["latent_mu"])
        swapped = swap_implicit_latent(
            latents["latent_mu"], bank, donors["within_idx"]
        )
        return actor_from_parts(
            actor_critic, state["obs"], swapped, latents["vel_mu"]
        )

    def act_wrong(
        self,
        actor_critic: Any,
        state: Dict[str, torch.Tensor],
        donors: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Cross-mass swap: opposite-mass donor latent_mu; own vel_mu."""
        if donors is None or "cross_idx" not in donors:
            raise KeyError("DreamWaQ act_wrong needs donors['cross_idx']")
        latents = self.extract_latent(actor_critic, state)
        bank = donors.get("latent_bank", latents["latent_mu"])
        swapped = swap_implicit_latent(
            latents["latent_mu"], bank, donors["cross_idx"]
        )
        return actor_from_parts(
            actor_critic, state["obs"], swapped, latents["vel_mu"]
        )

    def decode_features(self, latents: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Mass decoder must NOT see vel_mu (gait/velocity confound).
        return latents["latent_mu"]

    def teacher_features(self, latents: Dict[str, torch.Tensor]):
        return None
