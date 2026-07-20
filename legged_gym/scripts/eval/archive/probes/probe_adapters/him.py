"""HIM adapter: vel_hat + normalized implicit z; swap only z."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from legged_gym.scripts.eval.probe_physics_logic import swap_implicit_latent


def encode_him(
    actor_critic: Any, obs_history: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(vel_hat, z) with L2-normalized z — same as estimator.get_latent/encode."""
    vel, z = actor_critic.estimator.encode(obs_history)
    return vel, z


def actor_from_parts(
    actor_critic: Any,
    obs_history: torch.Tensor,
    vel: torch.Tensor,
    latent: torch.Tensor,
) -> torch.Tensor:
    num_one = actor_critic.num_one_step_obs
    obs = obs_history[:, :num_one]
    return actor_critic.actor(torch.cat([obs, vel, latent], dim=-1))


class HIMAdapter:
    name = "HIM"
    has_teacher = False
    use_test_kind = "student_latent_swap"

    def extract_latent(
        self, actor_critic: Any, state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # HIM env observation IS stacked history.
        obs_history = state.get("obs_history", state.get("obs"))
        vel, z = encode_him(actor_critic, obs_history)
        return {
            "latent": z,
            "z": z,
            "vel_hat": vel,
            "vel": vel,
            "obs_history": obs_history,
        }

    def act_normal(
        self, actor_critic: Any, state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        obs_history = state.get("obs_history", state.get("obs"))
        return actor_critic.act_inference(obs_history)

    def act_control(
        self,
        actor_critic: Any,
        state: Dict[str, torch.Tensor],
        donors: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if donors is None or "within_idx" not in donors:
            raise KeyError("HIM act_control needs donors['within_idx']")
        latents = self.extract_latent(actor_critic, state)
        bank = donors.get("latent_bank", latents["z"])
        swapped = swap_implicit_latent(latents["z"], bank, donors["within_idx"])
        return actor_from_parts(
            actor_critic, latents["obs_history"], latents["vel_hat"], swapped
        )

    def act_wrong(
        self,
        actor_critic: Any,
        state: Dict[str, torch.Tensor],
        donors: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if donors is None or "cross_idx" not in donors:
            raise KeyError("HIM act_wrong needs donors['cross_idx']")
        latents = self.extract_latent(actor_critic, state)
        bank = donors.get("latent_bank", latents["z"])
        swapped = swap_implicit_latent(latents["z"], bank, donors["cross_idx"])
        return actor_from_parts(
            actor_critic, latents["obs_history"], latents["vel_hat"], swapped
        )

    def decode_features(self, latents: Dict[str, torch.Tensor]) -> torch.Tensor:
        return latents["z"]

    def teacher_features(self, latents: Dict[str, torch.Tensor]):
        return None
