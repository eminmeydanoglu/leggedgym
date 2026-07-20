"""RMA adapter: student z_s interventions for the use-test; teacher kept separate.

Primary scientific path for "does the student use mass latent?":
  normal  = actor(obs, z_s_own)
  control = actor(obs, z_s_within_mass_donor)   # same-mass student latent
  wrong   = actor(obs, z_s_cross_mass_donor)    # opposite-mass student latent

Teacher privilege correct/wrong is exposed as optional diagnostics only
(act_teacher_correct / act_teacher_wrong) — not the primary Δuse path.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from legged_gym.scripts.eval.probe_physics_logic import (
    MASS_GRID_KG,
    MASS_NORM_RANGE,
    build_rma_wrong_privilege,
    swap_implicit_latent,
)


def history_encoder_input(ac: Any, history: torch.Tensor) -> torch.Tensor:
    if getattr(ac, "history_encoder_type", "MLP") == "TCN":
        return history.unsqueeze(1)
    return history


def student_latent(ac: Any, history: torch.Tensor) -> torch.Tensor:
    return ac.history_encoder(history_encoder_input(ac, history))


def actor_from_student_z(
    ac: Any, obs: torch.Tensor, z_s: torch.Tensor
) -> torch.Tensor:
    return ac.actor(torch.cat([obs, z_s], dim=-1))


class RMAAdapter:
    name = "RMA"
    has_teacher = True
    # Primary use-test is student latent swap (same contract as DW/HIM).
    use_test_kind = "student_latent_swap"

    def extract_latent(
        self, actor_critic: Any, state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        obs = state["obs"]
        priv = state["priv_obs"]
        history = state["history"]
        z_t = actor_critic.privilege_encoder(priv)
        z_s = student_latent(actor_critic, history)
        return {
            "latent": z_s,
            "z_s": z_s,
            "z_t": z_t,
            "obs": obs,
            "priv_obs": priv,
            "history": history,
        }

    def act_normal(
        self, actor_critic: Any, state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Student deploy path: actor(obs, history_encoder(history))."""
        return actor_critic.act_student(state["obs"], state["history"])

    def act_control(
        self,
        actor_critic: Any,
        state: Dict[str, torch.Tensor],
        donors: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Within-mass student latent swap (control for cross-mass)."""
        if donors is None or "within_idx" not in donors:
            raise KeyError("RMA act_control needs donors['within_idx'] (student z_s swap)")
        lat = self.extract_latent(actor_critic, state)
        bank = donors.get("latent_bank", lat["z_s"])
        swapped = swap_implicit_latent(lat["z_s"], bank, donors["within_idx"])
        return actor_from_student_z(actor_critic, state["obs"], swapped)

    def act_wrong(
        self,
        actor_critic: Any,
        state: Dict[str, torch.Tensor],
        donors: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Cross-mass student latent swap — primary student-use intervention."""
        if donors is None or "cross_idx" not in donors:
            raise KeyError("RMA act_wrong needs donors['cross_idx'] (student z_s swap)")
        lat = self.extract_latent(actor_critic, state)
        bank = donors.get("latent_bank", lat["z_s"])
        swapped = swap_implicit_latent(lat["z_s"], bank, donors["cross_idx"])
        return actor_from_student_z(actor_critic, state["obs"], swapped)

    # ------------------------------------------------------------------
    # Optional teacher privilege sensitivity (NOT primary Δuse)
    # ------------------------------------------------------------------

    def act_teacher_correct(
        self, actor_critic: Any, state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Diagnostic: teacher with true privilege (true mass + true vel)."""
        return actor_critic.act_teacher(state["obs"], state["priv_obs"])

    def act_teacher_wrong(
        self,
        actor_critic: Any,
        state: Dict[str, torch.Tensor],
        *,
        mass_grid: Sequence[float] = MASS_GRID_KG,
        mass_range: Tuple[float, float] = MASS_NORM_RANGE,
    ) -> torch.Tensor:
        """Diagnostic: teacher with opposite-end mass, true velocity.

        Label as teacher_privilege_sensitivity — not student-use evidence.
        """
        real_mass = state.get("real_mass_raw")
        if real_mass is None:
            raise KeyError("act_teacher_wrong needs state['real_mass_raw']")
        wrong_priv = build_rma_wrong_privilege(
            state["priv_obs"],
            real_mass,
            mass_grid=mass_grid,
            mass_range=mass_range,
            mass_slot=1,
        )
        z_wrong = actor_critic.privilege_encoder(wrong_priv)
        return actor_from_student_z(actor_critic, state["obs"], z_wrong)

    def decode_features(self, latents: Dict[str, torch.Tensor]) -> torch.Tensor:
        return latents["z_s"]

    def teacher_features(self, latents: Dict[str, torch.Tensor]):
        return latents.get("z_t")
