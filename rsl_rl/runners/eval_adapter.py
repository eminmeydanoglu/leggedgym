"""Deploy-time inference adapters shared by periodic eval and the V2 campaign.

Benchmark methods disagree only in *how the deployable policy is fed*: MLP/P5/HIM
consume a single observation tensor, RMA/DreamWaQ need ``(obs, history)``, SysID
needs the estimator feature history. The *metrics* (tracking / fall) are computed
from ``env.commands`` and ``env.simulator.*`` and are identical across methods, so
they stay in the caller (``run_indist_eval`` / ``campaign.rollout``).

An adapter therefore abstracts exactly three things and nothing else::

    state          = adapter.reset(env)          # env.reset() + first policy input(s)
    actions        = adapter.act(state)          # deterministic (mean) action
    state, rew, dn = adapter.step(env, actions)  # env.step() + next policy input(s)

``state`` is opaque to the caller (a tensor or a tuple of tensors). Every adapter
returns the *deterministic* action (mean, no exploration noise) so eval numbers are
comparable across checkpoints. Runners build the adapter that matches their method
(see ``OnPolicyRunner.get_eval_adapter`` and the adaptive overrides).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Sequence

import torch


def load_deploy_components(
    actor_critic: torch.nn.Module,
    model_state_dict: Dict[str, torch.Tensor],
    prefixes: Sequence[str],
    source: str = "<checkpoint>",
) -> None:
    """Strictly load exactly ``prefixes`` sub-modules from a checkpoint state-dict.

    Each prefix names a top-level sub-module of ``actor_critic`` (``"actor."``,
    ``"history_encoder."``, ``"vae."``, ``"estimator."``). For every prefix we slice
    the matching ``prefix + <param>`` tensors out of ``model_state_dict`` and load
    them into ``getattr(actor_critic, prefix[:-1])`` with ``strict=True``. An empty
    slice or any missing/unexpected key raises -- so a randomly-initialised
    estimator/encoder can never be silently deployed (Eval V2 blocker #2/#3).
    """
    for prefix in prefixes:
        name = prefix[:-1] if prefix.endswith(".") else prefix
        module = getattr(actor_critic, name, None)
        if module is None:
            raise RuntimeError(f"{source}: actor_critic has no deploy component '{name}'")
        sub_state = {k[len(prefix):]: v for k, v in model_state_dict.items()
                     if k.startswith(prefix)}
        if not sub_state:
            raise RuntimeError(f"{source}: checkpoint contains no '{prefix}*' tensors")
        missing, unexpected = module.load_state_dict(sub_state, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"{source}: '{name}' mismatch missing={missing} unexpected={unexpected}")


class EvalAdapter:
    """Base adapter: single-tensor obs contract (MLP / P5 / HIM).

    ``policy`` is the deterministic inference callable (``act_inference`` /
    ``act_student``) that maps the deployable input(s) to a mean action.
    """

    def __init__(self, policy: Callable[..., torch.Tensor], device: torch.device):
        self.policy = policy
        self.device = device

    def reset(self, env: Any) -> Any:
        env.reset()
        return env.get_observations().to(self.device)

    def act(self, state: Any) -> torch.Tensor:
        return self.policy(state.detach())

    def step(self, env: Any, actions: torch.Tensor):
        obs, _, rewards, dones, _ = env.step(actions)
        return obs.to(self.device), rewards, dones


class HistoryObsAdapter(EvalAdapter):
    """``(obs, history)`` contract for RMA (``act_student``) and DreamWaQ.

    Both envs return ``get_observations() -> (obs, priv, history, ...)`` and
    ``step() -> (obs, priv, history, ..., rewards, dones, infos)``; the deployable
    policy reads only the current single-frame ``obs`` and the stacked ``history``.
    ``obs_index``/``history_index`` locate those tensors inside the env tuples.
    """

    obs_index: int = 0
    history_index: int = 2

    def reset(self, env: Any) -> Any:
        env.reset()
        out = env.get_observations()
        return (out[self.obs_index].to(self.device), out[self.history_index].to(self.device))

    def act(self, state: Any) -> torch.Tensor:
        obs, history = state
        return self.policy(obs.detach(), history.detach())

    def step(self, env: Any, actions: torch.Tensor):
        out = env.step(actions)
        # env.step() returns (..., rewards, dones, infos); rewards/dones are the
        # last-three block regardless of how many obs tensors precede them.
        rewards, dones = out[-3], out[-2]
        state = (out[self.obs_index].to(self.device), out[self.history_index].to(self.device))
        return state, rewards, dones


class FeatureHistoryAdapter(EvalAdapter):
    """SysID (explicit estimator): the policy reads the estimator feature history.

    ``get_observations() -> (estimator_features, estimator_labels, privileged_obs)``
    and ``step() -> (estimator_features, estimator_labels, privileged_obs, rewards,
    dones, infos)``. The deployable input is ``estimator_features`` only; the actor
    runs the estimator internally, so labels never reach deployment.
    """

    features_index: int = 0

    def reset(self, env: Any) -> Any:
        env.reset()
        out = env.get_observations()
        return out[self.features_index].to(self.device)

    def act(self, state: Any) -> torch.Tensor:
        return self.policy(state.detach())

    def step(self, env: Any, actions: torch.Tensor):
        out = env.step(actions)
        rewards, dones = out[-3], out[-2]
        return out[self.features_index].to(self.device), rewards, dones


class RMAStudentDiagnosticAdapter(EvalAdapter):
    """RMA student rollout with paired teacher/student gap accumulation.

    The deployed student still controls the environment. At every state in the
    measured student trajectory, the privileged teacher is evaluated without
    stepping a second environment; latent MSE and action MAE are therefore paired
    over identical observations rather than sampled once after a reset.
    """

    def __init__(self, actor_critic: torch.nn.Module, device: torch.device):
        super().__init__(actor_critic.act_student, device)
        self.actor_critic = actor_critic
        self.begin_measurement()

    def begin_measurement(self) -> None:
        self.latent_squared_error = 0.0
        self.latent_elements = 0
        self.action_absolute_error = 0.0
        self.action_elements = 0

    def _state(self, out: Any):
        return tuple(out[index].to(self.device) for index in (0, 1, 2))

    def reset(self, env: Any) -> Any:
        env.reset()
        return self._state(env.get_observations())

    def act(self, state: Any) -> torch.Tensor:
        obs, privileged_obs, history = state
        history_input = history.unsqueeze(1) \
            if self.actor_critic.history_encoder_type == "TCN" else history
        teacher_latent = self.actor_critic.privilege_encoder(privileged_obs.detach())
        student_latent = self.actor_critic.history_encoder(history_input.detach())
        teacher_actions = self.actor_critic.act_teacher(obs.detach(), privileged_obs.detach())
        student_actions = self.actor_critic.act_student(obs.detach(), history.detach())
        latent_error = student_latent - teacher_latent
        action_error = student_actions - teacher_actions
        self.latent_squared_error += float(latent_error.square().sum().item())
        self.latent_elements += latent_error.numel()
        self.action_absolute_error += float(action_error.abs().sum().item())
        self.action_elements += action_error.numel()
        return student_actions

    def step(self, env: Any, actions: torch.Tensor):
        out = env.step(actions)
        return self._state(out), out[-3], out[-2]

    def diagnostics(self) -> Dict[str, float]:
        if self.latent_elements == 0 or self.action_elements == 0:
            raise RuntimeError("RMA diagnostics requested before a measured action")
        return {
            "latent_mse": self.latent_squared_error / self.latent_elements,
            "action_mae": self.action_absolute_error / self.action_elements,
        }
