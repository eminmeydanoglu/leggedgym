"""Source-faithful probe for the published upstream Go2 MoE-CTS policies.

This module is deliberately independent from the Genesis/Isaac environment
stack.  It has two small entry points:

``offline``
    Load a *raw* upstream training checkpoint or a student/actor-only
    deployment bridge, evaluate a fixed ``obs/history`` bank, and write every
    gate/expert/action intervention together with machine-readable metrics.

``closed_loop``
    Run independent, headless MuJoCo rollouts for learned, uniform, top-1 and
    fixed-expert routes.  Each route gets a fresh MuJoCo state and history;
    falls terminate that route and no post-terminal state is recorded.

``jit_parity``
    Compare the local state-dict adapter against a deployable TorchScript
    artifact on a deterministic observation/history sample.  This is an ABI
    parity check, not evidence that the missing privileged teacher is present.

The implementation mirrors the upstream ``ActorCriticMoECTS`` and
``rsl_rl.modules.utils`` source contracts.  The local copies of the tiny
network classes are intentional: importing a second package called
``rsl_rl`` from the reference checkout can otherwise silently resolve this
repository's package and produce a plausible-looking, wrong ABI.

Important source contracts (from ``go2_rl_gym``):

* observation: ``[ang_vel(3), projected_gravity(3), command(3),
  dof_pos_error(12), dof_vel(12), previous_action(12)]``;
* runner-owned history: five frames, oldest to newest, flattening as
  ``history.reshape(B, 5 * 45)``;
* student MoE: eight experts of width 32, weighted sum followed by L2
  normalization; actor input is ``[normalized_mixed_latent, obs]``;
* on reset the runner zeros an environment's history and appends the new
  observation.  There is no fabricated teacher/oracle input in this probe.

Only the standard library, NumPy and PyTorch are imported at module load.  The
optional MuJoCo dependency is imported inside ``run_closed_loop``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shlex
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Frozen upstream contract
# ---------------------------------------------------------------------------

OBS_TERMS = (
    "base_ang_vel",
    "projected_gravity",
    "commands",
    "dof_pos_error",
    "dof_vel",
    "previous_action",
)

UPSTREAM_MODEL_JOINT_NAMES = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)

# The values below are the deployment YAML values, not Genesis defaults.
UPSTREAM_DEFAULT_ANGLES = np.asarray(
    [0.1, 0.8, -1.5, -0.1, 0.8, -1.5,
     0.1, 1.0, -1.5, -0.1, 1.0, -1.5],
    dtype=np.float32,
)

PAPER_COMMANDS = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float32,
)
PAPER_COMMAND_LABELS = (
    "forward", "backward", "strafe_left", "strafe_right", "turn_left", "turn_right"
)

CLOSED_LOOP_ROUTE_MODES = (
    "learned",
    "uniform",
    "top1",
    *(f"fixed_expert_{index}" for index in range(8)),
)
OFFLINE_ONLY_ROUTE_MODES = ("shuffled",)

# MuJoCo can emit several finite but physically runaway control/observation
# rows before the height-based fall check fires.  They are not useful for
# same-state representation statistics and would dominate squared distances.
SPECIALIZATION_MAX_ABS_OBS = 1.0e3


@dataclass(frozen=True)
class UpstreamMoECTSContract:
    """Dimensions and ordering that are part of the upstream checkpoint ABI."""

    num_obs: int = 45
    num_privileged_obs: int = 263
    history_length: int = 5
    latent_dim: int = 32
    num_actions: int = 12
    expert_num: int = 8
    norm_type: str = "l2norm"
    action_scale: float = 0.25
    lin_vel_scale: float = 2.0
    ang_vel_scale: float = 0.25
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05
    command_scale: Tuple[float, float, float] = (2.0, 2.0, 0.25)
    obs_terms: Tuple[str, ...] = OBS_TERMS
    model_joint_names: Tuple[str, ...] = UPSTREAM_MODEL_JOINT_NAMES

    @property
    def history_dim(self) -> int:
        return self.history_length * self.num_obs

    def validate(self) -> "UpstreamMoECTSContract":
        if self.num_obs != 45 or self.num_privileged_obs != 263:
            raise ValueError("published upstream Go2 MoE-CTS expects 45/263 observations")
        if self.history_length != 5 or self.latent_dim != 32 or self.expert_num != 8:
            raise ValueError("published upstream MoE-CTS expects history=5, latent=32, experts=8")
        if len(self.model_joint_names) != self.num_actions:
            raise ValueError("model joint list must contain exactly 12 joints")
        if tuple(self.obs_terms) != OBS_TERMS:
            raise ValueError(f"observation term ordering changed: {self.obs_terms}")
        if self.norm_type != "l2norm":
            raise ValueError("this probe only supports source l2norm checkpoints")
        return self

    def as_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["history_dim"] = self.history_dim
        out["model_joint_names"] = list(self.model_joint_names)
        out["obs_terms"] = list(self.obs_terms)
        out["command_scale"] = list(self.command_scale)
        return out


CONTRACT = UpstreamMoECTSContract().validate()


# ---------------------------------------------------------------------------
# Exact small network implementation from upstream rsl_rl.modules.utils
# ---------------------------------------------------------------------------


def _get_activation(name: str) -> nn.Module:
    factories = {
        "elu": nn.ELU,
        "selu": nn.SELU,
        "relu": nn.ReLU,
        "crelu": nn.ReLU,
        "lrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    try:
        return factories[name]()
    except KeyError as exc:
        raise ValueError(f"unsupported upstream activation: {name!r}") from exc


class _SourceL2Norm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Exact upstream call: F.normalize(x, p=2.0, dim=-1).
        return F.normalize(x, p=2.0, dim=-1)


class _SourceSimNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dim = 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.view(*shape[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shape)


class _SourceMLP(nn.Module):
    def __init__(self, dims: Sequence[int], activation: str = "elu", last_activation: bool = False) -> None:
        super().__init__()
        # This mirrors the upstream source's module naming: ``network.0`` etc.
        act = _get_activation(activation)
        layers: List[nn.Module] = []
        last_dim = int(dims[0])
        for h_dim in dims[1:-1]:
            layers.extend([nn.Linear(last_dim, int(h_dim)), act])
            last_dim = int(h_dim)
        layers.append(nn.Linear(last_dim, int(dims[-1])))
        if last_activation:
            layers.append(act)
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class _SourceExperts(nn.Module):
    def __init__(
        self,
        expert_num: int,
        input_dim: int,
        backbone_hidden_dims: Sequence[int],
        expert_hidden_dim: int,
        output_dim: int,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.expert_num = int(expert_num)
        self.output_dim = int(output_dim)
        self.backbone = _SourceMLP(
            [input_dim, *backbone_hidden_dims, expert_num * expert_hidden_dim],
            activation,
            last_activation=True,
        )
        self.experts = nn.Conv1d(
            in_channels=expert_num * expert_hidden_dim,
            out_channels=expert_num * output_dim,
            kernel_size=1,
            groups=expert_num,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_features = self.backbone(x).unsqueeze(-1)
        expert_outs = self.experts(shared_features).squeeze(-1)
        return expert_outs.reshape(-1, self.expert_num, self.output_dim)


class _SourceMoE(nn.Module):
    def __init__(
        self,
        expert_num: int,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.experts = _SourceExperts(
            expert_num=expert_num,
            input_dim=input_dim,
            backbone_hidden_dims=hidden_dims[:-1],
            expert_hidden_dim=hidden_dims[-1],
            output_dim=output_dim,
            activation=activation,
        )
        self.gating_network = nn.Sequential(
            _SourceMLP([input_dim, *hidden_dims[:-1], expert_num], activation),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = self.gating_network(x)
        expert_outs = self.experts(x)
        return torch.sum(weights.unsqueeze(-1) * expert_outs, dim=1), weights


class _SourceStudentMoEEncoder(nn.Module):
    def __init__(
        self,
        expert_num: int,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: str = "elu",
        norm_type: str = "l2norm",
    ) -> None:
        super().__init__()
        self.norm_layer = _SourceL2Norm() if norm_type == "l2norm" else _SourceSimNorm()
        self.moe = _SourceMoE(
            expert_num=expert_num,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent, weights = self.moe(obs)
        return self.norm_layer(latent), weights


class SourceActorCriticMoECTS(nn.Module):
    """Minimal source-compatible ``ActorCriticMoECTS``.

    Parameter/module names intentionally match the upstream training source so
    raw checkpoints can be loaded with ``strict=True``.  The history buffer is
    non-persistent, as in upstream.
    """

    is_recurrent = False

    def __init__(
        self,
        num_obs: int = 45,
        num_critic_obs: int = 263,
        num_actions: int = 12,
        num_envs: int = 1,
        history_length: int = 5,
        actor_hidden_dims: Sequence[int] = (512, 256, 128),
        critic_hidden_dims: Sequence[int] = (512, 256, 128),
        teacher_encoder_hidden_dims: Sequence[int] = (512, 256),
        student_encoder_hidden_dims: Sequence[int] = (512, 256, 256),
        expert_num: int = 8,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        latent_dim: int = 32,
        norm_type: str = "l2norm",
    ) -> None:
        super().__init__()
        if norm_type not in ("l2norm", "simnorm"):
            raise ValueError(f"unsupported normalization: {norm_type}")
        self.num_actions = int(num_actions)
        self.history_length = int(history_length)
        self.register_buffer(
            "history",
            torch.zeros((int(num_envs), int(history_length), int(num_obs))),
            persistent=False,
        )
        self.teacher_encoder = nn.Sequential(
            _SourceMLP([num_critic_obs, *teacher_encoder_hidden_dims, latent_dim], activation),
            _SourceL2Norm() if norm_type == "l2norm" else _SourceSimNorm(),
        )
        self.student_moe_encoder = _SourceStudentMoEEncoder(
            expert_num=expert_num,
            input_dim=num_obs * history_length,
            hidden_dims=student_encoder_hidden_dims,
            output_dim=latent_dim,
            activation=activation,
            norm_type=norm_type,
        )
        self.actor = _SourceMLP([latent_dim + num_obs, *actor_hidden_dims, num_actions], activation)
        self.critic = _SourceMLP([latent_dim + num_critic_obs, *critic_hidden_dims, 1], activation)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))


class _SourceMoENGCTSStudentEncoder(nn.Module):
    """The 137k release's ``actor_critic_moe_ng_cts.StudentMoEEncoder``.

    The final 164k release uses the nested ``MoE`` above.  The 137k paper
    artifact was exported from the no-goal variant, whose expert backbone and
    gate are separate sequential modules.  Keeping this class here avoids
    renaming a raw checkpoint into a superficially compatible but semantically
    different model.
    """

    def __init__(
        self,
        expert_dim: int,
        gating_dim: int = 225,
        hidden_dims: Sequence[int] = (512, 256),
        expert_num: int = 8,
        expert_hidden_dim: int = 256,
        latent_dim: int = 32,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.expert_num = int(expert_num)
        self.latent_dim = int(latent_dim)
        self.expert_input_dim = int(expert_dim)
        self.norm_layer = _SourceL2Norm()
        act = _get_activation(activation)
        expert_layers: List[nn.Module] = []
        last_dim = int(expert_dim)
        for width in hidden_dims:
            expert_layers.extend([nn.Linear(last_dim, int(width)), act])
            last_dim = int(width)
        self.experts_backbone = nn.Sequential(*expert_layers)
        self.experts_hidden = nn.Sequential(
            nn.Linear(last_dim, expert_num * expert_hidden_dim),
            act,
        )
        self.experts_out = nn.Conv1d(
            in_channels=expert_num * expert_hidden_dim,
            out_channels=expert_num * latent_dim,
            kernel_size=1,
            groups=expert_num,
        )
        gate_layers: List[nn.Module] = []
        last_dim = int(gating_dim)
        for width in hidden_dims:
            gate_layers.extend([nn.Linear(last_dim, int(width)), act])
            last_dim = int(width)
        gate_layers.extend([nn.Linear(last_dim, expert_num), nn.Softmax(dim=-1)])
        self.gating_network = nn.Sequential(*gate_layers)

    def forward(self, obs: torch.Tensor, obs_no_goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = self.gating_network(obs)
        shared = self.experts_backbone(obs_no_goal)
        hidden = self.experts_hidden(shared).unsqueeze(-1)
        flat = self.experts_out(hidden)
        expert_latent = flat.reshape(-1, self.expert_num, self.latent_dim)
        raw = torch.sum(weights.unsqueeze(-1) * expert_latent, dim=1)
        return self.norm_layer(raw), weights, expert_latent


class SourceActorCriticMoENGCTS(nn.Module):
    """Raw source model used by the 137k no-goal release."""

    is_recurrent = False

    def __init__(
        self,
        expert_dim: int = 225,
        obs_no_goal_mask: Optional[Sequence[bool]] = None,
        num_obs: int = 45,
        num_critic_obs: int = 263,
        num_actions: int = 12,
        history_length: int = 5,
        actor_hidden_dims: Sequence[int] = (512, 256, 128),
        critic_hidden_dims: Sequence[int] = (512, 256, 128),
        teacher_encoder_hidden_dims: Sequence[int] = (512, 256),
        student_encoder_hidden_dims: Sequence[int] = (512, 256),
        expert_num: int = 8,
        latent_dim: int = 32,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        mask = tuple(obs_no_goal_mask) if obs_no_goal_mask is not None else (True,) * num_obs
        if len(mask) != num_obs:
            raise ValueError("obs_no_goal_mask must have one entry per observation term")
        self.history_length = int(history_length)
        self.register_buffer("obs_no_goal_mask", torch.tensor(mask, dtype=torch.bool), persistent=False)
        self.student_moe_encoder = _SourceMoENGCTSStudentEncoder(
            expert_dim=int(expert_dim),
            gating_dim=num_obs * history_length,
            hidden_dims=student_encoder_hidden_dims,
            expert_num=expert_num,
            latent_dim=latent_dim,
            activation=activation,
        )
        act = _get_activation(activation)
        teacher_layers: List[nn.Module] = []
        last = num_critic_obs
        for width in teacher_encoder_hidden_dims:
            teacher_layers.extend([nn.Linear(last, int(width)), act])
            last = int(width)
        teacher_layers.extend([nn.Linear(last, latent_dim), _SourceL2Norm()])
        self.teacher_encoder = nn.Sequential(*teacher_layers)
        actor_layers: List[nn.Module] = []
        last = latent_dim + num_obs
        for width in actor_hidden_dims:
            actor_layers.extend([nn.Linear(last, int(width)), act])
            last = int(width)
        actor_layers.append(nn.Linear(last, num_actions))
        self.actor = nn.Sequential(*actor_layers)
        critic_layers: List[nn.Module] = []
        last = latent_dim + num_critic_obs
        for width in critic_hidden_dims:
            critic_layers.extend([nn.Linear(last, int(width)), act])
            last = int(width)
        critic_layers.append(nn.Linear(last, 1))
        self.critic = nn.Sequential(*critic_layers)
        self.std = nn.Parameter(torch.ones(num_actions))


# ---------------------------------------------------------------------------
# Checkpoint adapter
# ---------------------------------------------------------------------------


def sha256_file(path: os.PathLike[str] | str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _extract_state_dict(checkpoint: Any) -> Tuple[Mapping[str, torch.Tensor], Dict[str, Any]]:
    if isinstance(checkpoint, torch.jit.ScriptModule):
        # The released deploy files are TorchScript wrappers, but their
        # state_dict still contains the student MoE and actor weights.  That is
        # sufficient for route/intervention analysis and MuJoCo rollouts; the
        # teacher/critic are intentionally unavailable.  A tiny unrelated JIT
        # module must still fail explicitly instead of being mistaken for a
        # policy bridge.
        state = {str(k): v for k, v in checkpoint.state_dict().items() if torch.is_tensor(v)}
        if not any(k.startswith("student_moe_encoder.") for k in state):
            raise TypeError(
                "TorchScript deployment checkpoint exposes only mixed latent/action "
                "or an unrelated module; pass a raw upstream training checkpoint "
                "with model_state_dict or a MoE deployment artifact"
            )
        return state, {"deployment_artifact": "torchscript", "teacher_available": False}
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"unsupported checkpoint object: {type(checkpoint).__name__}")
    info = dict(checkpoint.get("infos", {})) if isinstance(checkpoint.get("infos", {}), Mapping) else {}
    state: Any = checkpoint
    for key in ("model_state_dict", "state_dict", "model", "actor_critic"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            state = candidate
            break
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint has no tensor state_dict")
    state = {str(k): v for k, v in state.items() if torch.is_tensor(v)}
    if not state:
        raise ValueError("checkpoint state_dict contains no tensors")
    # Common wrappers from DDP/export scripts are harmless; remove one layer.
    for prefix in ("module.", "actor_critic."):
        if state and all(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]: v for k, v in state.items()}
    return state, info


def _undo_local_dof_permutation(
    state: Dict[str, torch.Tensor],
    permutation: Sequence[int],
) -> Dict[str, torch.Tensor]:
    """Undo ``import_go2_rl_gym_policy.apply_dof_permutation``.

    The bridge checkpoints already present in this repository were made for
    the local Genesis asset order (FR/FL/RR/RL), while this probe's contract is
    the upstream MuJoCo/IsaacGym order (FL/FR/RL/RR).  If ``perm[j]`` is the
    reference slot occupied by local slot ``j``, then ``reference[:, perm] =
    local`` for input blocks and ``reference[perm] = local`` for output rows.
    Raw upstream training checkpoints do not carry this metadata and are left
    untouched.
    """
    perm = [int(x) for x in permutation]
    if sorted(perm) != list(range(len(perm))) or len(perm) != 12:
        raise ValueError(f"invalid dof_permutation metadata: {permutation!r}")
    out = {k: v.clone() for k, v in state.items()}

    def reorder_input(key: str, offsets: Sequence[int]) -> None:
        if key not in state:
            return
        weight = state[key]
        if weight.ndim != 2:
            raise ValueError(f"expected linear input weight for {key}, got {tuple(weight.shape)}")
        restored = weight.clone()
        for offset in offsets:
            restored[:, offset + torch.as_tensor(perm, device=weight.device)] = weight[:, offset:offset + 12]
        out[key] = restored

    # Actor input: latent prefix followed by one 45-D observation frame.
    reorder_input("actor.network.0.weight", (32 + 9, 32 + 21, 32 + 33))
    # Both student entry networks consume five flattened source frames.
    offsets = tuple(frame * 45 + term for frame in range(5) for term in (9, 21, 33))
    reorder_input("student_moe_encoder.moe.experts.backbone.network.0.weight", offsets)
    reorder_input("student_moe_encoder.moe.gating_network.0.network.0.weight", offsets)
    output_key = "actor.network.6.weight"
    if output_key in state:
        out[output_key] = state[output_key][torch.as_tensor(perm, device=state[output_key].device)]
    bias_key = "actor.network.6.bias"
    if bias_key in state:
        out[bias_key] = state[bias_key][torch.as_tensor(perm, device=state[bias_key].device)]
    return out


def _map_deployment_state_to_source(
    state: Mapping[str, torch.Tensor],
    *,
    dof_permutation: Optional[Sequence[int]] = None,
) -> Dict[str, torch.Tensor]:
    """Map the local weights+latent bridge layout to source names.

    ``model_0.pt`` snapshots in this checkout were built from the upstream JIT
    files and use ``history_encoder``/``privilege_encoder`` names.  They are
    accepted for reproducibility, but marked teacher-unavailable by the caller;
    this path is never confused with a raw source training checkpoint.
    """
    mapped: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("privilege_encoder.0."):
            mapped["teacher_encoder.0.network." + key[len("privilege_encoder.0."):]] = value
        elif key.startswith("history_encoder."):
            mapped["student_moe_encoder." + key[len("history_encoder."):]] = value
        elif key.startswith("actor.") and not key.startswith("actor.network."):
            mapped["actor.network." + key[len("actor."):]] = value
        elif key.startswith("critic.") and not key.startswith("critic.network."):
            mapped["critic.network." + key[len("critic."):]] = value
        else:
            mapped[key] = value
    if dof_permutation is not None:
        mapped = _undo_local_dof_permutation(mapped, dof_permutation)
    return mapped


@dataclass
class LoadedMoECTSCheckpoint:
    model: nn.Module
    path: str
    sha256: str
    iteration: int
    schema: str
    teacher_available: bool
    critic_available: bool
    model_kind: str = "moe_cts"
    provenance: Dict[str, Any] = field(default_factory=dict)
    missing_keys: Tuple[str, ...] = ()
    unexpected_keys: Tuple[str, ...] = ()


def load_upstream_checkpoint(
    path: os.PathLike[str] | str,
    *,
    device: str | torch.device = "cpu",
    strict_raw: bool = True,
) -> LoadedMoECTSCheckpoint:
    """Load raw source weights or a documented local deployment bridge.

    Raw source checkpoints are loaded strictly.  A bridge checkpoint is only
    accepted when all student MoE + actor keys are present; missing random
    teacher/critic keys are reported rather than silently fabricated as valid
    oracle observations.
    """
    path = str(Path(path).expanduser().resolve())
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    # ``torch.jit.load`` is deliberately attempted first so the released
    # deployment files can be used for a short smoke/parity check.  A JIT file
    # is classified as a deployment bridge below; it never receives a
    # fabricated teacher or gets reported as a raw training checkpoint.
    is_torchscript = False
    try:
        checkpoint = torch.jit.load(path, map_location=device)
        is_torchscript = True
    except (RuntimeError, ValueError, EOFError, OSError):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    state, info = _extract_state_dict(checkpoint)
    source_keys = any(k.startswith("student_moe_encoder.") for k in state)
    bridge_keys = any(k.startswith("history_encoder.") for k in state)
    if not source_keys and not bridge_keys:
        raise ValueError(
            "checkpoint is neither upstream raw ActorCriticMoECTS nor the "
            "documented deployment bridge; expected student_moe_encoder/history_encoder keys"
        )
    flat_ng_keys = source_keys and any(k.startswith("student_moe_encoder.experts_backbone.") for k in state)
    model_kind = "moe_ng" if flat_ng_keys else "moe_cts"
    # A true source checkpoint carries teacher and critic weights.  JIT and
    # student/actor-only bridges intentionally do not; loading them with
    # strict=False is safe only after the required policy keys are checked.
    has_teacher = any(k.startswith("teacher_encoder.") for k in state)
    has_critic = any(k.startswith("critic.") for k in state)
    # The local ``model_0.pt`` deployment snapshots use the host names
    # ``history_encoder``/``actor`` rather than source names
    # ``student_moe_encoder``/``actor.network``.  They are still
    # student/actor-only deployment bridges and must not be labelled as raw
    # source-training checkpoints merely because their wrapper is a mapping.
    deployment_bridge = bool(
        is_torchscript
        or bridge_keys
        or (source_keys and not (has_teacher and has_critic))
    )
    schema = (
        "deployment_bridge"
        if deployment_bridge
        else ("source_training_moe_ng" if flat_ng_keys else "source_training")
    )
    dof_permutation = info.get("dof_permutation") if isinstance(info, Mapping) else None
    mapped = dict(state) if source_keys else _map_deployment_state_to_source(
        state,
        dof_permutation=dof_permutation,
    )
    if flat_ng_keys:
        expert_weight = state.get("student_moe_encoder.experts_backbone.0.weight")
        if expert_weight is None or expert_weight.ndim != 2:
            raise ValueError("raw MoE-NG checkpoint is missing experts_backbone.0.weight")
        # The source config's command mask is six non-command terms + 36
        # dof/velocity/action terms.  Some released artifacts nevertheless
        # store a full-width expert backbone; use the checkpoint shape as the
        # final authority and let the adapter preserve that behavior.
        mask = (True,) * 6 + (False,) * 3 + (True,) * 36
        model = SourceActorCriticMoENGCTS(
            expert_dim=int(expert_weight.shape[1]),
            obs_no_goal_mask=mask,
        ).to(device)
    else:
        model = SourceActorCriticMoECTS().to(device)
    load_result = model.load_state_dict(
        mapped,
        strict=bool(strict_raw and source_keys and not deployment_bridge),
    )
    required = [
        k for k in model.state_dict()
        if k.startswith("student_moe_encoder.") or k.startswith("actor.")
    ]
    missing_required = sorted(set(required).intersection(load_result.missing_keys))
    if missing_required:
        raise RuntimeError(f"checkpoint is missing required policy weights: {missing_required[:8]}")
    model.eval()
    raw_prov = info.get("provenance", "")
    if isinstance(raw_prov, Mapping):
        raw_prov_text = json.dumps(raw_prov, sort_keys=True)
    else:
        raw_prov_text = str(raw_prov)
    teacher_available = bool(
        source_keys and not deployment_bridge and "randomly initialized" not in raw_prov_text.lower()
    )
    critic_available = bool(
        source_keys and not deployment_bridge and "randomly initialized" not in raw_prov_text.lower()
    )
    iteration = -1
    if isinstance(checkpoint, Mapping):
        for key in ("iter", "iteration", "update"):
            if checkpoint.get(key) is not None:
                try:
                    iteration = int(checkpoint[key])
                except (TypeError, ValueError):
                    pass
                break
    provenance = {
        "infos": info,
        "raw_provenance": raw_prov,
        "deployment_artifact": "torchscript" if is_torchscript else None,
        "deployment_bridge": deployment_bridge,
        "teacher_available": teacher_available,
        "critic_available": critic_available,
        "strict_source_load": bool(strict_raw and source_keys and not deployment_bridge),
    }
    return LoadedMoECTSCheckpoint(
        model=model,
        path=path,
        sha256=sha256_file(path),
        iteration=iteration,
        schema=schema,
        teacher_available=teacher_available,
        critic_available=critic_available,
        model_kind="moe_ng" if flat_ng_keys else "moe_cts",
        provenance=provenance,
        missing_keys=tuple(load_result.missing_keys),
        unexpected_keys=tuple(load_result.unexpected_keys),
    )


@dataclass
class UpstreamMoECTSAdapter:
    loaded: LoadedMoECTSCheckpoint
    contract: UpstreamMoECTSContract = CONTRACT

    @classmethod
    def from_checkpoint(cls, path: os.PathLike[str] | str, **kwargs: Any) -> "UpstreamMoECTSAdapter":
        return cls(load_upstream_checkpoint(path, **kwargs))

    @property
    def model(self) -> nn.Module:
        return self.loaded.model

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _history(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim == 3:
            if tuple(history.shape[1:]) != (self.contract.history_length, self.contract.num_obs):
                raise ValueError(f"expected history [N,5,45], got {tuple(history.shape)}")
            return history.reshape(history.shape[0], -1)
        if history.ndim == 2 and history.shape[1] == self.contract.history_dim:
            return history
        raise ValueError(f"expected history [N,225] or [N,5,45], got {tuple(history.shape)}")

    @torch.inference_mode()
    def forward_components(self, obs: torch.Tensor, history: torch.Tensor) -> Dict[str, torch.Tensor]:
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        history_flat = self._history(torch.as_tensor(history, dtype=torch.float32, device=self.device))
        if obs.ndim != 2 or obs.shape[1] != self.contract.num_obs:
            raise ValueError(f"expected obs [N,45], got {tuple(obs.shape)}")
        if self.loaded.model_kind == "moe_ng":
            student = self.model.student_moe_encoder
            assert isinstance(student, _SourceMoENGCTSStudentEncoder)
            # Source MoE-NG gates on the full history and experts on the
            # command-masked history.  Preserve a full-width artifact's
            # historical behavior instead of slicing it into an incompatible
            # 210-D input.
            mask = self.model.obs_no_goal_mask
            masked = history_flat.reshape(-1, self.contract.history_length, self.contract.num_obs)[:, :, mask]
            masked = masked.reshape(history_flat.shape[0], -1)
            expert_input = history_flat if student.expert_input_dim == history_flat.shape[1] else masked
            gate = student.gating_network(history_flat)
            expert_hidden = student.experts_hidden(student.experts_backbone(expert_input)).unsqueeze(-1)
            expert_outputs = student.experts_out(expert_hidden).reshape(
                -1, student.expert_num, student.latent_dim
            )
            normalizer = student.norm_layer
        else:
            student = self.model.student_moe_encoder
            assert isinstance(student, _SourceStudentMoEEncoder)
            gate = student.moe.gating_network(history_flat)
            expert_outputs = student.moe.experts(history_flat)
            normalizer = student.norm_layer
        raw_weighted = torch.sum(gate.unsqueeze(-1) * expert_outputs, dim=1)
        normalized = normalizer(raw_weighted)
        action = self.model.actor(torch.cat([normalized, obs], dim=1))
        return {
            "obs": obs,
            "history_flat": history_flat,
            "gate": gate,
            "expert_outputs": expert_outputs,
            "raw_weighted_latent": raw_weighted,
            "normalized_mixed_latent": normalized,
            "learned_action": action,
        }

    @torch.inference_mode()
    def intervention_actions(
        self,
        components: Mapping[str, torch.Tensor],
        *,
        seed: int = 0,
        shuffled_gate: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate all requested gate interventions on the same fixed rows."""
        obs = components["obs"]
        gate = components["gate"]
        experts = components["expert_outputs"]
        n, k = gate.shape
        variants: Dict[str, torch.Tensor] = {"learned": gate}
        variants["uniform"] = torch.full_like(gate, 1.0 / k)
        if shuffled_gate is None:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            perm = torch.randperm(n, generator=generator).to(gate.device)
            variants["shuffled"] = gate[perm]
        else:
            shuffled_gate = torch.as_tensor(shuffled_gate, dtype=gate.dtype, device=gate.device)
            if tuple(shuffled_gate.shape) != tuple(gate.shape):
                raise ValueError("shuffled_gate must have the same [N,8] shape as gate")
            variants["shuffled"] = shuffled_gate
        top1 = torch.zeros_like(gate)
        top1.scatter_(1, gate.argmax(dim=1, keepdim=True), 1.0)
        variants["top1"] = top1
        normalizer = self.model.student_moe_encoder.norm_layer
        out: Dict[str, torch.Tensor] = {}
        for name, weights in variants.items():
            raw = torch.sum(weights.unsqueeze(-1) * experts, dim=1)
            latent = normalizer(raw)
            out[f"gate_{name}"] = weights
            out[f"raw_latent_{name}"] = raw
            out[f"normalized_latent_{name}"] = latent
            out[f"action_{name}"] = self.model.actor(torch.cat([latent, obs], dim=1))
        single_weights = torch.eye(k, dtype=gate.dtype, device=gate.device)
        single_raw = torch.einsum("ek,nkd->ned", single_weights, experts)
        single_latent = normalizer(single_raw)
        obs_rep = obs[:, None, :].expand(n, k, obs.shape[1]).reshape(n * k, -1)
        single_action = self.model.actor(
            torch.cat([single_latent.reshape(n * k, -1), obs_rep], dim=1)
        ).reshape(n, k, -1)
        out["single_expert_raw_latent"] = single_raw
        out["single_expert_normalized_latent"] = single_latent
        out["single_expert_action"] = single_action
        for expert_index in range(k):
            out[f"action_fixed_expert_{expert_index}"] = single_action[:, expert_index, :]
        return out

    @staticmethod
    def route_action(interventions: Mapping[str, torch.Tensor], route_mode: str) -> torch.Tensor:
        """Return the action that a named closed-loop route must physically use.

        ``shuffled`` deliberately has no route here: it is an offline
        same-state gate intervention because there is no reproducible causal
        single-environment shuffle semantics.  Keeping the dispatch explicit
        prevents a caller from accidentally driving MuJoCo with an action that
        was computed from another rollout's history.
        """
        if route_mode in ("learned", "uniform", "top1"):
            return interventions[f"action_{route_mode}"]
        match = re.fullmatch(r"(?:fixed_expert_|expert)([0-7])", str(route_mode))
        if match:
            return interventions[f"action_fixed_expert_{int(match.group(1))}"]
        if route_mode in OFFLINE_ONLY_ROUTE_MODES:
            raise ValueError(
                "route_mode='shuffled' is offline-only; its cross-row gate assignment "
                "has no defined single-environment closed-loop semantics"
            )
        raise ValueError(
            f"unknown route_mode {route_mode!r}; expected {CLOSED_LOOP_ROUTE_MODES} "
            f"or offline-only {OFFLINE_ONLY_ROUTE_MODES}"
        )


# ---------------------------------------------------------------------------
# History and MuJoCo observation/action adapters
# ---------------------------------------------------------------------------


class SourceHistoryCollector:
    """Runner-owned 5x45 history with source reset/append semantics."""

    def __init__(self, batch_size: int, contract: UpstreamMoECTSContract = CONTRACT, device: str | torch.device = "cpu") -> None:
        self.contract = contract.validate()
        self.state = torch.zeros(
            (int(batch_size), contract.history_length, contract.num_obs),
            dtype=torch.float32,
            device=device,
        )

    def reset(self, done: Optional[torch.Tensor] = None) -> None:
        if done is None:
            self.state.zero_()
            return
        done = torch.as_tensor(done, dtype=torch.bool, device=self.state.device).reshape(-1)
        if done.numel() != self.state.shape[0]:
            raise ValueError("done batch size does not match history")
        self.state[done] = 0.0

    def append(self, obs: torch.Tensor, done: Optional[torch.Tensor] = None) -> torch.Tensor:
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.state.device)
        if obs.ndim != 2 or tuple(obs.shape) != (self.state.shape[0], self.contract.num_obs):
            raise ValueError(f"expected obs [{self.state.shape[0]},45], got {tuple(obs.shape)}")
        # Upstream runner: history[dones > 0] = 0; history = cat(history[:,1:], obs).
        # ``done=None`` means "no reset" for a normal transition.  A full
        # reset is an explicit ``initial()`` call; treating None as a reset
        # here would make every history a one-frame history.
        if done is not None:
            self.reset(done)
        self.state = torch.cat([self.state[:, 1:], obs.unsqueeze(1)], dim=1)
        return self.state

    def initial(self, obs: torch.Tensor) -> torch.Tensor:
        self.reset()
        return self.append(obs)


@dataclass(frozen=True)
class JointOrderAdapter:
    """Named, testable model<->MuJoCo joint permutation."""

    mujoco_joint_names: Tuple[str, ...]
    model_joint_names: Tuple[str, ...] = UPSTREAM_MODEL_JOINT_NAMES

    def __post_init__(self) -> None:
        if set(self.mujoco_joint_names) != set(self.model_joint_names):
            raise ValueError("MuJoCo and model joint names must be the same set")
        if len(self.mujoco_joint_names) != len(set(self.mujoco_joint_names)):
            raise ValueError("duplicate MuJoCo joint name")

    @property
    def model_to_mujoco(self) -> Tuple[int, ...]:
        model_idx = {name: i for i, name in enumerate(self.model_joint_names)}
        return tuple(model_idx[name] for name in self.mujoco_joint_names)

    @property
    def mujoco_to_model(self) -> Tuple[int, ...]:
        mj_idx = {name: i for i, name in enumerate(self.mujoco_joint_names)}
        return tuple(mj_idx[name] for name in self.model_joint_names)

    def to_mujoco(self, model_values: np.ndarray) -> np.ndarray:
        x = np.asarray(model_values)
        return x[..., list(self.model_to_mujoco)]

    def to_model(self, mujoco_values: np.ndarray) -> np.ndarray:
        x = np.asarray(mujoco_values)
        return x[..., list(self.mujoco_to_model)]


def quat_rotate_inverse(q: Sequence[float], v: Sequence[float]) -> np.ndarray:
    """Exact quaternion convention used by upstream deploy_go2.py."""
    q = np.asarray(q, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    qw = q[0]
    qvec = q[1:]
    return v * (2 * qw * qw - 1) - np.cross(qvec, v) * qw * 2 + qvec * np.dot(qvec, v) * 2


def gravity_orientation(quaternion: Sequence[float]) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(quaternion, dtype=np.float32)
    return np.asarray(
        [2 * (-qz * qx + qw * qy), -2 * (qz * qy + qw * qx), 1 - 2 * (qw * qw + qz * qz)],
        dtype=np.float32,
    )


@dataclass
class MujocoObservationAdapter:
    """Build source 45-D observations and map policy actions to MuJoCo order."""

    joint_order: JointOrderAdapter = field(
        default_factory=lambda: JointOrderAdapter(UPSTREAM_MODEL_JOINT_NAMES)
    )
    default_angles_model: np.ndarray = field(default_factory=lambda: UPSTREAM_DEFAULT_ANGLES.copy())
    contract: UpstreamMoECTSContract = CONTRACT

    def __post_init__(self) -> None:
        self.default_angles_model = np.asarray(self.default_angles_model, dtype=np.float32)
        if self.default_angles_model.shape != (self.contract.num_actions,):
            raise ValueError("default angles must be [12]")

    @property
    def default_angles_mujoco(self) -> np.ndarray:
        return self.joint_order.to_mujoco(self.default_angles_model)

    def observation(
        self,
        *,
        quaternion: Sequence[float],
        world_linear_velocity: Sequence[float],
        world_angular_velocity: Sequence[float],
        command: Sequence[float],
        qpos_mujoco: Sequence[float],
        qvel_mujoco: Sequence[float],
        previous_action_model: Sequence[float],
    ) -> np.ndarray:
        qpos_mj = np.asarray(qpos_mujoco, dtype=np.float32).reshape(-1)
        qvel_mj = np.asarray(qvel_mujoco, dtype=np.float32).reshape(-1)
        prev = np.asarray(previous_action_model, dtype=np.float32).reshape(-1)
        command = np.asarray(command, dtype=np.float32).reshape(-1)
        if qpos_mj.size != 12 or qvel_mj.size != 12 or prev.size != 12 or command.size != 3:
            raise ValueError("MuJoCo observation inputs must be qpos/qvel/action=12 and command=3")
        ang = quat_rotate_inverse(quaternion, world_angular_velocity) * self.contract.ang_vel_scale
        grav = gravity_orientation(quaternion)
        cmd = command * np.asarray(self.contract.command_scale, dtype=np.float32)
        qerr = (self.joint_order.to_model(qpos_mj) - self.default_angles_model) * self.contract.dof_pos_scale
        dq = self.joint_order.to_model(qvel_mj) * self.contract.dof_vel_scale
        obs = np.concatenate([ang, grav, cmd, qerr, dq, prev]).astype(np.float32)
        if obs.shape != (self.contract.num_obs,):
            raise AssertionError(f"constructed observation has wrong shape: {obs.shape}")
        return obs

    def action_to_mujoco(self, action_model: Sequence[float]) -> np.ndarray:
        action_model = np.asarray(action_model, dtype=np.float32).reshape(-1)
        if action_model.size != self.contract.num_actions:
            raise ValueError("action must be [12]")
        return self.joint_order.to_mujoco(action_model)

    def target_q_mujoco(self, action_model: Sequence[float]) -> np.ndarray:
        return self.action_to_mujoco(action_model) * self.contract.action_scale + self.default_angles_mujoco


# ---------------------------------------------------------------------------
# Fixed bank I/O and analysis
# ---------------------------------------------------------------------------


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_bank_object(path: os.PathLike[str] | str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    path = str(path)
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: np.asarray(z[k]) for k in z.files}
        meta: Dict[str, Any] = {}
        if "metadata_json" in arrays:
            try:
                meta = json.loads(str(arrays["metadata_json"].reshape(-1)[0]))
            except (ValueError, TypeError, json.JSONDecodeError):
                meta = {"metadata_json_parse": "failed"}
        return arrays, meta
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, Mapping):
        raise TypeError(f"bank must be npz or mapping checkpoint, got {type(obj).__name__}")
    arrays = {str(k): _as_numpy(v) for k, v in obj.items() if k != "metadata" and k != "meta"}
    meta_obj = obj.get("metadata", obj.get("meta", {}))
    return arrays, dict(meta_obj) if isinstance(meta_obj, Mapping) else {}


def load_fixed_bank(path: os.PathLike[str] | str, contract: UpstreamMoECTSContract = CONTRACT) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    arrays, metadata = _load_bank_object(path)
    aliases = {
        "current_obs": "obs",
        "obs_history": "history",
        "privileged_obs": "privileged_obs",
        "commands": "commands",
    }
    for old, new in aliases.items():
        if new not in arrays and old in arrays:
            arrays[new] = arrays[old]
    if "obs" not in arrays:
        raise ValueError("bank must contain obs/current observation")
    arrays["obs"] = np.asarray(arrays["obs"], dtype=np.float32)
    if arrays["obs"].ndim != 2 or arrays["obs"].shape[1] != contract.num_obs:
        raise ValueError(f"bank obs must be [N,45], got {arrays['obs'].shape}")
    n = arrays["obs"].shape[0]
    if "history" not in arrays:
        raise ValueError("bank must contain source runner history as history or obs_history")
    hist = np.asarray(arrays["history"], dtype=np.float32)
    if hist.ndim == 2 and hist.shape[1] == contract.history_dim:
        hist = hist.reshape(n, contract.history_length, contract.num_obs)
    if hist.ndim != 3 or tuple(hist.shape[1:]) != (contract.history_length, contract.num_obs):
        raise ValueError(f"bank history must be [N,5,45] or [N,225], got {hist.shape}")
    if hist.shape[0] != n:
        raise ValueError("bank obs/history row counts differ")
    arrays["history"] = hist
    if "commands" not in arrays:
        # Official observation command slice is scaled; this fallback is only
        # an explicit inference and is recorded as such in metadata.
        arrays["commands"] = arrays["obs"][:, 6:9] / np.asarray(contract.command_scale, dtype=np.float32)
        metadata.setdefault("commands_source", "inferred_from_obs_scaled_slice")
    arrays["commands"] = np.asarray(arrays["commands"], dtype=np.float32).reshape(n, 3)
    defaults = {
        "terrain_id": np.full(n, -1, dtype=np.int64),
        "terrain_level": np.full(n, -1, dtype=np.int64),
        "command_id": np.full(n, -1, dtype=np.int64),
        "episode_id": np.full(n, -1, dtype=np.int64),
        "episode_step": np.full(n, -1, dtype=np.int64),
        "control_step": np.arange(n, dtype=np.int64),
        "done": np.zeros(n, dtype=np.bool_),
        "run_id": np.full(n, "unknown", dtype="U32"),
        "terrain_name": np.full(n, "unknown", dtype="U32"),
        "route_mode": np.full(n, "offline", dtype="U32"),
    }
    for key, value in defaults.items():
        if key not in arrays:
            arrays[key] = value
        elif np.asarray(arrays[key]).shape[0] != n:
            raise ValueError(f"bank metadata {key!r} has wrong row count")
    metadata.setdefault("teacher_oracle_available", False)
    metadata.setdefault("privileged_obs_source", "unavailable_not_fabricated")
    metadata.setdefault("history_semantics", "zero_on_done_then_append_current_obs")
    return arrays, metadata


def _safe_json(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe_json(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _ridge_predict(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    X_train = np.asarray(X_train, dtype=np.float64)
    Y_train = np.asarray(Y_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    mean = X_train.mean(axis=0)
    std = np.where(X_train.std(axis=0) < 1e-12, 1.0, X_train.std(axis=0))
    Xt = (X_train - mean) / std
    Xv = (X_test - mean) / std
    Xt = np.concatenate([Xt, np.ones((len(Xt), 1))], axis=1)
    Xv = np.concatenate([Xv, np.ones((len(Xv), 1))], axis=1)
    reg = np.eye(Xt.shape[1], dtype=np.float64) * alpha
    reg[-1, -1] = 0.0
    try:
        w = np.linalg.solve(Xt.T @ Xt + reg, Xt.T @ Y_train)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(Xt.T @ Xt + reg, Xt.T @ Y_train, rcond=None)[0]
    return Xv @ w


def _linear_probe(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, seed: int) -> Dict[str, Any]:
    X, Y, groups = np.asarray(X, dtype=np.float64), np.asarray(Y), np.asarray(groups)
    if len(X) < 10 or len(np.unique(groups)) < 2:
        return {"available": False, "reason": "fewer than two groups for deterministic OOF split"}
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    unique = unique[rng.permutation(len(unique))]
    test_groups = set(unique[::2].tolist())
    test = np.asarray([g in test_groups for g in groups], dtype=bool)
    if test.all() or (~test).all():
        test[:] = False
        test[np.arange(len(test)) % 5 == 0] = True
    pred = _ridge_predict(X[~test], Y[~test], X[test])
    y = np.asarray(Y[test], dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
        pred = pred[:, None]
    denom = np.sum((y - y.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2 = 1.0 - np.sum((y - pred) ** 2, axis=0) / np.maximum(denom, 1e-12)
    return {"available": True, "r2_per_target": r2.tolist(), "mean_r2": float(np.mean(r2)), "n_train": int((~test).sum()), "n_test": int(test.sum()), "split": "group_half"}


def _classification_probe(
    X: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
    *,
    target: str,
) -> Dict[str, Any]:
    """Deterministic group-aware, class-stratified classification metrics.

    Each group must belong to exactly one class.  This is intentionally a
    single deterministic train/test split rather than a row-level random
    split: command/terrain rollout rows are temporally correlated and row
    leakage would make a probe look stronger than it is.  A nearest-centroid
    classifier keeps this CPU-only analysis dependency-free; the reported
    ordinary/balanced accuracy is therefore a held-out classifier result, not
    a continuous-command regression score.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1)
    groups = np.asarray(groups).reshape(-1)
    if X.ndim != 2 or len(X) != len(labels) or len(labels) != len(groups):
        return {"available": False, "target": target, "reason": "feature, label and group row counts differ"}
    if len(labels) == 0:
        return {"available": False, "target": target, "reason": "no rows available"}
    if np.issubdtype(labels.dtype, np.number):
        label_valid = np.isfinite(labels.astype(np.float64)) & (labels >= 0)
    else:
        label_valid = np.asarray([str(x).strip() not in ("", "unknown", "-1") for x in labels], dtype=bool)
    finite = np.isfinite(X).all(axis=1)
    valid = label_valid & finite
    if not np.any(valid):
        return {"available": False, "target": target, "reason": f"{target} has no valid labelled finite rows"}
    X, labels, groups = X[valid], labels[valid], groups[valid]
    classes = np.unique(labels)
    if len(classes) < 2:
        return {
            "available": False,
            "target": target,
            "reason": f"{target} has one class or fewer after filtering; need at least two classes",
            "classes": classes.tolist(),
        }

    # A group crossing classes cannot be split without violating the group
    # boundary; fail loudly instead of assigning it by a majority heuristic.
    group_keys = np.asarray([str(g) for g in groups], dtype="U256")
    group_classes: Dict[str, Any] = {}
    for group in np.unique(group_keys):
        values = np.unique(labels[group_keys == group])
        if len(values) != 1:
            return {
                "available": False,
                "target": target,
                "reason": f"group {group!r} contains multiple {target} classes; group-stratified split undefined",
            }
        group_classes[str(group)] = values[0]

    rng = np.random.default_rng(int(seed))
    test_groups = set()
    groups_per_class: Dict[Any, List[str]] = {}
    for cls in classes:
        cls_groups = sorted(
            [g for g, value in group_classes.items() if value == cls],
            key=str,
        )
        groups_per_class[cls.item() if isinstance(cls, np.generic) else cls] = cls_groups
        if len(cls_groups) < 2:
            return {
                "available": False,
                "target": target,
                "reason": (
                    f"class {cls!r} has {len(cls_groups)} group(s); need at least two "
                    "groups so both train and test contain the class"
                ),
                "class_group_counts": {str(c): len(v) for c, v in groups_per_class.items()},
            }
        shuffled = list(rng.permutation(np.asarray(cls_groups, dtype=object)))
        n_test = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * 0.5))))
        test_groups.update(str(g) for g in shuffled[:n_test])

    test_mask = np.asarray([g in test_groups for g in group_keys], dtype=bool)
    train_mask = ~test_mask
    if not np.any(train_mask) or not np.any(test_mask):
        return {"available": False, "target": target, "reason": "stratified split produced an empty train or test set"}
    train_classes, test_classes = set(labels[train_mask].tolist()), set(labels[test_mask].tolist())
    missing_train = [c.item() if isinstance(c, np.generic) else c for c in classes if c not in train_classes]
    missing_test = [c.item() if isinstance(c, np.generic) else c for c in classes if c not in test_classes]
    if missing_train or missing_test:
        return {
            "available": False,
            "target": target,
            "reason": f"class missing from train/test after split: train={missing_train}, test={missing_test}",
        }

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = labels[train_mask], labels[test_mask]
    mean = X_train.mean(axis=0)
    std = np.where(X_train.std(axis=0) < 1e-12, 1.0, X_train.std(axis=0))
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    centroids = np.stack([X_train[y_train == cls].mean(axis=0) for cls in classes])
    distances = np.sum((X_test[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    prediction = classes[np.argmin(distances, axis=1)]
    class_to_index = {cls.item() if isinstance(cls, np.generic) else cls: i for i, cls in enumerate(classes)}
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for truth, pred in zip(y_test, prediction):
        confusion[class_to_index[truth.item() if isinstance(truth, np.generic) else truth],
                  class_to_index[pred.item() if isinstance(pred, np.generic) else pred]] += 1
    recalls = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    test_counts = {str(cls.item() if isinstance(cls, np.generic) else cls): int(np.sum(y_test == cls)) for cls in classes}
    train_counts = {str(cls.item() if isinstance(cls, np.generic) else cls): int(np.sum(y_train == cls)) for cls in classes}
    ordinary_accuracy = float(np.mean(y_test == prediction))
    return {
        "available": True,
        "target": target,
        "classifier": "nearest_centroid",
        "classes": [cls.item() if isinstance(cls, np.generic) else cls for cls in classes],
        "ordinary_accuracy": ordinary_accuracy,
        "accuracy": ordinary_accuracy,
        "balanced_accuracy": float(np.mean(recalls)),
        "majority_baseline": float(max(test_counts.values()) / len(y_test)),
        "confusion_matrix": confusion.tolist(),
        "class_counts": {"train": train_counts, "test": test_counts},
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_train_groups": int(len(set(group_keys[train_mask].tolist()))),
        "n_test_groups": int(len(set(group_keys[test_mask].tolist()))),
        "split": "deterministic_group_class_stratified",
        "split_seed": int(seed),
        "group_disjoint": bool(set(group_keys[train_mask]).isdisjoint(set(group_keys[test_mask]))),
        "test_fraction": 0.5,
    }


def _terrain_probe(X: np.ndarray, labels: np.ndarray, groups: np.ndarray, seed: int) -> Dict[str, Any]:
    """Backward-compatible terrain probe name using the new classifier contract."""
    return _classification_probe(X, labels, groups, seed, target="terrain_id")


def compute_specialization_metrics(
    arrays: Mapping[str, np.ndarray],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    raw_gate = np.asarray(arrays["gate"], dtype=np.float64)
    raw_experts = np.asarray(arrays["expert_outputs"], dtype=np.float64)
    raw_learned_latent = np.asarray(arrays["normalized_mixed_latent"], dtype=np.float64)
    raw_learned_action = np.asarray(arrays["learned_action"], dtype=np.float64)
    n = int(raw_experts.shape[0])
    finite_keys = (
        "gate",
        "expert_outputs",
        "normalized_mixed_latent",
        "learned_action",
        "single_expert_action",
        "action_uniform",
        "action_shuffled",
        "action_top1",
        "commands",
    )
    numeric_finite_rows = np.ones(n, dtype=bool)
    for key in finite_keys:
        value = np.asarray(arrays[key])
        if value.shape[0] != n:
            raise ValueError(f"metric array {key!r} has inconsistent first dimension {value.shape}")
        numeric_finite_rows &= np.isfinite(value.reshape(n, -1)).all(axis=1)
    valid_rows = numeric_finite_rows.copy()
    unbounded_obs_rows = np.zeros(n, dtype=bool)
    if "obs" in arrays:
        obs = np.asarray(arrays["obs"], dtype=np.float64)
        if obs.shape[0] != n:
            raise ValueError(f"metric array 'obs' has inconsistent first dimension {obs.shape}")
        obs_abs = np.max(np.abs(obs.reshape(n, -1)), axis=1)
        unbounded_obs_rows = numeric_finite_rows & (obs_abs > SPECIALIZATION_MAX_ABS_OBS)
        valid_rows &= ~unbounded_obs_rows
    # A route can become physically unstable before MuJoCo's height-based fall
    # guard fires, leaving inf/NaN observations and neural outputs in the final
    # row.  Exclude only those rows from offline specialization statistics;
    # closed-loop survival/fall metrics still use every recorded route row.
    gate = raw_gate[valid_rows]
    experts = raw_experts[valid_rows]
    learned_latent = raw_learned_latent[valid_rows]
    learned_action = raw_learned_action[valid_rows]
    metric_arrays = {
        key: np.asarray(arrays[key])[valid_rows]
        for key in finite_keys
    }
    n_finite = int(numeric_finite_rows.sum())
    n_valid = int(valid_rows.sum())
    if n_valid == 0:
        raise ValueError("no finite rows remain for specialization metrics")
    _, k, d = experts.shape
    eps = 1e-12
    gate_entropy = -np.sum(gate * np.log(np.maximum(gate, eps)), axis=1)
    expert_norm = np.linalg.norm(experts, axis=-1)
    normed_expert = experts / np.maximum(expert_norm[..., None], eps)
    expert_cos = np.einsum("nkd,njd->nkj", normed_expert, normed_expert).mean(axis=0)
    pairwise_delta = experts[:, :, None, :] - experts[:, None, :, :]
    expert_dist = np.linalg.norm(pairwise_delta, axis=-1).mean(axis=0)
    single = np.asarray(metric_arrays["single_expert_action"], dtype=np.float64)
    action_pairwise = np.mean((single[:, :, None, :] - single[:, None, :, :]) ** 2, axis=(0, 3))
    action_mse = {}
    for name in ("uniform", "shuffled", "top1"):
        action_mse[name] = float(np.mean((learned_action - metric_arrays[f"action_{name}"]) ** 2))
    groups = np.asarray(arrays.get("run_id", np.arange(n)))[valid_rows]
    command_id = np.asarray(arrays.get("command_id", np.full(n, -1)))[valid_rows]
    terrain_id = np.asarray(arrays.get("terrain_id", np.full(n, -1)))[valid_rows]
    commands = metric_arrays["commands"]
    # Continuous command regression remains a useful auxiliary diagnostic, but
    # it is deliberately separate from the required command_id classifier.
    command_regression = {
        "gate": _linear_probe(gate, commands, groups, seed),
        "normalized_mixed_latent": _linear_probe(learned_latent, commands, groups, seed + 1),
    }
    probe = {
        "command_id_from_gate": _classification_probe(gate, command_id, groups, seed, target="command_id"),
        "command_id_from_normalized_mixed_latent": _classification_probe(
            learned_latent, command_id, groups, seed + 1, target="command_id"
        ),
        "terrain_id_from_gate": _classification_probe(gate, terrain_id, groups, seed + 2, target="terrain_id"),
        "terrain_id_from_normalized_mixed_latent": _classification_probe(
            learned_latent, terrain_id, groups, seed + 3, target="terrain_id"
        ),
        "continuous_command_regression": command_regression,
        # Keep the old names as aliases for consumers of the previous probe,
        # but their values are explicitly continuous regressions, not class
        # accuracy and not causal closed-loop performance.
        "command_from_gate": command_regression["gate"],
        "command_from_normalized_mixed_latent": command_regression["normalized_mixed_latent"],
    }
    route_mode = np.asarray(arrays.get("route_mode", np.full(n, "offline")))[valid_rows]
    return {
        "schema_version": 2,
        "n_samples": int(n),
        "n_finite_samples": n_finite,
        "n_valid_samples": n_valid,
        "n_excluded_nonfinite": int(n - n_finite),
        "n_excluded_unbounded_observation": int(unbounded_obs_rows.sum()),
        "n_excluded_invalid": int(n - n_valid),
        "finite_row_filter": f"all_required_numeric_metric_arrays_finite_and_max_abs_obs_le_{SPECIALIZATION_MAX_ABS_OBS:g}",
        "expert_num": int(k),
        "latent_dim": int(d),
        "gate": {
            "entropy_mean": float(gate_entropy.mean()),
            "entropy_std": float(gate_entropy.std()),
            "effective_experts_mean": float(np.exp(gate_entropy).mean()),
            "effective_experts_std": float(np.exp(gate_entropy).std()),
            "mean_max_gate": float(gate.max(axis=1).mean()),
            "marginal_usage": gate.mean(axis=0).tolist(),
        },
        "expert_latent": {
            "norm_mean": expert_norm.mean(axis=0).tolist(),
            "norm_std": expert_norm.std(axis=0).tolist(),
            "pairwise_cosine_mean": expert_cos.tolist(),
            "pairwise_l2_mean": expert_dist.tolist(),
        },
        "action_interventions": {
            "learned_vs_uniform_mse": action_mse["uniform"],
            "learned_vs_shuffled_mse": action_mse["shuffled"],
            "learned_vs_top1_mse": action_mse["top1"],
            "single_expert_pairwise_mse": action_pairwise.tolist(),
            "single_expert_action_mean_norm": np.linalg.norm(single, axis=-1).mean(axis=0).tolist(),
            "semantics": "same_state_action_intervention; not a closed-loop performance metric",
        },
        "latent_norm": {
            "normalized_mixed_mean": float(np.linalg.norm(learned_latent, axis=1).mean()),
            "normalized_mixed_std": float(np.linalg.norm(learned_latent, axis=1).std()),
        },
        "probes": probe,
        "route_mode_counts": {
            str(mode): int(np.sum(route_mode == mode)) for mode in np.unique(route_mode)
        },
        "teacher_oracle": {
            "available": bool((metadata or {}).get("teacher_available", False)),
            "reason": "privileged observation is not present in the MuJoCo/source deployment bank; no synthetic teacher was created",
        },
    }


def _iter_batches(n: int, batch_size: int) -> Iterator[slice]:
    for start in range(0, n, batch_size):
        yield slice(start, min(start + batch_size, n))


def analyze_fixed_bank(
    adapter: UpstreamMoECTSAdapter,
    bank: Mapping[str, np.ndarray],
    *,
    seed: int = 0,
    batch_size: int = 4096,
) -> Dict[str, np.ndarray]:
    n = len(bank["obs"])
    component_outputs: Dict[str, List[np.ndarray]] = {}
    for sl in _iter_batches(n, batch_size):
        comp = adapter.forward_components(
            torch.from_numpy(np.asarray(bank["obs"][sl], dtype=np.float32)),
            torch.from_numpy(np.asarray(bank["history"][sl], dtype=np.float32)),
        )
        for key, value in comp.items():
            component_outputs.setdefault(key, []).append(value.detach().cpu().numpy())
    components = {key: np.concatenate(vals, axis=0) for key, vals in component_outputs.items()}

    # Shuffle gates over the complete fixed bank, not independently inside
    # each inference batch.  Per-batch shuffling would leak row locality into
    # the ablation whenever N > batch_size.
    permutation = np.random.default_rng(int(seed)).permutation(n)
    intervention_outputs: Dict[str, List[np.ndarray]] = {}
    for sl in _iter_batches(n, batch_size):
        comp = {
            key: torch.from_numpy(components[key][sl])
            for key in ("obs", "gate", "expert_outputs")
        }
        shuffled = torch.from_numpy(components["gate"][permutation[sl]])
        inter = adapter.intervention_actions(comp, seed=seed, shuffled_gate=shuffled)
        for key, value in inter.items():
            intervention_outputs.setdefault(key, []).append(value.detach().cpu().numpy())
    return {
        **components,
        **{key: np.concatenate(vals, axis=0) for key, vals in intervention_outputs.items()},
    }


def write_offline_result(
    adapter: UpstreamMoECTSAdapter,
    bank_path: os.PathLike[str] | str,
    out_dir: os.PathLike[str] | str,
    *,
    seed: int = 0,
    batch_size: int = 4096,
) -> Dict[str, Any]:
    bank, bank_meta = load_fixed_bank(bank_path)
    forward = analyze_fixed_bank(adapter, bank, seed=seed, batch_size=batch_size)
    arrays = {**bank, **forward}
    metrics = compute_specialization_metrics(
        arrays,
        metadata={**bank_meta, "teacher_available": adapter.loaded.teacher_available},
        seed=seed,
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    serializable_arrays = {k: np.asarray(v) for k, v in arrays.items() if k != "privileged_obs"}
    serializable_arrays["current_obs"] = np.asarray(arrays["obs"])
    serializable_arrays["metadata_json"] = np.asarray(
        json.dumps(
            {
                "schema_version": 2,
                "mode": "offline",
                "route_modes": ["offline_same_state_interventions"],
                "offline_only_modes": list(OFFLINE_ONLY_ROUTE_MODES),
                "intervention_semantics": "fixed-bank same-state actions; shuffled is not causal closed-loop",
                "checkpoint": adapter.loaded.path,
                "checkpoint_sha256": adapter.loaded.sha256,
                "checkpoint_schema": adapter.loaded.schema,
                "checkpoint_provenance": _safe_json(adapter.loaded.provenance),
                "bank": str(Path(bank_path).resolve()),
                "bank_sha256": sha256_file(bank_path),
                "seed": int(seed),
                "contract": CONTRACT.as_dict(),
                "teacher_available": adapter.loaded.teacher_available,
                "teacher_privileged_obs": "unavailable_not_fabricated",
                "bank_metadata": _safe_json(bank_meta),
            },
            sort_keys=True,
        )
    )
    np.savez_compressed(out / "probe.npz", **serializable_arrays)
    result = {
        "mode": "offline",
        "route_modes": ["offline_same_state_interventions"],
        "offline_only_modes": list(OFFLINE_ONLY_ROUTE_MODES),
        "checkpoint": adapter.loaded.path,
        "checkpoint_sha256": adapter.loaded.sha256,
        "checkpoint_schema": adapter.loaded.schema,
        "checkpoint_provenance": _safe_json(adapter.loaded.provenance),
        "bank": str(Path(bank_path).resolve()),
        "bank_sha256": sha256_file(bank_path),
        "seed": int(seed),
        "contract": CONTRACT.as_dict(),
        "teacher_available": adapter.loaded.teacher_available,
        "privileged_obs_source": "unavailable_not_fabricated",
        "metrics": metrics,
    }
    (out / "metrics.json").write_text(json.dumps(_safe_json(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# Headless MuJoCo closed-loop collector
# ---------------------------------------------------------------------------


TERRAIN_ASSET_MAP = {
    "flat": ("flat.xml", True, "official flat scene"),
    "stairs": ("stairs.xml", True, "official stairs scene"),
    # The reference checkout has no standalone wave/obstacle XML.  These are
    # deliberately labelled proxies, never silently presented as source terrain.
    "wave": ("cross_slope.xml", False, "cross_slope proxy; exact upstream wave asset unavailable"),
    "obstacle": ("race_track.xml", False, "race_track obstacle proxy; exact upstream obstacle asset unavailable"),
}


def resolve_terrain_asset(reference_root: os.PathLike[str] | str, terrain: str) -> Dict[str, Any]:
    if terrain not in TERRAIN_ASSET_MAP:
        raise ValueError(f"unknown terrain {terrain!r}; choose {sorted(TERRAIN_ASSET_MAP)}")
    filename, exact, note = TERRAIN_ASSET_MAP[terrain]
    path = Path(reference_root) / "resources" / "robots" / "go2" / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"name": terrain, "path": str(path.resolve()), "exact": exact, "note": note, "sha256": sha256_file(path)}


def _mujoco_joint_indices(model: Any, joint_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    import mujoco  # type: ignore
    qpos_indices, qvel_indices = [], []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint {name!r} missing from MuJoCo model")
        qpos_indices.append(int(model.jnt_qposadr[jid]))
        qvel_indices.append(int(model.jnt_dofadr[jid]))
    return np.asarray(qpos_indices, dtype=np.int64), np.asarray(qvel_indices, dtype=np.int64)


def _mujoco_actuator_joint_indices(model: Any, joint_names: Sequence[str]) -> np.ndarray:
    import mujoco  # type: ignore
    by_name = {name: i for i, name in enumerate(joint_names)}
    out = []
    for actuator_id in range(int(model.nu)):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name not in by_name:
            raise ValueError(f"actuator {actuator_id} targets unknown joint {name!r}")
        out.append(by_name[name])
    return np.asarray(out, dtype=np.int64)


def _initial_mujoco_state(model: Any, data: Any) -> None:
    import mujoco  # type: ignore
    mujoco.mj_resetData(model, data)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _normalize_route_modes(route_modes: Optional[Sequence[str]]) -> Tuple[str, ...]:
    """Validate and canonicalize closed-loop routes.

    ``expert0`` is accepted as a short CLI alias, while metadata always uses
    the unambiguous ``fixed_expert_0`` spelling.  ``shuffled`` is intentionally
    rejected here: it is a fixed-bank cross-row intervention with no reliable
    single-environment causal semantics.
    """
    requested = CLOSED_LOOP_ROUTE_MODES if route_modes is None else tuple(route_modes)
    normalized: List[str] = []
    for raw in requested:
        name = str(raw).strip().lower()
        match = re.fullmatch(r"(?:fixed_expert_|expert)([0-7])", name)
        if match:
            name = f"fixed_expert_{int(match.group(1))}"
        if name in OFFLINE_ONLY_ROUTE_MODES:
            raise ValueError(
                "route_mode='shuffled' is offline-only; omit it from --route-modes "
                "and use --mode offline for the fixed-bank intervention"
            )
        if name not in CLOSED_LOOP_ROUTE_MODES:
            raise ValueError(
                f"unknown route_mode {raw!r}; expected comma-separated values from "
                f"{CLOSED_LOOP_ROUTE_MODES}"
            )
        if name not in normalized:
            normalized.append(name)
    if not normalized:
        raise ValueError("at least one closed-loop route_mode is required")
    return tuple(normalized)


def _closed_loop_metric_summary(rollout_summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate physically measured route metrics without mixing route states."""
    by_mode: Dict[str, List[Mapping[str, Any]]] = {}
    for summary in rollout_summaries:
        by_mode.setdefault(str(summary["route_mode"]), []).append(summary)
    result: Dict[str, Any] = {}
    for route_mode, rows in by_mode.items():
        tracking = np.asarray([float(row["tracking_error_mean"]) for row in rows], dtype=np.float64)
        achieved = np.asarray([row["achieved_command_velocity_mean"] for row in rows], dtype=np.float64)
        result[route_mode] = {
            "n_rollouts": int(len(rows)),
            "fall_rate": float(np.mean([bool(row["fell"]) for row in rows])),
            "survival_duration_s_mean": float(np.mean([float(row["survival_duration_s"]) for row in rows])),
            "survival_duration_s_min": float(np.min([float(row["survival_duration_s"]) for row in rows])),
            "tracking_error_mean": float(np.mean(tracking)),
            "tracking_error_std": float(np.std(tracking)),
            "achieved_command_velocity_mean": achieved.mean(axis=0).tolist(),
            "rollouts": [_safe_json(row) for row in rows],
        }
    return result


def run_closed_loop(
    adapter: UpstreamMoECTSAdapter,
    *,
    reference_root: os.PathLike[str] | str,
    out_dir: os.PathLike[str] | str,
    terrains: Sequence[str] = ("flat",),
    commands: np.ndarray = PAPER_COMMANDS,
    command_labels: Sequence[str] = PAPER_COMMAND_LABELS,
    duration_s: float = 5.0,
    simulation_dt: float = 0.002,
    control_decimation: int = 10,
    seed: int = 0,
    max_rollouts: Optional[int] = None,
    route_modes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Collect independent, deterministic route rollouts in headless MuJoCo.

    ``max_rollouts`` is a per-route cap (not a global cap), so a short smoke
    with ``max_rollouts=1`` still runs one terrain/command rollout for every
    requested route.  A fallen route is terminated immediately after its final
    recorded control step; no action or state is collected from the fallen
    state for the remaining protocol duration.
    """
    import mujoco  # type: ignore

    if len(commands) != len(command_labels):
        raise ValueError("command labels and command rows differ")
    route_modes = _normalize_route_modes(route_modes)
    if max_rollouts is not None and int(max_rollouts) < 1:
        raise ValueError("max_rollouts must be positive when provided")
    np.random.seed(int(seed))
    contract = CONTRACT
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records: Dict[str, List[np.ndarray]] = {}
    terrain_meta = [resolve_terrain_asset(reference_root, name) for name in terrains]
    rollout_count = 0
    rollout_summaries: List[Dict[str, Any]] = []
    sim_steps = int(round(float(duration_s) / float(simulation_dt)))
    if sim_steps < control_decimation:
        raise ValueError("duration must contain at least one policy step")
    for route_mode in route_modes:
        route_rollouts = 0
        for terrain_index, terrain_info in enumerate(terrain_meta):
            if max_rollouts is not None and route_rollouts >= int(max_rollouts):
                break
            # Build a fresh model/data pair for every route.  In addition to
            # resetting qpos/qvel, this keeps accidental simulator state from
            # one route from becoming an intervention confounder.
            model = mujoco.MjModel.from_xml_path(terrain_info["path"])
            model.opt.timestep = float(simulation_dt)
            data = mujoco.MjData(model)
            qpos_idx, qvel_idx = _mujoco_joint_indices(model, UPSTREAM_MODEL_JOINT_NAMES)
            actuator_joint_idx = _mujoco_actuator_joint_indices(model, UPSTREAM_MODEL_JOINT_NAMES)
            obs_adapter = MujocoObservationAdapter(
                joint_order=JointOrderAdapter(UPSTREAM_MODEL_JOINT_NAMES),
                default_angles_model=UPSTREAM_DEFAULT_ANGLES.copy(),
            )
            kps = np.full(contract.num_actions, 20.0, dtype=np.float32)
            kds = np.full(contract.num_actions, 0.5, dtype=np.float32)
            for command_id, command in enumerate(np.asarray(commands, dtype=np.float32)):
                if max_rollouts is not None and route_rollouts >= int(max_rollouts):
                    break
                _initial_mujoco_state(model, data)
                target_q_mj = obs_adapter.default_angles_mujoco.copy()
                previous_action = np.zeros(contract.num_actions, dtype=np.float32)
                history = SourceHistoryCollector(1)
                rows = 0
                done = False
                tracking_errors: List[np.ndarray] = []
                achieved_velocities: List[np.ndarray] = []
                elapsed_s = 0.0
                for sim_step in range(sim_steps):
                    qpos_mj = np.asarray(data.qpos[qpos_idx], dtype=np.float32)
                    qvel_mj = np.asarray(data.qvel[qvel_idx], dtype=np.float32)
                    tau_mj = (target_q_mj - qpos_mj) * kps - qvel_mj * kds
                    data.ctrl[:] = tau_mj[actuator_joint_idx]
                    mujoco.mj_step(model, data)
                    if (sim_step + 1) % control_decimation != 0:
                        continue
                    qpos_mj = np.asarray(data.qpos[qpos_idx], dtype=np.float32)
                    qvel_mj = np.asarray(data.qvel[qvel_idx], dtype=np.float32)
                    obs = obs_adapter.observation(
                        quaternion=data.qpos[3:7],
                        world_linear_velocity=data.qvel[:3],
                        world_angular_velocity=data.qvel[3:6],
                        command=command,
                        qpos_mujoco=qpos_mj,
                        qvel_mujoco=qvel_mj,
                        previous_action_model=previous_action,
                    )
                    history_state = (
                        history.initial(torch.from_numpy(obs[None]))
                        if rows == 0
                        else history.append(torch.from_numpy(obs[None]))
                    )
                    comp = adapter.forward_components(torch.from_numpy(obs[None]), history_state)
                    inter = adapter.intervention_actions(
                        comp,
                        seed=seed + rows + command_id * 100003 + route_rollouts * 1000003,
                    )
                    route_action = adapter.route_action(inter, route_mode)
                    route_action_np = route_action[0].cpu().numpy().astype(np.float32)
                    target_q_mj = obs_adapter.target_q_mujoco(route_action_np)
                    body_linear = quat_rotate_inverse(data.qpos[3:7], data.qvel[:3])
                    body_angular = quat_rotate_inverse(data.qpos[3:7], data.qvel[3:6])
                    achieved_command_velocity = np.asarray(
                        [body_linear[0], body_linear[1], body_angular[2]], dtype=np.float32
                    )
                    tracking_error = achieved_command_velocity - np.asarray(command, dtype=np.float32)
                    done = bool(float(data.qpos[2]) < 0.15 or not np.isfinite(data.qpos[2]))
                    elapsed_s = float((sim_step + 1) * simulation_dt)
                    tracking_errors.append(tracking_error.astype(np.float64))
                    achieved_velocities.append(achieved_command_velocity.astype(np.float64))
                    payload = {
                        "obs": obs[None],
                        "history": history_state.cpu().numpy(),
                        "gate": comp["gate"].cpu().numpy(),
                        "expert_outputs": comp["expert_outputs"].cpu().numpy(),
                        "raw_weighted_latent": comp["raw_weighted_latent"].cpu().numpy(),
                        "normalized_mixed_latent": comp["normalized_mixed_latent"].cpu().numpy(),
                        "learned_action": comp["learned_action"].cpu().numpy(),
                        "single_expert_action": inter["single_expert_action"].cpu().numpy(),
                        "route_action": route_action.cpu().numpy(),
                        "commands": np.asarray(command, dtype=np.float32).reshape(1, 3),
                        "achieved_command_velocity": achieved_command_velocity.reshape(1, 3),
                        "tracking_error": tracking_error.reshape(1, 3),
                        "tracking_error_norm": np.asarray([np.linalg.norm(tracking_error)], dtype=np.float32),
                        "terrain_id": np.asarray([terrain_index], dtype=np.int64),
                        "terrain_level": np.asarray([-1], dtype=np.int64),
                        "terrain_name": np.asarray([terrain_info["name"]]),
                        "command_id": np.asarray([command_id], dtype=np.int64),
                        "command_label": np.asarray([command_labels[command_id]]),
                        "route_mode": np.asarray([route_mode]),
                        "run_id": np.asarray(
                            [f"{route_mode}:{terrain_info['name']}:cmd{command_id}"], dtype="U96"
                        ),
                        "episode_id": np.asarray([0], dtype=np.int64),
                        "episode_step": np.asarray([rows], dtype=np.int64),
                        "control_step": np.asarray([rows], dtype=np.int64),
                        "elapsed_s": np.asarray([elapsed_s], dtype=np.float32),
                        "done": np.asarray([done], dtype=np.bool_),
                        "fall": np.asarray([done], dtype=np.bool_),
                    }
                    # These are same-state offline interventions recorded for
                    # diagnosis; only route_action is applied to MuJoCo.
                    for name in ("uniform", "shuffled", "top1"):
                        for prefix in ("gate", "raw_latent", "normalized_latent", "action"):
                            payload[f"{prefix}_{name}"] = inter[f"{prefix}_{name}"].cpu().numpy()
                    for expert_index in range(contract.expert_num):
                        payload[f"action_fixed_expert_{expert_index}"] = inter[
                            f"action_fixed_expert_{expert_index}"
                        ].cpu().numpy()
                    for key, value in payload.items():
                        records.setdefault(key, []).append(np.asarray(value))
                    previous_action = route_action_np
                    rows += 1
                    if done:
                        # Termination is the paper-clean boundary: never keep
                        # observing or acting from a fallen state, and do not
                        # silently start a second episode under episode_id=0.
                        break
                if rows == 0:
                    raise RuntimeError("closed-loop rollout produced no policy rows")
                rollout_summaries.append(
                    {
                        "route_mode": route_mode,
                        "terrain_id": terrain_index,
                        "terrain_name": terrain_info["name"],
                        "command_id": command_id,
                        "command_label": command_labels[command_id],
                        "episode_id": 0,
                        "n_steps": rows,
                        "fell": bool(done),
                        "survival_duration_s": elapsed_s,
                        "tracking_error_mean": float(
                            np.mean([np.linalg.norm(error) for error in tracking_errors])
                        ),
                        "achieved_command_velocity_mean": np.mean(achieved_velocities, axis=0).tolist(),
                    }
                )
                route_rollouts += 1
                rollout_count += 1
    if not records:
        raise RuntimeError("closed-loop collector produced no rows")
    arrays = {key: np.concatenate(values, axis=0) for key, values in records.items()}
    # Each physical closed-loop step is evaluated with a batch of one, so the
    # per-step intervention above necessarily leaves ``shuffled`` unchanged.
    # Recompute that intervention over the complete recorded bank after all
    # routes are collected; this preserves its intended offline cross-row
    # semantics without ever driving MuJoCo with a shuffled action.
    n_rows = len(arrays["gate"])
    permutation = np.random.default_rng(int(seed)).permutation(n_rows)
    shuffled_outputs: Dict[str, List[np.ndarray]] = {}
    for sl in _iter_batches(n_rows, 4096):
        components = {
            key: torch.from_numpy(np.asarray(arrays[key][sl], dtype=np.float32))
            for key in ("obs", "gate", "expert_outputs")
        }
        shuffled_gate = torch.from_numpy(np.asarray(arrays["gate"][permutation[sl]], dtype=np.float32))
        interventions = adapter.intervention_actions(
            components,
            seed=seed,
            shuffled_gate=shuffled_gate,
        )
        for key in ("gate_shuffled", "raw_latent_shuffled", "normalized_latent_shuffled", "action_shuffled"):
            shuffled_outputs.setdefault(key, []).append(interventions[key].cpu().numpy())
    for key, values in shuffled_outputs.items():
        arrays[key] = np.concatenate(values, axis=0)
    arrays["current_obs"] = np.asarray(arrays["obs"])
    # Keep history as [N,5,45] and action tensors as the requested public ABI.
    arrays["metadata_json"] = np.asarray(
        json.dumps(
            {
                "schema_version": 2,
                "mode": "closed_loop",
                "route_modes": list(route_modes),
                "offline_only_modes": list(OFFLINE_ONLY_ROUTE_MODES),
                "seed": int(seed),
                "duration_s": float(duration_s),
                "simulation_dt": float(simulation_dt),
                "control_decimation": int(control_decimation),
                "fixed_protocol": "5s forward/command bank; no viewer; route_action drives MuJoCo",
                "route_action_semantics": {
                    "learned": "policy weighted latent/action",
                    "uniform": "uniform expert weights, physically rolled out",
                    "top1": "per-state argmax expert, physically rolled out",
                    "fixed_expert_0_to_7": "one-hot expert action, each physically rolled out separately",
                    "shuffled": "offline-only cross-row gate assignment over all recorded rows; no single-env causal semantics",
                },
                "shuffled_intervention_seed": int(seed),
                "termination_semantics": "terminate_on_fall; no post-terminal rows; fresh state/history per route",
                "max_rollouts_per_route": None if max_rollouts is None else int(max_rollouts),
                "commands": [np.asarray(c).tolist() for c in commands],
                "command_labels": list(command_labels),
                "terrains": terrain_meta,
                "exact_terrain_assets": [x["name"] for x in terrain_meta if x["exact"]],
                "terrain_proxy_or_blocker": [x for x in terrain_meta if not x["exact"]],
                "checkpoint": adapter.loaded.path,
                "checkpoint_sha256": adapter.loaded.sha256,
                "checkpoint_schema": adapter.loaded.schema,
                "checkpoint_provenance": _safe_json(adapter.loaded.provenance),
                "contract": contract.as_dict(),
                "teacher_available": adapter.loaded.teacher_available,
                "privileged_obs_source": "unavailable_not_fabricated",
                "rollout_summaries": rollout_summaries,
            },
            sort_keys=True,
        )
    )
    np.savez_compressed(out / "probe.npz", **arrays)
    metrics = compute_specialization_metrics(
        arrays,
        metadata={"teacher_available": adapter.loaded.teacher_available},
        seed=seed,
    )
    metrics["closed_loop"] = _closed_loop_metric_summary(rollout_summaries)
    result = {
        "mode": "closed_loop",
        "n_samples": int(len(arrays["obs"])),
        "rollouts": int(rollout_count),
        "route_modes": list(route_modes),
        "offline_only_modes": list(OFFLINE_ONLY_ROUTE_MODES),
        "checkpoint": adapter.loaded.path,
        "checkpoint_sha256": adapter.loaded.sha256,
        "checkpoint_provenance": _safe_json(adapter.loaded.provenance),
        "output": str((out / "probe.npz").resolve()),
        "terrains": terrain_meta,
        "metrics": metrics,
        "closed_loop_metrics": metrics["closed_loop"],
        "rollout_summaries": rollout_summaries,
        "teacher_available": adapter.loaded.teacher_available,
        "privileged_obs_source": "unavailable_not_fabricated",
    }
    (out / "metrics.json").write_text(json.dumps(_safe_json(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_jit_parity(
    adapter: UpstreamMoECTSAdapter,
    checkpoint: os.PathLike[str] | str,
    *,
    seed: int = 0,
    command_line: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare the adapter action against a stateful TorchScript deploy file."""
    path = str(Path(checkpoint).expanduser().resolve())
    try:
        jit_model = torch.jit.load(path, map_location=adapter.device)
    except Exception as exc:
        return {
            "available": False,
            "status": "BLOCKED",
            "reason": f"checkpoint is not loadable TorchScript: {type(exc).__name__}: {exc}",
            "checkpoint": path,
            "checkpoint_sha256": sha256_file(path) if os.path.isfile(path) else None,
            "command": command_line or shlex.join(sys.argv),
        }
    if not hasattr(jit_model, "history"):
        return {
            "available": False,
            "status": "BLOCKED",
            "reason": "TorchScript artifact has no runner-owned history buffer",
            "checkpoint": path,
            "checkpoint_sha256": sha256_file(path),
            "command": command_line or shlex.join(sys.argv),
        }
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    obs = torch.randn((1, CONTRACT.num_obs), generator=generator, dtype=torch.float32, device=adapter.device)
    history = torch.zeros(
        (1, CONTRACT.history_length, CONTRACT.num_obs), dtype=torch.float32, device=adapter.device
    )
    history[:, -1, :] = obs
    with torch.inference_mode():
        jit_model.history.zero_()
        jit_output = jit_model(obs)
        jit_action = jit_output[0] if isinstance(jit_output, (tuple, list)) else jit_output
        adapter_action = adapter.forward_components(obs, history)["learned_action"]
    error = (jit_action - adapter_action).detach().cpu().numpy()
    max_abs = float(np.max(np.abs(error)))
    mean_abs = float(np.mean(np.abs(error)))
    return {
        "available": True,
        "status": "PASS" if max_abs <= 1e-5 else "FAIL",
        "checkpoint": path,
        "checkpoint_sha256": sha256_file(path),
        "n_samples": 1,
        "input_seed": int(seed),
        "history_reset": "zero_then_append_current_observation",
        "max_abs_action_error": max_abs,
        "mean_abs_action_error": mean_abs,
        "tolerance": 1e-5,
        "command": command_line or shlex.join(sys.argv),
        "interpretation": "deploy ABI/action parity only; teacher/critic availability is unchanged",
    }


def write_jit_parity_result(
    adapter: UpstreamMoECTSAdapter,
    checkpoint: os.PathLike[str] | str,
    out_dir: os.PathLike[str] | str,
    *,
    seed: int = 0,
    command_line: Optional[str] = None,
) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "mode": "jit_parity",
        "adapter_schema": adapter.loaded.schema,
        "adapter_teacher_available": adapter.loaded.teacher_available,
        "parity": run_jit_parity(adapter, checkpoint, seed=seed, command_line=command_line),
    }
    (out / "metrics.json").write_text(
        json.dumps(_safe_json(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _default_reference_root() -> Path:
    return Path(__file__).resolve().parents[3] / "go2_rl_gym"


def _default_checkpoint() -> Optional[Path]:
    root = _default_reference_root().parents[0]
    candidates = [
        root / "logs/go2_moects/wty_go2_moe_cts_137k/model_0.pt",
        root / "logs/go2_moects/wty_go2_moe_cts_high_slope_thre_164k/model_0.pt",
    ]
    return next((p for p in candidates if p.is_file()), None)


def _parse_commands(name: str) -> Tuple[np.ndarray, Sequence[str]]:
    if name == "paper6":
        return PAPER_COMMANDS.copy(), PAPER_COMMAND_LABELS
    rows = []
    labels = []
    for idx, item in enumerate(name.split(";")):
        values = [float(v) for v in item.split(",")]
        if len(values) != 3:
            raise ValueError("custom commands must be vx,vy,yaw;vx,vy,yaw")
        rows.append(values)
        labels.append(f"custom_{idx}")
    return np.asarray(rows, dtype=np.float32), labels


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Source-faithful upstream Go2 MoE-CTS specialization probe")
    p.add_argument("--mode", choices=("offline", "closed_loop", "jit_parity"), required=True)
    p.add_argument(
        "--checkpoint",
        default=None,
        help="raw upstream model_*.pt or student/actor-only TorchScript deployment bridge",
    )
    p.add_argument("--bank", default=None, help="fixed .npz/.pt bank for offline mode")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--reference-root", default=None, help="go2_rl_gym reference checkout")
    p.add_argument("--terrains", default="flat", help="comma-separated flat,wave,stairs,obstacle")
    p.add_argument("--commands", default="paper6", help="paper6 or semicolon-separated vx,vy,yaw rows")
    p.add_argument("--duration-s", type=float, default=5.0)
    p.add_argument("--simulation-dt", type=float, default=0.002)
    p.add_argument("--control-decimation", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--max-rollouts", type=int, default=None, help="per-route rollout cap")
    p.add_argument(
        "--route-modes",
        default=",".join(CLOSED_LOOP_ROUTE_MODES),
        help="closed-loop learned,uniform,top1,fixed_expert_0..fixed_expert_7; shuffled is offline-only",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    cli = parse_args(argv)
    checkpoint = cli.checkpoint or (str(_default_checkpoint()) if _default_checkpoint() else None)
    if checkpoint is None:
        raise SystemExit("--checkpoint is required; no local raw model_0.pt bridge found")
    adapter = UpstreamMoECTSAdapter.from_checkpoint(checkpoint)
    if cli.mode == "offline":
        if not cli.bank:
            raise SystemExit("--bank is required for --mode offline")
        result = write_offline_result(adapter, cli.bank, cli.out, seed=cli.seed, batch_size=cli.batch_size)
    elif cli.mode == "jit_parity":
        result = write_jit_parity_result(
            adapter,
            checkpoint,
            cli.out,
            seed=cli.seed,
            command_line=shlex.join(sys.argv),
        )
    else:
        reference_root = Path(cli.reference_root) if cli.reference_root else _default_reference_root()
        commands, labels = _parse_commands(cli.commands)
        terrains = tuple(x.strip() for x in cli.terrains.split(",") if x.strip())
        result = run_closed_loop(
            adapter,
            reference_root=reference_root,
            out_dir=cli.out,
            terrains=terrains,
            commands=commands,
            command_labels=labels,
            duration_s=cli.duration_s,
            simulation_dt=cli.simulation_dt,
            control_decimation=cli.control_decimation,
            seed=cli.seed,
            max_rollouts=cli.max_rollouts,
            route_modes=tuple(x.strip() for x in cli.route_modes.split(",") if x.strip()),
        )
    print(json.dumps(_safe_json(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
