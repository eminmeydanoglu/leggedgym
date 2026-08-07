# Actor-Critic for MoE-CTS (Mixture-of-Experts Concurrent Teacher-Student).
# Ported from go2_rl_gym (wty-yy) onto the host ActorCriticCTS contract:
# the MoE student encoder replaces the MLP history encoder while keeping the
# same attribute name and call signature, so PPO_CTS/CTSRunner work unchanged.

import torch
import torch.nn as nn
from torch.distributions import Normal

from .actor_critic_cts import ActorCriticCTS
from .moe_utils import StudentMoEEncoder, StudentDenseEncoder, L2Norm, SimNorm, MLP


class ActorCriticMoECTS(ActorCriticCTS):
    """CTS actor-critic whose student encoder is a Mixture-of-Experts network.

    Differences from ActorCriticCTS (paper-faithful):
    - ``history_encoder`` is a StudentMoEEncoder (dense soft mixture of
      ``expert_num`` experts with a softmax gating network) instead of an MLP.
      The attribute name is kept so PPO_CTS builds the separate encoder
      optimizer from its parameters unchanged.
    - Privilege (teacher) encoder output is L2/Sim-normalized.
    - Student latent is computed under no_grad in the RL pass: the encoder is
      trained only by the distillation loss (separate optimizer).
    - Critic input is ``[latent.detach(), critic_observations]`` where the
      latent is role-aware (reference ``evaluate(privileged_obs, history,
      is_teacher)``): teacher/privilege latent for teacher envs, MoE history
      latent for student envs. The detach keeps critic-loss gradients out of
      both encoders.
    """

    def __init__(self,
                 num_actor_obs,
                 num_actions,
                 num_privilege_encoder_input,
                 num_history_encoder_input,
                 num_latent_dims,
                 num_critic_obs,
                 actor_hidden_dims=[512, 256, 128],
                 critic_hidden_dims=[512, 256, 128],
                 expert_num=8,
                 student_encoder_hidden_dims=[512, 256, 256],
                 student_encoder_type='moe',
                 norm_type='l2norm',
                 activation='elu',
                 **kwargs):
        assert norm_type in ['l2norm', 'simnorm'], f"Normalization type {norm_type} not supported!"
        super().__init__(num_actor_obs,
                         num_actions,
                         num_privilege_encoder_input,
                         num_history_encoder_input,
                         num_latent_dims,
                         num_critic_obs,
                         actor_hidden_dims=actor_hidden_dims,
                         critic_hidden_dims=critic_hidden_dims,
                         activation=activation,
                         **kwargs)

        if student_encoder_type == 'moe':
            self.history_encoder = StudentMoEEncoder(
                expert_num=expert_num, input_dim=num_history_encoder_input,
                hidden_dims=student_encoder_hidden_dims,
                output_dim=num_latent_dims, activation=activation,
                norm_type=norm_type)
        elif student_encoder_type == 'dense':
            self.history_encoder = StudentDenseEncoder(
                input_dim=num_history_encoder_input,
                hidden_dims=student_encoder_hidden_dims,
                output_dim=num_latent_dims, activation=activation,
                norm_type=norm_type)
        else:
            raise ValueError(
                f"Unknown student_encoder_type {student_encoder_type!r}; "
                "expected 'moe' or 'dense'.")
        # Normalize the teacher latent like the student latent.
        norm_layer = L2Norm() if norm_type == 'l2norm' else SimNorm()
        self.privilege_encoder = nn.Sequential(self.privilege_encoder, norm_layer)

        # Rebuild the critic with latent + critic_obs input (paper: 32 + 263).
        self.critic = MLP([num_critic_obs + num_latent_dims, *critic_hidden_dims, 1], activation=activation)

        print(f"Privilege Encoder (+{norm_type}): {self.privilege_encoder}")
        print(f"Student {student_encoder_type} Encoder: {self.history_encoder}")
        print(f"Critic MLP: {self.critic}")

    # Actor input is [latent, observations] -- latent FIRST. The host
    # ActorCriticCTS concatenates the other way round; the reference
    # (go2_rl_gym actor_critic_moe_cts.py / actor_critic_moe_ng_cts.py, and the
    # exported deploy policies) uses `torch.cat([latent, obs])`. Matching it
    # costs nothing for a freshly trained policy and makes the published
    # go2_rl_gym checkpoints loadable into this actor (see
    # legged_gym/scripts/import_go2_rl_gym_policy.py). Keep both roles on the
    # same order, hence the teacher override too.
    def update_distribution(self, observations, observation_history, privilege_observations, act_type):
        if act_type == "student":
            # The encoder is trained only by the distillation loss; block
            # RL-loss gradients (paper) and save the backward compute.
            with torch.no_grad():
                latent = self.history_encoder(observation_history)
        elif act_type == "teacher":
            latent = self.privilege_encoder(privilege_observations)
        else:
            raise ValueError("Invalid act_type. Must be 'teacher' or 'student'.")
        mean = self.actor(torch.cat((latent, observations), dim=-1))
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act_teacher(self, observations, privilege_observations, **kwargs):
        latent = self.privilege_encoder(privilege_observations)
        return self.actor(torch.cat((latent, observations), dim=-1))

    def act_student(self, observations, observation_history, **kwargs):
        # Deploy/inference path. The MoE encoder is trained only by the
        # distillation loss, so its latent is computed under no_grad here as
        # well (RL surrogate must never reach it).
        with torch.no_grad():
            latent = self.history_encoder(observation_history)
        actions_mean = self.actor(torch.cat((latent, observations), dim=-1))
        return actions_mean

    def evaluate(self, critic_observations, obs_history=None, is_teacher=True, **kwargs):
        # Role-aware critic evaluation (reference actor_critic_moe_cts.py:134-140).
        # critic_observations have the same layout as the privilege encoder
        # input; the critic additionally sees the role latent concatenated in
        # front. The latent is detached before the concat in BOTH roles
        # (reference: torch.cat([latent.detach(), privileged_obs])), so the
        # critic loss never trains either encoder.
        if is_teacher:
            latent = self.privilege_encoder(critic_observations)
        else:
            if obs_history is None:
                raise ValueError(
                    "obs_history is required when is_teacher=False: the student "
                    "latent comes from the history (MoE) encoder")
            latent = self.history_encoder(obs_history)
        value = self.critic(torch.cat((latent.detach(), critic_observations), dim=-1))
        return value
