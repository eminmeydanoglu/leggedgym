"""Rollout storage for the MoE-CTS port (go2_moects), go2_rl_gym-aligned.

Contract (fixed, shared with the algorithm and runner ports):

    mini_batch_generator(num_mini_batches, num_epochs=8) yields per minibatch:
        (teacher_batch, student_batch, teacher_critic, student_critic)

    teacher_batch : 7-tuple, field-for-field identical to what
        RolloutStorageCTS.mini_batch_generator yields for the teacher role:
        (obs, privileged_obs, actions, old_actions_log_prob, advantages,
         old_mu, old_sigma)
    student_batch : 6-tuple, identical to the host CTS student fields:
        (obs, privileged_obs, observation_histories, actions,
         old_actions_log_prob, advantages)
    teacher_critic / student_critic : CriticMiniBatch, role-homogeneous
        (teacher_critic only teacher-env transitions, student_critic only
        student-env transitions) and sample-aligned: within one
        CriticMiniBatch, index i of every field is the SAME transition.

Why this diverges from the base RolloutStorageCTS instead of reusing it:
the base generator samples its critic batch (critic_obs/values/returns) with
an independent randperm over ALL envs, so (a) teacher and student transitions
mix inside one critic minibatch and (b) critic_obs is decoupled from
observation_histories (which only rides the student stream). The reference
go2_rl_gym MoE-CTS update needs per-sample (critic_obs, history) pairs to
evaluate the value function and to regress the student MoE encoder latent
onto the teacher encoder latent, per role. One index stream per role —
the same design as the reference — gives aligned, role-pure critic batches
while keeping the teacher/student role tuples byte-identical to the base
class (same flattening, same per-role randperm, same slicing, permutation
generated once and reused across epochs exactly like the base class).

Env role layout follows the host convention: teacher envs are indices
[0, num_teacher), student envs [num_teacher, num_envs) (see
rsl_rl/algorithms/ppo_cts.py and legged_gym/envs/go2/go2_cts/go2_cts.py).
compute_returns(last_values, gamma, lam) is inherited unchanged from
RolloutStorageCTS (bootstrap values are supplied by the caller).
"""

from typing import NamedTuple

import torch

from .rollout_storage_cts import RolloutStorageCTS


class CriticMiniBatch(NamedTuple):
    critic_observations: torch.Tensor      # [B, *critic_obs_shape]
    observation_histories: torch.Tensor    # [B, *obs_history_shape], storage layout
    target_values: torch.Tensor            # [B, 1]
    returns: torch.Tensor                  # [B, 1]
    advantages: torch.Tensor               # [B, 1]
    old_actions_log_prob: torch.Tensor     # [B, 1]
    old_mu: torch.Tensor                   # [B, *actions_shape]
    old_sigma: torch.Tensor                # [B, *actions_shape]


class RolloutStorageMoECTS(RolloutStorageCTS):
    """RolloutStorageCTS with aligned, role-homogeneous critic minibatches."""

    # Part of the contract with PPO_MOE_CTS.update(): mini_batch_generator draws
    # its per-role permutation ONCE, outside the epoch loop (inherited
    # RolloutStorageCTS behaviour), so every epoch yields the same minibatches in
    # the same order. The algorithm relies on this to materialize a single epoch
    # and replay it instead of holding num_learning_epochs redundant copies of
    # the rollout. Flip this to False if the generator ever reshuffles per epoch.
    minibatches_are_epoch_invariant = True

    class Transition(RolloutStorageCTS.Transition):
        # Host CTS transition already carries every field the MoE-CTS port
        # needs (observations, privileged_observations, observation_histories,
        # critic_observations, actions, rewards, dones, values,
        # actions_log_prob, action_mean, action_sigma, hidden_states).
        pass

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        # Per-role batch sizes and index streams: identical construction to
        # RolloutStorageCTS.mini_batch_generator (teacher first, then student;
        # each randperm generated once and reused across epochs). The role
        # batches below are therefore exactly what the base class yields, and
        # the critic batches share the same per-role indices, so every field
        # of a CriticMiniBatch is aligned to the same transition.
        teacher_mini_batch_size = self.num_teacher * self.num_transitions_per_env // num_mini_batches
        student_mini_batch_size = (self.num_envs - self.num_teacher) * self.num_transitions_per_env // num_mini_batches
        teacher_indices = torch.randperm(num_mini_batches*teacher_mini_batch_size, requires_grad=False, device=self.device)
        student_indices = torch.randperm(num_mini_batches*student_mini_batch_size, requires_grad=False, device=self.device)

        # Split data into teacher group and student group (same slicing and
        # flattening as the base class; step-major within each role group,
        # applied identically to every tensor so alignment is preserved).
        teacher_observations = self.observations[:, 0:self.num_teacher].flatten(0, 1)
        teacher_privileged_observations = self.privileged_observations[:, 0:self.num_teacher].flatten(0, 1)
        teacher_obs_histories = self.observation_histories[:, 0:self.num_teacher].flatten(0, 1)
        teacher_actions = self.actions[:, 0:self.num_teacher].flatten(0, 1)
        teacher_old_actions_log_prob = self.actions_log_prob[:, 0:self.num_teacher].flatten(0, 1)
        teacher_advantages = self.teacher_advantages.flatten(0, 1)
        teacher_old_mu = self.mu[:, 0:self.num_teacher].flatten(0, 1)
        teacher_old_sigma = self.sigma[:, 0:self.num_teacher].flatten(0, 1)
        teacher_critic_observations = self.critic_observations[:, 0:self.num_teacher].flatten(0, 1)
        teacher_values = self.values[:, 0:self.num_teacher].flatten(0, 1)
        teacher_returns = self.returns[:, 0:self.num_teacher].flatten(0, 1)

        student_observations = self.observations[:, self.num_teacher:].flatten(0, 1)
        student_privileged_observations = self.privileged_observations[:, self.num_teacher:].flatten(0, 1)
        student_obs_histories = self.observation_histories[:, self.num_teacher:].flatten(0, 1)
        student_actions = self.actions[:, self.num_teacher:].flatten(0, 1)
        student_old_actions_log_prob = self.actions_log_prob[:, self.num_teacher:].flatten(0, 1)
        student_advantages = self.student_advantages.flatten(0, 1)
        student_old_mu = self.mu[:, self.num_teacher:].flatten(0, 1)
        student_old_sigma = self.sigma[:, self.num_teacher:].flatten(0, 1)
        student_critic_observations = self.critic_observations[:, self.num_teacher:].flatten(0, 1)
        student_values = self.values[:, self.num_teacher:].flatten(0, 1)
        student_returns = self.returns[:, self.num_teacher:].flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                # One index slice per role per minibatch, applied to every
                # tensor of that role (teacher-batch fields, student-batch
                # fields and both critic minibatches alike).
                start_teacher = i*teacher_mini_batch_size
                end_teacher = (i+1)*teacher_mini_batch_size
                teacher_batch_idx = teacher_indices[start_teacher:end_teacher]
                start_student = i*student_mini_batch_size
                end_student = (i+1)*student_mini_batch_size
                student_batch_idx = student_indices[start_student:end_student]

                # Role tuples: field order and content identical to the base
                # class generator's teacher / student sections.
                teacher_batch = (
                    teacher_observations[teacher_batch_idx],
                    teacher_privileged_observations[teacher_batch_idx],
                    teacher_actions[teacher_batch_idx],
                    teacher_old_actions_log_prob[teacher_batch_idx],
                    teacher_advantages[teacher_batch_idx],
                    teacher_old_mu[teacher_batch_idx],
                    teacher_old_sigma[teacher_batch_idx],
                )
                student_batch = (
                    student_observations[student_batch_idx],
                    student_privileged_observations[student_batch_idx],
                    student_obs_histories[student_batch_idx],
                    student_actions[student_batch_idx],
                    student_old_actions_log_prob[student_batch_idx],
                    student_advantages[student_batch_idx],
                )

                teacher_critic = CriticMiniBatch(
                    critic_observations=teacher_critic_observations[teacher_batch_idx],
                    observation_histories=teacher_obs_histories[teacher_batch_idx],
                    target_values=teacher_values[teacher_batch_idx],
                    returns=teacher_returns[teacher_batch_idx],
                    advantages=teacher_advantages[teacher_batch_idx],
                    old_actions_log_prob=teacher_old_actions_log_prob[teacher_batch_idx],
                    old_mu=teacher_old_mu[teacher_batch_idx],
                    old_sigma=teacher_old_sigma[teacher_batch_idx],
                )
                student_critic = CriticMiniBatch(
                    critic_observations=student_critic_observations[student_batch_idx],
                    observation_histories=student_obs_histories[student_batch_idx],
                    target_values=student_values[student_batch_idx],
                    returns=student_returns[student_batch_idx],
                    advantages=student_advantages[student_batch_idx],
                    old_actions_log_prob=student_old_actions_log_prob[student_batch_idx],
                    old_mu=student_old_mu[student_batch_idx],
                    old_sigma=student_old_sigma[student_batch_idx],
                )

                yield teacher_batch, student_batch, teacher_critic, student_critic
