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

Why this diverges from the HOST base RolloutStorageCTS instead of reusing it
(divergence from the host, NOT from the go2_rl_gym reference): the host
generator samples its critic batch (critic_obs/values/returns) with a third,
independent randperm over ALL envs (rollout_storage_cts.py:123), so (a)
teacher and student transitions mix inside one critic minibatch and (b)
critic_obs is decoupled from observation_histories (which only rides the
student stream). The reference go2_rl_gym MoE-CTS update needs per-sample
(critic_obs, history) pairs to evaluate the value function and to regress the
student MoE encoder latent onto the teacher encoder latent, per role.

The reference already has exactly that property: its generator draws ONLY the
two per-role streams (go2_rl_gym rollout_storage_cts.py:158-159) and applies
the same slice to every field — obs, critic_obs (= privileged_obs), history,
values, returns, mu/sigma — while moe_cts.py:122-124 calls act() and
evaluate() on the same [start:end] window. So one index stream per role here
is reference PARITY, not a "cleaner" deviation: it gives aligned, role-pure
critic batches while keeping the teacher/student role tuples byte-identical
to the base class (same flattening, same per-role randperm, same slicing,
permutation generated once and reused across epochs exactly like the base
class). The only remaining gap to the reference is packaging: the reference
yields one teacher-first concatenated tuple and slices roles by
[:teacher_samples] / [teacher_samples:]; this class yields them separately.

Env role layout is the reference's INTERLEAVED mapping (go2_rl_gym
moe_cts.py:96-102: every int(1/student_env_ratio)-th env is a student), owned
by PPO_MOE_CTS and injected into this storage -- NOT the host CTS convention
of contiguous teacher [0, num_teacher) / student [num_teacher, num_envs)
blocks. Every stored tensor stays in original env order (the runner writes
rewards/dones/timeouts in env order); only the role-partitioning points --
compute_returns GAE loops and the minibatch generator below -- gather by the
injected idxs. The per-role GAE recurrence matches
RolloutStorageCTS.compute_returns (bootstrap values are supplied by the
caller, in env order), but advantage normalization is GLOBAL over the
concatenated teacher+student advantages (vendored go2_rl_gym
rollout_storage_cts.py:141-143), deliberately NOT the host's per-role
normalization.
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

    def __init__(self, num_envs, num_teacher, num_transitions_per_env, obs_shape,
                 privileged_obs_shape, obs_history_shape, critic_obs_shape, actions_shape,
                 device='cpu', *, teacher_env_idxs, student_env_idxs):
        super().__init__(num_envs, num_teacher, num_transitions_per_env, obs_shape,
                         privileged_obs_shape, obs_history_shape, critic_obs_shape,
                         actions_shape, device)
        assert teacher_env_idxs is not None and student_env_idxs is not None, (
            "RolloutStorageMoECTS requires the interleaved role idxs "
            "(PPO_MOE_CTS.compute_role_env_idxs); the host CTS contiguous "
            "split is not a valid fallback for the MoE arm")
        assert len(teacher_env_idxs) == num_teacher, (
            f"{len(teacher_env_idxs)=} != {num_teacher=}")
        assert len(student_env_idxs) == num_envs - num_teacher, (
            f"{len(student_env_idxs)=} != {num_envs - num_teacher=}")
        self.teacher_env_idxs = teacher_env_idxs
        self.student_env_idxs = student_env_idxs

    def compute_returns(self, last_values, gamma, lam):
        # Same per-role GAE as RolloutStorageCTS.compute_returns, with the role
        # columns gathered by the interleaved env idxs instead of contiguous
        # slices. last_values arrives in env order; returns are scattered back
        # into the env-order buffer. Advantages are then normalized GLOBALLY
        # over both roles (vendored parity), after both GAE loops.
        ti, si = self.teacher_env_idxs, self.student_env_idxs

        # teacher role (base-class recurrence, slice [:, 0:num_teacher] -> idx gather)
        values_t = self.values[:, ti]
        rewards_t = self.rewards[:, ti]
        dones_t = self.dones[:, ti]
        returns_t = torch.empty_like(values_t)
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values[ti]
            else:
                next_values = values_t[step + 1]
            next_is_not_terminal = 1.0 - dones_t[step].float()
            delta = rewards_t[step] + next_is_not_terminal * \
                gamma * next_values - values_t[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            returns_t[step] = advantage + values_t[step]
        self.returns[:, ti] = returns_t

        # student role (base-class recurrence, slice [:, num_teacher:] -> idx gather)
        values_s = self.values[:, si]
        rewards_s = self.rewards[:, si]
        dones_s = self.dones[:, si]
        returns_s = torch.empty_like(values_s)
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values[si]
            else:
                next_values = values_s[step + 1]
            next_is_not_terminal = 1.0 - dones_s[step].float()
            delta = rewards_s[step] + next_is_not_terminal * \
                gamma * next_values - values_s[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            returns_s[step] = advantage + values_s[step]
        self.returns[:, si] = returns_s

        ## compute and normalize the advantages -- GLOBAL over both roles
        ## (vendored go2_rl_gym rollout_storage_cts.py:141-143: ONE mean/std
        ## over the concatenated teacher+student advantages), NOT per-role
        ## like the host RolloutStorageCTS. Both roles must be normalized
        ## here, after both GAE loops, because the student returns do not
        ## exist yet at the end of the teacher loop.
        self.teacher_advantages = returns_t - values_t
        self.student_advantages = returns_s - values_s
        all_advantages = torch.cat((self.teacher_advantages, self.student_advantages), dim=1)
        adv_mean, adv_std = all_advantages.mean(), all_advantages.std()
        # Raw (pre-normalization) per-role advantage scale, published for
        # telemetry. The single global mean/std above rescales both roles by
        # the SAME factor, so if one role's raw spread drifts away from the
        # other's, that role silently gets larger post-normalization
        # advantages -- i.e. a larger share of the policy gradient. Without
        # these numbers that drift is invisible in every downstream log (the
        # stored advantages are already normalized).
        self.raw_adv_stats = {
            "adv_mean_teacher": float(self.teacher_advantages.mean()),
            "adv_mean_student": float(self.student_advantages.mean()),
            "adv_std_teacher": float(self.teacher_advantages.std()),
            "adv_std_student": float(self.student_advantages.std()),
            "adv_std_global": float(adv_std),
        }
        self.raw_adv_stats["adv_std_ratio"] = (
            self.raw_adv_stats["adv_std_student"] /
            (self.raw_adv_stats["adv_std_teacher"] + 1e-8))
        self.teacher_advantages = (self.teacher_advantages - adv_mean) / (adv_std + 1e-8)
        self.student_advantages = (self.student_advantages - adv_mean) / (adv_std + 1e-8)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        # Per-role batch sizes and index streams: identical construction to
        # RolloutStorageCTS.mini_batch_generator (teacher first, then student;
        # each randperm generated once and reused across epochs). The role
        # batches below therefore have exactly the base class's structure, and
        # the critic batches share the same per-role indices, so every field
        # of a CriticMiniBatch is aligned to the same transition.
        teacher_mini_batch_size = self.num_teacher * self.num_transitions_per_env // num_mini_batches
        student_mini_batch_size = (self.num_envs - self.num_teacher) * self.num_transitions_per_env // num_mini_batches
        teacher_indices = torch.randperm(num_mini_batches*teacher_mini_batch_size, requires_grad=False, device=self.device)
        student_indices = torch.randperm(num_mini_batches*student_mini_batch_size, requires_grad=False, device=self.device)

        # Split data into teacher group and student group by the interleaved
        # role idxs (advanced indexing, then the same flattening as the base
        # class; step-major within each role group, applied identically to
        # every tensor so alignment is preserved).
        ti, si = self.teacher_env_idxs, self.student_env_idxs
        teacher_observations = self.observations[:, ti].flatten(0, 1)
        teacher_privileged_observations = self.privileged_observations[:, ti].flatten(0, 1)
        teacher_obs_histories = self.observation_histories[:, ti].flatten(0, 1)
        teacher_actions = self.actions[:, ti].flatten(0, 1)
        teacher_old_actions_log_prob = self.actions_log_prob[:, ti].flatten(0, 1)
        teacher_advantages = self.teacher_advantages.flatten(0, 1)
        teacher_old_mu = self.mu[:, ti].flatten(0, 1)
        teacher_old_sigma = self.sigma[:, ti].flatten(0, 1)
        teacher_critic_observations = self.critic_observations[:, ti].flatten(0, 1)
        teacher_values = self.values[:, ti].flatten(0, 1)
        teacher_returns = self.returns[:, ti].flatten(0, 1)

        student_observations = self.observations[:, si].flatten(0, 1)
        student_privileged_observations = self.privileged_observations[:, si].flatten(0, 1)
        student_obs_histories = self.observation_histories[:, si].flatten(0, 1)
        student_actions = self.actions[:, si].flatten(0, 1)
        student_old_actions_log_prob = self.actions_log_prob[:, si].flatten(0, 1)
        student_advantages = self.student_advantages.flatten(0, 1)
        student_old_mu = self.mu[:, si].flatten(0, 1)
        student_old_sigma = self.sigma[:, si].flatten(0, 1)
        student_critic_observations = self.critic_observations[:, si].flatten(0, 1)
        student_values = self.values[:, si].flatten(0, 1)
        student_returns = self.returns[:, si].flatten(0, 1)

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
