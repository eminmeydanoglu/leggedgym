# PPO for MoE-CTS (Mixture-of-Experts Concurrent Teacher-Student).
# Ported from go2_rl_gym (wty-yy) onto the host PPO_CTS substrate.
#
# Reference parity deltas vs plain PPO_CTS (reference: rsl_rl/algorithms/moe_cts.py):
# - Role-aware value bootstrap and value loss: teacher envs [0:num_teacher] use
#   the privilege-encoder latent, student envs use the history (MoE) encoder
#   latent; both latents are detached before the critic concat.
# - The student-encoder distillation loss carries an additional gating
#   load-balance term and is optimized by the separate encoder optimizer in a
#   second pass over the same minibatches.
# - update() returns (mean_value_loss, mean_teacher_surrogate_loss,
#   mean_student_surrogate_loss, mean_latent_loss, moe_stats).

import math

import torch
import torch.nn as nn

from rsl_rl.algorithms.ppo_cts import PPO_CTS


class PPO_MOE_CTS(PPO_CTS):
    def __init__(self, actor_critic, load_balance_coef=0.01, **kwargs):
        super().__init__(actor_critic, **kwargs)
        self.load_balance_coef = load_balance_coef

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape,
                     privileged_obs_shape, obs_history_shape, critic_obs_shape, action_shape):
        # Lazy import: RolloutStorageMoECTS is delivered by the storage track.
        from rsl_rl.storage import RolloutStorageMoECTS
        self.storage = RolloutStorageMoECTS(
            num_envs, self.num_teacher, num_transitions_per_env, actor_obs_shape,
            privileged_obs_shape, obs_history_shape, critic_obs_shape, action_shape, self.device)

    def _compute_rollout_values(self, critic_obs, obs_history):
        # Reference parity (moe_cts/cts act): rollout values are role-aware.
        # Overriding the PPO_CTS hook (instead of post-fixing transition.values
        # after super().act()) avoids a wasted teacher-latent critic forward over
        # every env on every rollout step.
        teacher_values = self.actor_critic.evaluate(critic_obs[:self.num_teacher], is_teacher=True)
        student_values = self.actor_critic.evaluate(
            critic_obs[self.num_teacher:], obs_history[self.num_teacher:], is_teacher=False)
        return torch.cat((teacher_values, student_values), dim=0).detach()

    def compute_returns(self, critic_obs, obs_history):
        # Role-aware bootstrap (reference cts.py:159-165): teacher envs
        # [0:num_teacher] bootstrap from the privilege-latent critic, student
        # envs from the history-latent critic.
        assert self.storage is not None  # storage is initialized in init_storage()
        teacher_values = self.actor_critic.evaluate(critic_obs[:self.num_teacher], is_teacher=True)
        student_values = self.actor_critic.evaluate(
            critic_obs[self.num_teacher:], obs_history[self.num_teacher:], is_teacher=False)
        last_values = torch.cat((teacher_values, student_values), dim=0).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        """One PPO update + one student-encoder distillation pass.

        Storage contract (RolloutStorageMoECTS): mini_batch_generator yields
        (teacher_batch, student_batch, teacher_critic, student_critic) where
          teacher_batch = (obs, privileged_obs, actions, old_actions_log_prob,
                           advantages, old_mu, old_sigma)
          student_batch = (obs, privileged_obs, obs_histories, actions,
                           old_actions_log_prob, advantages)
        (same field order as the per-role slices of
        RolloutStorageCTS.mini_batch_generator's flat yield), and
        teacher_critic/student_critic are CriticMiniBatch namedtuples with
        fields (critic_observations, observation_histories, target_values,
        returns, advantages, old_actions_log_prob, old_mu, old_sigma).
        """
        assert not self.actor_critic.is_recurrent
        mean_value_loss = 0.0
        mean_teacher_surrogate_loss = 0.0
        mean_student_surrogate_loss = 0.0
        # Materialize once so the RL pass and the student-encoder pass iterate
        # the same minibatches (reference moe_cts.py:111).
        #
        # RolloutStorageMoECTS draws its per-role permutation once and does NOT
        # reshuffle between epochs (inherited from RolloutStorageCTS), so all
        # num_learning_epochs epochs yield identical minibatches. Materializing
        # the full generator therefore held num_learning_epochs redundant copies
        # of the whole rollout (multiple GB at 8192 envs). Build one epoch and
        # replay it instead; storages that do not advertise the guarantee keep
        # the old behaviour. The total iteration count is identical either way.
        data, num_rl_epochs = self._materialize_minibatches()

        for _ in range(num_rl_epochs):
            for teacher_batch, student_batch, teacher_critic, student_critic in data:
                loss, teacher_surrogate_loss, student_surrogate_loss, value_loss = self._compute_rl_loss(
                    teacher_batch, student_batch, teacher_critic, student_critic)

                # Gradient step (RL params only; the student MoE encoder is excluded)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.rl_params, self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_teacher_surrogate_loss += teacher_surrogate_loss.item()
                mean_student_surrogate_loss += student_surrogate_loss.item()

        # Student MoE encoder pass: separate optimizer (reference moe_cts.py:197-224).
        # moe_stats are aggregated over every epoch x minibatch x encoder-epoch
        # computation, weighted by the minibatch student-sample count.
        stats_sums = {"latent_mse": 0.0, "load_balance": 0.0,
                      "student_encoder_total": 0.0, "gating_entropy": 0.0}
        usage_sum = None
        stats_weight = 0
        for _ in range(num_rl_epochs):
            for _, student_batch, _, _ in data:
                (_, student_privileged_obs_batch, student_obs_histories_batch,
                 _, _, _) = student_batch
                n_student = student_obs_histories_batch.shape[0]
                for _ in range(self.num_encoder_epochs):
                    encoder_loss, latent_loss, load_balance_loss, gating_weights = \
                        self._compute_encoder_losses(student_obs_histories_batch, student_privileged_obs_batch)

                    # Aggregate pre-update stats of this computation.
                    with torch.no_grad():
                        weights = gating_weights.detach()
                        gating_entropy = -torch.xlogy(weights, weights).sum(dim=-1).mean()
                        mean_usage = weights.mean(dim=0)
                    stats_sums["latent_mse"] += latent_loss.item() * n_student
                    stats_sums["load_balance"] += load_balance_loss.item() * n_student
                    stats_sums["student_encoder_total"] += encoder_loss.item() * n_student
                    stats_sums["gating_entropy"] += gating_entropy.item() * n_student
                    usage_sum = mean_usage * n_student if usage_sum is None else usage_sum + mean_usage * n_student
                    stats_weight += n_student

                    self.history_encoder_optimizer.zero_grad()
                    encoder_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.actor_critic.history_encoder.parameters(), self.max_grad_norm)
                    self.history_encoder_optimizer.step()

        mean_expert_usage = usage_sum / stats_weight
        moe_stats = {
            "latent_mse": stats_sums["latent_mse"] / stats_weight,
            "load_balance": stats_sums["load_balance"] / stats_weight,
            "student_encoder_total": stats_sums["student_encoder_total"] / stats_weight,
            "gating_entropy": stats_sums["gating_entropy"] / stats_weight,
            "effective_experts": math.exp(stats_sums["gating_entropy"] / stats_weight),
            "expert_usage_min": float(mean_expert_usage.min()),
            "expert_usage_max": float(mean_expert_usage.max()),
            "expert_usage_std": float(mean_expert_usage.std()),
            "expert_usage": mean_expert_usage.tolist(),
        }
        mean_latent_loss = moe_stats["latent_mse"]

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_teacher_surrogate_loss /= num_updates
        mean_student_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_teacher_surrogate_loss, mean_student_surrogate_loss, mean_latent_loss, moe_stats

    def _materialize_minibatches(self):
        """(minibatch list, number of times to replay it).

        Returns one epoch's worth of minibatches plus a replay count when the
        storage guarantees its permutation is epoch-invariant (the
        RolloutStorageCTS/MoECTS behaviour: the randperm is drawn once outside
        the epoch loop), otherwise the full generator and a replay count of 1.
        Both branches perform num_learning_epochs * num_mini_batches gradient
        steps over the same batches; only peak memory differs.
        """
        if getattr(self.storage, "minibatches_are_epoch_invariant", False):
            return (list(self.storage.mini_batch_generator(self.num_mini_batches, 1)),
                    self.num_learning_epochs)
        return (list(self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs)), 1)

    def _compute_rl_loss(self, teacher_batch, student_batch, teacher_critic, student_critic):
        (teacher_obs_batch, teacher_privileged_obs_batch, teacher_actions_batch,
         teacher_old_actions_log_prob_batch, teacher_advantages_batch,
         teacher_old_mu_batch, teacher_old_sigma_batch) = teacher_batch
        (student_obs_batch, student_privileged_obs_batch, student_obs_histories_batch,
         student_actions_batch, student_old_actions_log_prob_batch,
         student_advantages_batch) = student_batch

        # Teacher arm (identical surrogate/KL handling to PPO_CTS._compute_rl_loss)
        self.actor_critic.act(teacher_obs_batch, None, teacher_privileged_obs_batch, act_type='teacher')
        teacher_actions_log_prob_batch = self.actor_critic.get_actions_log_prob(teacher_actions_batch)
        teacher_entropy_batch = self.actor_critic.entropy
        teacher_mu_batch = self.actor_critic.action_mean
        teacher_sigma_batch = self.actor_critic.action_std

        ## Teacher KL, adapt learning rate
        self._adjust_learning_rate(teacher_sigma_batch, teacher_old_sigma_batch,
                                   teacher_mu_batch, teacher_old_mu_batch)

        ## Surrogate loss
        ratio = torch.exp(teacher_actions_log_prob_batch -
                          torch.squeeze(teacher_old_actions_log_prob_batch))
        teacher_surrogate_loss = self._compute_surrogate_loss(ratio, teacher_advantages_batch)

        # Student arm (the MoE history latent is computed under no_grad inside
        # ActorCriticMoECTS.update_distribution, so the surrogate never puts
        # gradients into the student encoder)
        self.actor_critic.act(student_obs_batch, student_obs_histories_batch, None, act_type='student')
        student_actions_log_prob_batch = self.actor_critic.get_actions_log_prob(student_actions_batch)
        student_entropy_batch = self.actor_critic.entropy

        ## Surrogate loss
        ratio = torch.exp(student_actions_log_prob_batch -
                          torch.squeeze(student_old_actions_log_prob_batch))
        student_surrogate_loss = self._compute_surrogate_loss(ratio, student_advantages_batch)

        # Role-aware value loss (reference moe_cts.py:171-179): per-sample
        # losses over the concatenated teacher+student batch with a single
        # mean, i.e. teacher/student arms weighted by their sample counts.
        teacher_value_batch = self.actor_critic.evaluate(
            teacher_critic.critic_observations, is_teacher=True)
        student_value_batch = self.actor_critic.evaluate(
            student_critic.critic_observations, student_critic.observation_histories, is_teacher=False)
        value_losses = torch.cat((
            self._value_losses_per_sample(
                teacher_value_batch, teacher_critic.returns, teacher_critic.target_values),
            self._value_losses_per_sample(
                student_value_batch, student_critic.returns, student_critic.target_values),
        ), dim=0)
        value_loss = value_losses.mean()

        total_entropy_batch = torch.cat((teacher_entropy_batch, student_entropy_batch), dim=0)

        loss = self.value_loss_coef * value_loss + \
            teacher_surrogate_loss + student_surrogate_loss \
            - self.entropy_coef * (total_entropy_batch.mean())

        return loss, teacher_surrogate_loss, student_surrogate_loss, value_loss

    def _value_losses_per_sample(self, value_batch, returns_batch, target_values_batch):
        # Per-sample counterpart of PPO._compute_value_function_loss so the two
        # roles can be concatenated before the mean (reference parity).
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -self.clip_param, self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            return torch.max(value_losses, value_losses_clipped)
        return (returns_batch - value_batch).pow(2)

    def _compute_encoder_loss(self, student_obs_histories_batch, student_privileged_obs_batch):
        return self._compute_encoder_losses(student_obs_histories_batch, student_privileged_obs_batch)[0]

    def _compute_encoder_losses(self, student_obs_histories_batch, student_privileged_obs_batch):
        """Student MoE encoder loss = distillation MSE + load-balance term.

        Returns (total, latent_mse, load_balance, gating_weights); the
        components feed the moe_stats aggregation in update().
        """
        # Explicit return, not the last_gating_weights snapshot: the snapshot is
        # detached (and the RL passes overwrite it from no_grad forwards), so it
        # would give the load-balance term no gradient at all.
        encoder_predictions, gating_weights = \
            self.actor_critic.history_encoder.forward_with_weights(student_obs_histories_batch)

        with torch.no_grad():  # don't backpropagate through the encoder targets
            encoder_targets = self.actor_critic.privilege_encoder(student_privileged_obs_batch)

        latent_loss = nn.functional.mse_loss(encoder_predictions, encoder_targets)

        # Load-balance loss: push mean gating usage towards uniform
        # (reference moe_cts.py:211-213)
        mean_usage = torch.mean(gating_weights, dim=0)
        target_usage = torch.full_like(mean_usage, 1.0 / gating_weights.shape[1])
        load_balance_loss = torch.mean((mean_usage - target_usage).pow(2))

        total = latent_loss + self.load_balance_coef * load_balance_loss
        return total, latent_loss, load_balance_loss, gating_weights
