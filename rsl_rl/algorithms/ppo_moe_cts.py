# PPO for MoE-CTS (Mixture-of-Experts Concurrent Teacher-Student).
# Ported from go2_rl_gym (wty-yy) onto the host PPO_CTS substrate.
#
# Reference parity deltas vs plain PPO_CTS (reference: rsl_rl/algorithms/moe_cts.py):
# - Teacher/student roles are INTERLEAVED env ids (every
#   int(1/student_env_ratio)-th env is a student; reference moe_cts.py:96-102),
#   not the host CTS contiguous [0:num_teacher] / [num_teacher:] split. Role
#   forward passes run on idx-gathered rows and every tensor crossing the
#   storage boundary is scattered back to env order, so rewards/dones/timeouts
#   written by the runner in env order stay correct.
# - Role-aware value bootstrap and value loss: teacher envs use the
#   privilege-encoder latent, student envs use the history (MoE) encoder
#   latent; both latents are detached before the critic concat.
# - The student-encoder distillation loss carries an additional gating
#   load-balance term and is optimized by the separate encoder optimizer in a
#   second pass over the same minibatches.
# - update() returns (mean_value_loss, mean_teacher_surrogate_loss,
#   mean_student_surrogate_loss, mean_latent_loss, moe_stats). moe_stats also
#   carries the standard PPO health telemetry: entropy / clip_frac (pre-update
#   means over the RL pass), kl_mean (last adaptive-LR KL of the update, see
#   the _adjust_learning_rate override) and explained_variance (stored rollout
#   values vs returns).
#   On top of those aggregates it carries the diagnostics that separate the
#   two arms -- which is what any post-mortem of this algorithm needs, since
#   teacher and student share one actor/critic but see different latents and
#   fail independently:
#     per-role   entropy_/clip_frac_/kl_/value_loss_/explained_variance_
#                {teacher,student}, plus ratio_dev_max
#     gradients  grad_norm, encoder_grad_norm (pre-clip)
#     advantages adv_{mean,std}_{teacher,student}, adv_std_global,
#                adv_std_ratio -- RAW, pre-normalization (the storage
#                normalizes globally, so the stored advantages hide it)
#     distill    latent_cosine, latent_ev (scale-free counterparts of
#                latent_mse), action_gap_mse (teacher vs student action means
#                on the same states -- the end-to-end CTS convergence metric)
#     gating     gating_max_weight, experts_alive (per-sample collapse, which
#                the batch-mean expert_usage vector cannot show)

import math

import torch
import torch.nn as nn

from rsl_rl.algorithms.ppo_cts import PPO_CTS


def compute_role_env_idxs(num_envs, teacher_env_ratio, device):
    """Interleaved teacher/student env split (reference moe_cts.py:96-102).

    Every int(1/student_env_ratio)-th env is a student, the rest are teachers.
    Count guard and asserts are the reference's: the split only exists for
    (num_envs, ratio) pairs where the idx counts come out exact, anything else
    fails loudly here instead of silently mis-assigning roles.
    """
    teacher_num_envs = max(int(num_envs * teacher_env_ratio), 1)
    student_num_envs = num_envs - teacher_num_envs
    student_env_ratio = 1 - teacher_env_ratio
    teacher_env_idxs = torch.tensor(
        [i for i in range(num_envs) if i % int(1 / student_env_ratio) != 0], device=device)
    student_env_idxs = torch.tensor(
        [i for i in range(num_envs) if i % int(1 / student_env_ratio) == 0], device=device)
    assert len(teacher_env_idxs) == teacher_num_envs, (
        f"{len(teacher_env_idxs)=} != {teacher_num_envs=}: the interleaved "
        f"role split does not come out exact for {num_envs=} / "
        f"{teacher_env_ratio=} (reference requires num_envs divisible by "
        "int(1/student_env_ratio))")
    assert len(student_env_idxs) == student_num_envs, (
        f"{len(student_env_idxs)=} != {student_num_envs=}: the interleaved "
        f"role split does not come out exact for {num_envs=} / "
        f"{teacher_env_ratio=} (reference requires num_envs divisible by "
        "int(1/student_env_ratio))")
    return teacher_env_idxs, student_env_idxs


class PPO_MOE_CTS(PPO_CTS):
    def __init__(self, actor_critic, load_balance_coef=0.01, teacher_env_ratio=0.75, **kwargs):
        super().__init__(actor_critic, **kwargs)
        self.load_balance_coef = load_balance_coef
        # Must match the env-side ratio WtyCurriculumMixin uses to rescale
        # num_teacher (checked in _build_role_env_idxs).
        self.teacher_env_ratio = teacher_env_ratio
        # Interleaved role mapping, built lazily once num_envs is known (the
        # runner learns it only at init_storage time; tests may drive
        # act()/compute_returns() without init_storage).
        self.teacher_env_idxs = None
        self.student_env_idxs = None
        self._role_num_envs = None
        # PPO health telemetry: last per-minibatch mean KL of the update
        # (captured by the _adjust_learning_rate override) and the per-minibatch
        # stats stash (_compute_rl_loss -> update()).
        self.last_kl_mean = None
        self._minibatch_telemetry = None

    def _role_env_idxs(self, num_envs):
        if self.teacher_env_idxs is None:
            self._build_role_env_idxs(num_envs)
        assert self._role_num_envs == num_envs, (
            f"role mapping was built for {self._role_num_envs} envs, got a "
            f"batch of {num_envs}")
        return self.teacher_env_idxs, self.student_env_idxs

    def _build_role_env_idxs(self, num_envs):
        teacher_env_idxs, student_env_idxs = compute_role_env_idxs(
            num_envs, self.teacher_env_ratio, self.device)
        expected_teacher = max(int(num_envs * self.teacher_env_ratio), 1)
        assert expected_teacher == self.num_teacher, (
            f"{self.num_teacher=} is inconsistent with "
            f"max(int({num_envs} * {self.teacher_env_ratio}), 1) "
            f"= {expected_teacher}: env-side teacher_env_ratio and the "
            "algorithm's teacher_env_ratio must match")
        self.teacher_env_idxs = teacher_env_idxs
        self.student_env_idxs = student_env_idxs
        self._role_num_envs = num_envs

    def _scatter_to_env_order(self, teacher_rows, student_rows):
        """Inverse of the teacher/student gather: role-grouped rows -> env order."""
        out = torch.empty(len(self.teacher_env_idxs) + len(self.student_env_idxs),
                          *teacher_rows.shape[1:],
                          device=teacher_rows.device, dtype=teacher_rows.dtype)
        out[self.teacher_env_idxs] = teacher_rows
        out[self.student_env_idxs] = student_rows
        return out

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape,
                     privileged_obs_shape, obs_history_shape, critic_obs_shape, action_shape):
        # Lazy import: RolloutStorageMoECTS is delivered by the storage track.
        from rsl_rl.storage import RolloutStorageMoECTS
        self._role_env_idxs(num_envs)
        self.storage = RolloutStorageMoECTS(
            num_envs, self.num_teacher, num_transitions_per_env, actor_obs_shape,
            privileged_obs_shape, obs_history_shape, critic_obs_shape, action_shape, self.device,
            teacher_env_idxs=self.teacher_env_idxs, student_env_idxs=self.student_env_idxs)

    def act(self, obs, privileged_obs, obs_history, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Reference parity (cts.py:112-142): roles are interleaved env ids, so
        # the role forward passes run on idx-gathered rows and the results are
        # scattered back to env order (the reference's real_actions). The
        # transition -- and with it the storage -- stays in env order.
        ti, si = self._role_env_idxs(obs.shape[0])
        teacher_actions = self.actor_critic.act(obs[ti], None, privileged_obs[ti], act_type='teacher').detach()
        teacher_actions_log_prob = self.actor_critic.get_actions_log_prob(teacher_actions).detach()
        teacher_action_mean = self.actor_critic.action_mean.detach()
        teacher_action_sigma = self.actor_critic.action_std.detach()
        student_actions = self.actor_critic.act(obs[si], obs_history[si], None, act_type='student').detach()
        student_actions_log_prob = self.actor_critic.get_actions_log_prob(student_actions).detach()
        student_action_mean = self.actor_critic.action_mean.detach()
        student_action_sigma = self.actor_critic.action_std.detach()
        # store the actions and log probs (env order)
        self.transition.actions = self._scatter_to_env_order(teacher_actions, student_actions)
        self.transition.actions_log_prob = self._scatter_to_env_order(
            teacher_actions_log_prob, student_actions_log_prob)
        self.transition.action_mean = self._scatter_to_env_order(teacher_action_mean, student_action_mean)
        self.transition.action_sigma = self._scatter_to_env_order(teacher_action_sigma, student_action_sigma)
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs.detach()
        self.transition.privileged_observations = privileged_obs.detach()
        self.transition.observation_histories = obs_history.detach()
        self.transition.critic_observations = critic_obs.detach()
        self.transition.values = self._compute_rollout_values(
            self.transition.critic_observations, self.transition.observation_histories)
        return self.transition.actions

    def _compute_rollout_values(self, critic_obs, obs_history):
        # Reference parity (moe_cts/cts act): rollout values are role-aware.
        # Overriding the PPO_CTS hook (instead of post-fixing transition.values
        # after super().act()) avoids a wasted teacher-latent critic forward over
        # every env on every rollout step. Returned in env order.
        ti, si = self._role_env_idxs(critic_obs.shape[0])
        teacher_values = self.actor_critic.evaluate(critic_obs[ti], is_teacher=True)
        student_values = self.actor_critic.evaluate(
            critic_obs[si], obs_history[si], is_teacher=False)
        return self._scatter_to_env_order(teacher_values, student_values).detach()

    def compute_returns(self, critic_obs, obs_history):
        # Role-aware bootstrap (reference cts.py:159-165): teacher envs
        # bootstrap from the privilege-latent critic, student envs from the
        # history-latent critic; last_values cross the storage boundary in env
        # order.
        assert self.storage is not None  # storage is initialized in init_storage()
        ti, si = self._role_env_idxs(critic_obs.shape[0])
        teacher_values = self.actor_critic.evaluate(critic_obs[ti], is_teacher=True)
        student_values = self.actor_critic.evaluate(
            critic_obs[si], obs_history[si], is_teacher=False)
        last_values = self._scatter_to_env_order(teacher_values, student_values).detach()
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
        # Pre-update PPO health sums (per-minibatch telemetry stashed by
        # _compute_rl_loss; simple means over the RL pass, same weighting as
        # the loss means below). Key set is whatever the telemetry helper
        # emitted, so adding a stat there needs no bookkeeping here.
        rl_sums = {}
        # Pre-clip gradient norms (clip_grad_norm_ returns them for free).
        # Without these, a run that dies from an exploding update is
        # indistinguishable from one that dies from a bad reward: every other
        # channel is recorded BEFORE the gradient step.
        grad_norm_sum = 0.0
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
                # Grab the pre-gradient minibatch telemetry BEFORE the
                # optimizer step (plain floats, no autograd ties).
                minibatch_telemetry = self._minibatch_telemetry

                # Gradient step (RL params only; the student MoE encoder is excluded)
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.rl_params, self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_teacher_surrogate_loss += teacher_surrogate_loss.item()
                mean_student_surrogate_loss += student_surrogate_loss.item()
                grad_norm_sum += float(grad_norm)
                for key, value in minibatch_telemetry.items():
                    rl_sums[key] = rl_sums.get(key, 0.0) + value

        # Student MoE encoder pass: separate optimizer (reference moe_cts.py:197-224).
        # moe_stats are aggregated over every epoch x minibatch x encoder-epoch
        # computation, weighted by the minibatch student-sample count.
        stats_sums = {"latent_mse": 0.0, "load_balance": 0.0,
                      "student_encoder_total": 0.0, "gating_entropy": 0.0,
                      # Scale-free distillation quality. latent_mse alone is
                      # uninterpretable (its floor depends on the teacher
                      # latent's own spread, which drifts as the privilege
                      # encoder trains), so a plateau at 0.05 could mean
                      # "converged" or "learning nothing". cosine and
                      # latent_ev are bounded and comparable across runs.
                      "latent_cosine": 0.0, "latent_ev": 0.0,
                      # Gating collapse: mean usage can look uniform while
                      # every single sample routes hard to one expert. The
                      # max-weight mean and the argmax-share alive count see
                      # that; the mean usage vector does not.
                      "gating_max_weight": 0.0, "experts_alive": 0.0,
                      "encoder_grad_norm": 0.0}
        usage_sum = None
        stats_weight = 0
        for _ in range(num_rl_epochs):
            for _, student_batch, _, _ in data:
                (_, student_privileged_obs_batch, student_obs_histories_batch,
                 _, _, _) = student_batch
                n_student = student_obs_histories_batch.shape[0]
                for _ in range(self.num_encoder_epochs):
                    encoder_loss, latent_loss, load_balance_loss, gating_weights, \
                        encoder_predictions, encoder_targets = \
                        self._compute_encoder_losses(student_obs_histories_batch, student_privileged_obs_batch)

                    # Aggregate pre-update stats of this computation.
                    with torch.no_grad():
                        weights = gating_weights.detach()
                        gating_entropy = -torch.xlogy(weights, weights).sum(dim=-1).mean()
                        mean_usage = weights.mean(dim=0)
                        gating_max_weight = weights.max(dim=-1).values.mean()
                        # An expert is "alive" if it wins the argmax on at
                        # least 1% of the samples of this minibatch.
                        argmax_share = torch.bincount(
                            weights.argmax(dim=-1), minlength=weights.shape[1]
                        ).float() / weights.shape[0]
                        experts_alive = float((argmax_share > 0.01).sum())
                        pred, target = encoder_predictions.detach(), encoder_targets.detach()
                        latent_cosine = nn.functional.cosine_similarity(
                            pred, target, dim=-1).mean()
                        target_var = target.var(unbiased=False)
                        latent_ev = (1.0 - (target - pred).var(unbiased=False) / target_var
                                     if target_var > 1e-12 else torch.zeros(()))
                    stats_sums["latent_mse"] += latent_loss.item() * n_student
                    stats_sums["load_balance"] += load_balance_loss.item() * n_student
                    stats_sums["student_encoder_total"] += encoder_loss.item() * n_student
                    stats_sums["gating_entropy"] += gating_entropy.item() * n_student
                    stats_sums["gating_max_weight"] += gating_max_weight.item() * n_student
                    stats_sums["experts_alive"] += experts_alive * n_student
                    stats_sums["latent_cosine"] += latent_cosine.item() * n_student
                    stats_sums["latent_ev"] += float(latent_ev) * n_student
                    usage_sum = mean_usage * n_student if usage_sum is None else usage_sum + mean_usage * n_student
                    stats_weight += n_student

                    self.history_encoder_optimizer.zero_grad()
                    encoder_loss.backward()
                    encoder_grad_norm = nn.utils.clip_grad_norm_(
                        self.actor_critic.history_encoder.parameters(), self.max_grad_norm)
                    stats_sums["encoder_grad_norm"] += float(encoder_grad_norm) * n_student
                    self.history_encoder_optimizer.step()

        mean_expert_usage = usage_sum / stats_weight
        moe_stats = {key: value / stats_weight for key, value in stats_sums.items()}
        moe_stats.update({
            "effective_experts": math.exp(stats_sums["gating_entropy"] / stats_weight),
            "expert_usage_min": float(mean_expert_usage.min()),
            "expert_usage_max": float(mean_expert_usage.max()),
            "expert_usage_std": float(mean_expert_usage.std()),
            "expert_usage": mean_expert_usage.tolist(),
        })
        mean_latent_loss = moe_stats["latent_mse"]
        # Teacher-vs-student action gap: the end-to-end CTS convergence metric.
        # The latent losses above only say the encoder tracks the teacher
        # latent; they cannot say whether that translates into the same
        # actions -- which is the only thing that matters at deploy time,
        # since deployment runs the student arm alone.
        moe_stats["action_gap_mse"] = self._compute_action_gap(data)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_teacher_surrogate_loss /= num_updates
        mean_student_surrogate_loss /= num_updates
        # Standard PPO health telemetry: pre-update means over the RL pass,
        # plus the last adaptive-LR KL and the rollout explained variance.
        for key, total in rl_sums.items():
            moe_stats[key] = total / num_updates
        moe_stats["grad_norm"] = grad_norm_sum / num_updates
        moe_stats["kl_mean"] = self.last_kl_mean
        moe_stats["explained_variance"] = self._compute_explained_variance()
        # Per-role critic fit: one critic body serves both roles but on
        # different latents, and the student half is the one that has to work
        # at deploy time. A healthy aggregate EV can hide a student critic
        # that fits nothing.
        teacher_ev, student_ev = self._compute_role_explained_variance()
        moe_stats["explained_variance_teacher"] = teacher_ev
        moe_stats["explained_variance_student"] = student_ev
        # Raw per-role advantage scale (see RolloutStorageMoECTS: advantages
        # are normalized GLOBALLY, so the stored ones no longer show it).
        moe_stats.update(getattr(self.storage, "raw_adv_stats", None) or {})
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

        # Teacher arm (identical surrogate handling to PPO_CTS._compute_rl_loss)
        self.actor_critic.act(teacher_obs_batch, None, teacher_privileged_obs_batch, act_type='teacher')
        teacher_actions_log_prob_batch = self.actor_critic.get_actions_log_prob(teacher_actions_batch)
        teacher_entropy_batch = self.actor_critic.entropy
        teacher_mu_batch = self.actor_critic.action_mean
        teacher_sigma_batch = self.actor_critic.action_std

        # Student arm (the MoE history latent is computed under no_grad inside
        # ActorCriticMoECTS.update_distribution, so the surrogate never puts
        # gradients into the student encoder)
        self.actor_critic.act(student_obs_batch, student_obs_histories_batch, None, act_type='student')
        student_actions_log_prob_batch = self.actor_critic.get_actions_log_prob(student_actions_batch)
        student_entropy_batch = self.actor_critic.entropy
        student_mu_batch = self.actor_critic.action_mean
        student_sigma_batch = self.actor_critic.action_std

        ## KL over the CONCATENATED teacher+student batch (reference
        ## moe_cts.py:131-149 parity -- the host PPO_CTS computes it over the
        ## teacher arm only). Student old mu/sigma come from the
        ## sample-aligned student critic minibatch (the student role tuple
        ## carries no mu/sigma fields). The formula itself is inherited
        ## (PPO._adjust_learning_rate, identical to the reference's).
        self._adjust_learning_rate(
            torch.cat((teacher_sigma_batch, student_sigma_batch)),
            torch.cat((teacher_old_sigma_batch, student_critic.old_sigma)),
            torch.cat((teacher_mu_batch, student_mu_batch)),
            torch.cat((teacher_old_mu_batch, student_critic.old_mu)))

        ## Surrogate loss
        teacher_ratio = torch.exp(teacher_actions_log_prob_batch -
                                  torch.squeeze(teacher_old_actions_log_prob_batch))
        teacher_surrogate_loss = self._compute_surrogate_loss(teacher_ratio, teacher_advantages_batch)

        student_ratio = torch.exp(student_actions_log_prob_batch -
                                  torch.squeeze(student_old_actions_log_prob_batch))
        student_surrogate_loss = self._compute_surrogate_loss(student_ratio, student_advantages_batch)

        # Role-aware value loss (reference moe_cts.py:171-179): per-sample
        # losses over the concatenated teacher+student batch with a single
        # mean, i.e. teacher/student arms weighted by their sample counts.
        teacher_value_batch = self.actor_critic.evaluate(
            teacher_critic.critic_observations, is_teacher=True)
        student_value_batch = self.actor_critic.evaluate(
            student_critic.critic_observations, student_critic.observation_histories, is_teacher=False)
        teacher_value_losses = self._value_losses_per_sample(
            teacher_value_batch, teacher_critic.returns, teacher_critic.target_values)
        student_value_losses = self._value_losses_per_sample(
            student_value_batch, student_critic.returns, student_critic.target_values)
        value_losses = torch.cat((teacher_value_losses, student_value_losses), dim=0)
        value_loss = value_losses.mean()

        total_entropy_batch = torch.cat((teacher_entropy_batch, student_entropy_batch), dim=0)
        # Pre-gradient minibatch telemetry; update() aggregates it over the RL
        # pass. Stashed on the instance (not returned) so the pinned 4-tuple
        # return contract stays unchanged.
        self._minibatch_telemetry = self._compute_minibatch_telemetry(
            teacher_ratio, student_ratio, total_entropy_batch,
            teacher_entropy=teacher_entropy_batch,
            student_entropy=student_entropy_batch,
            teacher_value_losses=teacher_value_losses,
            student_value_losses=student_value_losses,
            teacher_kl=(teacher_sigma_batch, teacher_old_sigma_batch,
                        teacher_mu_batch, teacher_old_mu_batch),
            student_kl=(student_sigma_batch, student_critic.old_sigma,
                        student_mu_batch, student_critic.old_mu))

        loss = self.value_loss_coef * value_loss + \
            teacher_surrogate_loss + student_surrogate_loss \
            - self.entropy_coef * (total_entropy_batch.mean())

        return loss, teacher_surrogate_loss, student_surrogate_loss, value_loss

    def _compute_minibatch_telemetry(self, teacher_ratio, student_ratio, total_entropy_batch,
                                     teacher_entropy=None, student_entropy=None,
                                     teacher_value_losses=None, student_value_losses=None,
                                     teacher_kl=None, student_kl=None):
        """Pre-gradient PPO health stats for one minibatch (plain floats).

        clip_frac: fraction of the CONCATENATED teacher+student surrogate
        samples with |ratio - 1| > clip_param (PPO clip activity). entropy:
        concatenated teacher+student mean -- the same quantity the entropy
        bonus in the loss uses.

        Everything after the first three positional args is the PER-ROLE
        split of those same quantities (entropy, clip_frac, KL, value loss).
        The aggregate numbers alone cannot tell a healthy update from one
        where the teacher arm is fine and the student arm is diverging (or
        vice versa) -- the two roles share one actor and one critic head but
        see different latents, so they routinely fail independently. All of it
        is optional so a caller (or a test) that only wants the two aggregate
        keys keeps the original contract.
        """
        with torch.no_grad():
            ratio = torch.cat((teacher_ratio, student_ratio), dim=0)
            tel = {
                "entropy": float(total_entropy_batch.mean()),
                "clip_frac": float((torch.abs(ratio - 1.0) > self.clip_param).float().mean()),
                # Worst single-sample importance ratio deviation of the
                # minibatch: the first thing to look at when a run goes
                # unstable but the mean clip_frac still looks tame.
                "ratio_dev_max": float(torch.abs(ratio - 1.0).max()),
            }
            for role, role_ratio, role_entropy, role_value_losses, role_kl in (
                    ("teacher", teacher_ratio, teacher_entropy, teacher_value_losses, teacher_kl),
                    ("student", student_ratio, student_entropy, student_value_losses, student_kl)):
                tel[f"clip_frac_{role}"] = float(
                    (torch.abs(role_ratio - 1.0) > self.clip_param).float().mean())
                if role_entropy is not None:
                    tel[f"entropy_{role}"] = float(role_entropy.mean())
                if role_value_losses is not None:
                    tel[f"value_loss_{role}"] = float(role_value_losses.mean())
                if role_kl is not None:
                    tel[f"kl_{role}"] = self._kl_mean(*role_kl)
            return tel

    @staticmethod
    def _kl_mean(sigma, old_sigma, mu, old_mu):
        """Mean of PPO._adjust_learning_rate's KL estimator (plain float)."""
        with torch.inference_mode():
            kl = torch.sum(
                torch.log(sigma / old_sigma + 1.e-5) +
                (torch.square(old_sigma) + torch.square(old_mu - mu)) /
                (2.0 * torch.square(sigma)) - 0.5, axis=-1)
            return float(torch.mean(kl))

    def _adjust_learning_rate(self, sigma_batch, old_sigma_batch, mu_batch, old_mu_batch):
        """Capture the per-minibatch mean KL for telemetry, then defer to the
        base implementation.

        PPO._adjust_learning_rate owns the actual adaptive-LR update and is
        unchanged; it only computes the KL when schedule == 'adaptive'. Here
        the estimate is recorded whenever desired_kl is set, so the kl_mean
        stat also stays alive on fixed-schedule runs. last_kl_mean holds the
        LAST minibatch's value of the update (plain float), mirroring the
        base's formula exactly (same concatenated teacher+student inputs this
        class passes in _compute_rl_loss).
        """
        if self.desired_kl is not None:
            self.last_kl_mean = self._kl_mean(
                sigma_batch, old_sigma_batch, mu_batch, old_mu_batch)
        super()._adjust_learning_rate(sigma_batch, old_sigma_batch, mu_batch, old_mu_batch)

    def _compute_explained_variance(self):
        """Explained variance of the stored rollout values vs returns.

        Standard 1 - Var(target - pred) / Var(target) over the concatenated
        (env-order) role-aware rollout values and GAE returns held by the
        storage -- the pre-update value-function fit. Population variance
        (unbiased=False). None when the storage exposes no values/returns or
        the returns are (near-)constant; the runner skips None stats.
        """
        values = getattr(self.storage, "values", None)
        returns = getattr(self.storage, "returns", None)
        if values is None or returns is None:
            return None
        return self._explained_variance(returns.flatten(), values.flatten())

    @staticmethod
    def _explained_variance(y_true, y_pred):
        var_y = torch.var(y_true, unbiased=False)
        if var_y < 1e-12:
            return None
        return float(1.0 - torch.var(y_true - y_pred, unbiased=False) / var_y)

    def _compute_role_explained_variance(self):
        """(teacher_ev, student_ev) of the rollout values vs returns.

        Same quantity as _compute_explained_variance, restricted to each
        role's env columns (interleaved idxs, env-order buffers). (None, None)
        when the storage exposes no value buffers or the role mapping cannot
        be built for its env count -- telemetry never kills a run.
        """
        values = getattr(self.storage, "values", None)
        returns = getattr(self.storage, "returns", None)
        if values is None or returns is None:
            return None, None
        try:
            teacher_idxs, student_idxs = self._role_env_idxs(values.shape[1])
        except AssertionError:
            return None, None
        evs = []
        for idxs in (teacher_idxs, student_idxs):
            evs.append(self._explained_variance(
                returns[:, idxs].flatten(), values[:, idxs].flatten()))
        return evs[0], evs[1]

    def _compute_action_gap(self, data):
        """Mean squared teacher-vs-student action-mean gap, or None.

        Evaluated on the FIRST student minibatch only (one extra no_grad
        forward per update, negligible next to the RL pass) -- the student
        role tuple carries both the obs history the student encoder needs and
        the privileged obs the teacher encoder needs for the same
        transitions, so the two arms are compared on identical states.
        Uses act_teacher/act_student, which return the action means directly
        and do NOT touch self.distribution.
        """
        if not data:
            return None
        _, student_batch, _, _ = data[0]
        (student_obs, student_privileged_obs, student_obs_histories, *_) = student_batch
        with torch.no_grad():
            teacher_actions = self.actor_critic.act_teacher(student_obs, student_privileged_obs)
            student_actions = self.actor_critic.act_student(student_obs, student_obs_histories)
            return float((teacher_actions - student_actions).pow(2).mean())

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

        Returns (total, latent_mse, load_balance, gating_weights,
        predictions, targets); the components feed the moe_stats aggregation
        in update() (the last two carry the scale-free distillation stats).
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
        return (total, latent_loss, load_balance_loss, gating_weights,
                encoder_predictions, encoder_targets)
