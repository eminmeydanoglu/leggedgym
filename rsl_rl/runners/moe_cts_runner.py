# MoE-CTS runner for the go2_rl_gym (wty-yy, RSS 2026) MoE port.
#
# Subclasses CTSRunner and overrides only the MoE-specific deltas:
# - storage: RolloutStorageMoECTS (rsl_rl/storage/rollout_storage_moe_cts.py)
#   is wired in _init_storage (lazy import so this module stays importable
#   while the storage port is in flight).
# - bootstrap: PPO_MOE_CTS.compute_returns is role-aware and needs the final
#   obs_history next to critic_obs (reference: go2_rl_gym
#   on_policy_runner_cts.py, compute_returns(privileged_obs, history)).
# - update(): PPO_MOE_CTS.update returns a 5-tuple
#   (mean_value_loss, mean_teacher_surrogate_loss, mean_student_surrogate_loss,
#   mean_latent_loss, moe_stats). The first four feed the unchanged CTS log
#   channels; moe_stats is already aggregated by the algorithm and is only
#   logged here (TensorBoard + terminal).
# - checkpoints: same host convention as CTSRunner (OnPolicyRunner.save/load),
#   plus the history-encoder optimizer state so a resume restores its Adam
#   moments (same pattern as TSRunner._aux_optimizers).
# - telemetry: per-role Train/* reward/episode-length split (reference
#   go2_rl_gym on_policy_runner_cts.py:138-174, 245-254) on top of the
#   unchanged aggregate CTS channels; done-env sums are split by the
#   algorithm's interleaved teacher_env_idxs. The algorithm's moe_stats also
#   carry the standard PPO health metrics (Policy/entropy, Policy/kl_mean,
#   Policy/clip_frac, Loss/explained_variance).
# - logging dedup: the inherited CTS 'Loss/reconstruction' channel is
#   identical to 'Loss/latent_mse' on this arm (both are PPO_MOE_CTS's
#   mean_latent_loss), so MoECTSRunner.log filters that one tag out of the
#   CTSRunner.log call. CTSRunner is shared and stays untouched, so plain
#   CTS tasks keep the channel.
#
# get_inference_policy / deploy behaviour is inherited from CTSRunner
# (student policy via actor_critic.act_student) unchanged.

import time
import os
import statistics
from collections import deque

import torch

from rsl_rl.algorithms import PPO_MOE_CTS  # noqa: F401  eval() in _init_agent_and_algo
from rsl_rl.modules import ActorCriticMoECTS  # noqa: F401  eval() in _init_agent_and_algo
from .cts_runner import CTSRunner


class _DropScalarTags:
    """SummaryWriter proxy that drops add_scalar calls for the given tags.

    MoECTSRunner.log uses it to suppress the duplicate 'Loss/reconstruction'
    CTS channel: on this arm that value IS moe_stats['latent_mse']
    (PPO_MOE_CTS.update returns mean_latent_loss in the 4th slot), already
    logged as 'Loss/latent_mse' by _log_moe_stats. CTSRunner itself is shared
    and stays untouched, so plain CTS tasks keep the channel.
    """

    def __init__(self, writer, tags):
        self._writer = writer
        self._tags = tags

    def add_scalar(self, tag, *args, **kwargs):
        if tag not in self._tags:
            self._writer.add_scalar(tag, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._writer, name)


class MoECTSRunner(CTSRunner):
    """CTSRunner variant for ActorCriticMoECTS + PPO_MOE_CTS."""

    def _init_storage(self):
        # Same shapes as CTSRunner; the MoE arm requires RolloutStorageMoECTS.
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env,
                              [self.env.num_obs], [self.env.num_privileged_obs],
                              [self.env.num_history_obs], [self.env.num_critic_obs], [self.env.num_actions])
        from rsl_rl.storage.rollout_storage_moe_cts import RolloutStorageMoECTS
        if not isinstance(self.alg.storage, RolloutStorageMoECTS):
            # PPO_MOE_CTS.init_storage may still wire the base CTS storage while
            # the port is in flight; enforce the MoE storage contract, reusing
            # the interleaved role idxs the algorithm built for this num_envs.
            self.alg.storage = RolloutStorageMoECTS(
                self.env.num_envs, self.alg.num_teacher, self.num_steps_per_env,
                [self.env.num_obs], [self.env.num_privileged_obs],
                [self.env.num_history_obs], [self.env.num_critic_obs],
                [self.env.num_actions], self.device,
                teacher_env_idxs=getattr(self.alg, "teacher_env_idxs", None),
                student_env_idxs=getattr(self.alg, "student_env_idxs", None))

    def deploy_state_prefixes(self):
        # Deployment runs the student arm: actor + MoE history encoder. The
        # privilege encoder and critic are training-only, and the imported
        # go2_rl_gym policies do not even contain them (see
        # legged_gym/scripts/import_go2_rl_gym_policy.py) -- listing them here
        # would make every such checkpoint fail the strict deploy load.
        return ("actor.", "history_encoder.")

    def get_eval_adapter(self, device=None):
        # Go2MoECTS.get_observations() -> (obs, privileged, history, critic) and
        # step() -> (..., rewards, dones, infos), i.e. exactly the tuple layout
        # HistoryObsAdapter indexes (obs_index=0, history_index=2).
        from rsl_rl.runners.eval_adapter import HistoryObsAdapter
        self.alg.actor_critic.eval()
        dev = torch.device(device) if device is not None else self.device
        if device is not None:
            self.alg.actor_critic.to(device)
        return HistoryObsAdapter(self.alg.actor_critic.act_student, dev)

    def _aux_optimizers(self):
        # PPO_MOE_CTS trains the MoE student encoder with a dedicated optimizer
        # (PPO_CTS.history_encoder_optimizer); persist its moments on resume.
        opt = getattr(self.alg, 'history_encoder_optimizer', None)
        return {"history_encoder_optimizer_state_dict": opt} if opt is not None else {}

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        self._pre_learn(init_at_random_ep_len)
        # Sync the env's ratio-based curricula with the effective budget
        # (incl. any --max_iterations CLI override); no-op for envs without
        # the wty substrate.
        if hasattr(self.env, "set_wty_total_iterations"):
            self.env.set_wty_total_iterations(num_learning_iterations)
        obs, privileged_obs, obs_history, critic_obs = self.env.get_observations()
        obs, privileged_obs, obs_history, critic_obs = obs.to(self.device), privileged_obs.to(self.device), \
            obs_history.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        # Per-role training-health split (reference on_policy_runner_cts.py:138-141).
        # The aggregate buffers stay fed so the CTSRunner.log channels
        # (Train/mean_reward, Train/mean_episode_length) are unchanged.
        teacher_rewbuffer = deque(maxlen=100)
        teacher_lenbuffer = deque(maxlen=100)
        student_rewbuffer = deque(maxlen=100)
        student_lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        # Interleaved role ids, built by the algorithm at init_storage time
        # (compute_role_env_idxs); students are the complement of these.
        teacher_env_idxs = self.alg.teacher_env_idxs

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, privileged_obs, obs_history, critic_obs)
                    obs, privileged_obs, obs_history, critic_obs, rewards, dones, infos = self.env.step(actions)
                    obs, privileged_obs, obs_history, rewards, dones, critic_obs = obs.to(self.device), \
                        privileged_obs.to(self.device), obs_history.to(self.device), rewards.to(self.device), dones.to(self.device), critic_obs.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        if new_ids.shape[0]:
                            # Reference on_policy_runner_cts.py:167-174: split the
                            # done envs by role (interleaved idxs, NOT contiguous
                            # blocks) so per-role health is visible next to the
                            # aggregate above.
                            done_ids = new_ids.flatten()
                            is_teacher = torch.isin(done_ids, teacher_env_idxs)
                            teacher_ids = done_ids[is_teacher]
                            student_ids = done_ids[~is_teacher]
                            teacher_rewbuffer.extend(cur_reward_sum[teacher_ids].cpu().numpy().tolist())
                            teacher_lenbuffer.extend(cur_episode_length[teacher_ids].cpu().numpy().tolist())
                            student_rewbuffer.extend(cur_reward_sum[student_ids].cpu().numpy().tolist())
                            student_lenbuffer.extend(cur_episode_length[student_ids].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                # MoE bootstrap is role-aware (interleaved teacher/student env
                # ids, reference moe_cts.py:96-102) and needs the latest
                # obs_history next to critic_obs.
                self.alg.compute_returns(critic_obs, obs_history)

            # 5-tuple: the 4th value is the latent (student-encoder) distillation
            # loss; it feeds the existing 'Loss/reconstruction' CTS log channel,
            # so the local name matches the CTS log contract on purpose.
            mean_value_loss, mean_teacher_surrogate_loss, \
                mean_student_surrogate_loss, mean_reconstruction_loss, moe_stats = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        # Dedup: CTSRunner.log would write Loss/reconstruction from
        # locs['mean_reconstruction_loss'], identical to Loss/latent_mse on
        # this arm. Filter that one tag out of the base call only; the base's
        # terminal print of the value is left as-is.
        writer = self.writer
        self.writer = _DropScalarTags(writer, {"Loss/reconstruction"})
        try:
            super().log(locs, width, pad)  # CTS TensorBoard + terminal block unchanged
        finally:
            self.writer = writer
        self._log_role_split_stats(locs, pad)
        self._log_moe_stats(locs.get('moe_stats') or {}, locs['it'], pad)

    def _log_role_split_stats(self, locs, pad=35):
        """Per-role Train/* telemetry (reference on_policy_runner_cts.py:245-254).

        The buffers are filled by MoECTSRunner.learn and arrive via locs; a
        missing/empty buffer (e.g. a hand-built locs in tests) only skips its
        own channels. The aggregate Train/mean_reward channels are written by
        CTSRunner.log and are not touched here. tot_time is already updated by
        super().log, so the /time variants use the same step as the reference.
        """
        it = locs['it']
        lines = []
        for role in ('teacher', 'student'):
            rewbuffer = locs.get(f'{role}_rewbuffer') or []
            lenbuffer = locs.get(f'{role}_lenbuffer') or []
            if len(rewbuffer) == 0:
                continue
            mean_reward = statistics.mean(rewbuffer)
            mean_length = statistics.mean(lenbuffer)
            self.writer.add_scalar(f'Train/mean_{role}_reward', mean_reward, it)
            self.writer.add_scalar(f'Train/mean_{role}_episode_length', mean_length, it)
            self.writer.add_scalar(f'Train/mean_{role}_reward/time', mean_reward, self.tot_time)
            self.writer.add_scalar(f'Train/mean_{role}_episode_length/time', mean_length, self.tot_time)
            lines += [f"{f'Mean {role} reward:':>{pad}} {mean_reward:.2f}",
                      f"{f'Mean {role} episode length:':>{pad}} {mean_length:.2f}"]
        if lines:
            print('\n'.join(lines))

    def _log_moe_stats(self, moe_stats, it, pad=35):
        """Log the pre-aggregated MoE telemetry from PPO_MOE_CTS.update().

        No aggregation here: the algorithm owns the numbers, the runner only
        writes them out. Missing keys are skipped so a partial stats dict never
        kills a training run mid-logging.
        """
        if not moe_stats:
            return
        for key in ('latent_mse', 'load_balance', 'student_encoder_total',
                    'value_loss_teacher', 'value_loss_student',
                    'grad_norm', 'encoder_grad_norm'):
            if moe_stats.get(key) is not None:
                self.writer.add_scalar('Loss/' + key, moe_stats[key], it)
        # Standard PPO health telemetry (aggregated by PPO_MOE_CTS.update).
        for key in ('explained_variance', 'explained_variance_teacher',
                    'explained_variance_student'):
            if moe_stats.get(key) is not None:
                self.writer.add_scalar('Loss/' + key, moe_stats[key], it)
        for key in ('entropy', 'kl_mean', 'clip_frac', 'ratio_dev_max',
                    'entropy_teacher', 'entropy_student',
                    'kl_teacher', 'kl_student',
                    'clip_frac_teacher', 'clip_frac_student'):
            if moe_stats.get(key) is not None:
                self.writer.add_scalar('Policy/' + key, moe_stats[key], it)
        # Raw per-role advantage scale: the storage normalizes advantages
        # globally, so this is the only place the per-role spread is visible.
        for key in ('adv_mean_teacher', 'adv_mean_student', 'adv_std_teacher',
                    'adv_std_student', 'adv_std_global', 'adv_std_ratio'):
            if moe_stats.get(key) is not None:
                self.writer.add_scalar('Train/' + key, moe_stats[key], it)
        for key in ('gating_entropy', 'effective_experts',
                    'expert_usage_min', 'expert_usage_max', 'expert_usage_std',
                    'gating_max_weight', 'experts_alive',
                    'latent_cosine', 'latent_ev', 'action_gap_mse'):
            if moe_stats.get(key) is not None:
                self.writer.add_scalar('MoE/' + key, moe_stats[key], it)
        # One scalar per expert; len(expert_usage) == policy expert_num by
        # contract, so the count is never hardcoded here.
        for i, usage in enumerate(moe_stats.get('expert_usage') or []):
            self.writer.add_scalar(f'MoE/expert_usage_{i}', usage, it)

        def _f(v):
            return v.item() if isinstance(v, torch.Tensor) else float(v)

        term_keys = [('latent_mse', 'MoE latent mse:'),
                     ('latent_cosine', 'MoE latent cosine:'),
                     ('latent_ev', 'MoE latent expl. var:'),
                     ('action_gap_mse', 'MoE teacher-student action gap:'),
                     ('load_balance', 'MoE load balance:'),
                     ('student_encoder_total', 'MoE student encoder total:'),
                     ('gating_entropy', 'MoE gating entropy:'),
                     ('effective_experts', 'MoE effective experts:'),
                     ('experts_alive', 'MoE experts alive (argmax):'),
                     ('gating_max_weight', 'MoE gating max weight:'),
                     ('expert_usage_min', 'MoE expert usage min:'),
                     ('expert_usage_max', 'MoE expert usage max:'),
                     ('expert_usage_std', 'MoE expert usage std:'),
                     ('explained_variance_teacher', 'Expl. var (teacher):'),
                     ('explained_variance_student', 'Expl. var (student):'),
                     ('kl_teacher', 'KL (teacher):'),
                     ('kl_student', 'KL (student):'),
                     ('grad_norm', 'Grad norm (RL, pre-clip):'),
                     ('encoder_grad_norm', 'Grad norm (encoder, pre-clip):'),
                     ('adv_std_ratio', 'Adv std ratio (student/teacher):')]
        lines = [f"{label:>{pad}} {_f(moe_stats[key]):.4f}"
                 for key, label in term_keys if moe_stats.get(key) is not None]
        if lines:
            print('\n'.join(lines))
