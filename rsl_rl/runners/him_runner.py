# Training runner for the Hybrid Internal Model (HIM).
# HIM follows the standard single-tensor contract (obs = stacked history,
# critic_obs = five-frame privileged history), so this runner keeps the base
# OnPolicyRunner fairness machinery: the iteration-based command_schedule and the
# Eval-V2 / best_tracking.pt selection. learn() is copied from
# OnPolicyRunner.learn() with HIM's next_critic_obs rollout + estimator logging.

import time
import os
from collections import deque
import statistics

import torch

from rsl_rl.algorithms import PPO_HIM
from rsl_rl.modules import HIMActorCritic
from rsl_rl.env import VecEnv
from .on_policy_runner import OnPolicyRunner


class HIMRunner(OnPolicyRunner):

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device='cpu'):
        super().__init__(env, train_cfg, log_dir, device)

    def _init_agent_and_algo(self):
        actor_critic_class = eval(self.cfg["policy_class_name"])  # HIMActorCritic
        actor_critic: HIMActorCritic = actor_critic_class(
            self.env.num_obs,
            self.env.num_privileged_obs,
            self.env.num_one_step_obs,
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO_HIM
        self.alg: PPO_HIM = alg_class(actor_critic, device=self.device, **self.alg_cfg)

    def _init_storage(self):
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
        )

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        self._pre_learn(init_at_random_ep_len)
        if self.training_seed is not None:
            print(f"[train] seed={self.training_seed} "
                  f"start_iter={self.current_learning_iteration}")
        # land on the correct schedule stage for the (possibly resumed) start iter
        self._apply_command_schedule(self.current_learning_iteration)
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # advance the command schedule if this iteration crosses a stage boundary
            self._apply_command_schedule(it)
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    # post-step critic obs is the HIM estimator's supervision target
                    self.alg.process_env_step(rewards, dones, infos, critic_obs)

                    if self.log_dir is not None:
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            completed_iteration = self.completed_iteration(it)
            if completed_iteration % self.save_interval == 0:
                assert self.log_dir is not None
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(completed_iteration)),
                          iteration=completed_iteration)

            # Periodic in-distribution eval -> Eval/* + best_tracking.pt selection.
            if self.eval_interval > 0 and completed_iteration % self.eval_interval == 0:
                with torch.inference_mode():
                    self._run_eval(completed_iteration)
                obs = self.env.get_observations().to(self.device)
                privileged_obs = self.env.get_privileged_observations()
                critic_obs = (privileged_obs if privileged_obs is not None else obs).to(self.device)
                cur_reward_sum.zero_()
                cur_episode_length.zero_()
                self.alg.actor_critic.train()

            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        assert self.log_dir is not None
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        # Extend the base log with the HIM estimator losses, then delegate.
        if self.writer is not None:
            self.writer.add_scalar('Loss/Estimation', locs['mean_estimation_loss'], locs['it'])
            self.writer.add_scalar('Loss/Swap', locs['mean_swap_loss'], locs['it'])
        super().log(locs, width, pad)

    # HIM keeps the standard single-tensor obs contract, so the base
    # StandardAdapter / act_inference eval path already applies. It only adds the
    # estimator: its Adam moments must survive a resume, and the estimator weights
    # must load strictly for deployment (act_inference reads estimator latents).
    def _aux_optimizers(self):
        return {"estimator_optimizer_state_dict": self.alg.actor_critic.estimator.optimizer}

    def deploy_state_prefixes(self):
        return ("actor.", "estimator.")
