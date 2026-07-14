# PPO for the Hybrid Internal Model (HIM).
# Subclasses the local PPO to reuse its surrogate/value-loss and adaptive-LR
# helpers (as PPO_DreamWaQ does). The estimator is trained by its own optimizer
# inside the actor-critic; the PPO optimizer covers only actor + critic + std.

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules import HIMActorCritic
from rsl_rl.storage import HIMRolloutStorage


class PPO_HIM(PPO):
    actor_critic: HIMActorCritic

    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 use_spo=False,
                 device='cpu'):

        super().__init__(
            actor_critic,
            num_learning_epochs,
            num_mini_batches,
            clip_param,
            gamma,
            lam,
            value_loss_coef,
            entropy_coef,
            learning_rate,
            max_grad_norm,
            use_clipped_value_loss,
            schedule,
            desired_kl,
            use_spo,
            device,
        )

        # PPO covers actor + critic + action noise only. The estimator keeps its
        # own optimizer (built in HIMEstimator) and the actor detaches its output,
        # so estimator params receive no PPO gradient.
        self.rl_parameters = list(self.actor_critic.actor.parameters()) + \
            list(self.actor_critic.critic.parameters()) + \
            [self.actor_critic.std]
        self.optimizer = optim.Adam(self.rl_parameters, lr=learning_rate)
        self.transition = HIMRolloutStorage.Transition()

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape,
                     critic_obs_shape, action_shape):
        self.storage = HIMRolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape,
            critic_obs_shape, action_shape, self.device)

    def process_env_step(self, rewards, dones, infos, next_critic_obs):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.next_critic_observations = next_critic_obs.clone()
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def update(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_estimation_loss = 0.0
        mean_swap_loss = 0.0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, \
            returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, \
            hid_states_batch, masks_batch, next_critic_obs_batch, terminated_batch in generator:

            # PPO RL loss (reuses the base surrogate/value machinery)
            loss, surrogate_loss, value_loss = self._compute_rl_loss(
                obs_batch, critic_obs_batch, actions_batch, target_values_batch,
                advantages_batch, returns_batch, old_actions_log_prob_batch,
                old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.rl_parameters, self.max_grad_norm)
            self.optimizer.step()

            # Estimator step (its own optimizer, inside the actor-critic)
            estimation_loss, swap_loss = self.actor_critic.estimator.update(
                obs_batch, next_critic_obs_batch, terminated_batch)

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_estimation_loss += estimation_loss
            mean_swap_loss += swap_loss

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimation_loss /= num_updates
        mean_swap_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss
