# Hybrid Internal Model (HIM) actor-critic.
# Ported from InternRobotics/HIMLoco (arXiv:2312.11460) and adapted to
# LeggedGym-Ex conventions. The estimator lives inside the actor-critic and the
# actor observation IS the stacked history, so act_inference(obs_history) takes a
# single tensor -> the shared eval harness (policy(obs)) works with no edits.

import torch
import torch.nn as nn
from torch.distributions import Normal

from .actor_critic import get_activation
from .him_estimator import HIMEstimator


class HIMActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self,
                 num_actor_obs,
                 num_critic_obs,
                 num_one_step_obs,
                 num_actions,
                 actor_hidden_dims=[512, 256, 128],
                 critic_hidden_dims=[512, 256, 128],
                 activation='elu',
                 init_noise_std=1.0,
                 **estimator_cfg):
        super().__init__()

        self.num_one_step_obs = num_one_step_obs

        # Internal estimator. temporal_steps is derived from the actor obs (the
        # stacked history) and the single-frame width, not hardcoded.
        self.estimator = HIMEstimator(
            temporal_steps=num_actor_obs // num_one_step_obs,
            num_one_step_obs=num_one_step_obs,
            activation=activation,
            **estimator_cfg,
        )

        # Actor input = current one-step frame + estimated [vel(3), latent].
        mlp_input_dim_a = num_one_step_obs + 3 + self.estimator.num_latent
        mlp_input_dim_c = num_critic_obs

        activation_fn = get_activation(activation)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation_fn)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation_fn)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Estimator Encoder: {self.estimator.encoder}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _actor_input(self, obs_history):
        """current one-step frame + detached estimator (vel, latent)."""
        vel, latent = self.estimator.get_latent(obs_history)
        obs = obs_history[:, :self.num_one_step_obs]
        return torch.cat((obs, vel, latent), dim=-1)

    def update_distribution(self, obs_history):
        mean = self.actor(self._actor_input(obs_history))
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, obs_history, **kwargs):
        self.update_distribution(obs_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_history, **kwargs):
        return self.actor(self._actor_input(obs_history))

    def evaluate(self, critic_obs, **kwargs):
        return self.critic(critic_obs)
