# Hybrid Internal Model (HIM) estimator.
# Ported from InternRobotics/HIMLoco (arXiv:2312.11460) and adapted to
# LeggedGym-Ex conventions (reuse the local get_activation, document our critic
# layout contract, mask the estimation loss by episode termination).

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .actor_critic import get_activation


class HIMEstimator(nn.Module):
    """HIM estimator: an explicit base-velocity head + an implicit SwAV latent.

    The encoder consumes the stacked observation history and outputs
    ``[vel_hat(3), latent(num_latent)]``. The latent is trained with a
    Sinkhorn/SwAV swapped-prediction loss against a target encoder that reads the
    *next* one-step observation, so the latent captures short-horizon dynamics.
    The explicit head regresses the next base linear velocity.

    Critic-obs layout contract (see Go2BenchHIM.compute_observations):
        next_critic_obs = [base_lin_vel(3), one_step_obs(num_one_step_obs), P(...)]
    so ``vel  = next_critic_obs[:, 0:3]`` and
       ``next_obs = next_critic_obs[:, 3:3 + num_one_step_obs]``.
    """

    def __init__(self,
                 temporal_steps,
                 num_one_step_obs,
                 enc_hidden_dims=[128, 64, 16],
                 tar_hidden_dims=[128, 64],
                 activation='elu',
                 learning_rate=1e-3,
                 max_grad_norm=10.0,
                 num_prototype=32,
                 temperature=3.0,
                 **kwargs):
        if kwargs:
            print("HIMEstimator.__init__ got unexpected arguments, which will be ignored: "
                  + str([key for key in kwargs.keys()]))
        super().__init__()

        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.num_latent = enc_hidden_dims[-1]
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature
        self.num_prototype = num_prototype
        self.learning_rate = learning_rate

        activation_fn = get_activation(activation)

        # Encoder: history -> [vel(3), latent(num_latent)]
        enc_input_dim = self.temporal_steps * self.num_one_step_obs
        enc_layers = []
        for l in range(len(enc_hidden_dims) - 1):
            enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[l]), activation_fn]
            enc_input_dim = enc_hidden_dims[l]
        enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[-1] + 3)]
        self.encoder = nn.Sequential(*enc_layers)

        # Target encoder: next one-step obs -> latent (SwAV target branch)
        tar_input_dim = self.num_one_step_obs
        tar_layers = []
        for l in range(len(tar_hidden_dims)):
            tar_layers += [nn.Linear(tar_input_dim, tar_hidden_dims[l]), activation_fn]
            tar_input_dim = tar_hidden_dims[l]
        tar_layers += [nn.Linear(tar_input_dim, enc_hidden_dims[-1])]
        self.target = nn.Sequential(*tar_layers)

        # Prototypes for the Sinkhorn/SwAV clustering
        self.proto = nn.Linear(enc_hidden_dims[-1], num_prototype, bias=False)

        # The estimator is trained by its own optimizer; the PPO path detaches
        # the estimator output so policy gradients never reach these parameters.
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def get_latent(self, obs_history):
        """Detached (vel, latent) for feeding the actor at act/inference time."""
        vel, z = self.encode(obs_history)
        return vel.detach(), z.detach()

    def encode(self, obs_history):
        out = self.encoder(obs_history.detach())
        vel, z = out[..., :3], out[..., 3:]
        z = F.normalize(z, dim=-1, p=2)
        return vel, z

    def update(self, obs_history, next_critic_obs, terminated):
        """One estimator optimization step.

        Args:
            obs_history: stacked history the encoder reads. Shape (B, T*one_step).
            next_critic_obs: critic obs AFTER the env step, laid out as
                ``[base_lin_vel(3), one_step_obs(num_one_step_obs), P(...)]``.
            terminated: ``(1 - dones)`` mask, shape (B, 1). Cross-boundary
                velocity targets are invalid (env resets on done), so the
                explicit estimation MSE is masked by it.

        Returns:
            (estimation_loss, swap_loss) as python floats.
        """
        vel = next_critic_obs[:, 0:3].detach()
        next_obs = next_critic_obs[:, 3:3 + self.num_one_step_obs].detach()

        out = self.encoder(obs_history)
        pred_vel, z_s = out[..., :3], out[..., 3:]
        z_s = F.normalize(z_s, dim=-1, p=2)
        z_t = self.target(next_obs)
        z_t = F.normalize(z_t, dim=-1, p=2)

        # normalize the prototypes onto the unit sphere before scoring
        with torch.no_grad():
            w = self.proto.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.proto.weight.copy_(w)

        score_s = self.proto(z_s)
        score_t = self.proto(z_t)
        with torch.no_grad():
            q_s = sinkhorn(score_s)
            q_t = sinkhorn(score_t)
        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)

        # Swapped prediction: q_s supervises p_t and vice versa. Left over the
        # full batch (minor simplification vs. masking; consistent w/ ref impl).
        swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).sum(dim=-1).mean()
        # Explicit velocity estimation, masked to valid (non-terminal) steps.
        estimation_loss = F.mse_loss(pred_vel * terminated, vel * terminated)
        losses = estimation_loss + swap_loss

        self.optimizer.zero_grad()
        losses.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return estimation_loss.item(), swap_loss.item()


def sinkhorn(scores, eps=0.05, niters=3):
    """Sinkhorn-Knopp normalization producing doubly-stochastic assignments.

    Returns a (B, K) matrix whose columns (prototypes) form a uniform marginal
    and whose rows sum to 1, used as the SwAV clustering target.
    """
    Q = torch.exp(scores / eps).t()
    Q = Q / torch.sum(Q)
    K, B = Q.shape
    r = torch.ones(K, device=Q.device) / K
    c = torch.ones(B, device=Q.device) / B
    for _ in range(niters):
        u = torch.sum(Q, dim=1)
        Q *= (r / u).unsqueeze(1)
        Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
    return (Q / torch.sum(Q, dim=0, keepdim=True)).t()
