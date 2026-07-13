"""Per-env metric accumulation for the eval harness.

The harness runs `auto_reset=True`, so the env re-spawns terminated robots
*inside* `env.step()` and zeroes its own `episode_length_buf` before returning.
That means we cannot read survival time off the env after the fact -- we keep our
own per-env step counter here, mirroring the env's episode boundaries via the
`reset_buf` / `time_out_buf` flags returned each step.

Terminology (matches legged_robot.check_termination):
  - done    = reset_buf                       (episode ended, any reason)
  - timeout = done &  time_out_buf            (survived the full episode)
  - fall    = done & ~time_out_buf            (terminated early == failure)

`fall` is the metric that actually separates methods under OOD; tracking error
and return saturate long before robots start falling.
"""

import torch


class MetricAccumulator:
    """Accumulates episode- and step-level statistics for every env in parallel.

    All buffers are shape (num_envs,). Call `update(...)` once per env step, then
    `compute()` at the end to get per-env scalars. `T` is the number of steps the
    harness actually ran (used for right-censoring envs that never terminated).
    """

    def __init__(self, num_envs: int, device, lin_err_threshold: float = 0.25):
        self.num_envs = num_envs
        self.device = device
        z = lambda dtype=torch.float: torch.zeros(num_envs, dtype=dtype, device=device)

        # running (current, not-yet-finished episode)
        self._alive_steps = z(torch.long)     # steps since last reset (our own count)
        self._return_run = z()                # reward sum in current episode
        self._reward_step_sum = z()           # all rewards, including censored episodes

        # completed-episode accumulators
        self._ep_len_sum = z()                # sum of lengths of finished episodes
        self._ep_count = z(torch.long)        # number of finished episodes
        self._fall_count = z(torch.long)      # finished episodes ending in a fall
        self._timeout_count = z(torch.long)   # finished episodes ending in a timeout
        self._ep_return_sum = z()             # sum of returns of finished episodes
        self._ever_fell = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # step-level accumulators (averaged over all steps)
        self._lin_err_sum = z()               # |cmd_xy - v_xy| summed over steps
        self._ang_err_sum = z()               # |cmd_yaw - w_yaw| summed over steps
        self._lin_err_sq_sum = z()            # squared tracking error (RMS / spikes)
        self._lin_err_bad_sum = z()           # steps above an explicit tolerance
        self._tilt_sum = z()                   # ||projected_gravity_xy|| (sine tilt)
        self._torque_sq_sum = z()              # mean_j torque^2
        self._mech_power_sum = z()             # sum_j |torque * joint_velocity|
        self._action_rate_sum = z()            # mean_j |a_t - a_(t-1)|
        self._foot_slip_sum = z()              # contact-foot horizontal speed
        self.lin_err_threshold = float(lin_err_threshold)
        self._steps = 0                       # global step counter (same for all envs)

    @torch.no_grad()
    def update(
        self, rew, reset_buf, time_out_buf, lin_err, ang_err, *,
        tilt=None, torques=None, dof_vel=None, actions=None, last_actions=None,
        feet_vel=None, feet_contact=None,
    ):
        """Ingest one env step.

        Args:
            rew:          (num_envs,) reward this step.
            reset_buf:    (num_envs,) 1 where the env terminated this step.
            time_out_buf: (num_envs,) 1 where the termination was a timeout.
            lin_err:      (num_envs,) |command_xy - base_lin_vel_xy| this step.
            ang_err:      (num_envs,) |command_yaw - base_ang_vel_yaw| this step.
        """
        done = reset_buf.bool()
        timeout = done & time_out_buf.bool()
        fall = done & ~time_out_buf.bool()

        self._alive_steps += 1
        self._return_run += rew
        self._reward_step_sum += rew
        self._lin_err_sum += lin_err
        self._ang_err_sum += ang_err
        self._lin_err_sq_sum += lin_err.square()
        self._lin_err_bad_sum += (lin_err > self.lin_err_threshold).float()
        if tilt is not None:
            self._tilt_sum += tilt
        if torques is not None:
            self._torque_sq_sum += torques.square().mean(dim=1)
        if torques is not None and dof_vel is not None:
            self._mech_power_sum += (torques * dof_vel).abs().sum(dim=1)
        if actions is not None and last_actions is not None:
            self._action_rate_sum += (actions - last_actions).abs().mean(dim=1)
        if feet_vel is not None and feet_contact is not None:
            slip_xy = torch.norm(feet_vel[..., :2], dim=-1)
            contact = feet_contact.float()
            self._foot_slip_sum += (
                (slip_xy * contact).sum(dim=1) / contact.sum(dim=1).clamp(min=1.0)
            )
        self._steps += 1

        # book a finished episode wherever `done`
        self._ep_len_sum += torch.where(done, self._alive_steps.float(), torch.zeros_like(self._return_run))
        self._ep_count += done.long()
        self._ep_return_sum += torch.where(done, self._return_run, torch.zeros_like(self._return_run))
        self._fall_count += fall.long()
        self._timeout_count += timeout.long()
        self._ever_fell |= fall

        # reset running trackers for envs that just finished
        zero_l = torch.zeros_like(self._alive_steps)
        zero_f = torch.zeros_like(self._return_run)
        self._alive_steps = torch.where(done, zero_l, self._alive_steps)
        self._return_run = torch.where(done, zero_f, self._return_run)

    @torch.no_grad()
    def compute(self) -> dict:
        """Return per-env metric tensors, each shape (num_envs,).

        Keys:
            fall_rate       : falls / finished episodes (0 if none finished).
            never_fell      : 1.0 if the env never fell over the whole run.
            mean_ep_len     : mean length of finished episodes, in steps
                              (right-censored: envs with no finished episode get T).
            falls_per_1k    : falls per 1000 steps (density, robust to episode count).
            mean_return     : mean return per finished episode.
            tracking_lin_err: mean |cmd_xy - v_xy| over all steps.
            tracking_ang_err: mean |cmd_yaw - w_yaw| over all steps.
        """
        ep = self._ep_count.clamp(min=1).float()
        steps = max(self._steps, 1)

        mean_ep_len = torch.where(
            self._ep_count > 0,
            self._ep_len_sum / ep,
            torch.full_like(self._ep_len_sum, float(self._steps)),  # censored
        )
        return {
            "fall_rate": self._fall_count.float() / ep,
            # V2 headline: a binary event over the fixed measurement window,
            # never divided by how many episodes auto-reset happened to create.
            "ever_fell": self._ever_fell.float(),
            "never_fell": (self._fall_count == 0).float(),
            "mean_ep_len": mean_ep_len,
            "falls_per_1k": self._fall_count.float() / steps * 1000.0,
            "mean_return": self._ep_return_sum / ep,
            "return_per_step": self._reward_step_sum / steps,
            "tracking_lin_err": self._lin_err_sum / steps,
            "tracking_ang_err": self._ang_err_sum / steps,
            "tracking_lin_rmse": torch.sqrt(self._lin_err_sq_sum / steps),
            "tracking_lin_bad_frac": self._lin_err_bad_sum / steps,
            "tilt_mean": self._tilt_sum / steps,
            "torque_sq_mean": self._torque_sq_sum / steps,
            "mech_power_abs": self._mech_power_sum / steps,
            "action_rate_mean": self._action_rate_sum / steps,
            "foot_slip_mean": self._foot_slip_sum / steps,
        }
