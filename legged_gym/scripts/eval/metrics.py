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

    def __init__(self, num_envs: int, device):
        self.num_envs = num_envs
        self.device = device
        z = lambda dtype=torch.float: torch.zeros(num_envs, dtype=dtype, device=device)

        # running (current, not-yet-finished episode)
        self._alive_steps = z(torch.long)     # steps since last reset (our own count)
        self._return_run = z()                # reward sum in current episode

        # completed-episode accumulators
        self._ep_len_sum = z()                # sum of lengths of finished episodes
        self._ep_count = z(torch.long)        # number of finished episodes
        self._fall_count = z(torch.long)      # finished episodes ending in a fall
        self._timeout_count = z(torch.long)   # finished episodes ending in a timeout
        self._ep_return_sum = z()             # sum of returns of finished episodes

        # step-level accumulators (averaged over all steps)
        self._lin_err_sum = z()               # |cmd_xy - v_xy| summed over steps
        self._ang_err_sum = z()               # |cmd_yaw - w_yaw| summed over steps
        self._steps = 0                       # global step counter (same for all envs)

    @torch.no_grad()
    def update(self, rew, reset_buf, time_out_buf, lin_err, ang_err):
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
        self._lin_err_sum += lin_err
        self._ang_err_sum += ang_err
        self._steps += 1

        # book a finished episode wherever `done`
        self._ep_len_sum += torch.where(done, self._alive_steps.float(), torch.zeros_like(self._return_run))
        self._ep_count += done.long()
        self._ep_return_sum += torch.where(done, self._return_run, torch.zeros_like(self._return_run))
        self._fall_count += fall.long()
        self._timeout_count += timeout.long()

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
            "never_fell": (self._fall_count == 0).float(),
            "mean_ep_len": mean_ep_len,
            "falls_per_1k": self._fall_count.float() / steps * 1000.0,
            "mean_return": self._ep_return_sum / ep,
            "tracking_lin_err": self._lin_err_sum / steps,
            "tracking_ang_err": self._ang_err_sum / steps,
        }
