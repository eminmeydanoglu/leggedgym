"""Runtime physics-switch support shared by V3 Go2 training tasks.

The legacy benchmark samples mass/CoM during Genesis scene construction only.
V3 needs a change while an episode is live, without pretending a reset occurred.

Dose (per expert review): AT MOST ONE switch per episode, fired at a random
step drawn uniformly inside a fraction window of the episode length (default
``[0.2, 0.8]`` of ``max_episode_length``).  This keeps the vast majority of
timesteps stationary -- enough for an estimator to converge before and observe
the change -- while still making a memoryless policy structurally suboptimal.
It is also the exact training analogue of the S2 eval protocol (a single switch
near mid-episode).  Repeated/high-frequency switching is deliberately avoided:
if physics changes faster than the estimator window, the optimal policy reverts
to conservatism and the headroom we are trying to open closes again.
"""

from __future__ import annotations

import torch


class Go2V3PhysicsResampleMixin:
    """Resample mass and CoM once, at a random time, during a live V3 episode."""

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        dr = cfg.domain_rand
        self._v3_switch_enabled = bool(
            getattr(dr, "resample_physics_within_episode", False)
        )
        self._v3_switch_mass = bool(getattr(dr, "physics_resample_mass", False))
        self._v3_switch_com = bool(getattr(dr, "physics_resample_com", False))
        window = getattr(dr, "physics_resample_switch_window", None)
        if self._v3_switch_enabled:
            if not self._v3_switch_mass and not self._v3_switch_com:
                raise ValueError("V3 physics resampling is enabled but no mutable axis is selected")
            if not isinstance(window, (list, tuple)) or len(window) != 2:
                raise ValueError("physics_resample_switch_window must be [lo_frac, hi_frac]")
            lo_f, hi_f = map(float, window)
            if not (0.0 <= lo_f <= hi_f <= 1.0):
                raise ValueError("physics_resample_switch_window must satisfy 0 <= lo <= hi <= 1")
            self._v3_switch_lo_steps = max(1, int(round(lo_f * self.max_episode_length)))
            self._v3_switch_hi_steps = max(
                self._v3_switch_lo_steps, int(round(hi_f * self.max_episode_length))
            )

    def _init_buffers(self):
        super()._init_buffers()
        if self._v3_switch_enabled:
            # Absolute within-episode step at which this env's single switch
            # fires, and whether that switch has already happened this episode.
            self._v3_switch_step = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self._v3_switch_done = torch.ones(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._schedule_v3_switch(torch.arange(self.num_envs, device=self.device))

    def _schedule_v3_switch(self, env_ids):
        """Draw one pending switch time (from episode start) for each env."""
        if len(env_ids) == 0:
            return
        # randint uses an exclusive upper bound.  Include ``hi`` deliberately.
        when = torch.randint(
            self._v3_switch_lo_steps,
            self._v3_switch_hi_steps + 1,
            (len(env_ids),),
            device=self.device,
        )
        self._v3_switch_step[env_ids] = when
        self._v3_switch_done[env_ids] = False

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if self._v3_switch_enabled:
            # episode_length_buf was just zeroed by the base reset, so the
            # scheduled step is measured from the start of the new episode.
            self._schedule_v3_switch(env_ids)

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        if not self._v3_switch_enabled:
            return
        due = (
            (~self._v3_switch_done)
            & (self.episode_length_buf >= self._v3_switch_step)
        ).nonzero(as_tuple=False).flatten()
        if len(due) == 0:
            return
        resample = getattr(self.simulator, "resample_v3_physics", None)
        if not callable(resample):
            raise RuntimeError(
                "V3 dynamic physics requires a simulator with resample_v3_physics(); "
                "use Genesis or explicitly disable the V3 switch contract."
            )
        resample(due, mass=self._v3_switch_mass, com=self._v3_switch_com)
        # One switch per episode: mark done and do NOT reschedule until reset.
        self._v3_switch_done[due] = True

    def _update_terrain_curriculum(self, env_ids):
        """Keep V4 on its declared terrain row, independent of performance."""
        if getattr(self.cfg.terrain, "fixed_terrain_level", None) is not None:
            return
        return super()._update_terrain_curriculum(env_ids)
