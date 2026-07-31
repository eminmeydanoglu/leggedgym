# Curriculum, command-resampling and reward machinery ported from go2_rl_gym
# (wty-yy, vendored IsaacGym fork of unitree_rl_gym, RSS 2026 MoE locomotion
# paper) onto the host LeggedRobot contract. Shared by both go2_moects arms
# (MoE-CTS and HIM): the mixin never defines ``compute_observations``, so each
# arm keeps its own observation contract.
#
# Vendored sources (go2_rl_gym/legged_gym/envs/base/legged_robot.py unless
# noted): _update_terrain_curriculum, _get_env_origins (round-robin part),
# _update_env_command_ranges, _resample_commands, update_reward_curriculum /
# get_current_scale + compute_reward ramping, _get_dynamic_sigma +
# _reward_tracking_{lin,ang}_vel, _get_base_height + _reward_correct_base_height
# + _reward_feet_regulation, and _reward_hip_to_default from
# go2_rl_gym/legged_gym/envs/go2/go2_env.py.
#
# Buffer-name mapping (vendored -> host): root_states -> simulator.base_pos,
# contact_forces[feet] -> feet_force_norm, rigid_body_states[feet] ->
# simulator.feet_pos / simulator.feet_vel, torques -> simulator.torques,
# terrain / terrain_levels / terrain_types -> simulator._terrain /
# simulator.terrain_levels / simulator.terrain_types. Buffers the host does not
# track (commands_resampling_step, commands_xy_accumulation, max_move_distance,
# stop_heading, last_is_limit_vel, env_command_ranges) are maintained by the
# mixin itself in _init_buffers.

from itertools import product

import torch

from legged_gym.utils.math_utils import wrap_to_pi, quat_apply


def _sample_disjoint_intervals(env_ids, limit_bound, cfg_min, cfg_max, device):
    """Sample uniformly from [cfg_min, -limit_bound] U [limit_bound, cfg_max].

    Ported from go2_rl_gym/legged_gym/utils/isaacgym_utils.py.
    """
    width_neg = torch.nn.functional.relu(-limit_bound - cfg_min)
    width_pos = torch.nn.functional.relu(cfg_max - limit_bound)

    total_width = width_neg + width_pos + 1e-6  # epsilon against division by zero
    u = torch.rand(len(env_ids), device=device) * total_width

    samples = torch.where(
        u < width_neg,
        cfg_min + u,
        cfg_max - width_pos + (u - width_neg)
    )
    return samples


def _sample_single_interval(env_ids, cfg_min, cfg_max, device):
    """Sample uniformly from [cfg_min, cfg_max].

    Ported from go2_rl_gym/legged_gym/utils/isaacgym_utils.py.
    """
    r = torch.rand(len(env_ids), device=device)
    samples = cfg_min + r * (cfg_max - cfg_min)
    return samples


class WtyCurriculumMixin:
    """go2_rl_gym terrain-curriculum / command / reward machinery for host envs.

    Cooperative overrides only (every method calls ``super()`` where a host
    method of the same name exists). Place the mixin first in the MRO:

        class Go2MoECTS(WtyCurriculumMixin, LeggedRobotCTS): ...
        class Go2MoECTSHIM(WtyCurriculumMixin, Go2BenchHIM): ...

    The mixin is active only on a ``moe_grid`` terrain (the Phase-2 builder
    records ``cols2id``/``name2cols`` on the Terrain instance). On any other
    terrain it degrades to global command ranges, default tracking sigma and no
    terrain curriculum -- the defensive fallback required by the port plan.
    """

    # ------------------------------------------------------------------
    # Config / buffer setup
    # ------------------------------------------------------------------

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        # Iteration-based curricula count in PPO iterations; the vendored repo
        # hardcodes 24 (its runner's num_steps_per_env). Keep
        # cfg.env.wty_steps_per_iteration in sync with
        # CfgPPO.runner.num_steps_per_env.
        self.num_steps_per_iter = getattr(self.cfg.env, "wty_steps_per_iteration", 24)
        # Keep the CTS teacher/student split proportional when num_envs is
        # overridden (e.g. --num_envs on small GPUs): cfg.env.num_teacher is a
        # class-level constant computed for the cfg's own num_envs. Uses
        # cfg.env.num_envs (self.num_envs is not set yet at _parse_cfg time).
        # No-op for the HIM arm (no num_teacher attribute).
        if hasattr(self, "num_teacher"):
            self.num_teacher = int(self.cfg.env.num_envs
                                   * getattr(self.cfg.env, "teacher_env_ratio", 0.75))
        # [vendored _parse_cfg]: iteration-based command range curriculum,
        # sorted descending so reached entries can be popped from the back.
        command_range_curriculum = getattr(self.cfg.commands, "command_range_curriculum", None)
        if command_range_curriculum:
            self.cfg.commands.command_range_curriculum = sorted(
                command_range_curriculum, key=lambda x: x["iter"], reverse=True)

    def _init_buffers(self):
        super()._init_buffers()
        # --- vendored command-resampling state (_init_buffers) ---
        self.commands_resampling_step = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_xy_accumulation = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.max_move_distance = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.stop_heading = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_is_limit_vel = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.zero_command_proba = 0.0
        self.max_lin_vel = max(
            abs(self.command_ranges["lin_vel_x"][0]), abs(self.command_ranges["lin_vel_x"][1]),
            abs(self.command_ranges["lin_vel_y"][0]), abs(self.command_ranges["lin_vel_y"][1]))
        self.limit_vel_prob = self.cfg.commands.limit_vel_prob
        self.limit_vel_comb = torch.tensor(list(product(
            self.cfg.commands.limit_vel["lin_vel_x"],
            self.cfg.commands.limit_vel["lin_vel_y"],
            self.cfg.commands.limit_vel["ang_vel_yaw"]
        )), device=self.device, requires_grad=False)
        # --- vendored reward-scale curriculum (__init__ / update_reward_curriculum) ---
        self.reward_curriculum_configs = getattr(self.cfg.rewards, "curriculum_rewards", None) or []
        self.reward_curriculum_scales = {
            config["reward_name"]: config["start_value"] for config in self.reward_curriculum_configs}
        # --- vendored dynamic tracking sigma (_create_envs) ---
        self.dynamic_sigma_cfg = getattr(self.cfg.rewards, "dynamic_sigma", None)
        if self.dynamic_sigma_cfg is not None:
            self.terrain_max_sigmas = torch.tensor(
                self.dynamic_sigma_cfg["max_sigma"], device=self.device, requires_grad=False)
        # --- vendored base-height scan mask (_init_buffers) ---
        # The mask selects the 0.4m x 0.3m window of the 17x11 height scan
        # directly under the base; ordering matches the simulator's
        # meshgrid(x, y, indexing='ij') height-point layout.
        if self.cfg.terrain.measure_heights:
            x_points = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device)
            y_points = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device)
            grid_x, grid_y = torch.meshgrid(x_points, y_points, indexing="ij")
            x_mask = (grid_x.flatten() >= -0.2) & (grid_x.flatten() <= 0.2)  # 0.4m length
            y_mask = (grid_y.flatten() >= -0.15) & (grid_y.flatten() <= 0.15)  # 0.3m width
            self.base_height_scan_mask = (x_mask & y_mask).float()
            self.num_base_height_scan_points = self.base_height_scan_mask.sum()
            assert self.num_base_height_scan_points > 0, "No height scan points within the specified area."
        # --- per-env command ranges (vendored env_command_ranges) ---
        self.env_command_ranges = {
            key: torch.tensor(value, device=self.device, dtype=torch.float,
                              requires_grad=False).repeat(self.num_envs, 1)
            for key, value in self.command_ranges.items()
        }
        # --- terrain curriculum assignment (vendored _get_env_origins port) ---
        self.wty_terrain_ids = None
        self._wty_name2cols = {}
        self._wty_curriculum_active = False
        self._wty_setup_terrain_curriculum()

    def _wty_setup_terrain_curriculum(self):
        """Round-robin terrain level/type assignment + semantic terrain ids.

        [vendored _get_env_origins]: the host simulator assigns random initial
        levels, while the vendored repo spreads envs round-robin over the
        initial levels and over all terrain columns, which the game-inspired
        curriculum and the per-terrain-type command limits rely on. Only
        active on a moe_grid terrain; otherwise the mixin stays inactive.
        """
        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            return
        terrain = getattr(self.simulator, "_terrain", None)
        cols2id = getattr(terrain, "cols2id", None)
        if not cols2id or not hasattr(self.simulator, "_terrain_origins"):
            return  # defensive fallback: no moe_grid bookkeeping -> curriculum off
        # vendored: curriculum=True -> start on levels 0..max_init_terrain_level
        # (the mixin owns the curriculum, so cfg.terrain.curriculum stays False
        # and max_init_terrain_level applies unconditionally here).
        max_init_level = self.cfg.terrain.max_init_terrain_level
        self.simulator._terrain_levels = torch.fmod(
            torch.arange(self.num_envs, device=self.device), max_init_level + 1)
        self.simulator._terrain_types = torch.div(
            torch.arange(self.num_envs, device=self.device),
            (self.num_envs / self.cfg.terrain.num_cols), rounding_mode="floor").to(torch.long)
        # semantic terrain id per env: column index -> vendored terrain id
        terrain_cols2id = torch.tensor(cols2id, device=self.device)
        self.wty_terrain_ids = terrain_cols2id[self.simulator.terrain_types]
        self.simulator._env_origins[:] = self.simulator._terrain_origins[
            self.simulator.terrain_levels, self.simulator.terrain_types]
        self._wty_name2cols = {
            name: torch.tensor(sorted(cols), device=self.device)
            for name, cols in terrain.name2cols.items()
        }
        self._wty_curriculum_active = True
        self._update_env_command_ranges()

    # ------------------------------------------------------------------
    # Per-step bookkeeping (vendored post_physics_step)
    # ------------------------------------------------------------------

    def _post_physics_step_callback(self):
        """[vendored post_physics_step]: per-env command-resample countdown,
        max-move-distance tracking, then resample; host push behavior kept.

        The host's fixed-period resample (episode_length % resampling_time) and
        every-step heading recompute are intentionally replaced: the vendored
        resampler owns per-env resampling and heading (stop_heading) logic.
        """
        self.commands_resampling_step -= 1
        # max xy distance from the env origin over the episode (drives the
        # terrain curriculum instead of the instantaneous distance).
        self.max_move_distance = self.max_move_distance.maximum(torch.norm(
            self.simulator.base_pos[:, :2] - self.simulator.env_origins[:, :2], dim=1))
        # resample commands after the episode's reward bookkeeping; the guard
        # skips the last step of an episode (vendored).
        resampling_env_ids = ((self.commands_resampling_step <= 0.0)
                              * (self.episode_length_buf < self.max_episode_length - 1)
                              ).nonzero(as_tuple=False).flatten()
        self._resample_commands(resampling_env_ids)

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self.simulator.push_robots()
        if self.cfg.domain_rand.push_links and (self.common_step_counter % self.cfg.domain_rand.push_links_interval == 0):
            self.simulator.push_links()

    # ------------------------------------------------------------------
    # Terrain curriculum (vendored _update_terrain_curriculum)
    # ------------------------------------------------------------------

    def _update_terrain_curriculum(self, env_ids):
        """[vendored _update_terrain_curriculum]: game-inspired curriculum.

        Promotion uses the episode's max move distance; demotion compares it
        against the accumulated xy command distance
        (move_down_by_accumulated_xy_command). The level update, max-level
        wraparound and env_origins reassignment live in the host simulator's
        update_terrain_curriculum (same semantics as the vendored inline code).
        """
        if not self.init_done or not self._wty_curriculum_active:
            # don't change on initial reset / without a moe_grid terrain
            return
        distance = self.max_move_distance[env_ids]
        # robots that walked far enough progress to harder terrains
        move_up = distance > self.simulator._terrain.env_length / 2
        if getattr(self.cfg.terrain, "move_down_by_accumulated_xy_command", False):
            move_down = (distance < torch.norm(self.commands_xy_accumulation[env_ids], dim=1)
                         * (self.cfg.commands.resampling_time * (1 - self.zero_command_proba)) * 0.5) * ~move_up
        else:
            # robots that walked less than half of their required distance go to simpler terrains
            move_down = (distance < torch.norm(
                self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5) * ~move_up
        self.simulator.update_terrain_curriculum(env_ids, move_up, move_down)
        self.max_move_distance[env_ids] = 0.0

    # ------------------------------------------------------------------
    # Command resampling (vendored _resample_commands / _update_env_command_ranges)
    # ------------------------------------------------------------------

    def _update_env_command_ranges(self):
        """[vendored _update_env_command_ranges]: per-terrain-type command
        limits, clamped to the current global command ranges."""
        if self.wty_terrain_ids is None:
            self.env_command_ranges = {
                key: torch.tensor(value, device=self.device, dtype=torch.float,
                                  requires_grad=False).repeat(self.num_envs, 1)
                for key, value in self.command_ranges.items()
            }
            return
        for terrain_id, terrain_command_ranges in enumerate(self.cfg.commands.terrain_max_command_ranges):
            env_ids = (self.wty_terrain_ids == terrain_id).nonzero(as_tuple=False).flatten()
            if len(env_ids) == 0:
                continue
            self.env_command_ranges["lin_vel_x"][env_ids, 0] = max(
                terrain_command_ranges["lin_vel_x"][0],
                self.command_ranges["lin_vel_x"][0])
            self.env_command_ranges["lin_vel_x"][env_ids, 1] = min(
                terrain_command_ranges["lin_vel_x"][1],
                self.command_ranges["lin_vel_x"][1])
            self.env_command_ranges["lin_vel_y"][env_ids, 0] = max(
                terrain_command_ranges["lin_vel_y"][0],
                self.command_ranges["lin_vel_y"][0])
            self.env_command_ranges["lin_vel_y"][env_ids, 1] = min(
                terrain_command_ranges["lin_vel_y"][1],
                self.command_ranges["lin_vel_y"][1])
            self.env_command_ranges["ang_vel_yaw"][env_ids, 0] = max(
                terrain_command_ranges["ang_vel_yaw"][0],
                self.command_ranges["ang_vel_yaw"][0])
            self.env_command_ranges["ang_vel_yaw"][env_ids, 1] = min(
                terrain_command_ranges["ang_vel_yaw"][1],
                self.command_ranges["ang_vel_yaw"][1])
            if self.cfg.commands.heading_command:
                self.env_command_ranges["heading"][env_ids, 0] = max(
                    terrain_command_ranges["heading"][0],
                    self.command_ranges["heading"][0])
                self.env_command_ranges["heading"][env_ids, 1] = min(
                    terrain_command_ranges["heading"][1],
                    self.command_ranges["heading"][1])

    def _resample_commands(self, env_ids):
        """[vendored _resample_commands]: dynamic (traverse-guaranteeing)
        command resampling with per-terrain-type limits, limit-velocity draws,
        a zero-command curriculum and an iteration-based command-range
        curriculum. Replaces the host's uniform resampler.

        The vendored turn_over zero-command block is dropped: the host has no
        turn_over machinery and the ported go2 config keeps turn_over=False.
        """
        if len(env_ids) == 0:
            return
        self.stop_heading[env_ids] = False
        # update command curriculum with train steps
        if len(self.cfg.commands.command_range_curriculum):
            current_iter = self.common_step_counter // self.num_steps_per_iter
            # iterate backwards to be able to pop entries
            for i in range(len(self.cfg.commands.command_range_curriculum) - 1, -1, -1):
                cmd_cfg = self.cfg.commands.command_range_curriculum[i]
                if current_iter >= cmd_cfg["iter"]:
                    self.command_ranges["lin_vel_x"] = cmd_cfg["lin_vel_x"]
                    self.command_ranges["lin_vel_y"] = cmd_cfg["lin_vel_y"]
                    self.command_ranges["ang_vel_yaw"] = cmd_cfg["ang_vel_yaw"]
                    self.command_ranges["heading"] = cmd_cfg["heading"]
                    self.max_lin_vel = max(
                        abs(self.command_ranges["lin_vel_x"][0]), abs(self.command_ranges["lin_vel_x"][1]),
                        abs(self.command_ranges["lin_vel_y"][0]), abs(self.command_ranges["lin_vel_y"][1]))
                    self.cfg.commands.command_range_curriculum.pop(i)
                    self._update_env_command_ranges()
                    print(f"Command range updated at iter {current_iter}: {self.command_ranges}")
        remaining_dist = torch.clip(
            0.625 * self.cfg.terrain.terrain_length
            - torch.norm(self.commands_xy_accumulation[env_ids], dim=1) * self.cfg.commands.resampling_time,
            0.0)
        self.commands_resampling_step[env_ids] = self.cfg.commands.resampling_time / self.dt
        if self.cfg.commands.dynamic_resample_commands:
            # arrive at boundary 0.625 times the width of the remaining distance.
            # The reset path (host reset_idx) calls this BEFORE zeroing
            # episode_length_buf, so the remaining step count can legitimately be
            # <= 0 here: exactly 0 for an env that terminates on contact at step
            # max_episode_length, -1 for a timeout. Both mean "no minimum speed
            # to enforce". The old timeout case (-1) already produced a negative
            # ratio that got clipped to 0, so mapping every non-positive budget
            # to 0 is behaviour-preserving there and removes the division-by-~0
            # (and the crash it used to raise) at exactly 0.
            remaining_steps = self.max_episode_length - self.episode_length_buf[env_ids]
            vel_low_bound = torch.where(
                remaining_steps > 0,
                remaining_dist / (torch.clamp(remaining_steps, min=1.0) * self.dt),
                torch.zeros_like(remaining_dist))
            vel_low_bound = torch.clip(vel_low_bound, 0.0)
            self.commands[env_ids, 0] = _sample_disjoint_intervals(
                env_ids,
                vel_low_bound,
                self.env_command_ranges["lin_vel_x"][env_ids, 0],
                self.env_command_ranges["lin_vel_x"][env_ids, 1],
                self.device)
            self.commands[env_ids, 1] = _sample_disjoint_intervals(
                env_ids,
                vel_low_bound,
                self.env_command_ranges["lin_vel_y"][env_ids, 0],
                self.env_command_ranges["lin_vel_y"][env_ids, 1],
                self.device)
            if self.cfg.commands.heading_command:
                r = torch.rand(len(env_ids), device=self.device)
                lower = self.env_command_ranges["heading"][env_ids, 0]
                upper = self.env_command_ranges["heading"][env_ids, 1]
                self.commands[env_ids, 3] = (upper - lower) * r + lower
            else:
                r = torch.rand(len(env_ids), device=self.device)
                lower = self.env_command_ranges["ang_vel_yaw"][env_ids, 0]
                upper = self.env_command_ranges["ang_vel_yaw"][env_ids, 1]
                self.commands[env_ids, 2] = (upper - lower) * r + lower
        else:
            self.commands[env_ids, 0] = _sample_single_interval(
                env_ids,
                self.env_command_ranges["lin_vel_x"][env_ids, 0],
                self.env_command_ranges["lin_vel_x"][env_ids, 1],
                self.device)
            self.commands[env_ids, 1] = _sample_single_interval(
                env_ids,
                self.env_command_ranges["lin_vel_y"][env_ids, 0],
                self.env_command_ranges["lin_vel_y"][env_ids, 1],
                self.device)
            if self.cfg.commands.heading_command:
                self.commands[env_ids, 3] = _sample_single_interval(
                    env_ids,
                    self.env_command_ranges["heading"][env_ids, 0],
                    self.env_command_ranges["heading"][env_ids, 1],
                    self.device)
            else:
                self.commands[env_ids, 2] = _sample_single_interval(
                    env_ids,
                    self.env_command_ranges["ang_vel_yaw"][env_ids, 0],
                    self.env_command_ranges["ang_vel_yaw"][env_ids, 1],
                    self.device)
            # set small commands to zero
            self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        rand_prob = torch.rand(len(env_ids), device=self.device)
        min_prob, max_prob = 0.0, 0.0
        # set limitation lin vel
        if self.limit_vel_prob > 0.0:
            max_prob += self.limit_vel_prob
            lim_mask = (rand_prob >= min_prob) * (rand_prob < max_prob)
            lim_env_ids = env_ids[lim_mask]
            if len(lim_env_ids) > 0:
                change_lim_env_ids = lim_env_ids
                if self.cfg.commands.limit_vel_invert_when_continuous:
                    was_limited = self.last_is_limit_vel[lim_env_ids]
                    invert_env_ids = lim_env_ids[was_limited]
                    self.commands[invert_env_ids, 0] *= -1.0
                    self.commands[invert_env_ids, 1] *= -1.0
                    self.commands[invert_env_ids, 2] *= -1.0
                    change_lim_env_ids = lim_env_ids[~was_limited]
                vel_idx = torch.randint(0, self.limit_vel_comb.shape[0], (len(change_lim_env_ids),), device=self.device)
                lin_vel_x_lim = torch.where(
                    self.limit_vel_comb[vel_idx, 0] == -1,
                    self.env_command_ranges["lin_vel_x"][change_lim_env_ids, 0],
                    self.env_command_ranges["lin_vel_x"][change_lim_env_ids, 1])
                lin_vel_x_lim[self.limit_vel_comb[vel_idx, 0] == 0] = 0.0
                lin_vel_y_lim = torch.where(
                    self.limit_vel_comb[vel_idx, 1] == -1,
                    self.env_command_ranges["lin_vel_y"][change_lim_env_ids, 0],
                    self.env_command_ranges["lin_vel_y"][change_lim_env_ids, 1])
                lin_vel_y_lim[self.limit_vel_comb[vel_idx, 1] == 0] = 0.0
                ang_vel_z_lim = torch.where(
                    self.limit_vel_comb[vel_idx, 2] == -1,
                    self.env_command_ranges["ang_vel_yaw"][change_lim_env_ids, 0],
                    self.env_command_ranges["ang_vel_yaw"][change_lim_env_ids, 1])
                ang_vel_z_lim[self.limit_vel_comb[vel_idx, 2] == 0] = 0.0
                self.commands[change_lim_env_ids, 0] = lin_vel_x_lim
                self.commands[change_lim_env_ids, 1] = lin_vel_y_lim
                self.commands[change_lim_env_ids, 2] = ang_vel_z_lim
                if self.cfg.commands.heading_command and self.cfg.commands.stop_heading_at_limit:
                    self.stop_heading[lim_env_ids] = True  # stop heading to current heading
                self.last_is_limit_vel[env_ids] = False
                self.last_is_limit_vel[lim_env_ids] = True
            else:
                self.last_is_limit_vel[env_ids] = False
            min_prob += self.limit_vel_prob

        # set all commands to zero with some probability
        if self.cfg.commands.zero_command_curriculum is not None:
            self.zero_command_proba = self._get_current_scale(self.cfg.commands.zero_command_curriculum)
        if self.zero_command_proba > 0.0:
            max_prob += self.zero_command_proba
            next_resampling_step = torch.clip(
                self.max_episode_length - self.episode_length_buf[env_ids]
                - (remaining_dist / (0.8 * self.max_lin_vel * self.dt + 1e-9)),
                min=0.0,
                max=self.cfg.commands.resampling_time / self.dt)
            zero_mask = (rand_prob >= min_prob) * (rand_prob < max_prob) * (next_resampling_step > 0.0)
            zero_env_ids = env_ids[zero_mask]
            if len(zero_env_ids) > 0:
                self.commands[zero_env_ids, :2] = 0.0
                self.commands_resampling_step[zero_env_ids] = next_resampling_step[zero_mask]
                if self.cfg.commands.limit_ang_vel_at_zero_command_prob > 0.0:
                    ang_vel_rand = torch.rand(len(zero_env_ids), device=self.device)  # independent distribution
                    add_ang_mask = ang_vel_rand < self.cfg.commands.limit_ang_vel_at_zero_command_prob
                    add_ang_env_ids = zero_env_ids[add_ang_mask]
                    if len(add_ang_env_ids) > 0:
                        direction_rand = torch.rand(len(add_ang_env_ids), device=self.device)
                        self.commands[add_ang_env_ids, 2] = torch.where(
                            direction_rand < 0.5,
                            self.env_command_ranges["ang_vel_yaw"][add_ang_env_ids, 0],
                            self.env_command_ranges["ang_vel_yaw"][add_ang_env_ids, 1])
                        if self.cfg.commands.heading_command:
                            self.stop_heading[add_ang_env_ids] = True
            min_prob += self.zero_command_proba

        self.commands_xy_accumulation[env_ids] += self.commands[env_ids, :2]

        if self.cfg.commands.heading_command:
            heading_env_ids = env_ids[self.stop_heading[env_ids] == 0.0]
            if len(heading_env_ids) > 0:
                forward = quat_apply(self.simulator.base_quat[heading_env_ids], self.forward_vec[heading_env_ids])
                heading = torch.atan2(forward[:, 1], forward[:, 0])
                self.commands[heading_env_ids, 2] = torch.clip(
                    0.5 * wrap_to_pi(self.commands[heading_env_ids, 3] - heading),
                    self.env_command_ranges["ang_vel_yaw"][heading_env_ids, 0],
                    self.env_command_ranges["ang_vel_yaw"][heading_env_ids, 1])

    # ------------------------------------------------------------------
    # Reset lifecycle (vendored reset_idx)
    # ------------------------------------------------------------------

    def reset_idx(self, env_ids):
        """[vendored reset_idx]: run the terrain curriculum and clear the
        command accumulators around the host reset, then log the
        per-terrain-type curriculum metrics."""
        if len(env_ids) > 0:
            # The host only calls _update_terrain_curriculum when
            # cfg.terrain.curriculum is set; moe_grid keeps that flag off (the
            # builder dispatch), so the mixin drives the vendored curriculum
            # explicitly. Early-out guards (init_done / active) live inside
            # _update_terrain_curriculum.
            if not self.cfg.terrain.curriculum:
                self._update_terrain_curriculum(env_ids)
            # Cleared BEFORE super().reset_idx() re-samples commands: the
            # vendored accumulation starts from the freshly sampled command.
            self.commands_xy_accumulation[env_ids] = 0.0
            self.commands_resampling_step[env_ids] = self.cfg.commands.resampling_time / self.dt
            # Vendored reset_idx zeroes episode_length_buf and only THEN calls
            # _resample_commands, so its dynamic velocity lower bound is computed
            # against a full fresh episode. The host reverses that order
            # (legged_robot.reset_idx: _resample_commands, ... , buf = 0), which
            # would make _resample_commands size the bound against the *old*
            # episode's leftover steps -- and hit the degenerate zero-budget case
            # for an env terminating exactly at max_episode_length. Zero it here
            # to restore the vendored ordering; the host's own zeroing below is
            # then a no-op, and nothing between here and there reads the buffer.
            self.episode_length_buf[env_ids] = 0
        super().reset_idx(env_ids)
        # per-terrain-type curriculum metrics (vendored reset_idx extras)
        if len(env_ids) > 0 and self._wty_curriculum_active:
            self.extras["episode"]["terrain_level_all"] = torch.mean(
                self.simulator.terrain_levels.float())
            for name, cols in self._wty_name2cols.items():
                self.extras["episode"]["terrain_level_" + name] = torch.mean(
                    self.simulator.terrain_levels[torch.isin(self.simulator.terrain_types, cols)].float())
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]

    # ------------------------------------------------------------------
    # Reward computation (vendored compute_reward + update_reward_curriculum)
    # ------------------------------------------------------------------

    def compute_reward(self):
        """[vendored compute_reward + update_reward_curriculum]: host reward
        loop plus per-term iteration-based scale ramping
        (cfg.rewards.curriculum_rewards)."""
        self._update_reward_curriculum()
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            if name in self.reward_curriculum_scales:
                rew = rew * self.reward_curriculum_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
        # host UED accounting tail, kept for contract parity (never active here)
        if self.ued_adapter is not None and getattr(self.cfg.env, "ued_enabled", False):
            self.ued_adapter.record_step(self.rew_buf)

    def _update_reward_curriculum(self, force_update: bool = False):
        """[vendored update_reward_curriculum]: refresh ramped reward scales
        once per PPO iteration."""
        if self.reward_curriculum_configs:
            if self.common_step_counter % self.num_steps_per_iter == 0 or force_update:
                for config in self.reward_curriculum_configs:
                    self.reward_curriculum_scales[config["reward_name"]] = self._get_current_scale(config)

    def _get_current_scale(self, config):
        """[vendored get_current_scale]: linear interpolation between
        start_value and end_value over [start_iter, end_iter] PPO iterations.

        config: {'start_iter': 0, 'end_iter': 1500, 'start_value': 1.0, 'end_value': 0.0}
        """
        current_iter = self.common_step_counter // self.num_steps_per_iter
        percentage = (current_iter - config["start_iter"]) / (config["end_iter"] - config["start_iter"])
        percentage = max(min(percentage, 1.0), 0.0)
        return (1.0 - percentage) * config["start_value"] + percentage * config["end_value"]

    # ------------------------------------------------------------------
    # Dynamic tracking sigma (vendored _get_dynamic_sigma + tracking rewards)
    # ------------------------------------------------------------------

    def _get_dynamic_sigma(self, target_vel_abs, v_min, v_max):
        """[vendored _get_dynamic_sigma]: per-terrain, per-level tracking-sigma
        scaling. The vendored gate on cfg.terrain.curriculum becomes the
        mixin's curriculum-active flag (moe_grid keeps the host flag off)."""
        default_sigma = self.cfg.rewards.tracking_sigma
        if not self._wty_curriculum_active or self.dynamic_sigma_cfg is None:
            return torch.full_like(target_vel_abs, default_sigma)
        target_sigmas = self.terrain_max_sigmas[self.wty_terrain_ids]
        sigma = torch.full_like(target_vel_abs, default_sigma)
        # based on velocity ranges, compute sigma
        # v_min <= v < v_max (linear interpolation)
        mask = (target_vel_abs >= v_min) & (target_vel_abs < v_max)
        if mask.any():
            ratio = (target_vel_abs[mask] - v_min) / (v_max - v_min)
            sigma[mask] = default_sigma + ratio * (target_sigmas[mask] - default_sigma)
        # v >= v_max
        mask = target_vel_abs >= v_max
        if mask.any():
            sigma[mask] = target_sigmas[mask]
        # based on terrain level, compute sigma
        level_scale = torch.clamp(torch.exp((self.simulator.terrain_levels.float() + 1.0) / 10.0) - 1.0, max=1.0)
        sigma = default_sigma + level_scale * (sigma - default_sigma)
        return sigma

    def _reward_tracking_lin_vel(self):
        """[vendored _reward_tracking_lin_vel]: per-axis dynamic sigma."""
        if self.dynamic_sigma_cfg is None:
            sigma_x = sigma_y = self.cfg.rewards.tracking_sigma
        else:
            vmin = self.dynamic_sigma_cfg["min_lin_vel"]
            vmax = self.dynamic_sigma_cfg["max_lin_vel"]
            sigma_x = self._get_dynamic_sigma(torch.abs(self.commands[:, 0]), vmin, vmax)
            sigma_y = self._get_dynamic_sigma(torch.abs(self.commands[:, 1]), vmin, vmax)
        lin_vel_error_sq = torch.square(self.commands[:, :2] - self.simulator.base_lin_vel[:, :2])
        scaled_error = lin_vel_error_sq[:, 0] / sigma_x + lin_vel_error_sq[:, 1] / sigma_y
        return torch.exp(-scaled_error)

    def _reward_tracking_ang_vel(self):
        """[vendored _reward_tracking_ang_vel]: dynamic sigma on yaw rate."""
        if self.dynamic_sigma_cfg is None:
            sigma = self.cfg.rewards.tracking_sigma
        else:
            vmin = self.dynamic_sigma_cfg["min_ang_vel"]
            vmax = self.dynamic_sigma_cfg["max_ang_vel"]
            sigma = self._get_dynamic_sigma(torch.abs(self.commands[:, 2]), vmin, vmax)
        ang_vel_error_sq = torch.square(self.commands[:, 2] - self.simulator.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error_sq / sigma)

    # ------------------------------------------------------------------
    # Base-height and foot-lift rewards (vendored _get_base_height,
    # _reward_correct_base_height, _reward_feet_regulation)
    # ------------------------------------------------------------------

    def _get_base_height(self):
        """[vendored _get_base_height]: base height above the mean ground
        estimated from the masked height scan."""
        if not self.cfg.terrain.measure_heights:
            return self.simulator.base_pos[:, 2]
        masked_heights = self.simulator.measured_heights * self.base_height_scan_mask.unsqueeze(0)
        estimated_ground_z = masked_heights.sum(dim=1) / self.num_base_height_scan_points
        return self.simulator.base_pos[:, 2] - estimated_ground_z

    def _reward_correct_base_height(self):
        """[vendored _reward_correct_base_height]"""
        base_height = self._get_base_height()
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_feet_regulation(self):
        """[vendored _reward_feet_regulation]: CTS foot-lift regularizer --
        with growing foot xy velocity, demand proportionally higher lift."""
        base_height = self._get_base_height()
        feet_pos = self.simulator.feet_pos                    # (N, 4, 3), world frame
        feet_xy_vel = self.simulator.feet_vel[:, :, :2]       # (N, 4, 2), world frame
        delta_feet = feet_pos - self.simulator.base_pos.unsqueeze(1)
        # foot height relative to the base, then relative to the ground (N, 4)
        feet2base_height = (delta_feet * self.simulator.projected_gravity.unsqueeze(1)).sum(-1)
        feet_height = torch.clamp(base_height.unsqueeze(1) - feet2base_height, min=0.0)
        return (feet_xy_vel.pow(2).sum(-1)
                * torch.exp(-feet_height / (0.025 * self.cfg.rewards.base_height_target))).sum(-1)

    def _reward_hip_to_default(self):
        """[vendored go2_env.py _reward_hip_to_default]: keep hip joints close
        to the default pose (prevents the overly narrow CTS gait)."""
        hip_dof_indices = [0, 3, 6, 9]
        hip_pos = self.simulator.dof_pos[:, hip_dof_indices]
        default_hip_pos = self.simulator.default_dof_pos[:, hip_dof_indices]
        return torch.sum(torch.abs(hip_pos - default_hip_pos), dim=1)
